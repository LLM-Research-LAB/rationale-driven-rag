import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from transformers import get_scheduler
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import spacy
from langchain_core.documents import Document


class T5ProjNetRetrievalEvaluator():

    def __init__(self):
        self.seed = 42
        self.batch_size = 1
        self.num_epochs = 50
        self.LOW_LABEL = 0
        self.AMBIGOUS_LABEL = 1
        self.HIGH_LABEL = 2
        self.nlp = spacy.load('en_core_web_sm')

    def fine_tune_model(self, train_texts, train_label, save_path='./outputs/'):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # main code from CRAG article
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForSequenceClassification.from_pretrained(
            "t5-large", num_labels=3)
        # tokenizer = self.tokenizer
        # model = self.model
        train_data = tokenizer(train_texts, padding="max_length",
                               max_length=512, truncation=True, return_tensors="pt")
        train = TensorDataset(
            train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
        train_dataloader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, sampler=None)
        optimizer = Adam(model.parameters(), lr=0.001,
                         betas=(0.9, 0.999), eps=1e-08)
        num_training_steps = self.num_epochs * len(train_dataloader)
        print(num_training_steps)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.to(self.device)
        for i, epoch in enumerate(range(self.num_epochs)):
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 10 == 0 and not step == 0:
                    print("step: ", step, "  loss:",
                          total_loss/(step*self.batch_size))
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                model.zero_grad()
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                loss.mean().backward()
                total_loss += loss.mean().item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            avg_train_loss = total_loss / len(train_dataloader)
            print("avg_loss:", avg_train_loss)
            print("train accuracy: ", accuracy(
                model, tokenizer, train_texts[:188], train_label[:188]))
            model.save_pretrained(save_path + "/ep{}".format(i))
            tokenizer.save_pretrained(save_path + "/ep{}".format(i))
            self.model = model
            self.tokenizer = tokenizer

    def load_pretrained_model(self, load_path='./outputs/ep8'):
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        # self.model = T5ForSequenceClassification.from_pretrained(load_path).to(self.device)
        self.model = SentenceTransformer(
            'sentence-transformers/gtr-t5-large').to('cuda')
        # self.proj_net = torch.load("/content/drive/MyDrive/Persian LLM/New Codes/CRAG/production_models/production_model_autoencoder(kmeans:91%,justT5,epoch20)-newprojectionmodel_acc:0.888,Resmodel.pt")

        # to load state_dict:
        self.proj_net = Net(768)  # create the model instance
        # self.proj_net.load_state_dict(torch.load('/content/drive/MyDrive/Persian LLM/New Codes/CRAG/production_models/production_model_autoencoder(kmeans:91%,justT5,epoch20)-newprojectionmodel_acc:0.888,Resmodel.pt'), strict=False)
        self.proj_net.load_state_dict(torch.load(
            '/content/drive/MyDrive/Persian LLM/New Codes/CRAG/production_models/production_model_autoencoder(kmeans:91%,justT5,epoch20)-newprojectionmodel_acc:0.888,Resmodel.pt'))
        return self

    def evaluate_single(self, q: str, doc: str) -> float:
        # we can use few shot to show mid levels
        if not (type(doc) is str):
          doc = doc.page_content
        prompt_template = "qnli question: qqqqqq sentence: cccccc"
        input_text = prompt_template.replace(
            'qqqqqq', q).replace('cccccc', doc)
        embeddings = torch.tensor(self.model.encode(input_text))
        output = torch.nn.functional.softmax(self.proj_net(embeddings), dim=0)
        _, predicted = torch.max(output, 0)
        return predicted
        # inputs = self.tokenizer(input_text, return_tensors="pt")
        # outputs = self.model(inputs["input_ids"].to(self.device),
        #                      attention_mask=inputs["attention_mask"].to(self.device))
        # print(outputs.logits)
        # return int(torch.argmax(F.softmax(outputs.logits, 1), dim=1)[0])

    def evaluate_batch(self, q: str, docs: list):
        # self.load_pretrained_model()
        high_docs = []
        medium_docs = []
        low_docs = []
        for d in docs:
            out = self.evaluate_single(q, d)
            if out == self.LOW_LABEL:
                low_docs.append(d)
            elif out == self.AMBIGOUS_LABEL:
                medium_docs.append(d)
            else:
                high_docs.append(d)
        # this condition means INCORRECT
        if len(low_docs) == len(docs):
            return [], "Yes"
        # this condition means CORRECT
        elif len(high_docs) > 0:
            return high_docs, "No"
        # this condition means ambigous
        else:
            return medium_docs, "Yes"

    def filter_highs(self, high_docs):
        possibilities = []
        print("high_docs len is: ", len(high_docs))
        for item in high_docs:
            d, q = item
            embeddings = torch.tensor(self.model.encode(d))
            output = torch.nn.functional.softmax(
                self.proj_net(embeddings), dim=0)
            possibilities.append(output.tolist()[2])
        print("here are possibilities: ", possibilities)
        array = np.array(possibilities)
        # Get the indices of the top 5 maximum values
        top_n_indices = np.argsort(array)[-5:][::-1]
        top_corresponding_values = [high_docs[i][1] for i in top_n_indices]
        return top_corresponding_values

    def evaluate_websearch(self, q: str, docs: list):
        # convert docs to strips. this is knowledge refinement
        # strips = []
        # for d in docs:
        #     for sentence in self.nlp(d).sents:
        #         strips.append(sentence)
        high_docs = []
        for d in docs:
            out = self.evaluate_single(q, str(d))
            if out == self.HIGH_LABEL:
                prompt_template = "qnli question: qqqqqq sentence: cccccc"
                input_text = prompt_template.replace(
                    'qqqqqq', q).replace('cccccc', str(d))
                high_docs.append((input_text, str(d)))
        return self.filter_highs(high_docs), "No"
