import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler
import torch.nn.functional as F


class T5RetrievalEvaluator():

    def __init__(self):
        self.seed = 42
        self.batch_size = 1
        self.num_epochs = 1
        self.LOW_LABEL = 0
        self.AMBIGOUS_LABEL = 1
        self.HIGH_LABEL = 2

    def fine_tune_model(self, train_texts, train_label, save_path='./outputs/'):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForSequenceClassification.from_pretrained(
            "t5-large", num_labels=3)
        # config
        train_data = tokenizer(train_texts, padding="max_length",
                               max_length=512, truncation=True, return_tensors="pt")
        train = TensorDataset(
            train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
        train_dataloader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, sampler=None)
        optimizer = AdamW(model.parameters(), lr=1e-4)
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
            model.save_pretrained(save_path + "/ep{}".format(i))
            tokenizer.save_pretrained(save_path + "/ep{}".format(i))

    def load_pretrained_model(self, load_path='./outputs/ep8'):
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(load_path)
        self.model = T5ForSequenceClassification.from_pretrained(
            load_path).to(self.device)

    def evaluate_single(self, q: str, doc: str) -> float:
        # we can use few shot to show mid levels
        prompt_template = "qnli question: qqqqqq sentence: cccccc"
        input_text = prompt_template.replace(
            'qqqqqq', q).replace('cccccc', doc)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model(inputs["input_ids"].to(self.device),
                             attention_mask=inputs["attention_mask"].to(self.device))
        return int(torch.argmax(F.softmax(outputs.logits, 1), dim=1)[0])

    def evaluate_batch(self, q: str, docs: list):
        self.load_pretrained_model()
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
            return medium_docs, "No"
        # this condition means ambigous
        else:
            return low_docs, "Yes"

    def evaluate_websearch(self, q: str, docs: list):
        self.load_pretrained_model()
        high_docs = []
        for d in docs:
            out = self.evaluate_single(q, d)
            if out == self.HIGH_LABEL:
                high_docs.append(d)
        return high_docs
