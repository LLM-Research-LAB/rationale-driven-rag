import re

from src.DataLoaders.DataLoader import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.prompts.prompt import PromptTemplate
from src.Helpers.Utils import check_different_shapes_of_true_false
import json
from tqdm import tqdm
import random
import pickle


class PubHealth(DataLoader):
    def __str__(self):
        return 'pubqa'
    # implement constructor if needed

    def __init__(self, contexts_path=None):
        self.template = 'default'
        self.generation_checkpoint = None
        self.websearch_count = 0
        self.strip = False
        self.contexts_path = contexts_path
        pass

    def load_data(self):
        file = open(
            'src/Dataset/eval_data/health_claims_processed.jsonl', 'r', encoding='utf-8')
        lines = file.readlines()
        self.input_test_data = []
        self.output_test_data = []
        self.contexts = []
        for line in lines:
            data = json.loads(line)
            self.input_test_data.append(data['question'])
            self.output_test_data.append(data['output'][0])
            docs = []
            for doc in data['ctxs']:
                docs.append(doc['title'].strip() + '//' + doc['text'].strip().replace("\n" , " "))
            self.contexts.append(docs)
        assert len(self.contexts) == len(
            self.input_test_data) == len(self.output_test_data)
        self.test_data = list(zip(self.input_test_data, self.output_test_data))
        if self.strip:
            self.strip_contexts()
        if self.contexts_path:
            self.load_contexts(path=self.contexts_path)
        return self.test_data

    # def index_documents(self):
    #     from models.EM import embedding
    #     directory = "src/Documents/pubhealth_documents/"
    #     all_docs = []
    #     for filename in os.listdir(directory):
    #         file_path = os.path.join(directory, filename)
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             content = f.read()
    #             all_docs.append(Document(page_content=content))

    #     splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #         chunk_size=1024, chunk_overlap=128
    #     ).split_documents(all_docs)

    #     vectorstore = FAISS.from_documents(splits, embedding)
    #     vectorstore.save_local("pubhealth_index")

    def train_test_split(self, split_point: float = 0.8):
        return None, self.test_data

    def get_llm_prompt_template(self,template='default'):
        if template == 'default':
            return PromptTemplate(
                template="Read the documents and answer the question: Is the following statement correct or not? \n\nDocuments: {context} \n\nStatement: {question}\n\nOnly say true if the statement is true; otherwise say false.\n\nAnswer:",
                input_variables=["context", "question"],
            )
        elif template == 'self-rag':
            return PromptTemplate(
                # template="### Instruction:\nIs the following statement correct or not? In Response say 'True' if it's correct; otherwise say 'False'.\n\n### Input:\n{question}\n\n### Response:\n{context}",
                template="### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false.\n\n## Input:\n\n{question}\n\n### Response:\n{context}",
                input_variables=["context", "question"],
            )
        else:
            return PromptTemplate(
                # template="### Instruction:\nIs the following statement correct or not? In Response say 'True' if it's correct; otherwise say 'False'.\n\n### Input:\n{question}\n\n### Response:\n{context}",
                template="### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false.\n\n### Input:\n{question}\n{context}\n\n### Response:",
                input_variables=["context", "question"],
            )

    def evaluate_rag_chain(self, rag_chain) -> float:
        if self.generation_checkpoint is None:
            self.generation_checkpoint = 0
            self.generations = []
        index = 0
        def remove_bracket_tokens(generated_text: str) -> str:
            """
            Removes any text enclosed in square brackets [] from a string.
            """
            # \[.*?\] matches '[' followed by any characters (non-greedy), followed by ']'
            cleaned_text = re.sub(r'\[.*?\]', '', generated_text)
            
            # .strip() removes any leftover leading/trailing whitespace
            return cleaned_text.strip()
        for input in tqdm(self.input_test_data[self.generation_checkpoint:], "evaluating on pubhealth... " if self.generation_checkpoint == 0 else "continuing evaluation from checkpoint..."):
            response = rag_chain(input.strip().replace(
                '\n', ' '), index)
            # print("the response of network is: ", response)
            self.generations.append(remove_bracket_tokens(response.replace('\n', ' ').strip()))
            index += 1
        # calculate accuracy here based on exact match
        total = len(self.generations)
        corrects = 0
        for input, output, generation in zip(self.input_test_data, self.output_test_data, self.generations):
            if check_different_shapes_of_true_false(output, generation):
                corrects += 1
        self.accuracy = (corrects / total) * 100
        return self.accuracy
    def accuracy(self):
        total = len(self.generations)
        corrects = 0
        for input, output, generation in zip(self.input_test_data, self.output_test_data, self.generations):
            if check_different_shapes_of_true_false(output, generation):
                corrects += 1
        self.accuracy = (corrects / total) * 100
        return self.accuracy

    def statistics(self):
        print("Number of websearches: ", self.websearch_count)
        print("Share of websearches: ", str(
            self.websearch_count * 100 / len(self.generations)) + "%")
        print("Accuracy: ", self.accuracy())
        
    def save_model_outputs(self, save_path: str = 'outputs/PubHealth_outputs.pickle') -> bool:
        with open(save_path, 'wb') as handle:
            pickle.dump({'input_test_data': self.input_test_data, 'generations': self.generations}, handle)
    def load_data_for_test(self, num_data: int = 100):
        random_indices = random.sample(range(len(self.input_test_data)), num_data)
        self.input_test_data = [self.input_test_data[i] for i in random_indices]
        self.output_test_data = [self.output_test_data[i] for i in random_indices]
        self.contexts = [self.contexts[i] for i in random_indices]
