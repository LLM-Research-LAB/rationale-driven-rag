from src.DataLoaders.DataLoader import DataLoader
import json
from langchain_core.prompts.prompt import PromptTemplate
from tqdm import tqdm
import random
import pickle


class TQA(DataLoader):
    def __str__(self):
        return 'tqa'
    # implement constructor if needed

    def __init__(self, contexts_path=None):
        self.template = 'default'
        self.generation_checkpoint = None
        self.websearch_count = 0
        self.strip = False
        self.contexts_path = contexts_path
        pass

    def load_data(self):
        file = open('src/Dataset/eval_data/triviaqa_test_w_gs.jsonl',
                    'r', encoding='utf-8')
        lines = file.readlines()
        self.input_test_data = []
        self.output_test_data = []
        self.contexts = []
        for line in lines:
            data = json.loads(line)
            self.input_test_data.append(data['question'].strip())
            self.output_test_data.append(data['answers'])
            docs = []
            for doc in data['ctxs']:
                docs.append(doc['title'].strip() + '//' + doc['text'].strip().replace("\n" , " "))
            self.contexts.append(docs)
        assert len(self.contexts) == len(
            self.input_test_data) == len(self.output_test_data)
        self.data = list(zip(self.input_test_data, self.output_test_data))
        if self.strip:
            self.strip_contexts()
        if self.contexts_path:
            self.load_contexts(path=self.contexts_path)
        return self.data

    # def index_documents(self):
    #     from models.EM import embedding
    #     file_path = "/content/data/corpora/wiki/enwiki-dec2020/text-list-100-sec.jsonl"
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         lines = file.readlines()
    #         for line in lines:
    #           data = json.loads(line)
    #           # docs = [{'text': item['text']} for item in content]
    #           # print(data.keys())
    #           break

    #     # splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     #     chunk_size=1024, chunk_overlap=128
    #     # ).split_documents(docs)

    #     # vectorstore = FAISS.from_documents(splits, embedding)
    #     # vectorstore.save_local("popqa_index")

    def train_test_split(self, split_point: float = 0.8):
        return None, self.data

    def get_llm_prompt_template(self, template='default'):
        if template == 'default':
            prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: {context}\n\nInstruction: provide a long-form answer including correct answers to the question. \n\n Question: {question}\n\nAnswer:"
            return PromptTemplate(
                template=prompt,
                input_variables=["context", "question"],
            )
        else:
            return PromptTemplate(
                template="### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n\n### Input:\n{question}\n\n### Response:\n{context}",
                input_variables=["context", "question"],
            )
            # return PromptTemplate(
            #     template="### Instruction:\n{question}\n\n### Response:\n{context}",
            #     input_variables=["context", "question"],
            # )
        
        # return PromptTemplate(
        #     template="### Instruction:\n{question}\n\n### Response:\n{context}",
        #     input_variables=["context", "question"],
        # )

        # elif template == 'self-rag':
        #     return PromptTemplate(
        #         template="### Instruction:\nProvide a short answer to the input question based on the retrieval information. The answer should be one or two words.\n\n### Input:\n{question}\n[Retrieval]<paragraph>{context}</paragraph>[/Retrieval]\n\n### Response:\n",
        #         input_variables=["context", "question"],
        #     )

    def evaluate_rag_chain(self, rag_chain) -> float:
        if self.generation_checkpoint is None:
            self.generation_checkpoint = 0
            self.generations = []
        index = 0
        for input in tqdm(self.input_test_data[self.generation_checkpoint:], "evaluating on tqa... " if self.generation_checkpoint == 0 else "continuing evaluation from checkpoint..."):
            response = rag_chain(input.strip().replace(
                '\n', ' '), index)
            # print("the response of network is: ", response)
            self.generations.append(
                response.strip().replace('\n', ' ').lower())
            index += 1
        accuracy = self.accuracy()
        return accuracy

    def save_model_outputs(self, save_path: str = 'outputs/tQA_outputs.txt') -> bool:
        with open(save_path, 'wb') as handle:
            pickle.dump({'input_test_data': self.input_test_data,
                        'generations': self.generations}, handle)

    def load_data_for_test(self, num_data: int = 100):
        random_indices = random.sample(range(len(self.input_test_data)), num_data)
        self.input_test_data = [self.input_test_data[i] for i in random_indices]
        self.output_test_data = [self.output_test_data[i] for i in random_indices]
        self.contexts = [self.contexts[i] for i in random_indices]

    def accuracy(self):
      total = len(self.generations)
      corrects = 0
      for input, output, generation in zip(self.input_test_data, self.output_test_data, self.generations):
          possible_answers = output
          for ans in possible_answers:
            if ans.lower() in generation.lower():
              corrects += 1
              print(
                  f"Correct! Input: {input}, Output: {output}, Generation: {generation}")
              break
          else:
            print(
                f"Incorrect! Input: {input}, Output: {output}, Generation: {generation}")
      self.acc = (corrects / total) * 100
      print(f"Total: {total}, Corrects: {corrects}, Accuracy: {self.acc}")
      return self.acc

    def statistics(self):
        print("Number of websearches: ", self.websearch_count)
        print("Share of websearches: ", str(
            self.websearch_count * 100 / len(self.generations)) + "%")
        print("Accuracy: ", self.accuracy())
