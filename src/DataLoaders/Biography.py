from src.DataLoaders.DataLoader import DataLoader
import pandas as pd  
import json
from factscore.factscorer import FactScorer
from langchain_core.prompts.prompt import PromptTemplate

class Biography(DataLoader):
    def __str__(self):
        return 'bio'
    #implement constructor if needed
    def __init__(self):
        pass
    
    def load_data(self):
        file = open(
            'src/Dataset/eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl', 'r', encoding='utf-8')
        lines = file.readlines()
        self.input_test_data = []
        self.contexts = []
        for line in lines:
            docs = []
            data = json.loads(line)
            # print("data : " , data)
            self.input_test_data.append(data['input'].strip())
            # print("data['ctxs'] : " , data['ctxs'])
            for doc in data['ctxs']:
                # print(doc['text'])
                docs.append(doc['text'].strip())
            self.contexts.append(docs)
        assert len(self.contexts) == len(self.input_test_data)
    def index_documents(self):
        pass
    
    def train_test_split(self , split_point : float=0.8) -> tuple:
        # all data is for testing
        return None , self.input_test_data
    
    def get_llm_prompt_template(self):  
        prompt = PromptTemplate(  
            template="""You are an AI model responsible for providing complete answers to people's biographies. \n 
            Here is the context:{context} \n
            Here is the question: \n\n {question} \n\n  
            Provide a complete biography based on the help context.""",  
            input_variables=["generation", "context", "question"],
        )  
        return prompt
    
    def evaluate_rag_chain(self, rag_chain) -> float:
        pass
    
    def save_model_outputs(self, save_path: str = '/biography_llama_outputs.txt') -> bool:
        pass
    # pass atomic_facts if preprocessed generations
    def __calculate_factscore(topics , generations , atomic_facts):
        fs = FactScorer()
        # topics = ['Lanny Flaherty']
        # generations = ['Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry.']
        # atomic_facts = [["He has appeared in television shows.",
        #                 "He has appeared in Law & Order."]]

        out = fs.get_score(topics, generations, atomic_facts=atomic_facts, gamma=10)
        # print(out["score"])  # FActScore
        # print(out["init_score"])  # FActScore w/o length penalty
        # print(out["respond_ratio"])  # % of responding (not abstaining from answering)
        # print(out["num_facts_per_response"])
        return out
    #TODO : skip for now
    def load_data_for_test(self, num_data: int = 50):
        # self.input_test_data = self.input_test_data[:num_data]
        # self.output_test_data = self.output_test_data[:num_data]
        # self.contexts = self.contexts[:num_data]
        pass
