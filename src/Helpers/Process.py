from src.DataLoaders.DataLoader import DataLoader
from src.RetrievalEvaluators.RetrievalEvaluator import RetrievalEvaluator

class Process:
    def __init__(self,dataloader):
        self.dataloader = dataloader
        
    def start(self,generate_response, load_sample_data_only = False, template="default"):
        self.dataloader.template=template
        self.dataloader.load_data()
        if load_sample_data_only:
            self.dataloader.load_data_for_test()
        # self.dataloader.index_documents()
        out = self.dataloader.evaluate_rag_chain(generate_response)
        self.accuracy = out
        # self.dataloader.save_model_outputs(save_path)
        return out