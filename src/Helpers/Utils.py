
def check_different_shapes_of_true_false(first : str , second : str) -> bool:
    if (first == 'True' or first == 'TRUE' or first == 'true' or first == 'TRUE.' or first == 'true.' or first == 'True.') and (second == 'True' or second == 'TRUE' or second == 'true' or second == 'TRUE.' or second == 'true.' or second == 'True.'):
        return True
    elif (first == 'False' or first == 'FALSE' or first == 'false' or first == 'FALSE.' or first == 'false.' or first == 'False.') and (second == 'False' or second == 'FALSE' or second == 'false' or second == 'FALSE.' or second == 'false.' or second == 'False.'):
        return True
    elif first == second:
        return True
    elif first.lower() in second.lower():
        return True
    else:
        return False
    
    
def check_different_shapes_of_choices(first : str , second : str) -> bool:
    if (first == 'A' or first == 'a' or first == 'A.' or first == 'A)' or first == '1') and (second == 'A)' or second == 'a' or second == 'A.' or first == 'A)' or first == '1'):
        return True
    elif (first == 'B' or first == 'b' or first == 'B.' or first == 'B)' or first == '2') and (second == 'B' or second == 'b' or second == 'B.' or first == 'B)' or first == '2'):
        return True
    elif (first == 'C' or first == 'c' or first == 'C.' or first == 'C)' or first == '3') and (second == 'C' or second == 'c' or second == 'C.' or first == 'C)' or first == '3'):
        return True
    elif (first == 'D' or first == 'd' or first == 'D.' or first == 'D)' or first == '4') and (second == 'D' or second == 'd' or second == 'D.' or first == 'D)' or first == '4'):
        return True
    elif first == second: #for 1) and 2) etc
        return True
    else:
        return False


def index_in_chroma_db(file_path="/content/drive/MyDrive/CRAG/ARC-V1-Feb2018-2/ARC_Corpus.txt", db_path='./chroma_db'):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from tqdm import tqdm
    with open(file_path, 'r') as f:
        counter = 0
        documents = []
        page_content = ""
        uuids = []
        for line in tqdm(f, 'processing line by line ...'):
            page_content += line
            if counter % 100 == 0:
                documents.append(Document(page_content=page_content))
                uuids.append(str(counter / 100))
                page_content = ""
                counter += 1

    # saving to vector store
    vector_store = Chroma(
        collection_name="ARC",
        embedding_function=embedding,
        # Where to save data locally, remove if not necessary
        persist_directory=db_path,
    )
    for i in tqdm(range(100, len(documents), 100), 'saving to vector store ...'):
        vector_store.add_documents(documents=documents[i-100:i], ids=uuids[i-100:i])

# not using below function anymore
def get_retrieved_docs(txt_file_path):
    contexts = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        num_question = -1
        docs = []
        current_question = ''
        for line in f:
            q, doc = line.split('[SEP]')
            q , doc = q.strip() , doc.strip()
            if len(doc) == 0:
                continue
            if q != current_question:
                num_question += 1
                current_question = q
                if len(docs) > 0:
                    contexts.append(docs)
                docs = []
            docs.append(doc)
    return contexts

def load_model_from_hf(model_id='mistralai/Mistral-7B-Instruct-v0.3',batch_size=1):
    import transformers
    import torch
    import os
    hf_token=os.environ['hf_token']
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={
        "torch_dtype": torch.bfloat16,
        # "quantization_config": bnb_config
        }, token=hf_token,batch_size=batch_size, device_map="auto")
    return pipeline

def call_pipeline(pipeline, inputs, max_generation_length=50):
    out = pipeline(inputs, do_sample=True,
                  max_new_tokens=max_generation_length, temperature=0.01, top_p=1.0, return_full_text=False)
    return out[0]['generated_text']


def process_arc_choices(choice):
    if choice == "1":
        return "A"
    if choice == "2":
        return "B"
    if choice == "3":
        return "C"
    if choice == "4":
        return "D"
    return choice