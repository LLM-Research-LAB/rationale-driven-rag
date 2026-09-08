from tools.search_tool import web_search_tool
from langchain_core.documents import Document
# from tools.log_tool import logger

def web_search(state):
    question = state["question"]
    documents = state["documents"]
    evaluator = state['retrieval_evaluator']
    dataloader = state['dataloader']
    llm = state['llm']
    index = state['index']
    websearch_tool = web_search_tool(str(state['dataloader']))
    results = websearch_tool.load_results()
    related_results = []
    results['questions'] = dataloader.input_test_data
    related_results= results['output_results'][index]
    # for i , q in enumerate(results['questions']):
    #     # print("here is the q: ", q)
    #     # print("here is the question: ", question)
    #     if q == question:
    #         related_results= results['output_results'][i]
    #         break
    # Here we should select only relevants using our evaluator
    filtered_docs, _ = evaluator.evaluate_websearch(state["question"], related_results, len(documents),index)
    print("Num of Websearch docs: ", len(filtered_docs))
    dataloader.websearch_count += 1
    for d in filtered_docs:
        documents.append(Document(page_content=d, metadata={"source": "Web Search"}))
    # print('Length of documents is: ', len(documents))
    # total_content_length = 0
    # for d in documents:
    #     total_content_length += len(d.page_content)
    # print("The total content length is: ", total_content_length)
    return {"documents": documents, "question": question,'dataloader' : dataloader, 'retrieval_evaluator' : evaluator, 'llm': llm}
