from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rag_chain(prompt_template, llm):
    return prompt_template | llm | StrOutputParser()


def formatter(docs, template):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_stripped(sentences):
        if len(sentences) > 0:
            knowledge_1 = ""
            knowledge_2 = ""
            internal_knowledge_docs = []
            external_knowledge_docs = []
            for s in sentences:
                if s.metadata['source'] == "Web Search":
                    external_knowledge_docs.append(s)
                else:
                    internal_knowledge_docs.append(s)
            if len(internal_knowledge_docs) > 0:
                if len(external_knowledge_docs) > 0:
                    knowledge_1 = "Knowledge1: " + " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in internal_knowledge_docs)
                else:
                    knowledge_1 = " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in internal_knowledge_docs)
            if len(external_knowledge_docs) > 0:
                if len(internal_knowledge_docs) > 0:
                    knowledge_2 = " [sep] Knowledge2: " +  " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in external_knowledge_docs)
                else:
                    knowledge_2 = " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in external_knowledge_docs)
            return "[Retrieval]<paragraph>" + knowledge_1 + knowledge_2 + "</paragraph>"
        return ""
    def format_docs_stripped_llama(sentences):
        if len(sentences) > 0:
            return " ".join(sentence.page_content.strip().replace("\n", " ") for sentence in sentences)
        return ""

    def format_passage_docs(docs):
        if len(docs) > 0:
            return "[Retrieval]<paragraph>" + "\n".join(doc.page_content for doc in docs) + "</paragraph>"
        return ""
    if template == 'default':
        return format_docs_stripped_llama(docs)
    else:
        return format_docs_stripped(docs)
