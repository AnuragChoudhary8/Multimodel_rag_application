from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

def create_summary_chain(task_type="text"):
    if task_type == "text":
        template = """You are an assistant tasked with summarizing text for retrieval. These summaries will be embedded and used to retrieve the raw text elements. Give a concise summary: {element}"""
    else:
        template = """You are an assistant tasked with summarizing tables for retrieval. These summaries will be embedded and used to retrieve the raw table elements. Give a concise summary: {element}"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatGroq(temperature=0, model="llama3-8b-8192", api_key="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")
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def summarize_elements(elements, chain):
    # Filter out empty or whitespace-only elements before summarizing
    clean_elements = [el for el in elements if el and el.strip()]
    if not clean_elements:
        return []
    return chain.batch(clean_elements, {"max_concurrency": 1})
