from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import uuid, base64, re, io
from PIL import Image

def create_retriever(text_summaries, texts, table_summaries, tables, image_summaries, images):
    all_summaries = text_summaries + table_summaries + image_summaries
    all_docs = texts + tables + images

    if not all_summaries or not all_docs or len(all_summaries) != len(all_docs):
        raise ValueError("Document summaries or original documents are empty or mismatched.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [
        Document(page_content=summary, metadata={"doc_id": str(i)})
        for i, summary in enumerate(all_summaries)
    ]

    vectorstore = Chroma.from_documents(docs, embedding=embeddings, collection_name="rag_collection", persist_directory="./chroma_db")

    store = InMemoryStore()
    id_key = "doc_id"

    for i, doc in enumerate(all_docs):
        uid = str(i)
        doc.metadata = {id_key: uid}

    store.mset(list(zip([str(i) for i in range(len(all_docs))], all_docs)))

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever

def split_modal_docs(docs):
    images, texts = [], []
    for doc in docs:
        content = doc.page_content if hasattr(doc, "page_content") else doc
        if re.match("^[A-Za-z0-9+/]+={0,2}$", content):
            try:
                decoded = base64.b64decode(content)[:8]
                if decoded.startswith(b"\xFF\xD8\xFF"):
                    img = Image.open(io.BytesIO(base64.b64decode(content)))
                    buf = io.BytesIO()
                    img.resize((800, 500)).save(buf, format="JPEG")
                    images.append(base64.b64encode(buf.getvalue()).decode())
                    continue
            except:
                pass
        texts.append(content)
    return {"images": images, "texts": texts}

def create_rag_chain(retriever):
    model = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key="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"
    )

    def format_input(inputs):
        question = inputs["question"]
        context = "\n".join(inputs["context"]["texts"])
        messages = [
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in inputs["context"]["images"]
            ],
            {
                "type": "text",
                "text": f"You are a helpful assistant.\nUser question: {question}\nContext:\n{context}"
            },
        ]
        return [HumanMessage(content=messages)]

    chain = (
        {
            "context": retriever | RunnableLambda(split_modal_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(format_input)
        | model
        | StrOutputParser()
    )
    return chain
