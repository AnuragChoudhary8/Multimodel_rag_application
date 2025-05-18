import streamlit as st
from utils.pdf_parser import extract_elements_from_pdf
from utils.summarizer import create_summary_chain, summarize_elements
from utils.image_utils import summarize_all_images
from utils.rag_chain import create_retriever, create_rag_chain
from langchain_core.documents import Document
import os

st.set_page_config(layout="wide")
st.title("📄🔍 Multimodal RAG PDF Assistant")

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf:
    pdf_path = f"temp_{uploaded_pdf.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("Extracting data from PDF...")
    elements = extract_elements_from_pdf(pdf_path)

    Text, Table, Image = [], [], []

    for el in elements:
        t = str(type(el))
        if "NarrativeText" in t:
            Text.append(str(el))
        elif "Table" in t:
            Table.append(str(el))
        elif "Image" in t:
            Image.append(el)

    text_docs, table_docs, image_docs = [], [], []
    text_summaries, table_summaries, image_summaries, img_base64s = [], [], [], []

    if Text:
        st.success("Summarizing Text...")
        text_summary_chain = create_summary_chain("text")
        text_summaries = summarize_elements(Text, text_summary_chain)
        text_docs = [Document(page_content=txt) for txt in Text]

    if Table:
        st.success("Summarizing Tables...")
        table_summary_chain = create_summary_chain("table")
        table_summaries = summarize_elements(Table, table_summary_chain)
        table_docs = [Document(page_content=tbl) for tbl in Table]

    if Image:
        st.success("Summarizing Images...")
        img_base64s, image_summaries = summarize_all_images(Image)
        image_docs = [Document(page_content=img) for img in img_base64s]

    # Combine summaries/docs while ensuring no empty list is passed
    all_summaries = text_summaries + table_summaries + image_summaries
    all_docs = text_docs + table_docs + image_docs

    if not all_summaries or not all_docs:
        st.error("❌ No valid summaries or documents to build retriever.")
        st.stop()

    if len(all_summaries) != len(all_docs):
        st.error(f"❌ Mismatch: {len(all_summaries)} summaries vs {len(all_docs)} documents.")
        st.stop()

    st.success("Creating retriever...")
    retriever = create_retriever(
        text_summaries, text_docs,
        table_summaries, table_docs,
        image_summaries, image_docs
    )

    st.success("Building RAG chain...")
    chain = create_rag_chain(retriever)

    st.subheader("Ask a question about your document")
    user_q = st.text_input("Question")

    if user_q:
        with st.spinner("Generating answer..."):
            answer = chain.invoke(user_q)
            st.markdown("### 🤖 Answer")
            st.write(answer)

    if os.path.exists(pdf_path):
        os.remove(pdf_path)
