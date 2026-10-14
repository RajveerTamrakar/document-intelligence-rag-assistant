import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Document Intelligence & RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Document Intelligence & RAG Assistant")
st.caption("Ask questions about your documents. Answers are grounded in retrieved passages and include source references.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    top_k = st.slider("Retrieved chunks (k)", 1, 8, 4)
    st.caption("Embeddings run locally using sentence-transformers. Chat answers require a Groq API key.")

api_key = os.getenv("GROQ_API_KEY", "").strip()
if not api_key:
    st.warning("Add GROQ_API_KEY to your .env file before asking questions. You can still upload and index documents.")

uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_uploaded_documents(uploaded_files):
    documents = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = tmp.name
        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(temp_path).load()
            elif suffix == ".txt":
                loaded = TextLoader(temp_path, encoding="utf-8").load()
            else:
                continue
            for doc in loaded:
                doc.metadata["source"] = uploaded.name
                if "page" in doc.metadata:
                    doc.metadata["page"] = int(doc.metadata["page"]) + 1
            documents.extend(loaded)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return documents

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if st.button("Process documents", type="primary", disabled=not uploads):
    try:
        with st.spinner("Extracting text, chunking documents, and creating embeddings…"):
            docs = load_uploaded_documents(uploads)
            if not docs or not any(d.page_content.strip() for d in docs):
                st.error("No readable text was found. If this is a scanned PDF, run OCR first and upload a searchable PDF.")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = splitter.split_documents([d for d in docs if d.page_content.strip()])
                embeddings = get_embeddings()
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.indexed_files = [f.name for f in uploads]
                st.session_state.chunk_count = len(chunks)
                st.success(f"Indexed {len(docs)} document pages/sections into {len(chunks)} chunks.")
    except Exception as exc:
        st.error(f"Could not process documents: {exc}")

if st.session_state.vector_store is not None:
    st.info(f"Indexed files: {', '.join(st.session_state.indexed_files)} · Chunks: {st.session_state.chunk_count}")
    question = st.text_input("Ask a question about your documents", placeholder="What are the main findings?")
    if st.button("Get answer", disabled=not question.strip()):
        if not api_key:
            st.error("Missing GROQ_API_KEY. Add it to .env and restart the app.")
        else:
            try:
                with st.spinner("Retrieving relevant passages and generating an answer…"):
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": top_k})
                    retrieved_docs = retriever.invoke(question)
                    context = "\n\n".join(
                        f"[Source: {d.metadata.get('source', 'Unknown')}, page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
                        for d in retrieved_docs
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are a document question-answering assistant. Answer using ONLY the supplied context. "
                         "If the context does not contain the answer, say: 'I couldn't find that information in the uploaded documents.' "
                         "Do not invent facts or citations. Keep the answer clear and concise.\n\nCONTEXT:\n{context}"),
                        ("human", "{question}")
                    ])
                    llm = ChatGroq(model=model_name, temperature=0, api_key=api_key)
                    response = (prompt | llm).invoke({"context": context, "question": question})
                st.subheader("Answer")
                st.write(response.content)
                st.subheader("Retrieved source passages")
                if not retrieved_docs:
                    st.write("No relevant passages were retrieved.")
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    with st.expander(f"{i}. {source} · Page {page}"):
                        st.write(doc.page_content)
                        st.caption(f"Chunk start index: {doc.metadata.get('start_index', 'N/A')}")
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

with st.expander("How this RAG pipeline works"):
    st.markdown("""
    1. **Ingestion:** extract text from PDF/TXT files.
    2. **Chunking:** split long text into overlapping passages.
    3. **Embeddings:** map passages into vectors using a sentence-transformer.
    4. **Retrieval:** FAISS finds passages semantically similar to the question.
    5. **Generation:** the LLM answers from retrieved context only.
    6. **Grounding:** show retrieved passages and source/page metadata for verification.
    """)
