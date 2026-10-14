# Document Intelligence & RAG Assistant

A complete starter project for asking questions about PDF/TXT documents with retrieval-augmented generation (RAG). It extracts text, chunks documents, creates local embeddings, indexes them in FAISS, retrieves relevant passages, and asks an LLM to answer from that context. The UI displays source filenames and page numbers so users can verify answers.

## Features

- Upload multiple PDF and TXT files
- Text extraction and overlapping chunking
- Local sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- FAISS vector similarity search
- LLM answers through Groq's API
- Source passages with document/page references
- Adjustable chunk size, overlap, and retrieval `k`
- A simple retrieval evaluation script with Source Hit Rate@k

## Architecture

`PDF/TXT → text extraction → chunking → embeddings → FAISS index → top-k retrieval → context-grounded LLM response → source references`

## Setup

Recommended: Python 3.10 or 3.11.

### 1. Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your LLM API key

Copy `.env.example` to `.env`, then add your Groq API key:
```env
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
Never commit `.env` or publish your API key. Provider model availability and free-tier limits can change; if the configured model is unavailable, set `GROQ_MODEL` to a currently supported model in your account.

### 4. Run the app
```bash
streamlit run app.py
```
Upload a searchable PDF or TXT file, click **Process documents**, enter a question, and click **Get answer**.

## Evaluate retrieval

1. Put test PDFs/TXT files in `test_docs/`.
2. Edit `evaluation/eval_questions.json`. Use exact filenames for `expected_source`, and write representative questions.
3. Run:
```bash
python evaluation/evaluate_retrieval.py --docs ./test_docs --k 4
```

This reports Source Hit Rate@k: how often the expected source appears among the retrieved chunks. This is a retrieval proxy, not a complete RAG evaluation. To support a stronger resume claim, also score answer correctness and unsupported answers against a manually reviewed question set, and report actual measured results only.

## Current limitations

- Scanned/image-only PDFs may need OCR before upload.
- The index is created in session memory; uploaded files must be processed again after restarting the app.
- Source references identify retrieved passages, but citations do not guarantee that every answer claim is supported. Always inspect the source passages.
- This starter does not provide authentication, multi-user isolation, persistent storage, or production security controls.

## Resume alignment

Once you have tested it with your own documents and evaluation set, you can accurately describe:
- A PDF/TXT RAG pipeline with chunking, embeddings, FAISS retrieval, and LLM-generated context-grounded answers.
- Source/page references and configurable retrieval settings.
- Retrieval evaluation using Source Hit Rate@k, with answer correctness and unsupported-answer checks performed separately.

Do not add performance percentages to your resume unless you measured them on your evaluation set.
