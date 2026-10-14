"""
Small, transparent retrieval evaluation helper.
1) Put your test documents in a folder.
2) Update evaluation/eval_questions.json with questions and expected_source filenames.
3) Run: python evaluation/evaluate_retrieval.py --docs ./test_docs --k 4
This evaluates whether the expected source file appears in top-k retrieved chunks.
Answer correctness and unsupported-answer rates should be reviewed separately with a human rubric.
"""
import argparse
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_docs(folder):
    docs = []
    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif path.suffix.lower() == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Folder containing evaluation PDF/TXT documents")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--questions", default="evaluation/eval_questions.json")
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        tests = json.load(f)

    docs = load_docs(args.docs)
    if not docs:
        raise SystemExit("No PDF/TXT documents found in the specified folder.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, add_start_index=True)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(chunks, embeddings)
    hits = 0

    for test in tests:
        found = store.similarity_search(test["question"], k=args.k)
        sources = [d.metadata.get("source", "") for d in found]
        hit = test.get("expected_source") in sources
        hits += int(hit)
        print(f"\nQuestion: {test['question']}")
        print(f"Expected source: {test.get('expected_source')}")
        print(f"Retrieved sources: {sources}")
        print(f"Expected source in top-{args.k}: {'YES' if hit else 'NO'}")

    total = len(tests)
    print(f"\nSource Hit Rate@{args.k}: {hits}/{total} = {hits / total:.1%}" if total else "No evaluation questions.")
    print("Note: source hit rate is only a retrieval proxy; it does not measure answer correctness or hallucination rate.")

if __name__ == "__main__":
    main()
