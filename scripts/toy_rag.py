"""
toy_rag.py — A minimal RAG pipeline over the Forget Me Not story bible.

Learning project: chunk markdown files by section, embed them,
store in a local ChromaDB, and retrieve relevant chunks for a query.

Install:
    pip install chromadb sentence-transformers

Usage:
    python toy_rag.py "is it consistent for Nick to tell Leo about the fake death?"
"""

import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

import chromadb
from sentence_transformers import SentenceTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(REPO_ROOT, ".rag_db")
TOP_K = 10
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


def chunk_markdown(filepath):
    """Split a markdown file into chunks by ## headers.
    Each chunk = (header_title, text, source_file)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"\n(?=## )", content)
    chunks = []
    rel_path = os.path.relpath(filepath, REPO_ROOT)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split("\n", 1)[0]
        title = first_line.lstrip("# ").strip() or "Untitled"
        if len(part) < 20:
            continue
        chunks.append((title, part, rel_path))

    return chunks


def build_index():
    """Walk the repo, chunk all .md files, embed, and store in ChromaDB."""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection("story_bible")
    except Exception:
        pass
    collection = client.create_collection("story_bible")

    md_files = glob.glob(os.path.join(REPO_ROOT, "**", "*.md"), recursive=True)
    md_files = [f for f in md_files if ".rag_db" not in f]

    all_chunks = []
    for filepath in md_files:
        all_chunks.extend(chunk_markdown(filepath))

    print(f"Found {len(md_files)} markdown files, {len(all_chunks)} chunks.")

    ids = []
    documents = []
    metadatas = []

    for i, (title, text, source) in enumerate(all_chunks):
        ids.append(f"chunk-{i}")
        documents.append(text)
        metadatas.append({"title": title, "source": source})

    print("Embedding chunks...")
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} chunks into {DB_DIR}")
    return model, collection


def load_index():
    """Load an existing index without rebuilding."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("story_bible")
    return model, collection


def ensure_index():
    """Load the existing index when it has data, otherwise rebuild it."""
    if not os.path.exists(DB_DIR):
        print("No index found. Building one now (this happens once)...\n")
        return build_index()

    try:
        model, collection = load_index()
        if collection.count() > 0:
            return model, collection
    except Exception:
        pass

    print("Existing index is empty or unreadable. Rebuilding it now...\n")
    return build_index()


def build_context(results):
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    context_parts = []

    for doc, meta in zip(docs, metadatas):
        if not doc or not str(doc).strip():
            continue
        title = meta.get("title", "Untitled") if meta else "Untitled"
        source = meta.get("source", "unknown") if meta else "unknown"
        text = str(doc).strip()
        if len(text) > 1800:
            text = text[:1800] + "..."
        context_parts.append(f"[{source}] {title}\n{text}")

    if not context_parts:
        return None

    return "\n\n".join(context_parts[:3])


def ask_llm(question, context):
    """Call a local Ollama server if it is available."""
    prompt = f"""You are helping review a story bible.
Answer the user's question using only the provided story context.
If the answer is not clearly supported by the context, say that clearly.

Question: {question}

Story context:
{context}
"""

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as exc:
        print(f"\nOllama request failed: {exc}")
        return None


def query(model, collection, question, top_k=TOP_K):
    embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
    )

    print(f"\nQuery: {question}\n")
    print(f"Top {top_k} retrieved chunks (by embedding similarity):\n")
    print("-" * 60)

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"[{meta['source']}] — {meta['title']}  (distance: {dist:.4f})")
        preview = doc[:300].replace("\n", " ")
        print(f"  {preview}...")
        print("-" * 60)

    context = build_context(results)

    if not context:
        print("\nNo relevant story context was retrieved from the index.")
        return results

    llm_answer = ask_llm(question, context)

    if llm_answer:
        print("\nLLM answer:\n")
        print(llm_answer)
    else:
        print("\nLLM answer unavailable. Install Ollama and run 'ollama pull llama3.2' to enable this feature.")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python toy_rag.py \"your question here\"")
        print("\nTip: try these to see RAG's strengths and weaknesses:")
        print('  python toy_rag.py "What does Nick wear during the amnesia arc?"')
        print('  python toy_rag.py "is it consistent for Nick to tell Leo about the fake death?"')
        print('  python toy_rag.py "What is the most significant unwritten scene?"')
        sys.exit(1)

    question = sys.argv[1]
    model, collection = ensure_index()
    query(model, collection, question)
