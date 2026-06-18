"""
toy_rag.py — A minimal RAG pipeline over the Forget Me Not story bible.

Learning project: chunk markdown files by section, embed them,
store in a local ChromaDB, and retrieve relevant chunks for a query.

Install:
    pip install chromadb sentence-transformers

Usage:
    python toy_rag.py "is it consistent for Nick to tell Leo about the fake death?"
"""

import os
import re
import sys
import glob
import chromadb
from sentence_transformers import SentenceTransformer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(REPO_ROOT, ".rag_db")
TOP_K = 4


def chunk_markdown(filepath):
    """Split a markdown file into chunks by ## headers.
    Each chunk = (header_title, text, source_file)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on level-2 headers, keep the header with its content
    parts = re.split(r"\n(?=## )", content)
    chunks = []
    rel_path = os.path.relpath(filepath, REPO_ROOT)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split("\n", 1)[0]
        title = first_line.lstrip("# ").strip() or "Untitled"
        # Skip trivially short chunks (e.g. stray separators)
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
    # Reset collection each run for simplicity
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

    if not os.path.exists(DB_DIR):
        print("No index found. Building one now (this happens once)...\n")
        model, collection = build_index()
    else:
        model, collection = load_index()

    query(model, collection, question)
