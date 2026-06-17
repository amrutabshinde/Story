# Toy RAG — Learning Project

A minimal retrieval-augmented generation (RAG) pipeline over this story bible,
built for learning purposes. See discussion in project chat for context on
why RAG is a good *learning* exercise here but not the recommended tool for
day-to-day story bible lookups (full-file reads work better for continuity
checking — see `notes/loose-ends.md`).

## Setup

```bash
pip install chromadb sentence-transformers
```

## Usage

First run builds the index (chunks every `.md` file by `##` header,
embeds with `all-MiniLM-L6-v2`, stores in local ChromaDB at `.rag_db/`):

```bash
python toy_rag.py "What does Nick wear during the amnesia arc?"
```

Subsequent runs reuse the existing index. Delete `.rag_db/` to rebuild
after editing the markdown files.

## Things to try

**A query RAG handles well** (the answer is concentrated in one section):
```bash
python toy_rag.py "What does Nick wear during the amnesia arc?"
```

**A query that exposes RAG's weakness** (the answer requires connecting
a specific scene to a continuity rule buried in a differently-themed file):
```bash
python toy_rag.py "is it consistent for Nick to tell Leo about the fake death?"
```
Watch whether `notes/loose-ends.md` (which contains the actual answer,
under "Continuity Flags") makes it into the top-k results — it often won't,
because the chunk's wording doesn't closely match the query's wording.

**A structural/meta query:**
```bash
python toy_rag.py "What is the most significant unwritten scene?"
```

## What this demonstrates

- Chunking strategy matters: splitting by `##` keeps related info together
  but can separate a fact from the file-level context that makes it relevant
- Embedding similarity ≠ semantic relevance — a chunk can be "about" the
  right topic in wording but miss the actual rule that governs it
- For small, well-organized, cross-referenced corpora like this one,
  retrieval-by-similarity can underperform simply reading the right
  whole file
