# Beatles Interview Chatbot – FastAPI RAG Backend

A FastAPI backend for a Beatles “interview chatbot” that answers user questions in a lightly persona-shaped Beatles voice while grounding responses in retrieved source material from *The Beatles Anthology* corpus.

## Features

- **Beatle-specific interview mode**
  - `john`
  - `paul`
  - `george`
  - `ringo`
- **RAG pipeline**
  - cleaned source text
  - chunked retrieval corpus
  - local embeddings
  - FAISS vector search
- **Grounded generation**
  - retrieves relevant Beatles text before answering
  - aims to avoid invented facts
- **Light persona styling**
  - John: dry, witty, dark/goofy
  - Paul: upbeat, confident, slightly domineering
  - George: brooding, thoughtful, reflective
  - Ringo: cheerful, friendly, warm
- **FastAPI endpoint**
  - `POST /interview`

---

## Architecture

This backend is the RAG/generation layer in a larger app stack:

- **React terminal frontend**
- **Express proxy**
- **FastAPI backend** ← this repo/component

### Backend flow

1. Receive `{ beatle, question }`
2. Retrieve relevant chunks from local FAISS index
3. Build a grounded prompt from retrieved excerpts
4. Send prompt to OpenRouter for generation
5. Return answer JSON

---

## Project Structure

```text
.
├── main.py
├── README.md
├── artifacts/
│   ├── chunks.jsonl
│   ├── faiss.index
│   └── index_meta.json
├── app/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       ├── retriever.py
│       ├── prompting.py
│       └── interview_service.py
└── scripts/
    ├── build_chunks.py
    ├── build_index.py
    └── test_retrieval.py
