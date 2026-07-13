#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"

# =========================================================
# Embedding model config
# =========================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# =========================================================
# Helpers
# =========================================================
def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """
    Read chunks from JSONL.
    Each line is one JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                chunks.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")

    if not chunks:
        raise ValueError("No chunks were loaded from chunks.jsonl")

    return chunks


def extract_texts(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Pull the main text field from each chunk.
    """
    texts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            raise ValueError(f"Chunk at position {i} is missing text")
        texts.append(text)
    return texts


def build_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Convert chunk texts into embedding vectors.

    We normalize embeddings so that inner product acts like cosine similarity.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a simple exact-search FAISS index using inner product.

    Because embeddings are normalized, inner product ~= cosine similarity.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_index_metadata(
    chunks: List[Dict[str, Any]],
    embedding_model_name: str,
    embedding_dim: int
) -> Dict[str, Any]:
    """
    Save metadata needed later when loading and querying the index.
    """
    return {
        "embedding_model": embedding_model_name,
        "embedding_dimension": embedding_dim,
        "chunk_count": len(chunks),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "chunks_path": str(CHUNKS_PATH),
    }


# =========================================================
# Main
# =========================================================
def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    texts = extract_texts(chunks)
    print(f"Loaded {len(chunks)} chunks")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nBuilding embeddings...")
    embeddings = build_embeddings(model, texts, batch_size=BATCH_SIZE)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving FAISS index to: {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))

    meta = build_index_metadata(
        chunks=chunks,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_dim=embeddings.shape[1],
    )

    print(f"Saving index metadata to: {INDEX_META_PATH}")
    with INDEX_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Indexed {len(chunks)} chunks")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"FAISS total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
