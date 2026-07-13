#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

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
# Retrieval config
# =========================================================
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_POOL = 12

VALID_BEATLES = {"john", "paul", "george", "ringo"}
UPPER_BEATLES = {"JOHN", "PAUL", "GEORGE", "RINGO"}

# Small metadata-based score adjustments
SPEAKER_MATCH_BONUS = 0.08
MENTION_MATCH_BONUS = 0.03
BIO_BONUS = 0.02
FRONT_MATTER_PENALTY = 0.05


# =========================================================
# Loading helpers
# =========================================================
def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")

    if not chunks:
        raise ValueError("No chunks loaded from chunks.jsonl")

    return chunks


def load_index_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    return faiss.read_index(str(path))


# =========================================================
# Embedding helpers
# =========================================================
def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    """
    Embed one question into a normalized vector.
    """
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(f"Unexpected query embedding shape: {vector.shape}")

    return vector.astype("float32")


# =========================================================
# Reranking
# =========================================================
def rerank_results(
    question_beatle: Optional[str],
    candidate_indices: List[int],
    candidate_scores: List[float],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply small metadata-based adjustments after FAISS retrieval.

    This is not a replacement for semantic search.
    It is just a gentle nudge using metadata.
    """
    reranked = []

    target = question_beatle.upper() if question_beatle else None

    for idx, base_score in zip(candidate_indices, candidate_scores):
        chunk = chunks[idx]
        final_score = float(base_score)

        speaker = chunk.get("speaker")
        mentions = chunk.get("mentions", [])
        content_type = chunk.get("content_type")

        reasons = [f"base={base_score:.4f}"]

        if target:
            if speaker == target:
                final_score += SPEAKER_MATCH_BONUS
                reasons.append(f"+speaker_match({SPEAKER_MATCH_BONUS})")

            if target in mentions:
                final_score += MENTION_MATCH_BONUS
                reasons.append(f"+mention_match({MENTION_MATCH_BONUS})")

            if content_type == "bio" and speaker == target:
                final_score += BIO_BONUS
                reasons.append(f"+bio_bonus({BIO_BONUS})")

        if content_type == "front_matter":
            final_score -= FRONT_MATTER_PENALTY
            reasons.append(f"-front_matter({FRONT_MATTER_PENALTY})")

        reranked.append({
            "index_position": idx,
            "chunk": chunk,
            "base_score": float(base_score),
            "final_score": final_score,
            "reasons": reasons,
        })

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked


# =========================================================
# Search
# =========================================================
def search_index(
    question: str,
    chunks: List[Dict[str, Any]],
    index: faiss.Index,
    model: SentenceTransformer,
    beatle: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
) -> List[Dict[str, Any]]:
    """
    Retrieve top candidate chunks, then rerank them slightly using metadata.
    """
    query_vector = embed_query(model, question)

    scores, indices = index.search(query_vector, candidate_pool)

    raw_scores = scores[0].tolist()
    raw_indices = indices[0].tolist()

    # Remove invalid indices in case FAISS returns -1 entries
    filtered_indices = []
    filtered_scores = []
    for idx, score in zip(raw_indices, raw_scores):
        if idx == -1:
            continue
        filtered_indices.append(idx)
        filtered_scores.append(score)

    reranked = rerank_results(
        question_beatle=beatle,
        candidate_indices=filtered_indices,
        candidate_scores=filtered_scores,
        chunks=chunks,
    )

    return reranked[:top_k]


# =========================================================
# Display helpers
# =========================================================
def preview_text(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def print_result(result: Dict[str, Any], rank: int):
    chunk = result["chunk"]

    print("=" * 80)
    print(f"Result #{rank}")
    print(f"Final score: {result['final_score']:.4f}")
    print(f"Base score:  {result['base_score']:.4f}")
    print(f"Why:         {' | '.join(result['reasons'])}")
    print(f"Chunk ID:    {chunk.get('chunk_id')}")
    print(f"Speaker:     {chunk.get('speaker')}")
    print(f"Mentions:    {chunk.get('mentions')}")
    print(f"Type:        {chunk.get('content_type')}")
    print(f"Source file: {chunk.get('source_file')}")
    print(f"Era:         {chunk.get('era')}")
    print(f"Section:     {chunk.get('section_title')}")
    print("-" * 80)
    print(preview_text(chunk.get("text", "")))
    print()


# =========================================================
# Main interactive loop
# =========================================================
def main():
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("Loading index metadata...")
    index_meta = load_index_meta(INDEX_META_PATH)

    print("Loading FAISS index...")
    index = load_faiss_index(INDEX_PATH)

    model_name = index_meta["embedding_model"]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("\nReady.")
    print("You can test retrieval now.")
    print("Example beatles: john, paul, george, ringo")
    print("Press Enter with no question to quit.\n")

    while True:
        beatle = input("Beatle (optional): ").strip().lower()
        if beatle and beatle not in VALID_BEATLES:
            print(f"Invalid beatle. Choose from: {sorted(VALID_BEATLES)}\n")
            continue

        question = input("Question: ").strip()
        if not question:
            print("Goodbye.")
            break

        print("\nSearching...\n")
        results = search_index(
            question=question,
            beatle=beatle if beatle else None,
            chunks=chunks,
            index=index,
            model=model,
            top_k=DEFAULT_TOP_K,
            candidate_pool=DEFAULT_CANDIDATE_POOL,
        )

        if not results:
            print("No results found.\n")
            continue

        for i, result in enumerate(results, start=1):
            print_result(result, i)

        print()


if __name__ == "__main__":
    main()
