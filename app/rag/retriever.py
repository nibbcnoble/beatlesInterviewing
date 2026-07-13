from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"

# =========================================================
# Retrieval config
# =========================================================
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_POOL = 12

VALID_BEATLES = {"john", "paul", "george", "ringo"}

SPEAKER_MATCH_BONUS = 0.08
MENTION_MATCH_BONUS = 0.03
BIO_BONUS = 0.02
FRONT_MATTER_PENALTY = 0.05


class BeatlesRetriever:
    """
    Loads chunk metadata, FAISS index, and embedding model once,
    then provides a retrieve() method for semantic search.
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        index_path: Path = INDEX_PATH,
        index_meta_path: Path = INDEX_META_PATH,
    ):
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.index_meta_path = index_meta_path

        self.chunks = self._load_chunks(self.chunks_path)
        self.index_meta = self._load_index_meta(self.index_meta_path)
        self.index = self._load_faiss_index(self.index_path)

        model_name = self.index_meta["embedding_model"]
        self.model = SentenceTransformer(model_name)

        self._validate_index_alignment()

    # =====================================================
    # Loading helpers
    # =====================================================
    def _load_chunks(self, path: Path) -> List[Dict[str, Any]]:
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
                    raise ValueError(f"Invalid JSON in chunks file at line {line_number}: {e}")

        if not chunks:
            raise ValueError("No chunks loaded from chunks.jsonl")

        return chunks

    def _load_index_meta(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Index metadata file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_faiss_index(self, path: Path) -> faiss.Index:
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        return faiss.read_index(str(path))

    def _validate_index_alignment(self) -> None:
        """
        Sanity checks:
        - chunk count matches FAISS vector count
        - metadata agrees with actual index
        """
        chunk_count = len(self.chunks)
        vector_count = self.index.ntotal
        meta_chunk_count = self.index_meta.get("chunk_count")

        if chunk_count != vector_count:
            raise ValueError(
                f"Chunk count ({chunk_count}) does not match FAISS vector count ({vector_count})"
            )

        if meta_chunk_count is not None and meta_chunk_count != chunk_count:
            raise ValueError(
                f"index_meta chunk_count ({meta_chunk_count}) does not match actual chunk count ({chunk_count})"
            )

    # =====================================================
    # Embedding helpers
    # =====================================================
    def _embed_query(self, question: str) -> np.ndarray:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        vector = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(f"Unexpected query embedding shape: {vector.shape}")

        return vector.astype("float32")

    # =====================================================
    # Reranking helpers
    # =====================================================
    def _rerank_results(
        self,
        beatle: Optional[str],
        candidate_indices: List[int],
        candidate_scores: List[float],
    ) -> List[Dict[str, Any]]:
        reranked = []

        target = beatle.upper() if beatle else None

        for idx, base_score in zip(candidate_indices, candidate_scores):
            chunk = self.chunks[idx]
            final_score = float(base_score)

            speaker = chunk.get("speaker")
            mentions = chunk.get("mentions", [])
            content_type = chunk.get("content_type")

            if target:
                if speaker == target:
                    final_score += SPEAKER_MATCH_BONUS

                if target in mentions:
                    final_score += MENTION_MATCH_BONUS

                if content_type == "bio" and speaker == target:
                    final_score += BIO_BONUS

            if content_type == "front_matter":
                final_score -= FRONT_MATTER_PENALTY

            reranked.append({
                "chunk": chunk,
                "base_score": float(base_score),
                "score": final_score,
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # =====================================================
    # Public API
    # =====================================================
    def retrieve(
        self,
        question: str,
        beatle: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a question.

        Returns a list of dictionaries containing:
        - chunk metadata
        - chunk text
        - retrieval score
        - base vector similarity score
        """
        if beatle:
            beatle = beatle.lower().strip()
            if beatle not in VALID_BEATLES:
                raise ValueError(f"Invalid beatle '{beatle}'. Must be one of {sorted(VALID_BEATLES)}")

        query_vector = self._embed_query(question)

        scores, indices = self.index.search(query_vector, candidate_pool)
        raw_scores = scores[0].tolist()
        raw_indices = indices[0].tolist()

        filtered_indices = []
        filtered_scores = []

        for idx, score in zip(raw_indices, raw_scores):
            if idx == -1:
                continue
            filtered_indices.append(idx)
            filtered_scores.append(score)

        reranked = self._rerank_results(
            beatle=beatle,
            candidate_indices=filtered_indices,
            candidate_scores=filtered_scores,
        )

        return reranked[:top_k]


# Optional singleton helper
_retriever_instance: Optional[BeatlesRetriever] = None


def get_retriever() -> BeatlesRetriever:
    """
    Lazily initialize one retriever instance and reuse it.
    This is useful in web apps so the model and FAISS index
    are not reloaded on every request.
    """
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = BeatlesRetriever()

    return _retriever_instance
