from __future__ import annotations

import os
from typing import Dict, Any, Optional

import requests

from app.rag.retriever import get_retriever
from app.rag.prompting import build_interview_prompt, build_fallback_answer

# =========================================================
# OpenRouter config
# =========================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Retrieval config
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_POOL = 12

# Generation config
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 350


class InterviewService:
    """
    High-level service for Beatles interview responses.

    Responsibilities:
    - retrieve relevant chunks
    - build a grounded prompt
    - call OpenRouter for generation
    - return the final answer
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = OPENROUTER_MODEL,
    ):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = openrouter_model
        self.retriever = get_retriever()

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

    def answer_question(self, beatle: str, question: str) -> Dict[str, Any]:
        beatle = beatle.lower().strip()
        question = question.strip()

        if not beatle:
            raise ValueError("Beatle is required")

        if not question:
            raise ValueError("Question is required")

        retrieved_chunks = self.retriever.retrieve(
            question=question,
            beatle=beatle,
            top_k=RETRIEVAL_TOP_K,
            candidate_pool=RETRIEVAL_CANDIDATE_POOL,
        )

        if not retrieved_chunks:
            return {
                "beatle": beatle,
                "answer": build_fallback_answer(beatle),
            }

        prompt = build_interview_prompt(
            beatle=beatle,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        try:
            answer = self._generate_with_openrouter(prompt)
        except Exception:
            answer = build_fallback_answer(beatle)

        if not answer.strip():
            answer = build_fallback_answer(beatle)

        return {
            "beatle": beatle,
            "answer": answer.strip(),
        }

    def _generate_with_openrouter(self, prompt: str) -> str:
        """
        Send the grounded prompt to OpenRouter and return the generated answer.
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.openrouter_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": GENERATION_TEMPERATURE,
            "max_tokens": GENERATION_MAX_TOKENS,
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "").strip()


# Optional singleton helper
_interview_service_instance: Optional[InterviewService] = None


def get_interview_service() -> InterviewService:
    global _interview_service_instance

    if _interview_service_instance is None:
        _interview_service_instance = InterviewService()

    return _interview_service_instance
