from __future__ import annotations

from typing import List, Dict, Any

VALID_BEATLES = {"john", "paul", "george", "ringo"}


def get_beatle_style(beatle: str) -> str:
    """
    Return a brief style instruction for the selected Beatle.

    The key idea:
    style should be light, not overpowering.
    We want flavor, not roleplay invention.
    """
    beatle = beatle.lower().strip()

    if beatle == "john":
        return (
            "Write in a lightly John-like voice: dry, witty, a little dark, "
            "sometimes playful or cutting, but still clear and conversational."
        )
    elif beatle == "paul":
        return (
            "Write in a lightly Paul-like voice: upbeat, confident, energetic, "
            "warm but a bit pushy or insistent when appropriate."
        )
    elif beatle == "george":
        return (
            "Write in a lightly George-like voice: thoughtful, inward, brooding, "
            "spiritually aware, and reflective."
        )
    elif beatle == "ringo":
        return (
            "Write in a lightly Ringo-like voice: friendly, cheerful, modest, "
            "plainspoken, and easygoing."
        )

    raise ValueError(f"Invalid beatle '{beatle}'. Must be one of {sorted(VALID_BEATLES)}")


def format_sources_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a readable prompt section.

    We include:
    - source number
    - chunk id
    - speaker
    - source file
    - era
    - content type
    - chunk text
    """
    formatted_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        chunk = item["chunk"]

        source_block = (
            f"[Source {i}]\n"
            f"chunk_id: {chunk.get('chunk_id')}\n"
            f"speaker: {chunk.get('speaker')}\n"
            f"mentions: {chunk.get('mentions')}\n"
            f"content_type: {chunk.get('content_type')}\n"
            f"source_file: {chunk.get('source_file')}\n"
            f"era: {chunk.get('era')}\n"
            f"section_title: {chunk.get('section_title')}\n"
            f"text: {chunk.get('text')}"
        )
        formatted_parts.append(source_block)

    return "\n\n".join(formatted_parts)


def build_interview_prompt(
    beatle: str,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build the final RAG prompt for the interview chatbot.

    This prompt tells the model:
    - who it is loosely channeling
    - what the user's question is
    - what source excerpts are available
    - what rules it must follow
    """
    beatle = beatle.lower().strip()

    if beatle not in VALID_BEATLES:
        raise ValueError(f"Invalid beatle '{beatle}'. Must be one of {sorted(VALID_BEATLES)}")

    style_instruction = get_beatle_style(beatle)
    sources_text = format_sources_for_prompt(retrieved_chunks)

    prompt = f"""
You are answering a user as if you are {beatle.title()} from The Beatles.

IMPORTANT:
- Your answer must be grounded only in the provided source excerpts.
- Do not invent facts, dates, memories, events, opinions, or quotes.
- If the excerpts do not give enough information to answer confidently, say so simply and honestly.
- Do not pretend to know anything beyond the excerpts.
- Use only a light stylistic resemblance to {beatle.title()}.
- Do not become theatrical, exaggerated, or cartoonish.
- Prioritize accuracy and grounding over performance.

STYLE:
{style_instruction}

ANSWERING RULES:
- Answer in first person, as the selected Beatle.
- Keep the response natural and conversational.
- You may summarize or paraphrase the source material, but do not fabricate.
- If multiple excerpts suggest different nuances, reflect that uncertainty honestly.
- Keep the answer fairly concise: usually 1 to 3 paragraphs.
- Do not mention "the provided excerpts" or "Source 1" in the body of the answer.
- Do not include citations in the answer text unless explicitly asked.

USER QUESTION:
{question}

SOURCE EXCERPTS:
{sources_text}

Now write the best grounded answer to the user's question.
""".strip()

    return prompt


def build_fallback_answer(beatle: str) -> str:
    """
    Optional fallback text in case retrieval fails or returns nothing useful.
    """
    beatle = beatle.lower().strip()

    fallback_map = {
        "john": "I'd rather not make something up. Ask me another one and I'll tell you what I can from the record.",
        "paul": "I don't want to pretend I know if the material doesn't support it. Give me another question and we'll have a go.",
        "george": "I shouldn't invent an answer when the material doesn't really say. Ask something else and we'll see what's there.",
        "ringo": "I don't want to make anything up, you know. Ask me something else and I'll do my best with what's there.",
    }

    if beatle not in fallback_map:
        raise ValueError(f"Invalid beatle '{beatle}'. Must be one of {sorted(VALID_BEATLES)}")

    return fallback_map[beatle]
