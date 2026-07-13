import os
import re
import json
import time
import argparse
from pathlib import Path

import requests

SPEAKERS = ("JOHN:", "PAUL:", "GEORGE:", "RINGO:")

SYSTEM_PROMPT = """You are cleaning raw book text for later use in a retrieval-augmented generation system.

Your task:
- Convert raw line-broken text into clean readable paragraph text.
- Preserve wording as closely as possible.
- Preserve speaker labels exactly when present: JOHN:, PAUL:, GEORGE:, RINGO:
- Preserve factual content.
- Do not summarize.
- Do not paraphrase unless absolutely necessary to repair broken formatting.
- Do not omit content.
- Do not add commentary.
- Output plain text only.

Rules:
1. Merge lines that are clearly part of the same paragraph.
2. Start a new paragraph when a new speaker label appears.
3. Preserve paragraph breaks where they make sense.
4. Remove indentation and excessive spacing.
5. If footnote numbers or citation artifacts obviously interrupt readability, remove only those artifacts.
6. Do not change chronology or speaker attribution.
7. Return only the cleaned text for the chunk, with no preamble and no explanation.
"""

USER_PROMPT_TEMPLATE = """Clean the following text chunk conservatively into readable paragraphs.

File: {filename}
Chunk: {chunk_number}/{chunk_total}

Text:
\"\"\"
{chunk_text}
\"\"\"
"""


def normalize_whitespace(line: str) -> str:
    line = line.replace("\t", " ")
    line = re.sub(r"[ ]{2,}", " ", line)
    return line.strip()


def is_probable_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 40 and re.fullmatch(r"[\d\- ]+", stripped):
        return True
    if len(stripped) <= 60 and stripped.isupper():
        return True
    if re.fullmatch(r"\d{4}(?:-\d{2,4})?", stripped):
        return True
    return False


def starts_with_speaker(line: str) -> bool:
    stripped = line.lstrip()
    return any(stripped.startswith(s) for s in SPEAKERS)


def rule_based_cleanup(raw_text: str) -> str:
    lines = raw_text.splitlines()

    cleaned_lines = []
    for line in lines:
        norm = normalize_whitespace(line)
        cleaned_lines.append(norm)

    paragraphs = []
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            text = " ".join(buffer).strip()
            text = re.sub(r"\s+([,.;:?!])", r"\1", text)
            paragraphs.append(text)
            buffer = []

    for line in cleaned_lines:
        if not line:
            flush()
            continue

        if is_probable_heading(line):
            flush()
            paragraphs.append(line)
            continue

        if starts_with_speaker(line):
            flush()
            buffer.append(line)
            continue

        if buffer:
            buffer.append(line)
        else:
            buffer = [line]

    flush()

    # light cleanup of dangling footnote-style numbers at line ends
    normalized_paragraphs = []
    for p in paragraphs:
        p = re.sub(r"(?<=\w)(\d{1,3})$", "", p).strip()
        p = re.sub(r"\s{2,}", " ", p)
        normalized_paragraphs.append(p)

    return "\n\n".join(normalized_paragraphs).strip()


def chunk_text(text: str, max_chars: int = 5000, overlap_chars: int = 400):
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []

    current = []

    def current_len(parts):
        return sum(len(p) for p in parts) + max(0, 2 * (len(parts) - 1))

    for para in paragraphs:
        if not current:
            current = [para]
            continue

        if current_len(current + [para]) <= max_chars:
            current.append(para)
        else:
            chunks.append("\n\n".join(current))

            # overlap using tail paragraphs
            overlap = []
            total = 0
            for p in reversed(current):
                total += len(p) + 2
                overlap.insert(0, p)
                if total >= overlap_chars:
                    break

            current = overlap + [para]

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def call_openrouter(api_key: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def process_file(path: Path, api_key: str, model: str, rule_dir: Path, llm_dir: Path,
                 max_chars: int, overlap_chars: int, sleep_seconds: float):
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    first_pass = rule_based_cleanup(raw_text)

    rule_output_path = rule_dir / f"{path.stem}.cleaned.txt"
    rule_output_path.write_text(first_pass, encoding="utf-8")

    chunks = chunk_text(first_pass, max_chars=max_chars, overlap_chars=overlap_chars)

    refined_chunks = []
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        print(f"  LLM cleaning chunk {i}/{total} for {path.name}...")
        user_prompt = USER_PROMPT_TEMPLATE.format(
            filename=path.name,
            chunk_number=i,
            chunk_total=total,
            chunk_text=chunk
        )
        cleaned_chunk = call_openrouter(api_key, model, user_prompt)
        refined_chunks.append(cleaned_chunk)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    final_text = f"=== FILE: {path.name} ===\n\n" + "\n\n".join(refined_chunks).strip() + "\n"

    llm_output_path = llm_dir / f"{path.stem}.llm_cleaned.txt"
    llm_output_path.write_text(final_text, encoding="utf-8")

    return {
        "filename": path.name,
        "rule_output": str(rule_output_path),
        "llm_output": str(llm_output_path),
        "chunk_count": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean Beatles text files with rule-based cleanup + OpenRouter chunk refinement.")
    parser.add_argument("--input", required=True, help="Input folder containing raw .txt files")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model name")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max chars per LLM chunk")
    parser.add_argument("--overlap-chars", type=int, default=400, help="Overlap chars between chunks")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between API calls")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
      raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rule_dir = output_dir / "cleaned_rulebased"
    llm_dir = output_dir / "cleaned_llm"

    rule_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {input_dir}")

    manifest = []
    master_parts = []

    for path in txt_files:
        print(f"Processing {path.name}...")
        result = process_file(
            path=path,
            api_key=api_key,
            model=args.model,
            rule_dir=rule_dir,
            llm_dir=llm_dir,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            sleep_seconds=args.sleep,
        )
        manifest.append(result)

        llm_text = Path(result["llm_output"]).read_text(encoding="utf-8")
        master_parts.append(llm_text.strip())

    master_path = output_dir / "master_cleaned.txt"
    master_path.write_text("\n\n".join(master_parts).strip() + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone.")
    print(f"Master file: {master_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
