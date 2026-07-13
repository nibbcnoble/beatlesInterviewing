#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
SOURCE_DIR = PROCESSED_DIR / "cleaned_rulebased"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "chunks.jsonl"

# =========================================================
# Chunking configuration
# =========================================================
TARGET_CHARS = 1200
MIN_CHARS = 500
MAX_CHARS = 1800
MIN_USEFUL_CHARS = 120

BEATLES = ["JOHN", "PAUL", "GEORGE", "RINGO"]

FULL_NAME_TO_BEATLE = {
    "JOHN LENNON": "JOHN",
    "PAUL MCCARTNEY": "PAUL",
    "GEORGE HARRISON": "GEORGE",
    "RINGO STARR": "RINGO",
}

# =========================================================
# Basic text cleanup
# =========================================================
def normalize_text(text: str) -> str:
    """
    Normalize line endings and whitespace.
    Also clean quote-year suffixes like:
      sentence.80  -> sentence. [80]
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # Trim each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse large runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Convert anthology-style year suffixes into a cleaner inline form
    # Example: "specials.80" -> "specials. [80]"
    text = re.sub(r"([.!?'\"])(\d{2})(\s|$)", r"\1 [\2]\3", text)

    return text.strip()


# =========================================================
# File metadata
# =========================================================
def parse_source_metadata(path: Path) -> Dict[str, Optional[str]]:
    """
    Example filenames:
      00_front_matter_and_bios.cleaned.txt
      03_1964.cleaned.txt
      08_1969-70.cleaned.txt
    """
    name = path.name
    m = re.match(r"(?P<chapter>\d+)_([^.]*)\.cleaned\.txt$", name)

    chapter_num = None
    era = None

    if m:
        chapter_num = m.group("chapter")
        remainder = name.split("_", 1)[1]
        era = remainder.replace(".cleaned.txt", "")

    return {
        "chapter_num": chapter_num,
        "era": era,
    }


# =========================================================
# Heading detection
# =========================================================
def is_heading_like(line: str) -> bool:
    """
    A heuristic (best-guess rule) for deciding whether a line
    looks like a section heading rather than normal prose.
    """
    line = line.strip()
    if not line:
        return False

    if len(line) > 100:
        return False

    # All-caps / title-ish lines
    if re.fullmatch(r"[A-Z0-9 ,.'’:/\-()]+", line):
        return True

    if re.match(r"^(chapter|part|section)\b", line, re.IGNORECASE):
        return True

    return False


def detect_heading_speaker(line: str) -> Optional[str]:
    """
    If a heading says JOHN LENNON / PAUL MCCARTNEY / etc.,
    return the matching Beatle name.
    """
    cleaned = re.sub(r"\s+", " ", line.strip().upper())

    for full_name, beatle in FULL_NAME_TO_BEATLE.items():
        if cleaned == full_name:
            return beatle

    return None


def infer_content_type(source_file: str, current_speaker: Optional[str]) -> str:
    """
    Very simple content classification.
    """
    if source_file.startswith("00_"):
        if current_speaker is not None:
            return "bio"
        return "front_matter"
    return "main"


# =========================================================
# Splitting text into blocks
# =========================================================
def split_into_blocks(text: str) -> List[str]:
    """
    First split on blank lines.
    Then split heading-like first lines away from body text if needed.
    """
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    blocks: List[str] = []

    for block in raw_blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        # If first line looks like a heading and there is body after it,
        # separate them into two blocks.
        if len(lines) >= 2 and is_heading_like(lines[0]):
            blocks.append(lines[0])
            rest = "\n".join(lines[1:]).strip()
            if rest:
                blocks.append(rest)
        else:
            blocks.append(block)

    return blocks


# =========================================================
# Chunk splitting helpers
# =========================================================
def hard_split(text: str, max_chars: int) -> List[str]:
    parts = []
    start = 0
    while start < len(text):
        part = text[start:start + max_chars].strip()
        if part:
            parts.append(part)
        start += max_chars
    return parts


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    If a chunk is too long, split by sentence boundaries if possible.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return hard_split(text, max_chars)

    parts = []
    current = ""

    for sent in sentences:
        candidate = sent if not current else current + " " + sent

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current.strip())

            if len(sent) <= max_chars:
                current = sent
            else:
                parts.extend(hard_split(sent, max_chars))
                current = ""

    if current:
        parts.append(current.strip())

    return parts


# =========================================================
# Metadata detection
# =========================================================
def detect_explicit_speaker(text: str) -> Optional[str]:
    """
    Detect explicit speaker labels like:
      JOHN:
      PAUL:
    If more than one appears, return None because the chunk is mixed.
    """
    found = set()

    for beatle in BEATLES:
        if re.search(rf"\b{beatle}\s*:", text):
            found.add(beatle)

    if len(found) == 1:
        return next(iter(found))

    return None


def detect_mentions(text: str) -> List[str]:
    found = set()

    for beatle in BEATLES:
        if re.search(rf"\b{beatle}\b", text, re.IGNORECASE):
            found.add(beatle)

    return sorted(found)


def looks_like_junk(text: str) -> bool:
    """
    Skip chunks that are too small or mostly just heading noise.
    """
    stripped = text.strip()

    if len(stripped) < MIN_USEFUL_CHARS:
        return True

    if is_heading_like(stripped) and len(stripped) < 150:
        return True

    return False


def normalize_for_dedupe(text: str) -> str:
    """
    Normalize text for duplicate detection.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_section_title(blocks: List[str], era: Optional[str]) -> Optional[str]:
    """
    Use the first heading-like block as the section title when possible.
    Otherwise fall back to era.
    """
    for block in blocks[:8]:
        candidate = block.replace("\n", " ").strip()
        if is_heading_like(candidate) and len(candidate) <= 100:
            return candidate

    return era


# =========================================================
# Main chunk building
# =========================================================
def build_chunks_from_blocks(
    blocks: List[str],
    source_file: str,
    chapter_num: Optional[str],
    era: Optional[str],
    section_title: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Build chunks while carrying forward section speaker context.

    This is the key learning idea in this version:
    we keep track of who the current section seems to be about.
    """
    chunks: List[Dict[str, Any]] = []

    current_blocks: List[str] = []
    current_len = 0

    current_section_speaker: Optional[str] = None

    def flush():
        nonlocal current_blocks, current_len, chunks, current_section_speaker

        if not current_blocks:
            return

        text = "\n\n".join(current_blocks).strip()

        if not text or looks_like_junk(text):
            current_blocks = []
            current_len = 0
            return

        split_parts = split_long_text(text, MAX_CHARS)

        for part in split_parts:
            if looks_like_junk(part):
                continue

            explicit_speaker = detect_explicit_speaker(part)
            final_speaker = explicit_speaker or current_section_speaker

            chunk = {
                "source_file": source_file,
                "chapter_num": chapter_num,
                "era": era,
                "section_title": section_title,
                "content_type": infer_content_type(source_file, final_speaker),
                "speaker": final_speaker,
                "mentions": detect_mentions(part),
                "text": part,
                "char_count": len(part),
                "block_count": len(current_blocks),
            }
            chunks.append(chunk)

        current_blocks = []
        current_len = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # If this block is a heading, it may change our speaker context.
        if is_heading_like(block):
            flush()

            heading_speaker = detect_heading_speaker(block)
            if heading_speaker:
                current_section_speaker = heading_speaker

            # We do not store heading-only blocks as chunks.
            continue

        block_len = len(block)

        if not current_blocks:
            current_blocks = [block]
            current_len = block_len
            continue

        candidate_len = current_len + 2 + block_len

        if candidate_len <= TARGET_CHARS:
            current_blocks.append(block)
            current_len = candidate_len
            continue

        if current_len < MIN_CHARS and candidate_len <= MAX_CHARS:
            current_blocks.append(block)
            current_len = candidate_len
            continue

        flush()
        current_blocks = [block]
        current_len = block_len

    flush()
    return chunks


# =========================================================
# Per-file processing
# =========================================================
def process_file(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(raw)

    if not text:
        return []

    meta = parse_source_metadata(path)
    blocks = split_into_blocks(text)
    section_title = infer_section_title(blocks, meta["era"])

    return build_chunks_from_blocks(
        blocks=blocks,
        source_file=path.name,
        chapter_num=meta["chapter_num"],
        era=meta["era"],
        section_title=section_title,
    )


# =========================================================
# Deduplication
# =========================================================
def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for chunk in chunks:
        key = normalize_for_dedupe(chunk["text"])
        if key in seen:
            continue

        seen.add(key)
        deduped.append(chunk)

    return deduped


# =========================================================
# Entry point
# =========================================================
def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(SOURCE_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {SOURCE_DIR}")

    all_chunks: List[Dict[str, Any]] = []

    for path in txt_files:
        file_chunks = process_file(path)
        all_chunks.extend(file_chunks)

    all_chunks = dedupe_chunks(all_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"chunk_{i:06d}"

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Built {len(all_chunks)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")

    counts = {}
    for chunk in all_chunks:
        counts.setdefault(chunk["source_file"], 0)
        counts[chunk["source_file"]] += 1

    print("\nChunks per file:")
    for source_file, count in sorted(counts.items()):
        print(f"  {source_file}: {count}")


if __name__ == "__main__":
    main()
