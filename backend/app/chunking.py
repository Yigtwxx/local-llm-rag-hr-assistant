"""Heading-aware chunking for Turkish Markdown policy documents.

Splitting on a fixed character window would cut tables in half and strip the
heading that gives a passage its meaning. Instead we split on the Markdown
heading tree, keep each block (paragraph, table, list) intact, and prefix every
chunk with its heading path so an isolated chunk still says what it is about.
"""

import hashlib
import re
from pathlib import Path

from app.schemas import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Blockquotes carry the "this is a fictional document" banner and other editorial
# notes. They are dropped entirely rather than unquoted: indexing them would let
# the assistant answer questions about the document instead of the policy.
_BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>")

# Turkish is agglutinative: words carry more characters per token than English.
# ~3.5 characters per token is a reasonable estimate for the Qwen/Gemma
# tokenizers and is only used for sizing chunks, never for billing.
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Rough token count used to size chunks."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _slugify(value: str) -> str:
    lowered = value.casefold()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "i̇": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug[:60] or "bolum"


def _split_blocks(text: str) -> list[str]:
    """Split a section body into atomic blocks (paragraph, table, list)."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [block for block in blocks if block]


def _pack_blocks(blocks: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack blocks into chunks, carrying an overlap tail forward.

    A single block larger than `chunk_size` is emitted on its own rather than
    being cut: splitting a Markdown table mid-row destroys its meaning, and an
    oversized chunk is still perfectly retrievable.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block)

        if current and current_tokens + block_tokens > chunk_size:
            chunks.append("\n\n".join(current))
            tail = _overlap_tail(current, overlap)
            current = [*tail, block]
            current_tokens = sum(estimate_tokens(part) for part in current)
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _overlap_tail(blocks: list[str], overlap: int) -> list[str]:
    """Take trailing blocks worth roughly `overlap` tokens, for continuity."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    budget = overlap
    for block in reversed(blocks):
        tokens = estimate_tokens(block)
        if tokens > budget:
            break
        tail.insert(0, block)
        budget -= tokens
    return tail


def chunk_markdown(
    text: str,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Turn one Markdown document into retrievable chunks."""
    lines = text.splitlines()

    doc_title = source_file
    for line in lines:
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            doc_title = match.group(2).strip()
            break

    # Walk the document, accumulating body lines under the current heading path.
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    body: list[str] = []

    def flush() -> None:
        if body and any(line.strip() for line in body):
            # Skip the H1: it is already carried separately as `doc_title`, and
            # repeating it in every section path just wastes context.
            path = [title for title in heading_stack[1:] if title]
            sections.append((" › ".join(path) if path else doc_title, list(body)))
        body.clear()

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)
        else:
            body.append(line)
    flush()

    chunks: list[Chunk] = []
    for section, section_lines in sections:
        kept = [line for line in section_lines if not _BLOCKQUOTE_LINE_RE.match(line)]
        raw = "\n".join(kept).strip()
        if not raw:
            continue

        blocks = _split_blocks(raw)
        for body_text in _pack_blocks(blocks, chunk_size, chunk_overlap):
            # The heading path travels with the text so a chunk retrieved on its
            # own still identifies which policy and section it came from.
            contextual = f"{doc_title}\n{section}\n\n{body_text}"
            digest = hashlib.sha1(
                f"{source_file}:{section}:{body_text}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{_slugify(source_file)}-{digest}",
                    text=contextual,
                    source_file=source_file,
                    doc_title=doc_title,
                    section=section,
                    token_estimate=estimate_tokens(contextual),
                )
            )
    return chunks


def load_knowledge_base(
    kb_dir: Path, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every Markdown file in the knowledge-base directory."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, path.name, chunk_size, chunk_overlap))
    return chunks
