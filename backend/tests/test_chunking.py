"""Unit tests for heading-aware chunking. No Ollama required."""

from app.chunking import chunk_markdown, estimate_tokens

SAMPLE = """# İzin Politikası

> Kurgusal doküman notu.

## 1. Yıllık İzin

### 1.1 Hak Ediş

| Hizmet süresi | İzin hakkı |
|---|---|
| 1 – 5 yıl arası | 16 iş günü |
| 5 – 15 yıl arası | 22 iş günü |

Bu süreler yasal asgari sürelerin üzerindedir.

## 2. Mazeret İzinleri

Evlilik izni 5 iş günüdür.
"""


def test_document_title_is_extracted_from_h1() -> None:
    chunks = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    assert chunks
    assert all(chunk.doc_title == "İzin Politikası" for chunk in chunks)


def test_heading_path_is_preserved() -> None:
    chunks = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    sections = {chunk.section for chunk in chunks}
    assert "1. Yıllık İzin › 1.1 Hak Ediş" in sections
    assert "2. Mazeret İzinleri" in sections


def test_chunk_text_carries_its_context() -> None:
    """An isolated chunk must still say which document and section it is from."""
    chunks = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    table_chunk = next(c for c in chunks if "16 iş günü" in c.text)
    assert table_chunk.text.startswith("İzin Politikası")
    assert "1.1 Hak Ediş" in table_chunk.text


def test_markdown_table_is_never_split() -> None:
    """A tiny chunk budget must still keep the table rows together."""
    chunks = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=10, chunk_overlap=0)
    table_chunks = [c for c in chunks if "| Hizmet süresi |" in c.text]
    assert len(table_chunks) == 1
    assert "16 iş günü" in table_chunks[0].text
    assert "22 iş günü" in table_chunks[0].text


def test_blockquote_banner_is_stripped() -> None:
    chunks = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    assert not any("Kurgusal doküman notu" in chunk.text for chunk in chunks)


def test_chunk_ids_are_unique_and_stable() -> None:
    first = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    second = chunk_markdown(SAMPLE, "01-izin.md", chunk_size=500, chunk_overlap=50)
    ids = [chunk.chunk_id for chunk in first]
    assert len(ids) == len(set(ids))
    assert ids == [chunk.chunk_id for chunk in second]


def test_token_estimate_scales_with_length() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 350) == 100
