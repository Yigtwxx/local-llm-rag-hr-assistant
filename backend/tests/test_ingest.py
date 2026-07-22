"""A failed ingest must leave the previous index exactly as it was.

The rebuild used to drop the collection first and write batch by batch, so
anything that failed part-way — Ollama not running is the ordinary case — left
the machine with no index at all. Measured against a real store: 39 chunks
became 0, and the assistant refused every question until someone noticed.
"""

from pathlib import Path

import pytest

from app import ingest as ingest_module
from app.config import Settings
from app.llm import OllamaError
from app.retrieval import VectorStore

DIMENSIONS = 4

KB_DOCUMENT = """# İzin Politikası

## 1. Yıllık Ücretli İzin

### 1.1 Hak Ediş

Yıllık ücretli izin hakkı hizmet süresine göre belirlenir.

### 1.2 Avans İzin

Avans izin en fazla 5 iş günüdür.
"""


class FakeOllama:
    """Embeds deterministically, or fails on the nth batch."""

    def __init__(self, *, fail_on_batch: int | None = None) -> None:
        self.fail_on_batch = fail_on_batch
        self.batches = 0
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.batches += 1
        if self.fail_on_batch is not None and self.batches >= self.fail_on_batch:
            raise OllamaError("Cannot reach Ollama at http://127.0.0.1:11999")
        return [[float(len(text) % 7)] * DIMENSIONS for text in texts]

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "01-izin-politikasi.md").write_text(KB_DOCUMENT, encoding="utf-8")

    settings = Settings(
        kb_dir=kb,
        storage_dir=tmp_path / "storage",
        collection_name="test_kb",
        suggestions_file=tmp_path / "missing-suggestions.yaml",
    )
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)
    return settings


async def test_a_successful_ingest_indexes_every_chunk(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "OllamaClient", lambda settings: FakeOllama())

    stored = await ingest_module.ingest()

    assert stored > 0
    assert VectorStore(workspace).count() == stored


async def test_a_failure_part_way_through_leaves_the_index_intact(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "OllamaClient", lambda settings: FakeOllama())
    before = await ingest_module.ingest()
    assert before > 0

    # Second run dies during embedding, after the point the old code had already
    # deleted the collection.
    monkeypatch.setattr(
        ingest_module, "OllamaClient", lambda settings: FakeOllama(fail_on_batch=1)
    )

    with pytest.raises(OllamaError):
        await ingest_module.ingest()

    assert VectorStore(workspace).count() == before


async def test_the_collection_is_not_dropped_before_the_vectors_exist(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "OllamaClient", lambda settings: FakeOllama())
    await ingest_module.ingest()

    seen: list[int] = []
    original_reset = VectorStore.reset

    def recording_reset(self: VectorStore) -> None:
        # How many chunks were still indexed at the moment the drop happened.
        seen.append(self.count())
        original_reset(self)

    monkeypatch.setattr(VectorStore, "reset", recording_reset)
    monkeypatch.setattr(
        ingest_module, "OllamaClient", lambda settings: FakeOllama(fail_on_batch=1)
    )

    with pytest.raises(OllamaError):
        await ingest_module.ingest()

    # The drop never ran at all: embedding failed first.
    assert seen == []


async def test_a_failed_first_run_leaves_no_storage_behind(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ingest_module, "OllamaClient", lambda settings: FakeOllama(fail_on_batch=1)
    )

    with pytest.raises(OllamaError):
        await ingest_module.ingest()

    # Opening a Chroma client writes `chroma.sqlite3`, and the Docker entrypoint
    # reads that file as "an index already exists" and skips ingest. A first run
    # that died before writing anything therefore came back up with an empty
    # index on every later boot, silently refusing every question.
    assert not (workspace.resolve_storage_dir() / "chroma.sqlite3").exists()


async def test_the_client_is_closed_even_when_ingest_fails(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOllama(fail_on_batch=1)
    monkeypatch.setattr(ingest_module, "OllamaClient", lambda settings: fake)

    with pytest.raises(OllamaError):
        await ingest_module.ingest()

    assert fake.closed


async def test_append_keeps_what_is_already_there(
    workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "OllamaClient", lambda settings: FakeOllama())
    first = await ingest_module.ingest()

    # Same documents, so the ids repeat and Chroma upserts rather than doubling.
    await ingest_module.ingest(rebuild=False)

    assert VectorStore(workspace).count() == first
