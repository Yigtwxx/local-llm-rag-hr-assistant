"""Tests for the vector store's resilience to being re-indexed underneath it.

`ingest` runs in its own process while the API keeps serving. Both cases here
were reproduced against a running server before they were fixed; neither needs
Ollama, because the failures are in the Chroma handle and the BM25 cache rather
than in anything a model does.
"""

from pathlib import Path

import pytest
from chromadb.errors import NotFoundError

from app.config import Settings
from app.lexical import BM25Index
from app.retrieval import Retriever, VectorStore
from app.schemas import Chunk

DIMENSIONS = 8


def settings_for(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway store, so no real index is touched."""
    return Settings(storage_dir=tmp_path / "storage", collection_name="test_kb")


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file="01-izin-politikasi.md",
        doc_title="İzin Politikası",
        section=f"Bölüm {chunk_id}",
        token_estimate=len(text),
    )


def vector(seed: float) -> list[float]:
    """A deterministic unit-ish vector; the values only need to be stable."""
    return [seed] * DIMENSIONS


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(settings_for(tmp_path))
    store.add([chunk("a", "yıllık izin hakkı")], [vector(0.1)])
    return store


class TestCollectionReplacedUnderneath:
    """`ingest --rebuild` deletes the collection and creates a new one.

    A Chroma handle is bound to a collection *id*, so the handle the API took at
    startup pointed at something that no longer existed. Every request after a
    re-index failed: `/api/health` returned 500 and `/api/chat` closed its
    stream without emitting one event, so the interface waited on an answer
    that could never arrive.
    """

    def test_count_survives_the_collection_being_deleted(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        # A second process rebuilding the index, which is what `ingest` is.
        VectorStore(settings_for(tmp_path)).reset()

        assert store.count() == 0

    def test_query_survives_the_collection_being_deleted(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        rebuilt = VectorStore(settings_for(tmp_path))
        rebuilt.reset()
        rebuilt.add([chunk("b", "harcırah tutarı")], [vector(0.2)])

        found = store.query(vector(0.2), top_k=4)

        assert [c.chunk_id for c in found] == ["b"]

    def test_add_survives_the_collection_being_deleted(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        VectorStore(settings_for(tmp_path)).reset()

        store.add([chunk("c", "eğitim bütçesi")], [vector(0.3)])

        assert store.count() == 1

    def test_reacquiring_the_collection_bumps_the_generation(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        before = store.generation
        VectorStore(settings_for(tmp_path)).reset()

        store.count()

        # The counter is what tells a cache built from the old collection that
        # it is looking at a corpus that no longer exists.
        assert store.generation == before + 1

    def test_reset_tolerates_an_already_missing_collection(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        # Deleted through the raw client, so nothing recreates it — two ingests
        # racing is the case this covers. Going through another `VectorStore`
        # would not: its constructor recreates the collection on the way in.
        store._client.delete_collection(settings_for(tmp_path).collection_name)

        store.reset()  # must not raise

        assert store.count() == 0


class TestLexicalIndexFreshness:
    """Word search and vector search must never disagree about what is indexed.

    The BM25 index was built once and cached forever, so after a re-index the
    lexical arm kept answering from the corpus as it stood when the API booted
    while the dense arm answered from the current one.
    """

    def test_rebuilds_after_the_collection_is_replaced(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path)
        store = VectorStore(settings)
        store.add([chunk("a", "yıllık izin hakkı")], [vector(0.1)])
        retriever = Retriever(settings, client=None, store=store)  # type: ignore[arg-type]

        first = retriever._lexical_index()
        assert len(first) == 1

        rebuilt = VectorStore(settings)
        rebuilt.reset()
        rebuilt.add(
            [chunk("b", "babalık izni on iş günü"), chunk("c", "harcırah")],
            [vector(0.2), vector(0.3)],
        )

        second = retriever._lexical_index()

        assert second is not first
        assert len(second) == 2

    def test_rebuilds_after_chunks_are_appended(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path)
        store = VectorStore(settings)
        store.add([chunk("a", "yıllık izin hakkı")], [vector(0.1)])
        retriever = Retriever(settings, client=None, store=store)  # type: ignore[arg-type]

        first = retriever._lexical_index()

        # `ingest --append` adds to the same collection, so the generation
        # counter never moves and only the count gives the change away.
        store.add([chunk("b", "babalık izni on iş günü")], [vector(0.2)])
        second = retriever._lexical_index()

        assert second is not first
        assert len(second) == 2

    def test_keeps_the_cached_index_when_nothing_changed(self, tmp_path: Path) -> None:
        settings = settings_for(tmp_path)
        store = VectorStore(settings)
        store.add([chunk("a", "yıllık izin hakkı")], [vector(0.1)])
        retriever = Retriever(settings, client=None, store=store)  # type: ignore[arg-type]

        first = retriever._lexical_index()

        # Rebuilding on every question would re-tokenize the whole corpus per
        # request; the cache is the reason it is built lazily at all.
        assert retriever._lexical_index() is first
        assert isinstance(first, BM25Index)


def test_missing_collection_raises_without_the_guard(tmp_path: Path) -> None:
    """Pin the underlying Chroma behaviour the guard exists to absorb.

    If a future Chroma version made a stale handle resolve by name instead of
    by id, `_run`'s retry would become dead code and this test would say so.
    """
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    store.add([chunk("a", "yıllık izin hakkı")], [vector(0.1)])
    stale = store._collection

    VectorStore(settings).reset()

    with pytest.raises(NotFoundError):
        stale.count()
