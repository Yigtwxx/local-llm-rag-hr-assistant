"""Vector store access and similarity search over the HR knowledge base."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.config import Settings
from app.lexical import BM25Index
from app.llm import OllamaClient
from app.schemas import Chunk, RetrievedChunk
from app.suggestions import QUESTION_SEPARATOR

# How many near-miss passages are carried out of retrieval for the follow-up
# chips. Larger than the chip count on purpose: several may share a section, and
# only one chip is taken per section.
NEIGHBOUR_POOL = 10

T = TypeVar("T")


@dataclass
class RetrievalResult:
    """What retrieval found: the passages to answer from, and the near misses."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    neighbours: list[RetrievedChunk] = field(default_factory=list)


class VectorStore:
    """Persistent Chroma collection holding the indexed policy chunks.

    Embeddings are computed by Ollama and handed to Chroma explicitly, so the
    same model serves both chat and retrieval and no second runtime (torch,
    sentence-transformers) is needed.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        storage = settings.resolve_storage_dir()
        storage.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(storage),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._open()
        # Bumped every time the underlying collection is replaced. Callers that
        # cache anything derived from the collection's contents — the BM25 index
        # is the one — compare it to know their copy is stale.
        self._generation = 0

    def _open(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._settings.collection_name,
            # Cosine distance keeps scores comparable regardless of vector norm.
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def generation(self) -> int:
        """How many times the collection has been replaced under this handle."""
        return self._generation

    def _run(self, action: Callable[[chromadb.Collection], T]) -> T:
        """Run a collection operation, re-opening the collection if it is gone.

        A Chroma handle is bound to a collection *id*, not to its name. `ingest`
        rebuilds by deleting the collection and creating a new one, which gets a
        new id — so a handle taken at startup keeps pointing at something that
        no longer exists, and every request after a re-index failed:
        `/api/health` returned 500 and `/api/chat` closed its stream without
        emitting a single event. Re-indexing while the API is running is the
        documented workflow, not an edge case.

        The retry costs nothing on the normal path: it only runs after Chroma
        has already told us the collection is missing.
        """
        try:
            return action(self._collection)
        except NotFoundError:
            self._collection = self._open()
            self._generation += 1
            return action(self._collection)

    def count(self) -> int:
        return self._run(lambda collection: collection.count())

    def all_documents(self) -> dict[str, str]:
        """Every indexed chunk's text, keyed by id — the BM25 index's input.

        Read from Chroma rather than re-chunking the knowledge base so that word
        search and vector search can never disagree about what is indexed.
        """
        result = self._run(lambda collection: collection.get(include=["documents"]))
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        return {
            chunk_id: text
            for chunk_id, text in zip(ids, documents, strict=False)
            if text
        }

    def reset(self) -> None:
        """Drop and recreate the collection for a clean re-index."""
        try:
            self._client.delete_collection(self._settings.collection_name)
        except NotFoundError:
            # Already gone — another process rebuilt it first. Nothing to drop.
            pass
        self._collection = self._open()
        self._generation += 1

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        metadatas = [
            {
                "source_file": chunk.source_file,
                "doc_title": chunk.doc_title,
                "section": chunk.section,
                "token_estimate": chunk.token_estimate,
                # Chroma metadata holds scalars only, so the reviewed
                # follow-up questions travel as one separated string.
                "suggested_questions": QUESTION_SEPARATOR.join(
                    chunk.suggested_questions
                ),
            }
            for chunk in chunks
        ]
        self._run(
            lambda collection: collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
        )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        count = self.count()
        if count == 0:
            return []
        result = self._run(
            lambda collection: collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, count),
            )
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = result.get("ids") or [[]]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            ids[0], documents[0], metadatas[0], distances[0], strict=False
        ):
            raw_questions = str(meta.get("suggested_questions", ""))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=str(meta.get("source_file", "")),
                    doc_title=str(meta.get("doc_title", "")),
                    section=str(meta.get("section", "")),
                    # Chroma returns cosine distance; convert to similarity.
                    score=round(1.0 - float(distance), 4),
                    suggested_questions=[
                        question
                        for question in raw_questions.split(QUESTION_SEPARATOR)
                        if question.strip()
                    ],
                )
            )
        return retrieved


class Retriever:
    """Embeds a question and returns the passages worth answering from.

    Retrieval has two independent arms and a passage enters the context if
    *either* admits it:

    dense    cosine similarity above the calibrated floor, capped at `top_k`.
             Unchanged from the vector-only design — same floor, same cap, same
             order — so every question that worked before still gets exactly the
             same passages in exactly the same positions.

    lexical  BM25 word match carrying a word rare enough to clear its own floor,
             capped at `lexical_max`, appended after the dense passages.

    The arms are combined this way rather than fused into a single ranking on
    purpose. A fusion such as RRF would reorder the dense passages among
    themselves, and with `temperature=0` and a fixed seed that changes generated
    output for questions that were already answered correctly. There is no
    evidence BM25 orders those better, so the reordering would be a risk bought
    for nothing. Appending keeps one property that is worth more than elegance:
    when the lexical arm does not fire, the result is byte-identical to before.
    """

    def __init__(
        self, settings: Settings, client: OllamaClient, store: VectorStore
    ) -> None:
        self._settings = settings
        self._client = client
        self._store = store
        self._lexical: BM25Index | None = None
        self._lexical_fingerprint: tuple[int, int] | None = None

    def _lexical_index(self) -> BM25Index:
        """The BM25 index, built on first use and rebuilt when the corpus moves.

        Built lazily rather than in the constructor because the API creates the
        retriever at startup, which on a first run happens before `ingest` has
        written anything to index.

        Cached against a fingerprint rather than built once. `ingest` runs in its
        own process while the API keeps serving, so a cache with no invalidation
        left word search answering from the corpus as it was at startup while
        vector search answered from the current one — the two arms disagreeing
        about what is indexed, which is the one thing `all_documents` exists to
        prevent. The generation counter catches a rebuild and the count catches
        an `--append`.
        """
        fingerprint = (self._store.generation, self._store.count())
        if self._lexical is None or self._lexical_fingerprint != fingerprint:
            self._lexical = BM25Index(self._store.all_documents())
            self._lexical_fingerprint = fingerprint
        return self._lexical

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        """The passages worth answering from, best first."""
        return (await self.retrieve_with_neighbours(question)).chunks

    async def retrieve_with_neighbours(self, question: str) -> RetrievalResult:
        """Passages to answer from, plus the near misses behind them.

        Filtering here rather than in the prompt is deliberate: if nothing
        clears either floor the caller can refuse outright instead of handing
        the model weak context and hoping it declines to invent an answer.

        The near misses are carried out alongside because they are already
        ranked and cost nothing more to return. They are what the follow-up
        chips are drawn from — passages close enough to the question to be
        related, but not close enough to have answered it.
        """
        embeddings = await self._client.embed([question])
        if not embeddings:
            return RetrievalResult(chunks=[], neighbours=[])

        settings = self._settings
        # The whole collection, ranked. At this scale a full ranking costs
        # nothing and avoids a subtle failure: a passage the lexical arm wants
        # may sit outside any truncated pool, and then its cosine score — which
        # the UI displays — would be unavailable. Revisit if the knowledge base
        # grows by orders of magnitude.
        ranking = self._store.query(embeddings[0], self._store.count())

        dense = [c for c in ranking if c.score >= settings.similarity_threshold]
        dense = dense[: settings.top_k]

        # The lexical arm stays shut unless the dense arm already found
        # something. This is the rule that keeps the two-defence guarantee the
        # reports make: a question that clears no passage at all is refused
        # outright, without a generation call, and word matching must never
        # quietly turn that hard refusal into "let the model decide". Seven of
        # the nine out-of-scope questions in the calibration set land here
        # (measured against `bench.questions.collect_questions`).
        if not dense:
            # Nothing was answerable, but the best-ranked passages still say
            # what this knowledge base *is* about, so a refusal can still offer
            # somewhere to go. See `RagPipeline.answer`.
            return RetrievalResult(chunks=[], neighbours=ranking[:NEIGHBOUR_POOL])

        chunks = dense + self._lexical_additions(question, ranking, dense)
        used = {chunk.chunk_id for chunk in chunks}
        return RetrievalResult(
            chunks=chunks,
            neighbours=[c for c in ranking if c.chunk_id not in used][:NEIGHBOUR_POOL],
        )

    def _lexical_additions(
        self,
        question: str,
        ranking: list[RetrievedChunk],
        dense: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Passages the dense arm missed but a rare word in the question names.

        Deliberately narrow. Only the single best word match is considered, and
        only when it carries a word occurring in exactly one chunk of the corpus.
        A looser rule was measured and rejected: at any floor low enough to admit
        more, it also appended irrelevant passages to questions that already
        answered correctly, which is a regression bought for nothing.
        """
        settings = self._settings
        chosen = {chunk.chunk_id for chunk in dense}
        by_id = {chunk.chunk_id: chunk for chunk in ranking}

        additions: list[RetrievedChunk] = []
        for hit in self._lexical_index().rank(question, settings.lexical_max):
            if hit.chunk_id in chosen or hit.rarity < settings.lexical_rarity_floor:
                continue
            chunk = by_id.get(hit.chunk_id)
            if chunk is not None:
                additions.append(chunk.model_copy(update={"matched_by": "lexical"}))
        return additions
