"""Vector store access and similarity search over the HR knowledge base."""

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.llm import OllamaClient
from app.schemas import Chunk, RetrievedChunk


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
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            # Cosine distance keeps scores comparable regardless of vector norm.
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection for a clean re-index."""
        self._client.delete_collection(self._settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_file": chunk.source_file,
                    "doc_title": chunk.doc_title,
                    "section": chunk.section,
                    "token_estimate": chunk.token_estimate,
                }
                for chunk in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        ids = result.get("ids") or [[]]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            ids[0], documents[0], metadatas[0], distances[0], strict=False
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=str(meta.get("source_file", "")),
                    doc_title=str(meta.get("doc_title", "")),
                    section=str(meta.get("section", "")),
                    # Chroma returns cosine distance; convert to similarity.
                    score=round(1.0 - float(distance), 4),
                )
            )
        return retrieved


class Retriever:
    """Embeds a question and returns the passages worth answering from."""

    def __init__(
        self, settings: Settings, client: OllamaClient, store: VectorStore
    ) -> None:
        self._settings = settings
        self._client = client
        self._store = store

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return chunks above the similarity floor, best first.

        Filtering here rather than in the prompt is deliberate: if nothing
        clears the floor the caller can refuse outright instead of handing the
        model weak context and hoping it declines to invent an answer.
        """
        embeddings = await self._client.embed([question])
        if not embeddings:
            return []

        candidates = self._store.query(embeddings[0], self._settings.top_k)
        return [c for c in candidates if c.score >= self._settings.similarity_threshold]
