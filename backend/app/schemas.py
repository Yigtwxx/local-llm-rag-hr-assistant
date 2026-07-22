"""Pydantic models shared by the ingestion pipeline and the HTTP API."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable slice of a source document."""

    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    section: str
    token_estimate: int


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with its relevance score."""

    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    section: str
    score: float = Field(description="Cosine similarity in [0, 1]; higher is closer.")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(
        default=None,
        description="Chat model tag. Falls back to the configured primary model.",
    )
    think: bool = Field(
        default=False,
        description="Enable the model's reasoning mode. Off by default so that "
        "latency measurements stay comparable across models.",
    )


class ChatSource(BaseModel):
    """Citation surfaced next to an answer."""

    doc_title: str
    section: str
    source_file: str
    score: float
    excerpt: str


class ChatMetrics(BaseModel):
    """Per-request performance numbers, mirrored from Ollama's response."""

    model: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    eval_count: int | None = None
    tokens_per_second: float | None = None
    retrieval_ms: float | None = None


class HealthResponse(BaseModel):
    ollama_reachable: bool
    ollama_version: str | None = None
    collection_ready: bool
    indexed_chunks: int
    available_models: list[str]
    configured_models: list[str]
    missing_models: list[str]
