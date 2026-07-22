"""RAG orchestration: retrieve, build the prompt, stream a grounded answer."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import Settings
from app.llm import GenerationStats, OllamaClient
from app.retrieval import Retriever
from app.schemas import RetrievedChunk

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Shown verbatim when retrieval finds nothing above the similarity floor, so the
# refusal never depends on the model behaving well.
NO_CONTEXT_ANSWER = (
    "Bu bilgi elimdeki İK dokümanlarında yer almıyor. "
    "İK ekibine ik@novatek.example adresinden ulaşabilirsiniz."
)


@lru_cache
def load_prompt(name: str) -> str:
    """Read a prompt template from disk, cached after first use."""
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, attributable context block."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Kaynak {index} — {chunk.doc_title} › {chunk.section}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


@dataclass
class RagEvent:
    """One step of a streaming RAG response."""

    kind: str  # "sources" | "token" | "done" | "error"
    text: str = ""
    sources: list[RetrievedChunk] | None = None
    stats: GenerationStats | None = None
    retrieval_ms: float | None = None
    grounded: bool = True


class RagPipeline:
    """Ties retrieval and generation together for the chat endpoint."""

    def __init__(
        self, settings: Settings, client: OllamaClient, retriever: Retriever
    ) -> None:
        self._settings = settings
        self._client = client
        self._retriever = retriever

    async def answer(
        self, question: str, model: str, *, think: bool = False
    ) -> AsyncIterator[RagEvent]:
        import time

        started = time.perf_counter()
        chunks = await self._retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - started) * 1000

        yield RagEvent(kind="sources", sources=chunks, retrieval_ms=retrieval_ms)

        if not chunks:
            # Refuse before spending a generation call. This is what makes the
            # "no hallucination on out-of-scope questions" guarantee hold
            # regardless of which model is selected.
            yield RagEvent(kind="token", text=NO_CONTEXT_ANSWER, grounded=False)
            yield RagEvent(kind="done", retrieval_ms=retrieval_ms, grounded=False)
            return

        messages = [
            {"role": "system", "content": load_prompt("system_tr.txt")},
            {
                "role": "user",
                "content": load_prompt("answer_tr.txt").format(
                    context=build_context(chunks), question=question
                ),
            },
        ]

        async for piece in self._client.chat_stream(model, messages, think=think):
            if piece.content:
                yield RagEvent(kind="token", text=piece.content)
            if piece.done:
                yield RagEvent(
                    kind="done", stats=piece.stats, retrieval_ms=retrieval_ms
                )
