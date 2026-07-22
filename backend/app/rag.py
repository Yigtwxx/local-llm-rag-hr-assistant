"""RAG orchestration: retrieve, build the prompt, stream a grounded answer."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import Settings
from app.lexical import tokenize
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


# Above this share of shared words, a suggestion is treated as a restatement of
# what was just asked and dropped. Offering someone the question they typed is
# worse than offering nothing.
#
# Word overlap catches restatements, not synonyms: "Evlilik izni kaç gün?" and
# "Evlilik izni ne kadar?" share too little to be caught, and no threshold fixes
# that — "Yıllık izin kaç gün?" and "Ücretsiz izin kaç gün?" overlap by exactly
# as much and are genuinely different questions. Keeping two near-synonyms out
# of the chip row is therefore a job for whoever reviews
# `data/suggested-questions.yaml`, not for this function.
_SIMILARITY_LIMIT = 0.6


def _overlap(left: str, right: str) -> float:
    """Jaccard overlap of two questions' words, after Turkish folding."""
    first, second = set(tokenize(left)), set(tokenize(right))
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def pick_suggestions(
    question: str, neighbours: list[RetrievedChunk], limit: int
) -> list[str]:
    """Follow-up questions drawn from the passages ranked behind the answer.

    Costs nothing at answer time: the passages are already ranked and the
    questions were written at ingest, so no model is called and the stream is
    not delayed. Every suggestion is answerable by construction — it came from a
    passage that is in the index — which is the property a generated-on-the-fly
    suggestion could not offer. Proposing a question the assistant would then
    refuse is worse than proposing nothing.

    One question per section, so three chips point at three different parts of
    the handbook rather than three angles on the same paragraph.
    """
    picked: list[str] = []
    seen_sections: set[str] = set()

    for chunk in neighbours:
        if len(picked) >= limit:
            break
        if chunk.section in seen_sections or not chunk.suggested_questions:
            continue
        for candidate in chunk.suggested_questions:
            if _overlap(candidate, question) > _SIMILARITY_LIMIT:
                continue
            if any(
                _overlap(candidate, chosen) > _SIMILARITY_LIMIT for chosen in picked
            ):
                continue
            picked.append(candidate)
            seen_sections.add(chunk.section)
            break

    return picked


@dataclass
class RagEvent:
    """One step of a streaming RAG response."""

    kind: str  # "sources" | "token" | "done" | "error"
    text: str = ""
    sources: list[RetrievedChunk] | None = None
    stats: GenerationStats | None = None
    retrieval_ms: float | None = None
    grounded: bool = True
    suggestions: list[str] = field(default_factory=list)


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
        found = await self._retriever.retrieve_with_neighbours(question)
        chunks = found.chunks
        retrieval_ms = (time.perf_counter() - started) * 1000

        suggestions = pick_suggestions(
            question, found.neighbours, self._settings.max_suggestions
        )

        yield RagEvent(kind="sources", sources=chunks, retrieval_ms=retrieval_ms)

        if not chunks:
            # Refuse before spending a generation call. This is what makes the
            # "no hallucination on out-of-scope questions" guarantee hold
            # regardless of which model is selected.
            #
            # The suggestions still ride along. A refusal is exactly where a
            # user has the least idea what this assistant does know, so the
            # chips are worth more here than after a successful answer.
            yield RagEvent(kind="token", text=NO_CONTEXT_ANSWER, grounded=False)
            yield RagEvent(
                kind="done",
                retrieval_ms=retrieval_ms,
                grounded=False,
                suggestions=suggestions,
            )
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
                    kind="done",
                    stats=piece.stats,
                    retrieval_ms=retrieval_ms,
                    suggestions=suggestions,
                )
