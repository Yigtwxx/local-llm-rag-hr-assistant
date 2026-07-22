"""FastAPI application exposing the local HR assistant.

Run with:  uv run uvicorn app.api:app --reload
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.llm import OllamaClient, OllamaError
from app.rag import RagPipeline
from app.retrieval import Retriever, VectorStore
from app.schemas import ChatRequest, HealthResponse

logger = logging.getLogger(__name__)

BENCH_RESULTS = (
    Path(__file__).resolve().parent.parent / "bench" / "results" / "latest.json"
)

# Shown when the stream ends for a reason the caller cannot be told anything
# more useful about. Turkish, because it is rendered verbatim in the interface.
STREAM_FAILED = "Yanıt tamamlanamadı. Lütfen tekrar deneyin."


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    client = OllamaClient(settings)
    store = VectorStore(settings)
    app.state.settings = settings
    app.state.client = client
    app.state.store = store
    app.state.pipeline = RagPipeline(
        settings, client, Retriever(settings, client, store)
    )
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="NovaTek İK Asistanı",
    description="Fully local document QA over internal HR policies.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report whether Ollama, the models, and the index are all ready."""
    settings = app.state.settings
    client: OllamaClient = app.state.client
    store: VectorStore = app.state.store

    version = await client.version()
    available = await client.list_models()
    configured = [*settings.chat_models, settings.embedding_model]
    # Ollama reports tags as "name:tag"; a bare configured name matches any tag.
    missing = [
        model
        for model in configured
        if model not in available
        and not any(a.split(":")[0] == model.split(":")[0] for a in available)
    ]
    indexed = store.count()

    return HealthResponse(
        ollama_reachable=version is not None,
        ollama_version=version,
        collection_ready=indexed > 0,
        indexed_chunks=indexed,
        available_models=available,
        configured_models=configured,
        missing_models=missing,
    )


@app.get("/api/models")
async def models() -> dict[str, object]:
    """The two benchmarked chat models, plus which are actually pulled."""
    settings = app.state.settings
    client: OllamaClient = app.state.client
    available = set(await client.list_models())
    return {
        "models": [
            {
                "name": name,
                "role": role,
                "available": name in available,
            }
            for name, role in zip(
                settings.chat_models, ("primary", "secondary"), strict=False
            )
        ],
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream a grounded answer as server-sent events."""
    settings = app.state.settings
    pipeline: RagPipeline = app.state.pipeline

    model = request.model or settings.chat_model_primary
    if model not in settings.chat_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Allowed: {settings.chat_models}",
        )

    async def event_stream() -> AsyncIterator[str]:
        def sse(payload: dict[str, object]) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # Every stream has to end in exactly one of `done` or `error`. The
        # client has no other way to learn that an answer is over: HTTP status
        # is already sent by the time the first token exists, so a stream that
        # simply stops looks identical to one still arriving, and the interface
        # waits on it forever. Two ways that happened — Ollama ending its stream
        # without a `done` line, and an exception that was not `OllamaError`
        # escaping mid-generator — are both covered below.
        finished = False

        try:
            async for event in pipeline.answer(
                request.question, model, think=request.think
            ):
                if event.kind == "sources":
                    yield sse(
                        {
                            "type": "sources",
                            "sources": [
                                {
                                    "doc_title": chunk.doc_title,
                                    "section": chunk.section,
                                    "source_file": chunk.source_file,
                                    "score": chunk.score,
                                    "excerpt": chunk.text[:400],
                                    "matched_by": chunk.matched_by,
                                }
                                for chunk in (event.sources or [])
                            ],
                            "retrieval_ms": round(event.retrieval_ms or 0, 1),
                        }
                    )
                elif event.kind == "token":
                    yield sse({"type": "token", "text": event.text})
                elif event.kind == "done":
                    stats = event.stats
                    yield sse(
                        {
                            "type": "done",
                            "grounded": event.grounded,
                            # Follow-up questions, drawn from passages already
                            # in the index — no extra model call, so nothing
                            # here delays the stream or skews the metrics below.
                            "suggestions": event.suggestions,
                            "metrics": {
                                "model": model,
                                "ttft_ms": round(stats.ttft_ms, 1)
                                if stats and stats.ttft_ms
                                else None,
                                "total_ms": round(stats.total_ms, 1)
                                if stats and stats.total_ms
                                else None,
                                "eval_count": stats.eval_count if stats else None,
                                "tokens_per_second": round(stats.tokens_per_second, 2)
                                if stats and stats.tokens_per_second
                                else None,
                                "retrieval_ms": round(event.retrieval_ms or 0, 1),
                            },
                        }
                    )
                    finished = True
        except OllamaError as exc:
            # `user_message`, not `str(exc)`: the exception text names hosts and
            # status codes in English for the log and the CLI tools, and it was
            # going straight into a Turkish chat bubble.
            finished = True
            logger.warning("Chat stream failed: %s", exc)
            yield sse({"type": "error", "message": exc.user_message})
        except Exception:
            # Anything else is a bug here, not a condition the caller caused —
            # the stale Chroma handle after a re-index was one. Log it in full,
            # then still close the stream properly: once the response has begun
            # there is no status code left to fail with, and a truncated body
            # tells the reader nothing at all.
            finished = True
            logger.exception("Chat stream failed")
            yield sse({"type": "error", "message": STREAM_FAILED})

        if not finished:
            # The generator ran to completion without a terminal event, which
            # happens when Ollama's own stream ends without its `done` line.
            logger.warning("Chat stream ended without a done event")
            yield sse({"type": "error", "message": STREAM_FAILED})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/bench")
async def bench() -> dict[str, object]:
    """Serve the most recent benchmark run, if one exists.

    A run that was interrupted leaves `latest.json` half-written, and parsing
    that raised straight out of the handler as a 500 — an unreadable file is a
    missing result, not a server fault, and the panel already knows how to say
    so. The reason still goes to the log, because a truncated results file is
    something whoever ran the benchmark needs to know about.
    """
    if not BENCH_RESULTS.exists():
        return {"available": False, "hint": "Run: uv run python -m bench.run_bench"}
    try:
        payload = json.loads(BENCH_RESULTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("Could not read %s", BENCH_RESULTS)
        return {
            "available": False,
            "hint": (
                "latest.json is unreadable — re-run: uv run python -m bench.run_bench"
            ),
        }
    return {"available": True, **payload}
