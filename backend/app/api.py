"""FastAPI application exposing the local HR assistant.

Run with:  uv run uvicorn app.api:app --reload
"""

import json
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

BENCH_RESULTS = (
    Path(__file__).resolve().parent.parent / "bench" / "results" / "latest.json"
)


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
        except OllamaError as exc:
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/bench")
async def bench() -> dict[str, object]:
    """Serve the most recent benchmark run, if one exists."""
    if not BENCH_RESULTS.exists():
        return {"available": False, "hint": "Run: uv run python -m bench.run_bench"}
    return {"available": True, **json.loads(BENCH_RESULTS.read_text(encoding="utf-8"))}
