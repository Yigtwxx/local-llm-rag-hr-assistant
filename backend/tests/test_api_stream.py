"""The `/api/chat` SSE contract: every stream ends in exactly one terminal event.

The client has no other way to learn an answer is over. HTTP status is already
sent by the time the first token exists, so a stream that simply stops looks
identical to one still arriving and the interface waits on it forever. These
tests drive the endpoint with a stubbed pipeline; no Ollama, no vector store.
"""

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api
from app.llm import GenerationStats, OllamaError
from app.rag import RagEvent


class StubPipeline:
    """Stands in for `RagPipeline`, emitting whatever a test needs."""

    def __init__(self, events: Callable[[], AsyncIterator[RagEvent]]) -> None:
        self._events = events

    async def answer(
        self, question: str, model: str, *, think: bool = False
    ) -> AsyncIterator[RagEvent]:
        async for event in self._events():
            yield event


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestClient]:
    """A client whose lifespan builds nothing real.

    `VectorStore` would otherwise open the project's own Chroma directory and
    the test would start depending on an index having been built.
    """

    class NullClient:
        async def aclose(self) -> None: ...

    monkeypatch.setattr(api, "OllamaClient", lambda settings: NullClient())
    monkeypatch.setattr(api, "VectorStore", lambda settings: object())
    monkeypatch.setattr(api, "Retriever", lambda settings, client, store: object())
    monkeypatch.setattr(
        api, "RagPipeline", lambda settings, client, retriever: object()
    )

    with TestClient(api.app) as test_client:
        yield test_client


def events_of(response_text: str) -> list[dict[str, object]]:
    """Parse an SSE body into its decoded payloads."""
    return [
        json.loads(line[6:])
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def ask(client: TestClient) -> list[dict[str, object]]:
    response = client.post("/api/chat", json={"question": "Harcırah ne kadar?"})
    assert response.status_code == 200
    return events_of(response.text)


class TestTerminalEvent:
    def test_a_complete_answer_ends_in_done(self, client: TestClient) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            yield RagEvent(kind="sources", sources=[], retrieval_ms=5.0)
            yield RagEvent(kind="token", text="750 TL")
            yield RagEvent(
                kind="done",
                stats=GenerationStats(model="qwen3.5:9b", ttft_ms=1.0, total_ms=2.0),
                retrieval_ms=5.0,
            )

        client.app.state.pipeline = StubPipeline(events)

        assert [e["type"] for e in ask(client)] == ["sources", "token", "done"]

    def test_a_stream_that_stops_early_still_reports_an_error(
        self, client: TestClient
    ) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            yield RagEvent(kind="sources", sources=[], retrieval_ms=5.0)
            yield RagEvent(kind="token", text="750")
            # Ollama's stream ending without its `done` line looks exactly like
            # this. The answer used to just stop, and the bubble streamed on.

        client.app.state.pipeline = StubPipeline(events)
        received = ask(client)

        assert received[-1]["type"] == "error"
        assert received[-1]["message"] == api.STREAM_FAILED

    def test_an_ollama_failure_reaches_the_reader_in_turkish(
        self, client: TestClient
    ) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            yield RagEvent(kind="sources", sources=[], retrieval_ms=5.0)
            raise OllamaError(
                "Cannot reach Ollama at http://localhost:11434",
                user_message="Ollama'ya ulaşılamıyor.",
            )

        client.app.state.pipeline = StubPipeline(events)
        received = ask(client)

        assert received[-1]["type"] == "error"
        # The English text names a host and a port. It belongs in the log; it
        # was being rendered verbatim in a Turkish chat bubble.
        assert received[-1]["message"] == "Ollama'ya ulaşılamıyor."
        assert "Cannot reach" not in str(received[-1]["message"])

    def test_an_ollama_failure_without_a_specific_message_still_reads(
        self, client: TestClient
    ) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            raise OllamaError("some internal detail")
            yield  # pragma: no cover - unreachable, keeps this a generator

        client.app.state.pipeline = StubPipeline(events)
        received = ask(client)

        assert received[-1]["message"] == OllamaError.DEFAULT_USER_MESSAGE
        assert "internal detail" not in str(received[-1]["message"])

    def test_an_unexpected_failure_closes_the_stream_rather_than_truncating_it(
        self, client: TestClient
    ) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            yield RagEvent(kind="sources", sources=[], retrieval_ms=5.0)
            # A stale Chroma handle after a re-index raised exactly here, and
            # the body was cut off mid-response with no event at all.
            raise RuntimeError("Collection [abc] does not exist.")
            yield  # pragma: no cover - unreachable, keeps this a generator

        client.app.state.pipeline = StubPipeline(events)
        received = ask(client)

        assert received[-1]["type"] == "error"
        assert received[-1]["message"] == api.STREAM_FAILED

    def test_the_internal_message_is_never_leaked_to_the_caller(
        self, client: TestClient
    ) -> None:
        async def events() -> AsyncIterator[RagEvent]:
            raise RuntimeError("/Users/someone/secret/path exploded")
            yield  # pragma: no cover - unreachable, keeps this a generator

        client.app.state.pipeline = StubPipeline(events)
        received = ask(client)

        # The traceback belongs in the log, not in a chat bubble.
        assert "secret" not in json.dumps(received, ensure_ascii=False)


def test_an_unknown_model_is_rejected_before_the_stream_opens(
    client: TestClient,
) -> None:
    """A bad model is the caller's mistake, so it can still be a real status."""
    response = client.post(
        "/api/chat", json={"question": "Harcırah?", "model": "llama3.1:8b"}
    )

    assert response.status_code == 400


class TestBenchEndpoint:
    def test_a_missing_results_file_is_reported_as_unavailable(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(api, "BENCH_RESULTS", tmp_path / "nothing.json")

        payload = client.get("/api/bench").json()

        assert payload["available"] is False

    def test_an_unreadable_results_file_is_not_a_server_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A benchmark run killed part-way leaves exactly this behind.
        truncated = tmp_path / "latest.json"
        truncated.write_text('{"summaries": [{"model": "qwen', encoding="utf-8")
        monkeypatch.setattr(api, "BENCH_RESULTS", truncated)

        response = client.get("/api/bench")

        assert response.status_code == 200
        assert response.json()["available"] is False

    def test_a_readable_results_file_is_served_through(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        results = tmp_path / "latest.json"
        results.write_text('{"summaries": [], "generated_at": "2026-07-21"}', "utf-8")
        monkeypatch.setattr(api, "BENCH_RESULTS", results)

        payload = client.get("/api/bench").json()

        assert payload["available"] is True
        assert payload["generated_at"] == "2026-07-21"
