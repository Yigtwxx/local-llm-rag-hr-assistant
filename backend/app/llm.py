"""Thin async client over the Ollama HTTP API.

We talk to Ollama directly over HTTP rather than through a wrapper library so
the request shape stays visible — which matters here, because the benchmark
depends on sending identical options to every model.
"""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error status.

    Carries two messages on purpose. The exception text is English and names
    hosts, status codes and model tags: it is what `ingest` and the benchmark
    harness print, and what lands in the server log. `user_message` is the
    Turkish sentence `/api/chat` puts in front of the reader, who gets no value
    from a status code and cannot act on an English one — the interface already
    had to stop repeating the browser's own untranslated network errors for the
    same reason.
    """

    #: Used when a raise site has nothing more specific to offer.
    DEFAULT_USER_MESSAGE = "Yanıt alınamadı. Lütfen tekrar deneyin."

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or self.DEFAULT_USER_MESSAGE


@dataclass
class GenerationStats:
    """Timing and token counts for a single generation."""

    model: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    load_duration_ns: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        """Generation throughput, excluding prompt processing.

        Derived from Ollama's own eval counters rather than wall-clock time so
        that network and streaming overhead do not distort the number.
        """
        if not self.eval_count or not self.eval_duration_ns:
            return None
        return self.eval_count / (self.eval_duration_ns / 1e9)


@dataclass
class StreamedChunk:
    """One streamed piece of an answer."""

    content: str = ""
    done: bool = False
    stats: GenerationStats | None = field(default=None)


_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)

# The one failure a reader can actually do something about, so it says what.
_UNREACHABLE_TR = (
    "Ollama'ya ulaşılamıyor. Servisin çalıştığından emin olup tekrar deneyin."
)


class OllamaClient:
    """Async client for the embedding and chat endpoints we use."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host.rstrip("/"),
            timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def version(self) -> str | None:
        try:
            response = await self._client.get("/api/version", timeout=5.0)
            response.raise_for_status()
            return response.json().get("version")
        except httpx.HTTPError:
            return None

    async def list_models(self) -> list[str]:
        """Model tags currently available locally."""
        try:
            response = await self._client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [model["name"] for model in response.json().get("models", [])]

    async def loaded_models(self) -> list[dict[str, object]]:
        """Models Ollama currently holds in memory, with its own size figures.

        Ollama reports `size_vram` per resident model. That is the figure to
        trust for "how much memory does this model need": summing process RSS
        cannot tell two co-resident models apart, and on Apple Silicon the
        unified memory pool makes the distinction invisible from the outside.
        """
        try:
            response = await self._client.get("/api/ps", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return response.json().get("models", [])

    async def unload(self, model: str) -> None:
        """Evict a model from memory immediately.

        Ollama keeps a model resident for minutes after its last use. When two
        models are benchmarked back to back, the second is therefore measured
        while the first still holds memory — inflating its peak figure by the
        size of its predecessor. Unloading in between is what makes each memory
        number attributable to a single model.
        """
        try:
            await self._client.post(
                "/api/chat",
                json={"model": model, "messages": [], "keep_alive": 0},
                timeout=30.0,
            )
        except httpx.HTTPError:
            # Best effort: a failed unload costs accuracy, not correctness.
            pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with the configured embedding model.

        Every way out of here is an `OllamaError`, which is the contract the
        rest of the app relies on: `/api/chat` turns that into an error event
        the reader can act on, and anything else escapes as an unhandled
        exception. Retries used to reraise `httpx.ConnectError` untouched, so a
        stopped Ollama reached the interface as a generic "could not finish"
        instead of saying which service was down.
        """
        if not texts:
            return []
        try:
            return await self._embed(texts)
        except _RETRYABLE as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self._settings.ollama_host}",
                user_message=_UNREACHABLE_TR,
            ) from exc

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            "/api/embed",
            json={"model": self._settings.embedding_model, "input": texts},
        )
        if response.status_code >= 400:
            # Surface Ollama's own message (e.g. "model not found") but never
            # echo the request body back, which may contain document text.
            raise OllamaError(
                f"Embedding request failed ({response.status_code}). "
                f"Is '{self._settings.embedding_model}' pulled?",
                user_message=(
                    f"Gömme modeli '{self._settings.embedding_model}' yanıt vermedi. "
                    "Modelin indirilmiş olduğundan emin olun."
                ),
            )
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise OllamaError(
                f"Expected {len(texts)} embeddings, received {len(embeddings)}."
            )
        return embeddings

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        think: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamedChunk]:
        """Stream a chat completion, yielding content as it arrives.

        `think` is always sent explicitly. Qwen models default to reasoning mode
        being ON and Gemma defaults to OFF; relying on either default would make
        the two models incomparable and would silently change behaviour when
        Ollama updates.

        `num_predict` is likewise always sent: an unbounded generation would let
        one rambling answer dominate the benchmark's averages.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {
                "temperature": (
                    self._settings.temperature if temperature is None else temperature
                ),
                "seed": self._settings.seed if seed is None else seed,
                "num_predict": (
                    self._settings.max_tokens if max_tokens is None else max_tokens
                ),
            },
        }

        started = time.perf_counter()
        first_token_at: float | None = None

        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise OllamaError(
                        f"Chat request failed ({response.status_code}). "
                        f"Is '{model}' pulled?",
                        user_message=(
                            f"'{model}' modeli yanıt vermedi. "
                            "Modelin indirilmiş olduğundan emin olun."
                        ),
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = event.get("message", {}).get("content", "")
                    if content and first_token_at is None:
                        first_token_at = time.perf_counter()

                    if event.get("done"):
                        total_ms = (time.perf_counter() - started) * 1000
                        stats = GenerationStats(
                            model=model,
                            ttft_ms=(
                                (first_token_at - started) * 1000
                                if first_token_at is not None
                                else None
                            ),
                            total_ms=total_ms,
                            prompt_eval_count=event.get("prompt_eval_count"),
                            eval_count=event.get("eval_count"),
                            eval_duration_ns=event.get("eval_duration"),
                            load_duration_ns=event.get("load_duration"),
                        )
                        yield StreamedChunk(content=content, done=True, stats=stats)
                        return

                    if content:
                        yield StreamedChunk(content=content)
        except _RETRYABLE as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self._settings.ollama_host}",
                user_message=_UNREACHABLE_TR,
            ) from exc
