"""Every failure the Ollama client can hit must leave as an `OllamaError`.

That is the one exception `/api/chat` knows how to turn into a message the
reader can act on. Anything else reaches them as "could not finish", which does
not say which service is down or what to restart.
"""

import httpx
import pytest
from tenacity import wait_none

from app.config import Settings
from app.llm import OllamaClient, OllamaError


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the exponential wait between retries.

    The backoff is real behaviour and stays in production, but sleeping through
    it here turned a 0.3s suite into a 9s one for no extra coverage: what these
    tests check is which exception comes out, not how long it took.
    """
    monkeypatch.setattr(OllamaClient._embed.retry, "wait", wait_none())


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(Settings(ollama_host="http://127.0.0.1:11999"))


def transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


class TestEmbed:
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("timed out"),
            httpx.RemoteProtocolError("peer closed"),
        ],
    )
    async def test_unreachable_ollama_names_the_host(
        self, client: OllamaClient, failure: Exception
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise failure

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        with pytest.raises(OllamaError) as caught:
            await client.embed(["merhaba"])

        # Retries used to reraise the raw httpx error, which escaped every
        # handler above and truncated the response body.
        assert "127.0.0.1:11999" in str(caught.value)

    async def test_a_missing_model_says_which_one(self, client: OllamaClient) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        with pytest.raises(OllamaError) as caught:
            await client.embed(["merhaba"])

        assert "qwen3-embedding:0.6b" in str(caught.value)

    async def test_a_short_batch_is_refused_rather_than_misaligned(
        self, client: OllamaClient
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # One vector for two inputs: silently accepting this would pair
            # every chunk with the wrong embedding from here on.
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        with pytest.raises(OllamaError):
            await client.embed(["bir", "iki"])

    async def test_an_empty_batch_never_reaches_the_network(
        self, client: OllamaClient
    ) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"embeddings": []})

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        assert await client.embed([]) == []
        assert calls == 0


class TestProbesDegradeQuietly:
    """Health probes report absence rather than raising: the strip must render."""

    async def test_version_returns_none_when_ollama_is_down(
        self, client: OllamaClient
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        assert await client.version() is None

    async def test_list_models_returns_empty_when_ollama_is_down(
        self, client: OllamaClient
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        assert await client.list_models() == []


class TestChatStream:
    async def test_unreachable_ollama_names_the_host(
        self, client: OllamaClient
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        with pytest.raises(OllamaError) as caught:
            async for _ in client.chat_stream("qwen3.5:9b", []):
                pass

        assert "127.0.0.1:11999" in str(caught.value)

    async def test_an_error_status_says_which_model(self, client: OllamaClient) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        client._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:11999", transport=transport(handler)
        )

        with pytest.raises(OllamaError) as caught:
            async for _ in client.chat_stream("qwen3.5:9b", []):
                pass

        assert "qwen3.5:9b" in str(caught.value)
