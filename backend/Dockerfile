# Build context is the repository root, so the knowledge base can be copied in
# alongside the backend source.
#
# Base images are pinned to exact patch versions so a rebuild months from now
# produces the same image; bumping them is a deliberate edit, not a side effect
# of `docker compose build`.
FROM python:3.12.13-slim

# uv resolves and installs from uv.lock; no pip, no drift from the dev setup.
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first: this layer is cached until the lockfile changes.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/app ./app
RUN uv sync --frozen --no-dev

# The knowledge base is baked in: changing it must rebuild the index anyway.
COPY data/kb /data/kb

COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# The empty named volume inherits this directory's ownership, so the non-root
# user can write the index into it.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/storage \
    && chown -R app:app /app /data
USER app

EXPOSE 8000

# /api/health always answers 200 — its body reports whether Ollama and the
# index are ready. That makes it a check of "is uvicorn up", which is exactly
# what the web service needs to wait for. The long start period covers the
# first-run ingest.
HEALTHCHECK --interval=10s --timeout=5s --start-period=300s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

ENTRYPOINT ["docker-entrypoint.sh"]
