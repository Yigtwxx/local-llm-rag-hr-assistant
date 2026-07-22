#!/bin/sh
# Builds the vector index on first start, then hands over to uvicorn.
#
# The index lives on a named volume, so this runs once per volume rather than
# once per container. `set -e` matters: if ingest fails (Ollama unreachable,
# embedding model not pulled) the container must exit with the error visible,
# not start up and answer "bilmiyorum" to every question.
set -e

STORAGE_DIR="${STORAGE_DIR:-/app/storage}"

if [ ! -f "$STORAGE_DIR/chroma.sqlite3" ]; then
    echo "→ Vektör indeksi bulunamadı, oluşturuluyor (bir defalık)…"
    python -m app.ingest
    echo "→ İndeks hazır."
fi

exec uvicorn app.api:app --host 0.0.0.0 --port "${API_PORT:-8000}"
