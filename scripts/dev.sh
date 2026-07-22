#!/usr/bin/env bash
#
# Starts the backend and the frontend together for local development.
#
# Both logs stream into this one terminal, each line tagged with its source, so
# a traceback and the request that caused it stay next to each other. Nothing
# is filtered or cleared — the point of running them together is to see both.
#
# The two processes are also torn down as a group: Ctrl+C, a closed terminal,
# or either server crashing stops the other one too. Leaving a stray uvicorn
# holding port 8000 is the usual way to lose ten minutes to a "connection
# refused" that has nothing to do with the code.
#
# Usage:  ./scripts/dev.sh            (from anywhere)
#         BACKEND_PORT=8001 ./scripts/dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

die() {
    echo "hata: $*" >&2
    exit 1
}

# --- Preflight ----------------------------------------------------------
# Checked up front rather than left to fail mid-startup, where the error
# surfaces as a stack trace from whichever server lost the race.

command -v uv >/dev/null || die "uv bulunamadı — https://docs.astral.sh/uv/"
command -v npm >/dev/null || die "npm bulunamadı — Node.js ≥ 20 gerekiyor"

port_busy() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

port_busy "$BACKEND_PORT" && die "$BACKEND_PORT portu dolu (BACKEND_PORT ile değiştirebilirsiniz)"
port_busy "$FRONTEND_PORT" && die "$FRONTEND_PORT portu dolu (FRONTEND_PORT ile değiştirebilirsiniz)"

# The vector store is built by `uv run python -m app.ingest`; without it every
# question comes back as "bilmiyorum" and the cause is not visible in the UI.
[ -f "$ROOT/backend/storage/chroma.sqlite3" ] ||
    die "indeks yok — önce: cd backend && uv run python -m app.ingest"

# Ollama being down is recoverable while the app runs, so it is a warning.
if ! curl -fsS --max-time 2 "$OLLAMA_URL" >/dev/null 2>&1; then
    echo "uyarı: Ollama $OLLAMA_URL adresinde yanıt vermiyor — 'ollama serve' çalıştırın"
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "→ frontend bağımlılıkları kuruluyor…"
    (cd "$ROOT/frontend" && npm install)
fi

# --- Run ----------------------------------------------------------------

# Both servers write through a pipe rather than to a terminal, which normally
# flips them into block buffering — logs would then arrive in bursts, minutes
# after the request that produced them.
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

# Job control puts each background job in its own process group, so the
# cleanup below can signal a server together with everything it spawned —
# `npm run dev` in particular is a wrapper that outlives a bare kill.
set -m

pids=()

# shellcheck disable=SC2317  # invoked through the trap
cleanup() {
    trap - EXIT INT TERM
    # `${pids[@]}` on an empty array is an unbound-variable error under
    # `set -u` in bash 3.2, which macOS still ships.
    if [ "${#pids[@]}" -gt 0 ]; then
        for pid in "${pids[@]}"; do
            # Negative PID signals the whole process group; the plain PID is
            # the fallback for the case where job control was unavailable.
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        done
    fi
    # Killed explicitly, or the `wait` below blocks until it finishes polling.
    [ -n "${opener_pid:-}" ] && kill "$opener_pid" 2>/dev/null
    wait 2>/dev/null || true
    return 0
}
trap cleanup EXIT INT TERM

# Tags every line with its source. Both streams are merged so stderr — where
# uvicorn logs and every traceback go — is never dropped.
start_tagged() {
    local tag=$1 color=$2 dir=$3
    shift 3
    (
        cd "$dir" || exit 1
        "$@" 2>&1 | while IFS= read -r line; do
            printf '\033[%sm%s\033[0m %s\n' "$color" "$tag" "$line"
        done
    ) &
    pids+=("$!")
}

start_tagged "[backend ]" "36" "$ROOT/backend" \
    uv run uvicorn app.api:app --reload --port "$BACKEND_PORT"

start_tagged "[frontend]" "35" "$ROOT/frontend" \
    npm run dev -- --port "$FRONTEND_PORT"

# Opens the UI itself rather than leaving it to a clicked link. A browser that
# is not already running restores its own session or start page first, so the
# link lands behind whatever was open last. Chrome's `--app` window has no
# session restore, no start page and no tab strip: the address given is the
# only thing that appears.
open_ui() {
    local url="http://localhost:$FRONTEND_PORT"
    local chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    local attempt

    # Waiting for a real response, not a fixed sleep — opening early shows a
    # connection error page that does not refresh itself.
    for attempt in $(seq 1 60); do
        curl -fsS --max-time 1 "$url" >/dev/null 2>&1 && break
        sleep 0.25
    done

    if [ -x "$chrome" ]; then
        "$chrome" --app="$url" >/dev/null 2>&1 &
    elif command -v open >/dev/null; then
        open "$url" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null; then
        xdg-open "$url" >/dev/null 2>&1 || true
    fi
}

opener_pid=""
if [ -z "${NO_OPEN:-}" ]; then
    open_ui &
    opener_pid=$!
fi

echo
echo "  backend   http://127.0.0.1:$BACKEND_PORT   (dokümantasyon: /docs)"
echo "  arayüz    http://localhost:$FRONTEND_PORT   (hazır olunca kendiliğinden açılır)"
echo "  durdurmak için Ctrl+C · tarayıcı açılmasın: NO_OPEN=1 ./scripts/dev.sh"
echo

# Watch both, and leave as soon as either one dies, so a crashed backend does
# not leave the frontend running against nothing. `wait -n` would say this in
# one line but does not exist in bash 3.2, which is what macOS ships.
while :; do
    for pid in "${pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo
            echo "sunuculardan biri durdu — diğeri de kapatılıyor."
            exit 1
        fi
    done
    sleep 1
done
