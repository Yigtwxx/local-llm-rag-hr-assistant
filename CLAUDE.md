# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

A fully local document QA system ("NovaTek İK Asistanı") plus the benchmark
harness that produced the numbers in its report. No request leaves the machine:
Ollama serves both the chat models and the embedding model.

Three deliverables share one measurement source — `docs/arastirma-raporu.md`
(research report), `docs/proje-raporu.md` (project report) and
`docs/slides/index.html`. **If a benchmark number changes, all three have to be
rewritten.** Do not update one in isolation.

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12+, FastAPI (SSE), `uv` for deps |
| Vector store | ChromaDB, persisted at `backend/storage/` (gitignored) |
| Models | `qwen3.5:9b` (primary), `gemma4:12b` (secondary), `qwen3-embedding:0.6b` (1024-dim) |
| Frontend | React 19, TypeScript strict, Vite, Tailwind v4, shadcn/ui |
| Tests | `pytest` (backend), `vitest` (frontend) |

## Layout

```
backend/app/      FastAPI app: api, rag, retrieval, chunking, llm, config
backend/app/prompts/  System and answer prompts (Turkish) — never inline these in code
backend/bench/    Benchmark harness + calibration; results in bench/results/
data/kb/          Turkish HR knowledge base (Markdown) — the only corpus
docs/             Reports, slides, UI screenshots
frontend/src/     UI; lib/modelSkin.ts drives the per-model visual identity
scripts/dev.sh    Runs backend + frontend together (dev.bat on Windows)
docker-compose.yml  Demo stack: API + built UI behind nginx; Ollama stays on the host
```

## Commands

```bash
# Backend (from backend/)
uv sync
uv run python -m app.ingest          # build the vector index — required before first run
uv run uvicorn app.api:app --reload
uv run pytest
uv run ruff check . && uv run ruff format --check .

# Frontend (from frontend/)
npm install
npm run dev
npm run test                          # vitest
npm run typecheck                     # tsc -b  — NOT `tsc --noEmit`
npm run build

# Both at once
./scripts/dev.sh                      # scripts\dev.bat on Windows

# Docker (from the repository root)
docker compose up --build             # UI :8180 · API :8100
docker compose down                   # add -v to drop the index volume too
```

`tsc --noEmit` resolves the root `tsconfig.json`, which references projects and
checks nothing. Always use `tsc -b`.

## Verification gate

Before claiming work is done: `ruff check` clean, `ruff format --check` clean,
pytest green, `tsc -b` clean, vitest green, `npm run build` clean.

## Language

- Communication, reports, docs, UI copy: **Turkish**.
- Code, identifiers, comments, commit messages: **English**.
- `README.md` is the one exception: English throughout, since it is the repo's
  public face. Match the surrounding language when editing it.

## Retrieval rules

- **Similarity threshold is 0.46 and was measured, not chosen.** It is the
  highest value that misses no answerable question. An earlier 0.52 was
  calibrated on long questions only and rejected short ones such as
  "Harcırah ne kadar?" (4 of 19 in-scope questions). See `app/config.py` for
  the full reasoning.
- Two independent defences against hallucination: the threshold, and the system
  prompt. Out-of-scope questions that clear the threshold are refused by the
  prompt — verified across two models × three runs.
- Changing the knowledge base, chunking or the embedding model invalidates the
  threshold. Re-run `bench/calibrate_threshold.py`.
- `qwen3.5` has thinking on by default, `gemma4` does not. Every call sends
  `think=False` explicitly; without it the speed comparison is meaningless.

## Docker rules

- **Ollama is never containerised.** Inside a container it cannot reach Metal on
  macOS, falls back to CPU, and every number in the reports stops describing the
  system. The backend talks to the host through
  `DOCKER_OLLAMA_HOST` (default `http://host.docker.internal:11434`), with
  `extra_hosts: host-gateway` so Linux works too.
- **Ports 8180 (UI) and 8100 (API) were chosen, not defaulted.** 8000 and 3000
  belong to another project's compose on this machine, 8000 and 5173 to
  `scripts/dev.sh`. Ask before changing them. Both are overridable via
  `WEB_PORT` / `BACKEND_HOST_PORT`.
- **Never interpolate an existing `.env` key into compose.** Compose reads the
  repo's `.env`, where `OLLAMA_HOST` points at localhost — correct on the host,
  wrong inside a container. Docker-only settings use a `DOCKER_` prefix.
- nginx resolves the backend through a variable plus `resolver 127.0.0.11`, not
  a literal upstream name. With a literal name nginx refuses to start until the
  backend is resolvable, then caches its first IP across restarts.
- `proxy_buffering off` in `frontend/nginx.conf` is what keeps `/api/chat`
  streaming. Without it the answer arrives as one block and the UI looks frozen.
- The index lives in the `chroma-storage` volume; the entrypoint builds it on
  first start only. After changing the knowledge base, chunking or the embedding
  model, run `docker compose down -v` — otherwise the stale index survives.
- **Stop the stack before benchmarking** (`docker compose down`); see below.

## Benchmark rules

- **Check `docker ps` first.** Any other container that loads a model into
  Ollama contaminates the run. Two runs (run8, run9) were discarded this way.
- Always pass `--output <run-name>.json`. Never overwrite `latest.json` — that
  is the file the UI reads.
- After a run, check `stdev_tokens_per_second` and the log for
  `WARNING: N chat models were resident during this run`. A clean run sits
  around ±0.3; a contaminated one around ±9.3.
- The harness reports the model size from Ollama's `/api/ps`, not process RSS —
  weights are mmap'd, so RSS swings between 13 and 22 GB and means nothing.
- ~5 min per run, ~10 min with thinking enabled. The machine must be idle.

## Hardware

Device selection order is always CUDA → MPS → CPU, never hardcoded. The project
runs on macOS (Apple Silicon) and Windows (NVIDIA) — keep both paths working,
including in `scripts/`.

## UI conventions

- **No purple/violet.** The palette is low-chroma navy
  (`--primary: oklch(0.4 0.1 250)`; keep chroma at 0.10 or it drifts to indigo)
  plus purposeful colour: gold = cited chunk, green = ready, amber = degraded,
  red = error. Use the `--success` / `--warning` / `--gold` tokens; hardcoded
  Tailwind colours failed WCAG AA in light theme.
- **Nothing that mimics an AI writing.** No fake typewriter animation, no
  self-describing filler paragraphs. Before adding a line of copy, ask what the
  user loses without it.
- **Each model has its own skin, and the difference is carried by form, not just
  colour.** Primary is soft (larger radius, elevated, round avatar); secondary
  is flat (small radius, no elevation, square avatar). Components must not know
  which skin is active — they read `--radius`, `--elevation`,
  `--avatar-radius`, `--border`. Use the `elevate` / `elevate-lift` utilities
  and `avatar-shape` instead of hardcoding `shadow-*` or `rounded-full`.
- The skin key is `ModelInfo.role` from the backend. Never parse the model name;
  it breaks on a version bump.
- Two places cannot read CSS variables and must be kept in sync by hand:
  `BorderGlow`'s `borderRadius` prop (`ModelSkin.composerRadius`) and the `Logo`
  variant (`staggered` vs `flush`).
- `src/components/Logo.tsx` and `public/favicon.svg` are the same mark drawn
  twice — update both together.
- React Bits components do not pass this project's `verbatimModuleSyntax`,
  `erasableSyntaxOnly` and `noUncheckedIndexedAccess` settings; add `?? fallback`
  on indexed access and explicit tuple return types.

## Known limitation

"Babalık izni kaç gün?" — the chunk containing the answer ranks 12th of 37
(score 0.419) and never reaches top-4, so the system refuses. This is dense
retrieval's vocabulary-mismatch problem; the fix is hybrid search (BM25 +
vector). It was left unfixed because rebuilding the index would invalidate six
benchmark runs and the threshold calibration. Documented honestly in report
§9.9. **If work continues, this is the first task.**

## Secrets

Never hardcode credentials. All configuration goes through environment
variables or `.env` (see `.env.example`). `.env` and the source assignment PDF
are gitignored and must stay that way.
