"""Benchmark harness: measures speed, memory and answer quality per model.

Every model receives byte-identical prompts with identical options
(temperature, seed, think) so the comparison is valid. Timing comes from
Ollama's own eval counters; memory is sampled from the model runner process.

Run with:
    uv run python -m bench.run_bench
    uv run python -m bench.run_bench --think          # reasoning mode ON
    uv run python -m bench.run_bench --models qwen3.5:9b
"""

import argparse
import asyncio
import json
import platform
import re
import statistics
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psutil
import yaml

from app.config import get_settings
from app.llm import OllamaClient
from app.rag import NO_CONTEXT_ANSWER, RagPipeline
from app.retrieval import Retriever, VectorStore

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def normalize(text: str) -> str:
    """Casefold and strip accents so keyword matching is robust in Turkish.

    Turkish has the dotted/dotless i problem: `İ`.lower() is not `i` in every
    locale, and models vary in how they write `İzmir` vs `Izmir`.
    """
    lowered = text.replace("İ", "i").replace("I", "ı").casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped)


def contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)


@dataclass
class CaseResult:
    """One model answering one prompt."""

    case_id: str
    suite: str
    label: str
    model: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    retrieval_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    tokens_per_second: float | None = None
    quality_pass: bool | None = None
    quality_note: str = ""
    refused: bool | None = None
    answer_preview: str = ""


@dataclass
class ModelSummary:
    model: str
    runs: int = 0
    # Σ tokens / Σ generation time — the headline throughput figure. Weighting
    # by token count keeps short answers, where prompt processing dominates,
    # from dragging the average down as much as a long answer lifts it.
    tokens_per_second_weighted: float | None = None
    # Unweighted mean over cases, kept so the two can be compared: the gap
    # between them is itself a finding about answer-length sensitivity.
    mean_tokens_per_second: float | None = None
    stdev_tokens_per_second: float | None = None
    min_tokens_per_second: float | None = None
    max_tokens_per_second: float | None = None
    median_ttft_ms: float | None = None
    # Ollama's own resident-size figure for this model (from /api/ps).
    reported_memory_gb: float | None = None
    # Summed RSS of the runner processes. Only attributable to this model when
    # `max_models_resident` stayed at 1 for the whole run.
    peak_memory_gb: float | None = None
    max_models_resident: int = 0
    total_eval_tokens: int = 0
    quality_passed: int = 0
    quality_total: int = 0
    grounding_passed: int = 0
    grounding_total: int = 0
    load_seconds: float | None = None
    order_index: int = 0


@dataclass
class BenchReport:
    generated_at: str
    hardware: dict[str, object]
    settings: dict[str, object]
    summaries: list[ModelSummary] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    # Retrieval is the embedding model's work and is identical for every chat
    # model, so it is reported once for the run rather than per model.
    retrieval: dict[str, object] = field(default_factory=dict)


def hardware_profile() -> dict[str, object]:
    """Describe the machine under test, for the report's methodology section."""
    profile: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }
    if platform.system() == "Darwin":
        try:
            profile["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            # Apple Silicon shares one memory pool between CPU and GPU, so
            # "VRAM" is not a separate number here — worth stating explicitly.
            profile["memory_architecture"] = "unified"
        except (OSError, subprocess.SubprocessError):
            pass
    return profile


_RUNNER_NAMES = ("llama-server", "ollama runner", "ollama_llama_server")


def runner_memory_gb() -> float:
    """Resident memory of Ollama's model runner processes only, in GB.

    Ollama runs each model in a child process — `llama-server` as of 0.32 — and
    keeps `ollama serve` plus the desktop app alongside it. Matching every
    process whose command line merely mentions "ollama" swept all of those in,
    and also counted runners for models loaded by unrelated applications. That
    is how a 9B and a 12B model both reported ~34 GB on this machine: the metric
    was describing the system, not the model.
    """
    total = 0
    for process in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            name = (process.info["name"] or "").lower()
            cmdline = " ".join(process.info["cmdline"] or []).lower()
            if any(marker in cmdline or marker in name for marker in _RUNNER_NAMES):
                total += process.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return round(total / 1024**3, 2)


def weighted_tokens_per_second(cases: list[CaseResult]) -> float | None:
    """Σ generated tokens / Σ generation seconds.

    A plain mean over cases weights a 19-token answer the same as a 400-token
    one, even though the short answer's throughput is mostly prompt-processing
    overhead. The token-weighted figure is what a user actually experiences
    across a working session.
    """
    tokens = sum(c.eval_count or 0 for c in cases if c.eval_duration_ns)
    duration_ns = sum(c.eval_duration_ns or 0 for c in cases if c.eval_count)
    if not tokens or not duration_ns:
        return None
    return round(tokens / (duration_ns / 1e9), 2)


def score(answer: str, case: dict) -> tuple[bool, str]:
    """Keyword-based quality check.

    Deliberately mechanical rather than LLM-judged: the report needs numbers a
    reviewer can re-derive, and an LLM judge would add its own variance to a
    benchmark that is measuring models in the first place.
    """
    missing = [kw for kw in case.get("expect_all", []) if not contains(answer, kw)]
    if missing:
        return False, f"eksik: {', '.join(missing)}"

    any_of = case.get("expect_any", [])
    if any_of and not any(contains(answer, kw) for kw in any_of):
        return False, f"hicbiri yok: {', '.join(any_of)}"

    forbidden = [kw for kw in case.get("expect_none", []) if contains(answer, kw)]
    if forbidden:
        return False, f"olmamali: {', '.join(forbidden)}"

    return True, "ok"


_REFUSAL_SIGNALS = (
    "yer almıyor",
    "yer almamaktadır",
    "bulunmuyor",
    "bulunmamaktadır",
    "bilgi yok",
    "belirtilmemiş",
    "geçmemektedir",
    "bulamadım",
)


def first_sentence(text: str) -> str:
    """The opening sentence, which is where a refusal always appears."""
    stripped = text.strip()
    end = len(stripped)
    for terminator in (". ", ".\n", "\n"):
        found = stripped.find(terminator)
        if found != -1:
            end = min(end, found + 1)
    return stripped[:end]


def looks_like_refusal(answer: str) -> bool:
    """Did the assistant decline instead of inventing an answer?

    Only the opening sentence is examined. A model that answers correctly and
    then adds "bu konuda başka ayrıntı belirtilmemiş" has not refused, and
    matching that phrase anywhere in the text would score a correct answer as a
    failure — which is exactly what happened before this was narrowed.
    """
    if contains(answer, NO_CONTEXT_ANSWER[:40]):
        return True
    opening = first_sentence(answer)
    return any(contains(opening, signal) for signal in _REFUSAL_SIGNALS)


async def run_generation_case(
    client: OllamaClient, model: str, case: dict, think: bool
) -> CaseResult:
    """Raw model call — no retrieval, measures the model itself."""
    result = CaseResult(
        case_id=case["id"],
        suite="generation",
        label=case.get("label", case["id"]),
        model=model,
    )
    answer_parts: list[str] = []
    messages = [{"role": "user", "content": case["prompt"].strip()}]

    async for piece in client.chat_stream(model, messages, think=think):
        if piece.content:
            answer_parts.append(piece.content)
        if piece.done and piece.stats:
            result.ttft_ms = piece.stats.ttft_ms
            result.total_ms = piece.stats.total_ms
            result.eval_count = piece.stats.eval_count
            result.eval_duration_ns = piece.stats.eval_duration_ns
            result.tokens_per_second = piece.stats.tokens_per_second

    answer = "".join(answer_parts)
    result.quality_pass, result.quality_note = score(answer, case)
    result.answer_preview = answer.strip()[:300]
    return result


async def run_rag_case(
    pipeline: RagPipeline, model: str, case: dict, think: bool
) -> CaseResult:
    """Full RAG path — measures retrieval plus grounded generation."""
    result = CaseResult(
        case_id=case["id"], suite="rag", label=case["question"][:60], model=model
    )
    answer_parts: list[str] = []

    async for event in pipeline.answer(case["question"], model, think=think):
        if event.kind == "token":
            answer_parts.append(event.text)
        elif event.kind == "sources":
            result.retrieval_ms = event.retrieval_ms
        elif event.kind == "done" and event.stats:
            result.ttft_ms = event.stats.ttft_ms
            result.total_ms = event.stats.total_ms
            result.eval_count = event.stats.eval_count
            result.eval_duration_ns = event.stats.eval_duration_ns
            result.tokens_per_second = event.stats.tokens_per_second

    answer = "".join(answer_parts)
    result.answer_preview = answer.strip()[:500]
    refused = looks_like_refusal(answer)
    result.refused = refused

    if case.get("grounded", True):
        result.quality_pass, result.quality_note = score(answer, case)
        # The keyword check is the authority: if the expected facts are present,
        # the model answered — whatever else the reply says. Only annotate a
        # refusal when the answer genuinely lacks the facts.
        if refused and not result.quality_pass:
            result.quality_note = "cevap vermesi gerekirken reddetti"
    else:
        # Out-of-scope question: refusing IS the correct answer.
        result.quality_pass = refused
        result.quality_note = "dogru reddetti" if refused else "HALUSINASYON: uydurdu"

    return result


async def benchmark_model(
    model: str,
    suite: dict,
    client: OllamaClient,
    pipeline: RagPipeline,
    think: bool,
    order_index: int,
) -> tuple[ModelSummary, list[CaseResult]]:
    print(f"\n=== {model} (think={think}) ===")

    # Evict every other resident chat model — including ones loaded by unrelated
    # applications, which is why this asks Ollama what is actually in memory
    # instead of only unloading the models named on the command line. Without
    # this the memory figure is the machine's, not the model's.
    #
    # The embedding model is deliberately spared: the RAG suite needs it on
    # every question, so evicting it would only charge its reload cost to the
    # first retrieval and change nothing about the chat model being measured.
    embedding_model = get_settings().embedding_model
    for resident in await client.loaded_models():
        name = str(resident.get("model") or resident.get("name") or "")
        if name and name != model and name != embedding_model:
            print(f"  evicting {name}")
            await client.unload(name)
    await asyncio.sleep(2.0)

    # Warm-up: the first call pays the model-load cost, which would otherwise
    # be charged entirely to the first measured prompt.
    load_started = time.perf_counter()
    async for _ in client.chat_stream(
        model, [{"role": "user", "content": "Merhaba"}], think=think
    ):
        pass
    load_seconds = round(time.perf_counter() - load_started, 2)
    print(f"  warm-up / load: {load_seconds}s")

    cases: list[CaseResult] = []
    peak_memory = runner_memory_gb()
    reported_memory: float | None = None
    max_resident = 0

    async def sample_memory() -> None:
        """Peak runner RSS, plus a check that no foreign chat model appeared.

        `max_resident` counts chat models only. The embedding model is always
        resident during the RAG suite by design, so counting it would raise a
        false alarm on every single run.
        """
        nonlocal peak_memory, reported_memory, max_resident
        peak_memory = max(peak_memory, runner_memory_gb())
        resident = await client.loaded_models()
        chat_models = 0
        for entry in resident:
            name = str(entry.get("model") or entry.get("name") or "")
            if name == embedding_model:
                continue
            chat_models += 1
            size = entry.get("size_vram") or entry.get("size")
            if name == model and isinstance(size, int | float):
                reported = round(float(size) / 1024**3, 2)
                reported_memory = max(reported_memory or 0.0, reported)
        max_resident = max(max_resident, chat_models)

    await sample_memory()

    for case in suite.get("generation", []):
        result = await run_generation_case(client, model, case, think)
        cases.append(result)
        await sample_memory()
        flag = "ok" if result.quality_pass else "FAIL"
        tps = f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "?"
        print(f"  [gen ] {result.case_id:<28} {tps:>6} tok/s  {flag}")

    for case in suite.get("rag", []):
        result = await run_rag_case(pipeline, model, case, think)
        cases.append(result)
        await sample_memory()
        flag = "ok" if result.quality_pass else "FAIL"
        tps = f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "?"
        note = result.quality_note
        print(f"  [rag ] {result.case_id:<28} {tps:>6} tok/s  {flag}  {note}")

    speeds = [c.tokens_per_second for c in cases if c.tokens_per_second]
    ttfts = [c.ttft_ms for c in cases if c.ttft_ms]
    grounding = [c for c in cases if c.suite == "rag"]

    summary = ModelSummary(
        model=model,
        runs=len(cases),
        tokens_per_second_weighted=weighted_tokens_per_second(cases),
        mean_tokens_per_second=round(statistics.fmean(speeds), 2) if speeds else None,
        stdev_tokens_per_second=round(statistics.stdev(speeds), 2)
        if len(speeds) > 1
        else None,
        min_tokens_per_second=round(min(speeds), 2) if speeds else None,
        max_tokens_per_second=round(max(speeds), 2) if speeds else None,
        median_ttft_ms=round(statistics.median(ttfts), 1) if ttfts else None,
        reported_memory_gb=reported_memory,
        peak_memory_gb=peak_memory,
        max_models_resident=max_resident,
        total_eval_tokens=sum(c.eval_count or 0 for c in cases),
        quality_passed=sum(1 for c in cases if c.quality_pass),
        quality_total=len(cases),
        grounding_passed=sum(1 for c in grounding if c.quality_pass),
        grounding_total=len(grounding),
        load_seconds=load_seconds,
        order_index=order_index,
    )
    if max_resident > 1:
        print(
            f"  WARNING: {max_resident} chat models were resident during this "
            "run — another application loaded one mid-measurement, so the "
            "memory figures are not attributable to this model alone."
        )
    return summary, cases


async def main_async(models: list[str], think: bool, output_name: str) -> None:
    settings = get_settings()
    suite = yaml.safe_load((BENCH_DIR / "prompts.yaml").read_text(encoding="utf-8"))

    client = OllamaClient(settings)
    store = VectorStore(settings)

    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run: uv run python -m app.ingest")

    pipeline = RagPipeline(settings, client, Retriever(settings, client, store))

    report = BenchReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        hardware=hardware_profile(),
        settings={
            "embedding_model": settings.embedding_model,
            "indexed_chunks": store.count(),
            "top_k": settings.top_k,
            "similarity_threshold": settings.similarity_threshold,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "max_tokens": settings.max_tokens,
            "think": think,
        },
    )

    try:
        for order_index, model in enumerate(models):
            summary, cases = await benchmark_model(
                model, suite, client, pipeline, think, order_index
            )
            report.summaries.append(summary)
            report.cases.extend(cases)
    finally:
        await client.aclose()

    # Retrieval timing belongs to the embedding model and the vector store, not
    # to whichever chat model happened to be answering. Reporting it per model
    # invited a false conclusion ("retrieval is faster with qwen") from what is
    # just sampling noise, so it is aggregated once across the whole run.
    retrievals = [c.retrieval_ms for c in report.cases if c.retrieval_ms]
    if retrievals:
        report.retrieval = {
            "embedding_model": settings.embedding_model,
            "samples": len(retrievals),
            "mean_ms": round(statistics.fmean(retrievals), 1),
            "median_ms": round(statistics.median(retrievals), 1),
            "stdev_ms": round(statistics.stdev(retrievals), 1)
            if len(retrievals) > 1
            else None,
            "min_ms": round(min(retrievals), 1),
            "max_ms": round(max(retrievals), 1),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": report.generated_at,
        "hardware": report.hardware,
        "settings": report.settings,
        "summaries": [asdict(s) for s in report.summaries],
        "retrieval": report.retrieval,
        "cases": [asdict(c) for c in report.cases],
    }
    output = RESULTS_DIR / output_name
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    for summary in report.summaries:
        print(
            f"{summary.model:<20} "
            f"{summary.tokens_per_second_weighted or 0:>6.1f} tok/s (ağırlıklı)  "
            f"{summary.mean_tokens_per_second or 0:>6.1f} "
            f"±{summary.stdev_tokens_per_second or 0:>5.2f} (vaka ort.)  "
            f"TTFT {summary.median_ttft_ms or 0:>7.0f} ms  "
            f"RAM {summary.reported_memory_gb or 0:>5.1f} GB (ollama) / "
            f"{summary.peak_memory_gb or 0:>5.1f} GB (rss)  "
            f"kalite {summary.quality_passed}/{summary.quality_total}  "
            f"grounding {summary.grounding_passed}/{summary.grounding_total}"
        )
    if report.retrieval:
        print(
            f"\nretrieval (tüm koşu): {report.retrieval['median_ms']} ms medyan, "
            f"n={report.retrieval['samples']}"
        )
    print(f"\nSaved: {output}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark local models.")
    parser.add_argument(
        "--models",
        default=",".join(settings.chat_models),
        help="Comma-separated model tags.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable reasoning mode (default: off, for comparable timings).",
    )
    parser.add_argument("--output", default=None, help="Result filename.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    output = args.output or ("latest-think.json" if args.think else "latest.json")
    asyncio.run(main_async(models, args.think, output))


if __name__ == "__main__":
    main()
