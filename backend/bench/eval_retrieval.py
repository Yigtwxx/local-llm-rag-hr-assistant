"""Measure whether retrieval actually delivers the passage that holds the answer.

`calibrate_threshold.py` only asks whether a question cleared the similarity
floor. It never asks whether the passage that came back *contains the answer* —
which is why it reported "0/19 missed" at a threshold where at least one
answerable question could not reach its answer. Research report §9.9 records
that gap as a flaw in the measurement, not just in the system.

This script closes it. Every labelled question in `prompts.yaml` carries a
`gold` passage, and three numbers are reported against it:

  sıra (rank)  where the gold passage sits in the full ranking of all chunks.
               Diagnostic: it explains *why* a question fails.
  Recall@k     whether the gold passage is inside the top-k the ranking would
               hand over. Measures the ranker alone, ignoring the floor.
  ulaştı       whether the gold passage survives the real pipeline — ranking
               *and* the similarity floor — and actually reaches the model.
               This is the number that describes what a user experiences.

The three come apart exactly where the interesting failures are. "Babalık izni
kaç gün?" has its answer written verbatim in the knowledge base, yet ranks 12th
and scores below the floor: Recall@4 fails and `ulaştı` fails, and the rank
column says the cause is ordering rather than the threshold.

Run with:
    uv run python -m bench.eval_retrieval --output retrieval-baseline.json
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.llm import OllamaClient
from app.retrieval import Retriever, VectorStore
from app.schemas import RetrievedChunk
from bench.questions import GoldChunk, LabelledQuestion, labelled_questions, load_suite

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def chunk_key(chunk: RetrievedChunk) -> tuple[str, str]:
    """Identity used to match a retrieved chunk against a gold label."""
    return (chunk.source_file, chunk.section)


def gold_key(gold: GoldChunk) -> tuple[str, str]:
    return (gold.file, gold.section)


@dataclass
class QuestionResult:
    """How retrieval performed on one labelled question."""

    question: str
    tag: str
    # 1-based position of the best-ranked gold passage among all chunks.
    gold_rank: int | None
    gold_score: float | None
    top1_score: float | None
    hit_at_k: bool
    delivered: bool
    refused: bool
    # Passages the model actually received, so two runs can be diffed to prove
    # whether a retrieval change altered what generation saw.
    delivered_chunks: list[str]


def verify_gold_labels(
    questions: list[LabelledQuestion], indexed: set[tuple[str, str]]
) -> None:
    """Fail loudly when a gold label matches nothing in the index.

    A mislabelled question would otherwise be scored as a permanent miss, and
    the whole point of this script is to be trusted about misses. A wrong
    instrument is worse than no instrument, so this refuses to run instead of
    reporting a plausible but false number.
    """
    unknown: list[str] = []
    for item in questions:
        for gold in item.gold:
            if gold_key(gold) not in indexed:
                unknown.append(f"  {item.question!r} → {gold.file} › {gold.section}")

    if unknown:
        raise SystemExit(
            "Gold labels that match no indexed chunk:\n"
            + "\n".join(unknown)
            + "\n\nThe knowledge base, the chunking or the labels have drifted "
            "apart. Fix prompts.yaml or re-run `python -m app.ingest` before "
            "trusting any number from this script."
        )


async def evaluate() -> tuple[list[QuestionResult], dict[str, object]]:
    settings = get_settings()
    client = OllamaClient(settings)
    store = VectorStore(settings)

    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run: uv run python -m app.ingest")

    retriever = Retriever(settings, client, store)
    suite = load_suite()
    questions = labelled_questions(suite)
    if not questions:
        raise SystemExit("No gold-labelled questions found in prompts.yaml.")

    total = store.count()

    try:
        # The full ranking, used to locate the gold passage and to collect every
        # indexed chunk's identity for label verification.
        probe = await client.embed(["kapsam kontrolu"])
        if not probe:
            raise SystemExit("Embedding model unreachable — is Ollama running?")
        indexed = {chunk_key(c) for c in store.query(probe[0], total)}
        verify_gold_labels(questions, indexed)

        results: list[QuestionResult] = []
        for item in questions:
            embeddings = await client.embed([item.question])
            ranking = store.query(embeddings[0], total)
            wanted = {gold_key(g) for g in item.gold}

            gold_rank: int | None = None
            gold_score: float | None = None
            for position, chunk in enumerate(ranking, start=1):
                if chunk_key(chunk) in wanted:
                    gold_rank, gold_score = position, chunk.score
                    break

            # The authoritative answer to "what did the model see" — the real
            # pipeline, floor included. Costs a second embedding call per
            # question; that is cheaper than duplicating the gate logic here
            # and letting the two drift apart.
            delivered_chunks = await retriever.retrieve(item.question)

            results.append(
                QuestionResult(
                    question=item.question,
                    tag=item.tag,
                    gold_rank=gold_rank,
                    gold_score=gold_score,
                    top1_score=ranking[0].score if ranking else None,
                    hit_at_k=gold_rank is not None and gold_rank <= settings.top_k,
                    delivered=any(chunk_key(c) in wanted for c in delivered_chunks),
                    refused=not delivered_chunks,
                    delivered_chunks=[
                        f"{c.source_file} › {c.section}" for c in delivered_chunks
                    ],
                )
            )
    finally:
        await client.aclose()

    meta: dict[str, object] = {
        "embedding_model": settings.embedding_model,
        "indexed_chunks": total,
        "top_k": settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
    }
    return results, meta


def summarize(results: list[QuestionResult], top_k: int) -> dict[str, object]:
    """Aggregate the per-question numbers."""
    count = len(results)
    ranked = [r for r in results if r.gold_rank is not None]
    # MRR over every question: a gold passage that never appears contributes 0,
    # which is the standard convention and the honest one here.
    reciprocal = sum(1.0 / r.gold_rank for r in ranked if r.gold_rank)
    return {
        "questions": count,
        "recall_at_k": round(sum(r.hit_at_k for r in results) / count, 4),
        "mrr": round(reciprocal / count, 4),
        "delivered": round(sum(r.delivered for r in results) / count, 4),
        "refused": sum(r.refused for r in results),
        "k": top_k,
    }


def print_report(
    results: list[QuestionResult], meta: dict[str, object], summary: dict[str, object]
) -> None:
    print(f"Embedding model : {meta['embedding_model']}")
    print(f"Indexed chunks  : {meta['indexed_chunks']}")
    print(f"top_k / eşik    : {meta['top_k']} / {meta['similarity_threshold']}\n")

    print(f"{'sıra':>5}  {'skor':>6}  {'@k':>3}  {'ulaştı':>7}  soru")
    for result in sorted(results, key=lambda r: r.gold_rank or 10**6):
        rank = str(result.gold_rank) if result.gold_rank else "—"
        score = f"{result.gold_score:.3f}" if result.gold_score is not None else "—"
        print(
            f"{rank:>5}  {score:>6}  {'✓' if result.hit_at_k else '✗':>3}"
            f"  {'✓' if result.delivered else '✗':>7}  {result.question}"
        )

    k = summary["k"]
    print(f"\nRecall@{k} : {summary['recall_at_k']:.3f}  (sıralama tek başına)")
    print(f"MRR      : {summary['mrr']:.3f}")
    print(
        f"Ulaştı   : {summary['delivered']:.3f}  "
        f"(sıralama + eşik — kullanıcının yaşadığı sayı)"
    )

    failures = [r for r in results if not r.delivered]
    if failures:
        print("\nCevabına ulaşamayan sorular:")
        for result in failures:
            rank = result.gold_rank or "—"
            reason = (
                "eşiğin altında"
                if result.hit_at_k
                else f"ilk {summary['k']}'e giremedi (sıra {rank})"
            )
            print(f"  {result.question}  →  {reason}")


async def main_async(output_name: str | None) -> None:
    results, meta = await evaluate()
    summary = summarize(results, int(meta["top_k"]))  # type: ignore[arg-type]
    print_report(results, meta, summary)

    if not output_name:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "settings": meta,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    output = RESULTS_DIR / output_name
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nYazıldı: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether retrieval delivers the answering passage."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Result filename under bench/results/. Omit to only print.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.output))


if __name__ == "__main__":
    main()
