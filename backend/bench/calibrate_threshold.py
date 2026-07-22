"""Derive the retrieval similarity threshold from the labelled question set.

The threshold decides when the assistant refuses instead of answering, so it
should come from measurement rather than intuition.

Two question styles are scored together: the long, well-formed questions of the
benchmark suite and the short ones people actually type ("Harcırah ne kadar?").
Calibrating on the long style alone produced 0.52 — a value that then rejected
short questions whose answers are plainly in the documents. Because the two
groups overlap once both styles are present, the script does not look for a
clean separating gap; it sweeps candidate thresholds and reports the two error
types at each, which is the trade-off the choice actually involves.

Run with:  uv run python -m bench.calibrate_threshold
"""

import asyncio
from pathlib import Path

import yaml

from app.config import get_settings
from app.llm import OllamaClient
from app.retrieval import VectorStore

BENCH_DIR = Path(__file__).resolve().parent
SWEEP = (0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60)


def collect_questions(
    suite: dict,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Label every question as in-scope or out-of-scope, from both sections."""
    in_scope: list[tuple[str, str]] = []
    out_of_scope: list[tuple[str, str]] = []

    for case in suite.get("rag", []):
        target = in_scope if case.get("grounded", True) else out_of_scope
        target.append((case["question"], case["id"]))

    calibration = suite.get("calibration", {}) or {}
    for question in calibration.get("in_scope", []):
        in_scope.append((question, "kısa"))
    for question in calibration.get("out_of_scope", []):
        out_of_scope.append((question, "kısa"))

    return in_scope, out_of_scope


async def calibrate() -> None:
    settings = get_settings()
    client = OllamaClient(settings)
    store = VectorStore(settings)

    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run: uv run python -m app.ingest")

    suite = yaml.safe_load((BENCH_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    in_questions, out_questions = collect_questions(suite)

    async def top_score(question: str) -> float:
        embedding = (await client.embed([question]))[0]
        hits = store.query(embedding, settings.top_k)
        return hits[0].score if hits else 0.0

    try:
        in_scope = [(await top_score(q), q, tag) for q, tag in in_questions]
        out_of_scope = [(await top_score(q), q, tag) for q, tag in out_questions]
    finally:
        await client.aclose()

    print(f"Embedding model : {settings.embedding_model}")
    print(f"Indexed chunks  : {store.count()}\n")

    print("Kapsam içi (top-1 benzerlik, artan):")
    for score, question, tag in sorted(in_scope):
        print(f"  {score:.3f}  [{tag:<4}] {question}")

    print("\nKapsam dışı (azalan):")
    for score, question, tag in sorted(out_of_scope, reverse=True):
        print(f"  {score:.3f}  [{tag:<4}] {question}")

    if not in_scope or not out_of_scope:
        return

    lowest_in = min(score for score, _, _ in in_scope)
    highest_out = max(score for score, _, _ in out_of_scope)
    print(f"\nEn düşük kapsam içi : {lowest_in:.3f}")
    print(f"En yüksek kapsam dışı: {highest_out:.3f}")

    if highest_out >= lowest_in:
        print(
            "\nGruplar örtüşüyor — tek bir eşik ikisini birden ayıramaz. "
            "Aşağıdaki tablo, seçilecek eşiğin hangi hatayı satın aldığını gösterir.\n"
        )
    else:
        print(f"Ayrım aralığı: {highest_out:.3f} .. {lowest_in:.3f}\n")

    print(f"{'eşik':>6}  {'kaçırılan doğru soru':>22}  {'sızan kapsam dışı':>19}")
    for threshold in SWEEP:
        missed = [q for score, q, _ in in_scope if score < threshold]
        leaked = [q for score, q, _ in out_of_scope if score >= threshold]
        marker = (
            " ←  yapılandırılmış" if threshold == settings.similarity_threshold else ""
        )
        print(
            f"{threshold:>6.2f}  {len(missed):>10}/{len(in_scope):<11}"
            f"  {len(leaked):>8}/{len(out_of_scope):<10}{marker}"
        )

    print(
        "\nKaçırılan doğru soru = cevabı dokümanda olduğu hâlde reddedilen soru.\n"
        "Sızan kapsam dışı  = eşiği geçip modele ulaşan cevapsız soru. Bunlar "
        "hâlâ sistem prompt'u tarafından reddedilir; eşik tek savunma değildir.\n"
        "Bu asimetri nedeniyle eşik, sıfır kaçırma sağlayan en yüksek değere "
        "ayarlanmalıdır."
    )


if __name__ == "__main__":
    asyncio.run(calibrate())
