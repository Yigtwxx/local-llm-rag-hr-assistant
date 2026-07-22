"""Shared loading of the labelled question set in `prompts.yaml`.

Both `calibrate_threshold.py` and `eval_retrieval.py` read the same questions and
must agree on which are in scope, so the parsing lives here rather than in either
of them. They ask different things of it: the calibrator only needs the question
text and its scope label, while the retrieval evaluation also needs the `gold`
passages that hold the answer.

Two entry shapes are accepted in `calibration.in_scope`, a bare string and a
mapping with `question` and `gold`. The bare form predates the gold labels and
still works; it simply carries no retrieval-quality label.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BENCH_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = BENCH_DIR / "prompts.yaml"

# Tag used for the short calibration questions, shown in the calibrator's table.
SHORT_TAG = "kısa"


@dataclass(frozen=True)
class GoldChunk:
    """A passage that holds the answer, identified the way a human would.

    Not `chunk_id`: that is a hash of the text, so it changes whenever the
    document is edited and the label would silently stop matching anything.
    """

    file: str
    section: str


@dataclass(frozen=True)
class LabelledQuestion:
    """An in-scope question together with the passages that answer it."""

    question: str
    tag: str
    gold: tuple[GoldChunk, ...]


def load_suite() -> dict[str, Any]:
    """Parse `prompts.yaml`."""
    return yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))


def _entry_text(entry: str | dict[str, Any]) -> str:
    """The question text of a calibration entry, in either shape."""
    return entry if isinstance(entry, str) else str(entry["question"])


def _entry_gold(entry: str | dict[str, Any]) -> tuple[GoldChunk, ...]:
    """The gold passages of an entry; empty when it carries no label."""
    if isinstance(entry, str):
        return ()
    return tuple(
        GoldChunk(file=str(item["file"]), section=str(item["section"]))
        for item in entry.get("gold", []) or []
    )


def collect_questions(
    suite: dict[str, Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Label every question as in-scope or out-of-scope, from both sections.

    Returns `(question, tag)` pairs, where the tag is the RAG case id for the
    benchmark questions and `SHORT_TAG` for the calibration ones.
    """
    in_scope: list[tuple[str, str]] = []
    out_of_scope: list[tuple[str, str]] = []

    for case in suite.get("rag", []):
        target = in_scope if case.get("grounded", True) else out_of_scope
        target.append((case["question"], case["id"]))

    calibration = suite.get("calibration", {}) or {}
    for entry in calibration.get("in_scope", []):
        in_scope.append((_entry_text(entry), SHORT_TAG))
    for entry in calibration.get("out_of_scope", []):
        out_of_scope.append((_entry_text(entry), SHORT_TAG))

    return in_scope, out_of_scope


def labelled_questions(suite: dict[str, Any]) -> list[LabelledQuestion]:
    """Every in-scope question that carries at least one gold passage.

    Questions without a gold label are skipped rather than counted as misses:
    an unlabelled question is a gap in the test set, not a retrieval failure,
    and silently scoring it as zero would understate the system.
    """
    labelled: list[LabelledQuestion] = []

    for case in suite.get("rag", []):
        if not case.get("grounded", True):
            continue
        gold = _entry_gold(case)
        if gold:
            labelled.append(
                LabelledQuestion(question=case["question"], tag=case["id"], gold=gold)
            )

    calibration = suite.get("calibration", {}) or {}
    for entry in calibration.get("in_scope", []):
        gold = _entry_gold(entry)
        if gold:
            labelled.append(
                LabelledQuestion(question=_entry_text(entry), tag=SHORT_TAG, gold=gold)
            )

    return labelled
