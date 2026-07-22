"""Unit tests for follow-up suggestions. No Ollama and no vector store."""

from pathlib import Path

import pytest

from app.rag import pick_suggestions
from app.schemas import RetrievedChunk
from app.suggestions import PassageKey, clean_question, dump, load


def chunk(section: str, *questions: str, score: float = 0.5) -> RetrievedChunk:
    """A retrieved passage carrying the given follow-up questions."""
    return RetrievedChunk(
        chunk_id=section,
        text=f"metin: {section}",
        source_file="01-izin-politikasi.md",
        doc_title="İzin Politikası",
        section=section,
        score=score,
        suggested_questions=list(questions),
    )


class TestCleanQuestion:
    @pytest.mark.parametrize(
        "raw",
        [
            "1. Evlilik izni kaç gün?",
            "2) Evlilik izni kaç gün?",
            "- Evlilik izni kaç gün?",
            "* Evlilik izni kaç gün?",
            '"Evlilik izni kaç gün?"',
            "  Evlilik izni kaç gün?  ",
        ],
    )
    def test_list_markup_is_stripped(self, raw: str) -> None:
        """The prompt forbids markup, but a local model obeys imperfectly."""
        assert clean_question(raw) == "Evlilik izni kaç gün?"

    def test_a_statement_is_rejected(self) -> None:
        assert clean_question("Evlilik izni 5 iş günüdür.") is None

    def test_too_short_is_rejected(self) -> None:
        assert clean_question("Ne?") is None

    def test_empty_is_rejected(self) -> None:
        assert clean_question("   ") is None


class TestPickSuggestions:
    def test_one_question_per_section(self) -> None:
        """Three chips should point at three parts of the handbook, not one."""
        neighbours = [
            chunk(
                "2. Mazeret İzinleri", "Evlilik izni kaç gün?", "Taşınma izni var mı?"
            ),
            chunk("5. Ücretsiz İzin", "Ücretsiz izin kaç gün?"),
        ]
        picked = pick_suggestions("Doğum izni kaç hafta?", neighbours, limit=3)
        assert picked == ["Evlilik izni kaç gün?", "Ücretsiz izin kaç gün?"]

    def test_limit_is_respected(self) -> None:
        neighbours = [
            chunk("1. Yıllık İzin", "Yıllık izin kaç gün?"),
            chunk("2. Mazeret İzinleri", "Evlilik izni kaç gün?"),
            chunk("3. Hastalık İzni", "Rapor nasıl yüklenir?"),
            chunk("4. Fazla Mesai", "Mesai ücreti kaç katı?"),
            chunk("5. Yemek Kartı", "Karta ne kadar yüklenir?"),
        ]
        assert len(pick_suggestions("Harcırah ne kadar?", neighbours, limit=3)) == 3

    def test_a_repeated_suggestion_is_offered_only_once(self) -> None:
        """Two chips reading identically waste one of only three slots."""
        neighbours = [
            chunk("2. Mazeret İzinleri", "Evlilik izni kaç gün?"),
            chunk("2.b Mazeret (tekrar)", "Evlilik izni kaç gün?"),
            chunk("5. Ücretsiz İzin", "Ücretsiz izin kaç gün?"),
        ]
        picked = pick_suggestions("Harcırah ne kadar?", neighbours, limit=3)
        assert picked == ["Evlilik izni kaç gün?", "Ücretsiz izin kaç gün?"]

    def test_questions_sharing_only_their_shape_are_both_kept(self) -> None:
        """The overlap check must not confuse a shared template with a repeat.

        "Yıllık izin kaç gün?" and "Ücretsiz izin kaç gün?" share three words of
        four and are still different questions. Any threshold tight enough to
        drop one of these would take real suggestions with it.
        """
        neighbours = [
            chunk("1. Yıllık İzin", "Yıllık izin kaç gün?"),
            chunk("5. Ücretsiz İzin", "Ücretsiz izin kaç gün?"),
        ]
        picked = pick_suggestions("Harcırah ne kadar?", neighbours, limit=3)
        assert len(picked) == 2

    def test_a_restatement_of_the_question_is_dropped(self) -> None:
        """Offering someone the question they just typed is worse than nothing."""
        neighbours = [
            chunk(
                "2. Mazeret İzinleri", "Babalık izni kaç gün?", "Evlilik izni kaç gün?"
            )
        ]
        picked = pick_suggestions("Babalık izni kaç gün?", neighbours, limit=3)
        assert picked == ["Evlilik izni kaç gün?"]

    def test_passages_without_questions_are_skipped(self) -> None:
        """A passage with no reviewed questions must not consume a slot."""
        neighbours = [
            chunk("6. İzin Bakiyesi"),
            chunk("5. Ücretsiz İzin", "Ücretsiz izin kaç gün?"),
        ]
        picked = pick_suggestions("Harcırah ne kadar?", neighbours, limit=3)
        assert picked == ["Ücretsiz izin kaç gün?"]

    def test_no_neighbours_yields_nothing(self) -> None:
        assert pick_suggestions("Harcırah ne kadar?", [], limit=3) == []


class TestPersistence:
    def test_round_trip_preserves_questions(self, tmp_path: Path) -> None:
        path = tmp_path / "suggested-questions.yaml"
        original = {
            PassageKey("01-izin-politikasi.md", "2. Mazeret İzinleri"): [
                "Evlilik izni kaç gün?"
            ]
        }
        dump(path, original)
        assert load(path) == original

    def test_a_missing_file_means_no_suggestions(self, tmp_path: Path) -> None:
        """Ingest must work before the file has ever been generated."""
        assert load(tmp_path / "absent.yaml") == {}
