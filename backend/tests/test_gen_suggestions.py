"""Regenerating one document must not delete the others' reviewed questions.

`data/suggested-questions.yaml` is hand-edited work — the whole reason drafting
detours through a file is that a person reads it first. Writing only the newly
generated passages replaced the file: measured against the real one, `--only`
on a single document took it from 37 reviewed passages to 9.
"""

from pathlib import Path

import pytest

from app import gen_suggestions
from app.config import Settings
from app.llm import StreamedChunk
from app.suggestions import PassageKey, dump, load

LEAVE_DOC = """# İzin Politikası

## 1. Yıllık Ücretli İzin

Yıllık ücretli izin hakkı hizmet süresine göre belirlenir.
"""

EXPENSE_DOC = """# Masraf Politikası

## 2. Seyahat ve Harcırah

Yurt içi harcırah 750 TL/gün olarak ödenir.
"""


class FakeOllama:
    """Returns two fixed questions for whatever passage it is handed."""

    async def chat_stream(self, model, messages, **kwargs):  # noqa: ANN001, ANN003
        yield StreamedChunk(content="Yeni soru bir kaç gün?\nYeni soru iki kaç gün?")
        yield StreamedChunk(done=True)

    async def aclose(self) -> None: ...


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "01-izin-politikasi.md").write_text(LEAVE_DOC, encoding="utf-8")
    (kb / "03-masraf-ve-yan-haklar.md").write_text(EXPENSE_DOC, encoding="utf-8")

    settings = Settings(
        kb_dir=kb,
        storage_dir=tmp_path / "storage",
        suggestions_file=tmp_path / "suggested-questions.yaml",
    )
    monkeypatch.setattr(gen_suggestions, "get_settings", lambda: settings)
    monkeypatch.setattr(gen_suggestions, "OllamaClient", lambda s: FakeOllama())
    return settings


REVIEWED = {
    PassageKey("01-izin-politikasi.md", "1. Yıllık Ücretli İzin"): [
        "Elle düzeltilmiş izin sorusu?"
    ],
    PassageKey("03-masraf-ve-yan-haklar.md", "2. Seyahat ve Harcırah"): [
        "Elle düzeltilmiş harcırah sorusu?"
    ],
}


async def test_regenerating_one_document_keeps_the_others(
    workspace: Settings,
) -> None:
    path = workspace.resolve_suggestions_file()
    dump(path, REVIEWED)

    await gen_suggestions.generate(only="01-izin-politikasi.md")

    after = load(path)
    untouched = PassageKey("03-masraf-ve-yan-haklar.md", "2. Seyahat ve Harcırah")
    assert untouched in after
    assert after[untouched] == ["Elle düzeltilmiş harcırah sorusu?"]


async def test_regenerating_one_document_replaces_its_own_questions(
    workspace: Settings,
) -> None:
    path = workspace.resolve_suggestions_file()
    dump(path, REVIEWED)

    await gen_suggestions.generate(only="01-izin-politikasi.md")

    regenerated = load(path)[
        PassageKey("01-izin-politikasi.md", "1. Yıllık Ücretli İzin")
    ]
    assert regenerated == ["Yeni soru bir kaç gün?", "Yeni soru iki kaç gün?"]


async def test_regenerating_one_document_drops_its_stale_sections(
    workspace: Settings,
) -> None:
    path = workspace.resolve_suggestions_file()
    dump(
        path,
        {
            **REVIEWED,
            # A heading that was renamed since the last run. Left in place it
            # would point at a section no chunk carries any more.
            PassageKey("01-izin-politikasi.md", "9. Kaldırılmış Bölüm"): ["Eski soru?"],
        },
    )

    await gen_suggestions.generate(only="01-izin-politikasi.md")

    after = load(path)
    assert PassageKey("01-izin-politikasi.md", "9. Kaldırılmış Bölüm") not in after
    assert PassageKey("03-masraf-ve-yan-haklar.md", "2. Seyahat ve Harcırah") in after


async def test_a_full_run_writes_every_document(workspace: Settings) -> None:
    await gen_suggestions.generate(only=None)

    after = load(workspace.resolve_suggestions_file())
    assert {key.file for key in after} == {
        "01-izin-politikasi.md",
        "03-masraf-ve-yan-haklar.md",
    }


async def test_an_unknown_document_stops_rather_than_emptying_the_file(
    workspace: Settings,
) -> None:
    path = workspace.resolve_suggestions_file()
    dump(path, REVIEWED)

    # A typo in the filename must not be a way to wipe the reviewed questions.
    with pytest.raises(SystemExit):
        await gen_suggestions.generate(only="01-izin-politikasii.md")

    assert load(path) == REVIEWED
