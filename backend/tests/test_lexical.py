"""Unit tests for BM25 word search. No Ollama and no vector store required."""

import pytest

from app.lexical import BM25Index, fold, tokenize

# A miniature stand-in for the leave policy, shaped to reproduce the failure
# documented in research report §9.9: the passage that literally answers a
# paternity-leave question shares no distinctive word with the *topic* of leave,
# while several passages that cannot answer it are about leave throughout.
CORPUS = {
    "mazeret": (
        "İzin Politikası\n2. Mazeret İzinleri\n\n"
        "| Evlilik | 5 iş günü |\n"
        "| Eş doğumu (babalık izni) | 10 iş günü |\n"
        "| Taşınma | 1 iş günü |"
    ),
    "bakiye": (
        "İzin Politikası\n6. İzin Bakiyesi Sorgulama\n\n"
        "Çalışanlar güncel izin bakiyelerini İK portalındaki İzinlerim "
        "ekranından görüntüleyebilir. İzin bakiyesi her ay güncellenir."
    ),
    "hak-edis": (
        "İzin Politikası\n1.1 Hak Ediş\n\n"
        "Yıllık ücretli izin hakkı hizmet süresine göre belirlenir. "
        "1-5 yıl arası izin hakkı 16 iş günüdür."
    ),
    "harcirah": (
        "Masraf Politikası\n2.3 Günlük Yemek Harcırahı\n\n"
        "Yurt içi harcırah 750 TL/gün olarak ödenir."
    ),
}


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(CORPUS)


class TestTurkishFolding:
    def test_dotless_and_dotted_i_are_unified(self) -> None:
        """`I`/`ı` and `İ`/`i` must not split a word into two tokens."""
        assert fold("İZİN") == fold("izin")
        assert fold("IŞIK") == fold("ışık")

    def test_diacritics_are_stripped_so_ascii_typing_matches(self) -> None:
        """Employees routinely type without Turkish characters."""
        assert fold("harcırah") == fold("harcirah")
        assert fold("çğöşü") == "cgosu"

    def test_tokenize_drops_punctuation_and_table_pipes(self) -> None:
        assert tokenize("| Eş doğumu (babalık izni) | 10 iş günü |") == [
            "es",
            "dogumu",
            "babalik",
            "izni",
            "10",
            "is",
            "gunu",
        ]

    def test_tokenize_of_empty_text_is_empty(self) -> None:
        assert tokenize("") == []


class TestRanking:
    def test_rare_word_pulls_its_passage_to_the_top(self, index: BM25Index) -> None:
        """The regression this module exists for.

        "babalık" occurs in exactly one chunk, so IDF gives it nearly all of the
        query's weight and the answering passage ranks first — the position
        dense retrieval put 12th.
        """
        hits = index.rank("Babalık izni kaç gün?", limit=4)
        assert hits, "expected at least one lexical match"
        assert hits[0].chunk_id == "mazeret", (
            f"expected 'mazeret' first, got {[h.chunk_id for h in hits]}"
        )

    def test_a_query_of_only_common_words_reaches_no_chunk_of_full_rarity(
        self, index: BM25Index
    ) -> None:
        """ "izin" is in every leave document, so it must not admit anything.

        The gate demands rarity 1.0, so a question built only from corpus-wide
        words has to fall short of it everywhere.
        """
        for hit in index.rank("izin", limit=4):
            assert hit.rarity < 1.0, (
                f"a corpus-wide word identified {hit.chunk_id} on its own"
            )

    def test_unknown_words_produce_no_hits(self, index: BM25Index) -> None:
        assert index.rank("kreş servisi güzergâhı", limit=4) == []

    def test_limit_is_respected(self, index: BM25Index) -> None:
        assert len(index.rank("izin", limit=2)) <= 2

    def test_empty_index_ranks_nothing(self) -> None:
        assert BM25Index({}).rank("izin", limit=4) == []


class TestRarity:
    def test_a_word_unique_to_one_chunk_gives_full_rarity(
        self, index: BM25Index
    ) -> None:
        """ "babalık" is in exactly one chunk, which is what opens the gate."""
        hits = index.rank("babalık izni", limit=4)
        assert hits[0].rarity == pytest.approx(1.0)

    def test_rarity_stays_within_bounds(self, index: BM25Index) -> None:
        for hit in index.rank("yıllık izin hakkı harcırah", limit=4):
            assert 0.0 <= hit.rarity <= 1.0, f"rarity out of range: {hit.rarity}"

    def test_partial_match_scores_below_full_match(self, index: BM25Index) -> None:
        """A chunk holding only the common half of a query must rank lower."""
        hits = {hit.chunk_id: hit.rarity for hit in index.rank("babalık izni", 4)}
        assert hits["mazeret"] > hits.get("bakiye", 0.0)

    def test_filler_words_alone_never_reach_the_gate(self, index: BM25Index) -> None:
        """Regression for a metric that shipped the opposite of what it meant.

        The first version of this gate measured *coverage*: the share of the
        query's IDF weight the chunk contained, computed over query words that
        exist in the corpus. Words appearing nowhere were dropped from the
        denominator, so an out-of-scope question scored full marks on whatever
        filler it happened to share — "Kreş yardımı var mı?" reached 1.000
        against an unrelated passage because `kreş` matches nothing and only
        "yardımı/var/mı" were counted. The absence of the one distinctive word
        was the strongest evidence available and it was being discarded.

        Rarity cannot be gamed that way: it asks what the chunk *does* contain.
        """
        hits = index.rank("harcırah var mı", limit=4)
        matched = {hit.chunk_id: hit.rarity for hit in hits}
        # The passage that actually holds "harcırah" clears the gate.
        assert matched["harcirah"] == pytest.approx(1.0)
        # Passages sharing only filler words must not.
        for chunk_id, rarity in matched.items():
            if chunk_id != "harcirah":
                assert rarity < 1.0, f"{chunk_id} cleared the gate on filler alone"
