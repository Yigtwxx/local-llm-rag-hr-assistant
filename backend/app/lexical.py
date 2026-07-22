"""Word-level (BM25) search over the same chunks the vector store holds.

Dense retrieval fails in a specific, reproducible way: it matches topics, not
words. "Babalık izni kaç gün?" ranks the passage holding `Eş doğumu (babalık
izni) | 10 iş günü` 12th of 37, because a four-word question carries too little
context for the embedding to catch anything but the general theme of leave. This
is the vocabulary-mismatch problem, documented with numbers in research report
§9.9.

BM25 fails the opposite way — it cannot see that "eş doğumu" and "babalık" mean
the same thing — which is exactly why the two are combined rather than swapped.
A rare word like "babalık" occurs in 1 of 37 chunks and therefore carries almost
all of the query's discriminative weight; a common one like "izin" occurs nearly
everywhere and carries almost none. IDF produces that ranking on its own, so no
Turkish stopword list is needed or maintained.

Nothing here touches the embeddings. The chunks, their vectors and their cosine
scores are untouched, which is what lets this be added without re-indexing and
without invalidating the benchmark runs that measured generation over them.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Standard BM25 parameters. Left at the literature defaults deliberately: with
# 37 chunks there is not enough data to tune them without overfitting the test
# set, and an untuned standard is easier to defend than a fitted guess.
K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fold(text: str) -> str:
    """Casefold Turkish text and strip diacritics, for word matching.

    Two separate problems are handled:

    `str.casefold()` alone is wrong for Turkish. `"I".casefold()` is `"i"`, but
    the lowercase of Turkish `I` is `ı`, and `İ` lowercases to `i`. Both are
    mapped explicitly before folding.

    Diacritics are then stripped, so `harcırah` and `harcirah` become the same
    token. Employees routinely type without Turkish characters, and for
    retrieval that should be a match rather than a miss.

    `bench/run_bench.py` has a similar-looking normalizer. It is deliberately
    *not* shared: that one scores model answers against expected keywords for
    six benchmark runs whose numbers are already published, and it keeps `ı`
    distinct from `i` to avoid false keyword hits. Merging them would change
    measured results to save a few lines.
    """
    lowered = text.replace("İ", "i").replace("I", "ı").casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Dotless ı survives NFKD (it has no combining mark), so fold it last.
    return stripped.replace("ı", "i")


def tokenize(text: str) -> list[str]:
    """Split text into folded word tokens, dropping punctuation and symbols."""
    return [fold(match.group()) for match in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class LexicalHit:
    """One chunk matched by word search."""

    chunk_id: str
    score: float
    # Rarity of the most distinctive query word this chunk contains, in [0, 1].
    # Used by the retrieval gate; see `rarity` for why it is not the score.
    rarity: float


class BM25Index:
    """In-memory BM25 over the indexed chunks.

    Built at startup from the documents already stored in Chroma, so there is no
    second persisted index to keep in step with the vector store. At 37 chunks
    the build is trivial; if the knowledge base grew by orders of magnitude this
    is the first thing that would need persisting.
    """

    def __init__(self, documents: dict[str, str]) -> None:
        self._doc_tokens: dict[str, Counter[str]] = {
            chunk_id: Counter(tokenize(text)) for chunk_id, text in documents.items()
        }
        self._doc_len: dict[str, int] = {
            chunk_id: sum(counts.values())
            for chunk_id, counts in self._doc_tokens.items()
        }
        total_docs = len(self._doc_tokens)
        self._avg_len = sum(self._doc_len.values()) / total_docs if total_docs else 0.0

        document_freq: Counter[str] = Counter()
        for counts in self._doc_tokens.values():
            document_freq.update(counts.keys())

        # Probabilistic IDF with the +1 smoothing that keeps every weight
        # positive. Without it a term present in more than half the corpus gets
        # a negative weight and a chunk is punished for containing it.
        self._idf: dict[str, float] = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }
        # The weight a term occurring in exactly one chunk would carry — the
        # most a matchable word can be worth here. `rarity` divides by it so the
        # gate reads as a fraction rather than as a corpus-dependent magnitude.
        self._max_idf = (
            math.log(1 + (total_docs - 1 + 0.5) / 1.5) if total_docs else 0.0
        )

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def _query_idf(self, query: str) -> dict[str, float]:
        """Query terms that exist in the corpus, with their IDF weights.

        Terms absent from every chunk are dropped: no chunk can match them, so
        they cannot contribute to any score.
        """
        return {
            term: self._idf[term] for term in set(tokenize(query)) if term in self._idf
        }

    def _score(self, chunk_id: str, weights: dict[str, float]) -> float:
        counts = self._doc_tokens[chunk_id]
        length = self._doc_len[chunk_id]
        norm = K1 * (1 - B + B * (length / self._avg_len if self._avg_len else 1.0))

        total = 0.0
        for term, idf in weights.items():
            freq = counts.get(term, 0)
            if freq:
                total += idf * (freq * (K1 + 1)) / (freq + norm)
        return total

    def rarity(self, chunk_id: str, weights: dict[str, float]) -> float:
        """How distinctive the rarest matched query word is, in [0, 1].

        1.0 means the chunk contains a word from the question that occurs in
        exactly one chunk of the whole corpus — the strongest lexical evidence
        available, and the signal that recovers the paternity-leave passage.

        Raw BM25 scores cannot serve as a gate: they are unbounded and grow with
        query length, so no fixed floor means the same thing for a four-word
        question and a twenty-word one. Rarity is bounded and answers a question
        that transfers between them.

        An earlier design measured *coverage* — the share of the query's IDF
        weight present in the chunk — and it was wrong in a way worth recording.
        Coverage divides by the weight of query words that exist in the corpus,
        so words appearing nowhere are discarded. That threw away the strongest
        evidence a question is out of scope: "Kreş yardımı var mı?" scored 1.000
        because `kreş` matches nothing and only the filler words were counted.
        """
        counts = self._doc_tokens[chunk_id]
        matched = [idf for term, idf in weights.items() if counts.get(term, 0)]
        if not matched or self._max_idf <= 0:
            return 0.0
        return min(max(matched) / self._max_idf, 1.0)

    def rank(self, query: str, limit: int) -> list[LexicalHit]:
        """Best-matching chunks for a query, highest BM25 score first."""
        weights = self._query_idf(query)
        if not weights:
            return []

        hits = [
            LexicalHit(
                chunk_id=chunk_id,
                score=score,
                rarity=self.rarity(chunk_id, weights),
            )
            for chunk_id in self._doc_tokens
            if (score := self._score(chunk_id, weights)) > 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]
