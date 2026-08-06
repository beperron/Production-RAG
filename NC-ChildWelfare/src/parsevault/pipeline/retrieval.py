"""Retrieval engine: analysis, BM25F, char-n-gram fusion, query expansion.

The competitive core. Pure-Python and CPU-only, but doing the things that
separate a toy index from a strong one:

* **Stemming** (Porter) so "analyze / analysis / analyzing" match.
* **BM25F field boosting** — a hit in a heading counts more than one buried in
  the body, because headings are what a reader actually navigates by.
* **Hybrid fusion** — BM25 (lexical) fused with character-trigram cosine
  (fuzzy/morphological/typo-robust) via Reciprocal Rank Fusion, so we recover
  matches either signal alone would miss.
* **Pseudo-relevance feedback** (optional, RM3-lite) — expand the query with
  salient terms from the top results to lift recall.

Retrieval sits behind ``Retriever`` so a dense/embedding backend (on the GPU
box) can be swapped in or fused without touching callers — see ``embeddings.py``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# D3-1: Unicode-aware word matcher. The previous pattern (``[A-Za-z0-9]+``)
# dropped every non-ASCII letter, so ``q=résumé`` split to ``caf``/``sum``,
# ``q=北京`` produced ``[]``, and ``q=💔`` produced ``[]`` — the dashboard
# silently reported "0 hits" with no signal that the tokenizer ate the
# query. ``\w`` with ``re.UNICODE`` matches Latin-1 letters, CJK ideographs,
# combining marks, etc. while still excluding pure-emoji / pure-punctuation.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have he in is it its of on that the
    to was were will with this these those or not but if then else when while we
    you your they their our his her she them i me my than into over under out up
    down can could should would may might must do does did done about above below
    after before between through during without within across also which who whom
    whose what where why how all any both each few more most other some such only
    own same so too very s t can just don now""".split()
)


# --------------------------------------------------------------------------- #
# Porter stemmer (classic algorithm, compact). Stems need not be real words —
# they only need to be *consistent* between documents and queries.
# --------------------------------------------------------------------------- #
def _is_consonant(w: str, i: int) -> bool:
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _is_consonant(w, i - 1)
    return True


def _measure(stem: str) -> int:
    form = "".join("v" if not _is_consonant(stem, i) else "c" for i in range(len(stem)))
    return form.count("vc")


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_consonant(w, len(w) - 1)


def _cvc(w: str) -> bool:
    if len(w) < 3:
        return False
    if _is_consonant(w, -1 + len(w)) and not _is_consonant(w, len(w) - 2) and _is_consonant(w, len(w) - 3):
        return w[-1] not in "wxy"
    return False


def stem(word: str) -> str:
    w = word.lower()
    if len(w) <= 2:
        return w

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf):
                stem_ = w[: -len(suf)]
                if _contains_vowel(stem_):
                    w = stem_
                    if w.endswith(("at", "bl", "iz")):
                        w += "e"
                    elif _ends_double_consonant(w) and w[-1] not in "lsz":
                        w = w[:-1]
                    elif _measure(w) == 1 and _cvc(w):
                        w += "e"
                break

    # Step 1c
    if w.endswith("y") and _contains_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2
    _step2 = {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
        "izer": "ize", "abli": "able", "alli": "al", "entli": "ent", "eli": "e",
        "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate",
        "alism": "al", "iveness": "ive", "fulness": "ful", "ousness": "ous",
        "aliti": "al", "iviti": "ive", "biliti": "ble",
    }
    for suf, rep in _step2.items():
        if w.endswith(suf):
            if _measure(w[: -len(suf)]) > 0:
                w = w[: -len(suf)] + rep
            break

    # Step 3
    _step3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
              "ical": "ic", "ful": "", "ness": ""}
    for suf, rep in _step3.items():
        if w.endswith(suf):
            if _measure(w[: -len(suf)]) > 0:
                w = w[: -len(suf)] + rep
            break

    # Step 4
    _step4 = ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
              "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize")
    for suf in _step4:
        if w.endswith(suf):
            stem_ = w[: -len(suf)]
            if suf == "ion":
                if stem_ and stem_[-1] in "st" and _measure(stem_) > 1:
                    w = stem_
            elif _measure(stem_) > 1:
                w = stem_
            break
    if w.endswith("ion") and len(w) > 3 and w[-4] in "st" and _measure(w[:-3]) > 1:
        w = w[:-3]

    # Step 5a
    if w.endswith("e"):
        if _measure(w[:-1]) > 1 or (_measure(w[:-1]) == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    # Step 5b
    if _ends_double_consonant(w) and w.endswith("l") and _measure(w) > 1:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords + 1-char tokens removed (no stem)."""
    return [
        w for w in (m.group(0).lower() for m in _WORD_RE.finditer(text))
        if len(w) > 1 and w not in _STOPWORDS
    ]


def analyze(text: str) -> list[str]:
    """Indexing/query analysis: tokenize then stem."""
    return [stem(t) for t in tokenize(text)]


def char_ngrams(text: str, n: int = 3) -> Counter:
    """Character n-gram bag over the lowercased alphanumeric-collapsed text."""
    s = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    grams: Counter = Counter()
    for tok in s.split():
        padded = f" {tok} "
        for i in range(len(padded) - n + 1):
            grams[padded[i : i + n]] += 1
    return grams


def reciprocal_rank_fusion(
    rankings: list[tuple[list[tuple[str, float]], float]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id-lists (each with a weight) into one ranking.

    Rank-based and scale-free, so lexical BM25 and dense cosine scores combine
    without normalization headaches. Shared by the sparse hybrid and the
    sparse+dense fusion in DocIndex.
    """
    scores: dict[str, float] = {}
    for ranked, weight in rankings:
        for rank, (iid, _s) in enumerate(ranked):
            scores[iid] = scores.get(iid, 0.0) + weight / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[g] * b[g] for g in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
@dataclass
class _Item:
    id: str
    group: str  # e.g. document id, for scoped search
    tokens: list[str]  # analyzed body+heading (heading repeated for boost)
    charvec: Counter
    body: str  # for snippets / dense backends


class Retriever:
    """Hybrid BM25F + char-trigram retriever. Stable surface: ``search``."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        heading_boost: int = 2,
        rrf_k: int = 60,
        char_weight: float = 0.6,
        char_ngram: int = 3,
        synonyms=None,
        synonym_weight: float = 0.5,
    ):
        self.k1 = k1
        self.b = b
        self.heading_boost = heading_boost
        self.rrf_k = rrf_k
        self.char_weight = char_weight
        self.char_ngram = char_ngram
        # Optional domain synonym/acronym expander (see synonyms.py). When set,
        # the query is augmented with low-weight equivalent terms so e.g. "TPR"
        # matches "termination of parental rights". None → no expansion.
        self.synonyms = synonyms
        self.synonym_weight = synonym_weight
        self._items: dict[str, _Item] = {}

    # -- indexing -------------------------------------------------------------
    def add(self, item_id: str, body: str, *, heading: str = "", group: str = "") -> None:
        tokens = analyze(body) + analyze(heading) * self.heading_boost
        self._items[item_id] = _Item(
            id=item_id,
            group=group,
            tokens=tokens,
            charvec=char_ngrams(body + " " + heading, self.char_ngram),
            body=body,
        )

    def remove_group(self, group: str) -> None:
        for iid in [i for i, it in self._items.items() if it.group == group]:
            self._items.pop(iid, None)

    def __len__(self) -> int:
        return len(self._items)

    # -- search ---------------------------------------------------------------
    def search(
        self, query: str, k: int = 5, *, group: str | None = None, expand: bool = False
    ) -> list[tuple[str, float]]:
        scope = [it for it in self._items.values() if group is None or it.group == group]
        q_terms = analyze(query)
        if not q_terms or not scope:
            return []

        syn_terms = self.synonyms.expand(query) if self.synonyms is not None else []
        bm25 = self._bm25(q_terms, scope, expand=expand, extra_terms=syn_terms,
                          extra_weight=self.synonym_weight)
        if self.char_weight > 0:
            qvec = char_ngrams(query, self.char_ngram)
            char = sorted(
                ((it.id, _cosine(qvec, it.charvec)) for it in scope),
                key=lambda x: -x[1],
            )
            char = [(i, s) for i, s in char if s > 0]
        else:
            char = []

        fused = reciprocal_rank_fusion(
            [(bm25, 1.0), (char, self.char_weight)], rrf_k=self.rrf_k
        )
        return fused[:k]

    # -- internals ------------------------------------------------------------
    def _bm25(
        self, q_terms: list[str], scope: list[_Item], *, expand: bool,
        extra_terms: list[str] | None = None, extra_weight: float = 0.0,
    ) -> list[tuple[str, float]]:
        n = len(scope)
        avgdl = sum(len(it.tokens) for it in scope) / n
        tf_per = {it.id: Counter(it.tokens) for it in scope}

        def score_terms(terms: list[str], weight: float, acc: dict[str, float]) -> None:
            df = {t: sum(1 for it in scope if t in tf_per[it.id]) for t in set(terms)}
            for it in scope:
                tf = tf_per[it.id]
                dl = len(it.tokens) or 1
                s = 0.0
                for t in terms:
                    f = tf.get(t, 0)
                    if not f:
                        continue
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                    )
                if s:
                    acc[it.id] = acc.get(it.id, 0.0) + weight * s

        acc: dict[str, float] = {}
        score_terms(q_terms, 1.0, acc)

        # Domain synonym/acronym expansion: equivalent terms at reduced weight, so
        # an abbreviated query still ranks the documents that spell it out.
        if extra_terms and extra_weight > 0:
            novel = [t for t in extra_terms if t not in set(q_terms)]
            if novel:
                score_terms(novel, extra_weight, acc)

        if expand and acc:
            # RM3-lite: take salient terms from the top results, add at low weight.
            top = sorted(acc.items(), key=lambda x: -x[1])[: min(5, len(acc))]
            top_ids = {i for i, _ in top}
            counts: Counter = Counter()
            for it in scope:
                if it.id in top_ids:
                    counts.update(t for t in it.tokens if t not in q_terms)
            exp_terms = [t for t, _ in counts.most_common(8)]
            if exp_terms:
                score_terms(exp_terms, 0.3, acc)

        return sorted(acc.items(), key=lambda x: -x[1])
