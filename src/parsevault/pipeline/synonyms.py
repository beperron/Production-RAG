"""Domain synonym / acronym expansion for the lexical retrieval lane.

A query for "TPR" should match documents that say "termination of parental
rights", and vice versa. On a child-welfare + legal corpus the vocabulary is
dense with such abbreviations (GAL, CPS, ICWA, IEP, ASFA, …) and statute short
names — the single highest-leverage *lexical* precision win, because the right
document otherwise never surfaces for the abbreviated query.

Expansion is conservative and explainable: a curated, bidirectional lexicon (no
learned/automatic synonyms that could drift), applied only when a surface form
actually appears in the query, and the added terms are scored at a *reduced*
weight in BM25 so they refine ranking without overpowering the literal query.

The lexicon is data; a jurisdiction/agency can extend ``DEFAULT_SYNONYMS``.
"""

from __future__ import annotations

import re

from .retrieval import analyze

# Each group is a set of equivalent surface forms. Matching any member in a query
# contributes the *other* members' terms as low-weight expansion. Keep groups
# tight — only true equivalents, never near-related terms (e.g. ICWA "active
# efforts" and "reasonable efforts" are legally distinct and must NOT be merged).
# Some short acronyms collide with ordinary English words on case-record corpora
# ("yoga mat", "no roi this quarter", an entry titled "cac"). Marking them
# *direction-only* means the spelled-out phrase expands to add the acronym
# (still useful — a query for "medication assisted treatment" benefits from
# acronym hits in the corpus), but the bare acronym does NOT expand to the
# spelled-out phrase. This removes the precision foot-gun without losing the
# uni-directional recall win. Audit M-2.
SHORT_ACRONYMS_DIRECTION_ONLY: set[str] = {
    "mat",   # medication-assisted treatment vs. "yoga mat" / "doormat"
    "dv",    # domestic violence vs. file extensions / abbreviations
    "cac",   # child advocacy center vs. document headers
    "roi",   # release of information vs. "return on investment"
    "sud",   # substance use disorder vs. nonce text
    "posc",  # plan of safe care vs. rare strings
}

DEFAULT_SYNONYMS: list[set[str]] = [
    {"termination of parental rights", "tpr"},
    {"guardian ad litem", "gal"},
    {"court appointed special advocate", "casa"},
    {"child protective services", "cps"},
    {"department of social services", "dss"},
    {"indian child welfare act", "icwa"},
    {"adoption and safe families act", "asfa"},
    {"adoption assistance and child welfare act", "aacwa"},
    {"individualized education program", "individualized education plan", "iep"},
    {"release of information", "roi"},
    {"multidisciplinary team", "mdt"},
    {"child advocacy center", "cac"},
    {"substance use disorder", "sud"},
    {"medication assisted treatment", "mat"},
    {"domestic violence", "dv"},
    {"post traumatic stress disorder", "ptsd"},
    {"plan of safe care", "posc"},
    {"out of home placement", "out-of-home placement"},
    {"permanency planning", "permanency plan"},
    {"foster care", "out of home care"},
    {"qualified individual", "qualified residential treatment program", "qrtp"},
    # statute / authority short names and citation forms
    {"united states code", "u.s.c.", "usc"},
    {"code of federal regulations", "c.f.r.", "cfr"},
    {"north carolina general statutes", "n.c.g.s.", "ncgs"},
    {"north carolina administrative code", "n.c.a.c.", "ncac"},
    {"social security act", "ssa"},
]


class SynonymExpander:
    """Expand a query with low-weight synonym/acronym terms from a lexicon.

    Short-acronym handling (M-2): forms listed in ``direction_only`` (default
    ``SHORT_ACRONYMS_DIRECTION_ONLY``) are *not* themselves triggers for
    expansion — the spelled-out phrase still expands to add them, but a
    bare 3-letter ambiguous acronym ("mat", "dv", "roi") in the query does
    NOT pull in the clinical phrase. This removes the precision foot-gun on
    case-record corpora where these strings recur in non-clinical text.
    """

    def __init__(self, groups: list[set[str]] | None = None,
                 direction_only: set[str] | None = None):
        self.groups = groups if groups is not None else DEFAULT_SYNONYMS
        self.direction_only = (direction_only if direction_only is not None
                               else SHORT_ACRONYMS_DIRECTION_ONLY)
        # Precompile a word-boundary matcher per surface form (incl. multi-word).
        self._compiled: list[list[tuple[str, re.Pattern]]] = [
            [(s, re.compile(rf"(?<!\w){re.escape(s)}(?!\w)", re.IGNORECASE)) for s in group]
            for group in self.groups
        ]

    def expand(self, query: str) -> list[str]:
        """Return stemmed expansion terms not already in the query (deduped).

        For every group with a surface form present in the query, the *other*
        forms' analyzed terms are contributed. Direction-only acronyms in the
        query do NOT trigger expansion (M-2). Returns [] when nothing matches,
        so the common case is free.
        """
        if not query:
            return []
        ql = query.lower()
        have = set(analyze(query))
        out: list[str] = []
        for group in self._compiled:
            # Direction-only short acronyms (M-2): they're allowed as
            # EXPANSION targets (so "medication assisted treatment" pulls in
            # "mat") but NOT as triggers, so a query whose only matching form
            # is "mat" does not pull in clinical terms.
            present = [s for s, rx in group
                       if rx.search(ql) and s not in self.direction_only]
            if not present:
                continue
            present_set = set(present)
            for s, _rx in group:
                if s in present_set:
                    continue
                for t in analyze(s):
                    if t not in have:
                        out.append(t)
        return list(dict.fromkeys(out))


_DEFAULT: SynonymExpander | None = None


def default_expander() -> SynonymExpander:
    """A shared default expander (the lexicon is immutable, so it is reusable)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SynonymExpander()
    return _DEFAULT
