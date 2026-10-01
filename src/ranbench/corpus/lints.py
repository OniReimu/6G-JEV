"""RANIntent v1 text lints (mirroring edgebench): cluster leakage, literal field tokens, length, duplicates.

The 4-gram copy lint and the diversity helpers are edgebench's own (src.edgebench.corpus.lints).
"""
from __future__ import annotations

import re

from src.edgebench.corpus.lints import extract_ngrams, normalize_words
from src.ranbench.corpus.spec import FIELDS, METRIC_COLUMNS, OPTIONS

# Any mention of a cluster (or an obvious synonym) counts; for a state-dependent text every cluster is forbidden.
CLUSTER_PATTERNS: dict[str, re.Pattern[str]] = {
    "north_cluster": re.compile(r"\bnorth\w*", re.IGNORECASE),
    "south_cluster": re.compile(r"\bsouth\w*", re.IGNORECASE),
    "stadium": re.compile(r"\b(?:stadium|arena)\w*", re.IGNORECASE),
    "hospital_zone": re.compile(r"\bhospital\w*", re.IGNORECASE),
}

# Field names and option ids in their underscore form, plus the telemetry column names.
LITERAL_TOKENS: list[str] = sorted(
    {f for f in FIELDS if "_" in f}
    | {o for opts in OPTIONS.values() for o in opts if "_" in o}
    | set(METRIC_COLUMNS)
    | {"measured_at", "E2_KPM", "O1_PM", "cell_id"}
)
LITERAL_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\b(" + "|".join(map(re.escape, LITERAL_TOKENS)) + r")\b", re.IGNORECASE)


def check_cluster_leak_lint(text: str, allowed: set[str] | frozenset[str] = frozenset()) -> tuple[bool, str | None]:
    """Reject a text that mentions a cluster outside `allowed` (state-dependent texts: allowed is empty)."""
    for cluster, pat in CLUSTER_PATTERNS.items():
        if cluster in allowed:
            continue
        m = pat.search(text)
        if m:
            return False, f"Cluster mention '{m.group()}' ({cluster})"
    return True, None


def check_literal_token_lint(text: str) -> tuple[bool, str | None]:
    """Reject underscore field names / option ids and telemetry column names written verbatim."""
    m = LITERAL_TOKEN_PATTERN.search(text)
    if m:
        return False, f"Literal field token '{m.group()}'"
    return True, None


def check_length_lint(text: str, min_words: int, max_words: int) -> tuple[bool, str | None]:
    n = len(normalize_words(text))
    if n < min_words or n > max_words:
        return False, f"{n} words, allowed {min_words}-{max_words}"
    return True, None


def normalized_text(text: str) -> str:
    return " ".join(normalize_words(text))


def diversity_violation(text: str, accepted_texts: list[str], total_n: int, max_opening: float, max_5gram: float) -> str | None:
    """edgebench's incremental diversity rule: None, or the reason the text would breach a cap if accepted."""
    max_opening_count = max(1, int(total_n * max_opening))
    max_5gram_count = max(1, int(total_n * max_5gram))
    words = normalize_words(text)
    accepted_words = [normalize_words(t) for t in accepted_texts]
    if len(words) >= 4:
        opening = tuple(words[:4])
        if sum(1 for w in accepted_words if len(w) >= 4 and tuple(w[:4]) == opening) + 1 > max_opening_count:
            return "lint_diversity_opening_4gram"
    accepted_grams = [extract_ngrams(w, 5) for w in accepted_words]
    for g in extract_ngrams(words, 5):
        if sum(1 for grams in accepted_grams if g in grams) + 1 > max_5gram_count:
            return "lint_diversity_5gram"
    return None
