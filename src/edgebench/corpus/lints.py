"""Corpus quality lints: forbidden scaffolds, 4-gram verbatim copy, and diversity lints."""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from src.edgebench.contract import DEFAULT_CRITERIA

FORBIDDEN_SCAFFOLD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcamera\s+(?:stream\s+)?\d+\b", re.IGNORECASE),
    re.compile(r"\bshift\s+[A-Za-z]\b", re.IGNORECASE),
    re.compile(r"\bTicket\s*#\s*\d+\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*Correction\s*:", re.IGNORECASE),
    re.compile(r"\bCorrection\s*:\s*", re.IGNORECASE),
]

CATALOG_LEAK_PATTERN: re.Pattern[str] = re.compile(
    r"\b(fallback|generic|general[- ]purpose|specialized|specialised|dedicated)\b",
    re.IGNORECASE,
)

# Field names in underscore form (only those containing an underscore), the spaced forms of
# "quality floor" and "latency class", and option ids containing an underscore. Single-word field
# names, "service type" and single-word option ids are plain English and are NOT rejected.
BASE_ENUM_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(" + "|".join(f for f in DEFAULT_CRITERIA if "_" in f) + r"|quality\s+floor|latency\s+class)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(" + "|".join(sorted({o for opts in DEFAULT_CRITERIA.values() for o in opts if "_" in o})) + r")\b",
        re.IGNORECASE,
    ),
]

# Fixed stopword sets for the code-switch lint: the most frequent function words of each language
# (articles, pronouns, prepositions, conjunctions, auxiliaries, common adverbs), no content words.
# Tokens come from normalize_words, so French elisions ("l'hôpital", "qu'il") leave the bare "l"/"qu".
# Words that are also common English content words (car, hay, hat, bin, man, plus) are left out, and a
# word present in both the English and the target set counts for neither side (see the lint below).
ENGLISH_FUNCTION_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "so", "than", "then", "because", "while", "until",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "about", "as", "over", "after",
    "before", "without", "within", "up", "out", "off",
    "this", "that", "these", "those", "there", "here", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "when", "where", "how",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "must", "may", "might", "shall",
    "not", "no", "please", "each", "every", "all", "any", "some", "only", "just", "also", "too", "very",
}

TARGET_LANG_FUNCTION_WORDS: dict[str, set[str]] = {
    "de": {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines",
        "kein", "keine", "keinen", "keinem", "keiner", "und", "oder", "aber", "doch", "denn", "weil", "dass",
        "wenn", "als", "ob", "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "mir", "dich", "dir",
        "uns", "euch", "ihm", "ihn", "ihnen", "sich", "mein", "dein", "sein", "unser", "ist", "sind", "war",
        "haben", "habe", "hast", "wird", "werden", "kann", "kannst", "können", "muss", "müssen", "soll",
        "sollte", "darf", "will", "bleiben", "bleibt", "nicht", "nur", "auch", "noch", "schon", "sehr",
        "ganz", "ruhig", "hier", "dort", "da", "jetzt", "bitte", "mit", "von", "vom", "zu", "zum", "zur",
        "auf", "aus", "bei", "beim", "nach", "für", "über", "unter", "vor", "im", "am", "an", "in", "ins",
        "ohne", "durch", "um", "bis", "alle", "alles", "nichts", "etwas", "was", "wie", "wo", "dieser",
        "diese", "dieses", "diesem", "diesen",
    },
    "fr": {
        "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux", "et", "ou", "mais",
        "donc", "ni", "que", "qu", "qui", "dont", "où", "ce", "cet", "cette", "ces", "cela", "ça",
        "c", "il", "elle", "ils", "elles", "on", "je", "j", "tu", "nous", "vous", "me", "te", "se", "lui",
        "leur", "leurs", "mon", "ma", "mes", "son", "sa", "ses", "notre", "nos", "votre", "vos", "est",
        "sont", "être", "était", "a", "ont", "avoir", "fait", "faut", "doit", "doivent", "peut", "peuvent",
        "ne", "n", "pas", "très", "sans", "avec", "pour", "par", "dans", "sur", "sous", "entre",
        "chez", "vers", "en", "y", "à", "aussi", "bien", "tout", "tous", "toute", "toutes", "aucun",
        "aucune", "si", "comme", "déjà", "encore", "ici", "là",
    },
    "es": {
        "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "del", "al", "y", "e", "o", "u",
        "pero", "sino", "que", "porque", "si", "cuando", "donde", "como", "yo", "tú", "tu", "te", "me",
        "se", "nos", "le", "les", "él", "ella", "ellos", "ellas", "nosotros", "usted", "ustedes", "mi",
        "mis", "su", "sus", "nuestro", "nuestra", "este", "esta", "esto", "estos", "estas", "ese", "esa",
        "eso", "es", "son", "está", "están", "ser", "estar", "ha", "han", "he", "has", "debe",
        "puede", "no", "sí", "ya", "muy", "más", "menos", "solo", "sólo", "también", "luego",
        "después", "antes", "aquí", "todo", "toda", "todos", "cada", "nada", "algo", "a", "de", "en",
        "con", "sin", "por", "para", "sobre", "entre", "hasta", "desde", "hacia", "según", "qué", "cuántas",
        "cuántos",
    },
}


def normalize_words(text: str) -> list[str]:
    """Lowercased, punctuation stripped, split by whitespace."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w]


def extract_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Extract all contiguous n-grams from a word list."""
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_scaffold_lint(text: str) -> tuple[bool, str | None]:
    """Reject texts that contain forbidden scaffolds."""
    for pattern in FORBIDDEN_SCAFFOLD_PATTERNS:
        match = pattern.search(text)
        if match:
            return False, f"Forbidden scaffold detected: '{match.group()}'"
    return True, None


def check_catalog_leak_lint(text: str) -> tuple[bool, str | None]:
    """Reject non-RQ4 texts that contain catalog wording leaks:
    fallback, generic, general-purpose, specialized, specialised, dedicated.
    """
    match = CATALOG_LEAK_PATTERN.search(text)
    if match:
        return False, f"Catalog wording leak detected: '{match.group()}'"
    return True, None


def check_enum_leak_lint(
    text: str,
    target_service_id: str | None = None,
    target_service_name: str | None = None,
) -> tuple[bool, str | None]:
    """Reject texts that contain underscore field names, "quality floor"/"latency class",
    underscore option IDs, or (for RQ4) the target service ID (underscore form) or its exact
    catalog name (case-insensitive).
    """
    for pat in BASE_ENUM_LEAK_PATTERNS:
        match = pat.search(text)
        if match:
            return False, f"Enum/field leak detected: '{match.group()}'"

    if target_service_id and "_" in target_service_id:
        pat_id = re.compile(rf"\b{re.escape(target_service_id)}\b", re.IGNORECASE)
        match_id = pat_id.search(text)
        if match_id:
            return False, f"Enum/field leak detected (target service id): '{match_id.group()}'"

    if target_service_name:
        pat_name = re.compile(rf"\b{re.escape(target_service_name)}\b", re.IGNORECASE)
        match_name = pat_name.search(text)
        if match_name:
            return False, f"Enum/field leak detected (target service name): '{match_name.group()}'"

    return True, None


def check_codeswitch_monolingual_lint(
    text: str, target_lang: str
) -> tuple[bool, str | None]:
    """Reject codeswitch texts that are monolingual or lack sufficient language mixing.
    For de/fr/es: requires >= 1 English and >= 1 target stopword token with >= 2 on at least one side,
    where a word in both sets (e.g. "in", "a", "no") counts for neither side. For zh: requires >= 1
    English stopword token and >= 4 CJK characters.
    """
    words = normalize_words(text)

    if target_lang == "zh":
        en_words = ENGLISH_FUNCTION_WORDS
        target_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        min_target = 4
    else:
        target_words = TARGET_LANG_FUNCTION_WORDS.get(target_lang, set())
        shared = ENGLISH_FUNCTION_WORDS & target_words
        en_words = ENGLISH_FUNCTION_WORDS - shared
        target_count = sum(1 for w in words if w in target_words and w not in shared)
        min_target = 1
    en_count = sum(1 for w in words if w in en_words)

    if en_count < 1 or target_count < min_target or (target_lang != "zh" and max(en_count, target_count) < 2):
        return (
            False,
            f"Insufficient code-switching: English function words={en_count} (min 1), {target_lang} tokens={target_count} (min {min_target})",
        )
    return True, None


def check_copy_4gram_lint(
    text: str,
    criteria_descriptions: list[str],
    target_service_desc: str | None = None,
) -> tuple[bool, str | None]:
    """Reject a text if it contains any contiguous 4-word span (lowercased, punctuation stripped)
    from any criteria description of the active fields or from its target service's catalog description.
    """
    text_words = normalize_words(text)
    text_4grams = extract_ngrams(text_words, 4)
    if not text_4grams:
        return True, None

    all_sources = list(criteria_descriptions)
    if target_service_desc:
        all_sources.append(target_service_desc)

    for src in all_sources:
        src_words = normalize_words(src)
        src_4grams = extract_ngrams(src_words, 4)
        overlap = text_4grams & src_4grams
        if overlap:
            matched_span = " ".join(next(iter(overlap)))
            return False, f"Verbatim 4-word span copied: '{matched_span}' from source '{src}'"

    return True, None


def compute_diversity_stats(
    texts: list[str],
) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], int]]:
    """Count number of distinct texts containing each opening 4-gram and word 5-gram."""
    opening_counts: Counter[tuple[str, ...]] = Counter()
    ngram_5_counts: Counter[tuple[str, ...]] = Counter()

    for txt in texts:
        words = normalize_words(txt)
        if len(words) >= 4:
            opening = tuple(words[:4])
            opening_counts[opening] += 1
        seen_5grams = extract_ngrams(words, 5)
        for g in seen_5grams:
            ngram_5_counts[g] += 1

    return dict(opening_counts), dict(ngram_5_counts)


def check_diversity_lint(
    texts: list[str],
    max_5gram_ratio: float = 0.10,
    max_opening_4gram_ratio: float = 0.05,
) -> tuple[bool, list[int], dict[str, Any]]:
    """Check that no word 5-gram occurs in more than max_5gram_ratio of texts
    and no opening 4-gram occurs in more than max_opening_4gram_ratio.

    Returns (passed, list_of_violating_text_indices, details_dict).
    """
    n = len(texts)
    if n <= 1:
        return True, [], {}

    max_opening_count = max(1, int(n * max_opening_4gram_ratio))
    max_5gram_count = max(1, int(n * max_5gram_ratio))

    violating_indices: set[int] = set()
    seen_openings: Counter[tuple[str, ...]] = Counter()
    seen_5grams: Counter[tuple[str, ...]] = Counter()

    details: dict[str, Any] = {
        "n_texts": n,
        "max_allowed_opening_count": max_opening_count,
        "max_allowed_5gram_count": max_5gram_count,
        "violations": [],
    }

    for idx, txt in enumerate(texts):
        words = normalize_words(txt)
        is_violating = False

        if len(words) >= 4:
            opening = tuple(words[:4])
            seen_openings[opening] += 1
            if seen_openings[opening] > max_opening_count:
                is_violating = True
                details["violations"].append({
                    "index": idx,
                    "type": "opening_4gram",
                    "span": " ".join(opening),
                    "count": seen_openings[opening],
                })

        text_5grams = extract_ngrams(words, 5)
        for g in text_5grams:
            seen_5grams[g] += 1
            if seen_5grams[g] > max_5gram_count and not is_violating:
                is_violating = True
                details["violations"].append({
                    "index": idx,
                    "type": "word_5gram",
                    "span": " ".join(g),
                    "count": seen_5grams[g],
                })

        if is_violating:
            violating_indices.add(idx)

    passed = len(violating_indices) == 0
    return passed, sorted(violating_indices), details


def get_top_repeated_5grams(texts: list[str], top_k: int = 5) -> list[tuple[str, int]]:
    """Return the top repeated 5-grams across the texts with their occurrence counts."""
    ngram_counts: Counter[str] = Counter()
    for txt in texts:
        words = normalize_words(txt)
        grams = extract_ngrams(words, 5)
        for g in grams:
            ngram_counts[" ".join(g)] += 1

    return ngram_counts.most_common(top_k)
