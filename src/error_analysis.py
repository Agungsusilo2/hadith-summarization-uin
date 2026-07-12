"""Heuristic error analysis for generated summaries."""

from __future__ import annotations

import re

import pandas as pd

from .utils import word_count


NUMBER_RE = re.compile(r"\d+")
INDONESIAN_HINTS = {"yang", "dan", "di", "dari", "dengan", "adalah", "tidak", "kepada", "rasulullah"}


def _numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(str(text)))


def _possibly_not_indonesian(text: str) -> bool:
    words = set(str(text).lower().split())
    if not words:
        return False
    return len(words & INDONESIAN_HINTS) == 0 and len(words) > 8


# Capitalized words that recur across almost any hadith summary regardless of
# the specific source sentence - flagging these alone as "unsourced" would be
# a false positive on nearly every row, so they are never flagged by
# themselves. A genuinely new full name/phrase built around one of these
# (e.g. "Mohammed bin Salman") is still caught via its other, more distinctive
# token(s) (e.g. "Salman").
COMMON_ISLAMIC_TERMS = {
    "muhammad", "nabi", "rasul", "rasulullah", "messenger", "allah",
    "islam", "muslim", "quran", "qur'an", "alquran", "al-quran",
}


def _unsourced_capitalized_terms(summary: str, source: str) -> list[str]:
    """Return mid-sentence capitalized tokens in `summary` absent from `source`.

    This is a heuristic signal for *possible* hallucination, not a definitive
    verdict: it flags likely-unsourced names/places (mT5 has been observed to
    introduce a narrator's supposed hometown, or an unrelated public figure,
    that does not appear anywhere in the hadith text it was summarizing), but
    a flagged row still needs human judgement. Only mid-sentence
    capitalization is checked because Indonesian capitalizes every
    sentence-initial word regardless of whether it is a proper noun, and
    common Islamic terms (see COMMON_ISLAMIC_TERMS) are excluded because they
    legitimately recur in almost every hadith regardless of the source
    sentence.
    """
    candidates = re.findall(r"(?<=[a-z]\s)([A-Z][a-zA-Z'-]{2,})", str(summary))
    source_lower = str(source).lower()
    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen or key in COMMON_ISLAMIC_TERMS:
            continue
        seen.add(key)
        if key not in source_lower:
            found.append(candidate)
    return found


def _has_repetition(text: str) -> bool:
    words = str(text).lower().split()
    if len(words) < 6:
        return False
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    return len(bigrams) != len(set(bigrams)) and len(bigrams) > 5


def analyze_row(row: pd.Series) -> list[str]:
    """Return heuristic flags for one processed row."""
    flags: list[str] = []
    source = str(row.get("teks_sumber_ringkasan", ""))
    extractive = str(row.get("ringkasan_extractive", ""))
    abstractive = str(row.get("ringkasan_abstractive", ""))
    source_words = word_count(source)

    if not source.strip():
        flags.append("input_kosong")
    if not extractive.strip():
        flags.append("extractive_kosong")
    if abstractive == "" and row.get("status_abstractive", "") not in {"disabled", "not_run"}:
        flags.append("abstractive_kosong")
    if extractive.strip() and extractive.strip() == source.strip():
        flags.append("extractive_sama_dengan_sumber")
    if abstractive.strip() and abstractive.strip() == source.strip():
        flags.append("abstractive_sama_dengan_sumber")
    if source_words > 0 and word_count(extractive) < 5:
        flags.append("extractive_terlalu_pendek")
    if source_words > 0 and word_count(extractive) > source_words:
        flags.append("extractive_terlalu_panjang")
    if abstractive.strip() and word_count(abstractive) < 5:
        flags.append("abstractive_terlalu_pendek")
    if source_words > 0 and word_count(abstractive) > source_words:
        flags.append("abstractive_terlalu_panjang")
    if _has_repetition(extractive):
        flags.append("repetisi_extractive")
    if _has_repetition(abstractive):
        flags.append("repetisi_abstractive")

    source_numbers = _numbers(source)
    for name, summary in [("extractive", extractive), ("abstractive", abstractive)]:
        summary_numbers = _numbers(summary)
        if summary_numbers - source_numbers:
            flags.append(f"angka_baru_{name}")
        if source_numbers - summary_numbers and summary.strip():
            flags.append(f"angka_sumber_hilang_{name}")
        if _possibly_not_indonesian(summary):
            flags.append(f"kemungkinan_bukan_indonesia_{name}")

    if row.get("status_extractive") == "single_sentence":
        flags.append("input_satu_kalimat")
    if row.get("compression_ratio_extractive") == 1:
        flags.append("extractive_tidak_kompresi")
    if bool(row.get("used_chunking", False)):
        flags.append("abstractive_menggunakan_chunking")
    if str(row.get("status_abstractive", "")) == "error":
        flags.append("inferensi_abstractive_gagal")

    unsourced_names = _unsourced_capitalized_terms(abstractive, source)
    if unsourced_names:
        flags.append(
            "abstractive_kemungkinan_hallucination_nama:" + ",".join(unsourced_names)
        )

    return flags


def create_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe containing flagged rows only."""
    result = df.copy()
    result["error_flags"] = result.apply(lambda row: ";".join(analyze_row(row)), axis=1)
    return result[result["error_flags"].astype(str).str.strip() != ""].copy()

