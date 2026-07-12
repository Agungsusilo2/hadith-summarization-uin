"""Automatic evaluation and descriptive statistics."""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from . import config
from .utils import compression_ratio, word_count


def has_reference_summaries(df: pd.DataFrame) -> bool:
    """Return True only when reference summaries exist and are non-empty."""
    column = config.REFERENCE_SUMMARY_COLUMN
    return column in df.columns and df[column].astype(str).str.strip().ne("").any()


def repetition_rate(text: str) -> float:
    """Measure repeated token ratio."""
    words = str(text).lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return round(repeated / len(words), 4)


def novel_ngram_ratio(summary: str, source: str, n: int) -> float:
    """Return ratio of summary ngrams absent from the source."""
    source_tokens = str(source).lower().split()
    summary_tokens = str(summary).lower().split()
    if len(summary_tokens) < n:
        return 0.0
    source_ngrams = {tuple(source_tokens[i : i + n]) for i in range(max(0, len(source_tokens) - n + 1))}
    summary_ngrams = [tuple(summary_tokens[i : i + n]) for i in range(len(summary_tokens) - n + 1)]
    if not summary_ngrams:
        return 0.0
    novel = sum(1 for ngram in summary_ngrams if ngram not in source_ngrams)
    return round(novel / len(summary_ngrams), 4)


def descriptive_statistics(df: pd.DataFrame) -> dict:
    """Compute descriptive summary statistics without treating them as ground truth."""
    stats = {
        "rows": len(df),
        "avg_source_words": round(df["teks_sumber_ringkasan"].map(word_count).mean(), 4) if len(df) else 0,
        "empty_extractive": int(df["ringkasan_extractive"].astype(str).str.strip().eq("").sum()),
        "identical_extractive": int((df["ringkasan_extractive"] == df["teks_sumber_ringkasan"]).sum()),
    }
    if "ringkasan_abstractive" in df.columns:
        stats.update(
            {
                "empty_abstractive": int(df["ringkasan_abstractive"].astype(str).str.strip().eq("").sum()),
                "identical_abstractive": int((df["ringkasan_abstractive"] == df["teks_sumber_ringkasan"]).sum()),
            }
        )
    return stats


def evaluate_with_references(df: pd.DataFrame) -> dict:
    """Compute ROUGE and BERTScore when human references are available."""
    if not has_reference_summaries(df):
        logging.info("Ringkasan referensi tidak tersedia; evaluasi otomatis dilewati.")
        return {"mode": "descriptive", "statistics": descriptive_statistics(df)}

    references = df[config.REFERENCE_SUMMARY_COLUMN].astype(str).tolist()
    results: dict[str, object] = {"mode": "reference"}

    try:
        import evaluate

        rouge = evaluate.load("rouge")
        for column in ["ringkasan_extractive", "ringkasan_abstractive"]:
            if column in df.columns:
                predictions = df[column].astype(str).tolist()
                rouge_result = rouge.compute(predictions=predictions, references=references)
                results[f"{column}_rouge"] = rouge_result
    except Exception as exc:
        logging.warning("ROUGE tidak dapat dihitung: %s", exc)
        results["rouge_error"] = str(exc)

    try:
        from bert_score import score

        for column in ["ringkasan_extractive", "ringkasan_abstractive"]:
            if column in df.columns:
                predictions = df[column].astype(str).tolist()
                precision, recall, f1 = score(predictions, references, lang="id", verbose=False)
                results[f"{column}_bertscore"] = {
                    "precision": float(precision.mean()),
                    "recall": float(recall.mean()),
                    "f1": float(f1.mean()),
                }
    except Exception as exc:
        logging.warning("BERTScore tidak dapat dihitung: %s", exc)
        results["bertscore_error"] = str(exc)

    results["statistics"] = descriptive_statistics(df)
    return results


def add_descriptive_columns(df: pd.DataFrame, summary_column: str, prefix: str) -> pd.DataFrame:
    """Add non-reference descriptive metrics for a summary column."""
    df[f"{prefix}_repetition_rate"] = df[summary_column].map(repetition_rate)
    df[f"{prefix}_novel_unigram_ratio"] = df.apply(
        lambda row: novel_ngram_ratio(row[summary_column], row["teks_sumber_ringkasan"], 1), axis=1
    )
    df[f"{prefix}_novel_bigram_ratio"] = df.apply(
        lambda row: novel_ngram_ratio(row[summary_column], row["teks_sumber_ringkasan"], 2), axis=1
    )
    df[f"{prefix}_compression_ratio_check"] = df.apply(
        lambda row: compression_ratio(row[summary_column], row["teks_sumber_ringkasan"]), axis=1
    )
    return df

