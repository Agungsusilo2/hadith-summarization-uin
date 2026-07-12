"""Explainable extractive summarization baseline."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import compression_ratio, word_count


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a conservative regex fallback."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    return sentences or [text.strip()]


class ExtractiveSummarizer:
    """TF-IDF sentence scoring summarizer."""

    def summarize(self, text: str, max_sentences: int = 2) -> dict:
        """Return an extractive summary and metadata."""
        source = text.strip() if isinstance(text, str) else ""
        source_words = word_count(source)
        if not source:
            return self._result("", "empty_input", source)

        sentences = split_sentences(source)
        if len(sentences) == 1:
            return self._result(sentences[0], "single_sentence", source)

        if source_words <= 30:
            return self._result(source, "too_short", source)

        selected_count = max(1, min(max_sentences, len(sentences)))
        try:
            vectorizer = TfidfVectorizer(lowercase=True)
            matrix = vectorizer.fit_transform(sentences)
            scores = np.asarray(matrix.sum(axis=1)).ravel()
            top_indices = sorted(np.argsort(scores)[-selected_count:].tolist())
            summary = " ".join(sentences[index] for index in top_indices)
            return self._result(summary, "ok", source)
        except ValueError:
            fallback = " ".join(sentences[:selected_count])
            return self._result(fallback, "fallback_tokenization", source)

    @staticmethod
    def _result(summary: str, status: str, source: str) -> dict:
        return {
            "summary": summary,
            "status": status,
            "source_word_count": word_count(source),
            "summary_word_count": word_count(summary),
            "compression_ratio": compression_ratio(summary, source),
        }

