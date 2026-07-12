"""Lightweight preprocessing for Indonesian hadith translations."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_indonesian_text(text: str) -> str:
    """Clean text without removing religious terms, names, numbers, or negation."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

