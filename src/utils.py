"""Shared utility functions."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def setup_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create required directories without touching source data."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """Set deterministic seeds for available libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logging.debug("Torch is not installed; skipping torch seed.")


def word_count(text: object) -> int:
    """Count whitespace-separated words safely."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\S+", text.strip()))


def compression_ratio(summary: object, source: object) -> float:
    """Return summary/source word ratio."""
    source_words = word_count(source)
    if source_words == 0:
        return 0.0
    return round(word_count(summary) / source_words, 4)

