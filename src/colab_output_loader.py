"""Helpers for reading abstractive summarization output from Colab."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .data_loader import read_dataset, validate_columns


DEFAULT_COLAB_OUTPUT_PATH = config.OUTPUT_DIR / "hasil_ringkasan_hadis_colab.csv"
REQUIRED_COLAB_COLUMNS = [
    config.HADITH_ID_COLUMN,
    "ringkasan_abstractive",
]
OPTIONAL_COLAB_COLUMNS = [
    "status_abstractive",
    "model_abstractive",
    "jumlah_kata_abstractive",
    "compression_ratio_abstractive",
    "used_chunking",
    "error_abstractive",
]


def load_colab_output(path: Path = DEFAULT_COLAB_OUTPUT_PATH) -> pd.DataFrame:
    """Load and validate a Colab-generated summary CSV or Excel file."""
    df = read_dataset(path)
    validate_columns(df, REQUIRED_COLAB_COLUMNS)

    duplicate_count = df[config.HADITH_ID_COLUMN].duplicated().sum()
    if duplicate_count:
        logging.warning("Duplikasi ID pada output Colab ditemukan: %s baris", duplicate_count)

    keep_columns = [
        column
        for column in [*REQUIRED_COLAB_COLUMNS, *OPTIONAL_COLAB_COLUMNS]
        if column in df.columns
    ]
    return df[keep_columns].fillna("").copy()


def merge_colab_output(local_df: pd.DataFrame, colab_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Colab abstractive summaries into a local processed dataframe."""
    validate_columns(local_df, [config.HADITH_ID_COLUMN])
    validate_columns(colab_df, REQUIRED_COLAB_COLUMNS)

    suffix = "_colab"
    merged = local_df.merge(
        colab_df,
        on=config.HADITH_ID_COLUMN,
        how="left",
        suffixes=("", suffix),
    )

    for column in ["ringkasan_abstractive", *OPTIONAL_COLAB_COLUMNS]:
        colab_column = f"{column}{suffix}"
        if colab_column in merged.columns:
            colab_values = merged[colab_column].fillna("")
            merged[column] = colab_values.where(
                colab_values.astype(str).str.strip() != "",
                merged.get(column, ""),
            )
            merged = merged.drop(columns=[colab_column])
        elif column not in merged.columns:
            merged[column] = ""

    missing = merged["ringkasan_abstractive"].astype(str).str.strip().eq("").sum()
    if missing:
        logging.warning("Baris tanpa ringkasan abstractive dari Colab: %s", missing)

    return merged
