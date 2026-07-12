"""Dataset loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config


def read_dataset(path: Path) -> pd.DataFrame:
    """Read CSV or Excel dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Letakkan dataset di data/input/ "
            "atau ubah INPUT_DATA_PATH di src/config.py."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Format dataset tidak didukung: {suffix}. Gunakan CSV atau Excel.")


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate required columns and raise a clear error."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + f". Kolom tersedia: {list(df.columns)}. "
            "Sesuaikan nama kolom di src/config.py tanpa mengubah dataset sumber."
        )


def load_dataset(path: Path = config.INPUT_DATA_PATH) -> pd.DataFrame:
    """Load, validate, and prepare a working dataframe."""
    df = read_dataset(path)
    validate_columns(df, config.REQUIRED_COLUMNS)

    logging.info("Jumlah baris dataset: %s", len(df))
    null_counts = df[config.REQUIRED_COLUMNS].isna().sum().to_dict()
    logging.info("Nilai kosong pada kolom wajib: %s", null_counts)

    duplicate_count = df[config.HADITH_ID_COLUMN].duplicated().sum()
    if duplicate_count:
        logging.warning("Duplikasi nomor hadis ditemukan: %s baris", duplicate_count)

    work_df = df.copy()
    work_df = work_df.fillna("")
    before = len(work_df)
    work_df = work_df[work_df[config.INDONESIAN_TEXT_COLUMN].astype(str).str.strip() != ""].copy()
    removed = before - len(work_df)
    if removed:
        logging.warning("Baris tanpa terjemahan Indonesia dihapus dari dataframe kerja: %s", removed)

    return work_df.reset_index(drop=True)

