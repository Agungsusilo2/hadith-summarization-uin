"""Generate a UAS experiment report from actual pipeline outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.data_loader import read_dataset
from src.evaluation import descriptive_statistics, evaluate_with_references, has_reference_summaries
from src.utils import ensure_directories, setup_logging, word_count


REPORT_PATH = config.OUTPUT_DIR / "laporan_eksperimen_uas.md"
EXAMPLE_OUTPUT_PATH = config.OUTPUT_DIR / "contoh_output_ringkasan.md"
METRICS_PATH = config.OUTPUT_DIR / "metrics_eksperimen.json"


def _require_pipeline_output() -> pd.DataFrame:
    if not config.OUTPUT_DATA_PATH.exists():
        raise FileNotFoundError(
            "Output pipeline belum ditemukan. Jalankan `python run_light_pipeline.py` "
            "atau `python run_pipeline.py` terlebih dahulu."
        )
    return read_dataset(config.OUTPUT_DATA_PATH)


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return round(float(numeric.mean()), 4)


def _status_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}


def _select_examples(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    required = ["teks_sumber_ringkasan", "ringkasan_extractive", "ringkasan_abstractive"]
    for column in required:
        if column not in df.columns:
            df[column] = ""

    candidates = df.copy()
    candidates["source_words_for_example"] = candidates["teks_sumber_ringkasan"].map(word_count)
    candidates = candidates[candidates["teks_sumber_ringkasan"].astype(str).str.strip() != ""]
    candidates = candidates.sort_values("source_words_for_example", ascending=False)
    return candidates.head(limit).copy()


def _write_examples(df: pd.DataFrame) -> None:
    examples = _select_examples(df)
    lines = [
        "# Contoh Output Ringkasan",
        "",
        "File ini dibuat otomatis dari output pipeline. Contoh berikut perlu dicek manusia, terutama untuk kesesuaian makna hadis.",
        "",
    ]

    if examples.empty:
        lines.append("Belum ada contoh karena output pipeline kosong.")
    else:
        for number, (_, row) in enumerate(examples.iterrows(), start=1):
            hadith_id = row.get(config.HADITH_ID_COLUMN, "")
            lines.extend(
                [
                    f"## Contoh {number}",
                    "",
                    f"- Nomor hadis: `{hadith_id}`",
                    f"- Status extractive: `{row.get('status_extractive', '')}`",
                    f"- Status abstractive: `{row.get('status_abstractive', '')}`",
                    "",
                    "Teks sumber:",
                    "",
                    str(row.get("teks_sumber_ringkasan", "")),
                    "",
                    "Ringkasan extractive:",
                    "",
                    str(row.get("ringkasan_extractive", "")),
                    "",
                    "Ringkasan abstractive:",
                    "",
                    str(row.get("ringkasan_abstractive", "")),
                    "",
                ]
            )

    EXAMPLE_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _build_metrics(df: pd.DataFrame) -> dict:
    metrics = {
        "rows": int(len(df)),
        "has_reference_summaries": bool(has_reference_summaries(df)),
        "descriptive_statistics": descriptive_statistics(df),
        "status_counts": {
            "extractive": _status_counts(df, "status_extractive"),
            "abstractive": _status_counts(df, "status_abstractive"),
        },
        "averages": {
            "source_words": _safe_mean(df.get("jumlah_kata_sumber", pd.Series(dtype=float))),
            "extractive_words": _safe_mean(df.get("jumlah_kata_extractive", pd.Series(dtype=float))),
            "abstractive_words": _safe_mean(df.get("jumlah_kata_abstractive", pd.Series(dtype=float))),
            "extractive_compression_ratio": _safe_mean(
                df.get("compression_ratio_extractive", pd.Series(dtype=float))
            ),
            "abstractive_compression_ratio": _safe_mean(
                df.get("compression_ratio_abstractive", pd.Series(dtype=float))
            ),
        },
    }

    metrics["reference_evaluation"] = evaluate_with_references(df)
    return metrics


def _write_report(metrics: dict) -> None:
    evaluation_mode = metrics["reference_evaluation"].get("mode", "unknown")
    lines = [
        "# Laporan Eksperimen UAS",
        "",
        "Laporan ini dibuat otomatis dari output pipeline. Angka di bawah hanya valid jika pipeline sudah dijalankan pada dataset asli.",
        "",
        "## Konfigurasi",
        "",
        f"- Input pipeline: `{config.INPUT_DATA_PATH}`",
        f"- Output pipeline: `{config.OUTPUT_DATA_PATH}`",
        f"- Model abstractive: `{config.ABSTRACTIVE_MODEL_NAME}`",
        f"- Mode sistem: inference, bukan fine-tuning",
        f"- Jumlah baris output: `{metrics['rows']}`",
        f"- Memiliki ringkasan referensi: `{metrics['has_reference_summaries']}`",
        "",
        "## Status Pipeline",
        "",
        "Status extractive:",
        "",
        "```json",
        json.dumps(metrics["status_counts"]["extractive"], ensure_ascii=False, indent=2),
        "```",
        "",
        "Status abstractive:",
        "",
        "```json",
        json.dumps(metrics["status_counts"]["abstractive"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Statistik Rata-rata",
        "",
        "```json",
        json.dumps(metrics["averages"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evaluasi Otomatis",
        "",
        f"Mode evaluasi: `{evaluation_mode}`",
        "",
    ]

    if evaluation_mode == "reference":
        lines.extend(
            [
                "ROUGE dan BERTScore dihitung karena kolom ringkasan referensi tersedia.",
                "",
                "```json",
                json.dumps(metrics["reference_evaluation"], ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "Ringkasan referensi manusia belum tersedia, sehingga ROUGE dan BERTScore tidak boleh dianggap sebagai hasil eksperimen utama.",
                "Bagian ini hanya menampilkan statistik deskriptif.",
                "",
                "```json",
                json.dumps(metrics["descriptive_statistics"], ensure_ascii=False, indent=2, default=str),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Kesimpulan Template",
            "",
            "Isi bagian ini setelah hasil diperiksa:",
            "",
            "- Apakah extractive terlalu sering sama dengan teks sumber?",
            "- Apakah abstractive menghasilkan ringkasan yang lebih pendek?",
            "- Apakah ada ringkasan kosong atau error model?",
            "- Apakah angka, nama tokoh, dan pesan utama hadis tetap terjaga?",
            "- Apakah hasil perlu diperiksa ulang oleh evaluator manusia?",
            "",
            "Catatan: sistem ini tidak memberi fatwa, syarah, atau penilaian agama. Ringkasan perlu divalidasi manusia.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Create report files from real output data."""
    setup_logging()
    ensure_directories([config.OUTPUT_DIR])

    df = _require_pipeline_output()
    metrics = _build_metrics(df)

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_examples(df)
    _write_report(metrics)

    logging.info("Metrics disimpan ke: %s", METRICS_PATH)
    logging.info("Contoh output disimpan ke: %s", EXAMPLE_OUTPUT_PATH)
    logging.info("Laporan eksperimen disimpan ke: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
