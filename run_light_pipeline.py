"""Run the lightweight local hadith summarization pipeline."""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src import config
from src.colab_output_loader import DEFAULT_COLAB_OUTPUT_PATH, load_colab_output, merge_colab_output
from src.data_loader import load_dataset
from src.error_analysis import create_error_analysis
from src.evaluation import add_descriptive_columns
from src.extractive_summarizer import ExtractiveSummarizer
from src.preprocessing import clean_indonesian_text
from src.utils import ensure_directories, set_random_seed, setup_logging


def _create_human_evaluation_template(df: pd.DataFrame) -> None:
    template = pd.DataFrame(
        {
            "evaluator_id": "",
            config.HADITH_ID_COLUMN: df.get(config.HADITH_ID_COLUMN, ""),
            "teks_sumber": df.get("teks_sumber_ringkasan", ""),
            "ringkasan_extractive": df.get("ringkasan_extractive", ""),
            "ringkasan_abstractive": df.get("ringkasan_abstractive", ""),
            "factuality_1_5": "",
            "relevance_1_5": "",
            "completeness_1_5": "",
            "fluency_1_5": "",
            "tidak_mengubah_pesan_hadis_1_5": "",
            "preferred_model": "",
            "catatan": "",
        }
    )
    template.to_csv(config.HUMAN_EVALUATION_TEMPLATE_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    """Execute only local-safe processing and optionally merge Colab output."""
    setup_logging()
    set_random_seed(config.RANDOM_SEED)
    ensure_directories([config.INPUT_DIR, config.OUTPUT_DIR, config.MODELS_DIR])

    df = load_dataset(config.INPUT_DATA_PATH)
    if config.LIMIT_ROWS:
        logging.info("LIMIT_ROWS aktif: memproses %s baris pertama.", config.LIMIT_ROWS)
        df = df.head(config.LIMIT_ROWS).copy()

    df["teks_sumber_ringkasan"] = df[config.INDONESIAN_TEXT_COLUMN].map(clean_indonesian_text)

    extractive = ExtractiveSummarizer()
    extractive_results = []
    for text in tqdm(df["teks_sumber_ringkasan"], desc="Extractive summarization"):
        extractive_results.append(extractive.summarize(text))

    df["ringkasan_extractive"] = [item["summary"] for item in extractive_results]
    df["status_extractive"] = [item["status"] for item in extractive_results]
    df["jumlah_kata_sumber"] = [item["source_word_count"] for item in extractive_results]
    df["jumlah_kata_extractive"] = [item["summary_word_count"] for item in extractive_results]
    df["compression_ratio_extractive"] = [item["compression_ratio"] for item in extractive_results]

    df["ringkasan_abstractive"] = ""
    df["status_abstractive"] = "not_run_local_light"
    df["model_abstractive"] = ""
    df["jumlah_kata_abstractive"] = 0
    df["compression_ratio_abstractive"] = 0.0
    df["used_chunking"] = False
    df["error_abstractive"] = ""
    df["waktu_inferensi_abstractive_total_detik"] = 0.0

    if DEFAULT_COLAB_OUTPUT_PATH.exists():
        logging.info("Output Colab ditemukan; menggabungkan: %s", DEFAULT_COLAB_OUTPUT_PATH)
        colab_df = load_colab_output(DEFAULT_COLAB_OUTPUT_PATH)
        df = merge_colab_output(df, colab_df)
    else:
        logging.info("Output Colab belum ada; kolom abstractive dibiarkan kosong.")

    df = add_descriptive_columns(df, "ringkasan_extractive", "extractive")
    df = add_descriptive_columns(df, "ringkasan_abstractive", "abstractive")

    error_samples = create_error_analysis(df)
    error_samples.to_csv(config.ERROR_ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    _create_human_evaluation_template(df)

    df.to_csv(config.OUTPUT_DATA_PATH, index=False, encoding="utf-8-sig")
    logging.info("Output lokal ringan disimpan ke: %s", config.OUTPUT_DATA_PATH)
    logging.info("Error analysis disimpan ke: %s", config.ERROR_ANALYSIS_PATH)
    logging.info("Template evaluasi manusia disimpan ke: %s", config.HUMAN_EVALUATION_TEMPLATE_PATH)


if __name__ == "__main__":
    main()
