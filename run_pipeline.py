"""Run the hadith summarization pipeline."""

from __future__ import annotations

import json
import logging
import time

import pandas as pd
from tqdm import tqdm

from src import config
from src.abstractive_summarizer import AbstractiveSummarizer, GenerationConfig
from src.data_loader import load_dataset
from src.error_analysis import create_error_analysis
from src.evaluation import add_descriptive_columns, evaluate_with_references
from src.extractive_summarizer import ExtractiveSummarizer
from src.preprocessing import clean_indonesian_text
from src.utils import ensure_directories, set_random_seed, setup_logging

# ringkasan_extractive is the primary/default output: it only quotes sentences
# from the source, so it cannot invent facts. ringkasan_abstractive (mT5) is
# an experimental comparison only - it has been observed to hallucinate
# details (names, places, ages) absent from the source text. This disclaimer
# is written into the output CSV itself so it travels with the data even
# outside this codebase.
ABSTRACTIVE_DISCLAIMER = (
    "EKSPERIMEN PEMBANDING, BUKAN OUTPUT UTAMA. Ringkasan abstractive (mT5) "
    "dapat mengandung hallucination (nama/tempat/detail yang tidak ada di "
    "sumber). status_abstractive=generated_ok hanya berarti proses generate "
    "berhasil secara teknis, bukan validasi kebenaran isi. Gunakan "
    "ringkasan_extractive sebagai output utama, dan wajib validasi manusia "
    "sebelum ringkasan_abstractive dipakai untuk rujukan apa pun."
)


def _build_abstractive_summarizer() -> AbstractiveSummarizer:
    generation_config = GenerationConfig(
        max_input_length=config.MAX_INPUT_LENGTH,
        max_summary_length=config.MAX_SUMMARY_LENGTH,
        min_summary_length=config.MIN_SUMMARY_LENGTH,
        num_beams=config.NUM_BEAMS,
        no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
        batch_size=config.BATCH_SIZE,
    )
    return AbstractiveSummarizer(config.ABSTRACTIVE_MODEL_NAME, generation_config)


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
    """Execute the full pipeline."""
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

    if config.RUN_ABSTRACTIVE:
        start_time = time.perf_counter()
        try:
            abstractive = _build_abstractive_summarizer()
            abstractive_results = []
            texts = df["teks_sumber_ringkasan"].tolist()
            for start in tqdm(range(0, len(texts), config.BATCH_SIZE), desc="Abstractive summarization"):
                batch = texts[start : start + config.BATCH_SIZE]
                abstractive_results.extend(abstractive.summarize_batch(batch))
        except Exception as exc:
            logging.exception("Model abstractive gagal dimuat. Tidak ada fallback model.")
            abstractive_results = [
                {
                    "summary": "",
                    "status": "model_load_error",
                    "model_name": config.ABSTRACTIVE_MODEL_NAME,
                    "source_word_count": count,
                    "summary_word_count": 0,
                    "compression_ratio": 0.0,
                    "used_chunking": False,
                    "error": str(exc),
                }
                for count in df["jumlah_kata_sumber"].tolist()
            ]
        elapsed = round(time.perf_counter() - start_time, 4)

        df["ringkasan_abstractive"] = [item["summary"] for item in abstractive_results]
        df["status_abstractive"] = [item["status"] for item in abstractive_results]
        df["model_abstractive"] = [item["model_name"] for item in abstractive_results]
        df["jumlah_kata_abstractive"] = [item["summary_word_count"] for item in abstractive_results]
        df["compression_ratio_abstractive"] = [item["compression_ratio"] for item in abstractive_results]
        df["used_chunking"] = [item["used_chunking"] for item in abstractive_results]
        df["error_abstractive"] = [item["error"] for item in abstractive_results]
        df["waktu_inferensi_abstractive_total_detik"] = elapsed
        df["abstractive_disclaimer"] = ABSTRACTIVE_DISCLAIMER
    else:
        df["ringkasan_abstractive"] = ""
        df["status_abstractive"] = "disabled"
        df["model_abstractive"] = ""
        df["jumlah_kata_abstractive"] = 0
        df["compression_ratio_abstractive"] = 0.0
        df["used_chunking"] = False
        df["error_abstractive"] = ""
        df["waktu_inferensi_abstractive_total_detik"] = 0.0
        df["abstractive_disclaimer"] = ""

    df = add_descriptive_columns(df, "ringkasan_extractive", "extractive")
    df = add_descriptive_columns(df, "ringkasan_abstractive", "abstractive")

    if config.RUN_EVALUATION:
        evaluation_result = evaluate_with_references(df)
        logging.info("Evaluasi: %s", json.dumps(evaluation_result, ensure_ascii=False, default=str))

    error_samples = create_error_analysis(df)
    error_samples.to_csv(config.ERROR_ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    _create_human_evaluation_template(df)

    df.to_csv(config.OUTPUT_DATA_PATH, index=False, encoding="utf-8-sig")
    logging.info("Output disimpan ke: %s", config.OUTPUT_DATA_PATH)
    logging.info("Error analysis disimpan ke: %s", config.ERROR_ANALYSIS_PATH)
    logging.info("Template evaluasi manusia disimpan ke: %s", config.HUMAN_EVALUATION_TEMPLATE_PATH)
    if config.RUN_ABSTRACTIVE:
        logging.info(
            "Ingat: ringkasan_extractive adalah output utama. ringkasan_abstractive "
            "adalah eksperimen pembanding dan wajib divalidasi manusia (lihat kolom "
            "abstractive_disclaimer)."
        )


if __name__ == "__main__":
    main()

