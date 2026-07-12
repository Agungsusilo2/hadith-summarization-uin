import pandas as pd

from generate_experiment_report import _build_metrics, _select_examples
from src import config


def test_build_metrics_without_reference():
    df = pd.DataFrame(
        {
            "teks_sumber_ringkasan": ["Rasulullah mengajarkan akhlak yang baik."],
            "ringkasan_extractive": ["Rasulullah mengajarkan akhlak yang baik."],
            "ringkasan_abstractive": [""],
            "status_extractive": ["single_sentence"],
            "status_abstractive": ["not_run_local_light"],
            "jumlah_kata_sumber": [5],
            "jumlah_kata_extractive": [5],
            "jumlah_kata_abstractive": [0],
            "compression_ratio_extractive": [1.0],
            "compression_ratio_abstractive": [0.0],
        }
    )

    metrics = _build_metrics(df)

    assert metrics["rows"] == 1
    assert metrics["has_reference_summaries"] is False
    assert metrics["reference_evaluation"]["mode"] == "descriptive"


def test_select_examples_prefers_long_sources():
    df = pd.DataFrame(
        {
            config.HADITH_ID_COLUMN: [1, 2],
            "teks_sumber_ringkasan": ["pendek.", "ini adalah teks sumber yang lebih panjang."],
            "ringkasan_extractive": ["pendek.", "teks sumber lebih panjang."],
            "ringkasan_abstractive": ["", ""],
        }
    )

    examples = _select_examples(df, limit=1)

    assert examples.iloc[0][config.HADITH_ID_COLUMN] == 2
