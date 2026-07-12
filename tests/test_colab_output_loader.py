import pandas as pd

from src import config
from src.colab_output_loader import load_colab_output, merge_colab_output


def test_load_colab_output_keeps_expected_columns(tmp_path):
    path = tmp_path / "hasil_ringkasan_hadis_colab.csv"
    pd.DataFrame(
        {
            config.HADITH_ID_COLUMN: [1],
            "ringkasan_abstractive": ["ringkasan"],
            "status_abstractive": ["ok"],
            "unused": ["ignored"],
        }
    ).to_csv(path, index=False)

    df = load_colab_output(path)

    assert list(df.columns) == [
        config.HADITH_ID_COLUMN,
        "ringkasan_abstractive",
        "status_abstractive",
    ]


def test_merge_colab_output_updates_abstractive_columns():
    local_df = pd.DataFrame(
        {
            config.HADITH_ID_COLUMN: [1, 2],
            "ringkasan_abstractive": ["", ""],
            "status_abstractive": ["not_run_local_light", "not_run_local_light"],
        }
    )
    colab_df = pd.DataFrame(
        {
            config.HADITH_ID_COLUMN: [1],
            "ringkasan_abstractive": ["hasil colab"],
            "status_abstractive": ["ok"],
        }
    )

    merged = merge_colab_output(local_df, colab_df)

    assert merged.loc[0, "ringkasan_abstractive"] == "hasil colab"
    assert merged.loc[0, "status_abstractive"] == "ok"
    assert merged.loc[1, "ringkasan_abstractive"] == ""
