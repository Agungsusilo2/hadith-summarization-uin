from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.abstractive_summarizer import AbstractiveSummarizer, GenerationConfig
from src.data_loader import load_dataset
from src.extractive_summarizer import ExtractiveSummarizer


def test_extractive_single_sentence():
    summarizer = ExtractiveSummarizer()
    result = summarizer.summarize("Rasulullah mengajarkan akhlak yang baik.")
    assert result["summary"] == "Rasulullah mengajarkan akhlak yang baik."
    assert result["status"] == "single_sentence"


def test_extractive_multiple_sentences():
    summarizer = ExtractiveSummarizer()
    text = (
        "Rasulullah mengajarkan akhlak yang baik. "
        "Hadis ini menjelaskan pentingnya niat dalam amal. "
        "Mahasiswa dapat mempelajari pesan utama dari terjemahan hadis tersebut."
    )
    result = summarizer.summarize(text, max_sentences=2)
    assert result["summary"]
    assert result["summary_word_count"] <= result["source_word_count"]


def test_extractive_empty_input():
    result = ExtractiveSummarizer().summarize("")
    assert result["status"] == "empty_input"
    assert result["compression_ratio"] == 0.0


def test_dataset_missing_required_column(tmp_path: Path):
    dataset = tmp_path / "bad.csv"
    pd.DataFrame({"Num_hadith": [1], "terjemahan": ["teks"]}).to_csv(dataset, index=False)
    with pytest.raises(ValueError):
        load_dataset(dataset)


def test_input_file_not_overwritten(tmp_path: Path):
    dataset = tmp_path / "dataset.csv"
    original = "Num_hadith,teks_arab,terjemahan\n1,arab,teks indonesia\n"
    dataset.write_text(original, encoding="utf-8")
    load_dataset(dataset)
    assert dataset.read_text(encoding="utf-8") == original


def test_abstractive_model_load_failure(monkeypatch):
    def fail_load(self):
        raise RuntimeError("load failed")

    monkeypatch.setattr(AbstractiveSummarizer, "_load_model", fail_load)
    generation_config = GenerationConfig(
        max_input_length=config.MAX_INPUT_LENGTH,
        max_summary_length=config.MAX_SUMMARY_LENGTH,
        min_summary_length=config.MIN_SUMMARY_LENGTH,
        num_beams=config.NUM_BEAMS,
        no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
        batch_size=1,
    )
    with pytest.raises(RuntimeError):
        AbstractiveSummarizer("dummy-model", generation_config)

