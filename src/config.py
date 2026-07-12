"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_DATA_PATH = INPUT_DIR / "dataset_hadis.csv"
OUTPUT_DATA_PATH = OUTPUT_DIR / "hasil_ringkasan_hadis.csv"
ERROR_ANALYSIS_PATH = OUTPUT_DIR / "error_analysis_samples.csv"
HUMAN_EVALUATION_TEMPLATE_PATH = OUTPUT_DIR / "human_evaluation_template.csv"

RANDOM_SEED = 42
BATCH_SIZE = 4
MAX_INPUT_LENGTH = 512
MAX_SUMMARY_LENGTH = 128
MIN_SUMMARY_LENGTH = 20

RUN_ABSTRACTIVE = True
RUN_EVALUATION = True
LIMIT_ROWS = 10

ABSTRACTIVE_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3

HADITH_ID_COLUMN = "Num_hadith"
ARABIC_TEXT_COLUMN = "teks_arab"
INDONESIAN_TEXT_COLUMN = "terjemahan"
REFERENCE_SUMMARY_COLUMN = "ringkasan_referensi"

REQUIRED_COLUMNS = [
    HADITH_ID_COLUMN,
    ARABIC_TEXT_COLUMN,
    INDONESIAN_TEXT_COLUMN,
]

