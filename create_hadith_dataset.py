"""Dataset builder untuk Sahih al-Bukhari dari sunnah.com.

Script ini BUKAN bagian dari pipeline summarization utama. Tujuannya hanya
membangun dataset mentah (data/output/dataset_hadis_minimal.csv) yang nanti
bisa disalin manual oleh pengguna menjadi data/input/dataset_hadis.csv untuk
dipakai oleh run_pipeline.py / run_light_pipeline.py.

Jalankan:
    python create_hadith_dataset.py

Default aman: hanya mengambil sekitar 10 hadis pertama dari 1 book (lihat
bagian KONFIGURASI di bawah). Untuk scrape penuh, ubah RUN_FULL_SCRAPE=True.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================
# KONFIGURASI AMAN (ubah manual sesuai kebutuhan)
# ============================================================

RUN_FULL_SCRAPE = False          # True = scrape seluruh Sahih al-Bukhari (97 book)
LIMIT_BOOKS = 1                  # jumlah book yang diambil jika RUN_FULL_SCRAPE=False
LIMIT_HADITHS = 10               # jumlah hadis maksimum jika RUN_FULL_SCRAPE=False
RUN_TRANSLATION = True           # True = terjemahkan english_translation -> Indonesia
SAVE_EVERY = 10                  # autosave checkpoint terjemahan setiap N baris
REQUEST_DELAY_SECONDS = 2        # jeda antar request HTTP
TIMEOUT_SECONDS = 20             # timeout request HTTP
MAX_RETRIES = 3                  # jumlah percobaan ulang request yang gagal
COPY_MINIMAL_TO_INPUT = False    # True = otomatis salin dataset minimal ke data/input/

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-id"
TRANSLATION_BATCH_SIZE = 8
# opus-mt-en-id is a MarianMT model trained on sentence-level pairs. Feeding
# it a full multi-sentence hadith paragraph in one shot degrades badly, so
# long text is split into chunks around this token budget before translation
# instead of being silently truncated.
TRANSLATION_CHUNK_MAX_TOKENS = 60
TRANSLATION_GENERATE_MAX_LENGTH = 128
TRANSLATION_RATIO_WARNING_THRESHOLD = 0.35

BASE_URL = "https://sunnah.com"
COLLECTION_SLUG = "bukhari"
COLLECTION_INDEX_URL = f"{BASE_URL}/{COLLECTION_SLUG}"
BOOK_URL_TEMPLATE = f"{BASE_URL}/{COLLECTION_SLUG}/{{book_number}}"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
    "HadithDatasetBuilder/1.0 (academic project, non-commercial)"
)

MIN_ARABIC_CHARS = 10
MIN_ENGLISH_CHARS = 10
ARABIC_CHAR_PATTERN = re.compile(r"[؀-ۿ]")
REFERENCE_HREF_PATTERN = re.compile(rf"^/{COLLECTION_SLUG}:(\d+)$")

# ============================================================
# PATH (relatif terhadap root project, aman untuk laptop/PyCharm)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_HTML_DIR = PROJECT_ROOT / "data" / "cache" / "html"
CACHE_JSON_DIR = PROJECT_ROOT / "data" / "cache" / "json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
INPUT_DIR = PROJECT_ROOT / "data" / "input"

OUTPUT_FULL_PATH = OUTPUT_DIR / "dataset_hadis_full.csv"
OUTPUT_MINIMAL_PATH = OUTPUT_DIR / "dataset_hadis_minimal.csv"
OUTPUT_REF_TEMPLATE_PATH = OUTPUT_DIR / "ringkasan_referensi_template.csv"
OUTPUT_REVIEW_PATH = OUTPUT_DIR / "dataset_hadis_review.csv"
OUTPUT_PROGRESS_PATH = OUTPUT_DIR / "scrape_progress.csv"
TRANSLATION_CHECKPOINT_PATH = OUTPUT_DIR / "dataset_hadis_full.partial.csv"
INPUT_MINIMAL_COPY_PATH = INPUT_DIR / "dataset_hadis.csv"

FULL_OUTPUT_COLUMNS = [
    "Num_hadith",
    "source_ref",
    "in_book_reference",
    "book_number",
    "book_title",
    "hadith_number",
    "teks_arab",
    "english_translation",
    "terjemahan",
    "ringkasan_referensi",
    "data_status",
    "data_warning",
    "translation_status",
    "translation_warning",
    "translation_error",
    "translation_model",
]

MINIMAL_OUTPUT_COLUMNS = ["Num_hadith", "teks_arab", "terjemahan", "ringkasan_referensi"]

# dataset_hadis_minimal.csv only includes rows whose machine translation
# passed the automatic checks (see translate_english_to_indonesian). Rows
# flagged needs_review/failed still exist in dataset_hadis_full.csv and
# dataset_hadis_review.csv, they are just excluded from what feeds the
# summarization pipeline by default.
MINIMAL_ALLOWED_TRANSLATION_STATUSES = ["ok", "chunked_ok"]
REVIEW_TRANSLATION_STATUSES = ["needs_review", "failed"]

REF_TEMPLATE_COLUMNS = [
    "Num_hadith",
    "source_ref",
    "teks_arab",
    "terjemahan",
    "ringkasan_referensi",
    "validator",
    "catatan_validasi",
]


def setup_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_directories() -> None:
    """Create dataset-builder-only directories without touching pipeline folders."""
    for path in (CACHE_HTML_DIR, CACHE_JSON_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _slugify_cache_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def fetch_url_with_cache(
    session: requests.Session, url: str, cache_name: str
) -> Optional[str]:
    """Fetch a URL as text, using an on-disk HTML cache.

    Returns None (without fabricating content) if the page does not exist
    (HTTP 404) or if all retry attempts fail.
    """
    cache_path = CACHE_HTML_DIR / f"{_slugify_cache_name(cache_name)}.html"
    if cache_path.exists():
        logging.info("Cache hit: %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = session.get(url, timeout=TIMEOUT_SECONDS)
            if response.status_code == 404:
                logging.warning("Halaman tidak ditemukan (404): %s", url)
                return None
            response.raise_for_status()
            html = response.text
            cache_path.write_text(html, encoding="utf-8")
            return html
        except requests.RequestException as exc:
            last_error = exc
            logging.warning(
                "Gagal fetch %s (percobaan %s/%s): %s", url, attempt, MAX_RETRIES, exc
            )
            time.sleep(REQUEST_DELAY_SECONDS * attempt)

    logging.error("Gagal fetch %s setelah %s percobaan: %s", url, MAX_RETRIES, last_error)
    return None


def load_book_records_cache(cache_name: str) -> Optional[list[dict]]:
    """Load previously parsed hadith records for a book, if cached."""
    cache_path = CACHE_JSON_DIR / f"{_slugify_cache_name(cache_name)}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Cache JSON rusak untuk %s, akan diparse ulang: %s", cache_name, exc)
        return None


def save_book_records_cache(cache_name: str, records: list[dict]) -> None:
    """Persist parsed hadith records for a book so re-parsing can be skipped."""
    cache_path = CACHE_JSON_DIR / f"{_slugify_cache_name(cache_name)}.json"
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def parse_bukhari_book_list(html: str) -> list[dict]:
    """Parse the sunnah.com/bukhari index page into a list of book entries."""
    soup = BeautifulSoup(html, "html.parser")
    books: list[dict] = []

    for book_div in soup.find_all("div", class_="book_title"):
        number_tag = book_div.find("div", class_="book_number")
        title_tag = book_div.find("div", class_="english_book_name")
        link_tag = book_div.find("a", href=re.compile(rf"^/{COLLECTION_SLUG}/\d+$"))

        if number_tag is None or link_tag is None:
            logging.warning("Melewati entri book yang tidak lengkap di halaman indeks.")
            continue

        try:
            book_number = int(number_tag.get_text(strip=True))
        except ValueError:
            logging.warning("Nomor book tidak valid: %r", number_tag.get_text(strip=True))
            continue

        book_title = title_tag.get_text(strip=True) if title_tag else ""
        book_url = BASE_URL + link_tag["href"]

        books.append(
            {
                "book_number": book_number,
                "book_title": book_title,
                "book_url": book_url,
            }
        )

    books.sort(key=lambda item: item["book_number"])
    return books


def _clean_reference_cell(text: str) -> str:
    return re.sub(r"^[\s:\xa0]+", "", text).strip()


def _extract_table_field(table, label: str) -> str:
    if table is None:
        return ""
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2 and label.lower() in cells[0].get_text(strip=True).lower():
            return _clean_reference_cell(cells[1].get_text(" ", strip=True))
    return ""


def _extract_hadith_number(reference_table, sticky_text: str) -> Optional[int]:
    if reference_table is not None:
        link = reference_table.find("a", href=REFERENCE_HREF_PATTERN)
        if link is not None:
            match = REFERENCE_HREF_PATTERN.match(link["href"])
            if match:
                return int(match.group(1))

    trailing_number = re.search(r"(\d+)\s*$", sticky_text or "")
    if trailing_number:
        return int(trailing_number.group(1))
    return None


def parse_bukhari_book_page(html: str, book_number: int, book_title: str) -> list[dict]:
    """Parse a single sunnah.com/bukhari/{book_number} page into hadith records.

    Every hadith container is parsed independently: a failure on one hadith
    does not stop parsing of the rest of the page.
    """
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.find_all("div", class_="actualHadithContainer")

    records: list[dict] = []
    for container in containers:
        record = {
            "source_ref": "",
            "in_book_reference": "",
            "book_number": book_number,
            "book_title": book_title,
            "hadith_number": None,
            "teks_arab": "",
            "english_translation": "",
            "data_status": "valid",
            "data_warning": "",
        }
        try:
            sticky = container.find("div", class_="hadith_reference_sticky")
            sticky_text = sticky.get_text(strip=True) if sticky else ""

            reference_table = container.find("table", class_="hadith_reference")
            source_ref = _extract_table_field(reference_table, "Reference")
            if not source_ref:
                source_ref = sticky_text
            if not source_ref:
                hadith_number_guess = _extract_hadith_number(reference_table, sticky_text)
                if hadith_number_guess is not None:
                    source_ref = f"Sahih al-Bukhari {hadith_number_guess}"

            in_book_reference = _extract_table_field(reference_table, "In-book reference")

            english_full = container.find("div", class_="english_hadith_full")
            english_parts = []
            if english_full is not None:
                narrated = english_full.find("div", class_="hadith_narrated")
                details = english_full.find("div", class_="text_details")
                if narrated is not None:
                    english_parts.append(narrated.get_text(" ", strip=True))
                if details is not None:
                    english_parts.append(details.get_text(" ", strip=True))
            english_translation = " ".join(part for part in english_parts if part).strip()

            arabic_full = container.find("div", class_="arabic_hadith_full")
            teks_arab = arabic_full.get_text(" ", strip=True) if arabic_full else ""

            hadith_number = _extract_hadith_number(reference_table, sticky_text)

            record.update(
                {
                    "source_ref": source_ref,
                    "in_book_reference": in_book_reference,
                    "hadith_number": hadith_number,
                    "teks_arab": teks_arab,
                    "english_translation": english_translation,
                }
            )

            warnings = []
            if hadith_number is None:
                warnings.append("hadith_number tidak ditemukan")
            if not teks_arab:
                record["data_status"] = "missing_arabic"
                warnings.append("teks_arab tidak ditemukan")
            elif not english_translation:
                record["data_status"] = "missing_english"
                warnings.append("english_translation tidak ditemukan")

            record["data_warning"] = "; ".join(warnings)

        except Exception as exc:  # noqa: BLE001 - keep scraping other hadith
            record["data_status"] = "parse_error"
            record["data_warning"] = f"parse_error: {exc}"
            logging.warning("Gagal parsing satu hadis di book %s: %s", book_number, exc)

        records.append(record)

    return records


def validate_scraped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add/refine data_status and data_warning without deleting any raw data."""
    if df.empty:
        return df

    df = df.copy()
    df["data_status"] = df["data_status"].fillna("valid")
    df["data_warning"] = df["data_warning"].fillna("")

    def _append_warning(index: int, message: str) -> None:
        existing = df.at[index, "data_warning"]
        df.at[index, "data_warning"] = f"{existing}; {message}" if existing else message

    for index, row in df.iterrows():
        teks_arab = str(row.get("teks_arab", "") or "")
        english_translation = str(row.get("english_translation", "") or "")

        if teks_arab and not ARABIC_CHAR_PATTERN.search(teks_arab):
            _append_warning(index, "teks_arab tidak mengandung karakter Arab")
            if df.at[index, "data_status"] == "valid":
                df.at[index, "data_status"] = "missing_arabic"

        if row["data_status"] == "valid" and len(teks_arab) < MIN_ARABIC_CHARS:
            df.at[index, "data_status"] = "too_short"
            _append_warning(index, "teks_arab terlalu pendek")

        if row["data_status"] == "valid" and len(english_translation) < MIN_ENGLISH_CHARS:
            df.at[index, "data_status"] = "too_short"
            _append_warning(index, "english_translation terlalu pendek")

    duplicate_ref_mask = df["source_ref"].astype(str).duplicated(keep="first")
    for index in df.index[duplicate_ref_mask]:
        if df.at[index, "source_ref"]:
            df.at[index, "data_status"] = "duplicate_ref"
            _append_warning(index, "duplicate source_ref")

    duplicate_number_mask = df.duplicated(subset=["book_number", "hadith_number"], keep="first")
    for index in df.index[duplicate_number_mask]:
        if pd.notna(df.at[index, "hadith_number"]):
            _append_warning(index, "duplicate hadith_number dalam book yang sama")

    return df


def _load_translation_checkpoint() -> Optional[pd.DataFrame]:
    if not TRANSLATION_CHECKPOINT_PATH.exists():
        return None
    try:
        return pd.read_csv(TRANSLATION_CHECKPOINT_PATH, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError) as exc:
        logging.warning("Gagal membaca checkpoint terjemahan, diabaikan: %s", exc)
        return None


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter for English hadith text (no extra NLP dependency)."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"‘“])", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_oversized_sentence(sentence: str, tokenizer, max_tokens: int) -> list[str]:
    """Fallback split (by clause) for a single sentence that alone exceeds max_tokens."""
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    if len(parts) <= 1:
        encoded = tokenizer(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens)
        return [tokenizer.decode(encoded["input_ids"])]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [sentence]


def _chunk_text_for_translation(text: str, tokenizer, max_tokens: int) -> tuple[list[str], bool]:
    """Split text into chunks that fit max_tokens, grouping whole sentences together.

    Returns (chunks, had_oversized_sentence). had_oversized_sentence is True only
    when a single sentence alone exceeded max_tokens and had to be split further
    by clause (flagged for review instead of being silently truncated).
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return [], False

    chunks: list[str] = []
    current = ""
    had_oversized_sentence = False

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) <= max_tokens:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(tokenizer(sentence, add_special_tokens=False)["input_ids"]) <= max_tokens:
            current = sentence
        else:
            had_oversized_sentence = True
            chunks.extend(_split_oversized_sentence(sentence, tokenizer, max_tokens))

    if current:
        chunks.append(current)

    return chunks, had_oversized_sentence


def _looks_repetitive(text: str) -> bool:
    """Heuristic flag for degenerate, highly repetitive NMT output."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return False
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values()) / len(words) > 0.4


# opus-mt-en-id was trained partly on religious parallel corpora (e.g. Quran
# translations). On fragmented/out-of-context input it sometimes leaks names
# or classical-Arabic-grammar (nahwu/tafsir) jargon from THAT training data
# instead of translating the actual hadith sentence. These are known-observed
# leaks from this project's own test runs, not translations of anything in
# Sahih al-Bukhari book 1.
SUSPICIOUS_LEAKED_TERMS = [
    "zakaria", "zakariya",
    "yusuf", "yūsuf",
    "musa", "ibrahim",
    "balqis", "sulaiman", "sulayman",
    "al-aziz", "al-'aziz", "aziz",
    "mubtada", "zharaf", "athaf bayan", "khabar", "ta'lil",
    "maf'ul", "mashdar", "qiraat", "tahqiq", "tas-hil", "dhamir", "isim", "fi'il",
]

# Capitalized words that legitimately recur across almost any hadith
# translation and should never by themselves count as an "unsourced name".
ALLOWED_CAPITALIZED_TERMS = {
    "allah", "rasul", "rasulullah", "messenger", "nabi", "muhammad",
    "islam", "muslim", "quran", "qur'an", "alquran", "al-quran",
    "jibril", "gabriel", "hira",
}


def _find_leaked_blocklist_terms(translated_text: str, english_source: str, arabic_source: str) -> list[str]:
    """Return blocklisted terms present in the translation but absent from BOTH sources.

    A term is only ever a "leak" if it has no basis in either english_translation
    or teks_arab. Matching against teks_arab is a weak check (Arabic script vs.
    Latin term) but is still performed in case a Latinized name appears there.
    """
    translated_lower = translated_text.lower()
    source_lower = f"{english_source} {arabic_source}".lower()
    return [
        term
        for term in SUSPICIOUS_LEAKED_TERMS
        if term in translated_lower and term not in source_lower
    ]


def _find_unsourced_midsentence_names(translated_text: str, english_source: str, arabic_source: str) -> list[str]:
    """Flag capitalized, mid-sentence tokens in the translation not found in either source.

    Only mid-sentence capitalization is checked (a token preceded by a lowercase
    letter) because Indonesian capitalizes every sentence-initial word regardless
    of whether it is a proper noun, which would otherwise make this check useless.
    A candidate is only flagged if absent from BOTH english_translation and
    teks_arab (source-aware: a name genuinely present in the source is not a leak).
    """
    candidates = re.findall(r"(?<=[a-z]\s)([A-Z][a-zA-Z'-]{2,})", translated_text)
    source_lower = f"{english_source} {arabic_source}".lower()
    found: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen or key in ALLOWED_CAPITALIZED_TERMS:
            continue
        seen.add(key)
        if key not in source_lower:
            found.append(candidate)
    return found


def _translate_chunks(
    chunks: list[str], tokenizer, model, device: str
) -> list[str]:
    """Translate chunks one at a time (chunks are already short/sentence-scale)."""
    import torch

    translations: list[str] = []
    for chunk in chunks:
        inputs = tokenizer(
            [chunk],
            return_tensors="pt",
            truncation=True,
            max_length=TRANSLATION_CHUNK_MAX_TOKENS + 8,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=TRANSLATION_GENERATE_MAX_LENGTH)
        translations.append(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())
    return translations


def translate_english_to_indonesian(df: pd.DataFrame) -> pd.DataFrame:
    """Translate english_translation to Indonesian using Helsinki-NLP/opus-mt-en-id.

    Long text is split into sentence-level chunks before translation instead of
    being silently truncated, because opus-mt-en-id is a sentence-level MarianMT
    model that degrades badly on long multi-sentence input. Supports resuming:
    rows that already have a non-empty `terjemahan` (loaded from a checkpoint
    file or already present in df) are skipped and marked skipped_existing.
    """
    df = df.copy()
    for column in ("terjemahan", "translation_status", "translation_warning", "translation_error", "translation_model"):
        if column not in df.columns:
            df[column] = ""
    df[["terjemahan", "translation_status", "translation_warning", "translation_error", "translation_model"]] = df[
        ["terjemahan", "translation_status", "translation_warning", "translation_error", "translation_model"]
    ].fillna("")

    checkpoint = _load_translation_checkpoint()
    if checkpoint is not None and "source_ref" in checkpoint.columns:
        checkpoint = checkpoint.set_index("source_ref")
        for index, row in df.iterrows():
            ref = row["source_ref"]
            if ref in checkpoint.index and str(checkpoint.loc[ref, "terjemahan"]).strip():
                df.at[index, "terjemahan"] = checkpoint.loc[ref, "terjemahan"]
                df.at[index, "translation_status"] = "skipped_existing"
                df.at[index, "translation_model"] = checkpoint.loc[ref, "translation_model"]
                if "translation_warning" in checkpoint.columns:
                    df.at[index, "translation_warning"] = checkpoint.loc[ref, "translation_warning"]

    if not RUN_TRANSLATION:
        pending = df["translation_status"] == ""
        df.loc[pending, "translation_status"] = "disabled"
        return df

    pending_mask = (df["terjemahan"].astype(str).str.strip() == "") & (
        df["translation_status"] != "skipped_existing"
    )
    empty_source_mask = pending_mask & (df["english_translation"].astype(str).str.strip() == "")
    df.loc[empty_source_mask, "translation_status"] = "empty_source"
    pending_mask = pending_mask & ~empty_source_mask

    pending_indices = df.index[pending_mask].tolist()
    if not pending_indices:
        logging.info("Tidak ada baris baru yang perlu diterjemahkan.")
        return df

    logging.info("Memuat model terjemahan: %s", TRANSLATION_MODEL_NAME)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rows_since_save = 0
    for index in tqdm(pending_indices, desc="Translating"):
        source_text = str(df.at[index, "english_translation"])
        arabic_text = str(df.at[index, "teks_arab"])
        token_count = len(tokenizer(source_text, add_special_tokens=False)["input_ids"])

        if token_count <= TRANSLATION_CHUNK_MAX_TOKENS:
            chunks, had_oversized_sentence = [source_text], False
        else:
            chunks, had_oversized_sentence = _chunk_text_for_translation(
                source_text, tokenizer, TRANSLATION_CHUNK_MAX_TOKENS
            )
            if not chunks:
                chunks = [source_text]
        was_chunked = len(chunks) > 1

        try:
            translated_chunks = _translate_chunks(chunks, tokenizer, model, device)
        except Exception as exc:  # noqa: BLE001 - isolate failing row, keep going
            df.at[index, "translation_status"] = "failed"
            df.at[index, "translation_error"] = str(exc)
            df.at[index, "translation_warning"] = ""
            continue

        combined = " ".join(piece for piece in translated_chunks if piece).strip()

        # Rule 1: clause-level splitting (as opposed to whole-sentence chunking)
        # means a fragment was translated without its full grammatical context,
        # so it can never be considered plain chunked_ok/ok, only needs_review.
        warnings: list[str] = []
        if had_oversized_sentence:
            warnings.append("ada kalimat sangat panjang yang terpaksa dipotong per-klausa")
        if not combined:
            warnings.append("output kosong setelah translasi")
        else:
            source_len = len(source_text)
            ratio = len(combined) / source_len if source_len else 0.0
            if ratio < TRANSLATION_RATIO_WARNING_THRESHOLD:
                warnings.append(f"rasio panjang ID/EN rendah ({ratio:.2f})")
            if _looks_repetitive(combined):
                warnings.append("output tampak repetitif/tidak wajar")

            # Rules 2-3: flag any character name or tafsir/nahwu jargon in the
            # translation that has no basis in the English source (known leak
            # pattern from opus-mt-en-id's religious training data).
            leaked_terms = _find_leaked_blocklist_terms(combined, source_text, arabic_text)
            if leaked_terms:
                warnings.append(
                    "istilah/nama tidak bersumber dari english_translation/teks_arab: "
                    + ", ".join(sorted(set(leaked_terms)))
                )
            unsourced_names = _find_unsourced_midsentence_names(combined, source_text, arabic_text)
            if unsourced_names:
                warnings.append(
                    "kemungkinan nama tokoh tidak bersumber: " + ", ".join(sorted(set(unsourced_names)))
                )

        # Rules 4-5: chunked_ok is only earned when chunking was plain
        # whole-sentence grouping AND nothing above raised a warning. Any
        # warning - including a partially-drifted translation - forces
        # needs_review, even for rows that were chunked cleanly.
        status = "chunked_ok" if was_chunked else "ok"
        if warnings:
            status = "needs_review"

        df.at[index, "terjemahan"] = combined
        df.at[index, "translation_status"] = status
        df.at[index, "translation_warning"] = "; ".join(warnings)
        df.at[index, "translation_model"] = TRANSLATION_MODEL_NAME
        df.at[index, "translation_error"] = ""

        rows_since_save += 1
        if rows_since_save >= SAVE_EVERY:
            df.to_csv(TRANSLATION_CHECKPOINT_PATH, index=False, encoding="utf-8-sig")
            rows_since_save = 0

    df.to_csv(TRANSLATION_CHECKPOINT_PATH, index=False, encoding="utf-8-sig")
    return df


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_{timestamp}.backup{path.suffix}")
    backup_path.write_bytes(path.read_bytes())
    logging.info("Backup dibuat: %s", backup_path.name)


def save_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Save all dataset outputs, backing up any file that already exists.

    Returns the saved full dataframe (including the generated Num_hadith
    column) so callers can build summaries/previews from it.
    """
    df = df.copy()
    df.insert(0, "Num_hadith", range(1, len(df) + 1))
    if "ringkasan_referensi" not in df.columns:
        df["ringkasan_referensi"] = ""
    df["ringkasan_referensi"] = ""

    for column in FULL_OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df_full = df[FULL_OUTPUT_COLUMNS]

    _backup_if_exists(OUTPUT_FULL_PATH)
    df_full.to_csv(OUTPUT_FULL_PATH, index=False, encoding="utf-8-sig")
    logging.info("Dataset full disimpan: %s (%s baris)", OUTPUT_FULL_PATH, len(df_full))

    clean_mask = df_full["translation_status"].isin(MINIMAL_ALLOWED_TRANSLATION_STATUSES)
    df_minimal = df_full.loc[clean_mask, MINIMAL_OUTPUT_COLUMNS]
    _backup_if_exists(OUTPUT_MINIMAL_PATH)
    df_minimal.to_csv(OUTPUT_MINIMAL_PATH, index=False, encoding="utf-8-sig")
    logging.info(
        "Dataset minimal disimpan: %s (%s dari %s baris; hanya translation_status %s)",
        OUTPUT_MINIMAL_PATH,
        len(df_minimal),
        len(df_full),
        MINIMAL_ALLOWED_TRANSLATION_STATUSES,
    )

    review_mask = df_full["translation_status"].isin(REVIEW_TRANSLATION_STATUSES)
    df_review = df_full.loc[review_mask]
    _backup_if_exists(OUTPUT_REVIEW_PATH)
    df_review.to_csv(OUTPUT_REVIEW_PATH, index=False, encoding="utf-8-sig")
    logging.info(
        "Dataset review (needs_review/failed) disimpan: %s (%s baris)",
        OUTPUT_REVIEW_PATH,
        len(df_review),
    )

    df_ref_template = df_full[["Num_hadith", "source_ref", "teks_arab", "terjemahan", "ringkasan_referensi"]].copy()
    df_ref_template["validator"] = ""
    df_ref_template["catatan_validasi"] = ""
    df_ref_template = df_ref_template[REF_TEMPLATE_COLUMNS]
    _backup_if_exists(OUTPUT_REF_TEMPLATE_PATH)
    df_ref_template.to_csv(OUTPUT_REF_TEMPLATE_PATH, index=False, encoding="utf-8-sig")
    logging.info("Template ringkasan referensi disimpan: %s", OUTPUT_REF_TEMPLATE_PATH)

    if COPY_MINIMAL_TO_INPUT:
        _backup_if_exists(INPUT_MINIMAL_COPY_PATH)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_minimal.to_csv(INPUT_MINIMAL_COPY_PATH, index=False, encoding="utf-8-sig")
        logging.info("Dataset minimal disalin ke pipeline: %s", INPUT_MINIMAL_COPY_PATH)
    else:
        logging.info(
            "COPY_MINIMAL_TO_INPUT=False. Salin manual %s ke %s jika ingin dipakai pipeline.",
            OUTPUT_MINIMAL_PATH,
            INPUT_MINIMAL_COPY_PATH,
        )

    return df_full


@dataclass
class ProgressEntry:
    book_number: int
    book_title: str
    book_url: str
    hadith_count: int
    status: str
    error_message: str
    timestamp: str


def save_progress(entries: list[ProgressEntry]) -> None:
    """Save/merge scraping progress so the run's state is inspectable and resumable."""
    new_df = pd.DataFrame([entry.__dict__ for entry in entries])
    if OUTPUT_PROGRESS_PATH.exists():
        try:
            old_df = pd.read_csv(OUTPUT_PROGRESS_PATH, encoding="utf-8-sig")
            merged = pd.concat([old_df, new_df]).drop_duplicates(
                subset=["book_number"], keep="last"
            )
        except (OSError, pd.errors.ParserError):
            merged = new_df
    else:
        merged = new_df

    merged = merged.sort_values("book_number").reset_index(drop=True)
    merged.to_csv(OUTPUT_PROGRESS_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    """Run the dataset builder end-to-end with safe defaults."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    setup_logging()
    setup_directories()

    logging.info(
        "Konfigurasi: RUN_FULL_SCRAPE=%s, LIMIT_BOOKS=%s, LIMIT_HADITHS=%s, RUN_TRANSLATION=%s",
        RUN_FULL_SCRAPE,
        LIMIT_BOOKS,
        LIMIT_HADITHS,
        RUN_TRANSLATION,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    index_html = fetch_url_with_cache(session, COLLECTION_INDEX_URL, "index")
    if index_html is None:
        logging.error("Tidak bisa mengambil halaman indeks. Menghentikan proses.")
        return

    all_books = parse_bukhari_book_list(index_html)
    if not all_books:
        logging.error("Tidak ada book yang berhasil diparse dari halaman indeks. Menghentikan.")
        return
    logging.info("Ditemukan %s book pada halaman indeks.", len(all_books))

    books_to_process = all_books if RUN_FULL_SCRAPE else all_books[:LIMIT_BOOKS]

    all_records: list[dict] = []
    progress_entries: list[ProgressEntry] = []

    for book in tqdm(books_to_process, desc="Books"):
        book_number = book["book_number"]
        book_title = book["book_title"]
        book_url = book["book_url"]
        cache_name = f"book_{book_number}"
        timestamp = datetime.now().isoformat(timespec="seconds")

        cached_records = load_book_records_cache(cache_name)
        if cached_records is not None:
            logging.info("Memakai cache JSON untuk book %s (%s)", book_number, book_title)
            records = cached_records
            progress_entries.append(
                ProgressEntry(book_number, book_title, book_url, len(records), "cached", "", timestamp)
            )
        else:
            html = fetch_url_with_cache(session, book_url, cache_name)
            if html is None:
                progress_entries.append(
                    ProgressEntry(
                        book_number, book_title, book_url, 0, "failed", "gagal fetch halaman", timestamp
                    )
                )
                continue
            records = parse_bukhari_book_page(html, book_number, book_title)
            save_book_records_cache(cache_name, records)
            progress_entries.append(
                ProgressEntry(book_number, book_title, book_url, len(records), "success", "", timestamp)
            )

        all_records.extend(records)

        if not RUN_FULL_SCRAPE and len(all_records) >= LIMIT_HADITHS:
            break

    save_progress(progress_entries)

    if not all_records:
        logging.error("Tidak ada hadis yang berhasil diparse. Menghentikan proses.")
        return

    if not RUN_FULL_SCRAPE:
        all_records = all_records[:LIMIT_HADITHS]

    df = pd.DataFrame(all_records)
    df = validate_scraped_dataframe(df)
    df = translate_english_to_indonesian(df)
    df = save_outputs(df)

    total_hadith = len(df)
    total_books = df["book_number"].nunique()
    valid_count = int((df["data_status"] == "valid").sum())
    problem_count = total_hadith - valid_count
    empty_arabic = int((df["teks_arab"].astype(str).str.strip() == "").sum())
    empty_english = int((df["english_translation"].astype(str).str.strip() == "").sum())
    empty_indo = int((df["terjemahan"].astype(str).str.strip() == "").sum())

    print("\n=== Ringkasan Dataset Builder ===")
    print(f"Total hadis     : {total_hadith}")
    print(f"Total book      : {total_books}")
    print(f"Data valid      : {valid_count}")
    print(f"Data bermasalah : {problem_count}")
    print(f"Arab kosong     : {empty_arabic}")
    print(f"English kosong  : {empty_english}")
    print(f"Terjemahan kosong: {empty_indo}")
    print("\nStatus terjemahan:")
    print(df["translation_status"].value_counts().to_string())
    clean_df = df[df["translation_status"].isin(MINIMAL_ALLOWED_TRANSLATION_STATUSES)]
    review_df = df[df["translation_status"].isin(REVIEW_TRANSLATION_STATUSES)]
    print(
        f"\ndataset_hadis_minimal.csv: {len(clean_df)} baris (status {MINIMAL_ALLOWED_TRANSLATION_STATUSES})"
    )
    print(
        f"dataset_hadis_review.csv : {len(review_df)} baris (status {REVIEW_TRANSLATION_STATUSES}, wajib dicek manusia)"
    )

    def _truncate(text: object, width: int = 60) -> str:
        text = str(text)
        return text if len(text) <= width else text[: width - 3] + "..."

    if not clean_df.empty:
        print("\n5 baris pertama dataset minimal (bersih):")
        minimal_preview = clean_df[MINIMAL_OUTPUT_COLUMNS[:1] + ["teks_arab", "terjemahan"]].head(5).copy()
        minimal_preview["teks_arab"] = minimal_preview["teks_arab"].map(_truncate)
        minimal_preview["terjemahan"] = minimal_preview["terjemahan"].map(_truncate)
        print(minimal_preview.to_string(index=False))
    else:
        print("\nTidak ada baris berstatus ok/chunked_ok pada run ini; dataset_hadis_minimal.csv kosong.")
    print(f"\nOutput tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
