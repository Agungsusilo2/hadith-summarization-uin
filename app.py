"""Gradio demo: hadith catalog (reads pipeline CSV output, no live inference)
plus an optional manual free-text summarizer tab.
"""

from __future__ import annotations

import re

import pandas as pd

from src import config
from src.abstractive_summarizer import AbstractiveSummarizer, GenerationConfig
from src.extractive_summarizer import ExtractiveSummarizer
from src.preprocessing import clean_indonesian_text

GENERAL_DISCLAIMER = (
    "Ringkasan dihasilkan oleh model AI dan perlu diperiksa kembali. "
    "Hasil ini bukan tafsir, syarah, atau fatwa."
)

ABSTRACTIVE_WARNING = (
    "PERINGATAN KHUSUS ABSTRACTIVE: metode ini adalah eksperimen pembanding, "
    "bukan output utama. Model mT5 dapat menghasilkan hallucination berupa "
    "nama, tempat, atau detail yang tidak ada di teks sumber. Metode "
    "Extractive lebih aman karena hanya mengutip kalimat asli. Jangan "
    "jadikan hasil Abstractive sebagai rujukan keagamaan tanpa validasi "
    "manusia."
)

NO_HALLUCINATION_FLAG_MESSAGE = "Tidak ada flag otomatis, tetapi tetap perlu validasi manusia."

CSV_MISSING_MESSAGE = (
    "File hasil_ringkasan_hadis.csv belum ditemukan. "
    "Jalankan run_pipeline.py terlebih dahulu."
)

DATASET_FULL_PATH = config.OUTPUT_DIR / "dataset_hadis_full.csv"

# Columns read from dataset_hadis_full.csv to enrich the catalog with book
# metadata. Purely optional: the catalog still works from Num_hadith alone
# if this file or these columns are missing.
METADATA_COLUMNS = [
    "book_number",
    "book_title",
    "hadith_number",
    "source_ref",
    "translation_status",
    "translation_warning",
]

# Summary columns attached from hasil_ringkasan_hadis.csv onto the full
# scraped catalog (dataset_hadis_full.csv). Hadiths missing these (not yet
# processed by run_pipeline.py) get an Extractive summary computed live in
# show_hadith_detail; Abstractive is never generated live.
SUMMARY_COLUMNS = [
    "ringkasan_extractive",
    "status_extractive",
    "jumlah_kata_sumber",
    "jumlah_kata_extractive",
    "compression_ratio_extractive",
    "ringkasan_abstractive",
    "status_abstractive",
    "jumlah_kata_abstractive",
    "compression_ratio_abstractive",
    "abstractive_disclaimer",
]

LIVE_EXTRACTIVE_NOTE = (
    "⚡ Dihitung langsung di aplikasi (Extractive TF-IDF), belum melalui pipeline resmi run_pipeline.py."
)

ABSTRACTIVE_NOT_AVAILABLE_MESSAGE = (
    "Ringkasan Abstractive (mT5) belum tersedia untuk hadis ini. mT5 tidak "
    "dijalankan otomatis di aplikasi karena berat (perlu memuat model). "
    "Jalankan run_pipeline.py untuk memprosesnya."
)


def _clean_number_label(value: object) -> str:
    """Render a possibly-float-typed number (from a CSV merge) as a plain string."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _load_catalog() -> tuple[pd.DataFrame, str]:
    """Load pipeline results for the catalog tab. Never runs any model.

    Returns (dataframe, error_message). error_message is empty on success.
    """
    if not config.OUTPUT_DATA_PATH.exists():
        return pd.DataFrame(), CSV_MISSING_MESSAGE

    try:
        df = pd.read_csv(config.OUTPUT_DATA_PATH, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - surface any read failure in the UI
        return pd.DataFrame(), f"Gagal membaca {config.OUTPUT_DATA_PATH.name}: {exc}"

    if config.HADITH_ID_COLUMN not in df.columns:
        return pd.DataFrame(), (
            f"Kolom {config.HADITH_ID_COLUMN} tidak ditemukan di {config.OUTPUT_DATA_PATH.name}."
        )

    if DATASET_FULL_PATH.exists():
        try:
            meta = pd.read_csv(DATASET_FULL_PATH, encoding="utf-8-sig")
            available = [c for c in METADATA_COLUMNS if c in meta.columns]
            if "Num_hadith" in meta.columns and available:
                meta = meta[["Num_hadith"] + available].rename(
                    columns={"Num_hadith": config.HADITH_ID_COLUMN}
                )
                df = df.drop(columns=[c for c in available if c in df.columns])
                df = df.merge(meta, on=config.HADITH_ID_COLUMN, how="left")
        except Exception:  # noqa: BLE001 - metadata is optional, never fatal
            pass

    for column in METADATA_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df.fillna("")
    return df, ""


def _load_hallucination_flags() -> dict[int, str]:
    """Read already-computed hallucination flags from error_analysis_samples.csv.

    That file's `error_flags` column packs several heuristic flags together
    (e.g. "extractive_sama_dengan_sumber;abstractive_kemungkinan_hallucination_nama:Medinah").
    This only extracts the hallucination-name portion, per Num_hadith. Nothing
    is recomputed - it is purely read from the existing pipeline output.
    """
    if not config.ERROR_ANALYSIS_PATH.exists():
        return {}
    try:
        err_df = pd.read_csv(config.ERROR_ANALYSIS_PATH, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001 - optional enrichment, never fatal
        return {}
    if config.HADITH_ID_COLUMN not in err_df.columns or "error_flags" not in err_df.columns:
        return {}

    flags: dict[int, str] = {}
    for _, row in err_df.iterrows():
        for part in str(row["error_flags"]).split(";"):
            if part.startswith("abstractive_kemungkinan_hallucination_nama:"):
                try:
                    flags[int(row[config.HADITH_ID_COLUMN])] = part.split(":", 1)[1]
                except (ValueError, TypeError):
                    continue
    return flags


def _load_all_hadith(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Load every already-scraped hadith (dataset_hadis_full.csv) and attach
    precomputed summary columns from catalog_df where available.

    This only reads existing local CSVs - no scraping, no model calls.
    Hadiths without a pipeline-computed extractive summary are still
    returned (with empty summary columns); show_hadith_detail computes an
    Extractive summary live for those. Falls back to catalog_df if the full
    dataset file is missing, so behavior is unchanged when it's absent.
    """
    if not DATASET_FULL_PATH.exists():
        return catalog_df

    try:
        full = pd.read_csv(DATASET_FULL_PATH, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001 - fall back to the smaller catalog, never fatal
        return catalog_df

    if config.HADITH_ID_COLUMN not in full.columns or "terjemahan" not in full.columns:
        return catalog_df

    if not catalog_df.empty:
        available = [c for c in SUMMARY_COLUMNS if c in catalog_df.columns]
        if available:
            summary = catalog_df[[config.HADITH_ID_COLUMN] + available]
            full = full.merge(summary, on=config.HADITH_ID_COLUMN, how="left")

    for column in SUMMARY_COLUMNS:
        if column not in full.columns:
            full[column] = ""
    return full.fillna("")


CATALOG_DF, CATALOG_LOAD_ERROR = _load_catalog()
HALLUCINATION_FLAGS = _load_hallucination_flags()
ALL_HADITH_DF = _load_all_hadith(CATALOG_DF)


def _book_label(row: pd.Series) -> str:
    book_title = str(row.get("book_title", "")).strip()
    if not book_title:
        return "Semua Hadis"
    book_number = _clean_number_label(row.get("book_number", ""))
    return f"Book {book_number}: {book_title}" if book_number else book_title


def _book_sort_key(label: str) -> tuple[int, object]:
    match = re.match(r"Book (\d+):", label)
    if match:
        return (0, int(match.group(1)))
    return (1, label)


def book_choices() -> list[str]:
    if ALL_HADITH_DF.empty:
        return []
    labels = {_book_label(row) for _, row in ALL_HADITH_DF.iterrows()}
    if labels == {"Semua Hadis"}:
        return ["Semua Hadis"]
    return sorted(labels, key=_book_sort_key)


def _hadith_label(row: pd.Series) -> str:
    base = f"Hadis #{row[config.HADITH_ID_COLUMN]}"
    book_title = str(row.get("book_title", "")).strip()
    if book_title:
        book_number = _clean_number_label(row.get("book_number", ""))
        book_part = f"Book {book_number}: {book_title}" if book_number else book_title
        base = f"{base} — {book_part}"
    if not str(row.get("ringkasan_extractive", "")).strip():
        base += " (belum diringkas)"
    return base


def hadith_choices(book_label: str | None) -> list[tuple[str, int]]:
    if ALL_HADITH_DF.empty:
        return []
    if book_label and book_label != "Semua Hadis":
        subset = ALL_HADITH_DF[ALL_HADITH_DF.apply(_book_label, axis=1) == book_label]
    else:
        subset = ALL_HADITH_DF
    choices = [
        (_hadith_label(row), int(row[config.HADITH_ID_COLUMN])) for _, row in subset.iterrows()
    ]
    choices.sort(key=lambda item: item[1])
    return choices


def _technical_details(row: pd.Series) -> str:
    return (
        f"Jumlah kata sumber: {row.get('jumlah_kata_sumber', '-')}\n"
        f"Jumlah kata extractive: {row.get('jumlah_kata_extractive', '-')}\n"
        f"Jumlah kata abstractive: {row.get('jumlah_kata_abstractive', '-')}\n"
        f"Compression ratio extractive: {row.get('compression_ratio_extractive', '-')}\n"
        f"Compression ratio abstractive: {row.get('compression_ratio_abstractive', '-')}"
    )


def show_hadith_detail(num_hadith: int | None):
    """Look up one hadith row and format it for display.

    Reads from ALL_HADITH_DF only. If a pipeline-computed Extractive summary
    exists it is used as-is; otherwise one is computed live here (cheap
    TF-IDF, no model download). Abstractive/mT5 is only ever read from the
    pipeline output - it is never generated live.
    """
    import gradio as gr

    empty_accordion = gr.update(open=False)
    if ALL_HADITH_DF.empty or num_hadith is None:
        return "", "", "", "", "", "", "", "", GENERAL_DISCLAIMER, "", empty_accordion

    matches = ALL_HADITH_DF[ALL_HADITH_DF[config.HADITH_ID_COLUMN] == int(num_hadith)]
    if matches.empty:
        return "", "", "", "", "", "", "", "", GENERAL_DISCLAIMER, "", empty_accordion
    row = matches.iloc[0]

    title = f"### Hadis #{row[config.HADITH_ID_COLUMN]}"
    source_ref = str(row.get("source_ref", "")).strip()
    if source_ref:
        title += f" ({source_ref})"

    extractive_text = str(row.get("ringkasan_extractive", "")).strip()
    if not extractive_text:
        live_result = extractive_summarizer.summarize(str(row.get("terjemahan", "")))
        extractive_text = live_result["summary"]
        if extractive_text:
            extractive_text += f"\n\n{LIVE_EXTRACTIVE_NOTE}"

    abstractive_text = str(row.get("ringkasan_abstractive", "")).strip()
    status_abstractive = str(row.get("status_abstractive", "")).strip()

    if abstractive_text:
        status_note = status_abstractive
        if status_abstractive == "generated_ok":
            status_note += (
                "\n\nCatatan: status ini hanya berarti model berhasil menghasilkan "
                "teks secara teknis, bukan berarti isinya valid."
            )
        hallucination_text = HALLUCINATION_FLAGS.get(
            int(row[config.HADITH_ID_COLUMN]), ""
        ) or NO_HALLUCINATION_FLAG_MESSAGE
        warning_text = f"⚠️ {ABSTRACTIVE_WARNING}"
        disclaimer_abstractive = str(row.get("abstractive_disclaimer", "")).strip()
        if disclaimer_abstractive:
            warning_text += f"\n\n{disclaimer_abstractive}"
    else:
        status_note = "belum_diproses"
        hallucination_text = "Belum ada data (abstractive belum dijalankan untuk hadis ini)."
        warning_text = f"⚠️ {ABSTRACTIVE_WARNING}\n\n{ABSTRACTIVE_NOT_AVAILABLE_MESSAGE}"

    return (
        title,
        str(row.get("teks_arab", "")),
        str(row.get("terjemahan", "")),
        extractive_text,
        abstractive_text,
        status_note,
        hallucination_text,
        warning_text,
        GENERAL_DISCLAIMER,
        _technical_details(row),
        gr.update(open=True),
    )


def on_book_change(book_label: str):
    import gradio as gr

    choices = hadith_choices(book_label)
    if not choices:
        return gr.update(choices=[], value=None)
    return gr.update(choices=choices, value=choices[0][1])


# ---------------------------------------------------------------------------
# Tab 2 (optional): manual free-text summarizer, kept from the previous demo.
# ---------------------------------------------------------------------------

extractive_summarizer = ExtractiveSummarizer()
abstractive_summarizer: AbstractiveSummarizer | None = None
abstractive_load_error = ""


def _get_abstractive() -> AbstractiveSummarizer:
    global abstractive_summarizer, abstractive_load_error
    if abstractive_summarizer is None:
        generation_config = GenerationConfig(
            max_input_length=config.MAX_INPUT_LENGTH,
            max_summary_length=config.MAX_SUMMARY_LENGTH,
            min_summary_length=config.MIN_SUMMARY_LENGTH,
            num_beams=config.NUM_BEAMS,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            batch_size=1,
        )
        try:
            abstractive_summarizer = AbstractiveSummarizer(config.ABSTRACTIVE_MODEL_NAME, generation_config)
        except Exception as exc:
            abstractive_load_error = str(exc)
            raise
    return abstractive_summarizer


def summarize_manual(text: str, method: str) -> tuple[str, int, int, float, str, str]:
    """Summarize free-text input on demand (this tab only)."""
    cleaned = clean_indonesian_text(text)
    if not cleaned:
        return "", 0, 0, 0.0, "empty_input", GENERAL_DISCLAIMER

    source_words = len(cleaned.split())
    summaries: list[str] = []
    statuses: list[str] = []
    result_words = 0
    ratio = 0.0

    if method in {"Extractive", "Keduanya"}:
        result = extractive_summarizer.summarize(cleaned)
        summaries.append(f"[Extractive - Ringkasan Utama]\n{result['summary']}")
        statuses.append(f"extractive={result['status']}")
        result_words += int(result["summary_word_count"])
        ratio = float(result["compression_ratio"])

    if method in {"Abstractive", "Keduanya"}:
        try:
            result = _get_abstractive().summarize(cleaned)
        except Exception:
            result = {
                "summary": "",
                "status": "model_load_error",
                "summary_word_count": 0,
                "compression_ratio": 0.0,
                "error": abstractive_load_error,
            }
        summaries.append(f"[Abstractive - Eksperimen Pembanding]\n{result['summary']}")
        statuses.append(f"abstractive={result['status']}")
        if result.get("error"):
            statuses.append(f"error={result['error']}")
        result_words += int(result["summary_word_count"])
        ratio = float(result["compression_ratio"])

    warning_text = GENERAL_DISCLAIMER
    if method in {"Abstractive", "Keduanya"}:
        warning_text = f"{GENERAL_DISCLAIMER}\n\n{ABSTRACTIVE_WARNING}"

    return "\n\n".join(summaries), source_words, result_words, ratio, "; ".join(statuses), warning_text


def main() -> None:
    """Launch the Gradio UI: hadith catalog (default tab) + manual summarizer."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install gradio terlebih dahulu untuk menjalankan app.py.") from exc

    with gr.Blocks(title="Demo Ringkasan Hadis") as demo:
        gr.Markdown("# Demo Ringkasan Hadis")
        gr.Markdown(GENERAL_DISCLAIMER)

        with gr.Tabs():
            with gr.Tab("Katalog Hadis"):
                if CATALOG_LOAD_ERROR:
                    gr.Markdown(f"**{CATALOG_LOAD_ERROR}**")
                else:
                    gr.Markdown(
                        "Semua hadis yang sudah pernah di-scrape bisa dipilih di sini. "
                        "Hadis berlabel **(belum diringkas)** akan langsung dihitung "
                        "Ringkasan Extractive-nya saat dipilih. Ringkasan Abstractive "
                        "(mT5) tetap hanya tersedia untuk hadis yang sudah diproses "
                        "lewat `run_pipeline.py`, karena modelnya berat dan tidak "
                        "dijalankan otomatis di aplikasi."
                    )
                    with gr.Row():
                        book_dropdown = gr.Dropdown(
                            choices=book_choices(),
                            value=(book_choices()[0] if book_choices() else None),
                            label="Pilih Book",
                        )
                        hadith_dropdown = gr.Dropdown(
                            choices=hadith_choices(book_choices()[0] if book_choices() else None),
                            value=(
                                hadith_choices(book_choices()[0])[0][1]
                                if book_choices() and hadith_choices(book_choices()[0])
                                else None
                            ),
                            label="Pilih Hadis",
                        )

                    detail_title = gr.Markdown()
                    teks_arab_box = gr.Textbox(label="Teks Arab", lines=4, interactive=False)
                    terjemahan_box = gr.Textbox(label="Terjemahan Indonesia", lines=4, interactive=False)
                    extractive_box = gr.Textbox(
                        label="Ringkasan Utama (Extractive TF-IDF)", lines=4, interactive=False
                    )
                    abstractive_box = gr.Textbox(
                        label="Ringkasan Eksperimen (mT5 Abstractive)", lines=4, interactive=False
                    )
                    status_box = gr.Textbox(label="Status Abstractive", lines=2, interactive=False)
                    hallucination_box = gr.Textbox(
                        label="Flag Kemungkinan Hallucination", lines=2, interactive=False
                    )
                    warning_box = gr.Markdown()
                    disclaimer_box = gr.Markdown()

                    with gr.Accordion("Detail Teknis", open=False) as technical_accordion:
                        technical_box = gr.Textbox(lines=5, interactive=False, show_label=False)

                    detail_outputs = [
                        detail_title,
                        teks_arab_box,
                        terjemahan_box,
                        extractive_box,
                        abstractive_box,
                        status_box,
                        hallucination_box,
                        warning_box,
                        disclaimer_box,
                        technical_box,
                        technical_accordion,
                    ]

                    book_dropdown.change(
                        on_book_change, inputs=book_dropdown, outputs=hadith_dropdown
                    ).then(show_hadith_detail, inputs=hadith_dropdown, outputs=detail_outputs)
                    hadith_dropdown.change(
                        show_hadith_detail, inputs=hadith_dropdown, outputs=detail_outputs
                    )
                    demo.load(show_hadith_detail, inputs=hadith_dropdown, outputs=detail_outputs)

            with gr.Tab("Ringkas Manual"):
                gr.Markdown(
                    "Mode input manual (opsional): tempel teks terjemahan hadis sendiri "
                    "untuk diringkas langsung. Katalog Hadis tetap jalur utama."
                )
                manual_text = gr.Textbox(lines=10, label="Teks terjemahan hadis bahasa Indonesia")
                manual_method = gr.Radio(
                    ["Extractive", "Abstractive", "Keduanya"],
                    value="Extractive",
                    label="Metode",
                    info=(
                        "Extractive = output utama (aman, mengutip kalimat asli). "
                        "Abstractive = eksperimen pembanding, dapat hallucination."
                    ),
                )
                manual_button = gr.Button("Ringkas")
                manual_summary = gr.Textbox(label="Hasil ringkasan")
                with gr.Row():
                    manual_source_words = gr.Number(label="Jumlah kata sumber")
                    manual_result_words = gr.Number(label="Jumlah kata hasil")
                    manual_ratio = gr.Number(label="Compression ratio")
                manual_status = gr.Textbox(label="Status model")
                manual_warning = gr.Textbox(label="Peringatan")

                manual_button.click(
                    summarize_manual,
                    inputs=[manual_text, manual_method],
                    outputs=[
                        manual_summary,
                        manual_source_words,
                        manual_result_words,
                        manual_ratio,
                        manual_status,
                        manual_warning,
                    ],
                )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
