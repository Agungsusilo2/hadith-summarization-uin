---
title: Ringkasan Hadis AI
emoji: 📖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Hadith Summarization

Proyek ini membuat pipeline ringkasan otomatis untuk teks terjemahan hadis bahasa Indonesia. Sistem ini adalah implementasi inference, bukan fine-tuning.

Media bantu pembelajaran mahasiswa UIN Jakarta: **Katalog Hadis** interaktif (Extractive TF-IDF sebagai ringkasan utama, mT5 Abstractive sebagai eksperimen pembanding dengan peringatan hallucination).

## Cara Menggunakan (untuk Pengunjung Space)

Aplikasi ini sudah berjalan otomatis di Space — **tidak perlu install apa pun**. Langsung saja:

1. Buka tab **"Katalog Hadis"** (tab utama, biasanya sudah aktif).
2. Pilih **Book** di dropdown pertama, lalu pilih **Hadis** di dropdown kedua.
3. Detailnya langsung muncul: teks Arab, terjemahan Indonesia, **Ringkasan Utama (Extractive)**, **Ringkasan Eksperimen (mT5 Abstractive)** beserta peringatan hallucination-nya, dan disclaimer.
4. Hadis berlabel **"(belum diringkas)"** tetap bisa dipilih — Ringkasan Extractive-nya dihitung otomatis saat itu juga; Abstractive untuk hadis ini belum tersedia (lihat penjelasan di tab tersebut).
5. Tab **"Ringkas Manual"** (opsional) untuk menempel teks terjemahan sendiri dan meringkasnya langsung.

**Penting:** ringkasan dihasilkan AI dan perlu diperiksa kembali — bukan tafsir, syarah, atau fatwa. Metode Abstractive (mT5) khususnya adalah eksperimen pembanding yang bisa menghasilkan hallucination (nama/tempat/detail yang tidak ada di sumber); Extractive lebih aman karena hanya mengutip kalimat asli.

Bagian di bawah ini (mulai "Hasil Audit Notebook") adalah **dokumentasi teknis untuk pengembangan/pipeline**, bukan wajib dibaca kalau kamu cuma ingin memakai Katalog Hadis di atas.

Arsitektur proyek dibuat hybrid:

- PyCharm/laptop lokal untuk proses ringan: validasi dataset, preprocessing, baseline extractive, membaca output Colab, error analysis, template evaluasi manusia, dan demo GUI.
- Google Colab untuk proses berat: inference abstractive dengan model Transformer dan evaluasi berat jika dibutuhkan.

## Hasil Audit Notebook

Notebook referensi `bukhari_ts (3).ipynb` dibaca secara read-only. Temuan utama:

- Dataset yang dipakai: `dataset_bukhari_bersih.csv`, `sanadset.csv`, dan hasil lanjutan `df2_dengan_terjemahan_lengkap.csv`.
- Semua dataset di notebook lama dipanggil dari Google Drive Colab, misalnya `/content/drive/MyDrive/Alii_Siraj`.
- Kolom yang terlihat: `arab`, `terjemah`, `Book`, `Num_hadith`, `Hadith`, `Matn`, `hadith_clean`, `arab_match`, `terjemahan`, `arab_sunnah`, `terjemahan_en`, `url_sunnah`, dan `terjemahan_helsinki`.
- Kolom nomor hadis yang relevan: `Num_hadith`.
- Kolom teks Arab yang relevan dari dataset sanad: `Hadith` atau hasil pencocokan `arab_match`.
- Kolom terjemahan Indonesia yang relevan: `terjemahan`.
- Notebook melakukan normalisasi teks Arab, pencocokan hadis, scraping Sunnah.com untuk data yang belum cocok, dan terjemahan English ke Indonesia memakai `Helsinki-NLP/opus-mt-en-id`.
- Notebook juga memuat contoh Gemini API, tetapi bagian itu tidak digunakan karena proyek ini tidak memakai API berbayar.

Logika yang digunakan kembali:

- Validasi kolom `Num_hadith`.
- Penggunaan kolom `terjemahan` sebagai input utama summarization.
- Pembersihan ringan tag HTML dan whitespace.
- Chunking untuk teks panjang pada model abstractive.

Bagian notebook yang sengaja tidak digunakan:

- `drive.mount` dan path `/content/` pada proyek lokal.
- Scraping Sunnah.com.
- Upload file Colab pada pipeline lokal.
- Gemini API.
- Penerjemahan ulang jika dataset terjemahan Indonesia sudah tersedia.
- Normalisasi Arab untuk pipeline summarization Indonesia.

## Struktur Folder

```text
hadith_summarization/
|-- data/
|   |-- input/
|   `-- output/
|-- models/
|-- notebooks/
|   `-- hadith_summarization_colab.ipynb
|-- src/
|   |-- colab_output_loader.py
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- extractive_summarizer.py
|   |-- abstractive_summarizer.py
|   |-- evaluation.py
|   |-- error_analysis.py
|   `-- utils.py
|-- tests/
|-- app.py
|-- run_pipeline.py
|-- run_light_pipeline.py
|-- requirements.txt
|-- requirements-local.txt
|-- requirements-colab.txt
`-- README.md
```

## Versi Python

Direkomendasikan Python 3.10 atau 3.11. Python 3.13 belum selalu aman untuk semua library NLP berat seperti `torch`, `transformers`, dan `bert-score`.

## Setup Lokal Ringan

Gunakan ini untuk laptop/PyCharm.

```powershell
cd C:\Users\agung\Downloads\NLP\hadith_summarization
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local.txt
```

`requirements.txt` dan `requirements-local.txt` berisi dependency lokal ringan. Keduanya tidak memuat `torch` dan `transformers`, sehingga lebih aman untuk laptop yang tidak dipakai menjalankan model besar.

## Dataset

Letakkan dataset di:

```text
data/input/dataset_hadis.csv
```

Kolom default yang dibutuhkan:

- `Num_hadith`
- `teks_arab`
- `terjemahan`

Kolom opsional:

- `ringkasan_referensi`

Jika dataset memakai nama kolom berbeda, ubah pemetaan di `src/config.py`. Jangan mengubah dataset sumber hanya untuk menyesuaikan nama kolom.

Contoh jika dataset memakai kolom dari notebook:

```python
HADITH_ID_COLUMN = "Num_hadith"
ARABIC_TEXT_COLUMN = "arab_match"
INDONESIAN_TEXT_COLUMN = "terjemahan"
REFERENCE_SUMMARY_COLUMN = "ringkasan_referensi"
```

## Jalur Lokal Ringan

Jalankan:

```powershell
python run_light_pipeline.py
```

Jalur ini menjalankan:

- load dan validasi dataset;
- preprocessing teks Indonesia;
- baseline extractive TF-IDF;
- merge hasil abstractive dari Colab jika file tersedia;
- error analysis ringan;
- template evaluasi manusia.

Jalur ini tidak memuat model Transformer.

Jika sudah ada hasil Colab, letakkan di:

```text
data/output/hasil_ringkasan_hadis_colab.csv
```

Lalu jalankan ulang:

```powershell
python run_light_pipeline.py
```

Output utama:

```text
data/output/hasil_ringkasan_hadis.csv
```

Output tambahan:

```text
data/output/error_analysis_samples.csv
data/output/human_evaluation_template.csv
```

## Template Laporan Eksperimen UAS

Setelah pipeline dijalankan dan `data/output/hasil_ringkasan_hadis.csv` tersedia, buat template laporan eksperimen:

```powershell
python generate_experiment_report.py
```

Script ini membuat:

```text
data/output/laporan_eksperimen_uas.md
data/output/contoh_output_ringkasan.md
data/output/metrics_eksperimen.json
```

File tersebut diisi dari output nyata pipeline. Jika output pipeline belum ada, script akan berhenti dengan pesan error dan tidak membuat angka palsu.

Gunakan file ini untuk mengisi bagian UAS:

- smoke test pipeline dengan dataset asli;
- hasil eksperimen;
- nilai ROUGE/BERTScore jika ada `ringkasan_referensi`;
- contoh output ringkasan nyata;
- kesimpulan kualitas model setelah dicek manusia.

## Jalur Google Colab

Gunakan notebook baru:

```text
notebooks/hadith_summarization_colab.ipynb
```

Alur Colab:

1. Upload notebook ke Google Colab.
2. Jalankan install dependency di runtime Colab.
3. Upload `dataset_hadis.csv`.
4. Jalankan inference abstractive.
5. Download `hasil_ringkasan_hadis_colab.csv`.
6. Pindahkan file itu ke `data/output/` di laptop.
7. Jalankan `python run_light_pipeline.py` untuk menggabungkan hasil.

Notebook Colab memakai:

```text
csebuetnlp/mT5_multilingual_XLSum
```

Model dipakai untuk inference abstractive, bukan fine-tuning.

## Pipeline Penuh Lokal

Jika laptop kuat atau memiliki GPU dan dependency berat sudah terpasang:

```powershell
pip install -r requirements-local.txt
pip install -r requirements-colab.txt
python run_pipeline.py
```

Secara default `LIMIT_ROWS = 10` di `src/config.py` untuk smoke test.

Untuk memproses seluruh data:

```python
LIMIT_ROWS = None
```

Jika tidak ingin menjalankan model berat di lokal:

```python
RUN_ABSTRACTIVE = False
RUN_EVALUATION = False
```

## Demo Gradio

Gradio bersifat opsional dan tidak dibutuhkan oleh pipeline utama.

```powershell
python app.py
```

Tab utama **Katalog Hadis**: pilih book lalu pilih hadis, detail langsung tampil (teks Arab, terjemahan, ringkasan, status, disclaimer) tanpa memanggil model apa pun saat diklik:

- Hadis yang sudah diproses `run_pipeline.py` menampilkan Extractive TF-IDF (ringkasan utama) dan mT5 Abstractive (eksperimen pembanding, dengan peringatan hallucination) langsung dari `data/output/hasil_ringkasan_hadis.csv`.
- Hadis yang belum diproses pipeline (ditandai **"(belum diringkas)"** di dropdown) tetap bisa dipilih dari seluruh dataset hasil scraping (`data/output/dataset_hadis_full.csv`); Extractive-nya dihitung langsung di aplikasi (ringan, TF-IDF, tanpa model AI berat), sementara Abstractive tetap menunggu `run_pipeline.py` karena mT5 sengaja tidak dijalankan otomatis di UI.

Tab kedua **Ringkas Manual** (opsional): tempel teks terjemahan sendiri, pilih metode extractive/abstractive/keduanya, lihat jumlah kata, compression ratio, status model, dan peringatan bahwa hasil AI perlu diperiksa manusia.

## Model

Model abstractive default:

```text
csebuetnlp/mT5_multilingual_XLSum
```

Model ini digunakan melalui Hugging Face `AutoTokenizer` dan `AutoModelForSeq2SeqLM`. Program tidak otomatis mengganti model jika loading gagal.

Baseline extractive memakai TF-IDF sentence scoring. Metode ini hanya memilih kalimat dari sumber dan tidak menambahkan informasi baru.

## Evaluasi

Jika `ringkasan_referensi` tersedia dan berisi ringkasan valid dari manusia, program mencoba menghitung ROUGE dan BERTScore.

Jika referensi tidak tersedia, program hanya menghitung statistik deskriptif seperti panjang ringkasan, compression ratio, repetition rate, dan novel n-gram ratio. Statistik ini bukan ukuran kebenaran makna hadis.

## Testing

```powershell
python -m pytest
```

Unit test tidak mengunduh model besar. Kegagalan load model diuji dengan mocking.

## Deploy ke Hugging Face Spaces

Space memakai SDK Gradio (metadata di bagian atas file ini) dan `requirements.txt` di root folder ini — sengaja **ringan** (tanpa `torch`/`transformers`) supaya build cepat. Katalog Hadis tetap berfungsi penuh (Extractive selalu tersedia, live-computed bila perlu); Ringkas Manual mode Abstractive akan menampilkan status `model_load_error` yang jelas karena mT5 memang tidak disertakan di Space ini.

Yang **wajib** ikut ter-upload (sudah diatur di `.gitignore` supaya tidak ke-exclude bila deploy lewat `git push`):

- `app.py`, `requirements.txt`, `src/`
- `data/output/hasil_ringkasan_hadis.csv`
- `data/output/dataset_hadis_full.csv`
- `data/output/error_analysis_samples.csv`

Tanpa ketiga CSV di atas, Katalog Hadis akan menampilkan pesan "File hasil_ringkasan_hadis.csv belum ditemukan."

Yang **tidak perlu** ikut ter-upload (sudah di-gitignore): `*.backup_*`, `data/cache/`, file CSV backup/partial lain di `data/output/` dan `data/input/`, `.venv/`, `.pytest_cache/`, `.gradio/`, `models/` (kosong, model diunduh runtime kalau dipakai).

Dua cara upload:

1. **Web UI (tanpa git)** — buat Space baru di huggingface.co/new-space (SDK: Gradio), lalu drag-and-drop isi folder ini (skip folder/file yang di atas).
2. **Git** — `git init` di folder `hadith_summarization/`, `git remote add origin <URL Space>`, lalu `git push`. `.gitignore` di folder ini sudah menyaring file yang tidak perlu.

## Batasan

- Sistem ini tidak memberi fatwa, syarah, atau penilaian agama.
- Ringkasan AI perlu diperiksa manusia.
- Evaluasi otomatis tidak dapat memastikan kesesuaian makna hadis.
- Kualitas output bergantung pada kualitas terjemahan Indonesia pada dataset.
- Proyek ini tidak melakukan scraping atau penerjemahan ulang jika dataset terjemahan sudah tersedia.
- Jangan mengklaim hasil eksperimen jika pipeline belum benar-benar dijalankan pada dataset final.
