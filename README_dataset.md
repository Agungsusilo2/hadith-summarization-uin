# Dataset Builder — Sahih al-Bukhari (Sunnah.com)

Dokumen ini menjelaskan **dataset builder** (`create_hadith_dataset.py` dan
`notebooks/create_hadith_dataset_colab.ipynb`), bukan pipeline summarization
utama. Dataset builder ini adalah alat terpisah untuk menghasilkan dataset
mentah hadis yang kompatibel dengan pipeline yang sudah ada di
`run_pipeline.py` / `run_light_pipeline.py` / `app.py`. Pipeline utama, folder
`src/`, dan `tests/` tidak diubah oleh alat ini.

## 1. Tujuan Dataset

Dataset ini dibuat untuk keperluan tugas akademik (UAS mata kuliah NLP UIN
Jakarta) berjudul *"Ringkasan Otomatis Kitab Hadis Berbasis AI sebagai Media
Bantu Pembelajaran Mahasiswa UIN Jakarta"*. Dataset menyediakan pasangan teks
Arab, teks Inggris, dan terjemahan Indonesia dari hadis Sahih al-Bukhari yang
bisa dipakai sebagai input pipeline ringkasan otomatis.

## 2. Sumber Data

Dataset ini merupakan dataset turunan dari sumber publik Sunnah.com pada
koleksi Sahih al-Bukhari.

Halaman yang dipakai:

- Indeks book: `https://sunnah.com/bukhari`
- Halaman per book: `https://sunnah.com/bukhari/{book_number}`

Data **tidak** diambil dari dataset Liputan6, XL-Sum, AraBART, PEGASUS, atau
dataset dari paper referensi manapun. Paper hanya dipakai sebagai dasar
metode summarization, bukan sumber data hadis.

## 3. Alasan Memilih Sunnah.com Sahih al-Bukhari

- Sunnah.com adalah sumber referensi hadis publik yang umum dipakai dan
  menyediakan teks Arab, terjemahan Inggris, dan metadata referensi
  (in-book reference) secara terstruktur per hadis.
- Sahih al-Bukhari adalah salah satu kitab hadis paling otoritatif dan
  banyak dipakai sebagai bahan ajar, sehingga relevan untuk media bantu
  pembelajaran mahasiswa.

Catatan: Sunnah.com adalah sumber **publik**, bukan otomatis berarti
**open-source**. Lihat bagian [Batasan Lisensi](#12-batasan-lisensi).

## 4. Struktur Output

```text
hadith_summarization/
├── create_hadith_dataset.py
├── requirements-dataset.txt
├── README_dataset.md
├── notebooks/
│   └── create_hadith_dataset_colab.ipynb
└── data/
    ├── cache/
    │   ├── html/                      # cache HTML mentah per halaman
    │   └── json/                      # cache hasil parsing per book
    ├── output/
    │   ├── dataset_hadis_full.csv
    │   ├── dataset_hadis_minimal.csv
    │   ├── ringkasan_referensi_template.csv
    │   └── scrape_progress.csv
    └── input/
        └── dataset_hadis.csv          # dibuat manual, atau otomatis jika COPY_MINIMAL_TO_INPUT=True
```

## 5. Arti Setiap Kolom

### `dataset_hadis_full.csv`

| Kolom | Arti |
|---|---|
| `Num_hadith` | Nomor urut baris pada hasil scraping (bukan nomor hadis resmi). |
| `source_ref` | Referensi sumber, contoh `"Sahih al-Bukhari 1"`. |
| `in_book_reference` | Referensi dalam kitab, contoh `"Book 1, Hadith 1"`. |
| `book_number` | Nomor book/kitab pada Sunnah.com. |
| `book_title` | Judul book dalam bahasa Inggris, contoh `"Revelation"`. |
| `hadith_number` | Nomor hadis global pada koleksi Sahih al-Bukhari. |
| `teks_arab` | Teks Arab hadis asli dari Sunnah.com. |
| `english_translation` | Terjemahan Inggris asli dari Sunnah.com. |
| `terjemahan` | Terjemahan Indonesia hasil model `Helsinki-NLP/opus-mt-en-id`. |
| `ringkasan_referensi` | Sengaja dikosongkan. Lihat [poin 14](#14-alasan-ringkasan_referensi-kosong). |
| `data_status` | Status validasi: `valid`, `missing_arabic`, `missing_english`, `duplicate_ref`, `too_short`, `parse_error`. |
| `data_warning` | Penjelasan singkat jika ada masalah pada baris tersebut. |
| `translation_status` | Lihat [bagian 5a](#5a-arti-translation_status-dan-translation_warning) di bawah. |
| `translation_warning` | Alasan spesifik kenapa baris ditandai `needs_review` (kosong jika tidak ada masalah terdeteksi). |
| `translation_error` | Pesan error jika penerjemahan baris tersebut gagal total (exception). |
| `translation_model` | Nama model yang dipakai untuk menerjemahkan baris tersebut. |

### 5a. Arti `translation_status` dan `translation_warning`

`english_translation` sering berupa paragraf panjang multi-kalimat, sedangkan
`Helsinki-NLP/opus-mt-en-id` adalah model MarianMT yang dilatih untuk
**kalimat pendek**, bukan paragraf. Karena itu teks panjang dipecah dulu
menjadi potongan kalimat sebelum diterjemahkan (bukan dipotong/`truncate`
diam-diam), lalu hasil potongan digabung kembali. Status berikut mencatat
seberapa jauh proses itu terpaksa "mengorbankan" konteks:

| `translation_status` | Arti |
|---|---|
| `ok` | Teks pendek, diterjemahkan dalam satu kali panggilan model tanpa pemecahan, dan lolos semua pemeriksaan kualitas di bawah. |
| `chunked_ok` | Teks dipecah **per kalimat utuh** (bukan per potongan kalimat), setiap kalimat tetap punya konteks gramatikal lengkap, dan lolos semua pemeriksaan kualitas. |
| `needs_review` | **Wajib dicek manusia sebelum dipakai.** Dipicu oleh salah satu atau lebih dari: <br>• ada kalimat yang saking panjangnya terpaksa dipotong per-klausa (bukan per kalimat utuh) — konteks gramatikal sebagian hilang;<br>• rasio panjang `terjemahan`/`english_translation` terlalu rendah (`< 0.35`) — indikasi output terlalu pendek/kehilangan isi;<br>• output kosong setelah diterjemahkan;<br>• output tampak repetitif (kata yang sama diulang berlebihan) — indikasi keluaran model rusak;<br>• muncul nama tokoh atau istilah nahwu/tafsir Arab klasik (lihat `SUSPICIOUS_LEAKED_TERMS` di kode) yang **tidak ada** di `english_translation` maupun `teks_arab` — indikasi model "membocorkan" hafalan dari data latihnya (mis. cerita Nabi lain) alih-alih menerjemahkan isi hadis yang sebenarnya;<br>• ditemukan kata berhuruf kapital di tengah kalimat yang tidak ada di sumber sama sekali (kemungkinan nama asing yang dikarang model). |
| `empty_source` | `english_translation` kosong, tidak ada yang diterjemahkan. |
| `skipped_existing` | Baris sudah punya `terjemahan` valid dari checkpoint/run sebelumnya, tidak diterjemahkan ulang. |
| `failed` | Proses terjemahan gagal total (exception saat memanggil model). Lihat `translation_error`. |
| `disabled` | `RUN_TRANSLATION = False`, kolom `terjemahan` sengaja dikosongkan. |

**Penting — `chunked_ok`/`ok` bukan jaminan benar secara agama/makna.**
Status ini hanya menandakan proses mekanis (pemecahan kalimat, panjang
output, deteksi nama nyasar) tidak menemukan masalah yang bisa dideteksi
otomatis. Ini **bukan** validasi kebenaran makna, istilah fiqih, atau
kesesuaian dengan hadis aslinya — itu hanya bisa dinilai manusia yang paham
konteks keagamaan. Bahkan pada uji coba 100 hadis pertama, sebagian baris
berstatus `chunked_ok` masih berpotensi memuat penyimpangan makna yang tidak
tertangkap oleh pemeriksaan otomatis di atas. **Jangan pakai kolom
`terjemahan` mentah-mentah untuk evaluasi akademik (ROUGE/BERTScore/analisis
kualitas) tanpa sampling manual terlebih dahulu** — idealnya sampel acak dari
setiap status (termasuk `ok` dan `chunked_ok`, tidak hanya `needs_review`)
diperiksa oleh yang memahami hadis sebelum dataset dipakai sebagai rujukan.

### `dataset_hadis_minimal.csv`

`Num_hadith`, `teks_arab`, `terjemahan`, `ringkasan_referensi` — kolom ini
sengaja dibuat sama persis dengan kebutuhan `src/config.py` pada pipeline
utama (`HADITH_ID_COLUMN`, `ARABIC_TEXT_COLUMN`, `INDONESIAN_TEXT_COLUMN`,
`REFERENCE_SUMMARY_COLUMN`).

### `ringkasan_referensi_template.csv`

`Num_hadith`, `source_ref`, `teks_arab`, `terjemahan`, `ringkasan_referensi`
(kosong), `validator` (kosong, diisi nama pemvalidasi manusia),
`catatan_validasi` (kosong, diisi catatan pemvalidasi).

### `scrape_progress.csv`

`book_number`, `book_title`, `book_url`, `hadith_count`, `status`
(`success`/`cached`/`failed`), `error_message`, `timestamp`. Dipakai untuk
memantau progres scraping dan menghindari fetch ulang book yang sudah
berhasil.

## 6. Cara Menjalankan di Laptop/PyCharm

```powershell
cd C:\Users\agung\Downloads\NLP\hadith_summarization
python -m venv .venv        # jika belum ada
.venv\Scripts\activate
pip install -r requirements-dataset.txt
python create_hadith_dataset.py
```

Dengan konfigurasi default, script hanya mengambil sekitar 10 hadis pertama
dari 1 book (smoke test), lalu langsung menerjemahkannya ke Indonesia.

## 7. Cara Menjalankan di Google Colab

1. Upload `notebooks/create_hadith_dataset_colab.ipynb` ke Google Colab.
2. Jalankan sel install dependency.
3. Jalankan seluruh sel secara berurutan.
4. `USE_GOOGLE_DRIVE = False` secara default — output disimpan ke runtime
   Colab (`data/output/` relatif terhadap notebook), lalu bisa didownload
   manual. Jika ingin menyimpan ke Google Drive, ubah menjadi `True` dan
   jalankan sel mount Drive.
5. Download hasil (`dataset_hadis_minimal.csv`, dll) dan pindahkan ke laptop
   jika diperlukan.

## 8. Cara Membatasi Jumlah Hadis

Ubah di `create_hadith_dataset.py` (atau sel konfigurasi di notebook):

```python
RUN_FULL_SCRAPE = False
LIMIT_BOOKS = 1        # jumlah book yang diambil
LIMIT_HADITHS = 10     # jumlah hadis maksimum yang diambil
```

## 9. Cara Menjalankan Full Scrape

```python
RUN_FULL_SCRAPE = True
```

Full scrape akan mengambil seluruh 97 book Sahih al-Bukhari dari Sunnah.com.
Proses ini memakan waktu lama karena ada jeda (`REQUEST_DELAY_SECONDS`)
antar request agar tidak membebani server Sunnah.com. Jalankan hanya jika
memang dibutuhkan.

## 10. Cara Mengaktifkan/Menonaktifkan Translation

```python
RUN_TRANSLATION = True   # terjemahkan english_translation -> Indonesia
RUN_TRANSLATION = False  # lewati translation, kolom terjemahan kosong,
                          # translation_status = "disabled"
```

## 11. Cara Memakai `dataset_hadis_minimal.csv` ke Pipeline Lama

Dataset builder **tidak** otomatis menaruh hasil ke `data/input/` (kecuali
`COPY_MINIMAL_TO_INPUT = True`). Langkah manual:

1. Jalankan `create_hadith_dataset.py` hingga `data/output/dataset_hadis_minimal.csv` terbentuk.
2. Salin/rename file tersebut secara manual menjadi:

   ```text
   data/input/dataset_hadis.csv
   ```

3. Jalankan pipeline seperti biasa, misalnya:

   ```powershell
   python run_light_pipeline.py
   ```

Kolom `Num_hadith`, `teks_arab`, `terjemahan` pada dataset minimal sudah
sesuai dengan `REQUIRED_COLUMNS` di `src/config.py`.

Jika ingin proses ini otomatis, ubah:

```python
COPY_MINIMAL_TO_INPUT = True
```

Defaultnya `False` agar tidak ada risiko file input pipeline tertimpa tanpa
sepengetahuan pengguna. Jika file input sudah ada dan opsi ini diaktifkan,
script tetap membuat backup bertimestamp sebelum menimpanya.

## 12. Batasan Lisensi

Dataset ini merupakan dataset turunan dari sumber publik Sunnah.com. Data
digunakan untuk kepentingan akademik/UAS. Pengguna perlu memeriksa kembali
ketentuan penggunaan sumber asli sebelum menyebarluaskan dataset hasil
olahan.

Sunnah.com bersifat publik untuk dibaca, tetapi ini **bukan** pernyataan
bahwa datanya berlisensi open-source atau bebas dipakai ulang tanpa batas.
Jangan mendistribusikan ulang dataset hasil scraping ini sebagai dataset
resmi/independen tanpa mencantumkan sumber aslinya.

## 13. Batasan Kualitas Terjemahan Mesin

Kolom `terjemahan` dihasilkan oleh model mesin `Helsinki-NLP/opus-mt-en-id`
dari teks `english_translation`, bukan dari teks Arab langsung. Artinya
kualitas terjemahan Indonesia bergantung pada:

- akurasi terjemahan Inggris asli dari Sunnah.com, dan
- kualitas model `opus-mt-en-id` untuk domain teks keagamaan, yang tidak
  dilatih khusus untuk teks hadis. Model ini teramati kadang "membocorkan"
  nama tokoh atau istilah tafsir/nahwu dari data latihnya sendiri saat
  menerjemahkan kalimat panjang yang terpaksa dipecah per-klausa (lihat
  [bagian 5a](#5a-arti-translation_status-dan-translation_warning)).

Terjemahan mesin ini **tidak boleh dianggap sebagai terjemahan resmi atau
tafsir**. Selalu periksa ulang oleh yang memahami konteks keagamaan sebelum
dipakai di luar keperluan eksperimen NLP — gunakan kolom `translation_status`
dan `translation_warning` sebagai titik awal untuk memprioritaskan baris mana
yang paling butuh dicek, tapi jangan berhenti di situ: status `ok`/`chunked_ok`
tetap bukan jaminan kebenaran makna.

## 14. Alasan `ringkasan_referensi` Kosong

`ringkasan_referensi` sengaja dikosongkan dan **tidak** diisi otomatis oleh
model AI mana pun. Ringkasan yang dihasilkan model tidak boleh dianggap
sebagai *ground truth* untuk evaluasi. Evaluasi ROUGE/BERTScore yang sah
membutuhkan ringkasan referensi yang dibuat atau divalidasi manusia yang
memahami hadis, bukan ringkasan buatan mesin.

## 15. Cara Mengisi `ringkasan_referensi` Manual

1. Buka `data/output/ringkasan_referensi_template.csv`.
2. Untuk setiap baris, isi kolom `ringkasan_referensi` dengan ringkasan yang
   dibuat atau divalidasi manusia (idealnya oleh yang memiliki pemahaman
   keagamaan/hadis yang memadai).
3. Isi kolom `validator` dengan nama pembuat/pemvalidasi ringkasan.
4. Isi kolom `catatan_validasi` jika ada catatan tambahan (misalnya sumber
   syarah yang dijadikan acuan, atau bagian yang masih meragukan).
5. Gunakan file yang sudah diisi ini sebagai referensi evaluasi pada
   pipeline utama, sesuai kolom `REFERENCE_SUMMARY_COLUMN` di
   `src/config.py`.

## 16. Peringatan

Hasil ringkasan otomatis dari pipeline AI (baik extractive maupun
abstractive) **bukan tafsir, bukan syarah, dan bukan fatwa**. Ringkasan AI
adalah alat bantu belajar, bukan sumber hukum atau penjelasan agama yang
otoritatif. Pengguna wajib memeriksa ulang hasil ringkasan dengan sumber
keagamaan yang sah sebelum menjadikannya rujukan.

---

Dataset ini merupakan dataset turunan dari sumber publik Sunnah.com. Data
digunakan untuk kepentingan akademik/UAS. Pengguna perlu memeriksa kembali
ketentuan penggunaan sumber asli sebelum menyebarluaskan dataset hasil
olahan.
