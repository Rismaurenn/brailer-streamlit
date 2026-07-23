# Sistem Bantu Baca Braille - Streamlit

Aplikasi terjemahan Braille ke Latin menggunakan **CNN** dan **YOLO**, dibangun dengan **Streamlit**.

## Struktur Folder

```
├── app.py                    # Aplikasi Streamlit utama
├── control/
│   ├── classify.py           # Klasifikasi CNN (arsitektur model)
│   ├── convert.py            # Parsing bounding box YOLO
│   └── segmentation.py       # Segmentasi Braille via YOLO
├── utils/
│   ├── class_labels.json     # Label kelas Braille
│   ├── braille_symbols.json  # Peta simbol Braille
│   ├── braille_numbers.json  # Peta angka Braille
│   └── braille_to_alphabet.json
├── weights/                  # Tempatkan model di sini
├── requirements.txt
└── .gitignore
```

## Persiapan Lokal

### 1. Salin model weights

Copy file model ke folder `weights/`:

- `weights/cnn_v1.hdf5` — model CNN untuk klasifikasi Braille
- `weights/yolov8_braille.pt` — model YOLO untuk segmentasi

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan

```bash
streamlit run app.py
```

Akses di `http://localhost:8501`

---

## Deploy ke Streamlit Cloud (Gratis)

### 1. Push ke GitHub

Buat repository GitHub dan push folder ini sebagai root repository.

### 2. Login ke Streamlit Cloud

Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan GitHub.

### 3. Deploy Aplikasi

| Langkah | Keterangan |
|---------|------------|
| **Klik "New app"** | Pilih repository yang sudah di-push |
| **Branch** | `main` (atau branch yang digunakan) |
| **Main file path** | `app.py` |
| **Klik "Deploy"** | Tunggu build selesai |

### 4. Konfigurasi (jika perlu)

Di **Advanced Settings**, tidak perlu konfigurasi tambahan. Semua dependency sudah di `requirements.txt`.

### Catatan Penting

- **Model weights** akan terupload via Git ke repository
- Streamlit Cloud **free plan** memberikan:
  - 1 aplikasi public
  - CPU 1 core
  - RAM 1 GB
  - Sleep setelah 7 hari tidak dipakai
- Jika butuh performa lebih, upgrade ke **Streamlit Team** atau hosting di **Hugging Face Spaces**

---

## Deploy Alternatif: Hugging Face Spaces

1. Buat Space di [huggingface.co/spaces](https://huggingface.co/spaces)
2. Pilih **SDK: Streamlit**
3. Push folder ini ke repository Space
4. Aplikasi akan otomatis build dan deploy

---

## Catatan Skripsi

- Arsitektur model CNN tidak diubah dari versi asli
- Pipeline: Image → YOLO Segmentation → CNN Classification → Syllable Assembly
- Auto-straightening dokumen menggunakan perspective transform
- Model di-cache dengan `@st.cache_resource` agar tidak reload setiap interaksi