# ═══════════════════════════════════════════════════════════════
# PANDUAN DEPLOY KE SNOWFLAKE WEB UI
# (Streamlit in Snowflake)
# ═══════════════════════════════════════════════════════════════

## LANGKAH 1: Setup External Access Integration (EAI)
## ═══════════════════════════════════════════════════════════════

1. Buka Snowflake Web UI
2. Klik "Worksheets" di sidebar kiri
3. Klik "+ Worksheet" untuk buat worksheet baru
4. Paste SQL berikut dan klik "Run" (atau tekan Ctrl+Enter):

```sql
-- Setup EAI agar pip bisa download package dari PyPI
CREATE NETWORK RULE pypi_network_rule
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'pypi.org:443',
        'pypi.python.org:443',
        'files.pythonhosted.org:443'
    );

CREATE EXTERNAL ACCESS INTEGRATION pypi_eai
    ALLOWED_NETWORK_RULES = (pypi_network_rule)
    ENABLED = TRUE;

-- Grant ke role yang kamu pakai
-- Ganti 'SYSADMIN' jika role kamu berbeda
GRANT USAGE ON INTEGRATION pypi_eai TO ROLE SYSADMIN;

-- Verifikasi
SHOW EXTERNAL ACCESS INTEGRATIONS;
```

5. Pastikan tidak ada error. Jika berhasil, EAI sudah aktif.


## LANGKAH 2: Buat Streamlit App
## ═══════════════════════════════════════════════════════════════

1. Klik "Streamlit" di sidebar kiri
2. Klik "+ Streamlit App"
3. Pilih "Create in a new app"
4. Isi form:
   - App name: braille_to_latin
   - Database: pilih database kamu (atau buat baru)
   - Schema: pilih schema (atau public)
   - Warehouse: pilih warehouse (X-Small atau Small)
5. Klik "Create" / "Create App"


## LANGKAH 3: Tambah Code App
## ═══════════════════════════════════════════════════════════════

1. Setelah app terbuat, kamu akan melihat code editor
2. Di sebelah kiri ada file explorer
3. Hapus code default di app.py
4. Buka file app.py lokal kamu, copy SELURUH isi file
5. Paste ke editor di Snowflake Web UI
6. Save (Ctrl+S)


## LANGKAH 4: Tambah Requirements
## ═══════════════════════════════════════════════════════════════

1. Di file explorer (kiri), klik tombol "+" atau "Create file"
2. Buat file baru bernama: requirements.txt
3. Paste isi berikut:

```
streamlit>=1.28,<2.0
opencv-python-headless>=4.7,<5.0
Pillow>=9.4,<11.0
numpy>=1.23,<2.0
tensorflow-cpu>=2.11,<3.0
keras>=2.11,<4.0
ultralytics>=8.0,<9.0
torch>=2.0,<3.0
torchvision>=0.15,<1.0
```

4. Save (Ctrl+S)


## LANGKAH 5: Buat Folder & File control/
## ═══════════════════════════════════════════════════════════════

Buat folder dan file satu per satu:

1. Klik "+" → "Create folder" → nama: control
2. Buka folder control/
3. Buat file: __init__.py (kosong)
4. Buat file: classify.py → paste isi dari file lokal
5. Buat file: segmentation.py → paste isi dari file lokal
6. Buat file: convert.py → paste isi dari file lokal


## LANGKAH 6: Buat Folder & File utils/
## ═══════════════════════════════════════════════════════════════

1. Klik "+" → "Create folder" → nama: utils
2. Buka folder utils/
3. Buat file: __init__.py (kosong)
4. Buat file: class_labels.json → paste isi dari file lokal
5. Buat file: braille_symbols.json → paste isi dari file lokal
6. Buat file: braille_numbers.json → paste isi dari file lokal


## LANGKAH 7: Upload Model Weights
## ═══════════════════════════════════════════════════════════════

⚠️ PENTING: Model weights harus di-upload!

1. Klik "+" → "Create folder" → nama: weights
2. Buka folder weights/
3. Upload file dari komputer kamu:
   - cnn_v1.hdf5 (2.81 MB)
   - yolov8_braille.pt (49.68 MB)

Cara upload:
- Di file explorer, klik kanan pada folder weights/
- Pilih "Upload" atau klik tombol upload
- Pilih file dari komputer kamu


## LANGKAH 8: Deploy & Test
## ═══════════════════════════════════════════════════════════════

1. Klik tombol "Run" atau "Deploy" di pojok kanan atas
2. Tunggu sampai selesai (biasanya 1-3 menit untuk install packages)
3. Jika berhasil, app akan terbuka
4. Upload gambar Braille untuk test


## TROUBLESHOOTING
## ═══════════════════════════════════════════════════════════════

### Error: "Failed to retrieve packages"
→ EAI belum aktif. Ulangi Langkah 1.

### Error: "No module named 'cv2'"
→ opencv-python-headless butuh libGL.so.1 yang tidak tersedia di SiS.
   → Solusi: Ganti ke Snowflake Container Services (pakai Docker)

### Error: "FileNotFoundError: weights/..."
→ Model weights belum di-upload. Ulangi Langkah 7.

### Error: Memory/Runtime
→ App terlalu berat untuk SiS. Pertimbangkan:
   → Pakai warehouse yang lebih besar (MEDIUM/LARGE)
   → Atau deploy ke Render/Streamlit Cloud pakai Docker


## STRUKTUR FILE DI SNOWFLAKE
## ═══════════════════════════════════════════════════════════════

```
app.py                    ← Main Streamlit app
requirements.txt          ← Python dependencies
control/
  __init__.py             ← (kosong)
  classify.py             ← Braille classifier
  segmentation.py         ← YOLO segmentation
  convert.py              ← Utility functions
utils/
  __init__.py             ← (kosong)
  class_labels.json       ← Label mapping
  braille_symbols.json    ← Symbol mapping
  braille_numbers.json    ← Number mapping
weights/
  cnn_v1.hdf5             ← CNN model (2.81 MB)
  yolov8_braille.pt       ← YOLO model (49.68 MB)
```
