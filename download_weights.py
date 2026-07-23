"""
Script untuk download weights file yang dibutuhkan project.
Digunakan sebagai fallback jika weights tidak tersedia di repository.

Cara pakai:
    python download_weights.py

Atau bisa juga dipanggil dari Dockerfile:
    RUN python download_weights.py
"""
import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Daftar weights yang dibutuhkan dengan URL download
# Ganti URL ini dengan lokasi hosting weights Anda
WEIGHTS_CONFIG = {
    "cnn_v1.hdf5": {
        # Contoh: Google Drive direct link
        # "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
        "url": None,  # Set None jika file sudah ada di repo
        "expected_size_mb": 2.81,
    },
    "yolov8_braille.pt": {
        "url": None,  # Set None jika file sudah ada di repo
        "expected_size_mb": 49.68,
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download file dari URL ke destination path."""
    try:
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  Gagal download {dest.name}: {e}")
        return False


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    all_present = True
    for filename, config in WEIGHTS_CONFIG.items():
        filepath = WEIGHTS_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} sudah ada ({size_mb:.2f} MB)")
        else:
            all_present = False
            if config["url"]:
                print(f"  ⏳ {filename} belum ada, mencoba download...")
                if download_file(config["url"], filepath):
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {filename} berhasil di-download ({size_mb:.2f} MB)")
                else:
                    print(f"  ✗ Gagal mendownload {filename}")
                    print(f"    Silakan download manual dan letakkan di {WEIGHTS_DIR}")
                    sys.exit(1)
            else:
                print(f"  ✗ {filename} belum ada dan tidak ada URL download.")
                print(f"    Silakan letakkan file di {WEIGHTS_DIR}")
                sys.exit(1)

    if all_present:
        print("\nSemua weights file tersedia! ✓")
    else:
        print("\nDownload selesai.")


if __name__ == "__main__":
    main()
