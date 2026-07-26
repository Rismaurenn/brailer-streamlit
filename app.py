import gc
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════
# CPU & Memory Optimization - Batasi thread untuk kurangi beban
# ═══════════════════════════════════════════════════════════════
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _lazy_import_cv2():
    """Lazy import OpenCV hanya saat dibutuhkan."""
    import cv2
    return cv2


def _lazy_import_streamlit():
    """Lazy import Streamlit."""
    import streamlit as st
    return st


# ═══════════════════════════════════════════════════════════════
# Fungsi utilitas gambar (menggunakan cv2 via lazy import)
# ═══════════════════════════════════════════════════════════════

def order_document_points(points):
    cv2 = _lazy_import_cv2()
    rect = np.zeros((4, 2), dtype="float32")
    points = points.astype("float32")
    point_sum = points.sum(axis=1)
    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]
    point_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]
    return rect


def four_point_transform(image, points):
    cv2 = _lazy_import_cv2()
    rect = order_document_points(points)
    top_left, top_right, bottom_right, bottom_left = rect
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_a, height_b))
    if max_width < 80 or max_height < 80:
        return image
    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def auto_straighten_document(image):
    """Auto-straightening: contour detection + Hough line fallback.
    
    Optimasi dari skripsi:
    - Tambah fallback pakai Hough Lines jika contour gagal
    - Threshold area diturunkan ke 6% (skripsi: gambar Braille sering tidak fill frame)
    - Validasi aspect ratio lebih longgar (skripsi: kertas B5/A4 ratio ~1.4-1.6)
    """
    cv2 = _lazy_import_cv2()
    if image is None or image.size == 0:
        return image
    original = image.copy()
    height, width = image.shape[:2]
    max_side = max(height, width)
    scale = 900.0 / max_side if max_side > 900 else 1.0
    if scale != 1.0:
        resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    if contours:
        image_area = resized.shape[0] * resized.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.06:  # Turunkan threshold (skripsi: gambar sering partial)
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
            else:
                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect).astype("float32")
            x, y, w, h = cv2.boundingRect(quad.astype("int32"))
            if w < resized.shape[1] * 0.2 or h < resized.shape[0] * 0.2:
                continue
            aspect_ratio = max(w / max(h, 1), h / max(w, 1))
            if aspect_ratio > 6.0:  # Longgar (skripsi: kertas bisa agak miring)
                continue
            if area > best_area:
                best_area = area
                best_quad = quad

    # Fallback: Hough Lines jika contour gagal mendeteksi 4 titik
    if best_quad is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)
        if lines is not None and len(lines) >= 4:
            best_quad = _hough_to_quad(lines, resized.shape)

    if best_quad is None:
        logger.info("Auto-straightening: tidak ada kandidat ditemukan, gunakan gambar asli")
        return original

    if scale != 1.0:
        best_quad = best_quad / scale
    corrected = four_point_transform(original, best_quad)
    corrected_area = corrected.shape[0] * corrected.shape[1]
    original_area = original.shape[0] * original.shape[1]
    if corrected_area < original_area * 0.10:
        return original
    logger.info("Auto-straightening: koreksi perspektif diterapkan")
    return corrected


def _hough_to_quad(lines, shape):
    """Konversi Hough lines → 4 titik sudut kandidat."""
    cv2 = _lazy_import_cv2()
    h, w = shape[:2]
    # Kumpulkan semua titik ujung garis
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    if len(endpoints) < 4:
        return None
    endpoints = np.array(endpoints, dtype="float32")
    # Cari 4 sudut: TL, TR, BR, BL
    point_sum = endpoints.sum(axis=1)
    point_diff = np.diff(endpoints, axis=1).flatten()
    tl = endpoints[np.argmin(point_sum)]
    br = endpoints[np.argmax(point_sum)]
    tr = endpoints[np.argmin(point_diff)]
    bl = endpoints[np.argmax(point_diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _preprocess_balanced(image):
    """Preprocessing balanced sesuai skripsi Table 4.5.
    
    Pipeline:
    1. Gaussian blur untuk noise reduction ringan
    2. Contrast/brightness adjustment (alpha=1.2, beta=20)
    3. CLAHE untuk konsistensi lighting
    """
    cv2 = _lazy_import_cv2()
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    adjusted = cv2.convertScaleAbs(denoised, alpha=1.2, beta=20)
    # CLAHE pada L channel LAB
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_image(image_bytes, enable_straightening=True):
    cv2 = _lazy_import_cv2()
    np_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        return None
    if enable_straightening:
        image = auto_straighten_document(image)
    return image


# ═══════════════════════════════════════════════════════════════
# Lazy classifier loader dengan cache & memory management
# ═══════════════════════════════════════════════════════════════

_classifier = None


def get_classifier():
    """Load classifier sekali saja dan cache di memory."""
    global _classifier
    if _classifier is None:
        from control.classify import BrailleClassifier
        _classifier = BrailleClassifier(
            model_path=str(BASE_DIR / "weights" / "cnn_v1.hdf5"),
            json_path=str(BASE_DIR / "utils" / "class_labels.json"),
            symbols_path=str(BASE_DIR / "utils" / "braille_symbols.json"),
            numbers_path=str(BASE_DIR / "utils" / "braille_numbers.json"),
            yolo_weight=str(BASE_DIR / "weights" / "yolov8_braille.pt"),
        )
    return _classifier


def main():
    st = _lazy_import_streamlit()
    cv2 = _lazy_import_cv2()

    # ═══════════════════════════════════════════════════════════
    # Page config - icon盲 (Braille unicode) untuk parent app
    # ═══════════════════════════════════════════════════════════
    st.set_page_config(
        page_title="Sistem Bantu Baca Braille",
        page_icon="盲",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ═══════════════════════════════════════════════════════════
    # Header sederhana untuk parent users (skripsi: UI harus intuitif)
    # ═══════════════════════════════════════════════════════════
    st.title("盲 Sistem Bantu Baca Braille")
    st.caption(
        "Untuk orang tua anak tunanetra SD Kelas 1 — "
        "unggah foto tulisan Braille, sistem akan menerjemahkan ke teks Latin."
    )

    # ═══════════════════════════════════════════════════════════
    # Tips singkat (skripsi: panduan pengambilan foto)
    # ═══════════════════════════════════════════════════════════
    with st.expander("📷 Tips Foto Braille yang Baik", expanded=False):
        st.markdown(
            "1. **Posisi kamera tegak lurus** di atas kertas Braille\n"
            "2. **Pencahayaan merata** — hindari bayangan dan lampu menyilang\n"
            "3. **Kertas rata** — tidak melipat atau menggulung\n"
            "4. **Fokus jelas** — pastikan titik-titik Braille terlihat tajam"
        )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Pilih gambar Braille",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Format: JPG, PNG, BMP, WebP",
        )
        enable_straightening = st.checkbox("Luruskan otomatis jika kertas miring", value=True)

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = process_image(image_bytes, enable_straightening)

        if image is None:
            st.error("❌ Gambar tidak dapat dibaca. Coba foto lain yang lebih jelas.")
            return

        with col1:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Foto yang akan diterjemahkan",
                use_column_width=True,
            )

        if st.button("🔍 Terjemahkan", type="primary", use_container_width=True):
            with st.spinner("Membaca Braille... mohon tunggu"):
                try:
                    # Simpan ke temporary file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        cv2.imwrite(tmp.name, image)
                        tmp_path = tmp.name

                    # Bersihkan memory sebelum inference
                    gc.collect()

                    classifier = get_classifier()
                    result = classifier.recognize_braille(tmp_path)

                    if len(result) == 6:
                        predicted_image, character_result, syllable_result, speech_text, character_cells, syllable_cells = result
                    else:
                        predicted_image, syllable_result, speech_text, syllable_cells = result
                        character_result = syllable_result
                        character_cells = syllable_cells

                    # Hapus temporary file
                    Path(tmp_path).unlink(missing_ok=True)

                    if predicted_image is None:
                        st.error("❌ Tidak ada titik Braille yang terdeteksi. Coba foto dengan kualitas lebih baik.")
                        return

                    with col2:
                        st.image(
                            cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB),
                            caption="Hasil Deteksi Braille",
                            use_column_width=True,
                        )

                    # ═══════════════════════════════════════════
                    # Hasil terjemahan - tab sederhana
                    # ═══════════════════════════════════════════
                    st.markdown("---")
                    st.subheader("📖 Hasil Terjemahan")

                    tab_syllable, tab_character = st.tabs(["Suku Kata", "Huruf A-Z"])

                    with tab_syllable:
                        st.text_area(
                            "Terjemahan per Suku Kata",
                            syllable_result,
                            height=180,
                            disabled=True,
                            key="syllable_result",
                        )

                    with tab_character:
                        st.text_area(
                            "Terjemahan per Huruf",
                            character_result,
                            height=180,
                            disabled=True,
                            key="character_result",
                        )

                    # ═══════════════════════════════════════════
                    # Naskah suara pembelajaran (skripsi: fitur TTS)
                    # ═══════════════════════════════════════════
                    if speech_text:
                        st.markdown("---")
                        st.subheader("🗣️ Panduan Pelafalan")
                        st.caption("Setiap suku kata dipecah untuk membantu pengucapan")
                        st.text_area(
                            "Pola Suku Kata",
                            speech_text,
                            height=200,
                            disabled=True,
                            key="speech_text",
                        )

                    # Bersihkan memory setelah selesai
                    del predicted_image
                    gc.collect()
                    logger.info("Terjemahan selesai: %d karakter", len(syllable_result))

                except Exception as exc:
                    logger.error("Gagal memproses: %s", exc, exc_info=True)
                    st.error(f"❌ Gagal memproses gambar: {exc}")
                    st.info("Pastikan gambar jelas, kertas rata, dan pencahayaan cukup.")
                    gc.collect()

    with col2:
        if uploaded_file is None:
            st.info(
                "📷 Upload gambar Braille di panel sebelah kiri untuk memulai.\n\n"
                "Sistem akan mendeteksi titik-titik Braille dan menerjemahkannya "
                "menjadi teks Latin yang bisa dibaca."
            )


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
