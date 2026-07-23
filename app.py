import gc
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Suppress TensorFlow dan OpenCV warnings untuk kurangi noise
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only untuk kurangi memory

BASE_DIR = Path(__file__).resolve().parent


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
    if not contours:
        return original
    image_area = resized.shape[0] * resized.shape[1]
    best_quad = None
    best_area = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype("float32")
        x, y, w, h = cv2.boundingRect(quad.astype("int32"))
        if w < resized.shape[1] * 0.25 or h < resized.shape[0] * 0.25:
            continue
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        if aspect_ratio > 5.0:
            continue
        if area > best_area:
            best_area = area
            best_quad = quad
    if best_quad is None:
        return original
    if scale != 1.0:
        best_quad = best_quad / scale
    corrected = four_point_transform(original, best_quad)
    corrected_area = corrected.shape[0] * corrected.shape[1]
    original_area = original.shape[0] * original.shape[1]
    if corrected_area < original_area * 0.12:
        return original
    return corrected


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
    # Page config harus dipanggil pertama kali
    # ═══════════════════════════════════════════════════════════
    st.set_page_config(page_title="Sistem Bantu Baca Braille", page_icon="盲", layout="wide")

    st.title("Sistem Bantu Baca Braille")
    st.write("Upload gambar Braille untuk diterjemahkan.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Pilih gambar Braille", type=["jpg", "jpeg", "png", "bmp", "webp"])
        enable_straightening = st.checkbox("Luruskan foto otomatis jika kertas miring", value=True)

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = process_image(image_bytes, enable_straightening)

        if image is None:
            st.error("Gambar tidak dapat dibaca. Coba gambar lain.")
            return

        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Gambar yang diproses", use_column_width=True)

        if st.button("Terjemahkan", type="primary"):
            with st.spinner("Memproses gambar..."):
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
                        st.error("Gambar tidak dapat diproses. Coba foto Braille yang lebih jelas.")
                        return

                    with col2:
                        st.image(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB), caption="Gambar Hasil Deteksi", use_column_width=True)

                    st.subheader("Hasil Terjemahan")

                    tab_syllable, tab_character = st.tabs(["Suku Kata", "Karakter A-Z"])

                    with tab_syllable:
                        st.text_area("Hasil Suku Kata", syllable_result, height=150, disabled=True)

                    with tab_character:
                        st.text_area("Hasil Karakter", character_result, height=150, disabled=True)

                    if speech_text:
                        st.subheader("Naskah Suara Pembelajaran")
                        st.text_area("Pola Suku Kata", speech_text, height=200, disabled=True)

                    # Bersihkan memory setelah selesai
                    del predicted_image
                    gc.collect()

                except Exception as exc:
                    st.error(f"Proses pengenalan Braille gagal: {exc}")
                    gc.collect()

    with col2:
        if uploaded_file is None:
            st.info("Upload gambar Braille di panel sebelah kiri untuk memulai.")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
