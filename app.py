import gc
import base64
import json
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
    """Mencari empat sudut kertas lalu melakukan koreksi perspektif."""
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

    logger.info("Auto-straightening: koreksi perspektif diterapkan")
    return corrected


def _hough_to_quad(lines, shape):
    """Konversi Hough lines -> 4 titik sudut kandidat."""
    h, w = shape[:2]
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    if len(endpoints) < 4:
        return None
    endpoints = np.array(endpoints, dtype="float32")
    point_sum = endpoints.sum(axis=1)
    point_diff = np.diff(endpoints, axis=1).flatten()
    tl = endpoints[np.argmin(point_sum)]
    br = endpoints[np.argmax(point_sum)]
    tr = endpoints[np.argmin(point_diff)]
    bl = endpoints[np.argmax(point_diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


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


# ═══════════════════════════════════════════════════════════════
# Image to base64 helper
# ═══════════════════════════════════════════════════════════════

def image_to_base64(image):
    """Convert OpenCV image (BGR) to base64 JPEG string."""
    cv2 = _lazy_import_cv2()
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


# ═══════════════════════════════════════════════════════════════
# Build HTML component: grid overlay + TTS + animation
# ═══════════════════════════════════════════════════════════════

def build_result_html(original_b64, detected_b64, character_cells, syllable_cells, speech_text):
    """Bangun HTML lengkap dengan grid overlay, toggle, TTS, dan animasi.
    
    Port dari Flask result.html ke self-contained HTML untuk Streamlit iframe.
    """
    character_cells_json = json.dumps(character_cells)
    syllable_cells_json = json.dumps(syllable_cells)
    speech_text_escaped = json.dumps(speech_text or "")

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f8f9fa;
    color: #333;
    padding: 16px;
  }}
  .result-container {{ max-width: 1200px; margin: 0 auto; }}
  .result-title h2 {{
    font-size: 1.4rem;
    color: #1a1a2e;
    margin-bottom: 12px;
    text-align: center;
  }}
  .result-content {{ margin-top: 8px; }}
  .result-images {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
  }}
  .box-container {{
    flex: 1;
    min-width: 280px;
  }}
  .box-container > p {{
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 6px;
    color: #444;
  }}
  .box {{
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
  }}
  .box img {{
    width: 100%;
    display: block;
  }}
  .detected-image-box {{ position: relative; }}
  .detected-image-wrapper {{
    position: relative;
    display: inline-block;
    width: 100%;
  }}
  .detected-image-wrapper img {{
    width: 100%;
    display: block;
  }}
  #detected-grid-overlay {{
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }}
  .detected-grid-cell {{
    position: absolute;
    border: 2px solid rgba(46, 125, 50, 0.7);
    background: rgba(46, 125, 50, 0.12);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: #1b5e20;
    text-transform: lowercase;
    transition: background 0.15s, border-color 0.15s, transform 0.15s;
    pointer-events: auto;
  }}
  .detected-grid-cell.active {{
    background: rgba(0, 0, 0, 0.75);
    border-color: #ffeb3b;
    color: #fff;
    transform: scale(1.06);
    z-index: 10;
    box-shadow: 0 0 12px rgba(255, 235, 59, 0.6);
  }}
  .detected-grid-cell.done {{
    background: rgba(46, 125, 50, 0.30);
    border-color: rgba(46, 125, 50, 0.5);
    color: #2e7d32;
  }}

  /* Grid mode toggle */
  .grid-mode-controls {{
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .mode-button {{
    flex: 1;
    padding: 8px 12px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    background: #fff;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
    color: #666;
    transition: all 0.2s;
  }}
  .mode-button:hover {{ border-color: #90caf9; color: #1565c0; }}
  .mode-button.active {{
    background: #1565c0;
    border-color: #1565c0;
    color: #fff;
  }}

  /* Speech result */
  .speech-script {{
    background: #f1f8e9;
    border-left: 4px solid #4caf50;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    line-height: 1.8;
    white-space: pre-wrap;
    margin-bottom: 12px;
  }}

  /* Audio controls */
  .audio-controls {{
    text-align: center;
    margin: 12px 0;
  }}
  .audio-button-group {{
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-bottom: 8px;
  }}
  .audio-button {{
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .audio-button:first-child {{
    background: #4caf50;
    color: #fff;
  }}
  .audio-button:first-child:hover {{ background: #388e3c; }}
  .audio-button:last-child {{
    background: #f44336;
    color: #fff;
  }}
  .audio-button:last-child:hover {{ background: #d32f2f; }}
  #voice-status {{
    font-size: 0.85rem;
    color: #666;
    margin-top: 4px;
  }}

  hr {{
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 16px 0;
  }}
</style>
</head>
<body>
<div class="result-container">
  <div class="result-title">
    <h2>Hasil Terjemahan Braille</h2>
  </div>
  <div class="result-content">
    <div class="result-images">
      <div class="box-container">
        <p>Gambar Asli</p>
        <div class="box">
          <img src="data:image/jpeg;base64,{original_b64}" alt="Gambar Braille asli">
        </div>
      </div>
      <div class="box-container">
        <p>Gambar Hasil Deteksi</p>
        <div class="grid-mode-controls">
          <button type="button" id="character-grid-btn" class="mode-button"
            onclick="setGridMode('character')">Grid Karakter A-Z</button>
          <button type="button" id="syllable-grid-btn" class="mode-button active"
            onclick="setGridMode('syllable')">Grid Suku Kata</button>
        </div>
        <div class="box detected-image-box">
          <div class="detected-image-wrapper">
            <img id="detected-image" src="data:image/jpeg;base64,{detected_b64}" alt="Gambar hasil deteksi">
            <div id="detected-grid-overlay"></div>
          </div>
        </div>
      </div>
    </div>

    <hr>

    <div id="result-text">
      <p style="font-weight:600; font-size:0.95rem; margin-bottom:6px;">Hasil Terjemahan</p>
      <div class="speech-script" id="speech-display"></div>
      <div class="audio-controls">
        <div class="audio-button-group">
          <button type="button" class="audio-button" onclick="speakDetectedText()">&#9654; Putar Suara</button>
          <button type="button" class="audio-button" onclick="stopSpeech()">&#9632; Stop</button>
        </div>
        <p id="voice-status">Tekan tombol Putar Suara untuk membacakan pola suku kata menjadi kata.</p>
      </div>
    </div>
  </div>
</div>

<script>
(function() {{
  // ── Data dari Python ──
  const characterGridCells = {character_cells_json};
  const syllableGridCells = {syllable_cells_json};
  const detectedSpeechText = {speech_text_escaped};

  // ── State ──
  let detectedGridCells = syllableGridCells.length ? syllableGridCells : characterGridCells;
  let currentGridMode = syllableGridCells.length ? 'syllable' : 'character';
  let gridElements = [];
  let isSpeaking = false;
  let speechSequence = 0;

  // ── Init speech display ──
  const speechDisplay = document.getElementById('speech-display');
  if (detectedSpeechText && detectedSpeechText.trim()) {{
    speechDisplay.textContent = detectedSpeechText;
  }} else {{
    speechDisplay.textContent = 'Tidak ada naskah suara yang berhasil dibuat.';
  }}

  // ── Utility ──
  function medianNumber(values) {{
    const numbers = values
      .map(v => Number(v))
      .filter(v => Number.isFinite(v) && v > 0)
      .sort((a, b) => a - b);
    if (!numbers.length) return 0;
    const mid = Math.floor(numbers.length / 2);
    return numbers.length % 2 === 0 ? (numbers[mid - 1] + numbers[mid]) / 2 : numbers[mid];
  }}

  function clampPercent(val, min, max) {{
    return Math.min(Math.max(val, min), max);
  }}

  function normalizeToken(text) {{
    return String(text || '').toLowerCase().replace(/[^a-z0-9]/g, '').trim();
  }}

  // ── Grid buttons ──
  function updateGridButtons() {{
    document.getElementById('character-grid-btn').classList.toggle('active', currentGridMode === 'character');
    document.getElementById('syllable-grid-btn').classList.toggle('active', currentGridMode === 'syllable');
  }}

  window.setGridMode = function(mode) {{
    stopSpeech(false);
    currentGridMode = mode;
    detectedGridCells = mode === 'character' ? characterGridCells : syllableGridCells;
    updateGridButtons();
    renderDetectedGrid();
    setVoiceStatus(mode === 'character'
      ? 'Mode grid karakter A-Z aktif.'
      : 'Mode grid suku kata aktif.');
  }};

  // ── Render grid overlay ──
  function renderDetectedGrid() {{
    const overlay = document.getElementById('detected-grid-overlay');
    if (!overlay) return;
    overlay.innerHTML = '';
    gridElements = [];

    const mult = currentGridMode === 'character' ? 1.10 : 1.05;
    const medW = medianNumber(detectedGridCells.map(c => c.width));
    const medH = medianNumber(detectedGridCells.map(c => c.height));

    detectedGridCells.forEach(cell => {{
      const oL = Number(cell.left) || 0;
      const oT = Number(cell.top) || 0;
      const oW = Number(cell.width) || 0;
      const oH = Number(cell.height) || 0;

      const aW = Math.max(oW, medW) * mult;
      const aH = Math.max(oH, medH) * mult;
      const cX = oL + oW / 2;
      const cY = oT + oH / 2;
      const left = clampPercent(cX - aW / 2, 0, 100 - aW);
      const top = clampPercent(cY - aH / 2, 0, 100 - aH);

      const el = document.createElement('span');
      el.className = 'detected-grid-cell';
      el.textContent = (cell.text || '').toLowerCase();
      el.dataset.speak = cell.speak || cell.text || '';
      el.style.left = left + '%';
      el.style.top = top + '%';
      el.style.width = aW + '%';
      el.style.height = aH + '%';

      overlay.appendChild(el);
      if (el.dataset.speak.trim()) gridElements.push(el);
    }});
  }}

  // ── Voice status ──
  function setVoiceStatus(msg) {{
    const el = document.getElementById('voice-status');
    if (el) el.textContent = msg;
  }}

  // ── TTS ──
  function chooseIndonesianVoice() {{
    const voices = window.speechSynthesis.getVoices();
    return voices.find(v => v.lang && v.lang.toLowerCase().startsWith('id')) || null;
  }}

  function clearHighlights() {{
    gridElements.forEach(c => {{
      c.classList.remove('active');
      c.classList.remove('done');
    }});
  }}

  window.stopSpeech = function(showMsg) {{
    if (showMsg === undefined) showMsg = true;
    isSpeaking = false;
    speechSequence += 1;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    clearHighlights();
    if (showMsg) setVoiceStatus('Pembacaan suara dihentikan.');
  }};

  function activateGridCell(index) {{
    gridElements.forEach((c, i) => {{
      c.classList.remove('active');
      if (i < index) c.classList.add('done');
      else c.classList.remove('done');
    }});
    if (gridElements[index]) gridElements[index].classList.add('active');
  }}

  function pauseGridOnFinalWord() {{
    gridElements.forEach(c => c.classList.remove('active'));
  }}

  function finishSpeech(msg) {{
    isSpeaking = false;
    gridElements.forEach(c => {{
      c.classList.remove('active');
      c.classList.add('done');
    }});
    setVoiceStatus(msg || 'Selesai membacakan hasil deteksi.');
  }}

  function parseLearningScript(text) {{
    const script = String(text || '').trim();
    if (!script) return [];
    return script.split(/[.\\n;]+/).map(p => p.trim()).filter(Boolean).map(part => {{
      const pieces = part.split('=');
      if (pieces.length < 2) {{
        const token = normalizeToken(part);
        return token ? {{ units: [token], word: token }} : null;
      }}
      const units = pieces[0].split(/[+\\s]+/).map(u => normalizeToken(u)).filter(Boolean);
      const word = normalizeToken(pieces.slice(1).join('='));
      if (!units.length && !word) return null;
      return {{ units, word }};
    }}).filter(Boolean);
  }}

  function findGridIndexForUnit(unit, startIdx) {{
    const target = normalizeToken(unit);
    if (!target || !gridElements.length) return startIdx;
    for (let i = startIdx; i < gridElements.length; i++) {{
      if (normalizeToken(gridElements[i].dataset.speak || gridElements[i].textContent) === target) return i;
    }}
    for (let i = 0; i < gridElements.length; i++) {{
      if (normalizeToken(gridElements[i].dataset.speak || gridElements[i].textContent) === target) return i;
    }}
    return Math.min(startIdx, Math.max(gridElements.length - 1, 0));
  }}

  function activateGridForUnit(unit, startIdx) {{
    if (!gridElements.length) return startIdx;
    const idx = findGridIndexForUnit(unit, startIdx);
    gridElements.forEach((c, i) => {{
      c.classList.remove('active');
      if (i < idx) c.classList.add('done');
    }});
    if (gridElements[idx]) {{
      gridElements[idx].classList.add('active');
      gridElements[idx].classList.remove('done');
    }}
    return idx + 1;
  }}

  function speakUtterance(text, seq, onStart, onEnd) {{
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'id-ID';
    utt.rate = 0.72;
    utt.pitch = 1;
    const voice = chooseIndonesianVoice();
    if (voice) utt.voice = voice;
    utt.onstart = function() {{ if (seq === speechSequence && onStart) onStart(); }};
    utt.onend = function() {{ if (seq === speechSequence && onEnd) onEnd(); }};
    utt.onerror = function() {{
      if (seq !== speechSequence) return;
      isSpeaking = false;
      setVoiceStatus('Suara gagal diputar. Tekan tombol Putar Suara sekali lagi.');
    }};
    window.speechSynthesis.speak(utt);
  }}

  function speakLearningSequence(items, phraseIdx, unitIdx, gridIdx, seq) {{
    if (!isSpeaking || seq !== speechSequence) return;
    if (phraseIdx >= items.length) {{
      finishSpeech('Selesai membacakan pola suku kata menjadi kata.');
      return;
    }}
    const item = items[phraseIdx];
    const units = item.units || [];
    const finalWord = item.word || units.join('');

    if (unitIdx < units.length) {{
      const unit = units[unitIdx];
      const nextGridIdx = activateGridForUnit(unit, gridIdx);
      speakUtterance(unit, seq,
        function() {{ setVoiceStatus('Membaca suku kata: ' + unit + '. Grid ikut berjalan.'); }},
        function() {{
          const pause = unitIdx < units.length - 1 ? 650 : 350;
          setTimeout(function() {{
            speakLearningSequence(items, phraseIdx, unitIdx + 1, nextGridIdx, seq);
          }}, pause);
        }}
      );
      return;
    }}

    pauseGridOnFinalWord();
    speakUtterance('menjadi ' + finalWord, seq,
      function() {{ setVoiceStatus('Membaca kata: ' + finalWord + '. Grid berhenti sejenak.'); }},
      function() {{
        setTimeout(function() {{
          speakLearningSequence(items, phraseIdx + 1, 0, gridIdx, seq);
        }}, 650);
      }}
    );
  }}

  function speakGridCellSequentially(index, seq) {{
    if (!isSpeaking || seq !== speechSequence) return;
    if (index >= gridElements.length) {{
      isSpeaking = false;
      gridElements.forEach(c => c.classList.remove('active'));
      setVoiceStatus('Selesai membacakan hasil deteksi.');
      return;
    }}
    const cell = gridElements[index];
    const text = (cell.dataset.speak || cell.textContent || '').trim();
    if (!text) {{ speakGridCellSequentially(index + 1, seq); return; }}
    activateGridCell(index);
    speakUtterance(text, seq,
      function() {{ setVoiceStatus('Membaca ' + currentGridMode + ': ' + text.toUpperCase()); }},
      function() {{
        if (gridElements[index]) {{
          gridElements[index].classList.remove('active');
          gridElements[index].classList.add('done');
        }}
        setTimeout(() => speakGridCellSequentially(index + 1, seq), 150);
      }}
    );
  }}

  window.speakDetectedText = function() {{
    if (!('speechSynthesis' in window)) {{
      setVoiceStatus('Browser ini belum mendukung suara. Gunakan Chrome atau Edge terbaru.');
      return;
    }}
    window.speechSynthesis.cancel();
    clearHighlights();
    speechSequence += 1;

    const learningText = (detectedSpeechText || '').trim();
    const learningItems = parseLearningScript(learningText);
    if (learningItems.length) {{
      if (syllableGridCells.length && currentGridMode !== 'syllable') {{
        currentGridMode = 'syllable';
        detectedGridCells = syllableGridCells;
        updateGridButtons();
        renderDetectedGrid();
      }}
      isSpeaking = true;
      speakLearningSequence(learningItems, 0, 0, 0, speechSequence);
      return;
    }}
    if (!gridElements.length) {{
      setVoiceStatus('Belum ada Braille yang dapat dibacakan.');
      return;
    }}
    isSpeaking = true;
    speakGridCellSequentially(0, speechSequence);
  }};

  // ── Init ──
  updateGridButtons();
  renderDetectedGrid();
  if ('speechSynthesis' in window) {{
    window.speechSynthesis.onvoiceschanged = function() {{}};
  }}
  setVoiceStatus('Mode grid ' + (currentGridMode === 'character' ? 'karakter A-Z' : 'suku kata') + ' aktif.');
}})();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# Main Streamlit App
# ═══════════════════════════════════════════════════════════════

def main():
    st = _lazy_import_streamlit()
    cv2 = _lazy_import_cv2()
    import streamlit.components.v1 as components

    # ═══════════════════════════════════════════════════════════
    # Page config
    # ═══════════════════════════════════════════════════════════
    st.set_page_config(
        page_title="Sistem Bantu Baca Braille",
        page_icon="盲",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ═══════════════════════════════════════════════════════════
    # Header
    # ═══════════════════════════════════════════════════════════
    st.title("盲 Sistem Bantu Baca Braille")
    st.caption(
        "Untuk orang tua anak tunanetra SD Kelas 1 — "
        "unggah foto tulisan Braille, sistem akan menerjemahkan ke teks Latin."
    )

    # ═══════════════════════════════════════════════════════════
    # Tips singkat
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

                    # ═══════════════════════════════════════════
                    # Convert images to base64 untuk HTML component
                    # ═══════════════════════════════════════════
                    original_b64 = image_to_base64(image)
                    detected_b64 = image_to_base64(predicted_image)

                    with col2:
                        # ═══════════════════════════════════════════
                        # HTML component: grid overlay + TTS + animasi
                        # ═══════════════════════════════════════════
                        result_html = build_result_html(
                            original_b64,
                            detected_b64,
                            character_cells or [],
                            syllable_cells or [],
                            speech_text or "",
                        )
                        components.html(result_html, height=900, scrolling=True)

                    # Bersihkan memory setelah selesai
                    del predicted_image, original_b64, detected_b64
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
