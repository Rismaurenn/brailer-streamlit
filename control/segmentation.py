import logging
import os

import cv2
import numpy as np

# Batasi thread sebelum import ultralytics/torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from ultralytics import YOLO

from .convert import parse_xywh_and_class

logger = logging.getLogger(__name__)


def compute_iou(box_a, box_b):
    """IoU antara dua box dalam format xywh."""
    x1_a = box_a[0] - box_a[2] / 2
    y1_a = box_a[1] - box_a[3] / 2
    x2_a = box_a[0] + box_a[2] / 2
    y2_a = box_a[1] + box_a[3] / 2
    x1_b = box_b[0] - box_b[2] / 2
    y1_b = box_b[1] - box_b[3] / 2
    x2_b = box_b[0] + box_b[2] / 2
    y2_b = box_b[1] + box_b[3] / 2
    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(row, iou_threshold=0.3, min_confidence=None):
    """NMS per baris: hapus duplikat box overlap tinggi."""
    if len(row) < 2:
        return row
    # Sortir confidence descending
    confs = row[:, 4] if row.shape[1] > 4 else np.ones(len(row))
    if min_confidence is not None:
        confs = np.where(confs < min_confidence, 0, confs)
    indices = np.argsort(-confs)
    keep = []
    suppressed = set()
    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        for j in indices:
            if j <= i or j in suppressed:
                continue
            iou = compute_iou(row[i], row[j])
            if iou > iou_threshold:
                suppressed.add(j)
    return row[keep] if len(keep) < len(row) else row


class BrailleSegmentation:
    def __init__(self, yolo_weight="weights/yolov8_braille.pt"):
        self.conf = 0.15
        self.iou_threshold = 0.3        # NMS threshold untuk hapus duplikat
        self.image_dim = (100, 150)
        self.yolo_weight = yolo_weight
        self._yolo_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_weight)
        return self._yolo_model

    def segment_braille(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
        results = self.yolo_model.predict(image, conf=self.conf, max_det=9999)
        boxes = results[0].boxes
        list_boxes = parse_xywh_and_class(boxes)
        logger.info("YOLO deteksi: %d box sebelum NMS", sum(len(r) for r in list_boxes))
        return results, list_boxes

    @staticmethod
    def get_distance(xs):
        """Distance estimation pakai median, lebih robust drpd Counter."""
        if len(xs) < 2:
            return np.array([]), 0
        xs_sorted = np.sort(np.array(xs, dtype=float))
        distances = np.diff(xs_sorted)
        if len(distances) == 0:
            return np.array([]), 0
        common_distance = float(np.median(distances))
        return distances, common_distance

    def get_box_properties(self, row):
        xs = [box[0] for box in row.astype(int)]
        classes = [box[-1] for box in row.astype(int)]
        distances, common = self.get_distance(xs)
        return xs, classes, distances, common

    def clean_bboxes(self, boxes):
        """Improved cleaning: NMS + size filter + overlap removal."""
        cleaned_boxes = []
        for row_idx, row in enumerate(boxes):
            if len(row) == 0:
                continue
            # Step 1: NMS - hapus duplikat overlap tinggi
            row = non_max_suppression(row, iou_threshold=self.iou_threshold)
            if len(row) <= 1:
                cleaned_boxes.append(row)
                continue

            # Step 2: Ukuran filter - hapus box terlalu kecil/besar
            median_w = np.median(row[:, 2])
            median_h = np.median(row[:, 3])
            valid = []
            for box in row:
                w, h = box[2], box[3]
                # Tolak jika <25% atau >250% ukuran median
                if w < median_w * 0.25 or w > median_w * 2.5:
                    continue
                if h < median_h * 0.25 or h > median_h * 2.5:
                    continue
                valid.append(box)
            if not valid:
                cleaned_boxes.append(row)
                continue
            row = np.array(valid)
            if len(row) <= 1:
                cleaned_boxes.append(row)
                continue

            # Step 3: Jarak filter - hapus box terlalu berdekatan
            xs, _, distances, common = self.get_box_properties(row)
            if common == 0:
                cleaned_boxes.append(row)
                continue
            # common yg valid: minimal seperempat median size
            min_valid_dist = max(common * 0.4, median_w * 0.3)
            delete_indices = set()
            for j in range(1, len(xs)):
                if j in delete_indices:
                    continue
                if distances[j - 1] < min_valid_dist:
                    # Pilih box dgn confidence lebih tinggi
                    prev_conf = row[j - 1][4] if row.shape[1] > 4 else 0.5
                    curr_conf = row[j][4] if row.shape[1] > 4 else 0.5
                    if prev_conf >= curr_conf:
                        delete_indices.add(j)
                    else:
                        delete_indices.add(j - 1)
            if delete_indices:
                row = np.delete(row, sorted(delete_indices), axis=0)
            cleaned_boxes.append(row)
        total_after = sum(len(r) for r in cleaned_boxes)
        logger.info("Box setelah clean: %d", total_after)
        return cleaned_boxes