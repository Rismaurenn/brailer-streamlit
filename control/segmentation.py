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
        self.image_dim = (100, 150)
        self.yolo_weight = yolo_weight
        self._yolo_model = None

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_weight)
        return self._yolo_model

    def segment_braille(self, image_path):
        """Segmentasi sel Braille dari gambar."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
        results = self.yolo_model.predict(image, conf=self.conf, max_det=9999)
        boxes = results[0].boxes
        list_boxes = parse_xywh_and_class(boxes)
        return results, list_boxes

    def get_distance(self, xs):
        """Hitung jarak horizontal antar kotak dalam satu baris."""
        from collections import Counter
        if len(xs) < 2:
            return np.array([]), 0
        distances = np.diff(xs)
        common_distance = Counter(distances).most_common(1)[0][0]
        return distances, common_distance

    def get_box_properties(self, row):
        xs = [box[0] for box in row.astype(int)]
        classes = [box[-1] for box in row.astype(int)]
        distances, common = self.get_distance(xs)
        return xs, classes, distances, common

    def clean_bboxes(self, boxes):
        """Hilangkan bounding box yang tumpang tindih."""
        cleaned_boxes = []
        for row in boxes:
            if len(row) <= 1:
                cleaned_boxes.append(row)
                continue
            xs, _, distances, common = self.get_box_properties(row)
            if common == 0:
                cleaned_boxes.append(row)
                continue
            delete_indices = []
            for j in range(1, len(xs)):
                if distances[j - 1] < common / 2:
                    delete_indices.append(j - 1)
            if delete_indices:
                row = np.delete(row, delete_indices, axis=0)
            cleaned_boxes.append(row)
        return cleaned_boxes