import json
import logging
import os

# Batasi thread SEBELUM import torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

logger = logging.getLogger(__name__)


def convert_to_braille_unicode(str_input: str, path: str = "utils/braille_map.json") -> str:
    with open(path, "r", encoding="utf-8") as fl:
        data = json.load(fl)
    return data.get(str_input, str_input)


def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """Parse YOLO boxes → rows, grouped by y-coordinate.
    
    Optimasi dari skripsi:
    - Pakai median height (bukan mean) untuk robustness
    - Gunakan threshold = 60% median height (bukan 50%) untuk mengurangi
      grid shifting pada Braille yg berdekatan vertikal
    """
    jumlah_box = len(boxes)
    if jumlah_box == 0:
        return []
    new_boxes = np.zeros((jumlah_box, 6), dtype=float)
    new_boxes[:, :4] = boxes.xywh.cpu().numpy()
    new_boxes[:, 4] = boxes.conf.cpu().numpy()
    new_boxes[:, 5] = boxes.cls.cpu().numpy()
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]
    if jumlah_box == 1:
        return [new_boxes]
    # Pakai median height lebih robust terhadap outlier
    tinggi_median = float(np.median(new_boxes[:, 3]))
    # 60% median height - threshold lebih ketat kurangi shifting
    y_threshold = max(tinggi_median * 0.6, 1)
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]
    rows = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    for row in rows:
        row = row[row[:, 0].argsort()]
        boxes_return.append(row)
    logger.info(
        "Row grouping: %d box → %d baris (median_h=%.1f, threshold=%.1f)",
        jumlah_box, len(boxes_return), tinggi_median, y_threshold,
    )
    return boxes_return