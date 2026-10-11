from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def sanitize_name(name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in Path(name).stem)
    return safe_name.strip("_") or "video"


def ensure_goggles_class(class_names: dict[int, str] | list[str] | None) -> dict[int, str]:
    if class_names is None:
        return {0: "goggles"}

    normalized: dict[int, str] = {}
    if isinstance(class_names, dict):
        items = class_names.items()
    else:
        items = enumerate(class_names)

    for index, label in items:
        cleaned = str(label).strip()
        if not cleaned:
            continue
        cleaned = cleaned.replace("-", "_").replace(" ", "_")
        cleaned = cleaned.lower()
        if cleaned in {"goggle", "goggles"}:
            cleaned = "goggles"
        normalized[int(index)] = cleaned

    if not any(name == "goggles" for name in normalized.values()):
        next_index = max(normalized, default=-1) + 1
        normalized[next_index] = "goggles"

    return normalized


def draw_detection_labels(
    image: np.ndarray,
    detections: Sequence[tuple[float, float, float, float, str]],
) -> np.ndarray:
    height, width = image.shape[:2]
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    occupied_labels: list[tuple[int, int, int, int]] = []
    boxes = [
        (
            max(0, min(width, round(x1))),
            max(0, min(height, round(y1))),
            max(0, min(width, round(x2))),
            max(0, min(height, round(y2))),
        )
        for x1, y1, x2, y2, _ in detections
    ]
    draw_order = sorted(range(len(detections)), key=lambda index: (boxes[index][1], boxes[index][0]))

    for index in draw_order:
        x1, y1, x2, y2 = boxes[index]
        label = detections[index][4]
        text_size, baseline = cv2.getTextSize(label, font, 0.45, 1)
        text_width, text_height = text_size
        label_width = text_width + 8
        label_height = text_height + baseline + 6
        gap = 2

        # Pin label strictly to the top of its own bounding box (or inside top / bottom)
        cand_x = max(0, min(width - label_width, x1))
        # Priority: 1. Just above box, 2. Inside top of box, 3. Just below box
        cand_y_positions = [
            y1 - label_height - gap,
            y1 + gap,
            y2 + gap,
            y2 - label_height - gap,
        ]

        best_rect = None
        min_overlap = float("inf")

        for cand_y in cand_y_positions:
            if 0 <= cand_y and cand_y + label_height <= height:
                rect = (cand_x, cand_y, cand_x + label_width, cand_y + label_height)
                overlap = sum(_intersection_area(rect, placed) for placed in occupied_labels)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_rect = rect
                    if overlap == 0:
                        break

        if best_rect is None:
            best_rect = (
                cand_x,
                max(0, min(height - label_height, y1 - label_height - gap)),
                cand_x + label_width,
                max(0, min(height - label_height, y1 - label_height - gap)) + label_height,
            )

        left, top, right, bottom = best_rect
        is_violation = any(kw in label.lower() for kw in ["no_", "hazard", "danger", "fire", "smoke"])
        bg_color = (0, 0, 180) if is_violation else (25, 25, 25)

        cv2.rectangle(annotated, (left, top), (right, bottom), bg_color, thickness=cv2.FILLED)
        cv2.rectangle(annotated, (left, top), (right, bottom), (255, 255, 255), thickness=1)
        cv2.putText(
            annotated,
            label,
            (left + 4, bottom - baseline - 3),
            font,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        occupied_labels.append(best_rect)

    return annotated


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def create_output_folder(base_dir: str | Path = "output", name: str = "ppe_detection") -> Path:
    output_root = Path(base_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root / f"{sanitize_name(name)}_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
