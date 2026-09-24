"""Annotated frame rendering for debugging and operators."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Detection, DetectionClass

_COLORS = {
    DetectionClass.FIRE: (0, 0, 255),  # BGR red
    DetectionClass.SMOKE: (160, 160, 160),  # BGR grey
    DetectionClass.OTHER: (0, 255, 0),
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and confidence labels drawn."""
    canvas = frame.copy()
    for detection in detections:
        color = _COLORS.get(detection.label, (0, 255, 0))
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        caption = f"{detection.label.value} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        # Keep the label inside the frame even when the box hugs the top edge.
        label_y = max(y1, text_h + baseline)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            caption,
            (x1, label_y - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return canvas
