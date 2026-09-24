"""Core data types shared across the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DetectionClass(str, Enum):
    """Label classes the fire/smoke model emits.

    ``OTHER`` exists as a catch-all so an unexpected class label from a
    swapped-in model does not silently disappear from the pipeline.
    """

    FIRE = "fire"
    SMOKE = "smoke"
    OTHER = "other"

    @classmethod
    def from_label(cls, label: str) -> "DetectionClass":
        key = label.strip().lower()
        for member in cls:
            if member.value == key:
                return member
        return cls.OTHER


@dataclass(frozen=True)
class Detection:
    """A single object detection within one frame.

    ``bbox`` is ``(x1, y1, x2, y2)`` in absolute pixel coordinates of the
    frame that was submitted to the model.
    """

    label: DetectionClass
    raw_label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "raw_label": self.raw_label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
        }


@dataclass
class FrameResult:
    """All detections produced for a single captured frame."""

    frame_index: int
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def fire_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.label is DetectionClass.FIRE]

    @property
    def smoke_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.label is DetectionClass.SMOKE]

    @property
    def max_fire_confidence(self) -> float:
        fires = self.fire_detections
        return max((d.confidence for d in fires), default=0.0)

    @property
    def max_smoke_confidence(self) -> float:
        smoke = self.smoke_detections
        return max((d.confidence for d in smoke), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "inference_ms": round(self.inference_ms, 2),
            "max_fire_confidence": round(self.max_fire_confidence, 4),
            "max_smoke_confidence": round(self.max_smoke_confidence, 4),
            "detections": [d.to_dict() for d in self.detections],
        }
