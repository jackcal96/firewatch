"""Object detection wrapper around an Ultralytics YOLO model.

The heavy lifting (letterboxing, NMS, tensor decoding) is delegated to
Ultralytics. This module owns everything downstream of the raw model output:
label mapping, confidence/area filtering and conversion into the dataclasses
the rest of the pipeline consumes.

``DetectionBackend`` is the seam that lets tests exercise that downstream
logic with a deterministic backend, while ``UltralyticsBackend`` is the real
path used at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .config import ModelConfig
from .types import Detection, DetectionClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDetections:
    """Backend output in frame coordinates.

    Attributes:
        xyxy: ``(N, 4)`` array of absolute pixel boxes.
        conf: ``(N,)`` array of confidences.
        cls: ``(N,)`` array of integer class ids.
        names: Mapping of class id to label string.
    """

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    names: dict[int, str]


@runtime_checkable
class DetectionBackend(Protocol):
    """Anything that can turn a BGR frame into :class:`RawDetections`."""

    def predict(self, frame: np.ndarray) -> RawDetections: ...


class UltralyticsBackend:
    """Runs inference with an Ultralytics YOLO model."""

    def __init__(self, model_path: str, config: ModelConfig) -> None:
        # Imported lazily: importing torch/ultralytics is slow and unnecessary
        # for callers that only touch config, logging or alerting.
        from ultralytics import YOLO

        logger.info("Loading detection model %s on device=%s", model_path, config.device)
        self._model = YOLO(model_path)
        self._config = config

    def predict(self, frame: np.ndarray) -> RawDetections:
        results = self._model.predict(
            frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.imgsz,
            max_det=self._config.max_detections,
            device=self._config.device,
            half=self._config.half_precision,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        names = {int(k): str(v) for k, v in dict(result.names).items()}

        if boxes is None or len(boxes) == 0:
            return RawDetections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32),
                names=names,
            )

        return RawDetections(
            xyxy=boxes.xyxy.detach().cpu().numpy().astype(np.float32),
            conf=boxes.conf.detach().cpu().numpy().astype(np.float32),
            cls=boxes.cls.detach().cpu().numpy().astype(np.float32),
            names=names,
        )


class FireSmokeDetector:
    """Filters and normalises backend detections into ``Detection`` objects."""

    def __init__(self, backend: DetectionBackend, config: ModelConfig) -> None:
        self._backend = backend
        self._config = config

    @classmethod
    def from_config(cls, config: ModelConfig) -> "FireSmokeDetector":
        return cls(UltralyticsBackend(config.model_path, config), config)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections that pass the configured thresholds.

        Confidence filtering is also applied here even though Ultralytics
        already filters, because a non-Ultralytics backend (or a future
        low-level backend) may not honour ``confidence_threshold``.
        """
        raw = self._backend.predict(frame)
        detections: list[Detection] = []

        for bbox, confidence, class_id in zip(raw.xyxy, raw.conf, raw.cls):
            score = float(confidence)
            if score < self._config.confidence_threshold:
                continue

            label = self._resolve_label(int(class_id), raw.names)
            if label is None:
                continue

            detections.append(
                Detection(
                    label=label,
                    raw_label=raw.names.get(int(class_id), "unknown"),
                    confidence=score,
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                )
            )

        return detections

    def _resolve_label(
        self, class_id: int, names: dict[int, str]
    ) -> DetectionClass | None:
        raw_label = names.get(class_id)
        if raw_label is None:
            logger.warning("Model returned class id %s with no name mapping", class_id)
            return None

        label = DetectionClass.from_label(raw_label)
        if label is DetectionClass.OTHER and self._config.unknown_labels_policy == "ignore":
            logger.debug("Ignoring unmapped model class %r", raw_label)
            return None
        return label
