"""Helpers for the ``analyze`` CLI command."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .config import AppConfig
from .detector import FireSmokeDetector
from .logging_utils import DetectionLogger
from .pipeline import FireSmokePipeline, build_supervisor

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def analyze_file(config: AppConfig, input_path: str) -> int:
    """Run the pipeline over a local image or video file.

    Useful for validating a model or config change without a camera, and for
    rehearsing the actuation decision logic in dry-run against real footage.
    Returns a process exit code: 0 on success, 2 if the file could not be read.
    """
    path = Path(input_path)
    if not path.is_file():
        logger.error("Input file not found: %s", path)
        return 2

    detector = FireSmokeDetector.from_config(config.model)
    detection_logger = DetectionLogger(config.logging)
    supervisor = build_supervisor(config, detection_logger)
    pipeline = FireSmokePipeline(
        config, detector, detection_logger=detection_logger, supervisor=supervisor
    )

    if path.suffix.lower() in _IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.error("Could not decode image: %s", path)
            return 2
        result = pipeline.process_frame(frame, 0)
        logger.info(
            "Image %s: %s detection(s), max fire=%.3f, max smoke=%.3f",
            path.name,
            len(result.detections),
            result.max_fire_confidence,
            result.max_smoke_confidence,
        )
        pipeline.close()
        return 0

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.error("Could not open video: %s", path)
        return 2

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info("Analyzing %s (%s frames)", path.name, total_frames or "unknown")

    def frames():
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame
        finally:
            capture.release()

    processed = pipeline.run(frames())
    logger.info(
        "Done. frames_processed=%s alerts=%s", processed, pipeline.alerts_raised
    )
    return 0
