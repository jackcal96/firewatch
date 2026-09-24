"""Tests for the detection layer's filtering and label mapping."""

from __future__ import annotations

import pytest

from firewatch.config import ModelConfig
from firewatch.detector import FireSmokeDetector
from firewatch.types import DetectionClass

from .conftest import FakeBackend, make_raw


def _detector(backend, **overrides) -> FireSmokeDetector:
    config = ModelConfig(model_path="unused.pt", **overrides)
    return FireSmokeDetector(backend, config)


def test_labels_are_mapped_to_enum(frame):
    backend = FakeBackend([make_raw([(0, 0, 50, 50)], [0.9], [1])])
    detections = _detector(backend, confidence_threshold=0.4).detect(frame)
    assert len(detections) == 1
    assert detections[0].label is DetectionClass.FIRE
    assert detections[0].raw_label == "fire"


def test_smoke_class_is_recognised(frame):
    backend = FakeBackend([make_raw([(0, 0, 10, 10)], [0.8], [0])])
    detections = _detector(backend).detect(frame)
    assert detections[0].label is DetectionClass.SMOKE


def test_low_confidence_is_filtered(frame):
    backend = FakeBackend([make_raw([(0, 0, 50, 50)], [0.2], [1])])
    detections = _detector(backend, confidence_threshold=0.4).detect(frame)
    assert detections == []


def test_threshold_is_inclusive(frame):
    backend = FakeBackend([make_raw([(0, 0, 50, 50)], [0.4], [1])])
    detections = _detector(backend, confidence_threshold=0.4).detect(frame)
    assert len(detections) == 1


def test_multiple_detections_are_all_returned(frame):
    backend = FakeBackend(
        [make_raw([(0, 0, 10, 10), (20, 20, 40, 40)], [0.9, 0.7], [1, 0])]
    )
    detections = _detector(backend).detect(frame)
    assert {d.label for d in detections} == {DetectionClass.FIRE, DetectionClass.SMOKE}


def test_empty_backend_result_yields_no_detections(frame):
    backend = FakeBackend()
    assert _detector(backend).detect(frame) == []


def test_unknown_label_ignored_by_default(frame):
    backend = FakeBackend([make_raw([(0, 0, 10, 10)], [0.9], [2], names={2: "person"})])
    detections = _detector(backend, unknown_labels_policy="ignore").detect(frame)
    assert detections == []


def test_unknown_label_can_be_kept_as_other(frame):
    backend = FakeBackend([make_raw([(0, 0, 10, 10)], [0.9], [2], names={2: "person"})])
    detections = _detector(backend, unknown_labels_policy="as_other").detect(frame)
    assert detections[0].label is DetectionClass.OTHER


def test_missing_name_mapping_is_skipped(frame):
    backend = FakeBackend([make_raw([(0, 0, 10, 10)], [0.9], [7], names={})])
    assert _detector(backend).detect(frame) == []


def test_bbox_is_preserved_exactly(frame):
    backend = FakeBackend([make_raw([(1.5, 2.5, 30.0, 40.0)], [0.9], [1])])
    detection = _detector(backend).detect(frame)[0]
    assert detection.bbox == (1.5, 2.5, 30.0, 40.0)
    assert detection.area == pytest.approx((30.0 - 1.5) * (40.0 - 2.5))


def test_label_enum_handles_case_and_whitespace():
    assert DetectionClass.from_label(" FiRe ") is DetectionClass.FIRE
    assert DetectionClass.from_label("nonsense") is DetectionClass.OTHER
