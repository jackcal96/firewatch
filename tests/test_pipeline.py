"""Tests for pipeline orchestration and JSONL logging."""

from __future__ import annotations

import json

import numpy as np
import pytest

from firewatch.alerts import AlertPolicy
from firewatch.detector import FireSmokeDetector
from firewatch.logging_utils import DetectionLogger, JsonlWriter
from firewatch.pipeline import FireSmokePipeline

from .conftest import FakeBackend, RecordingNotifier, make_raw


def _build_pipeline(app_config, backend, notifier=None):
    detector = FireSmokeDetector(backend, app_config.model)
    policy = AlertPolicy(
        app_config.alert,
        clock=lambda: 0.0,
        notifiers=[notifier or RecordingNotifier()],
    )
    return FireSmokePipeline(app_config, detector, alert_policy=policy)


def test_frame_with_fire_produces_detection_and_alert(app_config, frame):
    backend = FakeBackend([make_raw([(0, 0, 40, 40)], [0.95], [1])])
    notifier = RecordingNotifier()
    pipeline = _build_pipeline(app_config, backend, notifier)

    result = pipeline.process_frame(frame, 0)

    assert len(result.detections) == 1
    # Confidences round-trip through float32 in the backend, hence approx.
    assert result.max_fire_confidence == pytest.approx(0.95)
    assert pipeline.alerts_raised == 1
    assert len(notifier.sent) == 1
    pipeline.close()


def test_clean_frame_produces_no_alert(app_config, frame):
    notifier = RecordingNotifier()
    pipeline = _build_pipeline(app_config, FakeBackend(), notifier)

    result = pipeline.process_frame(frame, 0)

    assert result.detections == []
    assert pipeline.alerts_raised == 0
    assert notifier.sent == []
    pipeline.close()


def test_run_processes_all_frames_and_records_counts(app_config, frame):
    backend = FakeBackend([make_raw([(0, 0, 40, 40)], [0.9], [1])])
    pipeline = _build_pipeline(app_config, backend)

    processed = pipeline.run([np.zeros((8, 8, 3), np.uint8)] * 5)

    assert processed == 5
    assert pipeline.frames_processed == 5
    assert pipeline.alerts_raised == 5


def test_run_honours_process_every_n_frames(app_config, frame):
    app_config.runtime.process_every_n_frames = 2
    pipeline = _build_pipeline(app_config, FakeBackend())

    processed = pipeline.run([np.zeros((8, 8, 3), np.uint8)] * 6)

    assert processed == 3


def test_run_stops_at_max_frames(app_config, frame):
    app_config.runtime.max_frames = 2
    pipeline = _build_pipeline(app_config, FakeBackend())

    processed = pipeline.run([np.zeros((8, 8, 3), np.uint8)] * 10)

    assert processed == 2


def test_run_respects_stop_callback(app_config, frame):
    pipeline = _build_pipeline(app_config, FakeBackend())
    processed = pipeline.run(
        [np.zeros((8, 8, 3), np.uint8)] * 10, stop=lambda: True
    )
    assert processed == 0


def test_detections_are_written_to_jsonl(app_config, frame):
    backend = FakeBackend([make_raw([(0, 0, 40, 40)], [0.91], [1])])
    pipeline = _build_pipeline(app_config, backend)
    pipeline.run([np.zeros((8, 8, 3), np.uint8)])

    path = app_config.logging.log_dir + "/" + app_config.logging.detection_log
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 1
    assert records[0]["max_fire_confidence"] == 0.91
    assert records[0]["detections"][0]["label"] == "fire"


def test_clean_frames_are_not_logged(app_config):
    pipeline = _build_pipeline(app_config, FakeBackend())
    pipeline.run([np.zeros((8, 8, 3), np.uint8)] * 3)

    path = app_config.logging.log_dir + "/" + app_config.logging.detection_log
    with open(path) as handle:
        assert handle.read().strip() == ""


def test_alerts_are_written_to_alert_log(app_config, frame):
    backend = FakeBackend([make_raw([(0, 0, 40, 40)], [0.93], [1])])
    pipeline = _build_pipeline(app_config, backend)
    pipeline.run([np.zeros((8, 8, 3), np.uint8)])

    path = app_config.logging.log_dir + "/" + app_config.logging.alert_log
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert records[0]["event"] == "alert"
    assert records[0]["label"] == "fire"


def test_annotated_frames_saved_when_enabled(app_config, tmp_path):
    out = tmp_path / "annotated"
    app_config.logging.write_frames_with_detections = True
    app_config.logging.annotated_frame_dir = str(out)
    backend = FakeBackend([make_raw([(1, 1, 30, 30)], [0.9], [1])])
    pipeline = _build_pipeline(app_config, backend)

    pipeline.process_frame(np.zeros((32, 32, 3), np.uint8), 0)

    assert len(list(out.glob("*.jpg"))) == 1
    pipeline.close()


def test_jsonl_writer_appends_across_instances(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"a": 1})
    with JsonlWriter(path) as writer:
        writer.write({"b": 2})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_jsonl_writer_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"x": True})
    assert path.is_file()


def test_log_event_includes_timestamp(app_config):
    with DetectionLogger(app_config.logging) as logger:
        logger.log_event("stream_connected", url="rtsp://cam")
        record = json.loads(
            (__import__("pathlib").Path(app_config.logging.log_dir) / app_config.logging.detection_log)
            .read_text()
            .strip()
        )
    assert record["event"] == "stream_connected"
    assert record["url"] == "rtsp://cam"
    assert "timestamp" in record
