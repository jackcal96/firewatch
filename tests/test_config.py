"""Tests for configuration loading and precedence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from firewatch.config import AppConfig, ModelConfig, SourceConfig


def test_defaults_are_sane():
    config = AppConfig()
    assert config.model.confidence_threshold == 0.35
    assert config.alert.alert_on_fire is True
    assert config.source.source_type == "rtsp"
    assert config.runtime.process_every_n_frames == 1


def test_yaml_file_is_loaded(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "model:\n  confidence_threshold: 0.9\nsource:\n  rtsp_url: rtsp://cam/1\n"
    )
    config = AppConfig.load(path)
    assert config.model.confidence_threshold == 0.9
    assert config.source.rtsp_url == "rtsp://cam/1"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AppConfig.load(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        AppConfig.load(path)


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("source:\n  rtsp_url: rtsp://from-file\n")
    monkeypatch.setenv("FIREWATCH_RTSP_URL", "rtsp://from-env")
    config = AppConfig.load(path)
    assert config.source.rtsp_url == "rtsp://from-env"


def test_cli_overrides_beat_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWATCH_RTSP_URL", "rtsp://from-env")
    config = AppConfig.load(None, source__rtsp_url="rtsp://from-cli")
    assert config.source.rtsp_url == "rtsp://from-cli"


def test_nested_override_sets_field():
    config = AppConfig.load(None, model__device="cuda:0", alert__min_confidence=0.7)
    assert config.model.device == "cuda:0"
    assert config.alert.min_confidence == 0.7


def test_override_for_unknown_section_raises():
    with pytest.raises(ValidationError):
        AppConfig.load(None, bogus__field=1)


def test_override_without_separator_is_ignored():
    # Sans "__" the key cannot address a nested field, so it is skipped.
    config = AppConfig.load(None, rtsp_url="rtsp://ignored")
    assert config.source.rtsp_url.startswith("rtsp://user:password")


def test_confidence_threshold_bounds_enforced():
    with pytest.raises(ValidationError):
        ModelConfig(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        ModelConfig(confidence_threshold=-0.1)


def test_file_source_requires_file_path():
    with pytest.raises(ValidationError, match="file_path is required"):
        SourceConfig(source_type="file")
    assert SourceConfig(source_type="file", file_path="a.mp4").file_path == "a.mp4"


def test_reconnect_backoff_must_be_positive():
    with pytest.raises(ValidationError):
        SourceConfig(reconnect_initial_backoff=0)
