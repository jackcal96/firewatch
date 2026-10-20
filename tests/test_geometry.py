"""Tests for zone geometry and the pipeline's actuation wiring."""

from __future__ import annotations

import numpy as np

from firewatch.config import ActuationConfig, AppConfig, ZoneConfig
from firewatch.geometry import Zone, ZoneFilter
from firewatch.pipeline import build_supervisor


class TestZoneGeometry:
    def test_point_inside_square(self) -> None:
        zone = Zone(name="square", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert zone.contains_point((0.5, 0.5)) is True

    def test_point_outside_square(self) -> None:
        zone = Zone(name="square", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert zone.contains_point((1.5, 0.5)) is False

    def test_point_outside_left_half_zone(self) -> None:
        zone = Zone(
            name="left",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        )
        assert zone.contains_point((0.25, 0.5)) is True
        assert zone.contains_point((0.75, 0.5)) is False

    def test_coverage_is_full_inside(self) -> None:
        zone = Zone(name="full", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        coverage = zone.coverage((100.0, 100.0, 200.0, 200.0), (1000, 1000))
        assert coverage == 1.0

    def test_coverage_is_zero_outside(self) -> None:
        zone = Zone(
            name="left",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        )
        coverage = zone.coverage((600.0, 100.0, 800.0, 300.0), (1000, 1000))
        assert coverage == 0.0

    def test_coverage_partial_for_straddling_box(self) -> None:
        zone = Zone(
            name="left",
            polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        )
        # Box spans x=0.25..0.75, so roughly half its area is in the zone.
        coverage = zone.coverage((250.0, 100.0, 750.0, 300.0), (1000, 1000))
        assert 0.3 < coverage < 0.7

    def test_degenerate_bbox_is_zero_coverage(self) -> None:
        zone = Zone(name="full", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert zone.coverage((100.0, 100.0, 100.0, 100.0), (1000, 1000)) == 0.0

    def test_zero_frame_size_is_handled(self) -> None:
        zone = Zone(name="full", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert zone.coverage((10.0, 10.0, 20.0, 20.0), (0, 0)) == 0.0


class TestZoneFilter:
    def test_no_zones_is_permissive(self) -> None:
        zone_filter = ZoneFilter([])
        assert zone_filter.has_zones is False
        match = zone_filter.evaluate((10.0, 10.0, 20.0, 20.0), (100, 100))
        assert match.matched is True
        assert match.zone_name is None

    def test_disabled_zones_are_ignored(self) -> None:
        zone_filter = ZoneFilter.from_configs(
            [
                ZoneConfig(
                    name="off",
                    enabled=False,
                    polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                )
            ]
        )
        assert zone_filter.has_zones is False

    def test_picks_zone_with_most_coverage(self) -> None:
        zone_filter = ZoneFilter.from_configs(
            [
                ZoneConfig(
                    name="left",
                    polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
                ),
                ZoneConfig(
                    name="right",
                    polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
                ),
            ]
        )
        # Box firmly in the right half.
        match = zone_filter.evaluate((700.0, 100.0, 900.0, 300.0), (1000, 1000))
        assert match.matched is True
        assert match.zone_name == "right"


class TestBuildSupervisor:
    def test_returns_none_when_disabled(self) -> None:
        config = AppConfig()
        assert config.actuation.enable_actuation is False
        assert build_supervisor(config) is None

    def test_builds_when_enabled(self) -> None:
        config = AppConfig(
            actuation=ActuationConfig(
                enable_actuation=True,
                require_zone=False,
            )
        )
        supervisor = build_supervisor(config)
        assert supervisor is not None
        assert supervisor.state.value == "idle"


class TestPipelineActuation:
    def test_pipeline_drives_supervisor(self) -> None:
        from firewatch.pipeline import FireSmokePipeline

        class FakeDetector:
            def detect(self, frame):
                from firewatch.types import Detection, DetectionClass

                return [
                    Detection(
                        label=DetectionClass.FIRE,
                        raw_label="fire",
                        confidence=0.95,
                        bbox=(100.0, 100.0, 300.0, 300.0),
                    )
                ]

        config = AppConfig(
            actuation=ActuationConfig(
                enable_actuation=True,
                require_zone=False,
                consecutive_frames=2,
                arm_delay_seconds=0.0,
            )
        )
        supervisor = build_supervisor(config)
        assert supervisor is not None
        pipeline = FireSmokePipeline(config, FakeDetector(), supervisor=supervisor)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame, 0)
        assert supervisor.shots_used == 0
        pipeline.process_frame(frame, 1)
        assert supervisor.shots_used == 1
        pipeline.close()

    def test_pipeline_without_supervisor_never_actuates(self) -> None:
        from firewatch.pipeline import FireSmokePipeline

        class FakeDetector:
            def detect(self, frame):
                from firewatch.types import Detection, DetectionClass

                return [
                    Detection(
                        label=DetectionClass.FIRE,
                        raw_label="fire",
                        confidence=0.99,
                        bbox=(10.0, 10.0, 200.0, 200.0),
                    )
                ]

        config = AppConfig()  # actuation disabled by default
        pipeline = FireSmokePipeline(config, FakeDetector())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(20):
            pipeline.process_frame(frame, i)
        pipeline.close()
