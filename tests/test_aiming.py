"""Tests for the pixel-to-bearing calibration and the aim solver."""

from __future__ import annotations

import json

import pytest

from firewatch.aiming import (
    AffineAxis,
    Calibration,
    CalibrationError,
    CalibrationPoint,
    angular_error,
    solve_bearing,
)


def _points() -> list[CalibrationPoint]:
    """A synthetic rig: 60 deg across the frame width, 30 deg down its height."""
    return [
        CalibrationPoint(u=0.0, v=0.0, pan=-30.0, tilt=15.0),
        CalibrationPoint(u=1.0, v=0.0, pan=30.0, tilt=15.0),
        CalibrationPoint(u=0.0, v=1.0, pan=-30.0, tilt=-15.0),
        CalibrationPoint(u=1.0, v=1.0, pan=30.0, tilt=-15.0),
    ]


class TestAffineAxis:
    def test_apply_maps_coord_to_angle(self) -> None:
        axis = AffineAxis(scale=60.0, offset=-30.0)
        assert axis.apply(0.5) == pytest.approx(0.0)
        assert axis.apply(0.0) == pytest.approx(-30.0)
        assert axis.apply(1.0) == pytest.approx(30.0)

    def test_invert_round_trips(self) -> None:
        axis = AffineAxis(scale=60.0, offset=-30.0)
        for coord in (0.0, 0.25, 0.5, 1.0):
            assert axis.invert(axis.apply(coord)) == pytest.approx(coord)

    def test_invert_rejects_zero_scale(self) -> None:
        with pytest.raises(CalibrationError):
            AffineAxis(scale=0.0, offset=0.0).invert(10.0)


class TestCalibrationFit:
    def test_recovers_known_mapping(self) -> None:
        calibration = Calibration.fit(_points())
        assert calibration.pan_axis.scale == pytest.approx(60.0)
        assert calibration.pan_axis.offset == pytest.approx(-30.0)
        assert calibration.tilt_axis.scale == pytest.approx(-30.0)
        assert calibration.tilt_axis.offset == pytest.approx(15.0)

    def test_needs_at_least_two_points(self) -> None:
        with pytest.raises(CalibrationError, match="at least 2"):
            Calibration.fit([CalibrationPoint(u=0.5, v=0.5, pan=0.0, tilt=0.0)])

    def test_rejects_collinear_points_on_an_axis(self) -> None:
        # Every point shares u, so no pan scale can be fitted. Silently
        # producing scale=0 here would aim every target at one bearing.
        with pytest.raises(CalibrationError, match="same image coordinate"):
            Calibration.fit(
                [
                    CalibrationPoint(u=0.5, v=0.0, pan=0.0, tilt=15.0),
                    CalibrationPoint(u=0.5, v=1.0, pan=0.0, tilt=-15.0),
                ]
            )

    def test_extra_points_are_averaged(self) -> None:
        # One badly-measured point should not dominate a fit built on many.
        points = _points()
        points.append(CalibrationPoint(u=0.5, v=0.5, pan=3.0, tilt=1.0))
        calibration = Calibration.fit(points)
        assert calibration.pan_axis.scale == pytest.approx(60.0, abs=5.0)
        assert calibration.pan_axis.offset == pytest.approx(-30.0, abs=5.0)

    def test_degenerate_detection(self) -> None:
        calibration = Calibration(
            AffineAxis(scale=0.0, offset=0.0), AffineAxis(scale=1.0, offset=0.0)
        )
        assert calibration.is_degenerate


class TestCalibrationIO:
    def test_round_trip(self, tmp_path) -> None:
        original = Calibration.fit(_points())
        target = tmp_path / "cal.json"
        original.save(target, points=_points())

        loaded = Calibration.load(target)
        assert loaded.pan_axis.scale == pytest.approx(original.pan_axis.scale)
        assert loaded.tilt_axis.offset == pytest.approx(original.tilt_axis.offset)

    def test_save_records_raw_points(self, tmp_path) -> None:
        target = tmp_path / "cal.json"
        Calibration.fit(_points()).save(target, points=_points())
        payload = json.loads(target.read_text())
        assert len(payload["points"]) == 4
        assert payload["version"] == 1

    def test_load_missing_file_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(CalibrationError, match="calibrate"):
            Calibration.load(tmp_path / "absent.json")

    def test_load_rejects_bad_json(self, tmp_path) -> None:
        target = tmp_path / "cal.json"
        target.write_text("{not json")
        with pytest.raises(CalibrationError, match="not valid JSON"):
            Calibration.load(target)

    def test_load_rejects_missing_axis(self, tmp_path) -> None:
        target = tmp_path / "cal.json"
        target.write_text(json.dumps({"pan_axis": {"scale": 1.0, "offset": 0.0}}))
        with pytest.raises(CalibrationError, match="missing axis"):
            Calibration.load(target)

    def test_creates_parent_directory(self, tmp_path) -> None:
        target = tmp_path / "nested" / "deeper" / "cal.json"
        Calibration.fit(_points()).save(target)
        assert target.is_file()


class TestBearingForBbox:
    def test_aims_at_box_centre(self) -> None:
        calibration = Calibration.fit(_points())
        # A box centred in a 640x480 frame maps to the centre of the mapping.
        pan, tilt = calibration.bearing_for_bbox((320 - 50, 240 - 50, 320 + 50, 240 + 50), (640, 480))
        assert pan == pytest.approx(0.0, abs=0.01)
        assert tilt == pytest.approx(0.0, abs=0.01)

    def test_offset_box_maps_off_centre(self) -> None:
        calibration = Calibration.fit(_points())
        pan, _ = calibration.bearing_for_bbox((600, 200, 640, 280), (640, 480))
        assert pan > 0

    def test_rejects_zero_frame_size(self) -> None:
        calibration = Calibration.fit(_points())
        with pytest.raises(CalibrationError, match="invalid frame size"):
            calibration.bearing_for_bbox((0, 0, 10, 10), (0, 480))

    def test_normalised_so_resolution_does_not_matter(self) -> None:
        calibration = Calibration.fit(_points())
        # The same relative box at two resolutions must give the same bearing.
        small = calibration.bearing_for_bbox((160, 120, 480, 360), (640, 480))
        large = calibration.bearing_for_bbox((320, 240, 960, 720), (1280, 960))
        assert small == pytest.approx(large)

    def test_point_for_bearing_is_the_inverse(self) -> None:
        calibration = Calibration.fit(_points())
        pan, tilt = calibration.bearing_for_point(0.3, 0.7)
        u, v = calibration.point_for_bearing(pan, tilt)
        assert (u, v) == pytest.approx((0.3, 0.7))


class TestSolveBearing:
    def test_returns_none_outside_envelope(self) -> None:
        calibration = Calibration.fit(_points())
        # A box at the far right needs pan=+30, which a [-10, 10] envelope
        # forbids. Returning None rather than clamping is the point: a clamped
        # bearing would report success while aiming at the wrong place.
        result = solve_bearing(
            calibration,
            (630, 200, 640, 280),
            (640, 480),
            min_pan=-10.0,
            max_pan=10.0,
            min_tilt=-45.0,
            max_tilt=45.0,
        )
        assert result is None

    def test_returns_bearing_inside_envelope(self) -> None:
        calibration = Calibration.fit(_points())
        result = solve_bearing(
            calibration,
            (300, 220, 340, 260),
            (640, 480),
            min_pan=-60.0,
            max_pan=60.0,
            min_tilt=-45.0,
            max_tilt=45.0,
        )
        assert result is not None
        pan, tilt = result
        assert -60.0 <= pan <= 60.0
        assert -45.0 <= tilt <= 45.0


class TestAngularError:
    def test_reports_largest_axis(self) -> None:
        # Not the Euclidean distance: either axis being off is enough to miss.
        assert angular_error((0.0, 0.0), (3.0, 4.0)) == pytest.approx(4.0)

    def test_identical_bearings_are_zero(self) -> None:
        assert angular_error((10.0, -5.0), (10.0, -5.0)) == 0.0
