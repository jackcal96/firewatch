"""Turning a detection box into a turret bearing.

The detector reports boxes in *pixels*. The turret moves in *degrees*. This
module is the only place that bridges the two, so a calibration error can be
traced to exactly one place rather than being spread across the pipeline.

Because the camera is fixed to the turret mount and both are rigid, the
mapping between a pixel and a bearing does not change while the rig is
running. It is therefore measured once - see ``firewatch calibrate`` - and
stored as a small set of correspondences.

The model is a plain affine map per axis::

    pan  = a_p * u + b_p
    tilt = a_t * v + b_t

where ``u``/``v`` are normalised image coordinates in 0-1 (not pixels), so a
resolution change on the camera does not invalidate a calibration. An affine
fit is the right level of complexity here: it is exactly determined by two
calibration points per axis, it is trivially invertible for the overlay, and
it cannot oscillate the way a higher-order fit would on a handful of samples.

There is deliberately no parallax correction. Camera and nozzle share a
mount, so the offset between them is fixed and is absorbed into the fitted
coefficients. A non-coaxial rig would need a depth term, which is a different
piece of work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class CalibrationError(ValueError):
    """Raised when a calibration is missing, malformed or unusable."""


@dataclass(frozen=True)
class CalibrationPoint:
    """One correspondence between an image point and a turret bearing.

    ``u``/``v`` are normalised image coordinates in 0-1, measured from the
    top-left of the frame. ``pan``/``tilt`` are the turret angles, in degrees,
    that put the nozzle on that image point.
    """

    u: float
    v: float
    pan: float
    tilt: float

    def to_dict(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v, "pan": self.pan, "tilt": self.tilt}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "CalibrationPoint":
        try:
            return cls(
                u=float(data["u"]),
                v=float(data["v"]),
                pan=float(data["pan"]),
                tilt=float(data["tilt"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed calibration point: {data!r}") from exc


@dataclass(frozen=True)
class AffineAxis:
    """``angle = scale * coord + offset`` for one axis."""

    scale: float
    offset: float

    def apply(self, coord: float) -> float:
        return self.scale * coord + self.offset

    def invert(self, angle: float) -> float:
        if self.scale == 0:
            raise CalibrationError("cannot invert an axis with zero scale")
        return (angle - self.offset) / self.scale

    def to_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "offset": self.offset}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "AffineAxis":
        try:
            return cls(scale=float(data["scale"]), offset=float(data["offset"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed axis: {data!r}") from exc


class Calibration:
    """The fitted pixel -> bearing mapping.

    Fit from at least two points per axis, ideally spread across the frame.
    Two points exactly determine the affine map; extra points are averaged in
    the least-squares sense, which is why the guided routine asks for more
    than the minimum.
    """

    def __init__(self, pan_axis: AffineAxis, tilt_axis: AffineAxis) -> None:
        self.pan_axis = pan_axis
        self.tilt_axis = tilt_axis

    @property
    def is_degenerate(self) -> bool:
        """True when an axis has no meaningful range in the image."""
        return self.pan_axis.scale == 0 or self.tilt_axis.scale == 0

    def bearing_for_point(self, u: float, v: float) -> tuple[float, float]:
        """Return the ``(pan, tilt)`` that aims at normalised point ``(u, v)``."""
        return self.pan_axis.apply(u), self.tilt_axis.apply(v)

    def point_for_bearing(self, pan: float, tilt: float) -> tuple[float, float]:
        """Return the normalised image point a bearing corresponds to.

        Used by the overlay to draw where the turret is actually pointing,
        which is how an operator confirms aiming before any water flows.
        """
        return self.pan_axis.invert(pan), self.tilt_axis.invert(tilt)

    def bearing_for_bbox(
        self, bbox: tuple[float, float, float, float], frame_size: tuple[int, int]
    ) -> tuple[float, float]:
        """Aim at the centre of a box given in absolute pixels.

        ``frame_size`` is ``(width, height)``. The centre is used because the
        spray target is the middle of the detected fire, and the nozzle
        pattern is symmetric about its axis.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise CalibrationError(f"invalid frame size {frame_size}")
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0 / width
        v = (y1 + y2) / 2.0 / height
        return self.bearing_for_point(u, v)

    @classmethod
    def fit(cls, points: list[CalibrationPoint]) -> "Calibration":
        """Least-squares fit of both axes from correspondences."""
        if len(points) < 2:
            raise CalibrationError(
                f"need at least 2 calibration points, got {len(points)}"
            )
        return cls(
            pan_axis=_fit_axis([(p.u, p.pan) for p in points], "pan"),
            tilt_axis=_fit_axis([(p.v, p.tilt) for p in points], "tilt"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pan_axis": self.pan_axis.to_dict(),
            "tilt_axis": self.tilt_axis.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Calibration":
        try:
            pan = data["pan_axis"]
            tilt = data["tilt_axis"]
        except KeyError as exc:
            raise CalibrationError(f"calibration missing axis: {exc}") from exc
        if not isinstance(pan, dict) or not isinstance(tilt, dict):
            raise CalibrationError("calibration axes must be mappings")
        return cls(AffineAxis.from_dict(pan), AffineAxis.from_dict(tilt))

    def save(self, path: str | Path, points: list[CalibrationPoint] | None = None) -> None:
        """Write the calibration, keeping the raw points for later refitting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "version": 1,
            **self.to_dict(),
        }
        if points is not None:
            payload["points"] = [p.to_dict() for p in points]
        target.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Wrote turret calibration to %s", target)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        source = Path(path)
        if not source.is_file():
            raise CalibrationError(
                f"no turret calibration at {source}. Run 'firewatch calibrate' "
                "with the turret connected and follow the prompts."
            )
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"calibration at {source} is not valid JSON") from exc
        if not isinstance(data, dict):
            raise CalibrationError(f"calibration at {source} must be a mapping")
        return cls.from_dict(data)


def _fit_axis(pairs: list[tuple[float, float]], name: str) -> AffineAxis:
    """Least-squares fit of ``angle = scale * coord + offset``."""
    n = len(pairs)
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
    if denominator == 0:
        raise CalibrationError(
            f"all calibration points share the same image coordinate on the "
            f"{name} axis, so no scale can be fitted. Spread the points out "
            "across the frame."
        )
    scale = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator
    offset = mean_y - scale * mean_x
    return AffineAxis(scale=scale, offset=offset)


def solve_bearing(
    calibration: Calibration,
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
    min_pan: float,
    max_pan: float,
    min_tilt: float,
    max_tilt: float,
) -> tuple[float, float] | None:
    """Aim at a box, or None when the target is outside the travel envelope.

    Returning None rather than clamping is deliberate. A clamped bearing would
    point the nozzle at the edge of the envelope and then report success,
    spraying somewhere other than the fire. Refusing is the honest answer, and
    the caller turns it into a logged deny.
    """
    pan, tilt = calibration.bearing_for_bbox(bbox, frame_size)
    if not (min_pan <= pan <= max_pan and min_tilt <= tilt <= max_tilt):
        return None
    return pan, tilt


def angular_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Largest per-axis absolute difference between two bearings, in degrees.

    The largest axis is used rather than the Euclidean distance so the check
    is conservative: if either axis is off by more than the tolerance, the
    aim is not good enough.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
