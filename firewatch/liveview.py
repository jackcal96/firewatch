"""Live preview overlay for turret trials.

The overlay exists to answer one question at a glance during a trial: *is the
rig looking where it thinks it is looking?* It draws the fire box, the aim
point the nozzle is actually on, the travel envelope, and a status block
carrying the current state and aim error.

Two design rules matter here:

* The overlay never hides a failure. When the turret is off target, or the
  spray was refused, that is drawn prominently rather than left implicit in
  a log file. A trial where the water misses should look wrong on screen.
* The display is optional and cannot affect control. ``show_window`` defaults
  to false, and every failure to open or write a window is downgraded to a
  log line, so a headless machine runs the same control path as a desktop.
"""

from __future__ import annotations

import logging
import os
import sys

import cv2
import numpy as np

from .aiming import Calibration, CalibrationError
from .config import TurretConfig
from .supervisor import ActuationDecision
from .types import Detection, DetectionClass, FrameResult

logger = logging.getLogger(__name__)

_COLORS: dict[DetectionClass, tuple[int, int, int]] = {
    DetectionClass.FIRE: (0, 140, 255),
    DetectionClass.SMOKE: (200, 200, 200),
    DetectionClass.OTHER: (0, 255, 0),
}

_TEXT = (255, 255, 255)
_DIM = (170, 170, 170)
_OK = (80, 220, 80)
_WARN = (0, 200, 255)
_BAD = (60, 60, 235)
_PANEL = (32, 32, 32)


def _put(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = _TEXT,
    scale: float = 0.5,
) -> None:
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def draw_aim_overlay(
    frame: np.ndarray,
    result: FrameResult,
    config: TurretConfig,
    calibration: Calibration | None = None,
    aim_bearing: tuple[float, float] | None = None,
    aim_ok: bool | None = None,
    aim_detail: str = "",
    state: str = "",
    decision: ActuationDecision | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` annotated for a live aiming trial.

    ``aim_bearing`` is where the turret *is*, in degrees. It is mapped back
    through the calibration so the overlay shows the nozzle position in image
    space - which is the whole point of the display: you can see the gap
    between the fire and the water without leaving the video.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for detection in result.detections:
        _draw_detection(canvas, detection)

    if calibration is not None and aim_bearing is not None:
        _draw_aim_marker(canvas, calibration, aim_bearing, aim_ok, width, height)

    _draw_envelope(canvas, config, calibration, width, height)
    _draw_status(canvas, result, aim_bearing, aim_ok, aim_detail, state, decision)

    return canvas


def _draw_detection(canvas: np.ndarray, detection: Detection) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
    color = _COLORS.get(detection.label, (0, 255, 0))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    caption = f"{detection.label.value} {detection.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    top = max(0, y1 - th - 6)
    cv2.rectangle(canvas, (x1, top), (x1 + tw + 6, y1), color, -1)
    _put(canvas, caption, (x1 + 3, y1 - 4), (0, 0, 0), 0.5)


def _draw_aim_marker(
    canvas: np.ndarray,
    calibration: Calibration,
    bearing: tuple[float, float],
    aim_ok: bool | None,
    width: int,
    height: int,
) -> None:
    """Draw where the nozzle is actually pointing, in image space."""
    try:
        u, v = calibration.point_for_bearing(*bearing)
    except CalibrationError:
        return

    cx, cy = int(round(u * width)), int(round(v * height))
    if not (0 <= cx < width and 0 <= cy < height):
        # Off-screen aim is itself the finding: label it at the nearest edge
        # rather than silently dropping the marker.
        cx = min(max(cx, 12), width - 12)
        cy = min(max(cy, 12), height - 12)
        color = _BAD
        _put(canvas, "AIM OFF-FRAME", (cx - 60, cy), color, 0.5)
    else:
        color = _OK if aim_ok else _BAD

    cv2.circle(canvas, (cx, cy), 14, color, 2)
    cv2.line(canvas, (cx - 22, cy), (cx - 6, cy), color, 2)
    cv2.line(canvas, (cx + 6, cy), (cx + 22, cy), color, 2)
    cv2.line(canvas, (cx, cy - 22), (cx, cy - 6), color, 2)
    cv2.line(canvas, (cx, cy + 6), (cx, cy + 22), color, 2)


def _draw_envelope(
    canvas: np.ndarray,
    config: TurretConfig,
    calibration: Calibration | None,
    width: int,
    height: int,
) -> None:
    """Outline the region the turret is allowed to point at.

    Drawn as the image-space footprint of the travel envelope, so an operator
    can see at a glance whether a fire is somewhere the rig can legally aim.
    """
    if calibration is None:
        return

    corners: list[tuple[float, float]] = []
    for pan in (config.min_pan, config.max_pan):
        for tilt in (config.min_tilt, config.max_tilt):
            try:
                corners.append(calibration.point_for_bearing(pan, tilt))
            except CalibrationError:
                return
    if len(corners) != 4:
        return

    (u0, v0), (u1, v1), (u2, v2), (u3, v3) = corners
    xs = [u0, u1, u2, u3]
    ys = [v0, v1, v2, v3]
    # The affine map makes the envelope a rectangle in image space; corners
    # arrive in (pan, tilt) nesting order, so min/max recovers it directly.
    x1 = int(round(min(xs) * width))
    x2 = int(round(max(xs) * width))
    y1 = int(round(min(ys) * height))
    y2 = int(round(max(ys) * height))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 40), 1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    _put(canvas, "travel envelope", (x1 + 4, max(12, y1 - 4)), (120, 120, 40), 0.42)


def _draw_status(
    canvas: np.ndarray,
    result: FrameResult,
    aim_bearing: tuple[float, float] | None,
    aim_ok: bool | None,
    aim_detail: str,
    state: str,
    decision: ActuationDecision | None,
) -> None:
    """Draw the status block: state, aim, and the last decision."""
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"frame {result.frame_index}", _DIM))

    if state:
        lines.append((f"state: {state}", _TEXT))

    if aim_bearing is not None:
        lines.append(
            (
                f"aim: pan={aim_bearing[0]:.1f} tilt={aim_bearing[1]:.1f}",
                _TEXT,
            )
        )
    if aim_ok is not None:
        lines.append(
            (
                "aim: ON TARGET" if aim_ok else f"aim: OFF TARGET - {aim_detail}",
                _OK if aim_ok else _BAD,
            )
        )

    if decision is not None:
        label = f"decision: {decision.action}"
        if decision.reason:
            label += f" ({decision.reason})"
        lines.append((label, _OK if decision.allowed else _WARN))

    if result.max_fire_confidence or result.max_smoke_confidence:
        lines.append(
            (
                f"fire {result.max_fire_confidence:.2f}  smoke {result.max_smoke_confidence:.2f}",
                _DIM,
            )
        )

    pad, line_h = 8, 20
    panel_h = pad * 2 + line_h * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (330, 8 + panel_h), _PANEL, -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    y = 8 + pad + 14
    for text, color in lines:
        _put(canvas, text, (16, y), color, 0.5)
        y += line_h


class LiveView:
    """Optional ``cv2.imshow`` window.

    Display availability is decided *before* OpenCV is asked to show anything.
    That ordering matters: on a machine with no display, ``cv2.imshow`` does
    not raise a catchable exception, it aborts the process. Catching
    ``cv2.error`` around it is therefore useless, and a preview window on a
    headless box would take down the whole detection loop - including any
    spray in progress. The firmware's serial watchdog covers the water in that
    case, but killing the process is not an acceptable failure mode.
    """

    def __init__(self, enabled: bool, title: str = "firewatch") -> None:
        self._requested = enabled
        self._title = title
        self._failed = False
        self._warned = False

        if enabled and not display_available():
            self._failed = True
            self._warned = True
            logger.warning(
                "Live preview requested but no display is available; "
                "running without a window. Set runtime.show_window: false to "
                "silence this, or install 'opencv-python' (not the headless "
                "build) and run on a machine with a desktop session."
            )

    @property
    def active(self) -> bool:
        return self._requested and not self._failed

    def show(self, frame: np.ndarray) -> bool:
        """Display a frame. Returns True if it was actually shown.

        The caller uses the return value to decide whether to honour a quit
        keypress, so a windowless run is never blocked waiting for input.
        """
        if not self.active:
            return False
        try:
            cv2.imshow(self._title, frame)
            return cv2.waitKey(1) & 0xFF not in (27, ord("q"))
        except cv2.error as exc:
            # Kept as a belt-and-braces path for a GUI build that fails at
            # show time. The pre-flight check above is what handles the
            # common headless case, since that failure aborts rather than
            # raises.
            self._failed = True
            if not self._warned:
                self._warned = True
                logger.warning("Live preview failed at show time: %s", exc)
            return False

    def close(self) -> None:
        if not self.active:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def display_available() -> bool:
    """Best-effort check that a GUI window can actually be opened.

    Deliberately conservative: a false negative costs a preview window, while
    a false positive aborts the process. Callers treat False as "do not call
    imshow", never as "the display is broken".
    """
    if not _opencv_has_gui():
        return False

    # On Linux a window needs a display server. Windows and macOS always have
    # one attached to an interactive session, so no check is meaningful there.
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        )
    return True


def _opencv_has_gui() -> bool:
    """True when this OpenCV build was compiled with a GUI backend.

    ``opencv-python-headless`` is built without one, and that is the usual
    cause of a preview that cannot work. Checking the build string avoids
    importing a backend we do not have.

    The parse is whitespace-insensitive because OpenCV pads the value to a
    fixed column, and that column has moved between versions.
    """
    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001 - an unreadable build string is not fatal
        return False
    for line in info.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("GUI:"):
            continue
        backend = stripped.split(":", 1)[1].strip().upper()
        return backend not in {"", "NONE"}
    return False
