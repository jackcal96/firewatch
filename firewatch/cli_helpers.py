"""Helpers for the ``analyze``, ``calibrate`` and turret CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .aim_controller import AimController, build_aim_controller
from .aiming import Calibration, CalibrationPoint
from .config import AppConfig
from .detector import FireSmokeDetector
from .logging_utils import DetectionLogger
from .pipeline import FireSmokePipeline, build_supervisor

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# The guided routine aims at a grid of image points rather than a couple of
# corners. More samples than the minimum two per axis is deliberate: a bad
# single measurement is averaged down instead of tilting the whole fit, and a
# grid exposes a non-linear rig (badly geared axis, lens distortion) as a
# residual rather than hiding it in an otherwise plausible line.
_CALIBRATION_GRID: list[tuple[float, float]] = [
    (0.25, 0.30),
    (0.75, 0.30),
    (0.25, 0.60),
    (0.75, 0.60),
    (0.50, 0.45),
]

_GRID_LABELS = [
    "upper left",
    "upper right",
    "lower left",
    "lower right",
    "centre",
]


def calibrate_turret(config: AppConfig, output_path: str) -> int:
    """Walk an operator through building the pixel-to-bearing mapping.

    The procedure is interactive because the mapping depends on where the rig
    physically points, which only the operator can see. For each grid point
    the routine asks for the pan and tilt that put the nozzle on that spot in
    the camera image.

    Nothing here moves the turret. The operator drives it with whatever
    manual control they have (the firmware's jog commands, a hand controller)
    and reports the angles. That keeps the routine usable before the software
    is trusted to command motion, which is exactly when it is needed.
    """
    if not config.turret.enable_turret:
        logger.error("Turret is disabled; set turret.enable_turret: true first.")
        return 2

    print()
    print("Turret calibration")
    print("=" * 60)
    print("For each point below, aim the nozzle at that spot in the camera")
    print("image, then enter the pan and tilt the turret is at.")
    print()
    print("The camera view is divided like this (u across, v down):")
    print()
    print("        u=0.25   u=0.50   u=0.75")
    print("  v=0.30  (1)              (2)")
    print("  v=0.45            (5)")
    print("  v=0.60  (3)              (4)")
    print()
    print("Enter 'skip' to leave a point out, or 'abort' to stop.")
    print("Aim at the fire's centre on screen, not the nozzle's own housing.")
    print()

    points: list[CalibrationPoint] = []
    used_labels: list[str] = []
    for (u, v), label in zip(_CALIBRATION_GRID, _GRID_LABELS):
        response = _prompt_point(u, v, label)
        if response is None:
            logger.info("Calibration aborted by operator")
            return 1
        if response is _SKIP:
            continue
        assert isinstance(response, CalibrationPoint)
        points.append(response)
        used_labels.append(label)

    if len(points) < 2:
        logger.error(
            "Need at least 2 calibration points; got %d. Nothing written.", len(points)
        )
        return 2

    try:
        calibration = Calibration.fit(points)
    except ValueError as exc:
        logger.error("Calibration failed: %s", exc)
        return 2

    calibration.save(output_path, points=points)
    _report_fit(points, used_labels, calibration)
    logger.info("Calibration written to %s", output_path)
    print()
    print("Next: run 'firewatch turret' to review, then 'firewatch run'.")
    return 0


# Distinct from None so the caller can tell "operator skipped this point" from
# "operator aborted the run". Returning None for both would make a single skip
# silently discard the whole calibration.
_SKIP = object()


def _prompt_point(u: float, v: float, label: str) -> CalibrationPoint | object | None:
    """Ask for one correspondence.

    Returns a :class:`CalibrationPoint`, ``_SKIP`` when the operator skips this
    point, or ``None`` when they abort the run.
    """
    while True:
        try:
            raw = input(f"[{label}] u={u:.2f} v={v:.2f} -> pan,tilt (e.g. -12.5,8.0): ")
        except EOFError:
            return None
        text = raw.strip().lower()
        if text in {"abort", "quit", "q"}:
            return None
        if text in {"skip", ""}:
            logger.info("Skipping %s", label)
            return _SKIP

        parts = [p for p in text.replace(" ", "").split(",") if p]
        if len(parts) != 2:
            print("  Expected two numbers separated by a comma, e.g. -12.5,8.0")
            continue
        try:
            pan, tilt = float(parts[0]), float(parts[1])
        except ValueError:
            print("  Could not parse those as numbers; try again.")
            continue
        return CalibrationPoint(u=u, v=v, pan=pan, tilt=tilt)


def _report_fit(
    points: list[CalibrationPoint], labels: list[str], calibration: Calibration
) -> None:
    """Print the fitted map and the residual at each measured point.

    Residuals are the useful part: they show how far the fitted line sits from
    what was actually measured. A large residual means the rig is not behaving
    affinely - usually a slipping coupling or a mis-scaled axis - and no
    amount of extra calibration points will fix it.
    """
    print()
    print("Fitted mapping:")
    print(
        f"  pan  = {calibration.pan_axis.scale:+.4f} * u "
        f"{calibration.pan_axis.offset:+.4f}"
    )
    print(
        f"  tilt = {calibration.tilt_axis.scale:+.4f} * v "
        f"{calibration.tilt_axis.offset:+.4f}"
    )
    print()
    print("Residuals (measured vs fitted):")
    worst = 0.0
    for point, label in zip(points, labels):
        pan, tilt = calibration.bearing_for_point(point.u, point.v)
        dpan = point.pan - pan
        dtilt = point.tilt - tilt
        worst = max(worst, abs(dpan), abs(dtilt))
        print(
            f"  {label:<12} pan {point.pan:+7.2f} (fit {pan:+7.2f}, d={dpan:+5.2f})  "
            f"tilt {point.tilt:+7.2f} (fit {tilt:+7.2f}, d={dtilt:+5.2f})"
        )
    print()
    print(f"  worst residual: {worst:.2f} deg")
    if worst > 5.0:
        print(
            "  WARNING: residual is large. Check for a slipping coupling, a "
            "stalled axis, or points measured against the wrong reference."
        )
    print()


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

    aim_controller = build_aim_controller(config)
    try:
        supervisor = build_supervisor(
            config, detection_logger, aim_controller=aim_controller
        )
    except Exception:
        if aim_controller is not None:
            aim_controller.close()
        raise

    if aim_controller is not None:
        # Home before analysing: without a datum the aim gate refuses every
        # spray, and a dry run would report denials that say nothing about
        # whether the decision logic works.
        outcome = aim_controller.home()
        if not outcome.ok:
            logger.error("Turret homing failed: %s", outcome.detail)
            aim_controller.close()
            return 2

    pipeline = FireSmokePipeline(
        config,
        detector,
        detection_logger=detection_logger,
        supervisor=supervisor,
        aim_controller=aim_controller,
    )

    if path.suffix.lower() in _IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.error("Could not decode image: %s", path)
            _close_aim(aim_controller)
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
        _close_aim(aim_controller)
        return 0

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.error("Could not open video: %s", path)
        _close_aim(aim_controller)
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

    try:
        processed = pipeline.run(frames())
    finally:
        _close_aim(aim_controller)
    logger.info(
        "Done. frames_processed=%s alerts=%s", processed, pipeline.alerts_raised
    )
    return 0


def _close_aim(aim_controller: AimController | None) -> None:
    """Release turret hardware, tolerating a controller that never opened."""
    if aim_controller is None:
        return
    try:
        aim_controller.close()
    except Exception:  # noqa: BLE001 - cleanup must not mask the real outcome
        logger.exception("Error closing the turret")
