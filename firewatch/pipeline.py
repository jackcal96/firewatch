"""Pipeline orchestration: capture -> detect -> alert -> log.

``FireSmokePipeline`` is transport-agnostic; it consumes any iterable of
frames, which is what makes the whole system testable without a camera. The
``run_forever`` helper wires that iterable to a real source for production use.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Iterable

import numpy as np
import cv2

from .alerts import AlertPolicy, build_notifiers
from .aim_controller import AimController, build_aim_controller
from .supervisor import ActuationSupervisor, ActuationDecision
from .config import AppConfig
from .detector import FireSmokeDetector
from .liveview import LiveView, draw_aim_overlay
from .logging_utils import DetectionLogger
from .stream import RtspFrameSource
from .types import Detection, DetectionClass, FrameResult
from .visualize import draw_detections

logger = logging.getLogger(__name__)


class FireSmokePipeline:
    """Turns frames into detections, alerts and log records."""

    def __init__(
        self,
        config: AppConfig,
        detector: FireSmokeDetector,
        alert_policy: AlertPolicy | None = None,
        detection_logger: DetectionLogger | None = None,
        supervisor: ActuationSupervisor | None = None,
        aim_controller: AimController | None = None,
        live_view: LiveView | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._logger = detection_logger or DetectionLogger(config.logging)
        self._alert_policy = alert_policy or AlertPolicy(
            config.alert, notifiers=build_notifiers(config.alert)
        )
        self._owns_logger = detection_logger is None
        self._supervisor = supervisor
        self._aim_controller = aim_controller
        self._live_view = live_view or LiveView(
            config.runtime.show_window, config.runtime.window_title
        )
        self.frames_processed = 0
        self.alerts_raised = 0
        self.quit_requested = False

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Run detection on one frame, aim, alert, and log the outcome."""
        started = time.perf_counter()
        detections = self._detector.detect(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = FrameResult(
            frame_index=frame_index,
            timestamp=datetime.now(timezone.utc),
            detections=detections,
            inference_ms=elapsed_ms,
        )

        self._logger.log_frame(result)
        for alert in self._alert_policy.evaluate(result):
            self.alerts_raised += 1
            self._logger.log_alert(alert)

        height, width = frame.shape[:2]
        frame_size = (width, height)

        # Track first, decide second. Aiming is free: the turret follows the
        # strongest fire every frame so an operator can watch it work on the
        # live view long before any water is armed. The supervisor then
        # independently re-checks the achieved bearing as its final gate, so
        # tracking never implies permission to spray.
        if self._aim_controller is not None:
            self._track(result, frame_size)

        decision = None
        if self._supervisor is not None:
            decision = self._supervisor.observe(result, frame_size=frame_size)
            if decision.action != "none":
                self._logger.log_actuation(decision)

        self.frames_processed += 1

        if detections and self._config.logging.write_frames_with_detections:
            self._save_annotated(frame, frame_index, detections)

        if self._live_view.active:
            self._render_preview(frame, result, decision)

        return result

    def _track(self, result: FrameResult, frame_size: tuple[int, int]) -> None:
        """Aim at the strongest fire in this frame, if there is one.

        Tracking is decoupled from spraying on purpose. The mount follows the
        fire so the operator can see aiming working on the live view, and the
        supervisor separately decides whether the aim is good enough to spray.
        A tracking failure is recorded on the controller's last outcome and
        picked up by the aim gate; it is not raised, because a stalled turret
        must not take down the detection loop.
        """
        assert self._aim_controller is not None
        target = _strongest_fire(result.detections)
        if target is None:
            return
        self._aim_controller.aim_at(target.bbox, frame_size)

    def _render_preview(
        self,
        frame: np.ndarray,
        result: FrameResult,
        decision: ActuationDecision | None,
    ) -> None:
        """Draw the aiming overlay and show it, if a display is available."""
        aim_outcome = (
            self._aim_controller.last_outcome if self._aim_controller is not None else None
        )
        calibration = (
            self._aim_controller.calibration if self._aim_controller is not None else None
        )
        state = self._supervisor.state.value if self._supervisor is not None else ""

        preview = draw_aim_overlay(
            frame,
            result,
            self._config.turret,
            calibration=calibration,
            aim_bearing=aim_outcome.bearing if aim_outcome is not None else None,
            aim_ok=aim_outcome.ok if aim_outcome is not None else None,
            aim_detail=aim_outcome.detail if aim_outcome is not None else "",
            state=state,
            decision=decision,
        )
        if not self._live_view.show(preview):
            # The window closed, or the user pressed q/Esc. Stop cleanly rather
            # than leaving a nozzle pointed at a fire with nobody watching.
            self.quit_requested = True
            logger.info("Live preview closed; requesting shutdown")

    def run(self, frames: Iterable[np.ndarray], stop: Callable[[], bool] = lambda: False) -> int:
        """Process an iterable of frames until exhausted or stopped."""
        min_interval = (
            1.0 / self._config.runtime.max_fps if self._config.runtime.max_fps else 0.0
        )
        every_n = self._config.runtime.process_every_n_frames
        max_frames = self._config.runtime.max_frames
        processed = 0

        try:
            for index, frame in enumerate(frames):
                if stop():
                    logger.info("Stop requested; exiting pipeline")
                    break
                if index % every_n != 0:
                    continue

                tick = time.monotonic()
                self.process_frame(frame, index)
                processed += 1

                if max_frames and processed >= max_frames:
                    logger.info("Reached max_frames=%s; exiting", max_frames)
                    break

                if min_interval:
                    remaining = min_interval - (time.monotonic() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.close()

        return processed

    def _save_annotated(
        self, frame: np.ndarray, frame_index: int, detections: list
    ) -> None:
        annotated = draw_detections(frame, detections)
        out_dir = Path(self._config.logging.annotated_frame_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        cv2.imwrite(str(out_dir / f"frame_{frame_index:07d}_{stamp}.jpg"), annotated)

    def close(self) -> None:
        if self._owns_logger:
            self._logger.close()


def _strongest_fire(detections: list[Detection]) -> Detection | None:
    """The most confident fire detection, or None.

    Only fire is tracked. Tracking smoke would swing the mount toward steam
    and dust, which is exactly the false-positive class the aiming trials are
    meant to expose rather than amplify.
    """
    fires = [d for d in detections if d.label is DetectionClass.FIRE]
    if not fires:
        return None
    return max(fires, key=lambda d: d.confidence)


def build_supervisor(
    config: AppConfig,
    detection_logger: DetectionLogger | None = None,
    aim_controller: AimController | None = None,
) -> ActuationSupervisor | None:
    """Assemble an actuation supervisor, or None when actuation is disabled.

    Returns None when ``enable_actuation`` is false so the detection-only path
    carries no actuation code at all - there is no valve to accidentally open.

    ``aim_controller`` is passed as the supervisor's aim gate when a turret is
    configured. With no turret the gate is absent and the fixed-nozzle
    behaviour is unchanged.
    """
    if not config.actuation.enable_actuation:
        return None

    # Decisions are logged by the pipeline (which sees every frame); the
    # supervisor reports actuator commands directly, since those are the events
    # a human most needs to audit.
    on_event = detection_logger.log_actuation if detection_logger is not None else None
    supervisor = ActuationSupervisor(
        config.actuation,
        on_actuator_event=on_event,
        aim_gate=aim_controller,
    )

    mode = "DRY RUN" if config.actuation.dry_run else "LIVE"
    aim_note = (
        f", aim gate on (tolerance {config.turret.aim_tolerance_degrees}deg)"
        if aim_controller is not None
        else ", no aim gate"
    )
    summary = (
        f"Actuation ENABLED ({mode}): min_conf={config.actuation.min_confidence}, "
        f"streak={config.actuation.consecutive_frames}, "
        f"max_spray={config.actuation.max_spray_seconds}s, "
        f"max_shots={config.actuation.max_shots}, "
        f"zones={supervisor.zone_names or 'whole frame'}"
        f"{aim_note}"
    )
    if config.actuation.dry_run:
        logger.warning(summary)
    else:
        logger.warning("%s", summary)
        logger.warning("LIVE ACTUATION: this process can now command physical hardware")
    return supervisor


def run_forever(config: AppConfig) -> int:
    """Production entry point: capture from the configured source until Ctrl-C."""
    source = RtspFrameSource(config.source)
    detector = FireSmokeDetector.from_config(config.model)
    actuation_logger = DetectionLogger(config.logging)

    aim_controller = build_aim_controller(config)
    try:
        supervisor = build_supervisor(
            config, actuation_logger, aim_controller=aim_controller
        )
    except Exception:
        if aim_controller is not None:
            aim_controller.close()
        raise

    pipeline = FireSmokePipeline(
        config,
        detector,
        detection_logger=actuation_logger,
        supervisor=supervisor,
        aim_controller=aim_controller,
    )

    stop_event = Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down", signum)
        stop_event.set()

    # SIGINT/SIGTERM only exist on POSIX; guard for portability.
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _handle_signal)

    def _should_stop() -> bool:
        # A closed preview window stops the run: a turret left aiming at a
        # fire with nobody watching is not a state to leave the rig in.
        return stop_event.is_set() or pipeline.quit_requested

    logger.info(
        "Starting fire/smoke detection on %s (model=%s, device=%s, conf=%.2f)",
        source.url,
        config.model.model_path,
        config.model.device,
        config.model.confidence_threshold,
    )

    if aim_controller is not None:
        # Home before capturing. Without a datum the supervisor's aim gate
        # refuses every spray, so failing here is better than running blind.
        outcome = aim_controller.home()
        if not outcome.ok:
            logger.error("Turret homing failed: %s", outcome.detail)
            source.release()
            aim_controller.close()
            return 2

    try:
        processed = pipeline.run(source.frames(stop=_should_stop), stop=_should_stop)
    finally:
        source.release()
        if aim_controller is not None:
            aim_controller.close()
        pipeline.close()

    logger.info(
        "Stopped. frames_processed=%s alerts=%s read_failures=%s reconnects=%s",
        processed,
        pipeline.alerts_raised,
        source.stats.read_failures,
        source.stats.reconnects_succeeded,
    )
    return processed
