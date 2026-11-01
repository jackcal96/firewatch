"""Command-line interface for the fire/smoke detection system.

Examples:
    firewatch run --config configs/default.yaml
    firewatch run --source__rtsp_url rtsp://cam/stream --model__device cuda:0
    firewatch analyze --input clip.mp4
    firewatch check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AppConfig

logger = logging.getLogger("firewatch")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _split_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse ``section__field=value`` strings into an override mapping."""
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid override {pair!r}; expected section__field=value")
        overrides[key] = value
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firewatch",
        description="Fire and smoke detection from RTSP camera feeds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="Path to a YAML config file.")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="SECTION__FIELD=VALUE",
            help="Override any config value, e.g. --set model__device=cuda:0",
        )

    run = sub.add_parser("run", help="Run continuously on the configured stream.")
    add_common(run)

    analyze = sub.add_parser(
        "analyze", help="Analyze a local video/image file (no camera needed)."
    )
    add_common(analyze)
    analyze.add_argument("--input", required=True, help="Path to a video or image file.")

    check = sub.add_parser("check", help="Validate config and load the model.")
    add_common(check)

    home = sub.add_parser(
        "home",
        help="Home the turret and report the datum it established.",
    )
    add_common(home)

    calibrate = sub.add_parser(
        "calibrate",
        help="Build the pixel-to-bearing mapping for the turret.",
    )
    add_common(calibrate)
    calibrate.add_argument(
        "--points",
        default="configs/turret_calibration.json",
        help="Where to write the calibration.",
    )

    turret_status = sub.add_parser(
        "turret",
        help="Show turret configuration, envelope and calibration status.",
    )
    add_common(turret_status)

    return parser


def _load_config(args: argparse.Namespace, **extra: str) -> AppConfig:
    overrides = _split_overrides(args.overrides)
    overrides.update(extra)
    return AppConfig.load(args.config, **overrides)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        if args.command == "run":
            from .pipeline import run_forever

            run_forever(_load_config(args))
            return 0

        if args.command == "analyze":
            from .cli_helpers import analyze_file

            config = _load_config(
                args,
                source__source_type="file",
                source__file_path=args.input,
            )
            return analyze_file(config, args.input)

        if args.command == "check":
            return _check(_load_config(args))

        if args.command == "home":
            return _home(_load_config(args))

        if args.command == "calibrate":
            from .cli_helpers import calibrate_turret

            return calibrate_turret(_load_config(args), args.points)

        if args.command == "turret":
            return _turret_status(_load_config(args))

    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    parser.error("Unknown command")
    return 2


def _check(config: AppConfig) -> int:
    """Load config and model without touching the network/camera."""
    from .detector import FireSmokeDetector

    model_path = Path(config.model.model_path)
    if not model_path.is_file():
        logger.error("Model weights not found: %s", model_path)
        logger.error("Download them with: python scripts/download_model.py")
        return 2

    detector = FireSmokeDetector.from_config(config.model)
    logger.info("Config OK; model %s loaded via %s", model_path, type(detector).__name__)
    _report_actuation_status(config)
    _report_turret_status(config)
    return 0


def _home(config: AppConfig) -> int:
    """Home the turret, establishing the datum all bearings are measured from."""
    from .aim_controller import build_aim_controller
    from .aiming import CalibrationError

    if not config.turret.enable_turret:
        logger.error("Turret is disabled; set turret.enable_turret: true first.")
        return 2

    try:
        controller = build_aim_controller(config)
    except CalibrationError as exc:
        # Homing does not need a calibration - it only needs the mount - so a
        # missing calibration is not fatal here. Report it and carry on.
        logger.warning("%s", exc)
        controller = None

    from .turret import build_turret

    turret = controller.turret if controller is not None else build_turret(config.turret)
    if turret is None:
        logger.error("Could not build a turret from this config.")
        return 2

    try:
        event = turret.home()
    except Exception as exc:  # noqa: BLE001 - report, never crash the operator
        logger.error("Homing failed: %s", exc)
        turret.close()
        return 2
    finally:
        if controller is not None:
            controller.close()

    logger.info(
        "Turret homed at pan=%.2f tilt=%.2f (%s)",
        event.pan or 0.0,
        event.tilt or 0.0,
        "simulated" if event.simulated else "live",
    )
    logger.info(
        "Travel envelope: pan [%.1f, %.1f], tilt [%.1f, %.1f]",
        config.turret.min_pan,
        config.turret.max_pan,
        config.turret.min_tilt,
        config.turret.max_tilt,
    )
    return 0


def _turret_status(config: AppConfig) -> int:
    """Report the aiming configuration without moving anything."""
    from .aiming import Calibration, CalibrationError

    turret = config.turret
    if not turret.enable_turret:
        logger.info("Turret: DISABLED (fixed nozzle, no aiming)")
        return 0

    mode = "DRY RUN (no motion)" if turret.dry_run else "LIVE (hardware will move)"
    logger.warning("Turret: ENABLED - %s", mode)
    logger.warning("  port          : %s", turret.serial_port or "(none)")
    logger.warning(
        "  envelope      : pan [%.1f, %.1f], tilt [%.1f, %.1f]",
        turret.min_pan,
        turret.max_pan,
        turret.min_tilt,
        turret.max_tilt,
    )
    logger.warning(
        "  home datum    : pan=%.1f tilt=%.1f", turret.home_pan, turret.home_tilt
    )
    logger.warning("  aim tolerance : %.1f deg", turret.aim_tolerance_degrees)
    logger.warning("  settle        : %.2f s", turret.settle_seconds)

    try:
        calibration = Calibration.load(turret.calibration_path)
    except CalibrationError as exc:
        logger.warning("  calibration   : MISSING - %s", exc)
        return 0

    logger.warning("  calibration   : %s", turret.calibration_path)
    logger.warning(
        "    pan  = %.4f * u + %.4f",
        calibration.pan_axis.scale,
        calibration.pan_axis.offset,
    )
    logger.warning(
        "    tilt = %.4f * v + %.4f",
        calibration.tilt_axis.scale,
        calibration.tilt_axis.offset,
    )
    if calibration.is_degenerate:
        logger.warning(
            "    WARNING: degenerate axis - every target maps to one angle. "
            "Re-run 'firewatch calibrate'."
        )
    return 0


def _report_actuation_status(config: AppConfig) -> None:
    """Print exactly what the actuator would do, so enabling it is a choice.

    Reporting the resolved actuator type and zone list here means an operator
    can confirm a live configuration is what they intended *before* running,
    rather than discovering it after a spray.
    """
    from .actuator import build_actuator
    from .geometry import ZoneFilter

    actuation = config.actuation
    if not actuation.enable_actuation:
        logger.info("Actuation: DISABLED (detection and alerting only)")
        return

    actuator = build_actuator(actuation)
    zones = ZoneFilter.from_configs(actuation.zones)
    mode = "DRY RUN (no hardware will move)" if actuation.dry_run else "LIVE (hardware)"
    logger.warning("Actuation: ENABLED - %s", mode)
    logger.warning("  actuator      : %s", type(actuator).__name__)
    logger.warning("  trigger       : %s", ", ".join(actuation.trigger_classes))
    logger.warning(
        "  evidence      : conf>=%.2f, streak>=%d, held>=%.1fs, box>=%.0fpx",
        actuation.min_confidence,
        actuation.consecutive_frames,
        actuation.arm_delay_seconds,
        actuation.min_bbox_area_px,
    )
    logger.warning(
        "  spray limits  : max %.1fs per shot, %d shot(s), %.0fs cooldown",
        actuation.max_spray_seconds,
        actuation.max_shots,
        actuation.cooldown_seconds,
    )
    logger.warning(
        "  zones         : %s",
        ", ".join(zones.zone_names) if zones.has_zones else "whole frame",
    )


def _report_turret_status(config: AppConfig) -> None:
    """Report the aiming setup, including whether the spray gate is active."""
    turret = config.turret
    if not turret.enable_turret:
        logger.info("Turret: DISABLED (fixed nozzle, no aiming)")
        return

    mode = "DRY RUN (no motion)" if turret.dry_run else "LIVE (hardware)"
    logger.warning("Turret: ENABLED - %s", mode)
    logger.warning("  port          : %s", turret.serial_port or "(none)")
    logger.warning(
        "  envelope      : pan [%.1f, %.1f], tilt [%.1f, %.1f]",
        turret.min_pan,
        turret.max_pan,
        turret.min_tilt,
        turret.max_tilt,
    )
    logger.warning("  aim tolerance : %.1f deg", turret.aim_tolerance_degrees)

    if config.actuation.enable_actuation:
        logger.warning(
            "  spray gate    : ACTIVE - the valve will not open unless the "
            "turret reports a bearing within %.1f deg of the target",
            turret.aim_tolerance_degrees,
        )
    else:
        logger.warning("  spray gate    : inactive (actuation is disabled)")


if __name__ == "__main__":
    sys.exit(main())
