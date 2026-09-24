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
    return 0


if __name__ == "__main__":
    sys.exit(main())
