"""Download fire/smoke detection weights from the Hugging Face Hub.

Default model: ``rabahdev/fire-smoke-yolov8n`` - a YOLOv8n fine-tune on the
D-Fire dataset with classes ``{0: smoke, 1: fire}``.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo <owner/name> --filename best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "rabahdev/fire-smoke-yolov8n"
DEFAULT_FILENAME = "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="File within the repo.")
    parser.add_argument(
        "--output-dir", default="models", help="Directory to place the weights in."
    )
    parser.add_argument(
        "--revision", default=None, help="Optional git revision (branch/tag/commit)."
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo}/{args.filename} -> {output_dir}")
    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        revision=args.revision,
        local_dir=str(output_dir),
    )
    print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
