from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a DETR training run")
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", default="facebook/detr-resnet-50")
    args = parser.parse_args()

    if not args.coco_json.exists() or not args.images.exists():
        raise SystemExit("The COCO annotation file and image directory must exist.")
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, DetrForObjectDetection  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install requirements-full.txt before preparing DETR training.") from exc

    print("Dataset paths and DETR dependencies are available.")
    print("Next step: implement a COCO Dataset adapter for the exact annotation schema you own.")
    print(f"Selected base model: {args.model}")


if __name__ == "__main__":
    main()
