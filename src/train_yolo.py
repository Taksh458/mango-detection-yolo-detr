from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset configuration not found: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-full.txt before training YOLO.") from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        project="runs",
        name="mango_yolo",
        seed=42,
    )


if __name__ == "__main__":
    main()
