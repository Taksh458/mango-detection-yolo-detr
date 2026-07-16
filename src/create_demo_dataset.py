from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "demo_data" / "mango"
SPLITS = {"train": 12, "val": 4, "test": 4}


def create_image(path: Path, label_path: Path, seed: int) -> None:
    random.seed(seed)
    width, height = 640, 420
    image = Image.new("RGB", (width, height), (225, 234, 218))
    draw = ImageDraw.Draw(image)

    mango_count = random.randint(2, 5)
    labels = []
    for _ in range(mango_count):
        cx = random.randint(70, width - 70)
        cy = random.randint(70, height - 70)
        box_w = random.randint(45, 85)
        box_h = random.randint(55, 95)
        x0, y0 = cx - box_w // 2, cy - box_h // 2
        x1, y1 = cx + box_w // 2, cy + box_h // 2
        draw.ellipse((x0, y0, x1, y1), fill=(238, 171, 44), outline=(72, 117, 56), width=3)
        labels.append(f"0 {cx/width:.6f} {cy/height:.6f} {box_w/width:.6f} {box_h/height:.6f}")

    image.save(path)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def main() -> None:
    for split, count in SPLITS.items():
        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"synthetic_{split}_{index:03d}"
            create_image(image_dir / f"{stem}.jpg", label_dir / f"{stem}.txt", seed=index + len(split) * 100)
    print(f"Synthetic demonstration dataset created at {DATA_ROOT}")
    print("This data is for pipeline validation only, not model-performance claims.")


if __name__ == "__main__":
    main()
