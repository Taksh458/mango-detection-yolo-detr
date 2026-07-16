from __future__ import annotations

import argparse
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_config(config_path: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()
    return root, config


def validate_label_file(path: Path) -> list[str]:
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}: line {line_number} must contain 5 values")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}: line {line_number} contains non-numeric values")
            continue
        if class_id < 0 or any(value < 0 or value > 1 for value in coords):
            errors.append(f"{path}: line {line_number} contains out-of-range values")
    return errors


def validate_dataset(config_path: Path) -> dict:
    root, config = resolve_config(config_path)
    report = {"root": str(root), "splits": {}, "errors": []}

    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        if not image_dir.exists():
            report["errors"].append(f"Missing image directory: {image_dir}")
            continue
        if not label_dir.exists():
            report["errors"].append(f"Missing label directory: {label_dir}")
            continue

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        missing_labels = []
        label_errors = []
        object_count = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing_labels.append(str(label))
                continue
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            object_count += len(lines)
            label_errors.extend(validate_label_file(label))

        report["splits"][split] = {
            "images": len(images),
            "objects": object_count,
            "missing_labels": len(missing_labels),
        }
        report["errors"].extend(missing_labels)
        report["errors"].extend(label_errors)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    args = parser.parse_args()
    report = validate_dataset(args.config.resolve())
    print(f"Dataset root: {report['root']}")
    for split, values in report["splits"].items():
        print(f"{split}: {values['images']} images, {values['objects']} objects, {values['missing_labels']} missing labels")
    if report["errors"]:
        print("\nValidation errors:")
        for error in report["errors"][:30]:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
