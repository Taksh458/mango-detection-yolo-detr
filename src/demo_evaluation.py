from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    # Deliberately labelled synthetic demonstration values. These are not orchard results.
    metrics = {
        "data_type": "synthetic_pipeline_demo",
        "warning": "Do not report these values as real model performance.",
        "models": {
            "YOLO demo": {"precision": 0.78, "recall": 0.74, "mAP50": 0.76},
            "DETR demo": {"precision": 0.75, "recall": 0.79, "mAP50": 0.77},
        },
    }
    (OUTPUT / "synthetic_demo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    names = list(metrics["models"])
    precision = [metrics["models"][name]["precision"] for name in names]
    recall = [metrics["models"][name]["recall"] for name in names]
    x = range(len(names))
    plt.figure(figsize=(7, 5))
    plt.bar([value - 0.18 for value in x], precision, width=0.36, label="Precision")
    plt.bar([value + 0.18 for value in x], recall, width=0.36, label="Recall")
    plt.xticks(list(x), names)
    plt.ylim(0, 1)
    plt.ylabel("Synthetic score")
    plt.title("Synthetic pipeline demonstration — not real model results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "synthetic_demo_comparison.png", dpi=160)
    plt.close()
    print(f"Synthetic demonstration outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
