from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from config import MODEL_DIR
from pipeline import generate_training_report


def main() -> None:
    report = generate_training_report()
    print("Loan default risk training pipeline completed.")
    print(f"Dataset shape: {report['dataset_shape']}")
    print(f"Selected model: {report['selected_model']}")
    print(f"Selected threshold: {report['selected_threshold']:.4f}")
    print(f"Baseline accuracy: {report['baseline_accuracy']:.4f}")
    print(f"Silhouette score: {report['segmentation']['silhouette_score']:.4f}")
    print(f"Artifacts saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
