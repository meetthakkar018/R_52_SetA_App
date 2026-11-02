import json
from pathlib import Path

import joblib
import pandas as pd

from config import DATASET_PATH, MODEL_DIR, PROJECT_NAME, ROLL_NUMBER, TEST_SIZE, RANDOM_STATE
from pipeline import clean_dataset, build_models_and_select_final, save_training_artifacts, train_kmeans_segmentation


def run_training_pipeline():
    dataset_path = DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {dataset_path}. Please add the real CSV before running training."
        )
    df = clean_dataset(pd.read_csv(dataset_path))
    final_summary = build_models_and_select_final()
    save_training_artifacts(final_summary)
    segmentation = train_kmeans_segmentation(df)
    metadata = {
        "roll_number": ROLL_NUMBER,
        "project_name": PROJECT_NAME,
        "dataset_shape": df.shape,
        "selected_model": final_summary["selected_model"],
        "selected_threshold": final_summary["trained_models"][0]["threshold"] if final_summary["trained_models"] else 0.5,
        "comparison_df": final_summary["comparison_df"],
        "segmentation": segmentation,
    }
    joblib.dump(metadata, MODEL_DIR / "training_metadata.joblib")
    return metadata


if __name__ == "__main__":
    result = run_training_pipeline()
    print("Training complete.")
    print(f"Dataset shape: {result['dataset_shape']}")
    print(f"Selected model: {result['selected_model']}")
    print(f"Selected threshold: {result['selected_threshold']}")
    print("Artifacts saved to model/ folder.")
