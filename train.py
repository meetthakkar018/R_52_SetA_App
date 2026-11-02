from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATASET_PATH,
    K,
    MODEL_ARTIFACTS,
    MODEL_DIR,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    RISK_THRESHOLDS,
    SEGMENT_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
)


def resolve_dataset_path() -> Path:
    if DATASET_PATH.exists():
        return DATASET_PATH

    fallback = Path(__file__).resolve().parent / "data" / "credit_risk_dataset.csv"
    if fallback.exists():
        return fallback

    candidates = [
        Path(r"D:\roo52\data\credit_risk_dataset.csv"),
        Path("data/credit_risk_dataset.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Place the real credit_risk_dataset.csv in D:/roo52/data/ or the project data/ folder."
    )


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    actual_path = resolve_dataset_path() if path is None else path
    df = pd.read_csv(actual_path)
    df.columns = [str(col).strip() for col in df.columns]
    if "loan_status" not in df.columns:
        raise ValueError("loan_status column is required but was not found in the dataset.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(0).astype(int)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].clip(0, 1)

    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown")

    for col in [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cleaned = cleaned[REQUIRED_COLUMNS].copy()
    cleaned["cb_person_default_on_file"] = cleaned["cb_person_default_on_file"].astype(str).str.strip().str.upper()
    return cleaned


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, [col for col in numeric_features if col in X.columns]))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, [col for col in categorical_features if col in X.columns]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def select_validation_threshold(y_true: pd.Series, probabilities: np.ndarray) -> Dict[str, float]:
    thresholds = np.linspace(0.10, 0.90, 81)
    best = None
    for threshold in thresholds:
        metrics = compute_threshold_metrics(y_true, probabilities, float(threshold))
        selection_score = metrics["f1"] + 0.4 * metrics["recall"] + 0.2 * metrics["precision"]
        candidate = {**metrics, "selection_score": float(selection_score)}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    return best


def baseline_accuracy(y_true: pd.Series) -> float:
    return float((y_true == 0).mean())


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> Dict[str, object]:
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "y_pred": y_pred,
        "probabilities": probabilities,
    }


def train_classifier(name: str, estimator) -> Dict[str, object]:
    df = clean_dataset(load_dataset())
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_f, X_valid, y_train_f, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train_f)),
            ("classifier", estimator),
        ]
    )
    model_pipeline.fit(X_train_f, y_train_f)

    val_probs = model_pipeline.predict_proba(X_valid)[:, 1]
    validation_thresholds = select_validation_threshold(y_valid, val_probs)
    threshold = validation_thresholds["threshold"]

    test_result = evaluate_model(model_pipeline, X_test, y_test, threshold)
    return {
        "name": name,
        "pipeline": model_pipeline,
        "threshold": threshold,
        "validation_metrics": validation_thresholds,
        "test_metrics": test_result,
    }


def train_all_models() -> Dict[str, object]:
    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    trained_models = [train_classifier(name, estimator) for name, estimator in model_specs.items()]

    df = clean_dataset(load_dataset())
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    baseline = baseline_accuracy(y_test)

    scores = []
    for model in trained_models:
        metrics = model["test_metrics"]
        selection_value = metrics["f1"] + 0.4 * metrics["recall"] + 0.2 * metrics["precision"]
        scores.append({
            "model": model["name"],
            "threshold": model["threshold"],
            "selection_value": selection_value,
            "metrics": metrics,
        })

    selected_model = max(scores, key=lambda item: item["selection_value"])
    selected_name = selected_model["model"]
    selected_record = next(model for model in trained_models if model["name"] == selected_name)

    comparison_rows = []
    for model in trained_models:
        metrics = model["test_metrics"]
        comparison_rows.append({
            "Model Name": model["name"],
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1 Score": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "Improvement over Baseline": round(metrics["accuracy"] - baseline, 4),
            "Threshold": round(model["threshold"], 4),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    return {
        "trained_models": trained_models,
        "selected_model": selected_name,
        "selected_threshold": float(selected_model["threshold"]),
        "baseline_accuracy": baseline,
        "comparison_df": comparison_df,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_training_artifacts(summary: Dict[str, object]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model in summary["trained_models"]:
        name = model["name"]
        artifact_name = {
            "Logistic Regression": MODEL_ARTIFACTS["logistic_regression"],
            "Random Forest": MODEL_ARTIFACTS["random_forest"],
            "Gradient Boosting": MODEL_ARTIFACTS["gradient_boosting"],
        }[name]
        joblib.dump(model["pipeline"], MODEL_DIR / artifact_name)

    selected_name = summary["selected_model"]
    final_pipeline = next(model["pipeline"] for model in summary["trained_models"] if model["name"] == selected_name)
    joblib.dump(final_pipeline, MODEL_DIR / MODEL_ARTIFACTS["final_model"])
    joblib.dump(summary["selected_threshold"], MODEL_DIR / MODEL_ARTIFACTS["threshold"])
    joblib.dump(summary["comparison_df"], MODEL_DIR / MODEL_ARTIFACTS["comparison"])
    joblib.dump({
        "selected_model": selected_name,
        "selected_threshold": summary["selected_threshold"],
        "baseline_accuracy": summary["baseline_accuracy"],
        "comparison_df": summary["comparison_df"],
    }, MODEL_DIR / MODEL_ARTIFACTS["training_metadata"])


def load_saved_artifacts() -> Dict[str, object]:
    model_map = {
        "Logistic Regression": MODEL_DIR / MODEL_ARTIFACTS["logistic_regression"],
        "Random Forest": MODEL_DIR / MODEL_ARTIFACTS["random_forest"],
        "Gradient Boosting": MODEL_DIR / MODEL_ARTIFACTS["gradient_boosting"],
    }

    model_artifacts = {name: joblib.load(path) for name, path in model_map.items()}
    final_model = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["final_model"])
    comparison_df = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["comparison"])
    metadata = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["training_metadata"])
    selected_threshold = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["threshold"])

    return {
        "models": model_artifacts,
        "final_model": final_model,
        "comparison_df": comparison_df,
        "metadata": metadata,
        "selected_threshold": selected_threshold,
    }


def risk_band(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "High Risk"


def segment_name_from_profile(profile: pd.Series) -> str:
    avg_interest = profile["avg_interest_rate"]
    avg_income = profile["avg_income"]
    avg_loan_ratio = profile["avg_loan_percent_income"]
    avg_credit = profile["avg_credit_history"]

    if avg_loan_ratio >= 0.35:
        return "High Loan-to-Income Borrowers"
    if avg_income >= 80000:
        return "High-Income Established Borrowers"
    if avg_interest >= 15:
        return "Large High-Interest Loan Borrowers"
    if avg_credit <= 4:
        return "Short Credit History Borrowers"
    return "Low-Risk Stable Borrowers"


def prepare_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = df[SEGMENT_FEATURES].copy()
    for col in feature_frame.columns:
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median())
    return feature_frame


def train_kmeans_segmentation(df: pd.DataFrame) -> Dict[str, object]:
    feature_frame = prepare_segmentation_features(df)
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(feature_frame), columns=feature_frame.columns)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = model.fit_predict(scaled)

    profile_data = imputed.copy()
    profile_data["segment_id"] = cluster_ids
    profile_data["loan_status"] = df[TARGET_COLUMN].reset_index(drop=True).to_numpy()

    profile = (
        profile_data.groupby("segment_id")
        .agg(
            borrowers=("loan_status", "size"),
            avg_age=("person_age", "mean"),
            avg_income=("person_income", "mean"),
            avg_emp_length=("person_emp_length", "mean"),
            avg_loan_amount=("loan_amnt", "mean"),
            avg_interest_rate=("loan_int_rate", "mean"),
            avg_loan_percent_income=("loan_percent_income", "mean"),
            avg_credit_history=("cb_person_cred_hist_length", "mean"),
            default_rate=("loan_status", "mean"),
        )
        .reset_index()
    )

    profile["Segment Name"] = profile.apply(segment_name_from_profile, axis=1)
    profile = profile.sort_values("segment_id").reset_index(drop=True)

    silhouette = silhouette_score(scaled, cluster_ids)

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "cluster_ids": cluster_ids,
        "segment_profile": profile,
        "silhouette_score": float(silhouette),
    }


def predict_segment_for_record(record: dict | pd.Series | pd.DataFrame, segmentation: Dict[str, object]) -> Tuple[int, str]:
    if isinstance(record, pd.DataFrame):
        row = record.iloc[0]
    elif isinstance(record, pd.Series):
        row = record
    else:
        row = pd.Series(record)

    feature_row = pd.DataFrame([row])[SEGMENT_FEATURES]
    imputed = pd.DataFrame(segmentation["imputer"].transform(feature_row), columns=feature_row.columns)
    scaled = segmentation["scaler"].transform(imputed)
    segment_id = int(segmentation["model"].predict(scaled)[0])
    profile = segmentation["segment_profile"][segmentation["segment_profile"]["segment_id"] == segment_id].iloc[0]
    return segment_id, profile["Segment Name"]


def predict_single_borrower(record: dict, final_model, threshold: float, segmentation: Dict[str, object]) -> Dict[str, object]:
    record_df = pd.DataFrame([record])
    feature_columns = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
    ]
    model_input = record_df[feature_columns]
    probabilities = final_model.predict_proba(model_input)[0]
    default_prob = float(probabilities[1])
    no_default_prob = float(probabilities[0])
    segment_id, segment_name = predict_segment_for_record(record, segmentation)

    return {
        "default_probability": default_prob,
        "no_default_probability": no_default_prob,
        "risk_band": risk_band(default_prob),
        "segment_id": segment_id,
        "segment_name": segment_name,
        "selected_model": final_model.__class__.__name__,
        "threshold": float(threshold),
    }


def bulk_predict_records(df: pd.DataFrame, final_model, threshold: float, segmentation: Dict[str, object]) -> pd.DataFrame:
    required = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    results = df.copy()
    probabilities = final_model.predict_proba(df[required])[:, 1]
    results["default_probability"] = probabilities
    results["no_default_probability"] = 1 - probabilities
    results["risk_band"] = results["default_probability"].apply(risk_band)
    results["segment_id"] = None
    results["segment_name"] = None

    for index, row in results.iterrows():
        segment_id, segment_name = predict_segment_for_record(row, segmentation)
        results.at[index, "segment_id"] = segment_id
        results.at[index, "segment_name"] = segment_name

    results = results.sort_values("default_probability", ascending=False).reset_index(drop=True)
    return results


def generate_training_report() -> Dict[str, object]:
    dataset = clean_dataset(load_dataset())
    summary = train_all_models()
    save_training_artifacts(summary)
    segmentation = train_kmeans_segmentation(dataset)
    summary["segmentation"] = segmentation
    summary["dataset_shape"] = dataset.shape
    summary["selected_model_threshold"] = summary["selected_threshold"]
    return summary


if __name__ == "__main__":
    report = generate_training_report()
    print("Training complete.")
    print(f"Dataset shape: {report['dataset_shape']}")
    print(f"Selected model: {report['selected_model']}")
    print(f"Selected threshold: {report['selected_threshold']:.4f}")
    print(f"Silhouette score: {report['segmentation']['silhouette_score']:.4f}")
