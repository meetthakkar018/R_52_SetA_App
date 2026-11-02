from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

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


def resolve_dataset_path() -> str:
    candidates = [
        DATASET_PATH,
        DATASET_PATH.parent / "data" / "credit_risk_dataset.csv",
        Path("data") / "credit_risk_dataset.csv",
        Path(r"D:\roll 52\data\credit_risk_dataset.csv"),
        Path(r"D:\roll52\data\credit_risk_dataset.csv"),
        Path(r"D:\roo52\data\credit_risk_dataset.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(DATASET_PATH)


def load_dataset(path: str | None = None) -> pd.DataFrame:
    actual_path = resolve_dataset_path() if path is None else path
    df = pd.read_csv(actual_path)
    df.columns = [str(col).strip() for col in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(0).astype(int)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].clip(0, 1)

    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        mode_value = cleaned[col].mode()
        if not mode_value.empty:
            cleaned[col] = cleaned[col].fillna(mode_value.iloc[0])
        else:
            cleaned[col] = cleaned[col].fillna("Unknown")

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
    num_cols = [col for col in numeric_features if col in X.columns]
    cat_cols = [col for col in categorical_features if col in X.columns]

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
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


def select_validation_threshold(y_true: pd.Series, probabilities: np.ndarray) -> Dict[str, Any]:
    best = None
    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = compute_threshold_metrics(y_true, probabilities, float(threshold))
        selection_score = metrics["f1"] + 0.4 * metrics["recall"] + 0.2 * metrics["precision"]
        candidate = {**metrics, "selection_score": float(selection_score)}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    return best


def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> Dict[str, Any]:
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "probabilities": probabilities,
        "predictions": y_pred,
    }


def baseline_accuracy(y_true: pd.Series) -> float:
    return float((y_true == 0).mean())


def train_classifier(name: str, estimator: Any) -> Dict[str, Any]:
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train_f)),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(X_train_f, y_train_f)

    valid_probs = pipeline.predict_proba(X_valid)[:, 1]
    validation_threshold = select_validation_threshold(y_valid, valid_probs)
    threshold = validation_threshold["threshold"]
    test_result = evaluate_model(pipeline, X_test, y_test, threshold)

    return {
        "name": name,
        "pipeline": pipeline,
        "threshold": threshold,
        "validation_metrics": validation_threshold,
        "test_metrics": test_result,
    }


def train_all_models() -> Dict[str, Any]:
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
        selection_score = metrics["f1"] + 0.4 * metrics["recall"] + 0.2 * metrics["precision"]
        scores.append(
            {
                "model": model["name"],
                "threshold": model["threshold"],
                "selection_score": float(selection_score),
                "metrics": metrics,
            }
        )

    selected = max(scores, key=lambda item: item["selection_score"])
    selected_name = selected["model"]

    comparison_rows = []
    for model in trained_models:
        metrics = model["test_metrics"]
        comparison_rows.append(
            {
                "Model Name": model["name"],
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "Improvement over Baseline": round(metrics["accuracy"] - baseline, 4),
                "Threshold": round(model["threshold"], 4),
            }
        )

    summary = {
        "trained_models": trained_models,
        "selected_model": selected_name,
        "selected_threshold": float(selected["threshold"]),
        "baseline_accuracy": float(baseline),
        "comparison_df": pd.DataFrame(comparison_rows),
        "X_test": X_test,
        "y_test": y_test,
    }
    return summary


def save_training_artifacts(summary: Dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model in summary["trained_models"]:
        artifact_name = {
            "Logistic Regression": MODEL_ARTIFACTS["logistic_regression"],
            "Random Forest": MODEL_ARTIFACTS["random_forest"],
            "Gradient Boosting": MODEL_ARTIFACTS["gradient_boosting"],
        }.get(model["name"])
        if artifact_name:
            joblib.dump(model["pipeline"], MODEL_DIR / artifact_name)

    selected_model = summary["selected_model"]
    final_pipeline = next(model["pipeline"] for model in summary["trained_models"] if model["name"] == selected_model)
    joblib.dump(final_pipeline, MODEL_DIR / MODEL_ARTIFACTS["final_model"])
    joblib.dump(summary["selected_threshold"], MODEL_DIR / MODEL_ARTIFACTS["threshold"])
    joblib.dump(summary["comparison_df"], MODEL_DIR / MODEL_ARTIFACTS["comparison"])
    joblib.dump(
        {
            "selected_model": selected_model,
            "selected_threshold": summary["selected_threshold"],
            "baseline_accuracy": summary["baseline_accuracy"],
            "comparison_df": summary["comparison_df"],
        },
        MODEL_DIR / MODEL_ARTIFACTS["training_metadata"],
    )


def load_saved_artifacts() -> Dict[str, Any]:
    model_map = {
        "Logistic Regression": MODEL_DIR / MODEL_ARTIFACTS["logistic_regression"],
        "Random Forest": MODEL_DIR / MODEL_ARTIFACTS["random_forest"],
        "Gradient Boosting": MODEL_DIR / MODEL_ARTIFACTS["gradient_boosting"],
    }
    models = {name: joblib.load(path) for name, path in model_map.items()}
    return {
        "models": models,
        "final_model": joblib.load(MODEL_DIR / MODEL_ARTIFACTS["final_model"]),
        "selected_threshold": joblib.load(MODEL_DIR / MODEL_ARTIFACTS["threshold"]),
        "comparison_df": joblib.load(MODEL_DIR / MODEL_ARTIFACTS["comparison"]),
        "metadata": joblib.load(MODEL_DIR / MODEL_ARTIFACTS["training_metadata"]),
    }


def risk_band(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "High Risk"


def prepare_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = df[SEGMENT_FEATURES].copy()
    for col in feature_frame.columns:
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median())
    return feature_frame


def segment_name_from_profile(profile: pd.Series) -> str:
    avg_interest = float(profile["avg_interest_rate"])
    avg_income = float(profile["avg_income"])
    avg_ratio = float(profile["avg_loan_percent_income"])
    avg_credit = float(profile["avg_credit_history"])

    if avg_ratio >= 0.35:
        return "High Loan-to-Income Borrowers"
    if avg_income >= 80000:
        return "High-Income Established Borrowers"
    if avg_interest >= 15:
        return "Large High-Interest Loan Borrowers"
    if avg_credit <= 4:
        return "Short Credit History Borrowers"
    return "Low-Risk Stable Borrowers"


def train_kmeans_segmentation(df: pd.DataFrame) -> Dict[str, Any]:
    feature_frame = prepare_segmentation_features(df)
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(feature_frame), columns=feature_frame.columns)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = model.fit_predict(scaled)

    cluster_df = imputed.copy()
    cluster_df["segment_id"] = cluster_ids
    cluster_df["loan_status"] = df[TARGET_COLUMN].reset_index(drop=True).to_numpy()

    segment_profile = (
        cluster_df.groupby("segment_id")
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
    segment_profile["Segment Name"] = segment_profile.apply(segment_name_from_profile, axis=1)
    segment_profile = segment_profile.sort_values("segment_id").reset_index(drop=True)

    silhouette = silhouette_score(scaled, cluster_ids)

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "cluster_ids": cluster_ids,
        "segment_profile": segment_profile,
        "silhouette_score": float(silhouette),
    }


def predict_segment_for_record(record: dict | pd.Series | pd.DataFrame, segmentation: Dict[str, Any]) -> Tuple[int, str]:
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


def predict_single_borrower(record: dict, final_model, threshold: float, segmentation: Dict[str, Any]) -> Dict[str, Any]:
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
    input_df = pd.DataFrame([record])[feature_columns]
    probabilities = final_model.predict_proba(input_df)[0]
    default_probability = float(probabilities[1])
    no_default_probability = float(probabilities[0])
    segment_id, segment_name = predict_segment_for_record(record, segmentation)
    return {
        "default_probability": default_probability,
        "no_default_probability": no_default_probability,
        "risk_band": risk_band(default_probability),
        "segment_id": segment_id,
        "segment_name": segment_name,
        "selected_model": final_model.__class__.__name__,
        "threshold": float(threshold),
    }


def bulk_predict_records(input_df: pd.DataFrame, final_model, segmentation: Dict[str, Any]) -> pd.DataFrame:
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
    missing = [col for col in required if col not in input_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    results = input_df.copy()
    probabilities = final_model.predict_proba(input_df[required])[:, 1]
    results["default_probability"] = probabilities
    results["no_default_probability"] = 1 - probabilities
    results["risk_band"] = results["default_probability"].apply(risk_band)
    results["segment_id"] = None
    results["segment_name"] = None

    for idx, row in results.iterrows():
        segment_id, segment_name = predict_segment_for_record(row, segmentation)
        results.at[idx, "segment_id"] = segment_id
        results.at[idx, "segment_name"] = segment_name

    return results.sort_values("default_probability", ascending=False).reset_index(drop=True)


def generate_training_report() -> Dict[str, Any]:
    dataset = clean_dataset(load_dataset())
    summary = train_all_models()
    save_training_artifacts(summary)
    segmentation = train_kmeans_segmentation(dataset)
    summary["segmentation"] = segmentation
    summary["dataset_shape"] = dataset.shape
    return summary


if __name__ == "__main__":
    report = generate_training_report()
    print("Training complete.")
    print(f"Dataset shape: {report['dataset_shape']}")
    print(f"Selected model: {report['selected_model']}")
    print(f"Selected threshold: {report['selected_threshold']:.4f}")
    print(f"Silhouette score: {report['segmentation']['silhouette_score']:.4f}")
