from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import silhouette_score

from config import (
    K,
    MODEL_ARTIFACTS,
    MODEL_DIR,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    RISK_THRESHOLDS,
    SEGMENT_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
    DATASET_PATH,
)


def ensure_dataset_exists(path: str | None = None) -> pd.DataFrame:
    dataset_path = path or str(DATASET_PATH)
    if not DATASET_PATH.exists() and path is None:
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Place the CSV in data/credit_risk_dataset.csv before training."
        )
    if path is not None and not pd.read_csv(path).empty:
        return pd.read_csv(path)
    if path is None and DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
    else:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in cleaned.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").astype("Int64")
    for col in cleaned.columns:
        if col == TARGET_COLUMN:
            continue
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).str.strip()
        if cleaned[col].dtype.kind in "if":
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            median_value = cleaned[col].median()
            if pd.notna(median_value):
                cleaned[col].fillna(median_value, inplace=True)

    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown")

    cleaned["loan_status"] = cleaned["loan_status"].fillna(0).astype(int)
    cleaned = cleaned[REQUIRED_COLUMNS].copy()
    cleaned["loan_status"] = cleaned["loan_status"].clip(0, 1)
    return cleaned


def get_feature_types(df: pd.DataFrame):
    numerical_columns = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    categorical_columns = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    return [c for c in numerical_columns if c in df.columns], [c for c in categorical_columns if c in df.columns]


def build_preprocessor(X: pd.DataFrame):
    numeric_cols, categorical_cols = get_feature_types(X)
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
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def sigmoid_threshold_metrics(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def find_best_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)
    best = None
    for threshold in thresholds:
        metrics = sigmoid_threshold_metrics(y_true, probabilities, threshold)
        score = metrics["f1"] + 0.5 * metrics["recall"] + 0.25 * metrics["precision"]
        candidate = {
            **metrics,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def baseline_accuracy(y_true):
    majority = 0 if (y_true == 0).mean() >= 0.5 else 1
    return (y_true == majority).mean()


def evaluate_model(model_pipeline, X_test, y_test, validation_threshold=None):
    proba = model_pipeline.predict_proba(X_test)[:, 1]
    threshold = validation_threshold if validation_threshold is not None else 0.5
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]),
        "threshold": threshold,
        "y_pred": pred,
        "probabilities": proba,
    }


def train_model_family(name: str, estimator):
    df = clean_dataset(ensure_dataset_exists())
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(X_train_sub, y_train_sub)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, val_proba)
    validation_threshold = best_threshold["threshold"]
    test_metrics = evaluate_model(pipeline, X_test, y_test, validation_threshold)

    return {
        "name": name,
        "pipeline": pipeline,
        "threshold": validation_threshold,
        "validation_threshold_metrics": best_threshold,
        "test_metrics": test_metrics,
    }


def build_models_and_select_final():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    trained_models = []
    for model_name, estimator in models.items():
        trained_models.append(train_model_family(model_name, estimator))

    baseline = baseline_accuracy(ensure_dataset_exists()[TARGET_COLUMN].iloc[:])
    # Recompute on held-out test set after splitting to keep a strict comparison.
    df = clean_dataset(ensure_dataset_exists())
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    baseline_model_score = baseline_accuracy(y_test)

    validation_scores = []
    for model in trained_models:
        metric = model["validation_threshold_metrics"]
        validation_scores.append(
            {
                "model": model["name"],
                "score": metric["f1"] + 0.5 * metric["recall"] + 0.25 * metric["precision"],
                "threshold": model["threshold"],
                "val_precision": metric["precision"],
                "val_recall": metric["recall"],
                "val_f1": metric["f1"],
                "test_metrics": model["test_metrics"],
            }
        )

    selected = max(validation_scores, key=lambda item: item["score"])
    selected_model = next(model for model in trained_models if model["name"] == selected["model"])
    selected_model["selected_by_rule"] = selected
    selected_model["baseline_accuracy"] = baseline_model_score

    comparison_rows = []
    for model in trained_models:
        metrics = model["test_metrics"]
        improvement = metrics["accuracy"] - baseline_model_score
        comparison_rows.append(
            {
                "Model Name": model["name"],
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "Improvement over Baseline": round(improvement, 4),
                "Threshold": model["threshold"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    final_summary = {
        "selected_model": selected["model"],
        "selected_threshold": selected["threshold"],
        "baseline_accuracy": baseline_model_score,
        "comparison_df": comparison_df,
        "trained_models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
    }
    return final_summary


def save_training_artifacts(final_summary):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_record in final_summary["trained_models"]:
        artifact_name = {
            "Logistic Regression": MODEL_ARTIFACTS["logistic_regression"],
            "Random Forest": MODEL_ARTIFACTS["random_forest"],
            "Gradient Boosting": MODEL_ARTIFACTS["gradient_boosting"],
        }[model_record["name"]]
        joblib.dump(model_record["pipeline"], MODEL_DIR / artifact_name)

    selected_model_name = final_summary["selected_model"]
    selected_model_record = next(m for m in final_summary["trained_models"] if m["name"] == selected_model_name)
    final_model_path = MODEL_DIR / MODEL_ARTIFACTS["final_model"]
    joblib.dump(selected_model_record["pipeline"], final_model_path)
    joblib.dump(selected_model_record["threshold"], MODEL_DIR / "selected_threshold.joblib")
    joblib.dump(final_summary["comparison_df"], MODEL_DIR / "comparison_table.joblib")
    joblib.dump({
        "selected_model": selected_model_name,
        "threshold": selected_model_record["threshold"],
        "baseline_accuracy": final_summary["baseline_accuracy"],
        "comparison_df": final_summary["comparison_df"],
    }, MODEL_DIR / MODEL_ARTIFACTS["metadata"])


def load_saved_models():
    model_paths = {
        "Logistic Regression": MODEL_DIR / MODEL_ARTIFACTS["logistic_regression"],
        "Random Forest": MODEL_DIR / MODEL_ARTIFACTS["random_forest"],
        "Gradient Boosting": MODEL_DIR / MODEL_ARTIFACTS["gradient_boosting"],
    }
    artifacts = {name: joblib.load(path) for name, path in model_paths.items()}
    selected_threshold = joblib.load(MODEL_DIR / "selected_threshold.joblib")
    metadata = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["metadata"])
    final_model = joblib.load(MODEL_DIR / MODEL_ARTIFACTS["final_model"])
    comparison_table = joblib.load(MODEL_DIR / "comparison_table.joblib")
    return {
        "models": artifacts,
        "final_model": final_model,
        "selected_threshold": selected_threshold,
        "metadata": metadata,
        "comparison_table": comparison_table,
    }


def risk_band(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "High Risk"


def default_probability(prediction: object) -> float:
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    if isinstance(prediction, (list, tuple, np.ndarray)) and len(prediction) > 1:
        return float(prediction[1])
    return float(prediction)


def run_training_pipeline():
    df = clean_dataset(ensure_dataset_exists())
    final_summary = build_models_and_select_final()
    save_training_artifacts(final_summary)
    segmentation_summary = train_kmeans_segmentation(df)
    metadata = {
        "data": df,
        "final_summary": final_summary,
        "segmentation": segmentation_summary,
        "selected_model": final_summary["selected_model"],
        "selected_threshold": final_summary["trained_models"][0]["threshold"] if final_summary["trained_models"] else 0.5,
    }
    return metadata


def prepare_feature_matrix(df: pd.DataFrame):
    feature_df = df.copy()
    for col in SEGMENT_FEATURES:
        if col not in feature_df.columns:
            raise ValueError(f"Missing clustering feature: {col}")
    return feature_df[SEGMENT_FEATURES]


def train_kmeans_segmentation(df: pd.DataFrame):
    feature_df = prepare_feature_matrix(df)
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(feature_df), columns=feature_df.columns)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = model.fit_predict(scaled)
    cluster_df = imputed.copy()
    cluster_df["segment_id"] = cluster_labels
    cluster_df["loan_status"] = df[TARGET_COLUMN].reset_index(drop=True).values

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
    segment_profile["Segment Name"] = [create_segment_name(row) for _, row in segment_profile.iterrows()]
    segment_profile = segment_profile.sort_values("segment_id").reset_index(drop=True)

    silhouette = silhouette_score(scaled, cluster_labels)
    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "cluster_labels": cluster_labels,
        "segment_profile": segment_profile,
        "silhouette_score": silhouette,
    }


def create_segment_name(row):
    if row["avg_loan_percent_income"] >= 0.35:
        return "High Loan-to-Income Borrowers"
    if row["avg_income"] >= row["avg_income"].max() * 0.75:
        return "High-Income Established Borrowers"
    if row["avg_interest_rate"] >= 15:
        return "Large High-Interest Loan Borrowers"
    if row["avg_credit_history"] <= 3:
        return "Short Credit History Borrowers"
    return "Low-Risk Stable Borrowers"


def predict_segment_for_record(record: pd.DataFrame, segmentation_artifacts):
    if isinstance(record, pd.DataFrame):
        feature_df = record[SEGMENT_FEATURES]
    else:
        feature_df = pd.DataFrame([record])[SEGMENT_FEATURES]
    imputed = pd.DataFrame(segmentation_artifacts["imputer"].transform(feature_df), columns=feature_df.columns)
    scaled = segmentation_artifacts["scaler"].transform(imputed)
    segment_id = int(segmentation_artifacts["model"].predict(scaled)[0])
    profile = segmentation_artifacts["segment_profile"][segmentation_artifacts["segment_profile"]["segment_id"] == segment_id].iloc[0]
    return segment_id, profile["Segment Name"]


def predict_single_borrower(record: dict, final_model, threshold, segmentation_artifacts):
    borrower_df = pd.DataFrame([record])
    features = borrower_df[[
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
    ]]
    default_prob = float(final_model.predict_proba(features)[0, 1])
    no_default_prob = 1 - default_prob
    risk_label = risk_band(default_prob)
    segment_id, segment_name = predict_segment_for_record(features, segmentation_artifacts)
    return {
        "default_probability": default_prob,
        "no_default_probability": no_default_prob,
        "risk_band": risk_label,
        "segment_id": segment_id,
        "segment_name": segment_name,
        "selected_model": final_model,  # kept for downstream UI display
        "threshold": threshold,
        "features": features,
    }


def bulk_predict_records(input_df: pd.DataFrame, final_model, threshold, segmentation_artifacts):
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
    missing = [c for c in required if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required borrower columns: {missing}")

    results = input_df.copy()
    results["default_probability"] = final_model.predict_proba(input_df[required])[:, 1]
    results["no_default_probability"] = 1 - results["default_probability"]
    results["risk_band"] = results["default_probability"].apply(risk_band)
    results["segment_id"] = None
    results["segment_name"] = None

    for idx, row in results.iterrows():
        segment_id, segment_name = predict_segment_for_record(row.to_dict(), segmentation_artifacts)
        results.at[idx, "segment_id"] = segment_id
        results.at[idx, "segment_name"] = segment_name

    results = results.sort_values("default_probability", ascending=False).reset_index(drop=True)
    return results


def load_training_data(path: str | None = None) -> pd.DataFrame:
    return clean_dataset(pd.read_csv(path or DATASET_PATH))
