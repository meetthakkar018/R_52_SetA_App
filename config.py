from pathlib import Path

ROLL_NUMBER = 52
PROJECT_NAME = "Loan Default Risk Prediction and Borrower Segmentation Platform"
ROOT_DIR = Path(__file__).resolve().parent

LOCAL_DATA_CANDIDATES = [
    Path(r"D:\roll 52\data\credit_risk_dataset.csv"),
    Path(r"D:\roll52\data\credit_risk_dataset.csv"),
    Path(r"D:\roo52\data\credit_risk_dataset.csv"),
    ROOT_DIR / "data" / "credit_risk_dataset.csv",
    Path("data") / "credit_risk_dataset.csv",
]
DATASET_PATH = next((p for p in LOCAL_DATA_CANDIDATES if p.exists()), ROOT_DIR / "data" / "credit_risk_dataset.csv")
MODEL_DIR = ROOT_DIR / "model"
TEST_SIZE = 0.20
RANDOM_STATE = 52
K = 5
TARGET_COLUMN = "loan_status"
RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}

REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

SEGMENT_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

MODEL_ARTIFACTS = {
    "logistic_regression": "logistic_regression_model.joblib",
    "random_forest": "random_forest_model.joblib",
    "gradient_boosting": "gradient_boosting_model.joblib",
    "final_model": "final_model.joblib",
    "threshold": "selected_threshold.joblib",
    "comparison": "comparison_table.joblib",
    "training_metadata": "training_metadata.joblib",
    "kmeans": "kmeans_model.joblib",
    "scaler": "segment_scaler.joblib",
    "imputer": "segment_imputer.joblib",
}
