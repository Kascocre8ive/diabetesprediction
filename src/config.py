"""
Central configuration for the Hybrid Ensemble Diabetes Prediction project.

Every other module (preprocessing, train_models, ensemble, app.py) imports
its paths, feature lists, and constants from here so there is exactly one
source of truth.
"""
import os

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Kaggle: iammustafatz/diabetes-prediction-dataset
# Rename the downloaded CSV to this filename and place it in data/
RAW_CSV_PATH = os.path.join(DATA_DIR, "diabetes_prediction_dataset.csv")

TARGET_COL = "diabetes"

NUMERIC_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Train/test/val split + reproducibility
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Saved-artifact paths
# ---------------------------------------------------------------------------
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_model.joblib")

METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
EVAL_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "eval_artifacts.joblib")