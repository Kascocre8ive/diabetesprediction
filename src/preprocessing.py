"""
Step 1 of the pipeline: load the raw Kaggle CSV, clean it, build a
reusable preprocessing pipeline (impute -> encode -> scale), split into
train/val/test, and balance the TRAINING split only with SMOTE.

Run directly:
    python -m src.preprocessing
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def load_data(path: str = config.RAW_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download it from Kaggle (iammustafatz/diabetes-prediction-dataset) "
            f"and place the CSV at that path (rename to "
            f"'diabetes_prediction_dataset.csv')."
        )
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Physiological validation + outlier capping, per proposal Section 3.2.1."""
    df = df.copy()

    # Drop exact duplicate rows (common in this dataset)
    df = df.drop_duplicates()

    # Physiologically implausible zero/negative readings -> treat as missing
    for col in ["bmi", "HbA1c_level", "blood_glucose_level", "age"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    # Median imputation for any missing numeric values introduced above
    for col in config.NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # IQR-based outlier capping (winsorizing) for numeric features
    for col in config.NUMERIC_FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Normalize smoking_history text categories seen in the raw dataset
    if "smoking_history" in df.columns:
        df["smoking_history"] = df["smoking_history"].replace(
            {"ever": "former", "not current": "former"}
        )

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, config.NUMERIC_FEATURES),
        ("bin", "passthrough", config.BINARY_FEATURES),
        ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable column names for the transformed matrix (used for
    XGBoost feature-importance reporting in the app)."""
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(config.CATEGORICAL_FEATURES)
    )
    return config.NUMERIC_FEATURES + config.BINARY_FEATURES + cat_names


def run():
    print("Loading raw data...")
    df = load_data()
    print(f"  {len(df):,} rows loaded")

    df = clean_data(df)
    print(f"  {len(df):,} rows after cleaning/deduplication")

    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_COL].astype(int)

    # 3-way split: train (for fitting base models) / val (for tuning the
    # ensemble's voting weights) / test (held out, touched only once, at
    # final evaluation).
    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainfull, y_trainfull, test_size=0.15,
        stratify=y_trainfull, random_state=config.RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    y_train_arr = y_train.to_numpy()
    print("Balancing training data with SMOTE "
          f"(before: {np.bincount(y_train_arr)})...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train_arr)
    print(f"  after SMOTE: {np.bincount(y_train_bal)}")

    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    joblib.dump(
        {
            "X_train": X_train_bal,
            "y_train": y_train_bal,
            "X_val": X_val_t,
            "y_val": y_val.to_numpy(),
            "X_test": X_test_t,
            "y_test": y_test.to_numpy(),
            "feature_names": feature_names,
        },
        os.path.join(config.MODEL_DIR, "splits.pkl"),
    )
    print(f"Saved preprocessor -> {config.PREPROCESSOR_PATH}")
    print(f"Saved train/test splits -> {os.path.join(config.MODEL_DIR, 'splits.pkl')}")


if __name__ == "__main__":
    run()