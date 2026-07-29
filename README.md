# Hybrid Ensemble Diabetes Prediction (SVM + Logistic Regression + XGBoost)

Streamlit implementation of the project proposal *"Predicting Diabetes
Using a Hybrid Ensemble of SVM, Logistic Regression, and XGBoost"*.

## 1. Project structure

```
diabetes_app/
├── app.py                     # Streamlit UI (prediction + performance dashboard)
├── requirements.txt
├── data/
│   └── diabetes_prediction_dataset.csv   <- put the Kaggle CSV here
├── models/                    # created automatically by the scripts below
│   ├── preprocessor.pkl
│   ├── splits.pkl
│   ├── svm_model.pkl
│   ├── lr_model.pkl
│   ├── xgb_model.pkl
│   ├── ensemble_model.pkl
│   ├── metrics.json
│   └── feature_importance.json
└── src/
    ├── config.py               # paths + feature list (edit here if needed)
    ├── preprocessing.py        # Step 1: clean, encode, scale, split, SMOTE
    ├── train_models.py         # Step 2: tune + train SVM/LR/XGBoost + ensemble
    └── ensemble.py             # SoftVotingEnsemble class used by both scripts
```

## 2. Get the dataset

Dataset: **Diabetes Prediction Dataset** (Kaggle, ~100,000 rows)
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

1. Download `diabetes_prediction_dataset.csv` from the link above (Kaggle
   account required).
2. Place it at:
   ```
   diabetes_app/data/diabetes_prediction_dataset.csv
   ```

   (the folder/filename must match exactly, or update `RAW_CSV_PATH` in
   `src/config.py`).

## 3. Environment setup

```bash
cd diabetes_app
python -m venv venv

# activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**streamlit run app.pyLibraries used** (all in `requirements.txt`):
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`,
`scipy`, `matplotlib`, `seaborn`, `streamlit`, `plotly`.

## 4. Run the pipeline (in order)

**Step 1 — Preprocess the data** (cleaning, encoding, scaling, train/val/test
split, SMOTE on the training split only):

```bash
python -m src.preprocessing
```

This creates `models/preprocessor.pkl` and `models/splits.pkl`.

**Step 2 — Train the models** (SVM, Logistic Regression, XGBoost, hyperparameter
tuning, soft-voting ensemble, evaluation, McNemar's test):

```bash
python -m src.train_models
```

This creates the `.pkl` model files plus `models/metrics.json` and
`models/feature_importance.json`. Expect this to take a while — SVM tuning
is the slowest part (see the note below).

**Step 3 — Launch the app:**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with two tabs:

- **Predict** — enter a patient's details and get a risk probability from
  the hybrid ensemble plus each individual base model.
- **Model Performance** — Accuracy/Precision/Recall/F1/AUC-ROC table,
  McNemar's test results, ensemble weights, and XGBoost feature importance.

## 5. Important implementation notes

- **Why a subsample for SVM?** An RBF-kernel SVM scales roughly
  quadratically-to-cubically with the number of rows. On the full ~85,000-row
  balanced training set, a hyperparameter search can take extremely long
  (your own literature review — Fadli Kurniawan & Megawaty, 2025 — reports
  one SVM run taking ~3500 seconds). `src/train_models.py` therefore trains/tunes
  SVM on a stratified subsample (`SVM_TRAIN_SAMPLE_SIZE = 15000` at the top
  of the file). You can raise this, or set it to `None` to use the full
  set, if you have more time/compute — this is a good discussion point for
  Chapter 4 (computational efficiency trade-off).
- **SMOTE is applied only to the training split**, never to validation or
  test, to avoid data leakage — exactly as described in Section 3.2.1 of
  the proposal.
- **Ensemble weights** (SVM : LR : XGBoost) are chosen by a small grid
  search evaluated on a held-out validation split, then the final metrics
  in `metrics.json` are computed once on the untouched test split.
- **McNemar's test** is implemented manually in `train_models.py` (exact
  binomial for small discordant-pair counts, chi-square with continuity
  correction otherwise) — no extra statistics library required.
- To retrain after changing hyperparameter grids or the SVM subsample size,
  just rerun `python -m src.train_models` (no need to rerun preprocessing
  unless you change cleaning/encoding logic).

## 6. Troubleshooting

- **"Dataset not found"** — check the CSV is at
  `data/diabetes_prediction_dataset.csv` exactly.
- **App shows "No trained model found"** — you skipped Steps 1–2; run them
  first, then relaunch `streamlit run app.py`.
- **Training is too slow** — lower `SVM_TRAIN_SAMPLE_SIZE`, `N_ITER_SEARCH`,
  or `CV_FOLDS` near the top of `src/train_models.py`.
