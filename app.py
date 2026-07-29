"""
Streamlit front-end for the Hybrid Ensemble Diabetes Prediction system
(SVM + Logistic Regression + XGBoost, soft-voting ensemble).

5 pages (sidebar navigation):
1. Dataset Overview   - EDA: distributions, class balance, correlation heatmap
2. Predict            - single-patient risk prediction
3. Batch Prediction    - upload a CSV of many patients, get risk for each, download results
4. Model Performance  - metrics, confusion matrices, ROC/PR curves, McNemar's test
5. Thesis Report      - auto-generated write-up (downloadable) for Chapters 4 & 5

Run:
    streamlit run app.py
"""
import os
import sys
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.ensemble import SoftVotingEnsemble  # noqa: F401 (needed for joblib.load)
from src.preprocessing import load_data, clean_data

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

MODEL_ORDER = ["SVM", "Logistic Regression", "XGBoost", "Hybrid Ensemble"]
MODEL_COLORS = {
    "SVM": "#636EFA",
    "Logistic Regression": "#EF553B",
    "XGBoost": "#00CC96",
    "Hybrid Ensemble": "#AB63FA",
}

# ---------------------------------------------------------------------------
# Light custom styling — bigger metric cards, tidier headers, accent color
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #eaecef;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
    }
    div[data-testid="stMetricLabel"] {font-weight: 600; opacity: 0.75;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = [p for p in [config.PREPROCESSOR_PATH, config.ENSEMBLE_MODEL_PATH] if not os.path.exists(p)]
    if missing:
        return None
    return {
        "preprocessor": joblib.load(config.PREPROCESSOR_PATH),
        "ensemble": joblib.load(config.ENSEMBLE_MODEL_PATH),
    }


@st.cache_data
def load_metrics():
    if not os.path.exists(config.METRICS_PATH):
        return None
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    if not os.path.exists(config.FEATURE_IMPORTANCE_PATH):
        return None
    with open(config.FEATURE_IMPORTANCE_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_eval_artifacts():
    if not os.path.exists(config.EVAL_ARTIFACTS_PATH):
        return None
    return joblib.load(config.EVAL_ARTIFACTS_PATH)


@st.cache_data
def load_raw_dataset():
    if not os.path.exists(config.RAW_CSV_PATH):
        return None
    df = load_data()
    df = clean_data(df)
    return df


artifacts = load_artifacts()
metrics_data = load_metrics()
feature_importance = load_feature_importance()
eval_artifacts = load_eval_artifacts()
raw_df = load_raw_dataset()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Predictor")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dataset Overview",
        "🔍 Predict",
        "🧪 Batch Prediction",
        "📊 Model Performance",
        "📄 Thesis Report",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Hybrid Ensemble: SVM + Logistic Regression + XGBoost (soft voting)")
if metrics_data:
    best_model = max(metrics_data["metrics"].items(), key=lambda kv: kv[1]["f1_score"])[0]
    st.sidebar.success(f"Best F1-score: {best_model}")

with st.sidebar.expander("ℹ️ About this project"):
    st.write(
        "This app predicts diabetes risk from routine clinical and lifestyle "
        "variables (age, BMI, HbA1c, blood glucose, hypertension, heart "
        "disease, gender, smoking history) using three base classifiers "
        "combined with a weighted soft-voting ensemble. Results here are "
        "for educational/research purposes only and are not a medical "
        "diagnosis."
    )

# ===========================================================================
# PAGE 1 — Dataset Overview (EDA)
# ===========================================================================
if page == "🏠 Dataset Overview":
    st.title("🏠 Dataset Overview")

    if raw_df is None:
        st.error(f"Dataset not found at {config.RAW_CSV_PATH}. Place the CSV there first.")
        st.stop()

    n_rows, n_cols = raw_df.shape
    prevalence = raw_df[config.TARGET_COL].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{n_rows:,}")
    c2.metric("Features", n_cols - 1)
    c3.metric("Diabetes prevalence", f"{prevalence:.1f}%")
    c4.metric("Missing values", int(raw_df.isna().sum().sum()))

    st.subheader("Sample of the cleaned dataset")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Descriptive statistics (numeric features)")
    st.dataframe(raw_df[config.NUMERIC_FEATURES].describe().T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        counts = raw_df[config.TARGET_COL].value_counts().rename({0: "No Diabetes", 1: "Diabetes"})
        fig = px.pie(
            values=counts.values, names=counts.index, hole=0.4,
            color=counts.index,
            color_discrete_map={"No Diabetes": "#00CC96", "Diabetes": "#EF553B"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation heatmap")
        corr_cols = config.NUMERIC_FEATURES + config.BINARY_FEATURES + [config.TARGET_COL]
        corr = raw_df[corr_cols].corr()
        fig_hm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
        )
        st.pyplot(fig_hm, use_container_width=True)

    st.subheader("Feature distributions by diabetes status")
    dist_col = st.selectbox("Choose a numeric feature", config.NUMERIC_FEATURES, key="dist_feature")
    fig_dist = px.histogram(
        raw_df, x=dist_col, color=raw_df[config.TARGET_COL].map({0: "No Diabetes", 1: "Diabetes"}),
        barmode="overlay", opacity=0.6, nbins=40,
        color_discrete_map={"No Diabetes": "#00CC96", "Diabetes": "#EF553B"},
        labels={"color": "Status"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Diabetes rate by category")
    cat_col1, cat_col2 = st.columns(2)
    for col, ui_col in zip(config.CATEGORICAL_FEATURES, [cat_col1, cat_col2]):
        rate = raw_df.groupby(col)[config.TARGET_COL].mean().sort_values(ascending=False) * 100
        fig_cat = px.bar(
            x=rate.index, y=rate.values, labels={"x": col, "y": "Diabetes rate (%)"},
            title=f"Diabetes rate by {col}", color=rate.values, color_continuous_scale="Reds",
        )
        ui_col.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("Hypertension / Heart disease vs. diabetes"):
        bc1, bc2 = st.columns(2)
        for col, ui_col in zip(config.BINARY_FEATURES, [bc1, bc2]):
            rate = raw_df.groupby(col)[config.TARGET_COL].mean() * 100
            rate.index = rate.index.map({0: "No", 1: "Yes"})
            fig_b = px.bar(x=rate.index, y=rate.values, labels={"x": col, "y": "Diabetes rate (%)"})
            ui_col.plotly_chart(fig_b, use_container_width=True)

# ===========================================================================
# PAGE 2 — Predict
# ===========================================================================
elif page == "🔍 Predict":
    st.title("🔍 Predict Diabetes Risk")

    if artifacts is None:
        st.error(
            "No trained model found. Run the training pipeline first:\n\n"
            "```\npython -m src.preprocessing\npython -m src.train_models\n```"
        )
        st.stop()

    st.subheader("Enter patient information")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 1, 100, 40)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    with col2:
        hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120)
        smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "No Info"])
    with col3:
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age, "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "gender": gender, "smoking_history": smoking_history,
        }])[config.ALL_FEATURES]

        X = artifacts["preprocessor"].transform(row)
        ensemble = artifacts["ensemble"]
        ens_proba = float(ensemble.predict_proba(X)[0, 1])
        per_model = {name: float(p[0]) for name, p in ensemble.per_model_proba(X).items()}

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=ens_proba * 100,
                title={"text": "Hybrid Ensemble — Diabetes Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if ens_proba >= 0.5 else "seagreen"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            if ens_proba >= 0.6:
                st.error(f"⚠️ Higher risk of diabetes (probability {ens_proba:.1%}). Not a medical diagnosis.")
            elif ens_proba >= 0.3:
                st.warning(f"🟡 Moderate risk of diabetes (probability {ens_proba:.1%}). Not a medical diagnosis.")
            else:
                st.success(f"✅ Lower risk of diabetes (probability {ens_proba:.1%}). Not a medical diagnosis.")

            n_agree_high = sum(1 for p in per_model.values() if p >= 0.5)
            st.caption(f"Model agreement: {n_agree_high}/3 base models flag elevated risk (≥50%).")

        with res_col2:
            st.markdown("Individual model predictions")
            model_df = pd.DataFrame({
                "Model": list(per_model.keys()) + ["Hybrid Ensemble"],
                "Diabetes Probability": [v * 100 for v in per_model.values()] + [ens_proba * 100],
            })
            fig2 = px.bar(
                model_df, x="Model", y="Diabetes Probability", color="Model",
                range_y=[0, 100], text_auto=".1f",
                color_discrete_map=MODEL_COLORS,
            )
            fig2.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        if feature_importance:
            with st.expander("Why this prediction? (top global risk factors)"):
                top5 = list(feature_importance.items())[:5]
                fi_names = [f for f, _ in top5]
                fi_df = pd.DataFrame(top5, columns=["Feature", "Importance"])
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
                fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=260)
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption(
                    "These are the features the XGBoost model relies on most across "
                    "the whole dataset, not specifically for this patient — useful "
                    "context, not a per-patient explanation."
                )

# ===========================================================================
# PAGE 3 — Batch Prediction
# ===========================================================================
elif page == "🧪 Batch Prediction":
    st.title("🧪 Batch Prediction")
    st.caption(
        "Upload a CSV containing one row per patient with the columns: "
        + ", ".join(config.ALL_FEATURES)
    )

    if artifacts is None:
        st.error(
            "No trained model found. Run the training pipeline first:\n\n"
            "```\npython -m src.preprocessing\npython -m src.train_models\n```"
        )
        st.stop()

    template_df = pd.DataFrame([{
        "age": 45, "bmi": 27.5, "HbA1c_level": 6.1, "blood_glucose_level": 140,
        "hypertension": 0, "heart_disease": 0, "gender": "Female", "smoking_history": "never",
    }])
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="batch_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_cols = [c for c in config.ALL_FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required column(s): {', '.join(missing_cols)}")
                st.stop()

            X_batch = artifacts["preprocessor"].transform(batch_df[config.ALL_FEATURES])
            ensemble = artifacts["ensemble"]
            proba = ensemble.predict_proba(X_batch)[:, 1]

            results_df = batch_df.copy()
            results_df["diabetes_risk_probability"] = (proba * 100).round(2)
            results_df["risk_category"] = pd.cut(
                proba, bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Moderate", "High"]
            )

            st.success(f"Scored {len(results_df):,} patients.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Low risk", int((results_df["risk_category"] == "Low").sum()))
            m2.metric("Moderate risk", int((results_df["risk_category"] == "Moderate").sum()))
            m3.metric("High risk", int((results_df["risk_category"] == "High").sum()))

            st.dataframe(results_df, use_container_width=True)

            fig_hist = px.histogram(
                results_df, x="diabetes_risk_probability", nbins=30,
                color="risk_category",
                color_discrete_map={"Low": "#00CC96", "Moderate": "#FFA15A", "High": "#EF553B"},
                labels={"diabetes_risk_probability": "Predicted diabetes risk (%)"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.download_button(
                "⬇️ Download results as CSV",
                data=results_df.to_csv(index=False),
                file_name="batch_prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")

# ===========================================================================
# PAGE 4 — Model Performance
# ===========================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    if metrics_data is None or eval_artifacts is None:
        st.info("No evaluation artifacts found yet — run `python -m src.train_models`.")
        st.stop()

    st.subheader("Test-set performance metrics")
    m_df = pd.DataFrame(metrics_data["metrics"]).T.reindex(MODEL_ORDER)
    st.dataframe(m_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    fig_bar = px.bar(
        m_df.reset_index().melt(id_vars="index", var_name="metric", value_name="score"),
        x="index", y="score", color="metric", barmode="group",
        labels={"index": "Model"}, color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("Confusion matrices")
    cm_cols = st.columns(4)
    for ui_col, name in zip(cm_cols, MODEL_ORDER):
        cm = np.array(eval_artifacts["models"][name]["confusion_matrix"])
        fig_cm, ax = plt.subplots(figsize=(3.2, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10)
        ui_col.pyplot(fig_cm, use_container_width=True)

    st.divider()
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.subheader("ROC curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), name="Random (AUC=0.50)",
        ))
        for name in MODEL_ORDER:
            d = eval_artifacts["models"][name]
            auc = metrics_data["metrics"][name]["auc_roc"]
            fig_roc.add_trace(go.Scatter(
                x=d["roc_fpr"], y=d["roc_tpr"], mode="lines",
                name=f"{name} (AUC={auc:.3f})",
                line=dict(color=MODEL_COLORS[name]),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with curve_col2:
        st.subheader("Precision-Recall curves")
        fig_pr = go.Figure()
        for name in MODEL_ORDER:
            d = eval_artifacts["models"][name]
            fig_pr.add_trace(go.Scatter(
                x=d["pr_recall"], y=d["pr_precision"], mode="lines",
                name=name, line=dict(color=MODEL_COLORS[name]),
            ))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=420)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()
    st.subheader("McNemar's test — Hybrid Ensemble vs. each base model")
    mc_df = pd.DataFrame(metrics_data["mcnemar"]).T
    mc_df["significant (p<0.05)"] = mc_df["p_value"] < 0.05
    st.dataframe(mc_df, use_container_width=True)

    fig_mc = px.bar(
        mc_df.reset_index(), x="index", y="p_value",
        labels={"index": "Comparison"}, title="McNemar's test p-values (0.05 significance line)",
    )
    fig_mc.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="α = 0.05")
    st.plotly_chart(fig_mc, use_container_width=True)

    st.divider()
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.subheader("Ensemble voting weights")
        weights = metrics_data["ensemble_weights"]
        fig_w = px.pie(values=list(weights.values()), names=list(weights.keys()), hole=0.4)
        st.plotly_chart(fig_w, use_container_width=True)

    with perf_col2:
        if feature_importance:
            st.subheader("XGBoost feature importance")
            fi_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_fi, use_container_width=True)

# ===========================================================================
# PAGE 5 — Thesis Report (auto-generated write-up for Chapters 4 & 5)
# ===========================================================================
elif page == "📄 Thesis Report":
    st.title("📄 Auto-Generated Report (for Chapters 4 & 5)")
    st.caption(
        "A structured draft you can adapt into your Results/Discussion and "
        "Conclusion chapters. Numbers are pulled directly from your trained models."
    )

    if metrics_data is None:
        st.info("No metrics found yet — run `python -m src.train_models`.")
        st.stop()

    m = metrics_data["metrics"]
    mc = metrics_data["mcnemar"]
    weights = metrics_data["ensemble_weights"]
    best_model = max(m.items(), key=lambda kv: kv[1]["f1_score"])[0]
    ens = m["Hybrid Ensemble"]

    lines = []
    lines.append("# Diabetes Prediction — Results Report")
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if raw_df is not None:
        n_rows = len(raw_df)
        prevalence = raw_df[config.TARGET_COL].mean() * 100
        lines.append("## 4.1 Dataset Summary")
        lines.append(
            f"The dataset used contains {n_rows:,} patient records after cleaning and "
            f"deduplication, with a diabetes prevalence of {prevalence:.1f}%. Features include "
            f"{', '.join(config.NUMERIC_FEATURES)} (numeric), {', '.join(config.BINARY_FEATURES)} "
            f"(binary), and {', '.join(config.CATEGORICAL_FEATURES)} (categorical). "
            f"Class imbalance in the training split was addressed using SMOTE, applied only to "
            f"the training partition to avoid data leakage into validation/test sets.\n"
        )

    lines.append("## 4.2 Model Performance Comparison")
    lines.append("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |")
    lines.append("|---|---|---|---|---|---|")
    for name in MODEL_ORDER:
        v = m[name]
        lines.append(
            f"| {name} | {v['accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} "
            f"| {v['f1_score']:.4f} | {v['auc_roc']:.4f} |"
        )
    lines.append(
        f"\nAmong the individual and combined models evaluated on the held-out test set, "
        f"{best_model} achieved the highest F1-score "
        f"({m[best_model]['f1_score']:.4f}). The Hybrid Ensemble combined SVM, Logistic "
        f"Regression, and XGBoost using soft voting with weights "
        f"(SVM={weights.get('SVM')}, LR={weights.get('Logistic Regression')}, "
        f"XGBoost={weights.get('XGBoost')}), tuned on a held-out validation split, and reached "
        f"an accuracy of {ens['accuracy']:.4f} and AUC-ROC of {ens['auc_roc']:.4f} on the test set.\n"
    )

    lines.append("## 4.3 Statistical Significance (McNemar's Test)")
    lines.append(
        "McNemar's test was used to determine whether the Hybrid Ensemble's prediction errors "
        "differ significantly from each base model's, using discordant prediction pairs on the "
        "test set:\n"
    )
    for comparison, res in mc.items():
        sig = "a statistically significant" if res["p_value"] < 0.05 else "no statistically significant"
        lines.append(
            f"- {comparison}: b={res['b']}, c={res['c']}, p-value={res['p_value']} "
            f"→ {sig} difference in error rates (α = 0.05)."
        )
    lines.append("")

    if feature_importance:
        top5 = list(feature_importance.items())[:5]
        lines.append("## 4.4 Key Predictive Features")
        lines.append(
            "Based on XGBoost's feature importance scores, the top predictors of diabetes "
            "risk in this dataset were:\n"
        )
        for i, (feat, score) in enumerate(top5, 1):
            lines.append(f"{i}. {feat} (importance = {score:.4f})")
        lines.append("")

    lines.append("## 5. Discussion & Conclusion (draft)")
    improves = ens["f1_score"] >= max(m[n]["f1_score"] for n in ["SVM", "Logistic Regression", "XGBoost"])
    any_sig = any(r["p_value"] < 0.05 for r in mc.values())
    lines.append(
        f"The results indicate that combining SVM, Logistic Regression, and XGBoost through a "
        f"soft-voting ensemble {'improves' if improves else 'does not clearly improve'} "
        f"predictive performance relative to the best individual base model on this dataset. "
        f"{'This improvement was statistically significant against at least one base model, ' if any_sig else 'However, McNemar tests did not show a statistically significant difference against the base models, suggesting the ensemble and base models make comparably distributed errors, '}"
        f"which should be discussed in light of the dataset size, class balance strategy (SMOTE), "
        f"and the computational cost of each model — particularly the SVM's slower training time "
        f"on large datasets, noted during implementation. Future work could explore additional "
        f"ensemble weighting strategies, external validation on other populations, and cost-sensitive "
        f"evaluation given the clinical asymmetry between false negatives and false positives."
    )

    report_text = "\n".join(lines)
    st.markdown(report_text)

    st.download_button(
        "⬇️ Download report as Markdown",
        data=report_text,
        file_name="diabetes_project_report.md",
        mime="text/markdown",
        use_container_width=True,
    )