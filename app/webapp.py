"""
webapp.py - Smart AutoML v2.0
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_data, get_file_info
from preprocessing import preprocess_data, split_data, analyze_dataset
from feature_selection import select_features
from feature_engineering import engineer_features
from model_trainer import train_and_select_best, save_model, get_feature_importance
from explainability import get_shap_explainer, plot_shap_summary, plot_shap_bar
from advanced_viz import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall,
    plot_residuals, plot_model_comparison, plot_cv_comparison
)
from profiling import generate_profile_report
from utils import (
    plot_histograms, plot_correlation_heatmap, plot_target_distribution,
    plot_feature_importance, model_to_bytes, format_metrics,
)
from api import save_model_metadata

st.set_page_config(page_title="Smart AutoML v2", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size:2.3rem; font-weight:800; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.1rem; }
    .sub-header { color:#777; font-size:.95rem; margin-bottom:1.2rem; }
    .section-title { font-size:1.2rem; font-weight:600; color:#2d3748; border-bottom:2px solid #667eea; padding-bottom:.35rem; margin:1.5rem 0 1rem 0; }
    .badge { border-radius:20px; padding:3px 12px; font-size:.78rem; font-weight:600; }
    .badge-classification { background:#ebf4ff; color:#3182ce; }
    .badge-regression { background:#f0fff4; color:#38a169; }
    .badge-best { background:#fef3c7; color:#d97706; }
    .info-box { background:linear-gradient(135deg,#f0f4ff,#faf5ff); border-left:3px solid #667eea; border-radius:8px; padding:10px 14px; font-size:13px; margin:8px 0; }
    div[data-testid="stMetricValue"] { font-size:1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in {
    "df": None, "analysis": None, "train_result": None,
    "X_selected": None, "y": None, "X_test": None, "y_test": None,
    "X_train": None, "y_train": None,
    "feature_names": None, "label_encoder": None,
    "shap_ready": False, "fe_summary": {}, "profile_html": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Smart AutoML v2")
    st.markdown("---")
    st.markdown("### 📁 Data Source")
    src_tab = st.radio("Source type", ["Upload file", "Paste CSV text", "Sample dataset"], label_visibility="collapsed")
    df_loaded = None

    if src_tab == "Upload file":
        uploaded = st.file_uploader("CSV / Excel / JSON", type=["csv","xlsx","xls","json"])
        if uploaded:
            try:
                df_loaded = load_data(uploaded)
                st.success(f"✅ {df_loaded.shape[0]:,} rows × {df_loaded.shape[1]} cols")
            except ValueError as e:
                st.error(str(e))

    elif src_tab == "Paste CSV text":
        csv_text = st.text_area("Paste CSV content here", height=140, placeholder="col1,col2,target\n1,2,0\n3,4,1")
        if csv_text.strip():
            try:
                import io as _io
                df_loaded = pd.read_csv(_io.StringIO(csv_text))
                st.success(f"✅ {df_loaded.shape[0]} rows parsed")
            except Exception as e:
                st.error(f"Parse error: {e}")

    else:
        sample = st.selectbox("Choose dataset", ["Iris (classification)", "California Housing (regression)"])
        if st.button("Load sample"):
            if "Iris" in sample:
                from sklearn.datasets import load_iris
                d = load_iris(as_frame=True)
                df_loaded = pd.concat([d.data, d.target.rename("species")], axis=1)
            else:
                from sklearn.datasets import fetch_california_housing
                d = fetch_california_housing(as_frame=True)
                df_loaded = pd.concat([d.data, d.target.rename("price")], axis=1)
            st.success(f"✅ Sample loaded: {df_loaded.shape}")

    if df_loaded is not None:
        st.session_state.df = df_loaded

    st.markdown("---")
    target_col, cv_folds, test_size = None, 3, 0.2
    fe_log = fe_poly = fe_inter = False
    run_btn = False

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 🎯 Configuration")
        target_col = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1)
        cv_folds = st.slider("CV folds", 2, 10, 3)
        test_size = st.slider("Test set %", 10, 40, 20, step=5) / 100
        st.markdown("### ⚙️ Feature Engineering")
        fe_log  = st.checkbox("Log transform (skewed cols)", value=False)
        fe_poly = st.checkbox("Polynomial features (degree 2)", value=False)
        fe_inter = st.checkbox("Interaction terms", value=False)
        st.markdown("---")
        run_btn = st.button("🚀 Run AutoML", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("<small style='color:#aaa'>Smart AutoML v2.0 · scikit-learn · SHAP · FastAPI</small>", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 Smart AutoML v2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Auto train · Tune · Explain · Deploy — any dataset, zero config.</div>', unsafe_allow_html=True)

if st.session_state.df is None:
    c1,c2,c3 = st.columns(3)
    c1.info("📁 **Step 1** — Upload dataset or use a sample")
    c2.info("🎯 **Step 2** — Select target column in sidebar")
    c3.info("🚀 **Step 3** — Click 'Run AutoML'")
    st.stop()

df = st.session_state.df

# ── Section 1: Dataset Preview ────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
info = get_file_info(df)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", f"{info['rows']:,}")
c2.metric("Columns", info['columns'])
c3.metric("Missing cells", sum(info['missing_values'].values()))
c4.metric("Duplicate rows", info['duplicates'])

tab_data, tab_types, tab_profile = st.tabs(["Raw data", "Column types", "Auto EDA Report"])
with tab_data:
    st.dataframe(df.head(200), use_container_width=True)
with tab_types:
    dtype_df = pd.DataFrame({
        "Column": list(info['dtypes'].keys()),
        "Type": list(info['dtypes'].values()),
        "Missing": [info['missing_values'][c] for c in info['dtypes'].keys()],
        "Unique": [df[c].nunique() for c in info['dtypes'].keys()],
    })
    st.dataframe(dtype_df, use_container_width=True)
with tab_profile:
    if st.button("🧪 Generate Full EDA Report"):
        with st.spinner("Building profiling report..."):
            html = generate_profile_report(df, title="Dataset EDA Report")
            st.session_state.profile_html = html
            st.success("Report ready!")
    if st.session_state.profile_html:
        st.components.v1.html(st.session_state.profile_html, height=800, scrolling=True)
        st.download_button("⬇️ Download EDA Report (HTML)", data=st.session_state.profile_html,
                           file_name="eda_report.html", mime="text/html")

# ── Section 2: Visualizations ─────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Data Visualizations</div>', unsafe_allow_html=True)
tab_hist, tab_corr = st.tabs(["Histograms", "Correlation Heatmap"])
with tab_hist:
    fig = plot_histograms(df)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("No numeric columns.")
with tab_corr:
    fig = plot_correlation_heatmap(df)
    if fig:
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Need ≥2 numeric columns.")

# ── AutoML Pipeline ───────────────────────────────────────────────────────────
if run_btn:
    if not target_col:
        st.error("Select a target column."); st.stop()

    progress = st.progress(0, text="Analyzing dataset...")
    try:
        analysis = analyze_dataset(df, target_col)
        st.session_state.analysis = analysis
    except Exception as e:
        st.error(f"Analysis failed: {e}"); st.stop()
    progress.progress(15, "Preprocessing...")

    try:
        X, y, label_encoder, feature_names, removed = preprocess_data(df, target_col)
        st.session_state.y = y
        st.session_state.label_encoder = label_encoder
        if removed:
            st.info(f"ℹ️ Removed {removed} duplicate rows.")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}"); st.stop()
    progress.progress(30, "Feature engineering...")

    if any([fe_log, fe_poly, fe_inter]):
        try:
            X, fe_summary = engineer_features(X, {"log_transform": fe_log, "polynomial": fe_poly,
                                                    "interactions": fe_inter, "top_n": 5, "poly_degree": 2})
            st.session_state.fe_summary = fe_summary
        except Exception as e:
            st.warning(f"Feature engineering skipped: {e}")
    progress.progress(45, "Selecting features...")

    try:
        X_selected, selected_features, _ = select_features(X, y, analysis["problem_type"])
        st.session_state.feature_names = selected_features
    except Exception as e:
        st.warning(f"Feature selection skipped: {e}")
        X_selected, selected_features = X, feature_names
        st.session_state.feature_names = feature_names
    progress.progress(55, "Training models with GridSearchCV...")

    try:
        X_train, X_test, y_train, y_test = split_data(X_selected, y, test_size=test_size)
        for k, v in [("X_selected", X_selected), ("X_train", X_train), ("X_test", X_test),
                     ("y_train", y_train), ("y_test", y_test)]:
            st.session_state[k] = v
        result = train_and_select_best(X_train, X_test, y_train, y_test, analysis, cv=cv_folds)
        st.session_state.train_result = result
        save_model(result["best_model"], path="models/model.pkl")
        save_model_metadata(result["best_model"], selected_features, analysis["problem_type"],
                            label_encoder, path="models/model_meta.json")
        progress.progress(100, "Done!")
        st.success(f"✅ Done! Best model: **{result['best_name']}**")
    except Exception as e:
        st.error(f"Training failed: {e}"); st.stop()

# ── Section 3: Model Results ──────────────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    analysis = st.session_state.analysis
    st.markdown('<div class="section-title">🏆 Model Results</div>', unsafe_allow_html=True)
    badge = "badge-classification" if result["problem_type"] == "classification" else "badge-regression"
    st.markdown(
        f'<span class="badge {badge}">{result["problem_type"].capitalize()}</span> &nbsp;'
        f'<span class="badge badge-best">🥇 {result["best_name"]}</span> &nbsp;'
        f'<small style="color:#888">Features used: {len(st.session_state.feature_names)}</small>',
        unsafe_allow_html=True)
    st.markdown("")
    fmt = format_metrics(result["best_metrics"], result["problem_type"])
    labels = {"accuracy":"Accuracy","f1_score":"F1 Score","r2_score":"R² Score",
              "rmse":"RMSE","cv_score":f"CV Score (k={cv_folds})"}
    mcols = st.columns(len(fmt))
    for i,(k,v) in enumerate(fmt.items()):
        mcols[i].metric(labels.get(k,k), v)
    if st.session_state.fe_summary:
        s = st.session_state.fe_summary
        parts = []
        if s.get("log_transformed"): parts.append(f"Log-transformed {len(s['log_transformed'])} cols")
        if s.get("polynomial_features"): parts.append(f"+{len(s['polynomial_features'])} poly features")
        if s.get("interaction_features"): parts.append(f"+{len(s['interaction_features'])} interaction features")
        if parts:
            st.markdown(f'<div class="info-box">⚙️ Feature engineering: {" · ".join(parts)} → {s.get("total_features_after","")} total features</div>', unsafe_allow_html=True)
    fig_td = plot_target_distribution(st.session_state.y, result["problem_type"])
    if fig_td:
        st.pyplot(fig_td, use_container_width=True); plt.close(fig_td)

# ── Section 4: Model Comparison Dashboard ─────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    st.markdown('<div class="section-title">⚖️ Model Comparison Dashboard</div>', unsafe_allow_html=True)
    rows = []
    for r in result["all_results"]:
        row = {"Model": r["name"]}
        row.update({k:v for k,v in r["metrics"].items() if k != "error"})
        row["Status"] = f"❌ {r['metrics']['error'][:60]}" if "error" in r["metrics"] else "✅"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_mc = plot_model_comparison(result["all_results"], result["problem_type"])
        if fig_mc:
            st.pyplot(fig_mc, use_container_width=True); plt.close(fig_mc)
    with col_b:
        fig_cv = plot_cv_comparison(result["all_results"])
        if fig_cv:
            st.pyplot(fig_cv, use_container_width=True); plt.close(fig_cv)

# ── Section 5: Advanced Charts ────────────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    label_encoder = st.session_state.label_encoder
    model = result["best_model"]
    st.markdown('<div class="section-title">📈 Advanced Charts</div>', unsafe_allow_html=True)
    if result["problem_type"] == "classification":
        tab_cm, tab_roc, tab_pr = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall"])
        with tab_cm:
            try:
                fig_cm = plot_confusion_matrix(model, X_test, y_test, label_encoder)
                st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)
            except Exception as e:
                st.error(f"CM error: {e}")
        with tab_roc:
            try:
                fig_roc = plot_roc_curve(model, X_test, y_test, label_encoder)
                if fig_roc:
                    st.pyplot(fig_roc, use_container_width=True); plt.close(fig_roc)
                else:
                    st.info("ROC requires predict_proba.")
            except Exception as e:
                st.error(f"ROC error: {e}")
        with tab_pr:
            try:
                fig_pr = plot_precision_recall(model, X_test, y_test)
                if fig_pr:
                    st.pyplot(fig_pr, use_container_width=True); plt.close(fig_pr)
                else:
                    st.info("PR curve: binary + predict_proba only.")
            except Exception as e:
                st.error(f"PR error: {e}")
    else:
        st.markdown("**Residual Analysis**")
        try:
            fig_res = plot_residuals(model, X_test, y_test)
            st.pyplot(fig_res, use_container_width=True); plt.close(fig_res)
        except Exception as e:
            st.error(f"Residual plot error: {e}")

# ── Section 6: Feature Importance ────────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    feature_names = st.session_state.feature_names
    st.markdown('<div class="section-title">📌 Feature Importance</div>', unsafe_allow_html=True)
    fi_df = get_feature_importance(result["best_model"], feature_names)
    if fi_df is not None and not fi_df.empty:
        col_c, col_d = st.columns([2,1])
        with col_c:
            fig_fi = plot_feature_importance(fi_df, top_n=15)
            if fig_fi:
                st.pyplot(fig_fi, use_container_width=True); plt.close(fig_fi)
        with col_d:
            st.dataframe(fi_df.head(15), use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

# ── Section 7: SHAP Explainability ───────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    X_selected = st.session_state.X_selected
    st.markdown('<div class="section-title">🔮 SHAP Explainability</div>', unsafe_allow_html=True)
    if st.button("⚡ Compute SHAP values"):
        with st.spinner("Computing SHAP values..."):
            try:
                X_shap = X_selected.sample(min(100, len(X_selected)), random_state=42)
                explainer, shap_values = get_shap_explainer(result["best_model"], X_shap, result["problem_type"])
                st.session_state["shap_values"] = shap_values
                st.session_state["X_shap"] = X_shap
                st.session_state.shap_ready = True
                st.success("✅ SHAP values computed!")
            except Exception as e:
                st.error(f"SHAP failed: {e}")
    if st.session_state.shap_ready:
        tab_sbar, tab_sbee = st.tabs(["Bar Plot", "Beeswarm"])
        sv = st.session_state["shap_values"]
        Xs = st.session_state["X_shap"]
        with tab_sbar:
            try:
                fig = plot_shap_bar(sv, Xs); st.pyplot(fig, use_container_width=True); plt.close(fig)
            except Exception as e:
                st.error(str(e))
        with tab_sbee:
            try:
                fig = plot_shap_summary(sv, Xs); st.pyplot(fig, use_container_width=True); plt.close(fig)
            except Exception as e:
                st.error(str(e))

# ── Section 8: Prediction UI ──────────────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    feature_names = st.session_state.feature_names
    X_selected = st.session_state.X_selected
    label_encoder = st.session_state.label_encoder
    st.markdown('<div class="section-title">🎯 Make a Prediction</div>', unsafe_allow_html=True)
    input_values = {}
    for i in range(0, len(feature_names), 3):
        chunk = feature_names[i:i+3]
        cols = st.columns(len(chunk))
        for col, feat in zip(cols, chunk):
            s = X_selected[feat]
            default_val = float(s.median())
            if s.nunique() <= 2 and set(s.unique()).issubset({0,1,0.0,1.0}):
                input_values[feat] = col.selectbox(feat, [0,1], index=int(default_val), key=f"p_{feat}")
            else:
                input_values[feat] = col.number_input(feat, value=default_val,
                    min_value=float(s.min()), max_value=float(s.max()), key=f"p_{feat}", format="%.4f")
    if st.button("🔮 Predict", type="primary"):
        try:
            input_df = pd.DataFrame([input_values])
            prediction = result["best_model"].predict(input_df)[0]
            pred_label = label_encoder.inverse_transform([int(prediction)])[0] if label_encoder else prediction
            if result["problem_type"] == "classification":
                st.success(f"🏷️ Predicted class: **{pred_label}**")
                if hasattr(result["best_model"], "predict_proba"):
                    proba = result["best_model"].predict_proba(input_df)[0]
                    classes = label_encoder.classes_ if label_encoder else range(len(proba))
                    st.table(pd.DataFrame({"Class": list(classes), "Probability": [f"{p*100:.1f}%" for p in proba]}))
            else:
                st.success(f"📈 Predicted value: **{float(pred_label):.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Section 9: Export & Deploy ────────────────────────────────────────────────
if st.session_state.train_result:
    result = st.session_state.train_result
    st.markdown('<div class="section-title">💾 Export & Deploy</div>', unsafe_allow_html=True)
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.markdown("**Trained Model**")
        st.download_button("⬇️ Download model.pkl", data=model_to_bytes(result["best_model"]),
                           file_name="model.pkl", mime="application/octet-stream", use_container_width=True)
        st.caption(f"Model: `{result['best_name']}`")
    with col_dl2:
        st.markdown("**EDA Report**")
        if st.session_state.profile_html:
            st.download_button("⬇️ Download EDA Report", data=st.session_state.profile_html,
                               file_name="eda_report.html", mime="text/html", use_container_width=True)
        else:
            st.info("Generate EDA report first.")
    with col_dl3:
        st.markdown("**REST API (FastAPI)**")
        st.markdown("```bash\nuvicorn src.api:app --port 8000\n```\nDocs at `localhost:8000/docs`")
    st.markdown("---")
    st.markdown('<div class="info-box">🐳 <b>Docker:</b> <code>docker build -t smart-automl . && docker run -p 8501:8501 smart-automl</code></div>', unsafe_allow_html=True)
    st.markdown("<center><small>Smart AutoML v2 · Streamlit · scikit-learn · SHAP · FastAPI</small></center>", unsafe_allow_html=True)
