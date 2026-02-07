# app.py
import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import glob

# --- Helper: Embed SHAP JS plot ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150, scrolling=True)

# --- Load model and feature names (NO SCALER for tree models) ---
@st.cache_resource
def load_artifacts():
    # Find model file: *_Optimized.pkl in current directory
    model_files = glob.glob("*_Optimized.pkl")
    if not model_files:
        raise FileNotFoundError("No model file found. Expected a file like 'LightGBM_Optimized.pkl'.")

    # Prefer known tree models
    preferred_order = ["LightGBM", "XGBoost", "RandomForest"]
    best_model_path = None
    for name in preferred_order:
        for f in model_files:
            if name in f:
                best_model_path = f
                break
        if best_model_path:
            break
    if not best_model_path:
        best_model_path = model_files[0]

    model = joblib.load(best_model_path)
    model_name = os.path.basename(best_model_path).replace("_Optimized.pkl", "")

    # Load feature names (required for column order)
    feature_names_path = "feature_names.pkl"
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"'{feature_names_path}' not found. Please ensure it's in the current directory.")
    
    feature_names = joblib.load(feature_names_path)

    return model, feature_names, model_name

# Initialize
try:
    model, feature_names, model_name = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model or feature names:\n{e}")
    st.stop()

# --- Page config ---
st.set_page_config(page_title="Clinical Risk Prediction", layout="wide", page_icon="🌳")
st.title(f"🌳 {model_name}-based Clinical Risk Prediction System")
st.sidebar.info(f"✅ Model: **{model_name}** (Tree-based, no scaling needed)")

# --- Tabs ---
tab1, tab2 = st.tabs(["📝 Single-instance Prediction", "📂 Batch Prediction"])

# ==========================================
# Tab 1: Single Prediction (Raw Input)
# ==========================================
with tab1:
    st.info("Enter **raw clinical values** (no normalization needed for tree models). Inputs are rounded to 2 decimal places.")
    
    with st.form("single_form"):
        inputs = {}
        n_cols = min(4, len(feature_names))
        cols = st.columns(n_cols)
        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                inputs[feat] = st.number_input(
                    label=f"{feat}",
                    value=0.00,
                    format="%.2f",
                    step=0.01
                )
        
        submitted = st.form_submit_button("🚀 Predict")
        
        if submitted:
            # Use raw input — NO SCALING
            x_df = pd.DataFrame([inputs], columns=feature_names)
            
            # Predict
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_df)[0, 1]
            else:
                prob = model.predict(x_df)[0]
            
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Prediction Result")
                st.metric("Risk Probability", f"{prob:.2%}")
                if prob > 0.5:
                    st.error("🔴 High Risk")
                else:
                    st.success("🟢 Low Risk")
            
            with c2:
                st.subheader("SHAP Explanation")
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.TreeExplainer(model)
                    shap_vals_all = explainer.shap_values(x_df)
                    
                    # Handle multi-class output
                    if isinstance(shap_vals_all, list):
                        shap_vals = shap_vals_all[1]  # positive class
                        base_val = explainer.expected_value[1]
                    else:
                        shap_vals = shap_vals_all
                        base_val = explainer.expected_value
                    
                    if isinstance(base_val, np.ndarray):
                        base_val = base_val.item()

                    # Waterfall Plot
                    exp = shap.Explanation(
                        values=shap_vals[0],
                        base_values=base_val,
                        data=x_df.iloc[0],
                        feature_names=feature_names
                    )
                    fig = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Force Plot
                    st.markdown("**Force Plot (Interactive)**")
                    force_html = shap.force_plot(
                        base_val,
                        shap_vals[0],
                        x_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force_html, height=160)

# ==========================================
# Tab 2: Batch Prediction (Raw Input)
# ==========================================
with tab2:
    st.info("Upload CSV/Excel with **raw feature values**. Column names must match exactly.")
    
    with st.expander("📥 Download Template (CSV)"):
        template = pd.DataFrame(columns=["Patient_ID"] + feature_names)
        csv_data = template.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV Template",
            csv_data,
            "prediction_template.csv",
            "text/csv"
        )
        st.write("Required features:", ", ".join(feature_names))
    
    uploaded = st.file_uploader("Upload File", type=["csv", "xlsx"])
    
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            
            df.columns = df.columns.str.strip()
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                # Use raw features — NO SCALING
                X = df[feature_names]
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    probs = model.predict(X)
                
                result = df.copy()
                result['Risk_Probability'] = np.round(probs, 4)
                result['Risk_Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probs]
                
                # Highlight
                def highlight_risk(val):
                    return 'background-color: #ffcccc' if val == 'High Risk' else 'background-color: #ccffcc'
                st.dataframe(result.style.applymap(highlight_risk, subset=['Risk_Level']))
                
                # Download
                csv_out = result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "💾 Download Results (.csv)",
                    csv_out,
                    "prediction_results.csv",
                    "text/csv"
                )
                
                # SHAP deep dive
                st.divider()
                st.subheader("🔍 Explain a Specific Sample")
                idx_options = result.index.tolist()
                selected_idx = st.selectbox(
                    "Select row to analyze",
                    options=idx_options,
                    format_func=lambda i: f"Row {i} | Prob: {result.loc[i, 'Risk_Probability']:.2%}"
                )
                
                if st.button("Explain This Sample"):
                    x_single = X.iloc[[selected_idx]]
                    
                    explainer = shap.TreeExplainer(model)
                    sv_all = explainer.shap_values(x_single)
                    
                    if isinstance(sv_all, list):
                        sv = sv_all[1][0]
                        bv = explainer.expected_value[1]
                    else:
                        sv = sv_all[0]
                        bv = explainer.expected_value
                    
                    if isinstance(bv, np.ndarray):
                        bv = bv.item()
                    
                    exp = shap.Explanation(
                        values=sv,
                        base_values=bv,
                        data=x_single.iloc[0],
                        feature_names=feature_names
                    )
                    
                    fig2 = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(fig2, bbox_inches='tight')
                    plt.close(fig2)
                    
                    st.markdown("**Force Plot (Interactive)**")
                    fp = shap.force_plot(bv, sv, x_single.iloc[0], feature_names=feature_names, matplotlib=False)
                    st_shap(fp, height=160)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e) 
