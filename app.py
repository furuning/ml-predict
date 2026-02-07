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

# --- Load artifacts from current directory (not subfolder) ---
@st.cache_resource
def load_artifacts():
    # Find model file: any *_Optimized.pkl in current dir
    model_files = glob.glob("*_Optimized.pkl")
    if not model_files:
        raise FileNotFoundError("No model file found. Expected a file like 'LightGBM_Optimized.pkl' in the current directory.")
    
    # Optional: prioritize known models
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
        best_model_path = model_files[0]  # fallback to first

    model = joblib.load(best_model_path)
    model_name = os.path.basename(best_model_path).replace("_Optimized.pkl", "")

    # Load scaler and feature names from current dir
    scaler_path = "scaler.pkl"
    feature_names_path = "feature_names.pkl"

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found in current directory.")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file '{feature_names_path}' not found in current directory.")

    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)

    return model, scaler, feature_names, model_name

# Initialize
try:
    model, scaler, feature_names, model_name = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model or dependencies:\n{e}")
    st.stop()

# --- Page config ---
st.set_page_config(page_title="Clinical Risk Prediction", layout="wide", page_icon="🏥")
st.title(f"🏥 {model_name}-based Clinical Risk Prediction System")
st.sidebar.info(f"✅ Model: **{model_name}**")

# --- Tabs ---
tab1, tab2 = st.tabs(["📝 Single-instance Prediction", "📂 Batch Prediction"])

# ==========================================
# Tab 1: Single Prediction
# ==========================================
with tab1:
    st.info("Enter clinical values for a single patient. All inputs are numeric and rounded to 2 decimal places.")
    
    with st.form("single_form"):
        inputs = {}
        n_cols = min(4, len(feature_names))
        cols = st.columns(n_cols)
        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                # ⭐ 关键修改：保留两位小数
                inputs[feat] = st.number_input(
                    label=f"{feat}",
                    value=0.00,          # 默认值两位小数
                    format="%.2f",       # 输入框显示格式
                    step=0.01            # 步长为0.01，便于调整
                )
        
        submitted = st.form_submit_button("🚀 Predict")
        
        if submitted:
            x_df = pd.DataFrame([inputs])
            x_scaled = scaler.transform(x_df)
            
            prob = model.predict_proba(x_scaled)[0, 1]
            
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
                    shap_vals_all = explainer.shap_values(x_scaled)
                    
                    # Handle multi-output SHAP (e.g., LightGBM/XGBoost)
                    if isinstance(shap_vals_all, list):
                        shap_vals = shap_vals_all[1]  # positive class
                        base_val = explainer.expected_value[1]
                    else:
                        shap_vals = shap_vals_all
                        base_val = explainer.expected_value
                    
                    if isinstance(base_val, np.ndarray):
                        base_val = base_val.item()  # scalar

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
# Tab 2: Batch Prediction
# ==========================================
with tab2:
    st.info("Upload a CSV or Excel file. Column names must exactly match the feature list below.")
    
    with st.expander("📥 Download Template (CSV)"):
        template = pd.DataFrame(columns=["Patient_ID"] + feature_names)
        csv_data = template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=csv_data,
            file_name="prediction_template.csv",
            mime="text/csv"
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
            extra = [c for c in df.columns if c not in feature_names and c != "Patient_ID"]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
            else:
                X = df[feature_names]
                X_scaled = scaler.transform(X)
                probs = model.predict_proba(X_scaled)[:, 1]
                
                result = df.copy()
                result['Risk_Probability'] = np.round(probs, 4)
                result['Risk_Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probs]
                
                # Highlight risk levels
                def highlight_risk(val):
                    color = '#ffcccc' if val == 'High Risk' else '#ccffcc'
                    return f'background-color: {color}'
                
                st.dataframe(result.style.applymap(highlight_risk, subset=['Risk_Level']))
                
                # Download button
                csv_out = result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "💾 Download Prediction Results (.csv)",
                    csv_out,
                    "prediction_results.csv",
                    "text/csv"
                )
                
                # Deep dive SHAP
                st.divider()
                st.subheader("🔍 Explain a Specific Sample")
                idx_options = result.index.tolist()
                selected_idx = st.selectbox(
                    "Select row index to analyze",
                    options=idx_options,
                    format_func=lambda i: f"Row {i} | Prob: {result.loc[i, 'Risk_Probability']:.2%}"
                )
                
                if st.button("Explain This Sample"):
                    x_single = X.iloc[[selected_idx]]
                    x_scaled_single = X_scaled[selected_idx].reshape(1, -1)
                    
                    explainer = shap.TreeExplainer(model)
                    sv_all = explainer.shap_values(x_scaled_single)
                    
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
                    
                    # Waterfall
                    fig2 = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(fig2, bbox_inches='tight')
                    plt.close(fig2)
                    
                    # Force Plot
                    st.markdown("**Force Plot (Interactive)**")
                    fp = shap.force_plot(bv, sv, x_single.iloc[0], feature_names=feature_names, matplotlib=False)
                    st_shap(fp, height=160)
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)