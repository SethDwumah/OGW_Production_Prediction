import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import io

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="OGW Prediction Dashboard", layout="wide")
st.title("📈 OGW Prediction Dashboard")

# =========================
# Cached Loaders (faster restarts & reruns)
# =========================
@st.cache_resource(show_spinner=False)
def load_models():
    # Add back other models later if needed
    return {
        "XGBoost": joblib.load("xgb_model.pkl"),
        # "Random Forest": joblib.load("rf_model.pkl"),
        # "Extra Trees": joblib.load("extr_model.pkl"),
    }

@st.cache_resource(show_spinner=False)
def load_preproc_and_scaler():
    preprocessor = joblib.load("preprocessor.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    return preprocessor, y_scaler

# Load once (cached)
try:
    models = load_models()
    preprocessor, y_scaler = load_preproc_and_scaler()
except Exception as e:
    st.error(f"🔧 Failed to load models or transformers: {e}")
    st.stop()

# =========================
# Helper Functions
# =========================
def validate_columns(df_input: pd.DataFrame, preproc):
    """Validate that the uploaded DataFrame has all columns expected by the preprocessor."""
    expected = getattr(preproc, "feature_names_in_", None)
    if expected is None:
        return True, []
    missing = [c for c in expected if c not in df_input.columns]
    return len(missing) == 0, missing

def preprocess_and_predict(model, df_input):
    X_processed = preprocessor.transform(df_input)
    y_pred_scaled = model.predict(X_processed)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred

def prepare_plot_data(df_filtered, smooth=True):
    df_filtered = df_filtered.copy()
    df_filtered['PRODUCTION DATE'] = pd.to_datetime(df_filtered['PRODUCTION DATE'], errors='coerce')
    df_filtered = df_filtered.sort_values("PRODUCTION DATE")

    if smooth:
        date_diffs = df_filtered['PRODUCTION DATE'].diff().dropna().dt.days
        avg_gap = date_diffs.mean() if not date_diffs.empty else 9999

        if avg_gap <= 2:
            df_filtered['Period'] = df_filtered['PRODUCTION DATE'].dt.to_period('M')
            plot_df = df_filtered.groupby('Period').mean(numeric_only=True).reset_index()
            plot_df['Date'] = plot_df['Period'].dt.to_timestamp()
        elif avg_gap <= 40:
            df_filtered['Period'] = df_filtered['PRODUCTION DATE'].dt.to_period('W')
            plot_df = df_filtered.groupby('Period').mean(numeric_only=True).reset_index()
            plot_df['Date'] = plot_df['Period'].dt.start_time
        else:
            plot_df = df_filtered.rename(columns={'PRODUCTION DATE': 'Date'})
    else:
        plot_df = df_filtered.rename(columns={'PRODUCTION DATE': 'Date'})
    return plot_df

def plot_production_trends(df_filtered, smooth=True):
    plot_df = prepare_plot_data(df_filtered, smooth)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Oil Production (stb/day)'],
                             mode='lines', name="Actual Oil", line_shape='spline'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Predicted Oil'],
                             mode='lines', name="Predicted Oil", line_shape='spline'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Gas Volume (scf/day)'],
                             mode='lines', name="Actual Gas", line_shape='spline'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Predicted Gas'],
                             mode='lines', name="Predicted Gas", line_shape='spline'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Water Production (stb/day)'],
                             mode='lines', name="Actual Water", line_shape='spline'))
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Predicted Water'],
                             mode='lines', name="Predicted Water", line_shape='spline'))

    fig.update_layout(
        title="Actual vs. Predicted Production Trends",
        xaxis_title="Date",
        yaxis_title="Production Rate",
        template="plotly_dark",
        legend_title="Toggle Lines",
        yaxis_tickformat=",.0f"
    )
    return fig

def visualize_data(df):
    st.subheader("📊 Feature-wise Visualizations")
    if df is None or df.empty:
        st.warning("No data available. Please upload and predict first.")
        return

    plot_type = st.selectbox("Choose plot type", ["Histogram", "Line Plot", "Scatter", "Correlation Heatmap"], key="viz_type")

    if plot_type in ["Histogram", "Line Plot", "Scatter"]:
        col_x = st.selectbox("X-axis", df.columns, key="viz_x")
        col_y = None
        if plot_type in ["Line Plot", "Scatter"]:
            col_y = st.selectbox("Y-axis", df.columns, key="viz_y")

        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_x, nbins=30, title=f"Distribution of {col_x}")
        elif plot_type == "Line Plot":
            fig = px.line(df, x=col_x, y=col_y, title=f"{col_y} over {col_x}")
        elif plot_type == "Scatter":
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")

        fig.update_layout(yaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.warning("No numeric columns available for correlation.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

def generate_operational_recommendations(df_sample, shap_values, feature_names, target_name):
    if hasattr(shap_values, "values"):  # Explanation object
        shap_vals = shap_values.values
    else:
        shap_vals = np.array(shap_values)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Impact": np.abs(shap_vals).mean(axis=0)
    }).sort_values(by="SHAP_Impact", ascending=False)
    top_features = importance_df.head(3)
    recs = []
    for _, row in top_features.iterrows():
        feat = row["Feature"]
        if "Choke" in feat:
            recs.append(f"Consider adjusting {feat}; model strongly influences {target_name}.")
        elif "Pressure" in feat:
            recs.append(f"Monitor and optimize {feat} to stabilize {target_name}.")
        elif "Temperature" in feat:
            recs.append(f"Temperature variations in {feat} affect {target_name}.")
        else:
            recs.append(f"Review operational control over {feat} as it impacts {target_name}.")
    return recs

# =========================
# Sidebar
# =========================
st.sidebar.header("📂 Upload a CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
model_choice = st.sidebar.selectbox("🤖 Select Model", options=list(models.keys()))
smooth_toggle = st.sidebar.radio("📊 Plot Style", ["Smoothed Trends", "Raw Data"])
predict_button = st.sidebar.button("🚀 Predict")

# Guardrail: show upload details & size hint
if uploaded_file is not None:
    size_mb = getattr(uploaded_file, "size", None)
    if size_mb is not None:
        size_mb = size_mb / (1024 * 1024)
        st.sidebar.caption(f"File size: ~{size_mb:.2f} MB")
        if size_mb > 25:
            st.sidebar.warning("Large CSVs may be slow or memory-heavy on Streamlit Cloud.")

# Persist uploaded data
if uploaded_file:
    try:
        # Read to memory buffer first to avoid partial reads
        content = uploaded_file.read()
        df_candidate = pd.read_csv(io.BytesIO(content))
        st.session_state.df_raw = df_candidate
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.session_state.pop("df_raw", None)
else:
    st.session_state.pop("df_raw", None)

st.write("### 📈 Production Dashboard")
tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Data Visualization", "🧠 SHAP Analysis"])

# =========================
# Prediction Flow
# =========================
if "df_raw" in st.session_state and predict_button:
    df = st.session_state.df_raw.copy()

    if df.empty:
        st.warning("Uploaded CSV is empty.")
    else:
        # Ensure PRODUCTION DATE exists for plotting (not strictly required for prediction)
        if "PRODUCTION DATE" not in df.columns:
            st.info("`PRODUCTION DATE` column not found. Plots will use raw ordering.")
        else:
            df['PRODUCTION DATE'] = pd.to_datetime(df['PRODUCTION DATE'], errors='coerce')

        # Add simple date features if PRODUCTION DATE present
        if "PRODUCTION DATE" in df.columns:
            df['year'] = df['PRODUCTION DATE'].dt.year
            df['month'] = df['PRODUCTION DATE'].dt.month
            df['day'] = df['PRODUCTION DATE'].dt.day

        # Column validation against preprocessor expectations (when available)
        ok, missing = validate_columns(df, preprocessor)
        if not ok:
            st.error(
                "Some required columns are missing for the model preprocessor:\n\n"
                + ", ".join(missing)
            )
        else:
            # Run prediction
            try:
                model = models[model_choice]
                preds = preprocess_and_predict(model, df)
                if preds.ndim != 2 or preds.shape[1] < 3:
                    raise ValueError("Model did not return 3 outputs (Oil, Gas, Water).")

                df['Predicted Oil'] = preds[:, 0]
                df['Predicted Gas'] = preds[:, 1]
                df['Predicted Water'] = preds[:, 2]
                st.session_state.df_pred = df
                st.success("✅ Predictions generated.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =========================
# Tabs
# =========================
if "df_pred" in st.session_state:
    df = st.session_state.df_pred

    # ===== TAB 1: Predictions =====
    with tab1:
        col1, col2, col3 = st.columns(3)
        try:
            col1.metric("Oil Production", f"{df['Predicted Oil'].mean():,.0f} stb/day")
            col2.metric("Gas Production", f"{(df['Predicted Gas'].mean())/1000:,.0f} mscf/day")
            col3.metric("Water Production", f"{df['Predicted Water'].mean():,.0f} stb/day")
        except Exception:
            st.caption("Metrics unavailable due to missing predictions.")

        try:
            fig = plot_production_trends(df, smooth=(smooth_toggle == "Smoothed Trends"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plotting failed: {e}")

    # ===== TAB 2: Data Visualization =====
    with tab2:
        visualize_data(df)

    # ===== TAB 3: SHAP Analysis =====
    with tab3:
        st.subheader(f"📊 SHAP Feature Importance for {model_choice}")
        try:
            sample_size = max(1, min(200, len(df)))
            df_sample = df.sample(sample_size, random_state=42)

            # Use only model input columns for SHAP to avoid ColumnTransformer issues
            expected = getattr(preprocessor, "feature_names_in_", None)
            if expected is not None:
                X_base = df_sample[expected].copy()
            else:
                X_base = df_sample.copy()

            X_processed = preprocessor.transform(X_base)
            feature_names = preprocessor.get_feature_names_out()
            target_names = ["Oil Production", "Gas Volume", "Water Production"]

            def _plot_shap_summary(shap_vals, X_proc, feat_names, title):
                st.write(f"### {title}")
                fig_shap = plt.figure()
                shap.summary_plot(
                    shap_vals if not isinstance(shap_vals, list) else shap_vals[0],
                    X_proc,
                    feature_names=feat_names,
                    show=False
                )
                st.pyplot(fig_shap)

            def _op_hints(vals, feats, tgt):
                recs = generate_operational_recommendations(
                    df_sample,
                    vals if not hasattr(vals, "values") else vals.values,
                    feats,
                    tgt
                )
                st.write("**Operational hints (auto-generated)**")
                for r in recs:
                    st.write(f"- {r}")

            model = models[model_choice]
            model_type = type(model).__name__.lower()
            is_xgb = "xgb" in model_type or "xgboost" in str(type(model)).lower()

            # ------------ XGBoost-specific handling ------------
            if is_xgb:
                # Case 1: MultiOutputRegressor(XGBRegressor) — explain per target
                if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) >= 1:
                    for i, tgt in enumerate(target_names[:len(model.estimators_)]):
                        try:
                            base = model.estimators_[i]  # XGBRegressor
                            explainer = shap.TreeExplainer(base)
                            shap_vals = explainer.shap_values(X_processed)

                            _plot_shap_summary(shap_vals, X_processed, feature_names, f"SHAP Summary – XGBoost – {tgt}")
                            _op_hints(shap_vals, feature_names, tgt)
                        except Exception as e:
                            st.error(f"SHAP failed for XGBoost ({tgt}): {e}")

                # Case 2: plain XGBRegressor (single-output) — explain once
                else:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_processed)
                        _plot_shap_summary(shap_vals, X_processed, feature_names, "SHAP Summary – XGBoost")
                        _op_hints(shap_vals, feature_names, "All Targets")
                    except Exception as e:
                        st.error(f"SHAP failed for XGBoost: {e}")

            # ------------ Other tree models (RF/ET/GB/DT) ------------
            elif any(t in model_type for t in ["forest", "extra", "gradientboost", "decisiontree"]):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_processed)
                    if not isinstance(shap_vals, list):
                        shap_vals = [shap_vals]
                    for i, tgt in enumerate(target_names[:len(shap_vals)]):
                        _plot_shap_summary(shap_vals[i], X_processed, feature_names, f"SHAP Summary – {type(model).__name__} – {tgt}")
                        _op_hints(shap_vals[i], feature_names, tgt)
                except Exception:
                    # Fallback for wrappers exposing per-target estimators
                    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) >= 1:
                        for i, tgt in enumerate(target_names[:len(model.estimators_)]):
                            try:
                                base = model.estimators_[i]
                                explainer = shap.TreeExplainer(base)
                                shap_vals = explainer.shap_values(X_processed)
                                _plot_shap_summary(shap_vals, X_processed, feature_names, f"SHAP Summary – {type(base).__name__} – {tgt}")
                                _op_hints(shap_vals, feature_names, tgt)
                            except Exception as e:
                                st.error(f"SHAP failed for {type(base).__name__} ({tgt}): {e}")
                    else:
                        st.info("SHAP fallback not available for this model structure.")

            # ------------ Generic non-tree fallback ------------
            else:
                try:
                    explainer = shap.Explainer(model.predict, X_processed)
                    shap_vals = explainer(X_processed)
                    _plot_shap_summary(shap_vals, X_processed, feature_names, f"SHAP Summary – {type(model).__name__}")
                    _op_hints(shap_vals, feature_names, "All Targets")
                except Exception as e:
                    st.info("SHAP could not be computed for this model on the current environment.")
                    st.error(f"Details: {e}")

        except Exception as e:
            st.error(f"SHAP setup failed: {e}")

elif uploaded_file and not predict_button:
    st.info("Click the 🚀 Predict button to generate predictions.")
else:
    st.warning("Please upload a CSV file to proceed.")
