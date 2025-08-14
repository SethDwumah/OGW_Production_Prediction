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

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="OGW Prediction Dashboard", layout="wide")

# =========================
# Load Models
# =========================
models = {
    "Random Forest": joblib.load("rf_model.pkl"),
    "XGBoost": joblib.load("xgb_model.pkl"),
    "Extra Trees": joblib.load("extr_model.pkl"),
}

# =========================
# Load Preprocessor & Scaler
# =========================
preprocessor = joblib.load("preprocessor.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# =========================
# Helper Functions
# =========================
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
        avg_gap = date_diffs.mean()

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
        legend_title="Toggle Lines"
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

        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.warning("No numeric columns available for correlation.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

#def calculate_metrics(y_true, y_pred):
    #rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))
    #r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    #return rmse, r2

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

# Persist uploaded data
if uploaded_file:
    st.session_state.df_raw = pd.read_csv(uploaded_file)
else:
    st.session_state.pop("df_raw", None)
    
st.write("### 📈 Production Dashboard")
tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Data Visualization", "🧠 SHAP Analysis"])

if "df_raw" in st.session_state and predict_button:
    df = st.session_state.df_raw.copy()
    df['PRODUCTION DATE'] = pd.to_datetime(df['PRODUCTION DATE'], errors='coerce')
    df['year'] = df['PRODUCTION DATE'].dt.year
    df['month'] = df['PRODUCTION DATE'].dt.month
    df['day'] = df['PRODUCTION DATE'].dt.day

    model = models[model_choice]
    preds = preprocess_and_predict(model, df)
    df['Predicted Oil'] = preds[:, 0]
    df['Predicted Gas'] = preds[:, 1]
    df['Predicted Water'] = preds[:, 2]
    st.session_state.df_pred = df

if "df_pred" in st.session_state:
    df = st.session_state.df_pred

    # ===== TAB 1: Predictions =====
    with tab1:
       
        col1, col2, col3 = st.columns(3)

        col1.metric("Oil Production", f"{df['Predicted Oil'].mean():,.0f} stb/day")
        col2.metric("Gas Production", f"{(df['Predicted Gas'].mean())/1000:,.0f} mscf/day")
        col3.metric("Water Production", f"{df['Predicted Water'].mean():,.0f} stb/day")

        fig = plot_production_trends(df, smooth=(smooth_toggle == "Smoothed Trends"))
        st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 2: Data Visualization =====
    with tab2:
        visualize_data(df)

    # ===== TAB 3: SHAP Analysis =====
    with tab3:
        st.subheader(f"📊 SHAP Feature Importance for {model_choice}")
        sample_size = min(200, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        X_processed = preprocessor.transform(df_sample)
        feature_names = preprocessor.get_feature_names_out()

        def explain_single_model(single_model, target_name):
            model_type = type(single_model).__name__.lower()
            try:
                if any(t in model_type for t in ["forest", "extra", "xgb"]):
                    explainer = shap.TreeExplainer(single_model)
                    shap_values = explainer.shap_values(X_processed)
                else:
                    explainer = shap.Explainer(single_model.predict, X_processed)
                    shap_values = explainer(X_processed)
                fig_shap = plt.figure()
                shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
                st.pyplot(fig_shap)
                recs = generate_operational_recommendations(
                    df_sample,
                    shap_values if not isinstance(shap_values, list) else shap_values[0],
                    feature_names,
                    target_name
                )
                for r in recs:
                    st.write(f"- {r}")
            except Exception as e:
                st.error(f"SHAP failed for {target_name}: {e}")

        if hasattr(models[model_choice], "estimators_"):
            for i, target in enumerate(["Oil Production", "Gas Volume", "Water Production"]):
                st.write(f"### SHAP Summary - {target}")
                explain_single_model(models[model_choice].estimators_[i], target)
        else:
            explain_single_model(models[model_choice], "All Targets")

elif uploaded_file and not predict_button:
    st.info("Click the 🚀 Predict button to generate predictions.")
else:
    st.warning("Please upload a CSV file to proceed.")
