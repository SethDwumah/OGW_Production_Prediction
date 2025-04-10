import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Oil, Gas, and Water Production Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #1e1e1e;
        }
        .stMetric {
            font-size: 18px;
            font-weight: bold;
            background: #2b2b2b;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .container {
            display: grid;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 10px;
            margin-top:10px;
        }
    </style>
""", unsafe_allow_html=True)

def generate_data():
    np.random.seed(42)
    years = np.arange(2010, 2031)
    return pd.DataFrame({
        "Year": years,
        "Oil Production (stb/day)": np.random.randint(5000, 20000, size=len(years)),
        "Gas Production (mscf/day)": np.random.randint(10000, 50000, size=len(years)),
        "Water Production (bbl/day)": np.random.randint(2000, 15000, size=len(years))
    })

def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }[model_type]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    return model, rmse, r2

def plot_production_trends(df):
    fig = px.line(df, x="Year", y=["Oil Production (stb/day)", "Predicted Oil",
                                     "Gas Production (mscf/day)", "Predicted Gas",
                                     "Water Production (bbl/day)", "Predicted Water"],
                  labels={"value": "Production Rate", "Year": "Year"},
                  title="📉 Actual vs. Predicted Production Trends",
                  template="plotly_dark")
    fig.update_layout(legend=dict(title="Toggle Lines"))
    return fig

def plot_performance_metrics(rmse, r2):
    return f"""
    <div class='container'>
        <div class='stMetric'>💡 Oil RMSE: {rmse[0]:,.2f}</div>
        <div class='stMetric'>🔥 Gas RMSE: {rmse[1]:,.2f}</div>
        <div class='stMetric'>💧 Water RMSE: {rmse[2]:,.2f}</div>
        <div class='stMetric'>📈 Oil R²: {r2[0]:,.2f}</div>
        <div class='stMetric'>📈 Gas R²: {r2[1]:,.2f}</div>
        <div class='stMetric'>📈 Water R²: {r2[2]:,.2f}</div>
    </div>
    """

def plot_shap_feature_importance(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    return fig

def main():
    #df = generate_data()
    
    with st.sidebar:
        
        # File uploader for custom dataset
        uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV File", type=["csv"])
    
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if not {"Year", "Oil Production (stb/day)", "Gas Production (mscf/day)", "Water Production (bbl/day)"}.issubset(df.columns):
                st.sidebar.error("Invalid CSV format. Ensure it contains 'Year', 'Oil Production', 'Gas Production', and 'Water Production' columns.")
                df = generate_data()
        else:
            df = generate_data()
        st.title("🎛 Dashboard Filters")
        selected_year = st.selectbox("📅 Select a Year", sorted(df["Year"], reverse=True))
        selected_model = st.selectbox("🧠 Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])
    
    st.markdown(f"### ⛽ Production in {selected_year}")
    col1, col2, col3 = st.columns(3)
    selected_data = df[df["Year"] == selected_year]
    col1.metric("🔴 Oil Production", f"{selected_data['Oil Production (stb/day)'].values[0]:,} stb/day")
    col2.metric("🟢 Gas Production", f"{selected_data['Gas Production (mscf/day)'].values[0]:,} mscf/day")
    col3.metric("🔵 Water Production", f"{selected_data['Water Production (bbl/day)'].values[0]:,} bbl/day")
    
    X = df[["Year"]]
    y = df[["Oil Production (stb/day)", "Gas Production (mscf/day)", "Water Production (bbl/day)"]]
    model, rmse, r2 = train_model(X, y, selected_model)
    df["Predicted Oil"], df["Predicted Gas"], df["Predicted Water"] = model.predict(X).T
    
    #st.markdown("## Production Trends and Performance Metrics")
    col = st.columns((1, 4.5), gap='medium')

    with col[1]:
        st.plotly_chart(plot_production_trends(df), use_container_width=True)
        st.pyplot(plot_shap_feature_importance(model, X), use_container_width=True)
    with col[0]:
        st.markdown(plot_performance_metrics(rmse, r2), unsafe_allow_html=True)
    
    #st.markdown("## 🎯 Feature Importance (SHAP Analysis)")
    
    
    #st.markdown("## 📌 Observations")
    #st.text_area("📝 Add Your Comments:")

if __name__ == "__main__":
    main()
