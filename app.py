import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit_echarts as ste
import matplotlib as plt




st.set_page_config(
    page_title="Prediction of Oil, Gas, and Water Production rate using ML",
    page_icon=":graph:",
    layout="wide",
    initial_sidebar_state="expanded"
    
    )
alt.themes.enable("dark")
df_data = pd.read_csv("dseats_2024_training_dataset.csv")

df_data['PRODUCTION DATE'] = pd.to_datetime(df_data["PRODUCTION DATE"],format='mixed')
df_data['year'] = df_data['PRODUCTION DATE'].dt.year
df =df_data.copy()
with st.sidebar:
    st.title("Real-Time Data Analytics with ML")
    year_list = list(df["year"].unique()[::-1])
    selected_year = st.selectbox("Select a year",year_list)
    df = df[df["year"]==selected_year]
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)   
    



def make_barplot(input_df, input_x,input_y):
    barplot =px.bar(input_df,input_x,input_y,color='Oil Production (stb/day)')
    
    return barplot

col = st.columns((1.5, 4.5, 2), gap='medium')

with col[1]:
    barplot = make_barplot(df_data, 'year', 'Oil Production (stb/day)')
    
    st.plotly_chart(barplot, use_container_width=True)
    