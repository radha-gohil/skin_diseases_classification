import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Skin Disease AI",
    page_icon="🩺",
    layout="wide"
)

st.sidebar.title("🩺 Skin Disease AI")
st.sidebar.markdown("""
Multi-Class Skin Disease  
Classification System  
Using Transfer Learning
""")

st.title("🧠 Skin Disease Classification Dashboard")

st.markdown("""
This application compares multiple deep learning models  
and deploys the best-performing model for real-time prediction.
""")

# Quick summary metrics (if training done)
if os.path.exists("outputs/model_comparison.csv"):
    df = pd.read_csv("outputs/model_comparison.csv")
    best = df.sort_values(by="Accuracy", ascending=False).iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best["Model"])
    col2.metric("Best Accuracy", f"{best['Accuracy']*100:.2f}%")
    col3.metric("Models Compared", len(df))

else:
    st.info("Train models to see performance summary.")