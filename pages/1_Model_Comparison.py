import streamlit as st
import pandas as pd
import os

st.title("📊 Model Comparison")

if not os.path.exists("outputs/model_comparison.csv"):
    st.warning("No model comparison data found. Train models first.")
else:
    df = pd.read_csv("outputs/model_comparison.csv")
    st.dataframe(df)

    st.subheader("Accuracy Comparison")
    st.bar_chart(df.set_index("Model")["Accuracy"])

    best = df.sort_values(by="Accuracy", ascending=False).iloc[0]
    st.success(f"🏆 Best Model: {best['Model']} ({best['Accuracy']*100:.2f}%)")