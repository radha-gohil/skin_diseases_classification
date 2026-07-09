import streamlit as st
import os

st.title("📌 Confusion Matrix")

cm_files = [f for f in os.listdir("outputs") if "confusion" in f.lower()]

if not cm_files:
    st.warning("No confusion matrix images found.")
else:
    selected = st.selectbox("Select Model", cm_files)
    st.image(os.path.join("outputs", selected), use_container_width=True)