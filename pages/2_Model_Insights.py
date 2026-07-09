import streamlit as st
import os

st.title("📈 Model Training Insights")

plot_dir = "outputs"

if not os.path.exists(plot_dir):
    st.warning("Outputs folder not found.")
else:
    plots = [f for f in os.listdir(plot_dir) if f.endswith(".png") and "accuracy" in f.lower()]

    if not plots:
        st.warning("No training plots found.")
    else:
        selected = st.selectbox("Select Model Plot", plots)
        st.image(os.path.join(plot_dir, selected), use_container_width=True)