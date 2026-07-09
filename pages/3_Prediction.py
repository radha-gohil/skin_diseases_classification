import streamlit as st
import numpy as np
import json
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("🔍 Skin Disease Prediction")

# Load comparison data
if not os.path.exists("outputs/model_comparison.csv"):
    st.warning("Train models first.")
    st.stop()

df = pd.read_csv("outputs/model_comparison.csv")

# Allow user to select model
model_list = df["Model"].tolist()
selected_model = st.selectbox("Select Model", model_list)

model_path = f"models/{selected_model}.h5"

if not os.path.exists(model_path):
    st.error("Selected model file not found.")
    st.stop()

model = load_model(model_path)

# Load class mapping
with open("outputs/class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg","png"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        img = image.load_img(file, target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            preds = model.predict(img_array)[0]

        top_index = np.argmax(preds)
        predicted_class = index_to_class[top_index]
        confidence = preds[top_index] * 100

        col1, col2 = st.columns([1,1])

        with col1:
            st.image(file, use_container_width=True)

        with col2:
            st.subheader(f"Prediction: {predicted_class}")

            if confidence > 85:
                st.success(f"High Confidence: {confidence:.2f}%")
            elif confidence > 60:
                st.warning(f"Moderate Confidence: {confidence:.2f}%")
            else:
                st.error(f"Low Confidence: {confidence:.2f}%")

            # Probability distribution
            prob_df = pd.DataFrame({
                "Disease": list(index_to_class.values()),
                "Probability": preds
            }).sort_values(by="Probability", ascending=False)

            st.bar_chart(prob_df.set_index("Disease"))