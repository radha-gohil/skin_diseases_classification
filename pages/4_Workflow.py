import streamlit as st

st.title("🛠 Project Workflow")

st.markdown("""
### 1️⃣ Data Preparation
- Balanced dataset
- Multi-class classification (8 diseases)

### 2️⃣ Preprocessing
- Image resizing (224x224)
- Normalization (1/255)
- Train/Validation/Test split

### 3️⃣ Transfer Learning
Pretrained models used:
- ResNet50
- VGG16
- InceptionV3
- EfficientNetB0

Base layers frozen  
Custom classification head added

### 4️⃣ Training Strategy
- Cross-entropy loss
- Adam optimizer
- Validation monitoring

### 5️⃣ Evaluation
- Accuracy comparison
- Confusion matrix
- Classification report

### 6️⃣ Deployment
- Interactive Streamlit dashboard
- Real-time prediction
- Multi-model comparison
""")