# 🩺 Skin Disease Classification Using Deep Learning

An end-to-end deep learning project for **multi-class skin disease classification** using **Transfer Learning** and an interactive **Streamlit dashboard**.

This project compares four ImageNet-pretrained convolutional neural network architectures:

- ResNet50
- VGG16
- InceptionV3
- EfficientNetB0

The complete workflow includes image preprocessing, transfer learning, model training, validation, independent test-set evaluation, model comparison, confusion-matrix analysis, and interactive image-based prediction.

> 🏆 **Best Experimental Result:** InceptionV3 achieved **93.13% test accuracy** and an **F1-score of 93.19%** on the project test set.

---

## 📌 Project Overview

Skin diseases can have visually similar symptoms, making image-based classification a challenging computer vision problem.

This project investigates how different pretrained CNN architectures perform on an **8-class skin disease classification task**.

Instead of relying on a single deep learning architecture, the system trains and evaluates multiple transfer-learning models under a common pipeline and compares them using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Accuracy Curves
- Validation Accuracy Curves
- Training Loss Curves
- Validation Loss Curves

The project also includes an interactive **Streamlit web application** for:

- Model performance comparison
- Training insight visualization
- Skin disease prediction
- Workflow visualization
- Confusion matrix inspection
- Confidence-based prediction analysis

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build an automated multi-class skin disease image classification pipeline
- Compare multiple pretrained CNN architectures
- Apply transfer learning using ImageNet weights
- Evaluate models on a separate test dataset
- Analyze model performance using multiple evaluation metrics
- Generate confusion matrices for class-level analysis
- Visualize training and validation behavior
- Deploy trained models through an interactive Streamlit interface
- Support image upload and confidence-based prediction

---

## 🦠 Skin Disease Classes

The system classifies skin images into the following eight categories:

| No. | Disease Class | Category |
|---|---|---|
| 1 | Cellulitis | Bacterial |
| 2 | Impetigo | Bacterial |
| 3 | Athlete's Foot | Fungal |
| 4 | Nail Fungus | Fungal |
| 5 | Ringworm | Fungal |
| 6 | Cutaneous Larva Migrans | Parasitic |
| 7 | Chickenpox | Viral |
| 8 | Shingles | Viral |

### Internal Dataset Labels

```text
BA- cellulitis
BA-impetigo
FU-athlete-foot
FU-nail-fungus
FU-ringworm
PA-cutaneous-larva-migrans
VI-chickenpox
VI-shingles
```

---

## 🧠 Deep Learning Models

The project compares four pretrained CNN architectures.

### 1. ResNet50

ResNet50 is a deep convolutional neural network that uses **residual connections** to improve gradient flow and support the training of deeper networks.

### 2. VGG16

VGG16 is a convolutional neural network architecture based on repeated **3×3 convolutional layers** with a simple and structured deep architecture.

### 3. InceptionV3

InceptionV3 uses multi-branch **Inception modules** to capture image features at different receptive-field scales.

### 4. EfficientNetB0

EfficientNetB0 is designed around **compound scaling**, which balances network depth, width, and input resolution.

---

## 🏗️ Model Architecture

Each model uses an ImageNet-pretrained CNN as a feature extractor.

```text
Input Skin Image
        │
        ▼
Resize to 224 × 224 × 3
        │
        ▼
Pretrained CNN Backbone
        │
        ├── ResNet50
        ├── VGG16
        ├── InceptionV3
        └── EfficientNetB0
        │
        ▼
Frozen Base Model
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense Layer
512 Units + ReLU
        │
        ▼
Output Layer
8 Units + Softmax
        │
        ▼
Predicted Disease Class
```

The pretrained convolutional base is frozen during training, while the custom classification head learns task-specific patterns for the eight skin disease classes.

---

## 🔄 End-to-End Project Pipeline

```text
Skin Disease Image Dataset
          │
          ▼
Directory-Based Data Loading
          │
          ▼
Resize Images to 224 × 224
          │
          ▼
Pixel Rescaling (1/255)
          │
          ▼
Training / Validation Split
       80% / 20%
          │
          ▼
Transfer Learning
          │
          ├── ResNet50
          ├── VGG16
          ├── InceptionV3
          └── EfficientNetB0
          │
          ▼
Frozen ImageNet Backbone
          │
          ▼
Custom Classification Head
          │
          ▼
Model Training
          │
          ▼
Independent Test Evaluation
          │
          ├── Accuracy
          ├── Precision
          ├── Recall
          ├── F1-Score
          └── Confusion Matrix
          │
          ▼
Cross-Model Comparison
          │
          ▼
Streamlit Dashboard
          │
          ▼
Interactive Image Prediction
```

---

## ⚙️ Data Preprocessing

Images are loaded using TensorFlow/Keras directory-based image generators.

### Preprocessing Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Image Channels | RGB |
| Batch Size | 32 |
| Training Portion | 80% |
| Validation Portion | 20% |
| Pixel Scaling | 1/255 |
| Class Mode | Categorical |
| Random Seed | 42 |

### Pixel Normalization

Image pixel values are rescaled using:

```text
Normalized Pixel = Pixel Value / 255
```

This transforms the original pixel range into normalized values.

### Current Implementation Note

The current preprocessing pipeline performs:

- Image resizing
- Pixel rescaling
- Training/validation splitting

The current implementation does **not apply explicit image augmentation** such as:

- Random rotation
- Horizontal flipping
- Random zoom
- Width shifting
- Height shifting

This README intentionally reflects the actual implementation.

---

## 🔬 Transfer Learning Strategy

Each CNN backbone is initialized using ImageNet-pretrained weights.

```python
weights="imagenet"
include_top=False
```

The original ImageNet classification head is removed.

The pretrained CNN backbone is frozen:

```python
base_model.trainable = False
```

A custom classification head is then added:

```text
Pretrained CNN Feature Maps
            │
            ▼
GlobalAveragePooling2D
            │
            ▼
Dense Layer
512 Units + ReLU
            │
            ▼
Output Layer
8 Units + Softmax
```

This approach reuses general visual representations learned from ImageNet while training a task-specific classifier for the eight skin disease categories.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 32 |
| Input Image Size | 224 × 224 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Training Metric | Accuracy |
| Output Activation | Softmax |
| Random Seed | 42 |

The models are compiled using:

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 🧪 Model Training Process

For every model, the training pipeline performs the following steps:

1. Load the training images
2. Resize images to 224 × 224
3. Normalize image pixels
4. Create an 80/20 training-validation split
5. Load ImageNet-pretrained CNN weights
6. Remove the original classification head
7. Freeze the pretrained base model
8. Add a custom classification head
9. Train the model for 20 epochs
10. Save the trained model
11. Generate training accuracy plots
12. Generate validation accuracy plots
13. Generate training loss plots
14. Generate validation loss plots
15. Evaluate on the separate test dataset
16. Generate a classification report
17. Generate a confusion matrix
18. Store model comparison metrics

---

## 📊 Model Performance Comparison

The models were evaluated on the project test set.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| ResNet50 | 36.48% | 34.55% | 36.48% | 29.13% |
| VGG16 | 83.69% | 86.60% | 83.69% | 84.17% |
| **InceptionV3** | **93.13%** | **93.29%** | **93.13%** | **93.19%** |
| EfficientNetB0 | 14.59% | 2.13% | 14.59% | 3.72% |

---

## 🏆 Best Performing Model

### InceptionV3

InceptionV3 achieved the strongest overall performance among the four evaluated architectures.

```text
Accuracy  : 93.13%
Precision : 93.29%
Recall    : 93.13%
F1-Score  : 93.19%
```

The experimental results indicate that InceptionV3 provided the best balance across all reported evaluation metrics on the project test set.

---

## 📈 Evaluation Metrics

The project evaluates model performance using multiple metrics.

### Accuracy

Accuracy measures the proportion of correctly classified test samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Precision measures how many samples predicted as belonging to a class are actually relevant to that class.

```text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many actual samples belonging to a class are correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

F1-score provides the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The project uses weighted precision, recall, and F1-score for cross-model comparison.

---

## 🔍 Confusion Matrix Analysis

A confusion matrix is generated for every trained model.

This enables analysis of:

- Correct classifications
- Incorrect classifications
- Frequently confused disease classes
- Class-specific model behavior
- Misclassification patterns

Generated confusion matrix files include:

```text
resnet50_confusion_matrix.png
vgg16_confusion_matrix.png
inceptionv3_confusion_matrix.png
efficientnet_confusion_matrix.png
```

---

## 📉 Training and Validation Curves

For each trained model, the project generates:

- Training accuracy curve
- Validation accuracy curve
- Training loss curve
- Validation loss curve

Example output files:

```text
resnet50_accuracy.png
resnet50_loss.png

vgg16_accuracy.png
vgg16_loss.png

inceptionv3_accuracy.png
inceptionv3_loss.png

efficientnet_accuracy.png
efficientnet_loss.png
```

These plots help analyze:

- Model convergence
- Generalization behavior
- Potential overfitting
- Potential underfitting
- Performance differences across architectures

---

## 🖥️ Streamlit Dashboard

The project includes an interactive multi-page Streamlit web application.

### 📊 1. Model Comparison

The Model Comparison page:

- Loads stored model comparison results
- Displays model performance metrics
- Creates an accuracy comparison chart
- Automatically identifies the best-performing model

---

### 📈 2. Model Training Insights

The Model Training Insights page is designed to:

- Display saved training plots
- Allow model-specific plot selection
- Visualize training behavior
- Support performance analysis

---

### 🔍 3. Skin Disease Prediction

The prediction page allows users to:

- Select a trained model
- Upload one or multiple images
- Generate disease predictions
- Display predicted disease classes
- Display prediction confidence
- Visualize class probability distributions

Supported image formats:

```text
JPG
PNG
```

---

### 🛠️ 4. Project Workflow

The workflow page explains:

- Data preparation
- Image preprocessing
- Transfer learning
- Model training
- Evaluation
- Deployment

---

### 📌 5. Confusion Matrix

The confusion matrix page is designed to:

- Display saved confusion matrices
- Allow model-specific selection
- Support visual error analysis

---

## 🔮 Prediction Workflow

The interactive prediction pipeline follows:

```text
Upload Image
     │
     ▼
Resize to 224 × 224
     │
     ▼
Convert Image to Array
     │
     ▼
Normalize Pixels by 1/255
     │
     ▼
Add Batch Dimension
     │
     ▼
Load Selected .h5 Model
     │
     ▼
Generate Softmax Probabilities
     │
     ▼
Select Highest Probability
     │
     ▼
Map Class Index to Disease Name
     │
     ▼
Display Predicted Disease
     │
     ▼
Display Confidence Score
     │
     ▼
Display Probability Distribution
```

---

## 🎯 Confidence Interpretation

The Streamlit prediction interface categorizes prediction confidence as:

| Confidence Score | Interpretation |
|---|---|
| Above 85% | High Confidence |
| 60% to 85% | Moderate Confidence |
| Below 60% | Low Confidence |

---

## 📁 Project Structure

```text
skin_diseases_classification/
│
├── data/
│   ├── train/
│   │   ├── BA- cellulitis/
│   │   ├── BA-impetigo/
│   │   ├── FU-athlete-foot/
│   │   ├── FU-nail-fungus/
│   │   ├── FU-ringworm/
│   │   ├── PA-cutaneous-larva-migrans/
│   │   ├── VI-chickenpox/
│   │   └── VI-shingles/
│   │
│   └── test/
│       ├── BA- cellulitis/
│       ├── BA-impetigo/
│       ├── FU-athlete-foot/
│       ├── FU-nail-fungus/
│       ├── FU-ringworm/
│       ├── PA-cutaneous-larva-migrans/
│       ├── VI-chickenpox/
│       └── VI-shingles/
│
├── models/
│   ├── resnet50.h5
│   ├── vgg16.h5
│   ├── inceptionv3.h5
│   └── efficientnet.h5
│
├── outputs/
│   │
│   ├── metrics/
│   │   ├── resnet50_metrics.json
│   │   ├── vgg16_metrics.json
│   │   ├── inceptionv3_metrics.json
│   │   └── efficientnet_metrics.json
│   │
│   ├── plots/
│   │   ├── resnet50_accuracy.png
│   │   ├── resnet50_loss.png
│   │   ├── resnet50_confusion_matrix.png
│   │   ├── vgg16_accuracy.png
│   │   ├── vgg16_loss.png
│   │   ├── vgg16_confusion_matrix.png
│   │   ├── inceptionv3_accuracy.png
│   │   ├── inceptionv3_loss.png
│   │   ├── inceptionv3_confusion_matrix.png
│   │   ├── efficientnet_accuracy.png
│   │   ├── efficientnet_loss.png
│   │   └── efficientnet_confusion_matrix.png
│   │
│   ├── class_indices.json
│   └── model_comparison.csv
│
├── pages/
│   ├── 1_Model_Comparison.py
│   ├── 2_Model_Insights.py
│   ├── 3_Prediction.py
│   ├── 4_Workflow.py
│   └── 5_Confusion_Matrix.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_builder.py
│   ├── plots.py
│   ├── train.py
│   └── utils.py
│
├── main.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The dataset and trained `.h5` model files may be excluded from Git version control because of storage constraints.

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/radha-gohil/skin_diseases_classification.git
```

### 2. Navigate to the Project Directory

```bash
cd skin_diseases_classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Organization

Organize the dataset in the following directory structure:

```text
data/
├── train/
│   ├── BA- cellulitis/
│   ├── BA-impetigo/
│   ├── FU-athlete-foot/
│   ├── FU-nail-fungus/
│   ├── FU-ringworm/
│   ├── PA-cutaneous-larva-migrans/
│   ├── VI-chickenpox/
│   └── VI-shingles/
│
└── test/
    ├── BA- cellulitis/
    ├── BA-impetigo/
    ├── FU-athlete-foot/
    ├── FU-nail-fungus/
    ├── FU-ringworm/
    ├── PA-cutaneous-larva-migrans/
    ├── VI-chickenpox/
    └── VI-shingles/
```

---

## 🏋️ Train the Models

Run the main training script:

```bash
python main.py
```

The training script sequentially trains:

```text
resnet50
vgg16
inceptionv3
efficientnet
```

After training, the pipeline:

1. Saves trained model files
2. Saves class-index mappings
3. Generates accuracy curves
4. Generates loss curves
5. Evaluates models on the test dataset
6. Generates classification reports
7. Creates confusion matrices
8. Stores model comparison metrics

---

## 🌐 Run the Streamlit Application

Run:

```bash
streamlit run streamlit_app.py
```

The application will launch in the browser and provide access to the multi-page dashboard.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Machine Learning and Evaluation

- Scikit-learn

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Web Application

- Streamlit

### Version Control

- Git
- GitHub

---

## ⚠️ Current Limitations

The current implementation has several limitations:

### 1. No Explicit Data Augmentation

The current training generator performs:

- Pixel rescaling
- Training-validation splitting

It does not currently apply:

- Rotation
- Flipping
- Zooming
- Translation

---

### 2. Frozen Feature Extractors

All pretrained CNN backbone layers remain frozen during training.

Architecture-specific fine-tuning may improve performance.

---

### 3. Shared Input Scaling Strategy

All models currently use the same `1/255` pixel scaling strategy.

Different pretrained architectures may benefit from their corresponding architecture-specific preprocessing functions.

---

### 4. Large Trained Model Files

The `.h5` model files may not be included directly in the GitHub repository because of storage constraints.

---

### 5. Performance Variability Across Architectures

The experimental results differ substantially across models.

In particular:

- InceptionV3 achieved strong performance
- VGG16 also performed well
- ResNet50 showed limited performance
- EfficientNetB0 requires further investigation and tuning

---

### 6. No Clinical Validation

The system is an experimental deep learning image-classification project and has not undergone clinical validation.

---

## 🔧 Future Improvements

Future improvements may include:

- Add image augmentation
- Apply architecture-specific preprocessing
- Fine-tune upper CNN backbone layers
- Add EarlyStopping
- Add ReduceLROnPlateau
- Add ModelCheckpoint
- Investigate class imbalance
- Add class-weighted training
- Evaluate per-class sensitivity
- Evaluate per-class specificity
- Add ROC-AUC analysis
- Perform cross-validation
- Add Grad-CAM explainability
- Improve probability calibration
- Analyze domain shift
- Optimize model size for deployment
- Compare ensemble learning approaches
- Add cloud deployment
- Improve Streamlit user experience

---

## 🔍 Potential Research Extensions

This project can be extended with:

### Explainable AI

Integrate Grad-CAM to visualize image regions influencing model predictions.

### Fine-Tuning

Unfreeze selected upper layers of pretrained backbones and train them with a smaller learning rate.

### Ensemble Learning

Combine predictions from multiple CNN architectures.

### Model Calibration

Improve the reliability of predicted confidence scores.

### External Validation

Evaluate the trained model on an independent dataset from a different source.

---

## ⚕️ Medical Disclaimer

> This project is intended strictly for **educational and research purposes only**.

The system:

- Is not a clinically validated diagnostic tool
- Should not replace professional medical evaluation
- Should not be used as the sole basis for diagnosis
- Should not be used as the sole basis for treatment decisions

Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

## 👩‍💻 Author

**Radha Gohil**

Areas explored in this project:

- Deep Learning
- Computer Vision
- Transfer Learning
- Medical Image Classification
- Model Evaluation
- Streamlit Deployment
