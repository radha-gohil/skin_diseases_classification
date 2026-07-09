import pandas as pd
from src.train import train_model
from src.config import TRAIN_DIR

MODELS = ["resnet50", "vgg16", "inceptionv3", "efficientnet"]

results = []

for model_name in MODELS:
    print(f"\nTraining {model_name}...")
    report = train_model(model_name, TRAIN_DIR)

    results.append({
        "Model": model_name,
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"]
    })

df = pd.DataFrame(results)
df.to_csv("outputs/model_comparison.csv", index=False)

print(df)