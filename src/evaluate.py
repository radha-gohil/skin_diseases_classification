import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from .config import METRICS_DIR

def evaluate_model(model, generator, model_name):

    os.makedirs(METRICS_DIR, exist_ok=True)

    predictions = model.predict(generator)
    y_pred = np.argmax(predictions, axis=1)

    report = classification_report(
        generator.classes,
        y_pred,
        target_names=list(generator.class_indices.keys()),
        output_dict=True
    )

    cm = confusion_matrix(generator.classes, y_pred)

    # Save report
    with open(os.path.join(METRICS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    return report, cm