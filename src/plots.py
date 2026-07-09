import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from .config import PLOTS_DIR

def save_training_plots(history, model_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_accuracy.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_loss.png"))
    plt.close()


def save_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()