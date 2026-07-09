import os
import json   # ✅ ADDED
import pandas as pd
from .config import MODEL_DIR, EPOCHS, TEST_DIR
from .data_loader import get_data_generators
from .model_builder import build_model
from .evaluate import evaluate_model
from .plots import save_training_plots, save_confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model(model_name, train_dir):

    train_gen, val_gen = get_data_generators(train_dir)

    # ✅ SAVE CLASS INDICES (ADDED BLOCK)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    model = build_model(model_name, train_gen.num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"{model_name}.h5"))

    # Save plots
    save_training_plots(history, model_name)

    # Test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    report, cm = evaluate_model(model, test_gen, model_name)
    save_confusion_matrix(cm, list(test_gen.class_indices.keys()), model_name)

    return report