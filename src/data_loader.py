import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import IMG_SIZE, BATCH_SIZE, SEED

def get_data_generators(train_dir):

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical",
        seed=SEED
    )

    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="categorical",
        seed=SEED
    )

    return train_generator, val_generator