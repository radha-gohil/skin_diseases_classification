from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, EfficientNetB0
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from .config import IMG_SIZE

def build_model(model_name, num_classes):

    if model_name == "resnet50":
        base_model = ResNet50(weights="imagenet", include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))

    elif model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))

    elif model_name == "inceptionv3":
        base_model = InceptionV3(weights="imagenet", include_top=False,
                                 input_shape=(IMG_SIZE, IMG_SIZE, 3))

    elif model_name == "efficientnet":
        base_model = EfficientNetB0(weights="imagenet", include_top=False,
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Invalid model name")

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model