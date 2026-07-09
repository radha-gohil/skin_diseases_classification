import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
TEST_DIR = os.path.join(BASE_DIR, "data/test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEST_DIR = os.path.join(BASE_DIR, "data/test")

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
# Training
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
