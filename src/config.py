import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training
EPOCHS = 10
LEARNING_RATE = 1e-2

# Classes
CLASSES = ["no_tumor", "tumor"]
NUM_CLASSES = len(CLASSES)