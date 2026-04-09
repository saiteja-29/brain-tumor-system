import os
from .config import DATA_DIR, MODEL_DIR, EPOCHS
from .preprocessing import get_data_generators
from .model import build_model

def train():

    train_dir = os.path.join(DATA_DIR, "processed/train")
    val_dir = os.path.join(DATA_DIR, "processed/val")

    train_gen, val_gen = get_data_generators(train_dir, val_dir)

    model = build_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "model.h5"))

    print("Model saved!")

if __name__ == "__main__":
    train()