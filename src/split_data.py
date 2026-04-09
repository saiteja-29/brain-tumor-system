import os
import shutil
import random

SOURCE_DIR = "data/raw"
DEST_DIR = "data/processed"

SPLIT_RATIO = (0.7, 0.15, 0.15)
MAX_IMAGES_PER_CLASS = 500
classes = ["tumor", "no_tumor"]

for cls in classes:
    images = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(images)
    images = images[:MAX_IMAGES_PER_CLASS]

    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split in splits:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

        for img in splits[split]:
            src = os.path.join(SOURCE_DIR, cls, img)
            dst = os.path.join(DEST_DIR, split, cls, img)
            shutil.copyfile(src, dst)

print("Data split complete!")