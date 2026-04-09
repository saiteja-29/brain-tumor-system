import os

print("Tumor images:", len(os.listdir("data/raw/tumor")))
print("No tumor images:", len(os.listdir("data/raw/no_tumor")))