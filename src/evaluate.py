import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from .config import DATA_DIR, MODEL_DIR
from .preprocessing import get_data_generators

from .preprocessing import get_test_generator
from sklearn.metrics import roc_auc_score


def evaluate():

    test_dir = os.path.join(DATA_DIR, "processed/test")

    test_gen = get_test_generator(test_dir)

    model = load_model(os.path.join(MODEL_DIR, "model.h5"))

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    y_prob = preds[:, 1]   # tumor probability
    roc = roc_auc_score(y_true, y_prob)

    print("ROC-AUC:", roc)

if __name__ == "__main__":
    evaluate()