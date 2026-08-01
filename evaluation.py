import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_predictions(y_true, y_pred, label_list=None):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    if label_list is None:
        matrix = confusion_matrix(y_true, y_pred)
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=label_list)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix,
    }


def time_fit_predict(model, x_train, y_train, x_eval):
    start_train = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    predictions = model.predict(x_eval)
    predict_seconds = time.perf_counter() - start_predict

    return predictions, train_seconds, predict_seconds
