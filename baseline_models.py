import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config
import data_prep
from evaluation import evaluate_predictions, time_fit_predict


def build_default_model(model_name):
    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000)
    if model_name == "SVM":
        return SVC(random_state=config.RANDOM_SEED)
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=config.RANDOM_SEED)
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=config.RANDOM_SEED)

    raise ValueError("Unknown model name: " + model_name)


def run_binary_baselines(splits):
    train_df = splits["binary"]["train"]
    val_df = splits["binary"]["val"]

    x_train, y_train, _ = data_prep.get_xy(train_df, "BINARY_TARGET")
    x_val, y_val, _ = data_prep.get_xy(val_df, "BINARY_TARGET")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    results = {}
    for model_name in ["LogisticRegression", "SVM", "KNN"]:
        model = build_default_model(model_name)
        predictions, train_seconds, predict_seconds = time_fit_predict(
            model, x_train_scaled, y_train, x_val_scaled
        )
        metrics = evaluate_predictions(y_val, predictions, label_list=[0, 1])
        metrics["train_seconds"] = train_seconds
        metrics["predict_seconds"] = predict_seconds
        metrics["num_features_used"] = 6
        results[model_name] = metrics

    return results


def run_multiclass_baselines(splits):
    train_df = splits["multiclass"]["train"]
    val_df = splits["multiclass"]["val"]

    x_train, y_train, _ = data_prep.get_xy(train_df, "FAULT_TYPE")
    x_val, y_val, _ = data_prep.get_xy(val_df, "FAULT_TYPE")

    results = {}
    for model_name in ["DecisionTree", "RandomForest"]:
        model = build_default_model(model_name)
        predictions, train_seconds, predict_seconds = time_fit_predict(
            model, x_train, y_train, x_val
        )
        metrics = evaluate_predictions(y_val, predictions, label_list=config.FAULT_TYPE_LABELS)
        metrics["train_seconds"] = train_seconds
        metrics["predict_seconds"] = predict_seconds
        metrics["num_features_used"] = 6
        results[model_name] = metrics

    return results


def print_results_table(binary_results, multiclass_results):
    print("\nBinary Classification (validation set)")
    print(f"{'Model':<20}{'Accuracy (%)':<15}{'Macro F1':<12}{'Train (s)':<12}{'Predict (s)':<12}")
    for model_name in binary_results:
        r = binary_results[model_name]
        print(f"{model_name:<20}{r['accuracy']*100:<15.2f}{r['macro_f1']:<12.4f}"
              f"{r['train_seconds']:<12.4f}{r['predict_seconds']:<12.4f}")

    print("\nMulti-class Classification (validation set)")
    print(f"{'Model':<20}{'Accuracy (%)':<15}{'Macro F1':<12}{'Train (s)':<12}{'Predict (s)':<12}")
    for model_name in multiclass_results:
        r = multiclass_results[model_name]
        print(f"{model_name:<20}{r['accuracy']*100:<15.2f}{r['macro_f1']:<12.4f}"
              f"{r['train_seconds']:<12.4f}{r['predict_seconds']:<12.4f}")


def save_results(binary_results, multiclass_results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Confusion matrices are numpy arrays; convert to plain lists for JSON.
    combined = {"binary": {}, "multiclass": {}}
    for model_name in binary_results:
        r = dict(binary_results[model_name])
        r["confusion_matrix"] = r["confusion_matrix"].tolist()
        combined["binary"][model_name] = r
    for model_name in multiclass_results:
        r = dict(multiclass_results[model_name])
        r["confusion_matrix"] = r["confusion_matrix"].tolist()
        combined["multiclass"][model_name] = r

    out_path = os.path.join(config.RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print("\nSaved baseline results to", out_path)


def main():
    splits = data_prep.load_and_prepare()
    binary_results = run_binary_baselines(splits)
    multiclass_results = run_multiclass_baselines(splits)
    print_results_table(binary_results, multiclass_results)
    save_results(binary_results, multiclass_results)


if __name__ == "__main__":
    main()
