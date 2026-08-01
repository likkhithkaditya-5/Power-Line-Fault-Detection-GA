import json
import os
import pandas as pd

import config
import data_prep
from evaluation import evaluate_predictions, time_fit_predict
from baseline_models import build_default_model
from run_ga_experiments import CLASSIFIER_SPECS

BINARY_MODELS = ["LogisticRegression", "SVM", "KNN"]


def load_ga_results():
    path = os.path.join(config.RESULTS_DIR, "ga_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find " + path + ". Run run_ga_experiments.py first."
        )
    with open(path) as f:
        return json.load(f)


def combine_train_and_val(splits, task):
    train_df = splits[task]["train"]
    val_df = splits[task]["val"]
    return pd.concat([train_df, val_df], axis=0)


def get_task_settings(task):
    if task == "binary":
        return "BINARY_TARGET", [0, 1]
    return "FAULT_TYPE", config.FAULT_TYPE_LABELS


def evaluate_baseline_on_test(name, task, splits):
    target_column, label_list = get_task_settings(task)

    fit_df = combine_train_and_val(splits, task)
    test_df = splits[task]["test"]

    x_fit, y_fit, _ = data_prep.get_xy(fit_df, target_column)
    x_test, y_test, _ = data_prep.get_xy(test_df, target_column)

    model = build_default_model(name)
    predictions, train_seconds, predict_seconds = time_fit_predict(model, x_fit, y_fit, x_test)

    metrics = evaluate_predictions(y_test, predictions, label_list=label_list)
    metrics["train_seconds"] = train_seconds
    metrics["predict_seconds"] = predict_seconds
    metrics["num_features_used"] = 6
    metrics["selected_features"] = list(config.FEATURE_COLUMNS)
    return metrics


def evaluate_ga_on_test(name, task, gene_specs, gene_names, model_builder, splits, ga_result):
    target_column, label_list = get_task_settings(task)

    fit_df = combine_train_and_val(splits, task)
    test_df = splits[task]["test"]

    selected_features = ga_result["selected_features"]
    hyperparams = ga_result["hyperparameters"]

    # Build a feature mask from the recorded feature NAMES, so this stays
    # correct even if the column order in config ever changes.
    feature_mask = []
    for column_name in config.FEATURE_COLUMNS:
        if column_name in selected_features:
            feature_mask.append(1)
        else:
            feature_mask.append(0)

    x_fit, y_fit, used_columns = data_prep.get_xy(fit_df, target_column, feature_mask=feature_mask)
    x_test, y_test, _ = data_prep.get_xy(test_df, target_column, feature_mask=feature_mask)

    model = model_builder(hyperparams)
    predictions, train_seconds, predict_seconds = time_fit_predict(model, x_fit, y_fit, x_test)

    metrics = evaluate_predictions(y_test, predictions, label_list=label_list)
    metrics["train_seconds"] = train_seconds
    metrics["predict_seconds"] = predict_seconds
    metrics["num_features_used"] = len(used_columns)
    metrics["selected_features"] = used_columns
    metrics["hyperparameters"] = hyperparams
    return metrics


def print_comparison(all_results):
    print("\n" + "=" * 94)
    print("FINAL TEST SET RESULTS  (models re-fit on train + validation)")
    print("=" * 94)
    header = f"{'Model':<20}{'Variant':<16}{'Feats':<7}{'Accuracy (%)':<15}{'Macro F1':<12}{'Sensors'}"
    print(header)
    print("-" * 94)

    for name in all_results:
        for variant in ["baseline", "ga"]:
            r = all_results[name][variant]
            variant_label = "Baseline" if variant == "baseline" else "GA-optimized"
            sensors = ", ".join(r["selected_features"])
            print(f"{name:<20}{variant_label:<16}{r['num_features_used']:<7}"
                  f"{r['accuracy']*100:<15.2f}{r['macro_f1']:<12.4f}{sensors}")
        print("-" * 94)


def print_deltas(all_results):
    print("\nCHANGE FROM BASELINE TO GA-OPTIMIZED (test set)")
    print(f"{'Model':<20}{'Acc change':<16}{'F1 change':<16}{'Sensors'}")
    for name in all_results:
        base_r = all_results[name]["baseline"]
        ga_r = all_results[name]["ga"]

        accuracy_change = (ga_r["accuracy"] - base_r["accuracy"]) * 100
        f1_change = ga_r["macro_f1"] - base_r["macro_f1"]
        feature_change = str(base_r["num_features_used"]) + " -> " + str(ga_r["num_features_used"])

        accuracy_text = f"{accuracy_change:+.2f} pp"
        f1_text = f"{f1_change:+.4f}"
        print(f"{name:<20}{accuracy_text:<16}{f1_text:<16}{feature_change}")


def save_results(all_results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    serializable = {}
    for name in all_results:
        serializable[name] = {}
        for variant in all_results[name]:
            r = dict(all_results[name][variant])
            r["confusion_matrix"] = r["confusion_matrix"].tolist()
            serializable[name][variant] = r

    out_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print("\nSaved test set results to", out_path)


def main():
    splits = data_prep.load_and_prepare()
    ga_results = load_ga_results()

    print("Test set sizes -- binary:", len(splits["binary"]["test"]),
          "| multiclass:", len(splits["multiclass"]["test"]))

    all_results = {}
    for name, task, gene_specs, gene_names, model_builder in CLASSIFIER_SPECS:
        print("Evaluating", name, "on the test set ...")
        baseline_metrics = evaluate_baseline_on_test(name, task, splits)
        ga_metrics = evaluate_ga_on_test(
            name, task, gene_specs, gene_names, model_builder, splits, ga_results[name]
        )
        all_results[name] = {"baseline": baseline_metrics, "ga": ga_metrics}

    print_comparison(all_results)
    print_deltas(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
