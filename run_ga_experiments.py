import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import config
import data_prep
import ga_common
from evaluation import evaluate_predictions, time_fit_predict


def build_logistic_regression(hyperparams):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=hyperparams["C"], random_state=config.RANDOM_SEED, max_iter=1000)),
    ])
    return model


def build_svm(hyperparams):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(C=hyperparams["C"], gamma=hyperparams["gamma"], random_state=config.RANDOM_SEED)),
    ])
    return model


def build_knn(hyperparams):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=hyperparams["n_neighbors"])),
    ])
    return model


def build_decision_tree(hyperparams):
    return DecisionTreeClassifier(max_depth=hyperparams["max_depth"], random_state=config.RANDOM_SEED)


def build_random_forest(hyperparams):
    return RandomForestClassifier(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        random_state=config.RANDOM_SEED,
    )


# One entry per classifier: (name, task, gene_specs, gene_names, model_builder)
CLASSIFIER_SPECS = [
    ("LogisticRegression", "binary", [("float", 0.01, 50.0)], ["C"], build_logistic_regression),
    ("SVM", "binary", [("float", 0.01, 50.0), ("float", 0.0001, 5.0)], ["C", "gamma"], build_svm),
    ("KNN", "binary", [("int", 1, 40)], ["n_neighbors"], build_knn),
    ("DecisionTree", "multiclass", [("int", 1, 25)], ["max_depth"], build_decision_tree),
    ("RandomForest", "multiclass", [("int", 10, 150), ("int", 1, 25)], ["n_estimators", "max_depth"], build_random_forest),
]


def make_search_subsample(x_train_full, y_train):
    sample_size = min(config.GA_SEARCH_SAMPLE_SIZE, len(x_train_full))
    if sample_size >= len(x_train_full):
        return x_train_full, y_train

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=config.RANDOM_SEED)
    sample_indices, _ = next(splitter.split(x_train_full, y_train))
    return x_train_full[sample_indices], y_train[sample_indices]


def run_one_classifier(name, task, gene_specs, gene_names, model_builder, splits,
                       penalty_lambda=None, seed=None):
    if seed is not None:
        ga_common.set_ga_seed(seed)

    if task == "binary":
        target_column = "BINARY_TARGET"
        label_list = [0, 1]
    else:
        target_column = "FAULT_TYPE"
        label_list = config.FAULT_TYPE_LABELS

    train_df = splits[task]["train"]
    val_df = splits[task]["val"]

    x_train_full, y_train, _ = data_prep.get_xy(train_df, target_column)
    x_val_full, y_val, _ = data_prep.get_xy(val_df, target_column)

    x_search, y_search = make_search_subsample(x_train_full, y_train)

    evaluate_function = ga_common.build_fitness_function(
        model_builder, gene_specs, gene_names, x_search, y_search, config.GA_CV_FOLDS,
        penalty_lambda=penalty_lambda
    )
    toolbox = ga_common.build_toolbox(gene_specs, evaluate_function)
    best_individual, fitness_history = ga_common.run_ga(toolbox)

    feature_mask, hyperparams = ga_common.decode_individual(best_individual, gene_specs, gene_names)

    selected_columns = []
    for i in range(ga_common.NUM_FEATURE_BITS):
        if feature_mask[i] == 1:
            selected_columns.append(i)
    selected_names = []
    for i in selected_columns:
        selected_names.append(config.FEATURE_COLUMNS[i])

    # Final fit on the FULL training set (not the search subsample) and
    # evaluation on the validation set, so this number is directly
    # comparable to the baseline table.
    x_train_selected = x_train_full[:, selected_columns]
    x_val_selected = x_val_full[:, selected_columns]

    final_model = model_builder(hyperparams)
    predictions, train_seconds, predict_seconds = time_fit_predict(
        final_model, x_train_selected, y_train, x_val_selected
    )
    metrics = evaluate_predictions(y_val, predictions, label_list=label_list)
    metrics["train_seconds"] = train_seconds
    metrics["predict_seconds"] = predict_seconds
    metrics["num_features_used"] = len(selected_columns)
    metrics["selected_features"] = selected_names
    metrics["feature_mask"] = feature_mask
    metrics["hyperparameters"] = hyperparams
    metrics["fitness_history"] = fitness_history
    metrics["best_fitness"] = float(best_individual.fitness.values[0])

    return metrics


def print_summary(all_results):
    print(f"\n{'Model':<20}{'Features':<10}{'Accuracy (%)':<15}{'Macro F1':<12}{'Selected sensors'}")
    for name in all_results:
        r = all_results[name]
        feature_list_text = ", ".join(r["selected_features"])
        print(f"{name:<20}{r['num_features_used']:<10}{r['accuracy']*100:<15.2f}"
              f"{r['macro_f1']:<12.4f}{feature_list_text}")


def save_results(all_results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    serializable = {}
    for name in all_results:
        r = dict(all_results[name])
        r["confusion_matrix"] = r["confusion_matrix"].tolist()
        serializable[name] = r

    out_path = os.path.join(config.RESULTS_DIR, "ga_results.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print("\nSaved GA results to", out_path)


def main():
    splits = data_prep.load_and_prepare()

    all_results = {}
    for name, task, gene_specs, gene_names, model_builder in CLASSIFIER_SPECS:
        print("\nRunning GA for", name, "...")
        result = run_one_classifier(name, task, gene_specs, gene_names, model_builder, splits)
        all_results[name] = result
        print(name, "done. Best val accuracy:", round(result["accuracy"] * 100, 2),
              "%  |  features used:", result["num_features_used"],
              "|  selected:", result["selected_features"])

    print_summary(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
