import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_prep
from run_ga_experiments import CLASSIFIER_SPECS, run_one_classifier

# Sweep values. 0.0 = accuracy only (no complexity penalty);
# 0.30 = a sensor must earn its place quite convincingly.
LAMBDA_VALUES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]


def run_sweep_for_classifier(name, task, gene_specs, gene_names, model_builder, splits):
    records = []

    for penalty_lambda in LAMBDA_VALUES:
        result = run_one_classifier(
            name, task, gene_specs, gene_names, model_builder, splits,
            penalty_lambda=penalty_lambda,
            seed=config.RANDOM_SEED,
        )
        records.append({
            "lambda": penalty_lambda,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "num_features_used": result["num_features_used"],
            "selected_features": result["selected_features"],
            "hyperparameters": result["hyperparameters"],
        })
        print("   lambda", penalty_lambda,
              "-> acc", round(result["accuracy"] * 100, 2),
              "% | features", result["num_features_used"],
              "|", result["selected_features"])

    return records


def print_summary(all_results):
    print("\n" + "=" * 100)
    print("LAMBDA SWEEP  (validation set)")
    print("=" * 100)

    for name in all_results:
        print("\n" + name)
        print(f"   {'lambda':<10}{'Features':<12}{'Accuracy (%)':<16}{'Macro F1':<12}{'Sensors'}")
        for record in all_results[name]:
            sensors = ", ".join(record["selected_features"])
            print(f"   {record['lambda']:<10}{record['num_features_used']:<12}"
                  f"{record['accuracy']*100:<16.2f}{record['macro_f1']:<12.4f}{sensors}")


def plot_pareto_frontier(all_results):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    binary_models = ["LogisticRegression", "SVM", "KNN"]
    colours = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    for panel_index in range(2):
        ax = axes[panel_index]
        if panel_index == 0:
            model_subset = []
            for name in all_results:
                if name in binary_models:
                    model_subset.append(name)
            panel_title = "Binary fault detection"
        else:
            model_subset = []
            for name in all_results:
                if name not in binary_models:
                    model_subset.append(name)
            panel_title = "Multi-class fault type"

        for i in range(len(model_subset)):
            name = model_subset[i]
            records = all_results[name]

            feature_counts = []
            accuracies = []
            for record in records:
                feature_counts.append(record["num_features_used"])
                accuracies.append(record["accuracy"] * 100)

            ax.plot(feature_counts, accuracies, marker="o", color=colours[i],
                    label=name, linewidth=1.5, markersize=7, alpha=0.85)

            for j in range(len(records)):
                ax.annotate(f"λ={records[j]['lambda']}",
                            (feature_counts[j], accuracies[j]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.set_xlabel("Number of sensor features retained")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title(panel_title)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    figure.suptitle("Accuracy vs. Complexity Trade-off as the Feature Penalty λ Varies")
    figure.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(config.PLOTS_DIR, "lambda_pareto_frontier.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("\nSaved", out_path)


def plot_features_vs_lambda(all_results):
    figure, ax = plt.subplots(figsize=(8, 5))
    colours = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    model_names = list(all_results.keys())
    for i in range(len(model_names)):
        name = model_names[i]
        records = all_results[name]

        lambda_values = []
        feature_counts = []
        for record in records:
            lambda_values.append(record["lambda"])
            feature_counts.append(record["num_features_used"])

        ax.plot(lambda_values, feature_counts, marker="o", color=colours[i],
                label=name, linewidth=1.5, markersize=6, alpha=0.85)

    ax.set_xlabel("Feature penalty λ")
    ax.set_ylabel("Sensors retained by the GA")
    ax.set_title("Effect of the Complexity Penalty on Feature Count")
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    figure.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "lambda_vs_feature_count.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("Saved", out_path)


def save_results(all_results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "lambda_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved lambda sweep results to", out_path)


def main():
    splits = data_prep.load_and_prepare()

    all_results = {}
    for name, task, gene_specs, gene_names, model_builder in CLASSIFIER_SPECS:
        print("\nSweeping lambda for", name, "...")
        all_results[name] = run_sweep_for_classifier(
            name, task, gene_specs, gene_names, model_builder, splits
        )

    print_summary(all_results)
    plot_pareto_frontier(all_results)
    plot_features_vs_lambda(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
