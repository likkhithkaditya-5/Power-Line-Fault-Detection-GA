import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

CLASSIFIER_ORDER = ["LogisticRegression", "SVM", "KNN", "DecisionTree", "RandomForest"]
BINARY_MODELS = ["LogisticRegression", "SVM", "KNN"]
MULTICLASS_MODELS = ["DecisionTree", "RandomForest"]


def load_results():
    with open(os.path.join(config.RESULTS_DIR, "baseline_results.json")) as f:
        baseline_results = json.load(f)
    with open(os.path.join(config.RESULTS_DIR, "ga_results.json")) as f:
        ga_results = json.load(f)
    return baseline_results, ga_results


def plot_convergence(ga_results):
    figure, axes = plt.subplots(1, len(CLASSIFIER_ORDER), figsize=(4 * len(CLASSIFIER_ORDER), 4))

    for i in range(len(CLASSIFIER_ORDER)):
        name = CLASSIFIER_ORDER[i]
        history = ga_results[name]["fitness_history"]
        generations = list(range(1, len(history) + 1))

        axes[i].plot(generations, history, color="#2563eb", linewidth=2)
        axes[i].set_title(name)
        axes[i].set_xlabel("Generation")
        if i == 0:
            axes[i].set_ylabel("Best fitness (macro-F1 - penalty)")
        axes[i].grid(alpha=0.3)

    figure.suptitle("GA Convergence: Best Fitness per Generation")
    figure.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "ga_convergence.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("Saved", out_path)


def plot_confusion_matrices(ga_results):
    figure, axes = plt.subplots(1, len(CLASSIFIER_ORDER), figsize=(4 * len(CLASSIFIER_ORDER), 4))

    for i in range(len(CLASSIFIER_ORDER)):
        name = CLASSIFIER_ORDER[i]
        matrix = np.array(ga_results[name]["confusion_matrix"])

        if name in BINARY_MODELS:
            tick_labels = ["No Fault", "Fault"]
        else:
            tick_labels = config.FAULT_TYPE_LABELS

        axes[i].imshow(matrix, cmap="Blues")
        axes[i].set_title(name)
        axes[i].set_xticks(range(len(tick_labels)))
        axes[i].set_yticks(range(len(tick_labels)))
        axes[i].set_xticklabels(tick_labels, rotation=45, ha="right")
        axes[i].set_yticklabels(tick_labels)
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("Actual")

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axes[i].text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)

    figure.suptitle("Confusion Matrices -- GA-Optimized Models (Validation Set)")
    figure.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "confusion_matrices.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("Saved", out_path)


def plot_feature_accuracy_tradeoff(baseline_results, ga_results):
    figure, ax = plt.subplots(figsize=(7, 5))

    for name in CLASSIFIER_ORDER:
        if name in BINARY_MODELS:
            baseline_metrics = baseline_results["binary"][name]
        else:
            baseline_metrics = baseline_results["multiclass"][name]
        ga_metrics = ga_results[name]

        baseline_x = baseline_metrics["num_features_used"]
        baseline_y = baseline_metrics["accuracy"] * 100
        ga_x = ga_metrics["num_features_used"]
        ga_y = ga_metrics["accuracy"] * 100

        ax.plot([baseline_x, ga_x], [baseline_y, ga_y], color="#9ca3af", linewidth=1, zorder=1)
        ax.scatter([baseline_x], [baseline_y], color="#94a3b8", marker="o", s=80, zorder=2,
                   label="Baseline (all 6 features)" if name == CLASSIFIER_ORDER[0] else None)
        ax.scatter([ga_x], [ga_y], color="#dc2626", marker="^", s=90, zorder=3,
                   label="GA-optimized" if name == CLASSIFIER_ORDER[0] else None)
        ax.annotate(name, (ga_x, ga_y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Number of sensor features used")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. Feature Count: Baseline vs. GA-Optimized")
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(alpha=0.3)
    ax.legend()

    figure.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "feature_accuracy_tradeoff.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("Saved", out_path)


def main():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    baseline_results, ga_results = load_results()

    plot_convergence(ga_results)
    plot_confusion_matrices(ga_results)
    plot_feature_accuracy_tradeoff(baseline_results, ga_results)


if __name__ == "__main__":
    main()
