import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_prep
from run_ga_experiments import CLASSIFIER_SPECS, run_one_classifier

NUM_SEEDS = 5


def summarize(values):
    array = np.array(values, dtype=float)
    return float(np.mean(array)), float(np.std(array))


def run_seeds_for_classifier(name, task, gene_specs, gene_names, model_builder, splits):
    per_seed_records = []

    for seed_index in range(NUM_SEEDS):
        seed_value = config.RANDOM_SEED + seed_index
        result = run_one_classifier(
            name, task, gene_specs, gene_names, model_builder, splits, seed=seed_value
        )
        per_seed_records.append({
            "seed": seed_value,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "num_features_used": result["num_features_used"],
            "selected_features": result["selected_features"],
            "hyperparameters": result["hyperparameters"],
            "best_fitness": result["best_fitness"],
        })
        print("   seed", seed_value,
              "-> acc", round(result["accuracy"] * 100, 2),
              "% | features", result["num_features_used"],
              "|", result["selected_features"])

    accuracy_values = []
    f1_values = []
    feature_count_values = []
    for record in per_seed_records:
        accuracy_values.append(record["accuracy"])
        f1_values.append(record["macro_f1"])
        feature_count_values.append(record["num_features_used"])

    accuracy_mean, accuracy_std = summarize(accuracy_values)
    f1_mean, f1_std = summarize(f1_values)
    feature_mean, feature_std = summarize(feature_count_values)

    # How often each sensor was selected across the seeds.
    selection_counts = {}
    for column_name in config.FEATURE_COLUMNS:
        selection_counts[column_name] = 0
    for record in per_seed_records:
        for column_name in record["selected_features"]:
            selection_counts[column_name] = selection_counts[column_name] + 1

    return {
        "per_seed": per_seed_records,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "num_features_mean": feature_mean,
        "num_features_std": feature_std,
        "selection_counts": selection_counts,
    }


def print_summary(all_results):
    print("\n" + "=" * 88)
    print("GA ROBUSTNESS ACROSS", NUM_SEEDS, "SEEDS  (validation set, mean +/- std)")
    print("=" * 88)
    print(f"{'Model':<20}{'Accuracy (%)':<22}{'Macro F1':<22}{'Features'}")
    print("-" * 88)
    for name in all_results:
        r = all_results[name]
        accuracy_text = f"{r['accuracy_mean']*100:.2f} +/- {r['accuracy_std']*100:.2f}"
        f1_text = f"{r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}"
        feature_text = f"{r['num_features_mean']:.1f} +/- {r['num_features_std']:.1f}"
        print(f"{name:<20}{accuracy_text:<22}{f1_text:<22}{feature_text}")

    print("\n" + "=" * 88)
    print("SENSOR SELECTION FREQUENCY  (out of", NUM_SEEDS, "seeds)")
    print("=" * 88)
    header = f"{'Model':<20}"
    for column_name in config.FEATURE_COLUMNS:
        header = header + f"{column_name:<8}"
    print(header)
    print("-" * 88)
    for name in all_results:
        line = f"{name:<20}"
        counts = all_results[name]["selection_counts"]
        for column_name in config.FEATURE_COLUMNS:
            line = line + f"{counts[column_name]}/{NUM_SEEDS}     "
        print(line)


def plot_selection_frequency(all_results):
    model_names = list(all_results.keys())
    matrix = np.zeros((len(model_names), len(config.FEATURE_COLUMNS)))

    for i in range(len(model_names)):
        counts = all_results[model_names[i]]["selection_counts"]
        for j in range(len(config.FEATURE_COLUMNS)):
            matrix[i, j] = counts[config.FEATURE_COLUMNS[j]] / NUM_SEEDS

    figure, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(config.FEATURE_COLUMNS)))
    ax.set_xticklabels(config.FEATURE_COLUMNS)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Sensor channel")
    ax.set_title("How often the GA selected each sensor (" + str(NUM_SEEDS) + " seeds)")

    for i in range(len(model_names)):
        for j in range(len(config.FEATURE_COLUMNS)):
            fraction = matrix[i, j]
            text_colour = "white" if fraction > 0.5 else "black"
            ax.text(j, i, f"{int(fraction * NUM_SEEDS)}/{NUM_SEEDS}",
                    ha="center", va="center", color=text_colour, fontsize=9)

    figure.colorbar(image, ax=ax, label="Selection frequency")
    figure.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(config.PLOTS_DIR, "sensor_selection_frequency.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("\nSaved", out_path)


def plot_accuracy_spread(all_results):
    model_names = list(all_results.keys())

    figure, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(len(model_names)):
        records = all_results[model_names[i]]["per_seed"]
        accuracy_points = []
        for record in records:
            accuracy_points.append(record["accuracy"] * 100)
        x_positions = [i] * len(accuracy_points)
        ax.scatter(x_positions, accuracy_points, color="#2563eb", alpha=0.7, s=50, zorder=3)

        mean_value = all_results[model_names[i]]["accuracy_mean"] * 100
        ax.scatter([i], [mean_value], color="#dc2626", marker="_", s=600, zorder=4)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Per-seed GA outcomes (red bar = mean across seeds)")
    ax.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "ga_seed_variability.png")
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    print("Saved", out_path)


def save_results(all_results):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved multi-seed results to", out_path)


def main():
    splits = data_prep.load_and_prepare()

    all_results = {}
    for name, task, gene_specs, gene_names, model_builder in CLASSIFIER_SPECS:
        print("\nRunning", NUM_SEEDS, "seeds for", name, "...")
        all_results[name] = run_seeds_for_classifier(
            name, task, gene_specs, gene_names, model_builder, splits
        )

    print_summary(all_results)
    plot_selection_frequency(all_results)
    plot_accuracy_spread(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
