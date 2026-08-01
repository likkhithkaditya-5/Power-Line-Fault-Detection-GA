import baseline_models
import run_ga_experiments
import make_plots


def main():
    print("=" * 60)
    print("STEP 1 / 3 -- Baseline models")
    print("=" * 60)
    baseline_models.main()

    print("\n" + "=" * 60)
    print("STEP 2 / 3 -- GA optimization")
    print("=" * 60)
    run_ga_experiments.main()

    print("\n" + "=" * 60)
    print("STEP 3 / 3 -- Plots")
    print("=" * 60)
    make_plots.main()

    print("\nAll done. Results in results/, plots in results/plots/.")


if __name__ == "__main__":
    main()
