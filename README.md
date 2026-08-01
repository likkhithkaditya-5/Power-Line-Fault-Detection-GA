# Power Line Fault Detection and Classification with a Genetic Algorithm

ECE 569A — Selected Topics in Computer Engineering: Artificial Intelligence
University of Victoria, Summer 2026 — Grad Group 19

**Victor Manoraj Shanmugam** (V01100086) · **Likkhith Krishnamurthy Aditya** (V01098937)

---

## Overview

Machine learning system for detecting and classifying faults on three-phase power
transmission lines, using a **Genetic Algorithm to jointly optimise sensor channel
selection and classifier hyperparameters**.

The research question is not just *how accurately can we detect faults*, but
**how few sensor channels are actually needed** to do so. Every channel removed
is one less physical device to purchase, install, calibrate and maintain.

Two tasks are addressed:

| Task | Classes | Models |
|---|---|---|
| Binary detection | Fault / No-Fault | Logistic Regression, SVM, KNN |
| Multi-class typing | LG, LL, LLG, LLL, LLLG | Decision Tree, Random Forest |

---

## Key Results (held-out test set)

| Task | Classifier | Baseline (6 ch.) | GA channels | GA accuracy | Δ |
|---|---|---|---|---|---|
| Binary | Logistic Regression | 72.40% | `Ia, Ic` | 74.79% | **+2.39 pp** |
| Binary | SVM | 98.45% | `Ia, Ib` | 99.00% | **+0.56 pp** |
| Binary | KNN | 99.22% | `Ia, Ib` | 99.22% | 0.00 pp |
| Multi-class | Decision Tree | 83.88% | `Ib, Ic, Vc` | 80.00% | −3.88 pp |
| Multi-class | Random Forest | 81.82% | `Ia, Ib, Ic` | 82.42% | **+0.61 pp** |

**Four of five classifiers match or beat their six-channel baseline while using
two or three channels** — a two-thirds hardware reduction at no accuracy cost.

Further findings:

- The GA consistently selects **line-current channels** and discards voltage
  channels for binary detection (stable across all 5 random seeds).
- **98% of all multi-class errors are a single LLL/LLLG confusion.** The other
  three fault categories classify at 99.6% accuracy — the performance gap is one
  physically-explicable failure mode, not general model weakness.
- Optimisation also cut runtime: Random Forest training 1,530 ms → 194 ms,
  inference 30.7 µs → 3.5 µs per sample.

---

## The Genetic Algorithm

Each chromosome is a **6-bit feature mask concatenated with hyperparameter genes**:

```
[ Ia Ib Ic Va Vb Vc | hyperparameter genes... ]
```

| Model | Hyperparameter genes |
|---|---|
| Logistic Regression | `C` (continuous) |
| SVM | `C`, `gamma` (continuous) |
| KNN | `n_neighbors` (integer) |
| Decision Tree | `max_depth` (integer) |
| Random Forest | `n_estimators`, `max_depth` (integer) |

Because `C` and `gamma` are continuous, the joint search space is **uncountable** —
exhaustive enumeration is impossible, which is what justifies evolutionary search.
(The 6-bit mask alone would only be 2⁶ = 64 subsets.)

**Fitness function:**

```
fitness(s) = MacroF1(s) − λ · (|s| / 6)
```

Macro-averaged F1 is used rather than plain accuracy so every fault class counts
equally under class imbalance. The penalty is linear in channel count and
normalised to [0,1], making λ interpretable directly against macro-F1.

**Configuration** (see `config.py`):

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 15 |
| Selection | Tournament (size 3), with elitism |
| Crossover | Uniform, p = 0.7 |
| Mutation | p = 0.3 (15% per-bit on mask, 30% per-gene on hyperparameters) |
| λ (complexity weight) | 0.05 |
| Fitness evaluation | 3-fold stratified cross-validated macro-F1 |
| Independent seeds | 5 |

---

## Dataset

Kaggle — [Electrical Fault Detection and Classification](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification)
(E. Sathyaprakash, 2021), generated from a MATLAB Simulink model of a
four-generator 11 kV system with faults applied at the transmission line midpoint.

**The dataset is not committed to this repository.** Download it and place the two
CSVs as follows:

```
data/detect_dataset.csv    # binary fault detection
data/classData.csv         # fault-type classification (G, C, B, A label bits)
```

Six input features are used throughout: `Ia, Ib, Ic, Va, Vb, Vc`.
The `G, C, B, A` columns in `classData.csv` are **label bits**, not features —
`data_prep.py` decodes them into the five fault categories.

If you want to run the pipeline without downloading the real data,
`make_synthetic_data.py` generates stand-in CSVs with the same schema for
smoke-testing. Results from synthetic data are **not** the results reported above.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11 recommended.

---

## Usage

Quick path — baselines, GA, and core plots:

```bash
python run_all.py
```

Full pipeline, in order (`run_all.py` covers steps 1–3 only):

```bash
python baseline_models.py      # 1. baselines on all 6 channels  -> baseline_results.json
python run_ga_experiments.py   # 2. GA optimisation              -> ga_results.json
python make_plots.py           # 3. convergence, confusion, tradeoff plots
python lambda_sweep.py         # 4. sweep λ                      -> lambda_sweep_results.json
python multi_seed_ga.py        # 5. 5-seed robustness            -> multiseed_results.json
python evaluate_test_set.py    # 6. held-out test evaluation     -> test_results.json
```

Step 6 requires step 2 to have run first (it reads `ga_results.json`).

---

## Repository Structure

```
├── config.py                  # all paths, split fractions, GA parameters
├── data_prep.py               # loading, label decoding, stratified 70/15/15 splits
├── evaluation.py              # accuracy / macro-F1 / confusion matrix, timing
├── baseline_models.py         # default-hyperparameter baselines, all 6 channels
├── ga_common.py               # DEAP setup: encoding, fitness, operators, evolution loop
├── run_ga_experiments.py      # per-classifier GA runs and model builders
├── lambda_sweep.py            # λ sensitivity sweep + Pareto plots
├── multi_seed_ga.py           # 5-seed robustness + sensor selection frequency
├── evaluate_test_set.py       # final held-out test evaluation (re-fits on train+val)
├── make_plots.py              # convergence, confusion matrices, accuracy/feature tradeoff
├── make_synthetic_data.py     # optional synthetic stand-in data
├── run_all.py                 # convenience runner (steps 1-3)
├── data/                      # (not committed) place Kaggle CSVs here
└── results/                   # JSON outputs + plots/
```

---

## Methodology Notes

- Data is split **70/15/15** with stratified sampling on the target label.
  The test partition is held out entirely and used only for final reported figures.
- **Cross-validation is used only inside the GA's fitness evaluation.** Baseline
  models and the final re-fit are each evaluated once on the single split.
- During search, fitness is evaluated on a **stratified subsample of up to 2,000
  training rows** for speed. The selected configuration is then **re-fit on the full
  training set** before any reported number is produced — no reported accuracy is
  measured on the subsample.
- A degenerate all-zero mask (no channels selected) is assigned fitness −1, below
  any achievable macro-F1, so selection discards it.
- All randomness is seeded from `config.RANDOM_SEED`; multi-seed runs use
  `RANDOM_SEED + 0..4`.

---

## References

1. M. Sajjad et al., "Robust fault detection and classification in power transmission
   lines via ensemble machine learning models," *Scientific Reports*, vol. 15,
   art. 86554, 2025. doi:10.1038/s41598-025-86554-2
2. L. Zhuo et al., "A genetic algorithm based wrapper feature selection method for
   classification of hyperspectral images using support vector machine,"
   *Int. Archives of the Photogrammetry, Remote Sensing and Spatial Information
   Sciences*, vol. 37, pp. 397–402, 2008.
3. O. Soufan et al., "DWFS: A wrapper feature selection tool based on a parallel
   genetic algorithm," *PLoS ONE*, vol. 10, no. 2, e0117988, 2015.
4. E. Sathyaprakash, "Electrical fault detection and classification," Kaggle, 2021.
