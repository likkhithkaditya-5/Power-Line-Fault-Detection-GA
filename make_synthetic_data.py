import os
import numpy as np
import pandas as pd

import config

NUM_ROWS = 12000
FRACTION_NO_FAULT = 0.54  # roughly matches published stats for this dataset


def make_fault_bit_row(rng):
    is_fault = rng.random() > FRACTION_NO_FAULT
    if not is_fault:
        return 0, 0, 0, 0

    ground = rng.integers(0, 2)
    num_phases = rng.choice([1, 2, 3])
    phase_bits = [0, 0, 0]
    chosen_idx = rng.choice(3, size=num_phases, replace=False)
    for idx in chosen_idx:
        phase_bits[idx] = 1

    # A single phase fault with no ground is not physically meaningful for
    # this dataset's label scheme, so force ground on in that case.
    if num_phases == 1:
        ground = 1

    c_bit, b_bit, a_bit = phase_bits
    return ground, c_bit, b_bit, a_bit


def make_sensor_readings(rng, is_fault):
    if is_fault:
        base_current = 400.0
        base_voltage = 0.6
        current_noise = 250.0
        voltage_noise = 0.35
    else:
        base_current = 40.0
        base_voltage = 1.0
        current_noise = 15.0
        voltage_noise = 0.05

    ia = rng.normal(base_current, current_noise)
    ib = rng.normal(base_current, current_noise)
    ic = rng.normal(base_current, current_noise)
    va = rng.normal(base_voltage, voltage_noise)
    vb = rng.normal(base_voltage, voltage_noise)
    vc = rng.normal(base_voltage, voltage_noise)
    return ia, ib, ic, va, vb, vc


def build_detect_dataset(rng):
    rows = []
    for i in range(NUM_ROWS):
        is_fault = 1 if rng.random() > FRACTION_NO_FAULT else 0
        ia, ib, ic, va, vb, vc = make_sensor_readings(rng, is_fault)
        rows.append([ia, ib, ic, va, vb, vc, is_fault])

    columns = config.FEATURE_COLUMNS + ["Output"]
    return pd.DataFrame(rows, columns=columns)


def build_class_data(rng):
    rows = []
    for i in range(NUM_ROWS):
        g_bit, c_bit, b_bit, a_bit = make_fault_bit_row(rng)
        is_fault = 1 if (g_bit or c_bit or b_bit or a_bit) else 0
        ia, ib, ic, va, vb, vc = make_sensor_readings(rng, is_fault)
        rows.append([ia, ib, ic, va, vb, vc, g_bit, c_bit, b_bit, a_bit])

    columns = config.FEATURE_COLUMNS + config.FAULT_BIT_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def main():
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    detect_df = build_detect_dataset(rng)
    detect_df.to_csv(config.BINARY_DATA_PATH, index=False)
    print("Synthetic detect_dataset.csv written:", len(detect_df), "rows")

    class_df = build_class_data(rng)
    class_df.to_csv(config.MULTICLASS_DATA_PATH, index=False)
    print("Synthetic classData.csv written:", len(class_df), "rows")


if __name__ == "__main__":
    main()
