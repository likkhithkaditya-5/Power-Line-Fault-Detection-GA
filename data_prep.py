import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def clean_column_names(data_frame):
    renamed = {}
    for column_name in data_frame.columns:
        renamed[column_name] = column_name.strip()
    return data_frame.rename(columns=renamed)


def drop_unnamed_columns(data_frame):
    columns_to_drop = []
    for column_name in data_frame.columns:
        if column_name.startswith("Unnamed"):
            columns_to_drop.append(column_name)
    if len(columns_to_drop) > 0:
        data_frame = data_frame.drop(columns=columns_to_drop)
    return data_frame


def require_columns(data_frame, required_columns, source_file_label):
    missing_columns = []
    for column_name in required_columns:
        if column_name not in data_frame.columns:
            missing_columns.append(column_name)

    if len(missing_columns) > 0:
        raise ValueError(
            "Missing expected column(s) " + str(missing_columns) + " in " + source_file_label + ". "
            "Columns found were: " + str(list(data_frame.columns)) + ". "
            "Check the real header names and update them here (or in config.py)."
        )


def find_binary_target_column(data_frame):
    for candidate in config.BINARY_TARGET_CANDIDATES:
        if candidate in data_frame.columns:
            return candidate

    raise ValueError(
        "Could not find a binary target column in detect_dataset.csv. "
        "Columns found were: " + str(list(data_frame.columns)) + ". "
        "Add the real column name to config.BINARY_TARGET_CANDIDATES."
    )


def load_binary_raw():
    data_frame = pd.read_csv(config.BINARY_DATA_PATH)
    data_frame = clean_column_names(data_frame)
    data_frame = drop_unnamed_columns(data_frame)
    require_columns(data_frame, config.FEATURE_COLUMNS, "detect_dataset.csv")

    target_column = find_binary_target_column(data_frame)
    data_frame = data_frame.copy()
    data_frame["BINARY_TARGET"] = data_frame[target_column].astype(int)
    return data_frame


def bits_to_fault_type(g_bit, c_bit, b_bit, a_bit):
    num_phases = c_bit + b_bit + a_bit

    if num_phases == 0 and g_bit == 0:
        return "NONE"  # no fault; caller should exclude these rows
    if num_phases == 1:
        return "LG"
    if num_phases == 2 and g_bit == 0:
        return "LL"
    if num_phases == 2 and g_bit == 1:
        return "LLG"
    if num_phases == 3 and g_bit == 0:
        return "LLL"
    if num_phases == 3 and g_bit == 1:
        return "LLLG"

    return "UNKNOWN"  # a combination that shouldn't occur (e.g. ground-only)


def load_multiclass_raw():
    data_frame = pd.read_csv(config.MULTICLASS_DATA_PATH)
    data_frame = clean_column_names(data_frame)
    data_frame = drop_unnamed_columns(data_frame)
    require_columns(data_frame, config.FEATURE_COLUMNS + config.FAULT_BIT_COLUMNS, "classData.csv")

    fault_types = []
    for i in range(len(data_frame)):
        row = data_frame.iloc[i]
        g_bit = int(row["G"])
        c_bit = int(row["C"])
        b_bit = int(row["B"])
        a_bit = int(row["A"])
        fault_types.append(bits_to_fault_type(g_bit, c_bit, b_bit, a_bit))

    data_frame = data_frame.copy()
    data_frame["FAULT_TYPE"] = fault_types
    return data_frame


def report_unknown_rows(data_frame):
    num_unknown = int((data_frame["FAULT_TYPE"] == "UNKNOWN").sum())
    if num_unknown > 0:
        print("WARNING:", num_unknown, "rows in classData.csv had an unexpected",
              "G-C-B-A combination and were excluded from the multiclass task.")


def make_binary_splits(data_frame):
    train_df, temp_df = train_test_split(
        data_frame,
        train_size=config.TRAIN_FRACTION,
        stratify=data_frame["BINARY_TARGET"],
        random_state=config.RANDOM_SEED,
    )

    val_share_of_temp = config.VAL_FRACTION / (config.VAL_FRACTION + config.TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        stratify=temp_df["BINARY_TARGET"],
        random_state=config.RANDOM_SEED,
    )
    return train_df, val_df, test_df


def make_multiclass_splits(data_frame):
    fault_only = data_frame[data_frame["FAULT_TYPE"].isin(config.FAULT_TYPE_LABELS)]

    train_df, temp_df = train_test_split(
        fault_only,
        train_size=config.TRAIN_FRACTION,
        stratify=fault_only["FAULT_TYPE"],
        random_state=config.RANDOM_SEED,
    )

    val_share_of_temp = config.VAL_FRACTION / (config.VAL_FRACTION + config.TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        stratify=temp_df["FAULT_TYPE"],
        random_state=config.RANDOM_SEED,
    )
    return train_df, val_df, test_df


def get_xy(data_frame, target_column, feature_mask=None):
    if feature_mask is None:
        columns_to_use = config.FEATURE_COLUMNS
    else:
        columns_to_use = []
        for i in range(len(config.FEATURE_COLUMNS)):
            if feature_mask[i] == 1:
                columns_to_use.append(config.FEATURE_COLUMNS[i])

    x_values = data_frame[columns_to_use].values
    y_values = data_frame[target_column].values
    return x_values, y_values, columns_to_use


def load_and_prepare():
    binary_frame = load_binary_raw()
    multiclass_frame = load_multiclass_raw()
    report_unknown_rows(multiclass_frame)

    binary_train, binary_val, binary_test = make_binary_splits(binary_frame)
    multi_train, multi_val, multi_test = make_multiclass_splits(multiclass_frame)

    return {
        "binary": {"train": binary_train, "val": binary_val, "test": binary_test},
        "multiclass": {"train": multi_train, "val": multi_val, "test": multi_test},
    }


if __name__ == "__main__":
    splits = load_and_prepare()

    print("Binary task  -- train:", len(splits["binary"]["train"]),
          "val:", len(splits["binary"]["val"]),
          "test:", len(splits["binary"]["test"]))
    print("Multiclass task -- train:", len(splits["multiclass"]["train"]),
          "val:", len(splits["multiclass"]["val"]),
          "test:", len(splits["multiclass"]["test"]))
    print("Multiclass label counts (train):")
    print(splits["multiclass"]["train"]["FAULT_TYPE"].value_counts())
