import re
from typing import Optional

import pandas as pd


def clean_str(string: str) -> str:
    return string.replace("_", "\n").capitalize()


def apply_filter(
    df_results: pd.DataFrame, or_filter_model_name: Optional[list[str]]
) -> pd.DataFrame:
    if or_filter_model_name:
        filtered_df = pd.DataFrame()
        for model_name_filter in or_filter_model_name:
            filtered_df = pd.concat(
                [
                    filtered_df,
                    df_results[
                        df_results["model_name"].str.contains(model_name_filter)
                    ],
                ]
            )
        return filtered_df
    return df_results


def get_groupby_columns(df_results: pd.DataFrame) -> list[str]:
    groupby_columns = [
        col
        for col in df_results.columns
        if col
        not in [
            "date",
            "model_name",
            "rmse",
            "wrmse (alpha=2)",
            "wprmse (beta=3)",
            "mae",
            "r2",
            "error_std",
            "elapsed_time",
            "relative_rmse",
            "smape",
        ]
    ]
    return groupby_columns


def extract_number_keys(model_name):
    # Pattern for searching for keywords followed by numeric values
    # such as "dropout0.1" or "maxDepth6"
    pattern = r"([a-zA-Z]+)(\d+(\.\d+)?)"
    matches = re.findall(pattern, model_name)
    return {match[0]: float(match[1]) for match in matches}


def extract_categorical(model_name):
    # Pattern for searching for categorical values (without number attached to them)
    pattern = r"([a-zA-Z]+)(?!\d+)(_|\b)"
    matches = re.findall(pattern, model_name)
    return [match[0] for match in matches[1:]]


def parse_model_names(df_results: pd.DataFrame) -> pd.DataFrame:
    df_results = df_results.copy()
    # Apply functions to create new columns for numerical values
    numbers_df = df_results["model_name"].apply(extract_number_keys).apply(pd.Series)

    # Merge the new numerical DataFrame with the original DataFrame
    df_results = pd.concat([df_results, numbers_df], axis=1)

    # Extract categorical values and create corresponding columns
    categorical_cols = (
        df_results["model_name"].apply(extract_categorical).explode().unique()
    )
    for cat in categorical_cols:
        if pd.isna(cat):
            continue
        df_results[cat] = df_results["model_name"].apply(lambda x: cat in x)

    # Parses the model name to only have the model type (first word separated by _)
    df_results["model_name"] = df_results["model_name"].apply(lambda x: x.split("_")[0])

    return df_results


def parse_group_index_to_name(group_by_columns: list, group_index: tuple) -> str:
    name = ""
    for i, column_name in enumerate(group_by_columns):
        name += f"{column_name}={group_index[i]}\n"
    return name
