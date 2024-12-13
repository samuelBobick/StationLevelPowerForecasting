from pathlib import Path
from typing import Optional

import pandas as pd

from .input_data_type import DataSchema

COLS_TO_NORMALIZE = ["power"]
NORM_PARAMETERS_PATH = Path(__file__).parent / "saved_normalization_parameters"
NORM_PARAMETERS_PATH.mkdir(parents=True, exist_ok=True)
SINGLE_EVSE_NORMALIZATION_PARAM = 6_600


def save_series_to_csv(series: pd.Series, name: str) -> None:
    series.to_csv(NORM_PARAMETERS_PATH / f"{name}.csv")


def load_series_from_csv(name: str):
    return pd.read_csv(NORM_PARAMETERS_PATH / f"{name}.csv", index_col=0).iloc[:, 0]


def get_train_mean_and_std(
    df_train: pd.DataFrame, dataset_name: Optional[str] = None
) -> tuple[pd.Series, pd.Series]:
    """Get the mean and standard deviation of the training data, used to normalize the rest of the data.

    Args:
        df_train (pd.DataFrame): Train dataframe, should be in the format of the DataSchema.
        dataset_name (Optional[str]): Name of the dataset, used to save the \
            normalization parameters in a csv. \
            If None, the parameters are not saved.

    Returns:
        tuple[pd.Series, pd.Series]: training data mean, training data standard deviation

    Example:
    >>> train_mean, train_std = get_train_mean_and_std(df_train)
    >>> df_val_eng = feature_engineering(df_val, train_mean, train_std)
    """
    # Check that the data is in the correct format
    DataSchema.validate(df_train)

    train_mean = df_train[COLS_TO_NORMALIZE].mean()
    train_std = df_train[COLS_TO_NORMALIZE].std()

    if dataset_name:
        filename = f"{dataset_name}_mean"
        save_series_to_csv(train_mean, filename)
        filename = f"{dataset_name}_std"
        save_series_to_csv(train_std, filename)

    return train_mean, train_std


def retrieve_train_mean_and_std(dataset_name: str):
    train_mean = load_series_from_csv(f"{dataset_name}_mean")
    train_std = load_series_from_csv(f"{dataset_name}_std")
    return train_mean, train_std


def standardize_data(
    data: pd.DataFrame, train_mean: pd.Series, train_std: pd.Series
) -> pd.DataFrame:
    assert data.columns.isin(
        train_mean.index
    ).all(), "Data columns do not match the parameters columns"
    return (data - train_mean) / train_std


def reverse_standardize_data(
    data: pd.DataFrame, train_mean: pd.Series, train_std: pd.Series
) -> pd.DataFrame:
    assert data.columns.isin(
        train_mean.index
    ).all(), "Data columns do not match the parameters columns"
    return data * train_std + train_mean


def get_train_min_and_max(
    df_train: pd.DataFrame, dataset_name: Optional[str] = None
) -> tuple[pd.Series, pd.Series]:
    DataSchema.validate(df_train)

    train_min = df_train[COLS_TO_NORMALIZE].min()
    train_max = df_train[COLS_TO_NORMALIZE].max()

    # save the data if we have the dataset name
    if dataset_name:
        filename = f"{dataset_name}_min"
        save_series_to_csv(train_min, filename)
        filename = f"{dataset_name}_max"
        save_series_to_csv(train_max, filename)

    return train_min, train_max


def retrieve_train_min_and_max(dataset_name: str):
    train_min = load_series_from_csv(f"{dataset_name}_min")
    train_max = load_series_from_csv(f"{dataset_name}_max")
    return train_min, train_max


def normalize_data(
    data: pd.DataFrame, train_min: pd.Series, train_max: pd.Series
) -> pd.DataFrame:
    assert data.columns.isin(
        train_min.index
    ).all(), "Data columns do not match the parameters columns"
    return (data - train_min) / (train_max - train_min)


def reverse_normalize_data(
    data: pd.DataFrame, train_min: pd.Series, train_max: pd.Series
) -> pd.DataFrame:
    assert data.columns.isin(
        train_min.index
    ).all(), "Data columns do not match the parameters columns"
    return data * (train_max - train_min) + train_min
