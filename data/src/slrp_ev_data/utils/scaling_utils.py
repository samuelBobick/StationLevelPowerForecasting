from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from slrp_ev_data.utils.data_utils import get_data_frequency, get_date_column_name
from slrp_ev_data.utils.input_data_type import DataSchema
from slrp_ev_ts_forecasting.default_parameters import (
    TIMESTEPS_ROLLING_WINDOW_FOR_SCALING,
    TypeDatasetName,
)

COLS_TO_NORMALIZE = ["power", "number_of_evses_available"]
NORM_PARAMETERS_PATH = Path(__file__).parent.parent / "saved_normalization_parameters"
NORM_PARAMETERS_PATH.mkdir(parents=True, exist_ok=True)
SINGLE_EVSE_NORMALIZATION_PARAM = 6_800


def save_series_to_csv(
    series: pd.Series, name: str, add_to_existing_save: bool
) -> None:
    if add_to_existing_save:
        existing_series = load_series_from_csv(name)
        series = pd.concat([existing_series, series])

    series.to_csv(NORM_PARAMETERS_PATH / f"{name}.csv")


def load_series_from_csv(name: str) -> pd.Series:
    return pd.read_csv(NORM_PARAMETERS_PATH / f"{name}.csv", index_col=0).iloc[:, 0]


def get_train_mean_and_std(
    df_train: pd.DataFrame,
    dataset_name: Optional[TypeDatasetName] = None,
    bypass_validation: bool = False,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
    add_to_existing_save: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Get the mean and standard deviation of the cols_to_normalize columns of the \
        training data, used to normalize the rest of the data.

    Args:
        df_train (pd.DataFrame): Train dataframe, should be in the format of the DataSchema.
        bypass_validation (bool): Whether to validate the input DataFrame type. \
            Set it to False unless you know what you are doing. Default is False.
        dataset_name (Optional[str]): Name of the dataset, used to save the \
            scaling parameters in a csv. If None, the parameters are not saved.
        cols_to_normalize (list[str]): Columns to normalize among df.columns. \
            Default is COLS_TO_NORMALIZE
        add_to_existing_save (bool): Whether to add the scaling parameters to an \
            existing file if saving the scaling parameters (if dataset_name is not None). \
            Default is False, which means we will erase existing files for this dataset.

    Returns:
        tuple[pd.Series, pd.Series]: training data mean, training data standard deviation

    Example:
    >>> train_mean, train_std = get_train_mean_and_std(df_train)
    >>> df_val_eng = feature_engineering(df_val, train_mean, train_std)
    """
    # Check that the data is in the correct format
    if not bypass_validation:
        DataSchema.validate(df_train)

    train_mean = df_train[cols_to_normalize].mean()
    train_std = df_train[cols_to_normalize].std()

    if dataset_name:
        filename = f"{dataset_name}_mean"
        save_series_to_csv(train_mean, filename, add_to_existing_save)
        filename = f"{dataset_name}_std"
        save_series_to_csv(train_std, filename, add_to_existing_save)

    return train_mean, train_std


def retrieve_train_mean_and_std(dataset_name: TypeDatasetName):
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
    df_train: pd.DataFrame,
    dataset_name: Optional[TypeDatasetName] = None,
    bypass_validation: bool = False,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
    add_to_existing_save: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Get the min and max of the cols_to_normalize columns of the training data,
    used to normalize the rest of the data.

    Args:
        df_train (pd.DataFrame): Train dataframe, should be in the format of the DataSchema.
        bypass_validation (bool): Whether to validate the input DataFrame type. \
            Set it to False unless you know what you are doing. Default is False.
        dataset_name (Optional[str]): Name of the dataset, used to save the \
            scaling parameters in a csv. If None, the parameters are not saved.
        cols_to_normalize (list[str]): Columns to normalize among df.columns. \
            Default is COLS_TO_NORMALIZE
        add_to_existing_save (bool): Whether to add the scaling parameters to an \
            existing file if saving the scaling parameters (if dataset_name is not None). \
            Default is False, which means we will erase existing files for this dataset.

    Returns:
        tuple[pd.Series, pd.Series]: training data min, training data max
    """
    if not bypass_validation:
        DataSchema.validate(df_train)

    train_min = df_train[cols_to_normalize].min()
    train_max = df_train[cols_to_normalize].max()

    # save the data if we have the dataset name
    if dataset_name:
        filename = f"{dataset_name}_min"
        save_series_to_csv(train_min, filename, add_to_existing_save)
        filename = f"{dataset_name}_max"
        save_series_to_csv(train_max, filename, add_to_existing_save)

    return train_min, train_max


def retrieve_train_min_and_max(dataset_name: TypeDatasetName):
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


def get_rolling_scaling_column(
    df: pd.DataFrame,
    scaling_mode: Literal["rolling_standardize", "rolling_normalize"],
    lookahead_15min_steps: int,
    lookback_15min_steps: int = TIMESTEPS_ROLLING_WINDOW_FOR_SCALING,
    bypass_validation: bool = False,
    dataset_name: Optional[TypeDatasetName] = None,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
    add_to_existing_save: bool = False,
) -> pd.DataFrame:
    """Creates a dataframe with the rolling mean and standard deviation of the cols_to_normalize
    column, for a given lookback and lookahead.

    Args:
        df (pd.DataFrame): Dataframe of type DataSchema
        scaling_mode (Literal["rolling_standardize", "rolling_normalize"]): \
            The rolling scaling mode to apply
        lookahead_15min_steps (int): Prediction horizon in 15min steps \
            (e.g. 96 for 24h)
        lookback_15min_steps (int): Lookback period on which to do the rolling \
            average. Recommended to be at least 30 days. Default is 96*30
        bypass_validation (bool): Whether to validate the input DataFrame type. \
            Set it to False unless you know what you are doing. Default is False.
        dataset_name (Optional[str]): Name of the dataset, used to save the \
            scaling parameters in a csv. If None, the parameters are not saved.
        cols_to_normalize (list[str]): Columns to normalize among df.columns. \
            Default is COLS_TO_NORMALIZE
        add_to_existing_save (bool): Whether to add the scaling parameters to an \
            existing file if saving the scaling parameters (if dataset_name is not None). \
            Default is False, which means we will erase existing files for this dataset.

    Returns:
        pd.DataFrame: A DataFrame with 2*cols_to_normalize columns, \
            one for the mean/min and one for the standard deviation/max \
            for each of the cols_to_normalize
    """
    if not bypass_validation:
        DataSchema.validate(df)

    freq = get_data_frequency(df)
    date_column_name = get_date_column_name(df)

    if freq == "5min":
        freq_factor = 3
    elif freq == "15min":
        freq_factor = 1
    else:
        raise ValueError(f"Unsupported frequency: {freq}, please update the code")

    lookback_min_steps_min = lookback_15min_steps * 15 * freq_factor
    rolling_power = (
        df.set_index(date_column_name)
        .asfreq(freq)[cols_to_normalize]
        .rolling(
            pd.Timedelta(minutes=lookback_min_steps_min),
            min_periods=int(lookback_15min_steps * freq_factor * 0.7),
            # closed="left",
        )
    )

    df_scaling_columns = pd.DataFrame(index=df[date_column_name])
    for column_name in cols_to_normalize:
        if scaling_mode == "rolling_standardize":
            df_scaling_columns[("mean_power_for_diff", column_name)] = (
                rolling_power[column_name]
                .mean()
                .shift(lookahead_15min_steps * freq_factor, freq=freq)
            )
            df_scaling_columns[("std_power_for_diff", column_name)] = (
                rolling_power[column_name]
                .std()
                .shift(lookahead_15min_steps * freq_factor, freq=freq)
            )
            # set to 1 the std when it is 0, otherwise we will have an issue in
            # the scaling
            # This is something done in sklearn: https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
            df_scaling_columns.loc[
                df_scaling_columns[("std_power_for_diff", column_name)] == 0,
                [("std_power_for_diff", column_name)],
            ] = 1

        elif scaling_mode == "rolling_normalize":
            df_scaling_columns[("min_power_for_diff", column_name)] = (
                rolling_power[column_name]
                .min()
                .shift(lookahead_15min_steps * freq_factor, freq=freq)
            )
            df_scaling_columns[("max_power_for_diff", column_name)] = (
                rolling_power[column_name]
                .max()
                .shift(lookahead_15min_steps * freq_factor, freq=freq)
            )
            # fix the issue when the max is equal to the min (then we will have a division by 0 during the scaling)
            df_scaling_columns.loc[
                df_scaling_columns[("max_power_for_diff", column_name)]
                == df_scaling_columns[("min_power_for_diff", column_name)],
                [("max_power_for_diff", column_name)],
            ] = (
                df_scaling_columns[("max_power_for_diff", column_name)] + 1
            )
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling_mode}")

    df_scaling_columns = df_scaling_columns[
        df_scaling_columns.index == df_scaling_columns.index.date
    ].reindex(df_scaling_columns.index, method="ffill")

    if dataset_name:
        if add_to_existing_save:
            # TODO
            raise ValueError(
                "add_to_existing_save is not yet supported for rolling scaling"
            )
        filename = f"{dataset_name}_rolling_{scaling_mode}_params"
        with open(NORM_PARAMETERS_PATH / f"{filename}.csv", "w") as f:
            df_scaling_columns.to_csv(f)
    return df_scaling_columns


def apply_rolling_scaling(
    df: pd.DataFrame,
    scaling_parameters: pd.DataFrame,
    scaling_mode: Literal["rolling_standardize", "rolling_normalize"],
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
) -> pd.DataFrame:
    """Applies rolling scaling (normalization or standardization) to the cols_to_normalize \
        column of the DataFrame.

    Args:
        df (pd.DataFrame): a DataFrame with a "power" and "date" or "startChargeTime" column
        scaling_parameters (pd.Series): scaling parameters for the rolling \
            scaling. Those are returned by the function get_rolling_scaling_column
        scaling_mode (Literal["rolling_standardize", "rolling_normalize"]): \
            The scaling mode to apply
        cols_to_normalize (list[str]): Columns to normalize among df.columns. \
            Default is COLS_TO_NORMALIZE

    Returns:
        pd.DataFrame: df with the cols_to_normalize columns normalized
    """
    date_column_name = get_date_column_name(df)

    scaling_column_names = []
    for column_name in cols_to_normalize:
        if scaling_mode == "rolling_standardize":
            scaling_column_names += [
                ("mean_power_for_diff", column_name),
                ("std_power_for_diff", column_name),
            ]
        elif scaling_mode == "rolling_normalize":
            scaling_column_names += [
                ("min_power_for_diff", column_name),
                ("max_power_for_diff", column_name),
            ]
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling_mode}")

    df = df.copy()
    df = df.merge(
        scaling_parameters[scaling_column_names],
        how="left",
        left_on=date_column_name,
        right_index=True,
    ).dropna(subset=scaling_column_names)

    for column_name in cols_to_normalize:
        if scaling_mode == "rolling_standardize":
            df[column_name] = (
                df[column_name] - df[("mean_power_for_diff", column_name)]
            ) / df[("std_power_for_diff", column_name)]
        elif scaling_mode == "rolling_normalize":
            df[column_name] = (
                df[column_name] - df[("min_power_for_diff", column_name)]
            ) / (
                df[("max_power_for_diff", column_name)]
                - df[("min_power_for_diff", column_name)]
            )

    df = df.drop(columns=scaling_column_names)
    return df


def reverse_rolling_scaling(
    df: pd.DataFrame,
    column_for_difference: pd.DataFrame,
    scaling_mode: Literal["rolling_standardize", "rolling_normalize"],
) -> pd.DataFrame:
    """Reverses the rolling standardization applied to the "power" column of the DataFrame.

    Args:
        df (pd.DataFrame): a DataFrame with a "power" and "date" column
        column_for_difference (pd.Series): _description_
        scaling_mode (Literal["rolling_standardize", "rolling_normalize"]): \
            The scaling mode to apply

    Returns:
        pd.DataFrame: _description_
    """
    for column_name in COLS_TO_NORMALIZE:
        if scaling_mode == "rolling_standardize":
            scaling_column_names = [
                ("mean_power_for_diff", column_name),
                ("std_power_for_diff", column_name),
            ]
        elif scaling_mode == "rolling_normalize":
            scaling_column_names = [
                ("min_power_for_diff", column_name),
                ("max_power_for_diff", column_name),
            ]

    df = df.copy()
    df = df.merge(
        column_for_difference,
        how="left",
        left_on="date",
        right_index=True,
    ).dropna(subset=scaling_column_names)

    for column_name in COLS_TO_NORMALIZE:
        if scaling_mode == "rolling_standardize":
            df[column_name] = (
                df[column_name] * df[("std_power_for_diff", column_name)]
            ) + df[("mean_power_for_diff", column_name)]
        elif scaling_mode == "rolling_normalize":
            df[column_name] = (
                df[column_name]
                * (
                    df[("max_power_for_diff", column_name)]
                    - df[("min_power_for_diff", column_name)]
                )
            ) + df[("min_power_for_diff", column_name)]

    df = df.drop(columns=scaling_column_names)
    return df
