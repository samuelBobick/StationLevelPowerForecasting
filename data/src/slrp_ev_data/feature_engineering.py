import numpy as np
import pandas as pd

from .input_data_type import DataSchema, FeaturedEngineeredSchema

COLS_TO_NORMALIZE = ["power"]


def convert_date_from_int_to_datetime(date_column: pd.Series) -> pd.Series:
    # Convert the date back to a Timestamp
    date_column = pd.to_datetime(date_column, unit="s")
    # Round dates to closest minute
    date_column = date_column.dt.round("5min")
    return date_column


def convert_date_from_datetime_to_int(date_column: pd.Series) -> pd.Series:
    # Convert the date back to a Timestamp
    date_column = date_column.astype("int64") // 10**9
    return date_column


def feature_engineering(
    data_input: pd.DataFrame,
    standardize_parameters: tuple[pd.Series, pd.Series] | None = None,
    normalize_parameters: tuple[pd.Series, pd.Series] | None = None,
) -> pd.DataFrame:
    """Processes the data so that it can be used in the models.
    Here are the different steps that are done:
        - Convert the date from a Timestamp to an int
        - Add the 4 hour time window (0 if the hour is between 0 and 4, 1 if the hour is between 4 and 8, etc.)
        - Standardize (or normalize) the data
        - Add time of day and time of year as sin and cos features
        - Remove the data from before March 2021 because it doesn't make sense. Plotting the monthly
            peak distribution shows that 2020 and the first two months of 2021 are outliers with no real usage.

    Args:
        data (pd.DataFrame): Data to be transformed and fed into the models (e.g. for the window generator).
        train_mean (pd.Series): Mean of the training data, used to normalize the data.
            Should come from get_train_mean_and_std.
        train_std (pd.Series): Standard deviation of the training data, used to normalize the data.
            Should come from get_train_mean_and_std.
        standardize_parameters (tuple[pd.Series, pd.Series]): Mean and standard deviation of the training data, used to standardize the data.
            Should come from get_train_mean_and_std.
        normalize_parameters (tuple[pd.Series, pd.Series]): Min and max of the training data, used to normalize the data.
            Should come from get_train_min_and_max.
    Returns:
        pd.DataFrame: Data ready to be used in the models and tensorflow.
    """
    data = data_input.copy()
    # Check that the data is in the correct format
    DataSchema.validate(data)

    data = data.loc[data["date"] >= "2021-03-01"]

    if normalize_parameters and standardize_parameters:
        raise ValueError(
            "Cannot normalize and standardize at the same time. Please pass normalize_parameters OR standardize_parameters"
        )
    if standardize_parameters:
        train_mean, train_std = standardize_parameters
        data[COLS_TO_NORMALIZE] = (data[COLS_TO_NORMALIZE] - train_mean) / train_std
    if normalize_parameters:
        train_min, train_max = normalize_parameters
        data[COLS_TO_NORMALIZE] = (data[COLS_TO_NORMALIZE] - train_min) / (
            train_max - train_min
        )

    # Add the 4 hour time window
    data["time_window"] = data["date"].dt.hour // 4

    # Convert the date to an int
    # we choose int instead of having a float, because the date magnitude (10**9)
    # is too large for a float32 (the tensorflow default dtype) to be precise
    data["date"] = convert_date_from_datetime_to_int(data["date"])

    s_in_day = 24 * 60 * 60  # number of seconds in a day
    s_in_week = 7 * s_in_day
    s_in_year = (365.2425) * s_in_day
    data["Day sin"] = np.sin(data["date"] * (2 * np.pi / s_in_day))
    data["Day cos"] = np.cos(data["date"] * (2 * np.pi / s_in_day))
    data["Week sin"] = np.sin(data["date"] * (2 * np.pi / s_in_week))
    data["Week cos"] = np.cos(data["date"] * (2 * np.pi / s_in_week))
    data["Year sin"] = np.sin(data["date"] * (2 * np.pi / s_in_year))
    data["Year cos"] = np.cos(data["date"] * (2 * np.pi / s_in_year))

    FeaturedEngineeredSchema.validate(data)
    return data


def reverse_feature_engineering(
    data_input: pd.DataFrame,
    standardize_parameters: tuple[pd.Series, pd.Series] | None = None,
    normalize_parameters: tuple[pd.Series, pd.Series] | None = None,
    bypass_output_validation: bool = False,
) -> pd.DataFrame:
    """Reverses the feature engineering done in feature_engineer.

    Args:
        data (pd.DataFrame): Data to be transformed back to its original form.

    Returns:
        pd.DataFrame: Data in its original form.
    """
    # convert table to DataFrame if it is in the tensor format
    if not isinstance(data_input, pd.DataFrame):
        data = pd.DataFrame(data_input)
        data.rename(
            columns={
                0: "date",
                1: "power",
                2: "workday",
                3: "time_window",
                4: "Day sin",
                5: "Day cos",
                6: "Week sin",
                7: "Week cos",
                8: "Year sin",
                9: "Year cos",
            },
            inplace=True,
        )

    else:
        data = data_input.copy()
    data = data.astype(
        {
            "date": "int64",
            "power": "float64",
            "workday": "int64",
            "time_window": "int32",
            "Day sin": "float64",
            "Day cos": "float64",
            "Week sin": "float64",
            "Week cos": "float64",
            "Year sin": "float64",
            "Year cos": "float64",
        }
    )

    # Check that the data is in the correct format
    FeaturedEngineeredSchema.validate(data)
    data = data.drop(
        ["Day sin", "Day cos", "Week sin", "Week cos", "Year sin", "Year cos"],
        axis=1,
        errors="ignore",
    )

    # Convert the date back to a Timestamp
    data["date"] = convert_date_from_int_to_datetime(data["date"])

    # Reverse the standardization
    if normalize_parameters and standardize_parameters:
        raise ValueError(
            "Cannot normalize and standardize at the same time. Please pass normalize_parameters OR standardize_parameters"
        )
    if standardize_parameters:
        train_mean, train_std = standardize_parameters
        data[COLS_TO_NORMALIZE] = data[COLS_TO_NORMALIZE] * train_std + train_mean
    if normalize_parameters:
        train_min, train_max = normalize_parameters
        data[COLS_TO_NORMALIZE] = (
            data[COLS_TO_NORMALIZE] * (train_max - train_min) + train_min
        )
    if not bypass_output_validation:
        DataSchema.validate(data)
    return data


def one_hot_encoding(
    data_input: pd.DataFrame, cols_to_encode: list[str]
) -> pd.DataFrame:
    """One hot encodes the columns specified in cols_to_encode.

    Args:
        data_input (pd.DataFrame): Data to be one hot encoded.
        cols_to_encode (list[str]): List of columns to be one hot encoded. Boolean columns are ignored.

    Returns:
       pd.DataFrame: Data with the columns specified in cols_to_encode one hot encoded.
    """

    one_hot_length = data_input[cols_to_encode].apply(lambda x: len(x.unique()))
    # we do not want to encode boolean arrays (with only two values)
    if one_hot_length[one_hot_length <= 2].shape[0] > 0:
        cols_to_ignore = one_hot_length[one_hot_length <= 2].index.tolist()
        print(
            f"WARNING: Column(s) {cols_to_ignore} are boolean (only two possible values) and will not be one-hot encoded"
        )
        cols_to_encode = [col for col in cols_to_encode if col not in cols_to_ignore]

    data_encoded = pd.get_dummies(data_input, columns=cols_to_encode, dtype=int)
    return data_encoded


def get_train_mean_and_std(df_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Get the mean and standard deviation of the training data, used to normalize the rest of the data.

    Args:
        df_train (pd.DataFrame): Train dataframe, should be in the format of the DataSchema.

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
    return train_mean, train_std


def get_train_min_and_max(df_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    DataSchema.validate(df_train)

    train_min = df_train[COLS_TO_NORMALIZE].min()
    train_max = df_train[COLS_TO_NORMALIZE].max()

    return train_min, train_max
