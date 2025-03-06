from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as UScalendar
from slrp_ev_ts_forecasting.default_parameters import TypeScalingMode

from slrp_ev_data.utils.data_utils import (
    add_missing_timesteps,
    convert_data_freq_to_minutes,
    convert_date_from_datetime_to_int,
    convert_date_from_int_to_datetime,
    get_data_frequency,
    get_workday_column_names,
)
from slrp_ev_data.utils.scaling_main import apply_scaling, reverse_scaling
from slrp_ev_data.utils.scaling_utils import (
    COLS_TO_NORMALIZE,
)
from slrp_ev_data.utils.USAcademicHolidayCalendar import USAcademicHolidayCalendar

from .utils.input_data_type import DataSchema, FeaturedEngineeredSchema


def feature_engineering(
    data_input: pd.DataFrame,
    add_nans_for_missing_data: bool,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
    cols_normalization_to_skip: list[str] = [],
    holiday_calendar: Literal["USFederal", "USAcademic"] = "USAcademic",
    lookahead: int = 96,
) -> pd.DataFrame:
    """Processes the data so that it can be used in the models.
    Here are the different steps that are done:
        - Convert the date from a Timestamp to an int
        - Add the 4 hour time window (0 if the hour is between 0 and 4, 1 if the hour is between 4 and 8, etc.)
        - Scale the data (Standardize, normalize or apply rolling standardization)
        - Add time of day and time of year as sin and cos features
        - Remove the data from before March 2021 because it doesn't make sense. Plotting the monthly
            peak distribution shows that 2020 and the first two months of 2021 are outliers with no real usage.

    Args:
        data (pd.DataFrame): Data to be transformed and fed into the models (e.g. for the window generator).
        add_nans_for_missing_data (bool): If True, adds the missing timesteps, and set their features to NaNs. \
            If False, there won't be any NaNs in the data, but some timesteps might be missing.
        standardize_parameters (tuple[pd.Series, pd.Series]): Mean and standard deviation \
            of the training data, used to standardize the data. \
            Should come from get_train_mean_and_std.
        normalize_parameters (tuple[pd.Series, pd.Series]): Min and max of the training data, \
            used to normalize the data. Should come from get_train_min_and_max.
    Returns:
        pd.DataFrame: Data ready to be used in the models and tensorflow.
    """
    # TODO: if we normalize, put workday between 0 and 1 and time features between 0 and 1
    # if we standardize, put workday between -1 and 1 and time features between -1 and 1

    data = data_input.copy()
    # Check that the data is in the correct format
    DataSchema.validate(data)

    cols_to_normalize = [
        col for col in COLS_TO_NORMALIZE if col not in cols_normalization_to_skip
    ]

    data = apply_scaling(data, scaling_mode, scaling_parameters, cols_to_normalize)

    if add_nans_for_missing_data:
        data = add_missing_timesteps(data)

    # We are going to add timesteps to the data, so that we can add the future workday
    # status for the last bit of the data
    data_freq = get_data_frequency(data)
    initial_data_end_date = data["date"].max()
    list_workday_column_names = get_workday_column_names(lookahead)
    number_of_rows_to_add = (len(list_workday_column_names) - 1) * 96 + 1
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                data={
                    "date": pd.date_range(
                        start=initial_data_end_date
                        + pd.Timedelta(minutes=convert_data_freq_to_minutes(data_freq)),
                        periods=number_of_rows_to_add,
                        freq=data_freq,
                    )
                }
            ),
        ],
        ignore_index=True,
    )

    data["workday_0"] = (data["date"].dt.dayofweek < 5).astype(int)
    # set public holidays to 0 (non workday)
    if holiday_calendar == "USAcademic":
        cal = USAcademicHolidayCalendar()
    elif holiday_calendar == "USFederal":
        cal = UScalendar()
    else:
        raise ValueError(
            f"holiday_calendar should be either 'USFederal' or 'USAcademic'. {holiday_calendar} was provided."
        )
    us_holidays = cal.holidays(start=data["date"].min(), end=data["date"].max())
    data.loc[pd.to_datetime(data["date"].dt.date).isin(us_holidays), "workday_0"] = 0
    # shift the workday column by 1, so that each timesteps knows the workday status of the next timestep
    # useful for regression models, when we make a prediction starting 00:00, knowing everything
    # up to 23:45 of the previous day
    workday_next_timestep = data["workday_0"].shift(-1).ffill().astype("Int32")
    # we are going to have 1 workday column for all the days we have to predict + 1
    for i, workday_column_name in enumerate(list_workday_column_names):
        data[workday_column_name] = (
            workday_next_timestep.shift(-i * 96).ffill().astype("Int32")
        )
    # drop the few timesteps that we added at the beginning
    data = data.iloc[:-number_of_rows_to_add].copy()

    # Add the 4 hour time window
    data["time_window"] = data["date"].dt.hour // 4
    data["time_window"] = data["time_window"].shift(-1).ffill().astype("Int32")

    engineer_time_features(data)

    data = data.astype({"power": "float32", "number_of_evses_available": "float32"})
    FeaturedEngineeredSchema.validate(data)

    return data


def engineer_time_features(data: pd.DataFrame, verbose_plot: bool = False):
    """Modifies data inplace by adding time features."""
    # Convert the date to an int
    # we choose int instead of having a float, because the date magnitude (10**9)
    # is too large for a float32 (the tensorflow default dtype) to be precise
    dates_ts = data["date"].copy()
    data["date"] = convert_date_from_datetime_to_int(data["date"])

    s_in_day = 24 * 60 * 60  # number of seconds in a day
    s_in_week = 7 * s_in_day
    s_in_year = (365.2425) * s_in_day
    data["Day sin"] = np.sin(data["date"] * (2 * np.pi / s_in_day), dtype=np.float32)
    data["Day cos"] = np.cos(data["date"] * (2 * np.pi / s_in_day), dtype=np.float32)
    data["Week sin"] = np.sin(data["date"] * (2 * np.pi / s_in_week), dtype=np.float32)
    data["Week cos"] = np.cos(data["date"] * (2 * np.pi / s_in_week), dtype=np.float32)
    data["Year sin"] = np.sin(data["date"] * (2 * np.pi / s_in_year), dtype=np.float32)
    data["Year cos"] = np.cos(data["date"] * (2 * np.pi / s_in_year), dtype=np.float32)

    if verbose_plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))

        # Hour of the day encoding
        sp = axs[0].scatter(
            data["Day sin"], data["Day cos"], c=dates_ts.dt.hour, cmap="viridis"
        )
        axs[0].set(
            xlabel="sin(hour)", ylabel="cos(hour)", title="Hour of the Day Encoding"
        )
        fig.colorbar(sp, ax=axs[0])

        # Day of the week encoding
        sp = axs[1].scatter(
            data["Week sin"], data["Week cos"], c=dates_ts.dt.dayofweek, cmap="viridis"
        )
        axs[1].set(
            xlabel="sin(day of week)",
            ylabel="cos(day of week)",
            title="Day of the Week Encoding",
        )
        fig.colorbar(sp, ax=axs[1])

        # Week of the year encoding
        sp = axs[2].scatter(
            data["Year sin"],
            data["Year cos"],
            c=dates_ts.dt.isocalendar().week,
            cmap="viridis",
        )
        axs[2].set(
            xlabel="sin(week of year)",
            ylabel="cos(week of year)",
            title="Week of the Year Encoding",
        )
        fig.colorbar(sp, ax=axs[2])

        plt.tight_layout()
        plt.show()


def reverse_feature_engineering(
    data_input: pd.DataFrame,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
    bypass_output_validation: bool = False,
    lookahead: int = 96,
) -> pd.DataFrame:
    """Reverses the feature engineering done in feature_engineer.

    Args:
        data (pd.DataFrame): Data to be transformed back to its original form.

    Returns:
        pd.DataFrame: Data in its original form.
    """
    list_workday_column_names = get_workday_column_names(lookahead)

    # convert table to DataFrame if it is in the tensor format
    if not isinstance(data_input, pd.DataFrame):
        data = pd.DataFrame(data_input)
        dict_rename = {
            0: "date",
            1: "power",
        }
        for i, workday_column_name in enumerate(list_workday_column_names):
            dict_rename = dict_rename | {2 + i: workday_column_name}
        dict_rename = dict_rename | {
            i + 3: "time_window",
            i + 4: "Day sin",
            i + 5: "Day cos",
            i + 6: "Week sin",
            i + 7: "Week cos",
            i + 8: "Year sin",
            i + 9: "Year cos",
        }
        data.rename(
            columns=dict_rename,
            inplace=True,
        )

    else:
        data = data_input.copy()
    dict_types = {
        "date": "int64",
        "power": "float32",
        "time_window": "int32",
        "Day sin": "float32",
        "Day cos": "float32",
        "Week sin": "float32",
        "Week cos": "float32",
        "Year sin": "float32",
        "Year cos": "float32",
    }
    for workday_column_name in list_workday_column_names:
        dict_types[workday_column_name] = "int32"
    data = data.astype(dict_types)

    # Check that the data is in the correct format
    FeaturedEngineeredSchema.validate(data)
    data = data.drop(
        ["Day sin", "Day cos", "Week sin", "Week cos", "Year sin", "Year cos"],
        axis=1,
        errors="ignore",
    )

    # Convert the date back to a Timestamp
    data["date"] = convert_date_from_int_to_datetime(data["date"])

    data = reverse_scaling(data, scaling_mode, scaling_parameters)

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
