import pandas as pd


def get_date_column_name(df: pd.DataFrame) -> str:
    """Check if the column name of this DataFrame for the date is 'date' or 'startChargeTime'."""
    list_possible_date_columns = ["date", "startChargeTime"]
    for possible_date_column in list_possible_date_columns:
        if possible_date_column in df.columns:
            return possible_date_column

    raise KeyError(
        "This DataFrame does not have a column named 'date' or 'startChargeTime'."
    )


def get_workday_column_names(lookahead: int) -> list[str]:
    # TODO: update for different data frequency
    workday_column_names = []

    # set the number of workday columns as the
    # number of days to predict ahead, + 1 for the current day,
    # + 1 for one day after the first day ahead
    number_of_workday_columns = (lookahead // 96) + 2
    for i in range(number_of_workday_columns):
        workday_column_names.append(f"workday_{int(i*96)}")
    return workday_column_names


def get_data_frequency(
    df, _data_size_for_freq_lookup=None, _date_column_name=None
) -> str:
    """Get the data frequency from the DataFrame.
    Leave `_data_size_for_freq_lookup` and `_date_column_name` to None, it is used for recursion.
    """
    if _data_size_for_freq_lookup is None:
        _data_size_for_freq_lookup = df.shape[0]
    if _date_column_name is None:
        _date_column_name = get_date_column_name(df)

    if isinstance(df[_date_column_name].iloc[0], pd.Timestamp):
        data_freq = pd.infer_freq(
            df[_date_column_name].iloc[-_data_size_for_freq_lookup:]
        )
    else:
        data_freq = pd.infer_freq(
            pd.to_datetime(
                df[_date_column_name].iloc[-_data_size_for_freq_lookup:], unit="s"
            )
        )

    # if the data frequency is not found, it might be because we have gaps.
    # Then, we recursively call the function with a smaller
    # data size lookup
    if not data_freq:
        if _data_size_for_freq_lookup < 20:
            raise ValueError("The data frequency could not be inferred.")
        else:
            _data_size_for_freq_lookup = int(_data_size_for_freq_lookup / 2)
            return get_data_frequency(df, _data_size_for_freq_lookup, _date_column_name)
    return data_freq


def convert_data_freq_to_minutes(data_freq) -> int:
    try:
        return int(data_freq.split("min")[0])  #
    except ValueError:
        raise ValueError(
            "The data frequency is not in minutes. "
            "Please edit this function to handle other frequencies."
        )


def add_missing_timesteps(data) -> pd.DataFrame:
    # make sure that date is a datetime before calling that function
    return (
        data.set_index("date").resample(get_data_frequency(data)).mean().reset_index()
    )


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
