import pandas as pd
from dateutil.relativedelta import FR
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


def get_date_column_name(df: pd.DataFrame) -> str:
    """Check if the column name of this DataFrame for the date is 'date' or 'startChargeTime'."""
    list_possible_date_columns = ["date", "startChargeTime"]
    for possible_date_column in list_possible_date_columns:
        if possible_date_column in df.columns:
            return possible_date_column

    raise KeyError(
        "This DataFrame does not have a column named 'date' or 'startChargeTime'."
    )


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
        if _data_size_for_freq_lookup < 100:
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


class USAcademicHolidayCalendar(AbstractHolidayCalendar):
    """
    US Academic Government Holiday Calendar based on rules specified by:
    - [UC Berkeley](https://hr.berkeley.edu/hr-network/personnel-resources/holiday-sick-vacation/holiday-schedule)
    - [UC San Diego](https://blink.ucsd.edu/HR/benefits/time-off/holidays.html)
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        Holiday(
            "Cesar Chavez Day", month=3, day=31, offset=pd.DateOffset(weekday=FR(-1))
        ),
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday(
            "Day after Thanksgiving",
            month=11,
            day=1,
            offset=pd.DateOffset(weekday=FR(4)),
        ),
        Holiday("Christmas Eve", month=12, day=24, observance=nearest_workday),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("New Year's Eve", month=12, day=31, observance=nearest_workday),
    ]
