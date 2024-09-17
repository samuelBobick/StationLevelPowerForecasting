import pandas as pd

from .input_data_type import DataSchema, FeaturedEngineeredSchema


def feature_engineering(data_input: pd.DataFrame) -> pd.DataFrame:
    """Processes the data so that it can be used in the models.
    Here are the different steps that are done:
        - Convert the date from a Timestamp to an int
        - Add the 4 hour time window (0 if the hour is between 0 and 4, 1 if the hour is between 4 and 8, etc.)

    Args:
        data (pd.DataFrame): Data to be transformed and fed into the models (e.g. for the window generator).

    Returns:
        pd.DataFrame: Data ready to be used in the models and tensorflow.
    """
    data = data_input.copy()
    # Check that the data is in the correct format
    DataSchema.validate(data)

    # Add the 4 hour time window
    data["time_window"] = data["date"].dt.hour // 4

    # Convert the date to an int
    # we choose int instead of having a float, because the date magnitude (10**9)
    # is too large for a float32 (the tensorflow default dtype) to be precise
    data["date"] = data["date"].astype("int64") // 10**9

    FeaturedEngineeredSchema.validate(data)
    return data


def reverse_feature_engineering(data_input: pd.DataFrame) -> pd.DataFrame:
    """Reverses the feature engineering done in feature_engineer.

    Args:
        data (pd.DataFrame): Data to be transformed back to its original form.

    Returns:
        pd.DataFrame: Data in its original form.
    """
    # convert table to DataFrame if it is in the tensor format
    if not isinstance(data_input, pd.DataFrame):
        data = pd.DataFrame(data_input)
        data.rename(columns={0: "date", 1: "power", 2: "workday"}, inplace=True)
        data = data.astype({"date": "int64", "power": "float64", "workday": "int64"})

    else:
        data = data_input.copy()

    # Check that the data is in the correct format
    FeaturedEngineeredSchema.validate(data)

    # Convert the date back to a Timestamp
    data["date"] = pd.to_datetime(data["date"], unit="s")
    # Round dates to closest minute
    data["date"] = data["date"].dt.round("5min")

    DataSchema.validate(data)
    return data
