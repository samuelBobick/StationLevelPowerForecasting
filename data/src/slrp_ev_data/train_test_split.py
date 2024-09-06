import pandas as pd
from .input_data_type import DataSchema


def train_test_split(
    data: pd.DataFrame, fraction_in_train: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing data.

    Args:
        data: The input data.
        fraction_in_train: the fraction of data to put in the training set

    Returns:
        A tuple containing the training and testing data.
    """
    # Check that the data is in the correct format
    DataSchema.validate(data)

    data_lenght = data.shape[0]
    split_index = int(data_lenght * 0.8)

    train = data.loc[:split_index]
    test = data.loc[split_index:]

    return train, test
