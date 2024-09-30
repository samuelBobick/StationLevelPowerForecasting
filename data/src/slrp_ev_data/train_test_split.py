import pandas as pd

from .input_data_type import DataSchema


def train_test_split(
    data: pd.DataFrame,
    generate_validation: bool = False,
    fraction_in_train: float = 0.7,
    fraction_in_val: float = 0.2,
) -> (
    tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Split the data into training and testing data. The default values for the sets sizes are from here:
    https://www.tensorflow.org/tutorials/structured_data/time_series#split_the_data

    Args:
        data: The input data.
        generate_validation: whether to generate a validation set.
        fraction_in_train: the fraction of data to put in the training set (default 0.7).
        fraction_in_val: the fraction of data to put in the validation set (default 0.2).

    Returns:
        A tuple containing (training, testing) data, or (training, validation, testing) data in case generate_validation is True
    """
    # Check that the data is in the correct format
    DataSchema.validate(data)

    data_length = data.shape[0]
    split_train_index = int(data_length * fraction_in_train)
    train = data.loc[: split_train_index - 1]

    if generate_validation:
        split_val_index = int(data_length * (fraction_in_train + fraction_in_val))
        val = data.loc[split_train_index : split_val_index - 1]
        test = data.loc[split_val_index:]
        return train, val, test
    else:
        test = data.loc[split_train_index:]
        return train, test
