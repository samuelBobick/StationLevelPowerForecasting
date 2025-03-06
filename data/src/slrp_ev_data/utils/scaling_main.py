import numpy as np
import pandas as pd
from slrp_ev_data.utils.scaling_utils import (
    COLS_TO_NORMALIZE,
    apply_rolling_scaling,
    get_rolling_scaling_column,
    get_train_mean_and_std,
    get_train_min_and_max,
    normalize_data,
    retrieve_train_mean_and_std,
    retrieve_train_min_and_max,
    reverse_normalize_data,
    reverse_rolling_scaling,
    reverse_standardize_data,
    standardize_data,
)
from slrp_ev_ts_forecasting.default_parameters import TypeDatasetName, TypeScalingMode


def get_scaling_parameters(
    train: pd.DataFrame | None,
    data: pd.DataFrame | None,
    scaling_mode: TypeScalingMode,
    dataset_name: TypeDatasetName,
    lookahead_15min_steps: int,
    retrieve_from_saved: bool = False,
    bypass_validation: bool = False,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
    add_to_existing_save: bool = False,
) -> tuple | pd.DataFrame:
    """_summary_

    Args:
        train (pd.DataFrame | None): Dataframe of type DataSchema with ONLY \
            the training data
        data (pd.DataFrame | None): Full dataframe of type DataSchema, \
            including the training data
        scaling_mode (Literal["rolling_standardize", "rolling_normalize"]): \
            The rolling scaling mode to apply
        dataset_name (Optional[str]): Name of the dataset, used to save the \
            scaling parameters in a csv. If None, the parameters are not saved.
        lookahead_15min_steps (int): Prediction horizon in 15min steps \
            (e.g. 96 for 24h)
        retrieve_from_saved (bool): Whether to retrieve the scaling parameters \
            from a saved file or to compute them again from the given train or data. \
            Default is False.
        bypass_validation (bool): Whether to validate the input DataFrame type. \
            Set it to False unless you know what you are doing. Default is False.
        cols_to_normalize (list[str]): Columns to normalize among df.columns. \
            Default is COLS_TO_NORMALIZE
        add_to_existing_save (bool): Whether to add the scaling parameters to an \
            existing file if saving the scaling parameters (if dataset_name is not None). \
            Default is False, which means we will erase existing files for this dataset.

    Returns:
        tuple | pd.DataFrame: the scaling parameters for each of the cols_to_normalize columns.
        (e.g. (mean, std) for standardize, (min, max) for normalize, etc...)
    """
    has_all_cols_to_normalize = True

    if scaling_mode == "normalize":
        if retrieve_from_saved:
            scaling_parameters = retrieve_train_min_and_max(dataset_name=dataset_name)
            has_all_cols_to_normalize = np.all(
                [col in scaling_parameters[0].index for col in cols_to_normalize]
            )
        if not retrieve_from_saved or not has_all_cols_to_normalize:
            if train is None:
                raise ValueError(
                    "train dataframe must be provided when data_scaling_mode is 'normalize'"
                )
            scaling_parameters = get_train_min_and_max(
                train,
                dataset_name=dataset_name,
                bypass_validation=bypass_validation,
                cols_to_normalize=cols_to_normalize,
                add_to_existing_save=add_to_existing_save
                or not has_all_cols_to_normalize,
            )

    elif scaling_mode == "standardize":
        if retrieve_from_saved:
            scaling_parameters = retrieve_train_mean_and_std(dataset_name=dataset_name)
            has_all_cols_to_normalize = np.all(
                [col in scaling_parameters[0].index for col in cols_to_normalize]
            )
        if not retrieve_from_saved or not has_all_cols_to_normalize:
            if train is None:
                raise ValueError(
                    "train dataframe must be provided when data_scaling_mode is 'standardize'"
                )
            scaling_parameters = get_train_mean_and_std(
                train,
                dataset_name=dataset_name,
                bypass_validation=bypass_validation,
                cols_to_normalize=cols_to_normalize,
                add_to_existing_save=add_to_existing_save
                or not has_all_cols_to_normalize,
            )

    elif scaling_mode in ["rolling_standardize", "rolling_normalize"]:
        if train is None or data is None:
            raise ValueError(
                "train and data dataframes must be provided when data_scaling_mode is 'rolling_standardize'"
            )
        if retrieve_from_saved:
            raise ValueError("retrieve_from_saved is not supported for rolling scaling")

        scaling_parameters = get_rolling_scaling_column(
            data,
            scaling_mode=scaling_mode,
            lookahead_15min_steps=lookahead_15min_steps,
            dataset_name=dataset_name,
            bypass_validation=bypass_validation,
            cols_to_normalize=cols_to_normalize,
            add_to_existing_save=add_to_existing_save,
        )

    else:
        raise ValueError(
            f"Data scaling mode {scaling_mode} is not defined. Please refer to "
            "TypeDataScalingMode for supported data scaling modes."
        )
    return scaling_parameters


def apply_scaling(
    data: pd.DataFrame,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
) -> pd.DataFrame:
    data = data.copy()
    if scaling_mode:
        if scaling_parameters is None:
            raise ValueError(
                "If scaling_mode is not None, scaling_parameters must be provided"
            )
        if scaling_mode == "standardize":
            if isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError("scaling_parameters should be a tuple of 2 pd.Series")
            train_mean, train_std = scaling_parameters
            data[cols_to_normalize] = standardize_data(
                data[cols_to_normalize],
                train_mean[cols_to_normalize],
                train_std[cols_to_normalize],
            )
        elif scaling_mode == "normalize":
            if isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError("scaling_parameters should be a tuple of 2 pd.Series")
            train_min, train_max = scaling_parameters
            data[cols_to_normalize] = normalize_data(
                data[cols_to_normalize],
                train_min[cols_to_normalize],
                train_max[cols_to_normalize],
            )
        elif scaling_mode in ["rolling_standardize", "rolling_normalize"]:
            if not isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError(
                    "scaling_parameters should be a pd.DataFrame, and 2 pd.Series"
                )
            data = apply_rolling_scaling(
                data,
                scaling_parameters,
                scaling_mode=scaling_mode,
                cols_to_normalize=cols_to_normalize,
            )
        else:
            raise ValueError(
                f"scaling_mode should be one of 'standardize', 'normalize', 'rolling_standardize', 'rolling_normalize'. {scaling_mode} was provided."
            )

    else:
        if scaling_parameters is not None:
            raise ValueError(
                "If scaling_parameters were provided but scaling_mode is None, please "
                "provide a scaling_mode or don't provide scaling_parameters"
            )
    return data


def reverse_scaling(
    data: pd.DataFrame,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
    cols_to_normalize: list[str] = COLS_TO_NORMALIZE,
):
    data = data.copy()
    # Reverse the standardization

    if scaling_mode:
        if scaling_parameters is None:
            raise ValueError(
                "If scaling_mode is not None, scaling_parameters must be provided"
            )
        if scaling_mode == "standardize":
            if isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError("scaling_parameters should be a tuple of 2 pd.Series")
            train_mean, train_std = scaling_parameters
            data[cols_to_normalize] = reverse_standardize_data(
                data[cols_to_normalize],
                train_mean[cols_to_normalize],
                train_std[cols_to_normalize],
            )
        elif scaling_mode == "normalize":
            if isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError("scaling_parameters should be a tuple of 2 pd.Series")
            train_min, train_max = scaling_parameters
            data[cols_to_normalize] = reverse_normalize_data(
                data[cols_to_normalize],
                train_min[cols_to_normalize],
                train_max[cols_to_normalize],
            )
        elif scaling_mode in ["rolling_standardize", "rolling_normalize"]:
            if not isinstance(scaling_parameters, pd.DataFrame):
                raise ValueError(
                    "scaling_parameters should be a pd.DataFrame, and 2 pd.Series"
                )

            data = reverse_rolling_scaling(
                data, scaling_parameters, scaling_mode=scaling_mode
            )
        else:
            raise ValueError(
                f"scaling_mode should be one of 'standardize', 'normalize', 'rolling_standardize'"
                f", 'rolling_normalize'. {scaling_mode} was provided."
            )
    else:
        if scaling_parameters is not None:
            raise ValueError(
                "If scaling_parameters were provided but scaling_mode is None, please "
                "provide a scaling_mode or don't provide scaling_parameters"
            )

    return data
