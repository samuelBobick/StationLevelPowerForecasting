import time

import numpy as np
from slrp_ev_data import (
    read_new_slrpev_data,
    read_old_slrpev_data,
    read_ucsd_data,
    train_test_split,
)
from slrp_ev_data.feature_engineering import (
    feature_engineering,
)

from slrp_ev_ts_forecasting.compute_losses import compute_losses
from slrp_ev_ts_forecasting.default_parameters import (
    ALPHA,
    DATASET,
    DEFAULT_RESULTS_FILENAME,
    GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
    LOOKAHEAD,
    SCALING_MODE,
    TypeDataSet,
    TypeModelChoice,
)
from slrp_ev_ts_forecasting.models.dict_models import DICT_MODEL
from slrp_ev_ts_forecasting.models.ffnn import FFNN
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel
from slrp_ev_ts_forecasting.save_losses import save_losses
from slrp_ev_ts_forecasting.utils_data_processing import (
    get_scaling_parameters,
    reverse_engineer_forecast,
)
from slrp_ev_ts_forecasting.visualization import visualize_forecast


def run_one_model(
    model_choice: TypeModelChoice,
    model_parameters={},
    verbose: bool = True,
    save_results_filename: str = DEFAULT_RESULTS_FILENAME,
    dataset: TypeDataSet = DATASET,
) -> None:
    # Read the data
    print("# Starting...")
    start_time = time.time()
    if dataset == "slrp-ev_old":
        data = read_old_slrpev_data.read_old_slrpev_data()
    elif dataset == "slrp-ev_new":
        data = read_new_slrpev_data.read_new_slrpev_data()
    elif dataset == "ucsd-all_garages":
        data = read_ucsd_data.read_ucsd_data()
    else:
        raise ValueError(
            f"Dataset of type {dataset} is not defined. Please refer to "
            "TypeDataSet for supported datasets."
        )

    session_based_mode = model_parameters.get("session_based_mode", None)
    peak_prediction = model_parameters.get("peak_prediction", None)
    if (session_based_mode or peak_prediction) and dataset != "slrp-ev_new":
        raise ValueError(
            "Session based mode and peak prediction are only available for the slrp-ev_new dataset"
        )

    get_val_data_from_shuffled_train = model_parameters.get(
        "get_val_data_from_shuffled_train", GET_VAL_DATA_FROM_SHUFFLED_TRAIN
    )
    # If we want to get the validation data from the shuffled train data,
    # we need to split the data into train and test only
    if get_val_data_from_shuffled_train:
        train, test = train_test_split.train_test_split(data, generate_validation=False, fraction_in_train=0.9)  # type: ignore
        val = None
    else:
        train, val, test = train_test_split.train_test_split(data, generate_validation=True)  # type: ignore

    model_class = DICT_MODEL[model_choice]["model"]
    is_regression_model = (RegressionBaseModel in model_class.__bases__) or (
        model_class == FFNN
    )

    scaling_mode = model_parameters.get("scaling_mode", SCALING_MODE)

    scaling_parameters = get_scaling_parameters(
        train,
        data,
        scaling_mode,
        dataset,
        lookahead_15min_steps=model_parameters.get("lookahead", LOOKAHEAD),
    )

    train_eng = feature_engineering(
        train,
        is_regression_model,
        scaling_mode=scaling_mode,
        scaling_parameters=scaling_parameters,
    )
    val_eng = (
        feature_engineering(
            val,
            is_regression_model,
            scaling_mode=scaling_mode,
            scaling_parameters=scaling_parameters,
        )
        if val is not None
        else None
    )
    test_eng = feature_engineering(
        test,
        is_regression_model,
        scaling_mode=scaling_mode,
        scaling_parameters=scaling_parameters,
    )

    # Add predefined model parameters to the model_parameters dictionary
    model_parameters = model_parameters | DICT_MODEL[model_choice]["model_params"]
    print(
        f"Model choice: {model_choice}, with the following parameters for the initialization: {model_parameters } "
    )
    # Add the scaling parameters to the model_parameters dictionary.
    # Since it is automatic, and a dataFrame, we do that after printing the
    # parameters information
    model_parameters["scaling_parameters"] = scaling_parameters

    model = model_class(**model_parameters)
    model_name = getattr(model, "model_str_name", model_choice)
    print("# Fitting...")
    model.fit(train_eng, val=val_eng, **DICT_MODEL[model_choice]["fit_params"])

    print("# Making prediction(s)...")
    df_predictions = model.predict(
        test_eng,
    )

    # Reverse engineer the forecast to get the original features back
    df_reversed_predictions = reverse_engineer_forecast(
        test_eng,
        df_predictions,
        scaling_mode=scaling_mode,
        scaling_parameters=scaling_parameters,
    )
    data_length_days = df_reversed_predictions.shape[0] // 96

    y_pred = df_reversed_predictions.filter(regex="^power").values.reshape(-1)
    y_true = df_reversed_predictions.filter(regex="real_power").values.reshape(-1)
    mask_nan = ~np.isnan(y_pred)
    y_pred = y_pred[mask_nan]
    y_true = y_true[mask_nan]

    losses = compute_losses(y_pred, y_true, model_parameters.get("alpha", ALPHA))
    print(
        f"{model_choice}: ",
        *[
            f"{loss_type.upper()}: {loss_value:.1f};"
            for loss_type, loss_value in losses.items()
        ],
        f"for around {data_length_days} days of predictions",
    )

    if verbose:
        visualize_forecast(
            test, df_reversed_predictions, data_length_days, model.model_str_name
        )

    model_parameters["dataset"] = dataset

    elapsed_time = time.time() - start_time
    save_losses(
        losses,
        model_name,
        elapsed_time=elapsed_time,
        model_params=model_parameters,
        filename=save_results_filename,
    )
