from slrp_ev_data import (
    read_new_slrpev_data,
    read_old_slrpev_data,
    read_ucsd_data,
    train_test_split,
)
from slrp_ev_data.feature_engineering import (
    feature_engineering,
    get_train_min_and_max,
    reverse_feature_engineering,
)

from slrp_ev_ts_forecasting.compute_losses import get_real_scale_losses
from slrp_ev_ts_forecasting.default_parameters import (
    DATASET,
    DEFAULT_RESULTS_FILENAME,
    GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
    TypeDataSet,
    TypeModelChoice,
)
from slrp_ev_ts_forecasting.models.dict_models import DICT_MODEL
from slrp_ev_ts_forecasting.models.ffnn import FFNN
from slrp_ev_ts_forecasting.models.regression_base import RegressionBaseModel
from slrp_ev_ts_forecasting.save_losses import save_losses
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

    normalize_parameters = get_train_min_and_max(train)
    train_eng = feature_engineering(train, is_regression_model, normalize_parameters)
    val_eng = (
        feature_engineering(val, is_regression_model, normalize_parameters)
        if val is not None
        else None
    )
    test_eng = feature_engineering(test, is_regression_model, normalize_parameters)

    model_parameters = model_parameters | DICT_MODEL[model_choice]["model_params"]
    print(
        f"Model choice: {model_choice}, with the following parameters for the initialization: {model_parameters } "
    )

    model = model_class(**model_parameters)
    model_name = getattr(model, "model_str_name", model_choice)
    print("# Fitting...")
    model.fit(train_eng, val=val_eng, **DICT_MODEL[model_choice]["fit_params"])

    print("# Making prediction(s)...")
    losses, forecast, forecast_dates = model.predict(test_eng)

    data_length_days = len(forecast) // 96
    losses = get_real_scale_losses(losses, normalize_parameters=normalize_parameters)
    print(
        f"{model_choice}: ",
        *[
            f"{loss_type.upper()}: {loss_value:.1f};"
            for loss_type, loss_value in losses.items()
        ],
        f"for around {data_length_days} days of predictions",
    )

    # Reverse engineer the forecast to get the original features back
    df_forecast = test_eng.copy()
    df_forecast = df_forecast.iloc[: len(forecast)]
    df_forecast["power"] = forecast
    df_forecast["date"] = forecast_dates
    df_forecast = reverse_feature_engineering(
        df_forecast, normalize_parameters, bypass_output_validation=True
    )

    if verbose:
        visualize_forecast(
            test,
            df_forecast["power"],
            data_length_days,
            forecast_dates=df_forecast["date"],
        )

    model_parameters["dataset"] = dataset
    save_losses(losses, model_name, model_parameters, filename=save_results_filename)
