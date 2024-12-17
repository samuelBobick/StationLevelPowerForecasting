import pandas as pd
from slrp_ev_data import (
    read_new_slrpev_data,
    read_old_slrpev_data,
    read_ucsd_data,
    train_test_split,
)
from slrp_ev_data.feature_engineering import (
    convert_date_from_int_to_datetime,
    feature_engineering,
    reverse_feature_engineering,
)
from slrp_ev_data.normalization_and_standardization import get_train_min_and_max

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


def reverse_engineer_forecast(
    df_test_example, df_predictions, normalize_parameters
) -> pd.DataFrame:
    # Reverse engineer the forecast to get the original features back
    # convert from float32 to int64
    df_predictions["date"] = df_predictions["date"].astype("int64")

    # initialize final dataframe
    df_reversed_predictions = pd.DataFrame()
    df_reversed_predictions["date"] = convert_date_from_int_to_datetime(
        df_predictions["date"]
    )

    next_power_column_number = len(df_predictions.columns) - 1
    for i in range(next_power_column_number):
        col_name = f"power_{i}"
        if (i == next_power_column_number - 1) and (
            "real_power" in df_predictions.columns
        ):
            col_name = "real_power"
        # merge_asof performs a left merge with the closest date
        df_reverse_helper = pd.merge_asof(
            df_predictions[["date", col_name]],
            df_test_example.drop(columns=["power"]),
            on="date",
            direction="nearest",
        ).rename(columns={col_name: "power"})
        df_reverse_helper = df_reverse_helper.dropna(subset=["power"])
        df_reverse_helper = reverse_feature_engineering(
            df_reverse_helper, normalize_parameters, bypass_output_validation=True
        )

        df_reverse_helper = df_reverse_helper[["date", "power"]]
        helper_date_mask = df_reversed_predictions["date"].isin(
            df_reverse_helper["date"]
        )
        df_reversed_predictions.loc[helper_date_mask, col_name] = df_reverse_helper[
            "power"
        ]

    return df_reversed_predictions


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

    normalize_parameters = get_train_min_and_max(train, dataset_name=dataset)
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
    losses, df_predictions = model.predict(test_eng)

    data_length_days = df_predictions.shape[0] // 96
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
    df_predictions = reverse_engineer_forecast(
        test_eng, df_predictions, normalize_parameters
    )

    if verbose:
        visualize_forecast(
            test,
            df_predictions,
            data_length_days,
        )

    model_parameters["dataset"] = dataset
    save_losses(losses, model_name, model_parameters, filename=save_results_filename)
