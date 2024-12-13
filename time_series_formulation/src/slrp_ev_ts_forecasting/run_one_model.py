import pandas as pd
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
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
from slrp_ev_ts_forecasting.models.ffnn import FFNN
from slrp_ev_ts_forecasting.models.knn import KNN
from slrp_ev_ts_forecasting.models.last_week import LastWeek
from slrp_ev_ts_forecasting.models.linear_model import LinearModel
from slrp_ev_ts_forecasting.models.lstm import LSTM
from slrp_ev_ts_forecasting.models.similar_day import SimilarDay
from slrp_ev_ts_forecasting.models.sktime_base import SktimeBaseModel
from slrp_ev_ts_forecasting.models.tcn import TCN
from slrp_ev_ts_forecasting.models.xgboost_model import XGBoost
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
        # merge_asof performs a left merge with the closest date
        df_reverse_helper = pd.merge_asof(
            df_predictions[["date", f"power_{i}"]],
            df_test_example.drop(columns=["power"]),
            on="date",
            direction="nearest",
        ).rename(columns={f"power_{i}": "power"})
        df_reverse_helper = df_reverse_helper.dropna(subset=["power"])
        df_reverse_helper = reverse_feature_engineering(
            df_reverse_helper, normalize_parameters, bypass_output_validation=True
        )

        df_reverse_helper = df_reverse_helper[["date", "power"]]
        helper_date_mask = df_reversed_predictions["date"].isin(
            df_reverse_helper["date"]
        )
        df_reversed_predictions.loc[helper_date_mask, f"power_{i}"] = df_reverse_helper[
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

    normalize_parameters = get_train_min_and_max(train, dataset_name=dataset)
    train_eng = feature_engineering(train, normalize_parameters)
    val_eng = (
        feature_engineering(val, normalize_parameters) if val is not None else None
    )
    test_eng = feature_engineering(test, normalize_parameters)

    dict_model: dict[TypeModelChoice, dict] = {
        "LinearRegression": {
            "model": LinearModel,
            "model_params": {},
            "fit_params": {},
        },
        "KNN": {
            "model": KNN,
            "model_params": {},
            "fit_params": {},
        },
        "XGBoost": {
            "model": XGBoost,
            "model_params": {},
            "fit_params": {},
        },
        "Last_Week": {"model": LastWeek, "model_params": {}, "fit_params": {}},
        "Similar_Day": {"model": SimilarDay, "model_params": {}, "fit_params": {}},
        "Basic_NN": {
            "model": FFNN,
            "model_params": {},
            "fit_params": {},
        },
        "LSTM": {
            "model": LSTM,
            "model_params": {},
            "fit_params": {},
        },
        "TCN": {
            "model": TCN,
            "model_params": {},
            "fit_params": {},
        },
        "AutoETS": {
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": AutoETS(auto=True, sp=24, n_jobs=-1, maxiter=20),
                "include_exogenous": True,
                "downsample_hours": 1,
                "refit_model_before_predictions": False,
            },  # default maxiter = 1000
            "fit_params": {},
        },
        "AutoARIMA": {
            # takes a very long time to run...
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": AutoARIMA(
                    sp=int(96 / (4 * 4)),
                    out_of_sample_size=int(96 / (4 * 4) * 7),
                    maxiter=15,
                    n_jobs=-1,
                    start_params=[1, 1],
                    max_order=7,
                    seasonal=True,
                    stationary=True,
                ),
                "include_exogenous": True,
                "downsample_hours": 4,
                "refit_model_before_predictions": False,
                "start_data_date": "2023-08",
            },  # default maxiter = 1000
            "fit_params": {},
        },
        "ARIMA": {
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": ARIMA(
                    order=(1, 0, 1),
                    seasonal_order=(1, 1, 1, 96 * 7 / (4 * 2)),
                ),
                "include_exogenous": False,
                "downsample_hours": 2,
                "refit_model_before_predictions": False,
                "start_data_date": "2023-01",
            },  # default maxiter = 1000
            "fit_params": {},
        },
        "Prophet": {
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": Prophet(
                    seasonality_mode="additive",
                    n_changepoints=40,
                    # add_country_holidays={"country_name": "US"},
                    yearly_seasonality=False,  # type: ignore
                    weekly_seasonality=True,  # type: ignore
                    daily_seasonality=True,  # type: ignore
                    # the three growth arguments go together.
                    # They make the model slightly more precise (by a few percents)
                    # but also much slower to make predictions
                    # Don't forget that the data is normalized (btw 0 and 1)
                    # growth_floor=0,
                    # growth_cap=1,
                    # growth="logistic",
                ),
                "include_exogenous": False,
                "downsample_hours": 2,
                "refit_model_before_predictions": False,
                # "start_data_date": "2023",
            },
            "fit_params": {},
        },
    }
    model_parameters = model_parameters | dict_model[model_choice]["model_params"]
    print(
        f"Model choice: {model_choice}, with the following parameters for the initialization: {model_parameters } "
    )

    model = dict_model[model_choice]["model"](**model_parameters)
    model_name = getattr(model, "model_str_name", model_choice)
    print("# Fitting...")
    model.fit(train_eng, val=val_eng, **dict_model[model_choice]["fit_params"])

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

    save_losses(losses, model_name, model_parameters, filename=save_results_filename)
