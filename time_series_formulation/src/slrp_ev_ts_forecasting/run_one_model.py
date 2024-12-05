from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from slrp_ev_data import read_old_data, read_ucsd_data, train_test_split
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


def run_one_model(
    model_choice: TypeModelChoice,
    model_parameters={},
    verbose: bool = True,
    save_results_filename: str = DEFAULT_RESULTS_FILENAME,
) -> None:
    # Read the data
    print("# Starting...")
    if DATASET == "slrp-ev_old":
        data = read_old_data.read_old_data()
    elif DATASET == "ucsd-all_garages":
        data = read_ucsd_data.read_ucsd_data()
    else:
        raise ValueError(
            f"Datset of type {DATASET} is not defined. Please refer to "
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

    normalize_parameters = get_train_min_and_max(train)
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

    save_losses(losses, model_name, model_parameters, filename=save_results_filename)
