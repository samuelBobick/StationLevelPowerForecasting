from typing import Literal

from ffnn import FFNN
from knn import KNN
from last_week import LastWeek
from lstm import LSTM
from old_ffnn2 import old_NeuralNet
from old_knn2 import OldKNN
from similar_day import SimilarDay
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from sktime_base import SktimeBaseModel
from slrp_ev_data import read_old_data, train_test_split
from slrp_ev_data.feature_engineering import (
    feature_engineering,
    get_train_min_and_max,
    reverse_feature_engineering,
)
from STL import STLARIMA
from tcn import TCN
from visualization import visualize_forecast
from xgboost_model import XGBoost

TypeModelChoice = Literal[
    "KNN",
    "XGBoost",
    "OldKNN",
    "Basic_NN",
    "Old_Basic_NN",
    "STL with ARIMA",
    "Last Week",
    "Similar Day",
    "LSTM",
    "TCN",
    "AutoETS",
    "AutoARIMA",
    "Prophet",
]
model_choice: TypeModelChoice = "Basic_NN"

if __name__ == "__main__":
    # Read the data
    print("# Starting...")
    data = read_old_data.read_old_data()
    train, val, test = train_test_split.train_test_split(data, generate_validation=True)  # type: ignore
    normalize_parameters = get_train_min_and_max(train)
    train_eng = feature_engineering(train, normalize_parameters)
    val_eng = feature_engineering(val, normalize_parameters)
    test_eng = feature_engineering(test, normalize_parameters)

    dict_model: dict[TypeModelChoice, dict] = {
        "KNN": {
            "model": KNN,
            "model_params": {},
            "fit_params": {
                "val": val_eng,
            },
        },
        "XGBoost": {
            "model": XGBoost,
            "model_params": {},
            "fit_params": {
                "val": val_eng,
            },
        },
        "OldKNN": {"model": OldKNN, "model_params": {}, "fit_params": {}},
        "Last Week": {"model": LastWeek, "model_params": {}, "fit_params": {}},
        "Similar Day": {"model": SimilarDay, "model_params": {}, "fit_params": {}},
        "Old_Basic_NN": {"model": old_NeuralNet, "model_params": {}, "fit_params": {}},
        "Basic_NN": {
            "model": FFNN,
            "model_params": {},
            "fit_params": {
                "val": val_eng,
            },
        },
        "STL with ARIMA": {"model": STLARIMA, "model_params": {}, "fit_params": {}},
        "LSTM": {
            "model": LSTM,
            "model_params": {},
            "fit_params": {
                "val": val_eng,
            },
        },
        "TCN": {
            "model": TCN,
            "model_params": {},
            "fit_params": {
                "val": val_eng,
            },
        },
        "AutoETS": {
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": AutoETS(auto=True, sp=24, n_jobs=-1, maxiter=20),
                "include_exogenous": True,
                "reduce_data_frequency": True,
                "refit_model_before_predictions": False,
            },  # default maxiter = 1000
            "fit_params": {
                "val": val_eng,
            },
        },
        "AutoARIMA": {
            # takes a very long time to run...
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": AutoARIMA(sp=24, maxiter=5, n_jobs=-1),
                "include_exogenous": False,
                "reduce_data_frequency": True,
                "refit_model_before_predictions": False,
                "start_data_date": "2023-09",
            },  # default maxiter = 1000
            "fit_params": {
                "val": val_eng,
            },
        },
        "Prophet": {
            "model": SktimeBaseModel,
            "model_params": {
                "forecaster": Prophet(
                    seasonality_mode="additive",
                    n_changepoints=40,
                    # add_country_holidays={"country_name": "US"},
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    # the three growth arguments go together.
                    # They make the model slightly more precise (by a few percents)
                    # but also much slower to make predictions
                    # Don't forget that the data is normalized (btw 0 and 1)
                    # growth_floor=0,
                    # growth_cap=1,
                    # growth="logistic",
                ),
                "include_exogenous": False,
                "reduce_data_frequency": True,
                "refit_model_before_predictions": False,
                # "start_data_date": "2023",
            },
            "fit_params": {
                "val": val_eng,
            },
        },
    }

    print(
        f"Model choice: {model_choice}, with the following parameters for the initialization: {dict_model[model_choice]['model_params']} "
    )

    model = dict_model[model_choice]["model"](
        **dict_model[model_choice]["model_params"]
    )
    print("# Fitting...")
    model.fit(train_eng, **dict_model[model_choice]["fit_params"])

    print("# Making prediction(s)...")
    # rmse, wrmse, forecast, forecast_dates = model.predict(test_eng)
    rmse, wrmse, forecast, forecast_dates = model.predict(test_eng)

    data_length_days = len(forecast) // 96
    train_min, train_max = normalize_parameters
    print(
        f"{model_choice}: ",
        "RMSE: {number:.0f}".format(
            number=rmse * (train_max["power"] - train_min["power"]) + train_min["power"]
        ),
        ", WRMSE: {number:.0f}".format(
            number=wrmse * (train_max["power"] - train_min["power"])
            + train_min["power"]
        ),
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

    visualize_forecast(
        test, df_forecast["power"], data_length_days, forecast_dates=df_forecast["date"]
    )
