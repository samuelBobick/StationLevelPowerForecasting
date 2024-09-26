from typing import Literal

from ffnn import FFNN
from knn import KNN
from last_week import LastWeek
from lstm import LSTM
from old_ffnn2 import old_NeuralNet
from old_knn2 import OldKNN
from similar_day import SimilarDay
from slrp_ev_data import read_old_data, train_test_split
from slrp_ev_data.feature_engineering import (
    feature_engineering,
    get_train_min_and_max,
    reverse_feature_engineering,
)
from STL import STLARIMA
from visualization import visualize_forecast

TypeModelChoice = Literal[
    "KNN",
    "OldKNN",
    "Basic_NN",
    "Old_Basic_NN",
    "STL with ARIMA",
    "Last Week",
    "Similar Day",
    "LSTM",
]
model_choice: TypeModelChoice = "LSTM"
number_of_initial_models = 1
x_dim = 16
lookahead = 96
epochs = 5
time_mode: Literal["cyclical", "window"] = "cyclical"

if __name__ == "__main__":
    # Read the data
    data = read_old_data.read_old_data()
    train, val, test = train_test_split.train_test_split(data, generate_validation=True)  # type: ignore
    normalize_parameters = get_train_min_and_max(train)
    train_eng = feature_engineering(train, normalize_parameters)
    val_eng = feature_engineering(val, normalize_parameters)
    test_eng = feature_engineering(test, normalize_parameters)

    dict_model: dict[TypeModelChoice, dict] = {
        "KNN": {
            "model": KNN,
            "model_params": {"time_mode": time_mode},
            "fit_params": {},
        },
        "OldKNN": {"model": OldKNN, "model_params": {}, "fit_params": {}},
        "Last Week": {"model": LastWeek, "model_params": {}, "fit_params": {}},
        "Similar Day": {"model": SimilarDay, "model_params": {}, "fit_params": {}},
        "Old_Basic_NN": {"model": old_NeuralNet, "model_params": {}, "fit_params": {}},
        "Basic_NN": {
            "model": FFNN,
            "model_params": {
                "time_mode": time_mode,
                "x_dim": x_dim,
                "number_of_initial_models": number_of_initial_models,
                "epochs": epochs,
                "lookahead": lookahead,
            },
            "fit_params": {
                "val": val_eng,
            },
        },
        "STL with ARIMA": {"model": STLARIMA, "model_params": {}, "fit_params": {}},
        "LSTM": {
            "model": LSTM,
            "model_params": {
                "time_mode": time_mode,
                "x_dim": x_dim,
                "epochs": epochs,
                "number_of_initial_models": number_of_initial_models,
                "lookahead": lookahead,
            },
            "fit_params": {
                "val": val_eng,
            },
        },
    }
    print("# Starting")
    print(
        f"Model choice: {model_choice}, with the following parameters for the initialization: {dict_model[model_choice]['model_params']} "
    )

    model = dict_model[model_choice]["model"](
        **dict_model[model_choice]["model_params"]
    )
    model.fit(train_eng, **dict_model[model_choice]["fit_params"])

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
