from collections.abc import Callable
from typing import Literal

from ffnn import NeuralNet
from knn import KNN
from last_week import LastWeek
from old_knn import OldKNN
from similar_day import SimilarDay
from slrp_ev_data import read_old_data, train_test_split
from STL import STLARIMA
from visualization import visualize_forecast

model_choice: Literal[
    "KNN", "OldKNN", "Neural Net", "STL with ARIMA", "Last Week", "Similar Day"
] = "KNN"

if __name__ == "__main__":
    dict_model: dict[str, Callable] = {
        "KNN": KNN,
        "OldKNN": OldKNN,
        "Last Week": LastWeek,
        "Similar Day": SimilarDay,
        "Neural Net": NeuralNet,
        "STL with ARIMA": STLARIMA,
    }
    # Read the data
    data = read_old_data.read_old_data()
    train, val, test = train_test_split.train_test_split(data, generate_validation=True)  # type: ignore

    model = dict_model[model_choice]()
    model.fit(train)

    df_eval = test
    rmse, wrmse, forecast, forecast_dates = model.predict(df_eval)
    data_length_days = len(forecast) // 96
    print(
        f"{model_choice}: ",
        "RMSE: {number:.0f}".format(number=rmse),
        ", WRMSE: {number:.0f}".format(number=wrmse),
        f"for around {data_length_days} days of predictions",
    )

    visualize_forecast(df_eval, forecast, 10, forecast_dates=forecast_dates)
