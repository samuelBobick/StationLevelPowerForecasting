import pandas as pd
import sys
from typing import Literal
from collections.abc import Callable

from knn import KNN
from ffnn import NeuralNet
from STL import STLARIMA
from last_week import LastWeek
from similar_day import SimilarDay
from visualization import visualize_forecast

from slrp_ev_data import read_old_data
from slrp_ev_data import train_test_split

model_choice: Literal[
    "KNN", "Neural Net", "STL with ARIMA", "Last Week", "Similar Day"
] = "Neural Net"

if __name__ == "__main__":
    dict_model: dict[str, Callable] = {
        "KNN": KNN,
        "Last Week": LastWeek,
        "Similar Day": SimilarDay,
        "Neural Net": NeuralNet,
        "STL with ARIMA": STLARIMA,
    }
    # Read the data
    data = read_old_data.read_old_data()
    train, test = train_test_split.train_test_split(data)

    model = dict_model[model_choice]()
    model.fit(train)

    rmse, wrmse, forecast = model.predict(test)
    data_lenght_days = len(forecast) // 96
    print(
        f"{model_choice}: ",
        "RMSE: {number:.0f}".format(number=rmse),
        ", WRMSE: {number:.0f}".format(number=wrmse),
        f"for around {data_lenght_days} days of predictions",
    )

    visualize_forecast(test, forecast, 10)
