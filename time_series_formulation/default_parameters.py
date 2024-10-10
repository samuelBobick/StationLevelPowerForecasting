from pathlib import Path
from typing import Literal

NUMBER_OF_INITIAL_MODELS = 1
X_DIM = 96 * 2
LOOKAHEAD = 96
EPOCHS = 2
TIME_MODE: Literal["cyclical", "window"] = "cyclical"
ALPHA = 2  # for underpredictions error
BETA = 3  # for weighted peaks error
BATCH_SIZE = 32
DROPOUT = 0.2
RESULTS_FILENAME = Path(__file__).parent / "results" / "results.csv"

TypeErrorMetric = Literal["mse"]  # TODO: add "wmse" for xgboost (and knn if possible)
ERROR_METRIC: TypeErrorMetric = "mse"

TypeModelChoice = Literal[
    "KNN",
    "XGBoost",
    "Basic_NN",
    "Last_Week",
    "Similar_Day",
    "LSTM",
    "TCN",
    "AutoETS",
    "AutoARIMA",
    "ARIMA",
    "Prophet",
]
