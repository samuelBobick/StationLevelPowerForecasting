from pathlib import Path
from typing import Literal, Optional

import torch

NUMBER_OF_INITIAL_MODELS = 1
X_DIM = 96 * 2
LOOKAHEAD = 96
EPOCHS = 5
TIME_MODE: Literal["cyclical", "window"] = "cyclical"
ALPHA = 2  # for underpredictions error
BETA = 3  # for weighted peaks error
BATCH_SIZE = 32
DROPOUT = 0.2
BATCH_NORM: bool = True

TypeOptimizeLags = Optional[Literal["short_opt", "long_opt"]]
OPTIMIZE_LAGS: TypeOptimizeLags = "long_opt"  # for regression models
NUMBER_OF_DAYS_FOR_PACF = 70

RESULTS_FILENAME = Path(__file__).parent / "results" / "results.csv"

TypeErrorMetric = Literal["mse"]  # TODO: add "wmse" for xgboost (and knn if possible)
ERROR_METRIC: TypeErrorMetric = "mse"

TypeDataSet = Literal["slrp-ev_old"]
DATASET: TypeDataSet = "slrp-ev_old"

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

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: " f"{torch.cuda.current_device()}")
    print(f"Name of current CUDA device: " f"{torch.cuda.get_device_name(cuda_id)}")
    DEVICE = "cuda"
else:
    print("CUDA is not available. Using CPU.")
    DEVICE = "cpu"
