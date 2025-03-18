from pathlib import Path
from typing import Literal, Optional

import torch

X_DIM = 96 * 2
LOOKAHEAD = 96
TIME_MODE: Literal["window", "cyclical"] = "cyclical"
GET_VAL_DATA_FROM_SHUFFLED_TRAIN = True

# Torch models
NUMBER_OF_INITIAL_MODELS = 1
EPOCHS = 7
BATCH_SIZE = 64
DROPOUT = 0.4
BATCH_NORM: bool = False

# KNN
N_NEIGHBORS = 4
PERCENTILE = 50

# Lags optimization
TypeOptimizeLags = Optional[Literal["short_opt", "long_opt"]]
OPTIMIZE_LAGS: TypeOptimizeLags = (
    None  # for regression models, such as Basic_NN or XGBoost
)
NUMBER_OF_DAYS_FOR_PACF = 35  # 70 was the old parameter, but it is too big for
# datasets with missing data

RESULTS_PATH = Path(__file__).parent / "results"
DEFAULT_RESULTS_FILENAME = "results"
SAVED_MODELS_PATH = Path(__file__).parent / "models" / "saved_models"
SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Error metrics
TypeErrorMetric = Literal["mse"]  # TODO: add "wmse" for xgboost (and knn if possible)
ERROR_METRIC: TypeErrorMetric = "mse"
ALPHA = 2  # for underpredictions error
BETA = 3  # for weighted peaks error

TypeDatasetName = Literal["slrp-ev_old", "slrp-ev_new", "ucsd-all_garages"]
DATASET: TypeDatasetName = "slrp-ev_new"

TypeScalingMode = Literal[
    "normalize", "standardize", "rolling_standardize", "rolling_normalize"
]
SCALING_MODE: TypeScalingMode = "normalize"
TIMESTEPS_ROLLING_WINDOW_FOR_SCALING = 96 * 30

SESSION_BASED_MODE = False
PEAK_PREDICTION = False
TypePeakPredictionMode = Literal["peak_of_day", "peak_next_8h"]
PEAK_PREDICTION_MODE = "peak_next_8h"
ADD_NUMBER_OF_SESSIONS = True
ADD_FRACTION_OF_REGULAR_SESSIONS = False
USE_ALL_ACTIVE_SESSIONS = True

# parameters to generate random sessions
NUMBER_OF_ARTIFICIAL_DATASETS = 0
RANDOM_START_TIME = True
# SHUFFLE_POWER_PROFILES and RANDOM_POWER_PROFILE_SHAPES can't be both True
SHUFFLE_POWER_PROFILES = True
RANDOM_POWER_PROFILE_SHAPES = False
RANDOM_USER_NEEDS = True
RANDOM_CHOICES = True

ADD_NUMBER_OF_EVSES_AVAILABLE = False

TypeModelChoice = Literal[
    "LinearRegression",
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
    "PeakPersistence",
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

RANDOM_SEED: int | None = None  # 42  # int(pd.Timestamp.now().timestamp())
VERBOSE = False
