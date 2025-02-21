from threading import Lock

import pandas as pd

from slrp_ev_ts_forecasting.compute_losses import Losses
from slrp_ev_ts_forecasting.default_parameters import (
    ADD_NUMBER_OF_EVSES_AVAILABLE,
    ALPHA,
    BATCH_SIZE,
    BETA,
    DATASET,
    DEFAULT_RESULTS_FILENAME,
    ERROR_METRIC,
    GET_VAL_DATA_FROM_SHUFFLED_TRAIN,
    LOOKAHEAD,
    NUMBER_OF_ARTIFICIAL_DATASETS,
    RANDOM_CHOICES,
    RANDOM_POWER_PROFILE_SHAPES,
    RANDOM_START_TIME,
    RANDOM_USER_NEEDS,
    RESULTS_PATH,
    SHUFFLE_POWER_PROFILES,
    TIME_MODE,
    X_DIM,
)

# Create a global lock
csv_lock = Lock()


def save_losses(
    losses: Losses,
    model_name: str,
    elapsed_time: float,
    model_params: dict = {},
    filename: str = DEFAULT_RESULTS_FILENAME,
) -> None:
    """Saves the losses to the results file.

    Args:
        losses (Losses): Losses to save.
        model_name (str): Name of the model.
    """
    additional_data = pd.DataFrame(
        {
            "date": pd.Timestamp.now(),
            "batch_size": model_params.get("batch_size", BATCH_SIZE),
            "x_dim": model_params.get("x_dim", X_DIM),
            "lookahead": model_params.get("lookahead", LOOKAHEAD),
            "time_mode": model_params.get("time_mode", TIME_MODE),
            "dataset": model_params.get("dataset", DATASET),
            "get_val_data_from_shuffled_train": model_params.get(
                "get_val_data_from_shuffled_train", GET_VAL_DATA_FROM_SHUFFLED_TRAIN
            ),
            "scaling_mode": model_params.get(
                "scaling_mode",
                model_params.get("data_scaling_mode", "rolling_standardize"),
            ),
            "add_number_of_evses_available": model_params.get(
                "add_number_of_evses_available", ADD_NUMBER_OF_EVSES_AVAILABLE
            ),
            "error_metric": model_params.get("error_metric", ERROR_METRIC),
            "model_name": model_name,
            "rmse": losses["rmse"],
            "relative_rmse": losses["relative_rmse"],
            f"wrmse (alpha={ALPHA})": losses["wrmse"],
            f"wprmse (beta={BETA})": losses["wprmse"],
            "mae": losses["mae"],
            "r2": losses["r2"],
            # "error_std": losses["error_std"],
            "smape": losses["smape"],
            "elapsed_time": elapsed_time,
            "number_of_artificial_datasets": model_params.get(
                "number_of_artificial_datasets", NUMBER_OF_ARTIFICIAL_DATASETS
            ),
            "random_start_time": model_params.get(
                "random_start_time", RANDOM_START_TIME
            ),
            "shuffle_power_profiles": model_params.get(
                "shuffle_power_profiles", SHUFFLE_POWER_PROFILES
            ),
            "random_power_profile_shapes": model_params.get(
                "random_power_profile_shapes", RANDOM_POWER_PROFILE_SHAPES
            ),
            "random_user_needs": model_params.get(
                "random_user_needs", RANDOM_USER_NEEDS
            ),
            "random_choices": model_params.get("random_choices", RANDOM_CHOICES),
        },
        index=[0],
    )

    results_file_path = RESULTS_PATH / f"{filename}.csv"

    with csv_lock:
        # read the file if it exists
        try:
            df_results = pd.read_csv(results_file_path, index_col=False)
        except FileNotFoundError:
            df_results = pd.DataFrame()
        df_results = pd.concat([df_results, additional_data], ignore_index=True)

        results_file_path.parent.mkdir(exist_ok=True)
        df_results.to_csv(results_file_path, index=False)
