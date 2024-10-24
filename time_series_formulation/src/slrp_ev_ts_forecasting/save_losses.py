import pandas as pd

from slrp_ev_ts_forecasting.compute_losses import Losses
from slrp_ev_ts_forecasting.default_parameters import (
    ALPHA,
    BATCH_SIZE,
    BETA,
    DATASET,
    DEFAULT_RESULTS_FILENAME,
    ERROR_METRIC,
    LOOKAHEAD,
    RESULTS_PATH,
    TIME_MODE,
    X_DIM,
)


def save_losses(
    losses: Losses,
    model_name: str,
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
            "error_metric": model_params.get("error_metric", ERROR_METRIC),
            "model_name": model_name,
            "rmse": losses["rmse"],
            f"wrmse (alpha={ALPHA})": losses["wrmse"],
            f"wprmse (beta={BETA})": losses["wprmse"],
            "mae": losses["mae"],
            "r2": losses["r2"],
        },
        index=[0],
    )

    results_file_path = RESULTS_PATH / f"{filename}.csv"

    # read the file if it exists
    try:
        df_results = pd.read_csv(results_file_path, index_col=False)
    except FileNotFoundError:
        df_results = pd.DataFrame()
    df_results = pd.concat([df_results, additional_data], ignore_index=True)

    results_file_path.parent.mkdir(exist_ok=True)
    df_results.to_csv(results_file_path, index=False)
