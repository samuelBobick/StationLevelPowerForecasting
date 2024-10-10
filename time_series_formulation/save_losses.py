import pandas as pd
from compute_losses import Losses
from default_parameters import (
    ALPHA,
    BATCH_SIZE,
    BETA,
    ERROR_METRIC,
    LOOKAHEAD,
    RESULTS_FILENAME,
    TIME_MODE,
    X_DIM,
)


def save_losses(losses: Losses, model_name: str) -> None:
    """Saves the losses to the results file.

    Args:
        losses (Losses): Losses to save.
        model_name (str): Name of the model.
    """
    additional_data = pd.DataFrame(
        {
            "date": pd.Timestamp.now(),
            "batch_size": BATCH_SIZE,
            "x_dim": X_DIM,
            "lookahead": LOOKAHEAD,
            "time_mode": TIME_MODE,
            "error_metric": ERROR_METRIC,
            "model_name": model_name,
            "rmse": losses["rmse"],
            f"wrmse (alpha={ALPHA})": losses["wrmse"],
            f"wprmse (beta={BETA})": losses["wprmse"],
            "mae": losses["mae"],
            "r2": losses["r2"],
        },
        index=[0],
    )

    # read the file if it exists
    try:
        df_results = pd.read_csv(RESULTS_FILENAME, index_col=False)
    except FileNotFoundError:
        df_results = pd.DataFrame()
    df_results = pd.concat([df_results, additional_data], ignore_index=True)

    df_results.to_csv(RESULTS_FILENAME, index=False)
