import numpy as np
from sklearn.metrics import root_mean_squared_error  # type: ignore


def asymmetric_rmse(alpha: float, forecast: np.ndarray, real: np.ndarray) -> float:
    weights = alpha ** (1 - np.sign(forecast - real))
    rwmse = root_mean_squared_error(forecast, real, sample_weight=weights)
    return rwmse


def asymmetric_rmse_detailed(y_true, y_pred, alpha: int) -> float:
    # Unused
    mse_loss = (y_true - y_pred) ** 2
    asymmetric_weight = alpha ** (1 - np.sign(y_pred - y_true))
    weighted_mse = asymmetric_weight * mse_loss
    rmse = np.sqrt(np.mean(weighted_mse))
    return rmse
