from typing import Literal, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torcheval.metrics import R2Score

from slrp_ev_ts_forecasting.asymmetric_loss import (
    AsymmetricRMSELoss,
    WeightedPeaksRMSELoss,
    asymmetric_rmse_detailed,
    weighted_peaks_rmse,
)


class Losses(TypedDict):
    rmse: float
    relative_rmse: float
    wrmse: float
    mae: float
    wprmse: float
    r2: float
    smape: float
    # error_std: float


TypeMetrics = Literal[
    "rmse",
    "relative_rmse",
    "wrmse (alpha=2)",
    "mae",
    "wprmse (beta=3)",
    "r2",
    "smape",
    "elapsed_time",
]


def compute_losses(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    y_naive_pred: Optional[np.ndarray] = None,
    y_naive_true: Optional[np.ndarray] = None,
) -> Losses:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # We compute the relative rmse error (scale invariant)
    # source: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.performance_metrics.forecasting.RelativeLoss.html
    # and https://robjhyndman.com/papers/mase.pdf part 2.4

    if (y_naive_pred is not None) and (y_naive_true is not None):
        naive_rmse = np.sqrt(mean_squared_error(y_naive_true, y_naive_pred))
        relative_rmse = rmse / naive_rmse
    else:
        relative_rmse = 999


    wrmse = asymmetric_rmse_detailed(y_true, y_pred, alpha)
    mae = mean_absolute_error(y_true, y_pred)
    wprmse = weighted_peaks_rmse(y_pred, y_true)
    r2 = r2_score(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    # error_std = np.std(np.array(y_pred) - np.array(y_true))
    return Losses(rmse=rmse, relative_rmse=relative_rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2, smape=smape)  # type: ignore


def compute_torch_losses(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float
) -> Losses:
    rmse = torch.sqrt(nn.functional.mse_loss(y_pred, y_true, reduction="mean")).item()
    wrmse = AsymmetricRMSELoss(alpha)(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    wprmse = WeightedPeaksRMSELoss()(y_pred, y_true).item()
    # Both tensors are on the same device but we had a problem with the R2Score
    # so we can just move them to the cpu
    # print(f"y_pred device: {y_pred.device}")
    # print(f"y_true device: {y_true.device}")
    r2 = R2Score().update(y_pred.cpu(), y_true.cpu()).compute().item()
    smape = symmetric_mean_absolute_percentage_error(
        y_true.cpu().numpy(), y_pred.cpu().numpy()
    )
    # error_std = torch.std(y_pred - y_true).item()
    return Losses(rmse=rmse, wrmse=wrmse, mae=mae, wprmse=wprmse, r2=r2, smape=smape)  # type: ignore


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """source: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
    Compared to classic percentage error, this measure is defined when the true value is zero.
    However is does not perform very well when the true value is close to zero (it will tend to give
    its max value - 200% - no matter how far away the prediction is to 0).
    """

    denom = np.abs(y_true) + np.abs(y_pred)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0] = 1.0
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom)
