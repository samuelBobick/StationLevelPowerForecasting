import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error  # type: ignore

from slrp_ev_ts_forecasting.default_parameters import BETA


def asymmetric_rmse(
    alpha: float, y_pred: np.ndarray | list, y_true: np.ndarray | list
) -> float:
    """This implementation gives different results than
    the torch implementation"""
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    weights = alpha ** (1 - np.sign(y_pred - y_true))
    rwmse = np.sqrt(mean_squared_error(y_pred, y_true, sample_weight=weights))
    return rwmse


def asymmetric_rmse_detailed(
    y_true: np.ndarray | list, y_pred: np.ndarray | list, alpha: float
) -> float:
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mse_loss = (y_true - y_pred) ** 2
    asymmetric_weight = alpha ** (1 - np.sign(y_pred - y_true))
    weighted_mse = asymmetric_weight * mse_loss
    rmse = np.sqrt(np.mean(weighted_mse))
    return rmse


class AsymmetricRMSELoss(nn.Module):
    def __init__(self, alpha):
        super(AsymmetricRMSELoss, self).__init__()
        self.multiplier = alpha

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        loss = torch.sqrt(
            torch.mean(
                torch.pow(self.multiplier, 1 - torch.sign(input - target)) * mse_loss
            )
        )
        return loss


def weighted_peaks_rmse(
    y_pred: np.ndarray | list, y_true: np.ndarray | list, beta: float = BETA
) -> float:
    """
    Calculate the weighted peaks RMSE. This loss puts more weight on
    the peaks.
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    # remember that the values are scaled and should be between 0 and 1
    # if not, we scale them the apply the weights
    if np.max(y_true) > 1:
        y_true_for_weights = y_true / np.max(y_true)
    else:
        y_true_for_weights = y_true
    weights = 1 + beta * y_true_for_weights
    # we need to add a 1 to avoid the weight being 0 when y_true is 0

    # wprmse = np.sqrt(mean_squared_error(y_pred, y_true, sample_weight=weights))
    mse_loss = (y_true - y_pred) ** 2
    weighted_mse = weights * mse_loss
    wprmse = np.sqrt(np.mean(weighted_mse))
    return wprmse


class WeightedPeaksRMSELoss(nn.Module):
    def __init__(self, beta: float = BETA):
        super(WeightedPeaksRMSELoss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        mse_loss = nn.functional.mse_loss(input, target, reduction="none")
        weights = 1 + self.beta * target
        loss = torch.sqrt(torch.mean(weights * mse_loss))
        return loss
