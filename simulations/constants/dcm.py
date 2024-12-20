from typing import Literal

import numpy as np
from scipy.special import softmax


# Default discrete choice model parameters
def get_dcm_theta(power_rate: float) -> np.ndarray:
    dcm_charging_sch_params = np.array(
        [[-power_rate * 0.0184 / 2], [power_rate * 0.0184 / 2], [0], [0]]
    )
    dcm_charging_reg_params = np.array(
        [
            [power_rate * 0.0184 / 2],
            [-power_rate * 0.0184 / 2],
            [0],
            [0.341],
        ]
    )
    dcm_leaving_params = np.array(
        [[power_rate * 0.005 / 2], [power_rate * 0.005 / 2], [0], [-1]]
    )

    theta = np.vstack(
        (
            dcm_charging_sch_params.T,
            dcm_charging_reg_params.T,
            dcm_leaving_params.T,
        )
    )
    return theta


def _compute_dcm_version_2023(z, theta):
    # Tugba's version of the DCM
    return softmax(
        theta @ z
    ).flatten()  # reshape to convert to a 3*1 matrix (initially array of 3 elements)


def _compute_dcm_version_2024(z: list, theta):
    """This version uses the same dcm parameters as in the 2023 version
    but does it in 2 steps:
    1. Compute the probability of leaving
    2. Compute the probability of charging scheduled or regular

    Args:
        z (_type_): list of prices
        theta (_type_): dcm parameters

    Returns:
        Probabilities (scheduled, regular, leave)
    """
    # first we compute the probability of leaving
    utility_leave = theta[2] @ z
    v_leave = softmax([-utility_leave, utility_leave])
    probability_charge = v_leave[0]
    probability_leave = v_leave[1]
    v_choice = softmax(theta[:2] @ z)
    probability_scheduled = v_choice[0] * probability_charge
    probability_regular = v_choice[1] * probability_charge

    return np.array([probability_scheduled, probability_regular, probability_leave])


def get_dcm_v(z, theta, version: Literal["2023", "2024"] = "2024"):
    if version == "2023":
        return _compute_dcm_version_2023(z, theta)
    elif version == "2024":
        return _compute_dcm_version_2024(z, theta)
    else:
        raise ValueError("Invalid version")
