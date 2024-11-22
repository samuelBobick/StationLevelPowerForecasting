import numpy as np


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
    dcm_leaving_params = np.array([[power_rate * 0.005], [0], [0], [-1]])

    theta = np.vstack(
        (
            dcm_charging_sch_params.T,
            dcm_charging_reg_params.T,
            dcm_leaving_params.T,
        )
    )
    return theta
