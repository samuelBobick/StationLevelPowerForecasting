from baseline_simulator import BaselineSimulator
from typing import Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants.dcm import get_dcm_theta
from constants.tariffs import DICT_TARIFFS, TypeTariffName
from scipy.special import softmax
from tqdm.auto import tqdm
from utils import (
    get_e_need,
    get_new_reg_obj,
    get_new_sch_obj,
    get_timestep_info,
)

class ThresholdSimulator(BaselineSimulator):
    def __init__(
        self,
        test_df,
        var_dim_constant: int = 96,
        delta_t: float = 0.25,
        power_rate: float = 6.6,
        flexibility_constant: float = 0.57,
        tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
        custom_cost_dc: Optional[float] = 500,
        monte_carlo: bool = False,
        verbose: bool = False,
        step: int = 1
    ):

        super().__init__(
            test_df,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            monte_carlo,
            verbose
        )

        self.step = step

    def argmin_u(
        self,
        z: list,
        v: np.ndarray,
        sub_df: pd.DataFrame,
        current_time: pd.Timestamp,
        running_peak: float,
        power_profiles: dict,
        prices: dict,
    ):
        """
        Function to minimize charging cost. Flexible charging with variable power schedule

        Inputs:
            z: array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1]
            v: array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
            sub_df: dataframe containing rows of sessions_df that represent active \
                sessions at the time of optimization
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            power_profiles: dictionary mapping dcosIds to power_profiles
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        u, e_delivered, J, J_array, p_dc_sch, p_dc_reg, current_peak_sch, current_peak_reg, constraints = self.initialize_problem(z, v, sub_df, current_time, running_peak, power_profiles, prices)

        # Hard threshold constraint
        constraints += [p_dc_sch <= running_peak]
        constraints += [p_dc_reg <= running_peak]

        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve()

        while running_peak <= self.power_rate * 8 and prob.status != "optimal":
            # Increment the hard threshold constraints by self.step
            constraints = constraints[:-2]
            running_peak += self.step
            constraints += [p_dc_sch <= running_peak]
            constraints += [p_dc_reg <= running_peak]

            obj = cp.Minimize(J)
            prob = cp.Problem(obj, constraints)
            prob.solve()

            if prob.status != "optimal":
                print(prob.status)
                print("Gurobi failed, cant solve for power")
                prob.solve(solver="GUROBI", verbose=True)

            return (
                u.value,
                e_delivered.value,
                p_dc_sch.value[0],
                p_dc_reg.value[0],
                current_peak_sch.value,
                current_peak_reg.value,
                J,
                J_array,
            )