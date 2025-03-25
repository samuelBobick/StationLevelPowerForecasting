from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from baseline_simulator import BaselineSimulator
from constants.tariffs import MODIFIED_DC, TypeTariffName


class ThresholdSimulator(BaselineSimulator):
    def __init__(
        self,
        test_df,
        var_dim_constant: int = 96,
        delta_t: float = 0.25,
        power_rate: float = 6.6,
        flexibility_constant: float = 0.57,
        tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
        custom_cost_dc: Optional[float] = MODIFIED_DC,
        initial_running_peak: float = 0,
        monte_carlo: bool = False,
        verbose: bool = False,
        step: float = 1,
    ):
        """
        Initialize child of BaselineSimulator which iteratively attempts to optimize with a hard threshold peak power threshold

        Args:
            var_dim_constant: 24-hour lookahead. Default is 96 (96 timesteps in a day with 15min data).
            delta_t: Size, in hour, of a timestep (e.g. 15min interval are 0.25h intervals). Default is 0.25.
            power_rate: Maximum power in kW of the chargers. Default is 6.6 kW.
            flexibility_constant: Proportion of flexibility to artificially reduce to the energy need of the \
                regular users, compared to the cumulative energy they used historically. \
                Default is 0.57 (the historical average flexibility of scheduled sessions).
            tariff_name: Name of the tariff to use. See constants.tariffs.TypeTariffName for available options. \
                Default is "BEV2S Secondary June 2023".
            custom_cost_dc: Custom demand charge cost in cents/kW that will replace the one from the tariffs. \
                Set to None to use the dc of the selected tariff. Default is 500 cents/kW.
            monte_carlo: Whether to re-evaluate choices with Discrete Choice Model (DCM). \
                If False, we assume the charging choice of each session is the one historical done. \
                If True, we will use the DCM to simulate the choice. Default is False.
            verbose: Print optimization information. Default is False
            step: how much to increase peak power threshold per iteration.
        """

        super().__init__(
            test_df,
            var_dim_constant,
            delta_t,
            power_rate,
            flexibility_constant,
            tariff_name,
            custom_cost_dc,
            initial_running_peak,
            monte_carlo,
            verbose,
        )

        self.step = step

    def argmin_u(
        self,
        z: list,
        v: np.ndarray,
        sub_df: pd.DataFrame,
        current_time: pd.Timestamp,
        running_peak: float,
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
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        (
            u,
            e_delivered,
            J,
            J_array,
            current_peak_sch,
            current_peak_reg,
            constraints,
        ) = self.initialize_problem(z, v, sub_df, current_time, running_peak, prices)

        # Hard threshold constraint
        constraints += [current_peak_sch <= running_peak]
        constraints += [current_peak_reg <= running_peak]

        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)

        while running_peak <= self.power_rate * 8 and prob.status != "optimal":
            # Increment the hard threshold constraints by self.step
            constraints = constraints[:-2]
            running_peak += self.step
            constraints += [current_peak_sch <= running_peak]
            constraints += [current_peak_reg <= running_peak]

            obj = cp.Minimize(J)
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)

        if prob.status != "optimal":
            raise Exception(f"Optimization failed with status {prob.status}")

        return (
            u,
            e_delivered.value,
            current_peak_sch.value,
            current_peak_reg.value,
            J,
            J_array,
        )
