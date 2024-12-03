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
            Function to minimize charging cost. Flexible charging with variable power schedule.
            Repeatedly attempts to solve optimization problem with an increasingly loose hard threshold.

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
            e_need_lst = []
            N_remain_lst = []
            price_lst = []

            last_row = sub_df.iloc[-1]
            TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                last_row, current_time, self.delta_t
            )
            e_need = get_e_need(
                last_row,
                current_time,
                power_profiles,
                self.delta_t,
                self.power_rate,
                self.flexibility_constant,
            )
            e_need_lst.append(e_need)
            N_remain_lst.append(N_remain)

            for index, row in (
                sub_df.iloc[:-1].loc[sub_df["choice"] == "SCHEDULED"].iterrows()
            ):
                TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                    row, current_time, self.delta_t
                )
                e_need = get_e_need(
                    row,
                    current_time,
                    power_profiles,
                    self.delta_t,
                    self.power_rate,
                    self.flexibility_constant,
                )
                e_need_lst.append(e_need)
                N_remain_lst.append(N_remain)

                if prices[row["dcosId"]]:
                    price_lst.append(prices[row["dcosId"]])
                else:
                    price_lst.append(row["sch_centsPerHr"])

            num_sch_user = len(e_need_lst)

            ### Decision Variables
            e_delivered = cp.Variable(
                shape=((self.var_dim_constant + 1) * num_sch_user, 1)
            )  # energy delivered
            u = cp.Variable(
                shape=(self.var_dim_constant * num_sch_user, 1)
            )  # charging profile (extra scheduled user profile added in case new user chooses scheduled
            p_dc_sch = cp.Variable(shape=1)
            p_dc_reg = cp.Variable(shape=1)

            ### Constraints incorporate all SCH users
            constraints = [u >= 0, u <= self.power_rate]

            # Iterate through all existing flex users
            for i in range(num_sch_user):
                e_need = e_need_lst[i]
                N_remain = N_remain_lst[i]

                # For now we don't have sessions longer than 1 day, but we
                # add that check in case it happens in the future
                assert N_remain <= self.var_dim_constant, (
                    f"This session lasts {N_remain} timesteps, which is longer than the power profile dimension {self.var_dim_constant}."
                    "Please check the length of the sessions or make the power profile longer in the optimizer."
                )

                u_start = int(i * self.var_dim_constant)
                u_end = int(i * self.var_dim_constant + N_remain)

                constraints += [cp.sum(u[u_start:u_end]) == e_need]
                # The user is only plugged in between u_start and u_end so below,
                # so we constraint the timesteps that the user is not plug in to 0
                constraints += [u[u_end : u_start + self.var_dim_constant] == 0]

            ### Solve
            J, J_array, current_peak_sch, current_peak_reg = self.get_J(
                u,
                z,
                v,
                p_dc_sch,
                p_dc_reg,
                sub_df,
                current_time,
                running_peak,
                power_profiles,
                prices,
            )

            # Demand charge constraints
            constraints += [running_peak <= p_dc_sch]
            constraints += [running_peak <= p_dc_reg]
            constraints += [current_peak_sch <= p_dc_sch]
            constraints += [current_peak_reg <= p_dc_reg]

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