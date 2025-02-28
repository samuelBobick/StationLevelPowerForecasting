from typing import Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from constants.dcm import get_dcm_theta, get_dcm_v
from constants.tariffs import DICT_TARIFFS, MODIFIED_DC, TypeTariffName
from slrp_ev_data.data_utils import USAcademicHolidayCalendar
from tqdm.auto import tqdm
from utils import (
    convert_power_profile_to_df,
    get_end_charge_time_row,
    get_new_reg_obj,
    get_new_sch_obj,
    get_next_reg_profile,
    get_remaining_e_need,
    get_sub_df,
    get_timestep_info,
    get_total_e_need,
    round_up_to_nearest_timestep,
)


class BaselineSimulator:
    """
    A class to replay the optimization of SLRP-EV sessions.
    """

    def __init__(
        self,
        test_df,
        var_dim_constant: int = 96,
        delta_t: float = 0.25,
        power_rate: float = 6.6,
        flexibility_constant: float = 0.57,
        tariff_name: TypeTariffName = "BEV2S Secondary June 2023",
        custom_cost_dc: Optional[float] = MODIFIED_DC,
        monte_carlo: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the BaselineSimulator with default or user-defined parameters.

        Args:
            test_df
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
            verbose: Print optimization information. Default is False.

        """
        self.test_df = test_df

        # Default simulation constants
        self.var_dim_constant = var_dim_constant
        self.delta_t = delta_t
        self.power_rate = power_rate
        self.flexibility_constant = flexibility_constant

        # Get the tariff
        self.TOU = DICT_TARIFFS[tariff_name]["TOU"]

        self.cost_dc = DICT_TARIFFS[tariff_name]["cost_dc"]
        if custom_cost_dc:
            self.cost_dc = custom_cost_dc  # our modification to make DC more relevant

        # Default price grid for optimization
        # prices = kwargs.get('prices', np.arange(20, 40, 5))
        # self.tariff_grid = kwargs.get(
        #     'tariff_grid',
        #     [(z_sch, z_reg) for z_reg in prices for z_sch in prices if z_sch < z_reg]
        # )
        # max price currently for fast charging, we would only be competitive if we are much lower than that
        self.max_price_per_kwh = 70  # cents/kWh
        # min price for home charging is around 15 cents/kWh.
        # This price would be very competitive for public level 2 charging
        # source: https://www.caranddriver.com/news/a45036169/electric-vehicle-ev-cost-to-charge/
        self.min_price_per_kwh = 15  # cents/kWh
        prices_kwh_regular = np.arange(
            self.max_price_per_kwh - 40, self.max_price_per_kwh, 5
        )
        prices_kwh_scheduled = np.arange(
            self.min_price_per_kwh, self.max_price_per_kwh, 5
        )
        self.tariff_grid = [
            (p_sch, p_reg)
            for p_reg in prices_kwh_regular
            for p_sch in prices_kwh_scheduled
            if p_sch < p_reg
        ]

        # Default discrete choice model parameters
        self.theta = get_dcm_theta(self.power_rate)

        # Default simulation options
        self.monte_carlo = monte_carlo
        self.verbose = verbose

        start_date = min(pd.to_datetime(self.test_df["startChargeTime"]))
        start_of_month = start_date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end_of_month = (start_date + pd.offsets.MonthEnd(1)).replace(
            hour=23, minute=59, second=59
        )
        intervals = pd.date_range(start=start_of_month, end=end_of_month, freq="15min")
        self.aggregate_power_profile = pd.DataFrame({"date": intervals, "power": 0.0})
        self.power_profiles = {}

        cal = USAcademicHolidayCalendar()
        self.holidays = cal.holidays(start=start_of_month, end=end_of_month)

    def get_dc_penalty(self, current_daily_peak, running_monthly_peak) -> cp.Expression:
        # having to use cp.maximum is much slower than putting this max into inequality constraints
        return cp.Constant(self.cost_dc) * cp.maximum(
            current_daily_peak - running_monthly_peak, 0
        )

    def get_current_peak_sch(
        self,
        num_reg_user: int,
        num_sch_user: int,
        u: cp.Variable,
        time=None,
        verbose=False,
    ) -> cp.Expression:
        """Helper function to get the peak, accounting for the optimized scheduled power profiles

        Args:
            num_reg_user (int): number of regular users
            num_reg_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile

        Returns:
            cp.Expression: current scheduled peak
        """
        # We add the +1 here because we haven't counted the new user yet (below we imagine
        # that the new user is scheduled)
        # initial shape of u: (self.var_dim_constant * (num_sch_user + 1), 1). The
        # first self.var_dim_constant elements of u are for the next session, that
        # we are trying to optimize
        sch_power_sum_profile = cp.reshape(
            u, (self.var_dim_constant, num_sch_user + 1)
        ).T  # Shape: (num_sch_user + 1, self.var_dim_constant)
        sch_power_sum_profile = cp.sum(
            sch_power_sum_profile, axis=0
        )  # Shape: (self.var_dim_constant,)

        return self.power_rate * num_reg_user + cp.max(sch_power_sum_profile)

    def get_current_peak_reg(
        self, num_reg_user: int, num_sch_user: int, u: cp.Variable, time=None, row=None
    ) -> cp.Expression:
        """Helper function to get the peak, accounting for the optimized scheduled power profiles

            add the + 1 because we imagine that the new user is regular here
            the second term is basically the max power from the current scheduled users
            (without considering that the new user is scheduled)

        Args:
            num_reg_user (int): number of regular users
            num_reg_user (int): number of scheduled users
            u (cp.Variable): scheduled power profile

        Returns:
            cp.Expression: current scheduled peak
        """
        return self.power_rate * (num_reg_user + 1) + cp.max(
            cp.sum(
                cp.reshape(
                    u[self.var_dim_constant :], (self.var_dim_constant, num_sch_user)
                ).T,
                axis=0,
            )
        )

    def get_J(
        self,
        u: cp.Variable,
        z: list,
        v: np.ndarray,
        sub_df: pd.DataFrame,
        current_time: pd.Timestamp,
        running_peak: float,
        prices: dict,
    ) -> tuple:
        """
        Helper function to set up the objective function

        Inputs:
            u: cvxpy variable for power profile
            z: array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ] (units: cents/kWh)
            v: array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
            sub_df: DataFrame containing rows of sessions_df that represent active sessions at the time of optimization
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        num_sch_user = 0
        num_reg_user = 0

        existing_sch_obj = cp.Constant(0)  # profit objective for existing scheduled
        existing_reg_obj = 0  # profit objective for existing regular

        for _, row in sub_df.iloc[:-1].iterrows():
            TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                row, current_time, self.delta_t
            )

            if row["choice"] == "SCHEDULED":
                price = prices[row["dcosId"]][0]
                adj_constant = int((num_sch_user + 1) * self.var_dim_constant)
                num_sch_user += 1
                power_profile = u[adj_constant : (adj_constant + N_remain)]
                power_profile = cp.reshape(power_profile, (power_profile.shape[0],)).T
                existing_sch_obj += (
                    self.delta_t
                    * power_profile
                    @ (self.TOU[TOU_current_idx:TOU_end_idx] - price).reshape(-1)
                )
            else:  # Assumes we know exactly how long they will stay
                price = prices[row["dcosId"]][1]
                if len(self.power_profiles[row["dcosId"]]) > 0 and N_remain > 0:
                    existing_reg_obj += (
                        self.delta_t
                        * self.power_profiles[row["dcosId"]][
                            (TOU_current_idx - TOU_start_idx) : (
                                TOU_end_idx - TOU_start_idx
                            )
                        ]  # TODO: check if this is correct
                        @ (self.TOU[TOU_current_idx:TOU_end_idx] - price)
                    )
                else:
                    existing_reg_obj += (
                        self.delta_t
                        * np.array([self.power_rate] * N_remain)
                        @ (self.TOU[TOU_current_idx:TOU_end_idx] - price)
                    )
                num_reg_user += 1

        last_row = sub_df.iloc[-1]
        new_sch_obj = get_new_sch_obj(last_row, z, u, self.delta_t, self.TOU)
        new_reg_obj = get_new_reg_obj(
            last_row,
            z,
            self.delta_t,
            self.TOU,
            self.power_rate,
            self.flexibility_constant,
        )
        new_leave_obj = 0

        current_peak_sch = self.get_current_peak_sch(
            num_reg_user, num_sch_user, u, current_time
        )
        current_peak_reg = self.get_current_peak_reg(
            num_reg_user, num_sch_user, u, current_time, last_row
        )

        J_scheduled = (
            (new_sch_obj + existing_sch_obj + existing_reg_obj)
            + self.get_dc_penalty(current_peak_sch, running_peak)
        ) * cp.Constant(v[0])
        J_regular = (
            (
                new_reg_obj
                + existing_sch_obj
                + existing_reg_obj
                + self.get_dc_penalty(current_peak_reg, running_peak)
            )
        ) * cp.Constant(v[1])
        J_leave = (new_leave_obj + existing_sch_obj + existing_reg_obj) * cp.Constant(
            v[2]
        )

        # J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj) * v[0]
        # J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj) * v[1]
        # J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj) * v[2]

        J_total = J_scheduled + J_regular + J_leave

        return (
            J_total,
            [
                J_scheduled / v[0],
                J_regular / v[1],
                J_leave / v[2],
                new_sch_obj,
                new_reg_obj,
                existing_sch_obj,
                existing_reg_obj,
            ],
            current_peak_sch,
            current_peak_reg,
        )

    def initialize_problem(self, z, v, sub_df, current_time, running_peak, prices):
        """Helper function to return the cvxpy variables to solve the optimization problem.

        Inputs:
            z: array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1]
            v: array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
            sub_df: dataframe containing rows of sessions_df that represent active \
                sessions at the time of optimization
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """

        e_need_lst = []
        N_remain_lst = []

        last_row = sub_df.iloc[-1]
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            last_row, current_time, self.delta_t
        )
        e_need = get_total_e_need(
            last_row, self.delta_t, self.flexibility_constant, self.power_rate
        )
        e_need_lst.append(e_need)
        N_remain_lst.append(N_remain)

        # Iter through all of the active scheduled sessions (except the new one)
        for index, row in (
            sub_df.iloc[:-1].loc[sub_df["choice"] == "SCHEDULED"].iterrows()
        ):
            TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                row, current_time, self.delta_t
            )

            e_need = get_remaining_e_need(
                row,
                current_time,
                self.power_profiles,
                self.delta_t,
                self.power_rate,
                self.flexibility_constant,
            )
            e_need_lst.append(e_need)
            N_remain_lst.append(N_remain)

        num_sch_user = len(e_need_lst)  # This considers the next user is scheduled

        ### Decision Variables
        e_delivered = cp.Variable(
            shape=((self.var_dim_constant + 1) * num_sch_user, 1)
        )  # energy delivered
        u = cp.Variable(
            shape=(self.var_dim_constant * num_sch_user, 1)
        )  # charging profile (extra scheduled user profile included in case new user chooses scheduled

        ### Constraints incorporate all SCH users
        constraints = [u >= 0, u <= self.power_rate]

        # Iterate through all existing scheduled users (considering the next one is also scheduled)
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
            sub_df,
            current_time,
            running_peak,
            prices,
        )

        return (
            u,
            e_delivered,
            J,
            J_array,
            current_peak_sch,
            current_peak_reg,
            constraints,
        )

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
            sub_df: DataFrame containing rows of sessions_df that represent active \
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

        obj = cp.Minimize(J)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5)
        if prob.status == "optimal_inaccurate":
            # TODO: look into why this is happening
            print("WARNING: optimal solution found, but is inaccurate")
        elif prob.status != "optimal":
            raise Exception(f"Optimization failed with status {prob.status}")

        return (
            u,
            e_delivered.value,
            current_peak_sch.value,
            current_peak_reg.value,
            J,
            J_array,
        )

    def grid_search(
        self,
        current_time: pd.Timestamp,
        running_peak: float,
        prices: dict,
    ) -> tuple[dict[tuple, dict], pd.DataFrame]:
        """
        Function to search over a grid of price combinations minimize charging cost.

        Inputs:
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        sub_df = get_sub_df(self.test_df, current_time, self.delta_t)

        assert len(sub_df) > 0, "sub_df is empty, nothing to grid search over!"

        # TODO: Grid search below can be parallelized
        grid_search_results = {}
        for z_sch_k, z_reg_k in self.tariff_grid:
            zk = [z_sch_k, z_reg_k, 1, 1]
            vk = get_dcm_v(zk, self.theta)

            (
                uk_flex,
                e_delivered,
                current_peak_sch,
                current_peak_reg,
                J,
                J_array,
            ) = self.argmin_u(zk, vk, sub_df, current_time, running_peak, prices)
            grid_search_results[(z_sch_k, z_reg_k)] = {
                "J": J.value[0] if isinstance(J.value, np.ndarray) else J.value,
                # value of the objective function (the array is of length 1)
                "J_arr": J_array,  # values of J_0=J_schedule, J_1=J_regular, J_2=J_leave, new_sch_obj, new_reg_obj, existing_sch_obj, existing_reg_obj, dc_charge_sch
                "u": uk_flex,
                "v": vk,  # array with probabilities of each choice [sch, reg, leave]
                "current_peak_sch": current_peak_sch,
                "current_peak_reg": current_peak_reg,
            }

        return grid_search_results, sub_df

    def update_aggregate_power_profile(
        self, previousStartChargeTime, startChargeTime, active_sessions
    ):
        """_summary_

        Args:
            previousStartChargeTime (_type_): _description_
            startChargeTime (_type_): _description_
            active_sessions (_type_): _description_
        """
        for row in active_sessions:
            power_profile = self.power_profiles[row["dcosId"]]

            power_profile_df = convert_power_profile_to_df(
                power_profile, pd.to_datetime(row["startChargeTime"]), self.delta_t
            )

            end_charge_time = get_end_charge_time_row(row)

            rounded_prev_time = round_up_to_nearest_timestep(
                previousStartChargeTime, self.delta_t
            )
            rounded_current_time = round_up_to_nearest_timestep(
                startChargeTime, self.delta_t
            )

            filtered_power_profile_df = power_profile_df[
                (power_profile_df["date"] >= rounded_prev_time)
                & (power_profile_df["date"] < rounded_current_time)
            ]
            self.aggregate_power_profile.loc[
                self.aggregate_power_profile["date"].isin(
                    filtered_power_profile_df["date"]
                ),
                "power",
            ] += filtered_power_profile_df["power"].values

            if end_charge_time < startChargeTime:
                active_sessions = [
                    a for a in active_sessions if a["dcosId"] != row["dcosId"]
                ]

        # get the row of the new (next) session
        new_session: pd.Series = self.test_df[
            pd.to_datetime(self.test_df["startChargeTime"]) == startChargeTime
        ].iloc[0]
        # the following line modifies the active_sessions list in place
        active_sessions.append(new_session)

        return active_sessions

    def simulate(self) -> tuple[dict, dict, dict, dict]:
        """
        Replay self.test_df and simulate the real-time optimization and control decisions
        """
        self.power_profiles = {c: np.array([]) for c in self.test_df["dcosId"]}
        prices = {c: (None, None) for c in self.test_df["dcosId"]}
        hourly_prices = {c: (None, None) for c in self.test_df["dcosId"]}
        user_computed_data_for_visualization = {c: {} for c in self.test_df["dcosId"]}
        # active_sessions = [{dcosId : start_time},......] TODO
        active_sessions: list[pd.Series] = []

        previousStartChargeTime = None
        u = None
        for startChargeTime in tqdm(
            pd.to_datetime(self.test_df["startChargeTime"]), desc="Optimizing sessions"
        ):
            # Update the aggregate power profile and the running peak
            active_sessions = self.update_aggregate_power_profile(
                previousStartChargeTime,
                startChargeTime,
                active_sessions,
            )
            # only used for timeseries_forecast TODO should we use self.power_profiles instead, using Thibaud's new utils function.
            timeseries_forecast = self.get_timeseries_forecast(
                active_sessions[-1],
                u,
                previousStartChargeTime,
                len(active_sessions),
                startChargeTime,
            )

            running_peak = self.aggregate_power_profile["power"].max()

            grid_search_results, sub_df = self.grid_search(
                startChargeTime, running_peak, prices
            )

            # Retrieve info for the optimal prices
            optimal_prices = min(
                grid_search_results, key=lambda k: grid_search_results[k]["J"]
            )
            min_J = grid_search_results[optimal_prices]["J"]
            min_J_arr = grid_search_results[optimal_prices]["J_arr"]
            u_cvxpy = grid_search_results[optimal_prices]["u"]
            u = u_cvxpy.value
            v = grid_search_results[optimal_prices]["v"]
            current_peak_sch = grid_search_results[optimal_prices][
                "current_peak_sch"
            ].item()
            current_peak_reg = grid_search_results[optimal_prices][
                "current_peak_reg"
            ].item()

            # update the power profiles of other active scheduled users
            num_sch_user = 0
            for index, row in (
                sub_df.iloc[:-1].loc[sub_df["choice"] == "SCHEDULED"].iterrows()
            ):
                TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = (
                    get_timestep_info(row, startChargeTime, self.delta_t)
                )

                adj_constant = int((num_sch_user + 1) * self.var_dim_constant)
                num_sch_user += 1
                self.power_profiles[row["dcosId"]][
                    (TOU_current_idx - TOU_start_idx) : (
                        TOU_current_idx - TOU_start_idx
                    )
                    + N_remain
                ] = u[adj_constant : (adj_constant + N_remain)].flatten()

            # Save hourly prices and choice of the new user
            last_row = sub_df.iloc[-1]
            prices[last_row["dcosId"]] = optimal_prices

            TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                last_row, pd.to_datetime(last_row["startChargeTime"]), self.delta_t
            )
            e_need = round(sum(u[: self.var_dim_constant])[0] * self.delta_t, 2)
            hours_if_reg = (
                e_need / self.power_rate
            )  # how many time steps would it take the user to charge if they chose regular?
            # convert the optimal prices from $/kWh to $/hour
            hourly_optimal_prices = (
                optimal_prices[0] * e_need / (N_remain * self.delta_t),
                optimal_prices[1] * e_need / (hours_if_reg),
            )
            hourly_prices[last_row["dcosId"]] = hourly_optimal_prices

            # Simulate next user's choice and update test_df if needed
            zk = [optimal_prices[0], optimal_prices[1], 1, 1]
            vk = get_dcm_v(zk, self.theta)
            if self.monte_carlo:
                normalized_probs = (
                    vk[:2] / vk[:2].sum()
                )  # only simulate sch/reg choices, no leaving in the simulation
                choice = np.random.choice(["SCHEDULED", "REGULAR"], p=normalized_probs)
                self.test_df.loc[
                    self.test_df["dcosId"] == last_row["dcosId"], "choice"
                ] = choice
            else:
                choice = last_row["choice"]

            # Save the power profile of the new user
            if choice == "SCHEDULED":
                self.power_profiles[last_row["dcosId"]] = u[
                    : self.var_dim_constant
                ].flatten()
                num_sch_user += 1
            else:
                self.power_profiles[last_row["dcosId"]] = get_next_reg_profile(
                    last_row, self.delta_t, self.flexibility_constant, self.power_rate
                )

            # Save other user data for visualization
            user_computed_data_for_visualization[last_row["dcosId"]] = {
                "Start charge time": startChargeTime,
                "Choice": choice,
                "Energy needed": e_need,
                "Duration (hours)": last_row["DurationHrs"],
            }

            previous_running_peak = running_peak
            previousStartChargeTime = startChargeTime

            if self.verbose:
                print(
                    "---------------------------------------------------------------------"
                )

                print("Done with optimization at", startChargeTime)
                print(
                    "Optimal prices (per kW; p_scheduled, p_regular):", optimal_prices
                )
                print(
                    "Optimal prices (per hour; p_scheduled, p_regular):",
                    hourly_optimal_prices,
                )
                print("Probabilities (scheduled, regular, leave):", vk)
                print("Selected choice:", choice)
                print("Utilities (scheduled, regular, leave):", self.theta @ zk)
                print(
                    "Optimized delivery of",
                    e_need,
                    f'kWh to session #{last_row["dcosId"]}',
                )
                print("Number of active sessions:", len(sub_df))
                print("Running peak thus far", round(running_peak, 2))
                print(
                    "Predicted peak options (scheduled, regular):",
                    round(current_peak_sch, 2),
                    round(current_peak_reg, 2),
                )
                print(
                    "Profit options (scheduled, regular, leave):",
                    np.round(min_J_arr[0].value),
                    (
                        min_J_arr[1]
                        if isinstance(min_J_arr[1], np.ndarray)
                        else np.round(min_J_arr[1].value)
                    ),
                    (
                        min_J_arr[2]
                        if isinstance(min_J_arr[2], np.ndarray)
                        else np.round(min_J_arr[2].value)
                    ),
                )
                print("If scheduled user:")
                print("  new_sch_obj (TOU-EV revenue) =", np.round(min_J_arr[3].value))
                print(
                    "  Demand charge penalty sch =",
                    np.round(
                        self.get_dc_penalty(
                            current_peak_sch, previous_running_peak
                        ).value
                    ),
                )
                print("If regular user:")
                print("  new_reg_obj (TOU-EV revenue) =", np.round(min_J_arr[4]))
                print(
                    "  Demand charge penalty reg =",
                    np.round(
                        self.get_dc_penalty(
                            current_peak_reg, previous_running_peak
                        ).value
                    ),
                )
                print(
                    "existing_sch_obj",
                    (
                        min_J_arr[5]
                        if isinstance(min_J_arr[5], int)
                        else np.round(min_J_arr[5].value)
                    ),
                )
                print("existing_reg_obj", np.round(min_J_arr[6]))
                print("Total daily cost so far", np.round(min_J))

                # visualize the predictions for peak_simulator
                num_reg_user_without_next = (
                    sub_df.iloc[:-1].loc[sub_df["choice"] == "REGULAR"].shape[0]
                )
                num_sch_user_without_next = num_sch_user - (choice == "SCHEDULED")
                self.get_current_peak_sch(
                    num_reg_user_without_next,
                    num_sch_user_without_next,
                    u_cvxpy,
                    startChargeTime,
                    verbose=True,
                )

                plot_prices_grid_profit_heatmap(grid_search_results)

        return (
            self.power_profiles,
            prices,
            hourly_prices,
            user_computed_data_for_visualization,
        )

    def get_timeseries(self, row, num_active_sessions, time, verbose=False):
        # This function is not implemented in the baseline simulator
        pass

    def get_timeseries_forecast(
        self, current_row, u, prev_start_charge_time, num_active_sessions, time
    ):
        # This function is not implemented in the baseline simulator
        pass


def plot_prices_grid_profit_heatmap(grid_search_results: dict):
    """
    Function to plot a heatmap of the profit for each price combination

    Inputs:
        grid_search_results: dictionary containing the results of the grid search
    """
    # Collect the data
    data = []
    for z_sch, z_reg in grid_search_results.keys():
        J = grid_search_results[(z_sch, z_reg)]["J"]
        data.append([z_sch, z_reg, J])

    # Create a DataFrame with the correct column names
    df = pd.DataFrame(data, columns=["z_sch", "z_reg", "Cost"])

    # Pivot the data correctly
    pivot_table = df.pivot(index="z_sch", columns="z_reg", values="Cost")

    # Plotting the heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar_kws={"label": "Cost"},
    )
    plt.title("Profit Heatmap")
    plt.xlabel("z_reg ($/kWh)")
    plt.ylabel("z_sch ($/kWh)")
    plt.show()
