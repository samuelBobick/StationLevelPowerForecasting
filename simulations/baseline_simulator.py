import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from utils import (
    get_e_need,
    get_new_reg_obj,
    get_new_sch_obj,
    get_timestep_info,
)


class BaselineSimulator:
    """
    A class to replay the optimization of SLRP-EV sessions.
    """

    # TODO: I actually think it would be better to have the parameters listed as class inputs rather than kwargs
    # (so that we know what the parameters are without having to look at the __init__ method)
    # Is there any specific reason why you chose to use kwargs?
    def __init__(self, test_df, **kwargs):
        """
        Initialize the BaselineSimulator with default or user-defined parameters.

        Args:
            **kwargs: Key-value pairs to override default parameters.
        """
        self.test_df = test_df

        # Default simulation constants
        self.var_dim_constant = kwargs.get("var_dim_constant", 96)  # 24-hour lookahead
        self.delta_t = kwargs.get("delta_t", 0.25)  # time step in hours
        self.power_rate = kwargs.get("power_rate", 6.6)  # max power in kW
        self.flexibility_constant = kwargs.get(
            "flexibility_constant", 0.57
        )  # proportion of flexibility

        # TODO: Should we add a parameter to select the tariffs? E.g. `tariff_name`?
        # link for source of rates: https://www.pge.com/tariffs/en/rate-information/electric-rates.html#accordion-a84c67dc1e-item-69d101345a
        # TOU A-10 Primary Tariff June 2023
        # self.cost_dc = kwargs.get('cost_dc', 1942)  # cents/kW
        # tou = np.ones((96,)) * 24.7  # off-peak cents/kWh
        # tou[:34] = 22.2  # off-peak
        # tou[86:] = 22.2  # 9:30pm super off-peak
        # self.TOU = kwargs.get('tou', np.concatenate([tou, tou, tou]))  # wrap around for multi-day sessions

        # Original Slrp-EV Tariffs
        # self.cost_dc = kwargs.get('cost_dc', 500)  # cents/kW
        # tou = np.ones((96,)) * 17.5  # off-peak cents/kWh
        # tou[64:84] = 36.7  # 4 pm - 9 pm peak
        # tou[36:56] = 14.9  # 9 am - 2 pm super off-peak
        # self.TOU = kwargs.get('tou', np.concatenate([tou, tou, tou]))  # wrap around for multi-day sessions

        # PGE BEV2S Secondary June 2023
        # self.cost_dc = kwargs.get("cost_dc", 191)  # cents/kW
        self.cost_dc = kwargs.get(
            "cost_dc", 500
        )  # our modification to make DC more relevant
        tou = np.ones((96,)) * 18.6  # off-peak cents/kWh
        tou[64:84] = 39.9  # 4 pm - 9 pm peak
        tou[36:56] = 16.3  # 9 am - 2 pm super off-peak
        self.TOU = kwargs.get(
            "tou", np.concatenate([tou, tou, tou])
        )  # wrap around for multi-day sessions

        # Default price grid for optimization
        # prices = kwargs.get('prices', np.arange(20, 40, 5))
        # self.tariff_grid = kwargs.get(
        #     'tariff_grid',
        #     [(z_sch, z_reg) for z_reg in prices for z_sch in prices if z_sch < z_reg]
        # )
        self.tariff_grid = [(i, 30) for i in np.arange(10, 30, 2.5)]

        # Default discrete choice model parameters
        dcm_charging_sch_params = np.array(
            [[-self.power_rate * 0.0184 / 2], [self.power_rate * 0.0184 / 2], [0], [0]]
        )
        dcm_charging_reg_params = np.array(
            [
                [self.power_rate * 0.0184 / 2],
                [-self.power_rate * 0.0184 / 2],
                [0],
                [0.341],
            ]
        )
        dcm_leaving_params = np.array([[self.power_rate * 0.005], [0], [0], [-1]])
        self.theta = kwargs.get(
            "theta",
            np.vstack(
                (
                    dcm_charging_sch_params.T,
                    dcm_charging_reg_params.T,
                    dcm_leaving_params.T,
                )
            ),
        )

        # Default simulation options
        self.monte_carlo = kwargs.get(
            "monte_carlo", False
        )  # Whether to re-evaluate choices with DCM
        self.verbose = kwargs.get("verbose", False)  # Print optimization information

    def get_J(
        self,
        u: cp.Variable,
        z: list,
        v: np.ndarray,
        p_dc_sch: cp.Variable,
        p_dc_reg: cp.Variable,
        sub_df: pd.DataFrame,
        current_time: pd.Timestamp,
        running_peak: float,
        power_profiles: dict,
        prices: dict,
    ) -> tuple:
        """
        Helper function to set up the objective function

        Inputs:
            u: cvxpy variable for power profile
            z: array where [tariff_flex, tariff_asap, tariff_overstay, leave = 1 ] (units: cents/kWh)
            v: array with softmax results [sm_c, sm_uc, sm_y] (sm_y = leave)
            p_dc_sch: cvxpy variable which represents the peak power if the most recent user chooses scheduled
            p_dc_reg: cvxpy variable which represents the peak power if the most recent user chooses regular
            sub_df: dataframe containing rows of sessions_df that represent active sessions at the time of optimization
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            power_profiles: dictionary mapping dcosIds to power_profiles
            prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        """
        num_sch_user = 0
        num_reg_user = 0

        # TODO: why only one of them in a cp.Constant?
        existing_sch_obj = cp.Constant(0)  # profit objective for existing scheduled
        existing_reg_obj = 0  # profit objective for existing regular

        for index, row in sub_df.iloc[:-1].iterrows():
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
                if (
                    len(power_profiles[row["dcosId"]]) > 0
                    and TOU_end_idx - TOU_current_idx > 0
                ):
                    existing_reg_obj += (
                        self.delta_t
                        * power_profiles[row["dcosId"]][
                            -(TOU_end_idx - TOU_current_idx) :
                        ]
                        @ (self.TOU[TOU_current_idx:TOU_end_idx] - price)
                    )
                else:
                    existing_reg_obj += (
                        self.delta_t
                        * np.array([self.power_rate] * N_remain)
                        @ (self.TOU[TOU_current_idx:TOU_end_idx] - price)
                    )
                num_reg_user += 1

        # We add the +1 is because we haven't counted the new user yet (below we imagine
        # that the new user is scheduled)
        sch_power_sum_profile = cp.reshape(
            u, (self.var_dim_constant, num_sch_user + 1)
        ).T  # Shape: (num_sch_user + 1, self.var_dim_constant)
        sch_power_sum_profile = cp.sum(
            sch_power_sum_profile, axis=0
        )  # Shape: (self.var_dim_constant,)

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

        current_peak_sch = self.power_rate * num_reg_user + cp.max(
            sch_power_sum_profile
        )
        # add the + 1 because we imagine that the new user is regular here
        # the second term is basically the max power from the current scheduled users
        # (without considering that the new user is scheduled)
        current_peak_reg = self.power_rate * (num_reg_user + 1) + cp.max(
            cp.sum(
                cp.reshape(
                    u[self.var_dim_constant :], (self.var_dim_constant, num_sch_user)
                ).T,
                axis=0,
            )
        )

        # TODO: Should this be self.cost_dc??
        COST_DC = 500
        # TODO: Should we rename J0, J1, J2 to something more descriptive?
        # J_schedule
        J0 = (
            (new_sch_obj + existing_sch_obj + existing_reg_obj)
            + cp.Constant(COST_DC) * (p_dc_sch - running_peak)
        ) * cp.Constant(v[0])
        # J_regular
        J1 = (
            (
                new_reg_obj
                + existing_sch_obj
                + existing_reg_obj
                + cp.Constant(COST_DC) * (p_dc_reg - running_peak)
            )
        ) * cp.Constant(v[1])
        # J_leave
        J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj) * cp.Constant(v[2])

        # J0 = (new_sch_obj + existing_sch_obj + existing_reg_obj) * v[0]
        # J1 = (new_reg_obj + existing_sch_obj + existing_reg_obj) * v[1]
        # J2 = (new_leave_obj + existing_sch_obj + existing_reg_obj) * v[2]

        # J_total
        J = J0 + J1 + J2

        return (
            J,
            [
                J0 / v[0],
                J1 / v[1],
                J2 / v[2],
                new_sch_obj,
                new_reg_obj,
                existing_sch_obj,
                existing_reg_obj,
                self.cost_dc * (p_dc_sch - running_peak),
            ],
            current_peak_sch,
            current_peak_reg,
        )

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

        # TODO: is this cleaner than `len(e_need_lst)`?
        num_sch_user = sub_df.loc[sub_df["choice"] == "SCHEDULED"].shape[0]

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

    def grid_search(
        self,
        current_time: pd.Timestamp,
        running_peak: float,
        power_profiles: dict,
        prices: dict,
    ) -> tuple[dict[tuple, dict], pd.DataFrame]:
        """
        Function to search over a grid of price combinations minimize charging cost.

        Inputs:
            current_time: time of optimization
            running_peak: running peak power this billing cycle
            power_profiles: dictionary mapping dcosIds to power_profiles
            prices: dictionary mapping dcodIds to (sch_price, reg_price) tuples
        """
        sub_df = self.test_df[
            pd.to_datetime(self.test_df["startChargeTime"]) <= current_time
        ]
        end_charge_times = (
            pd.to_datetime(sub_df["startChargeTime"])
            + pd.to_timedelta(sub_df["DurationHrs"], unit="h")
            - pd.Timedelta(minutes=15)
        ).dt.floor("15min")
        sub_df = sub_df[end_charge_times >= current_time]

        assert len(sub_df) > 0, "sub_df is empty, nothing to grid search over!"

        grid_search_results = {}
        for z_sch_k, z_reg_k in self.tariff_grid:
            zk = [z_sch_k, z_reg_k, 1, 1]
            vk = softmax(self.theta @ zk).reshape(
                3, 1
            )  # reshape to convert to a 3*1 matrix (initially array of 3 elements)

            (
                uk_flex,
                e_delivered,
                p_dc_sch_k,
                p_dc_reg_k,
                current_peak_sch,
                current_peak_reg,
                J,
                J_array,
            ) = self.argmin_u(
                zk, vk, sub_df, current_time, running_peak, power_profiles, prices
            )
            grid_search_results[(z_sch_k, z_reg_k)] = {
                "J": J.value[
                    0
                ],  # value of the objective function (the array is of length 1)
                "J_arr": J_array,  # values of J_0=J_schedule, J_1=J_regular, J_2=J_leave, new_sch_obj, new_reg_obj, existing_sch_obj, existing_reg_obj, dc_charge_sch
                "u": uk_flex,
                "v": vk,  # array with probabilities of each choice [sch, reg, leave]
                "p_dc_sch_k": p_dc_sch_k,
                "p_dc_reg_k": p_dc_reg_k,
                "current_peak_sch": current_peak_sch,
                "current_peak_reg": current_peak_reg,
            }

        return grid_search_results, sub_df

    def simulate(self) -> tuple[dict, dict, dict]:
        """
        Replay self.test_df and simulate the real-time optimization and control decisions
        """
        power_profiles = {c: np.array([]) for c in self.test_df["dcosId"]}
        prices = {c: () for c in self.test_df["dcosId"]}
        hourly_prices = {c: () for c in self.test_df["dcosId"]}
        running_peak = 0

        for startChargeTime in pd.to_datetime(self.test_df["startChargeTime"]):
            grid_search_results, sub_df = self.grid_search(
                startChargeTime, running_peak, power_profiles, prices
            )

            optimal_prices = min(
                grid_search_results, key=lambda k: grid_search_results[k]["J"]
            )
            min_J = grid_search_results[optimal_prices]["J"]
            min_J_arr = grid_search_results[optimal_prices]["J_arr"]
            u = grid_search_results[optimal_prices]["u"]
            v = grid_search_results[optimal_prices]["v"]
            dc_sch = grid_search_results[optimal_prices]["p_dc_sch_k"]
            dc_reg = grid_search_results[optimal_prices]["p_dc_reg_k"]
            current_peak_sch = grid_search_results[optimal_prices]["current_peak_sch"]
            current_peak_reg = grid_search_results[optimal_prices]["current_peak_reg"]

            num_sch_user = 0
            for index, row in (
                sub_df.iloc[:-1].loc[sub_df["choice"] == "SCHEDULED"].iterrows()
            ):
                TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = (
                    get_timestep_info(row, startChargeTime, self.delta_t)
                )

                adj_constant = int((num_sch_user + 1) * self.var_dim_constant)
                num_sch_user += 1
                power_profiles[row["dcosId"]][
                    (TOU_current_idx - TOU_start_idx) : (
                        TOU_current_idx - TOU_start_idx
                    )
                    + N_remain
                ] = u[adj_constant : (adj_constant + N_remain)].flatten()

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

            zk = [optimal_prices[0], optimal_prices[1], 1, 1]
            vk = softmax(self.theta @ zk).flatten()  # .reshape(3,1)
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

            if choice == "SCHEDULED":
                running_peak = max(running_peak, dc_sch)
                power_profiles[last_row["dcosId"]] = u[
                    : self.var_dim_constant
                ].flatten()
                num_sch_user += 1
            else:
                running_peak = max(running_peak, dc_reg)
                TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = (
                    get_timestep_info(last_row, startChargeTime, self.delta_t)
                )

                # TODO: Should we replace `last_row["cumEnergy_Wh"] / 1000` by e_need here?
                N_reg = (
                    last_row["cumEnergy_Wh"] / 1000 / self.power_rate / self.delta_t
                )  # how many time steps would it take the user to charge if they chose regular?
                N_reg_remainder = (
                    N_reg % 1
                )  # for that last timestep, what fraction of a timestep is charging needed to satisfy demand?
                N_reg = int(N_reg)
                power_profiles[last_row["dcosId"]] = np.zeros(self.var_dim_constant)
                power_profiles[last_row["dcosId"]][:N_reg] = np.array(
                    [self.power_rate] * N_reg
                )

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
                print("Utilities (scheduled, regular, leave):", self.theta @ zk)
                print(
                    "Optimized delivery of",
                    round(sum(u[: self.var_dim_constant])[0] * self.delta_t, 2),
                    f'kW to session #{last_row["dcosId"]}',
                )
                print("Number of active sessions:", len(sub_df))
                print(
                    "Current peak options (scheduled, regular):",
                    np.round(current_peak_sch, 2),
                    np.round(current_peak_reg, 2),
                )
                print(
                    "Running DC options (scheduled, regular):",
                    round(dc_sch, 2),
                    round(dc_reg, 2),
                )
                print("Peak thus far", round(running_peak, 2))
                print(
                    "Profit options (scheduled, regular, leave):",
                    min_J_arr[0].value,
                    (
                        min_J_arr[1]
                        if isinstance(min_J_arr[1], np.ndarray)
                        else min_J_arr[1].value
                    ),
                    (
                        min_J_arr[2]
                        if isinstance(min_J_arr[2], np.ndarray)
                        else min_J_arr[2].value
                    ),
                )
                print("new_sch_obj", min_J_arr[3].value)
                print("new_reg_obj", min_J_arr[4])
                print(
                    "existing_sch_obj",
                    (
                        min_J_arr[5]
                        if isinstance(min_J_arr[5], int)
                        else min_J_arr[5].value
                    ),
                )
                print("existing_reg_obj", min_J_arr[6])
                print("Total profit", min_J)

                # TODO: put this in a separate function? (e.g. `plot_prices_grid_profit_heatmap`)
                # Collect the data
                data = []
                for z_sch, z_reg in grid_search_results.keys():
                    J = grid_search_results[(z_sch, z_reg)]["J"]
                    data.append([z_sch, z_reg, J])

                # Create a DataFrame with the correct column names
                df = pd.DataFrame(data, columns=["z_sch", "z_reg", "Profit"])

                # Pivot the data correctly
                pivot_table = df.pivot(index="z_sch", columns="z_reg", values="Profit")

                # Plotting the heatmap
                plt.figure(figsize=(5, 4))
                sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    cbar_kws={"label": "Profit"},
                )
                plt.title("Profit Heatmap")
                plt.xlabel("z_reg")
                plt.ylabel("z_sch")
                plt.show()
                # print('Profit grid', grid_search_results[])
                # print('power_profile', u[:N_remain])
                # print('TOU slice', self.TOU[TOU_start_idx : TOU_end_idx])
                # print('')

        return power_profiles, prices, hourly_prices
