import ast
import re

import cvxpy as cp
import numpy as np
import pandas as pd
from utils.utils_e_need import get_number_timesteps_for_regular
from utils.utils_time_and_indexes import (
    get_end_charge_times,
    get_power_profile_idx,
    get_timestep_info,
)


def get_new_sch_obj(
    row: pd.Series, z: list, u: cp.Variable, delta_t: float, TOU: np.ndarray
) -> cp.Expression:
    """
    Helper function to generate the scheduled objective for the newest EV arrival if they choose scheduled.
    This objective is the cost of charging (TOU cost - charging revenue).
    If it is negative, the station operator is making money.

    Args:
        row: row from sessions_df
        z: list of (p_sch, p_reg)
        u: cp.Variable for power profile
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
    """
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, pd.to_datetime(row["startChargeTime"]), delta_t
    )
    next_session_profile = u[:N_remain]
    next_session_profile = cp.reshape(
        next_session_profile, (next_session_profile.shape[0],)
    )
    return delta_t * next_session_profile @ (TOU[TOU_start_idx:TOU_end_idx] - z[0])


def get_new_reg_obj(
    row: pd.Series,
    z: list,
    delta_t: float,
    TOU: np.ndarray,
    power_rate: float,
    flexibility_constant: float,
) -> float:
    """
    Helper function to generate the objective for the newest EV arrival if they choose regular.
    This objective is the cost of charging (TOU cost - charging revenue).
    If it is negative, the station operator is making money.

    Args:
        row: row from sessions_df
        z: tuple of (p_sch, p_reg)
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
        power_rate: max power of a single EV charger, in kW
        flexibility_constant: proportion of regular demand that a user would have demanded if they chose scheduled
    """
    # This code assumes that we know exactly how long the user will charge for regular (don't necessarily know in reality)
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, pd.to_datetime(row["startChargeTime"]), delta_t
    )

    N_reg = get_number_timesteps_for_regular(
        row, power_rate, delta_t, flexibility_constant
    )
    return delta_t * np.sum(
        power_rate * (TOU[TOU_start_idx : TOU_start_idx + N_reg] - z[1])
    )


def aggregate_power_profiles(
    test_df: pd.DataFrame, power_profiles: dict, delta_t: float
) -> np.ndarray:
    """
    Aggregate the power profiles from a month-long simulation

    Args:
        test_df: the pandas DataFrame used in the simulation
        power_profiles: dictionary mapping dcosIds to power profiles
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
    """

    filtered_power_profiles = {k: v for k, v in power_profiles.items() if len(v) > 0}
    agg_power_profile = np.zeros(int(32 * 24 / delta_t))

    for key, power_profile in filtered_power_profiles.items():
        matching_row = test_df.loc[test_df["dcosId"] == key]
        row = matching_row.squeeze()
        start_time = pd.to_datetime(row["startChargeTime"])
        start_of_month = start_time.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        # TODO: convert_timestep_to_index here
        i = int(np.ceil((start_time - start_of_month).total_seconds() / (15 * 60)))
        agg_power_profile[i : i + len(power_profile)] += power_profile

    return agg_power_profile


def get_profit(session_results: pd.DataFrame):
    """
    Aggregate the TOU cost and charging revenue from a simulation

    Args:
        session_results: results of the simulation by session, output DataFrame \
            of the function get_session_results
    """
    charging_revenue = session_results["charging_revenue"].sum()
    TOU_cost = session_results["TOU_cost"].sum()

    return charging_revenue, TOU_cost


def get_session_results(
    test_df: pd.DataFrame,
    power_profiles: dict,
    prices: dict,
    power_rate: float,
    TOU: np.ndarray,
    delta_t: float,
) -> pd.DataFrame:
    """
    Get simulation results for each session in the simulation.

    Args:
        test_df: the pandas DataFrame used in the simulation
        power_profiles: dictionary mapping dcosIds to power profiles
        power_rate: max power of a single EV charger, in kW
        prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
    """

    filtered_power_profiles = {k: v for k, v in power_profiles.items() if len(v) > 0}

    session_results = pd.DataFrame()
    for dcosId, power_profile in filtered_power_profiles.items():
        matching_row = test_df.loc[test_df["dcosId"] == dcosId]
        row = matching_row.squeeze()

        z_sch = prices[dcosId][0]
        z_reg = prices[dcosId][1]

        start_time = pd.to_datetime(row["startChargeTime"])
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            row, start_time, delta_t
        )

        if row["choice"] == "SCHEDULED":
            charging_revenue = sum(power_profile[:N_remain] * z_sch)
        else:
            charging_revenue = sum(power_profile[:N_remain] * z_reg)

        TOU_cost = sum(power_profile[:N_remain] * TOU[TOU_start_idx:TOU_end_idx])
        energy_delivered = sum(power_profile) * delta_t

        hours_if_reg = (
            energy_delivered / power_rate
        )  # how many hours would it take the user to charge if they chose regular?

        # convert the optimal prices from $/kWh to $/hour
        z_sch_hourly = float(z_sch * energy_delivered / (N_remain * delta_t))
        z_reg_hourly = z_reg * energy_delivered / (hours_if_reg)
        row_data = {
            "dcosId": dcosId,
            "choice": row["choice"],
            "z_sch": z_sch,
            "z_reg": z_reg,
            "hourly_scheduled_price": round(z_sch_hourly, 2),
            "hourly_regular_price": round(z_reg_hourly, 2),
            "start_time": start_time,
            "charging_revenue": round(charging_revenue, 1),
            "TOU_cost": round(TOU_cost, 1),
            "energy_delivered": round(energy_delivered, 1),
            "power_profile": np.round(power_profile, 2),
        }

        session_results = pd.concat(
            [session_results, pd.DataFrame([row_data])], ignore_index=True
        )

    return session_results


def unused_get_session_historical_power_profile(row: pd.Series) -> pd.DataFrame:
    """Helper function to parse the historical power profile of a session to a DataFrame.
    This is helpful to check the cumulative power consumption of a session.
    
    WARNING: Computing the cumulative power consumption of a session like that \
        does not give the same results as the column cumEnergy_Wh in the sessions_df. \
        But the value in cumEnergy_Wh makes more sense and is the one that should be used.
    """
    df_power = pd.DataFrame(columns=["timestamp", "power"])
    list_power = row["power"]
    list_power = re.sub(r"Decimal\('(\d+)'\)", r"\1", list_power)
    for i, row_power in enumerate(ast.literal_eval(list_power)):
        df_power.loc[i, "power"] = row_power["power_W"]
        df_power.loc[i, "timestamp"] = row_power["timestamp"]

    df_power["timestamp"] = pd.to_datetime(df_power["timestamp"], unit="s")
    return df_power


def get_sub_df(
    test_df: pd.DataFrame, current_time: pd.Timestamp, delta_t: float
) -> pd.DataFrame:
    """
    Given current time and the DataFrame to simulate on, return a sub DataFrame with only the active sessions
    and the current session being optimized as the last row
    """
    sub_df = test_df[pd.to_datetime(test_df["startChargeTime"]) <= current_time]
    end_charge_times = get_end_charge_times(sub_df)
    sub_df = sub_df[end_charge_times >= current_time]

    # we also want to drop the sessions that do not have any timestep remaining
    remaining_timesteps = sub_df.apply(
        lambda row: get_timestep_info(row, current_time, delta_t)[3], axis=1
    )
    sub_df = sub_df[remaining_timesteps > 0]

    return sub_df


def aggregate_u_scheduled_profiles(
    u: cp.Variable | cp.Expression, var_dim_constant: int
) -> cp.Expression:
    """Aggregate the power profiles of all scheduled users."""
    # order=C is crucial here to do the correct reshaping, see documentation of cp.reshape
    # initial shape of u: (self.var_dim_constant * (num_sch_user), 1),
    # with num_sch_user the number of scheduled users including the current one
    u_reshaped = cp.reshape(
        u, (u.shape[0] // var_dim_constant, var_dim_constant), order="C"
    )
    return cp.sum(u_reshaped, axis=0)  # type: ignore


def get_aggregate_active_reg_future_profiles(
    test_df: pd.DataFrame,
    current_time: pd.Timestamp,
    power_profiles: dict,
    delta_t: float,
) -> np.ndarray:
    """Aggregates all existing REGULAR power profiles.

    Args:
        test_df (pd.DataFrame): the pandas DataFrame used in the simulation
        current_time (pd.Timestamp): time of current optimization
        power_profiles (dict): dictionary mapping dcosIds to power profiles
        delta_t (float): timestep size, in hours (e.g. 0.25 for 15-minute timesteps).

    Returns:
        np.ndarray: aggregate power profile of all existing REGULAR profiles.
    """
    sub_df_without_last = get_sub_df(test_df, current_time, delta_t)[:-1]
    output = np.zeros(96)

    for index, row in sub_df_without_last.iterrows():
        if row["choice"] == "REGULAR":
            power_profile_current_idx = get_power_profile_idx(
                row, current_time, delta_t
            )

            p = power_profiles[row["dcosId"]][power_profile_current_idx:]
            output += np.pad(
                p, (0, max(0, 96 - len(p))), mode="constant", constant_values=0
            )

    return output


def get_next_reg_profile(
    row: pd.Series, delta_t: float, flexibility_constant: float, power_rate: float
):
    """
    Get the power profile or the session contained in row, assuming that user chooses REGULAR

    Args:
        row (pd.Series): row of test_df
        delta_t (float): timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
        flexibility_constant (float): tproportion of regular demand that a user would have demanded if they chose scheduled
        power_rate (float): max power of a single EV charger, in kW.

    Returns:
        np.ndarray: the session contained in row, assuming that user chooses REGULAR
    """
    N_reg = get_number_timesteps_for_regular(
        row, power_rate, delta_t, flexibility_constant
    )
    return np.array([power_rate] * N_reg + [0] * (96 - N_reg))
