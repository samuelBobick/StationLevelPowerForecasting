import ast
import re

import cvxpy as cp
import numpy as np
import pandas as pd
from utils.utils_e_need import get_total_e_need
from utils.utils_time_and_indexes import (
    get_end_charge_times,
    get_timestep_info,
)


def aggregate_u_scheduled_profiles(
    u: cp.Variable, var_dim_constant: int
) -> cp.Expression:
    """Aggregate the scheduled power profiles of all users"""
    u_reshaped = cp.reshape(
        u, (u.shape[0] // var_dim_constant, var_dim_constant), order="C"
    )
    return cp.sum(u_reshaped, axis=0)  # type: ignore


def get_new_sch_obj(row, z, u, delta_t, TOU):
    """
    Helper function to generate the scheduled objective for the newest EV arrival if they choose scheduled.
    This objective is the cost of charging (TOU cost - charging revenue).
    If it is negative, the station operator is making money.

        row: row from sessions_df
        z: tuple of (p_sch, p_reg)
        u: cp.Variable for power profile
        delta_t: timestep size, in hours
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
    """
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, pd.to_datetime(row["startChargeTime"]), delta_t
    )
    next_session_profile = u[:N_remain]
    next_session_profile = cp.reshape(
        next_session_profile, (next_session_profile.shape[0],)
    )  # .T
    return (
        delta_t
        * next_session_profile
        @ (TOU[TOU_start_idx:TOU_end_idx] - z[0])  # .reshape(-1)
    )


def get_number_timesteps_for_regular(row, power_rate, delta_t, flexibility_constant):
    """
    Helper function to get the number of timesteps needed to charge the EV if the user chose regular
    """
    e_need = get_total_e_need(row, delta_t, flexibility_constant, power_rate)
    N_reg = int(
        np.round(e_need / power_rate)
    )  # how many time steps would it take the user to charge if they chose regular?
    return N_reg


def get_new_reg_obj(row, z, delta_t, TOU, power_rate, flexibility_constant):
    """
    Helper function to generate the objective for the newest EV arrival if they choose regular.
    This objective is the cost of charging (TOU cost - charging revenue).
    If it is negative, the station operator is making money.

        row: row from sessions_df
        z: tuple of (p_sch, p_reg)
        delta_t: timestep size, in hours
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


def aggregate_power_profiles(test_df, power_profiles, delta_t):
    """
    Aggregate the power profiles from a month-long simulation

        Inputs:
        test_df: the pandas DataFrame used in the simulation
        power_profiles: dictionary mapping dcosIds to power profiles
        delta_t: timestep size, in hours
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

    Inputs:
        session_results: results of the simulation by session
    """

    charging_revenue = session_results["charging_revenue"].sum()
    TOU_cost = session_results["TOU_cost"].sum()

    return charging_revenue, TOU_cost


def get_session_results(test_df, power_profiles, prices, power_rate, TOU, delta_t):
    """
    Aggregate the power profiles from a month-long simulation

    Inputs:
        test_df: the pandas DataFrame used in the simulation
        power_profiles: dictionary mapping dcosIds to power profiles
        power_rate: max power of a single EV charger, in kW
        prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
        delta_t: timestep size, in hours
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


def get_session_historical_power_profile(row) -> pd.DataFrame:
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


def get_sub_df(test_df, current_time, delta_t):
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


def get_aggregate_active_reg_future_profiles(
    test_df, current_time, power_profiles, delta_t
):
    sub_df = get_sub_df(test_df, current_time, delta_t)[:-1]
    output = np.zeros(96)

    for index, row in sub_df.iterrows():
        if row["choice"] == "REGULAR":
            TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
                row, current_time, delta_t
            )

            p = power_profiles[row["dcosId"]][(TOU_current_idx - TOU_start_idx) :]
            output += np.pad(
                p, (0, max(0, 96 - len(p))), mode="constant", constant_values=0
            )

    return output


def get_next_reg_profile(row, delta_t, flexibility_constant, power_rate):
    N_reg = get_number_timesteps_for_regular(
        row, power_rate, delta_t, flexibility_constant
    )
    return np.array([power_rate] * N_reg + [0] * (96 - N_reg))
