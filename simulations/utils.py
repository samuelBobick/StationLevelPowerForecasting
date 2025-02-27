import ast
import re

import cvxpy as cp
import numpy as np
import pandas as pd


def get_timestep_info(row, current_time, delta_t):
    """
    Helper function to convert information from a row in sessions_df to array indices

        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        delta_t: timestep size, in hours
    """
    TOU_current_idx = int(
        np.ceil((current_time.hour + current_time.minute / 60) / delta_t)
    )  # current time, beginning of optimization horizon
    start_time = pd.to_datetime(row["startChargeTime"])
    TOU_start_idx = int(np.ceil((start_time.hour + start_time.minute / 60) / delta_t))
    # TODO: change np.floor to np.ceil below here, because ran into an issue
    # if N_remain is 0
    TOU_end_idx = int(
        np.ceil(TOU_start_idx + row["DurationHrs"] / delta_t)
    )  # end time index
    N_remain = TOU_end_idx - TOU_current_idx  # number of timesteps remaining
    return TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain


def get_new_sch_obj(row, z, u, delta_t, TOU):
    """
    Helper function to generate the scheduled objective for the newest EV arrival if they choose scheduled.

        row: row from sessions_df
        z: tuple of (p_sch, p_reg)
        u: cp.Variable for power profile
        delta_t: timestep size, in hours
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
    """
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, pd.to_datetime(row["startChargeTime"]), delta_t
    )
    power_profile = u[:N_remain]
    power_profile = cp.reshape(power_profile, (power_profile.shape[0],)).T
    return delta_t * power_profile @ (TOU[TOU_start_idx:TOU_end_idx] - z[0]).reshape(-1)


def get_new_reg_obj(row, z, delta_t, TOU, power_rate, flexibility_constant):
    """
    Helper function to generate the objective for the newest EV arrival if they choose regular.

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

    e_need = get_total_e_need(row, delta_t, flexibility_constant)

    N_reg = int(
        e_need // power_rate
    )  # how many time steps would it take the user to charge if they chose regular?
    return delta_t * np.sum(
        power_rate * (TOU[TOU_start_idx : TOU_start_idx + N_reg] - z[1])
    )


def get_power_profile_idx(row, current_time, delta_t):
    """
    Helper function to get the current index of the power profile (i.e. how many timesteps has the EV been charging so far)

        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        delta_t: timestep size, in hours
    """
    start_time = pd.to_datetime(row["startChargeTime"])
    current_time = (current_time + pd.Timedelta(minutes=15)).floor("15min")
    power_profile_current_idx = int(
        np.ceil((current_time - start_time).total_seconds() / 3600 / delta_t)
    )
    return power_profile_current_idx


def get_remaining_e_need(
    row, current_time, power_profiles, delta_t, power_rate, flexibility_constant
):
    """
    Helper function to calculate the energy demand of a particular session.

        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        power_profiles: dictionary mapping dcosIds to power_profiles
        delta_t: timestep size, in hours
        power_rate: max power of a single EV charger, in kW
        flexibility_constant: proportion of regular demand that a user would have demanded if they chose scheduled
    """
    e_need = get_total_e_need(row, delta_t, flexibility_constant)

    power_profile_current_idx = get_power_profile_idx(row, current_time, delta_t)
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, current_time, delta_t
    )

    if len(power_profiles[row["dcosId"]]) > 0:
        e_need -= sum(
            power_profiles[row["dcosId"]][:power_profile_current_idx]
        )  # if user has already consumed some power, subtract it from their demand
        if (
            e_need < 0
        ):  # handle numerical imprecision and set completed charging sessions to exactly zero
            e_need = 0
    if e_need > N_remain * power_rate:  # if user requests an infeasible amount of power
        e_need = N_remain * power_rate

    return e_need


def get_total_e_need(row, delta_t, flexibility_constant):
    if row["choice"] == "SCHEDULED" and not pd.isna(row["energyReq_Wh"]):
        # Case where the session was really scheduled and we have the energy needed
        # for the session given by the user
        e_need = row["energyReq_Wh"] / 1000 / delta_t
    elif row["choice"] == "SCHEDULED":
        # case for when we consider the session scheduled but the real one was regular
        # so we have to estimate the energy needed
        e_need = flexibility_constant * row["cumEnergy_Wh"] / 1000 / delta_t
    else:
        e_need = row["cumEnergy_Wh"] / 1000 / delta_t
    return e_need


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
        i = int(np.ceil((start_time - start_of_month).total_seconds() / (15 * 60)))
        agg_power_profile[i : i + len(power_profile)] += power_profile

    return agg_power_profile


def get_profit(test_df, power_profiles, prices, delta_t, TOU):
    """
    Aggregate the profit from a simulation

        Inputs:
        test_df: the pandas DataFrame used in the simulation
        power_profiles: dictionary mapping dcosIds to power_profiles
        prices: dictionary mapping dcosIds to (sch_price, reg_price) tuples
        delta_t: timestep size, in hours
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
    """

    charging_revenue = 0
    TOU_costs = 0
    for index, row in test_df.iterrows():
        current_time = pd.to_datetime(row["startChargeTime"])
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            row, current_time, delta_t
        )
        power_profile = power_profiles[row["dcosId"]]
        if row["choice"] == "SCHEDULED":
            charging_revenue += sum(power_profile[:N_remain] * prices[row["dcosId"]][0])
        else:
            charging_revenue += sum(power_profile[:N_remain] * prices[row["dcosId"]][1])

        TOU_costs += sum(power_profile[:N_remain] * TOU[TOU_start_idx:TOU_end_idx])

    return charging_revenue, TOU_costs


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
    agg_power_profile = np.zeros(int(32 * 24 / delta_t))

    df = pd.DataFrame()
    for dcosId, power_profile in filtered_power_profiles.items():
        matching_row = test_df.loc[test_df["dcosId"] == dcosId]
        row = matching_row.squeeze()
        start_time = pd.to_datetime(row["startChargeTime"])
        start_of_month = start_time.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        i = int(np.ceil((start_time - start_of_month).total_seconds() / (15 * 60)))
        agg_power_profile[i : i + len(power_profile)] += power_profile

        z_sch = prices[dcosId][0]
        z_reg = prices[dcosId][1]

        start_time = pd.to_datetime(row["startChargeTime"])
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            row, start_time, delta_t
        )

        power_profile = power_profiles[row["dcosId"]]
        if row["choice"] == "SCHEDULED":
            charging_revenue = sum(power_profile[:N_remain] * z_sch)
        else:
            charging_revenue = sum(power_profile[:N_remain] * z_reg)

        TOU_cost = sum(power_profile[:N_remain] * TOU[TOU_start_idx:TOU_end_idx])
        energy_delivered = sum(power_profile) * delta_t

        hours_if_reg = (
            energy_delivered / power_rate
        )  # how many time steps would it take the user to charge if they chose regular?

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

        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    return df


def get_session_power_profile(row) -> pd.DataFrame:
    """Helper function to parse the power profile of a session to a DataFrame.
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


def round_up_to_nearest_timestep(ts, delta_t):
    """
    Round pd.datetime object forward in time to the next 15-minute interval
    """
    round_interval = 60 * delta_t
    # Find the number of seconds since the last 15-minute interval
    seconds_to_next = (round_interval * 60) - (
        ts.minute % round_interval * 60 + ts.second
    )

    # Add the remaining seconds to the original timestamp
    rounded_ts = ts + pd.Timedelta(seconds=seconds_to_next)
    return rounded_ts.replace(second=0, microsecond=0)


def get_sub_df(test_df, current_time):
    """
    Given current time and the DataFrame to simulate on, return a sub DataFrame with only the active sessions
    """
    sub_df = test_df[pd.to_datetime(test_df["startChargeTime"]) <= current_time]
    end_charge_times = (
        pd.to_datetime(sub_df["startChargeTime"])
        + pd.to_timedelta(sub_df["DurationHrs"], unit="h")
        - pd.Timedelta(minutes=15)
    ).dt.floor("15min")
    sub_df = sub_df[end_charge_times >= current_time]

    return sub_df


def get_aggregate_reg_profiles(test_df, current_time, power_profiles, delta_t):
    sub_df = get_sub_df(test_df, current_time)[:-1]
    output = np.zeros(96)

    for index, row in sub_df.iterrows():
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            row, pd.to_datetime(row["startChargeTime"]), delta_t
        )

        p = power_profiles[row["dcosId"]][TOU_current_idx:]
        output += np.pad(
            p, (0, max(0, 96 - len(p))), mode="constant", constant_values=0
        )

    return output


def get_next_reg_profile(row, delta_t, flexibility_constant, power_rate):
    e_need = get_total_e_need(row, delta_t, flexibility_constant)
    N_reg = int(
        e_need // power_rate
    )  # how many time steps would it take the user to charge if they chose regular?
    return np.array([power_rate] * N_reg + [0] * (96 - N_reg))
