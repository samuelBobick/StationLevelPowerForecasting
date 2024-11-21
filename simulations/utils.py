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
    TOU_end_idx = int(
        np.floor(TOU_start_idx + row["DurationHrs"] / delta_t)
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

    if row["choice"] == "SCHEDULED" and not pd.isna(row["energyReq_Wh"]):
        e_need = row["energyReq_Wh"] / 1000 / delta_t
    elif row["choice"] == "SCHEDULED":
        e_need = flexibility_constant * row["cumEnergy_Wh"] / 1000 / delta_t
    else:
        e_need = row["cumEnergy_Wh"] / 1000 / delta_t

    N_reg = int(
        e_need // power_rate
    )  # how many time steps would it take the user to charge if they chose regular?
    return delta_t * np.sum(
        power_rate * (TOU[TOU_start_idx : TOU_start_idx + N_reg] - z[1])
    )


def get_power_profile_idx(row, current_time, delta_t):
    """
    Helper function to get the current index of the power profile (i.e. how many timeseteps has the EV been charging so far)

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


def get_e_need(
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
    if row["choice"] == "SCHEDULED" and not pd.isna(row["energyReq_Wh"]):
        e_need = row["energyReq_Wh"] / 1000 / delta_t
    elif row["choice"] == "SCHEDULED":
        e_need = flexibility_constant * row["cumEnergy_Wh"] / 1000 / delta_t
    else:
        e_need = row["cumEnergy_Wh"] / 1000 / delta_t

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
        prices: dictionary mapping dcodIds to (sch_price, reg_price) tuples
        delta_t: timestep size, in hours
        TOU: electricity price time series, with TOU[0] representing the price at midnight, in units of cents/kWh
    """

    profit = 0
    for index, row in test_df.iterrows():
        current_time = pd.to_datetime(row["startChargeTime"])
        TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
            row, current_time, delta_t
        )
        power_profile = power_profiles[row["dcosId"]]
        if row["choice"] == "SCHEDULED":
            profit += np.sum(
                power_profile[:N_remain]
                * (prices[row["dcosId"]][0] - TOU[TOU_start_idx:TOU_end_idx])
            )
        else:
            profit += np.sum(
                power_profile[:N_remain]
                * (prices[row["dcosId"]][1] - TOU[TOU_start_idx:TOU_end_idx])
            )

    return profit
