import pandas as pd
from utils.utils_time_and_indexes import get_timestep_info


def get_total_e_need(row, delta_t, flexibility_constant, power_rate):
    """Returns the total energy need, DIVIDED BY delta_t (in hour, e.g. 0.25)
    This e_need represent more the power needed at each timestep than the
    total energy needed for the session.
    """
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

    # make sure that the energy need is not infeasible
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, pd.to_datetime(row["startChargeTime"]), delta_t
    )
    if e_need > N_remain * power_rate:
        e_need = N_remain * power_rate

    return e_need


def get_remaining_e_need(
    row, current_time, power_profiles, delta_t, power_rate, flexibility_constant
):
    """
    Helper function to calculate the energy demand of a particular session.
    This e_need represent more the power needed at each timestep than the
    total energy needed for the session.

        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        power_profiles: dictionary mapping dcosIds to power_profiles
        delta_t: timestep size, in hours
        power_rate: max power of a single EV charger, in kW
        flexibility_constant: proportion of regular demand that a user would have demanded if they chose scheduled
    """
    e_need = get_total_e_need(row, delta_t, flexibility_constant, power_rate)

    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, current_time, delta_t
    )
    # power_profile_current_idx = get_power_profile_idx(row, current_time, delta_t)
    power_profile_current_idx = TOU_current_idx - TOU_start_idx

    if len(power_profiles[row["dcosId"]]) > 0:
        e_need -= sum(power_profiles[row["dcosId"]][:power_profile_current_idx])

        # if user has already consumed some power, subtract it from their demand
        if (
            e_need < 0
        ):  # handle numerical imprecision and set completed charging sessions to exactly zero
            e_need = 0

        # Check if user requests an infeasible amount of power.
        if e_need > N_remain * power_rate:
            # If the error is not too big, it is probably due to numerical imprecision
            # and we can set the demand to the maximum possible.
            # otherwise, we raise an error
            if e_need > N_remain * power_rate + 1:
                raise ValueError(
                    f"Remaining energy is infeasible e_need: {e_need}, "
                    f"max energy possible: {N_remain * power_rate}"
                )
            else:
                e_need = N_remain * power_rate

    return e_need
