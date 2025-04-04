import numpy as np
import pandas as pd


def round_up_to_nearest_timestep(ts, delta_t):
    """
    Round pd.datetime object forward in time to the next 15-minute interval
    """
    return ts.ceil(f"{int(delta_t * 60)}min")


def convert_time_to_index(timestep, delta_t: float):
    """
    Helper function to convert a timestep to an index in the power profile

    Args:
        timestep (pd.Timestamp | pd.Timedelta): timestep or timedelta to convert
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
    """
    try:
        return int(
            np.ceil(
                (timestep.hour + timestep.minute / 60 + timestep.second / 3600)
                / delta_t
            )
        )
    except AttributeError:
        return int(np.ceil((timestep.total_seconds() / 3600) / delta_t))


def get_timestep_info(row, current_time, delta_t):
    """
    Helper function to convert information from a row in sessions_df to array indices.
    Index 0 means midnight, index 1 means 15 minutes past midnight, etc.

    Args:
        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
    """
    TOU_current_idx = convert_time_to_index(
        current_time, delta_t
    )  # current time, beginning of optimization horizon

    TOU_start_idx = convert_time_to_index(
        pd.to_datetime(row["startChargeTime"]), delta_t
    )

    TOU_end_idx = convert_time_to_index(get_end_charge_time_row(row), delta_t)

    N_remain = TOU_end_idx - TOU_current_idx  # number of timesteps remaining
    return TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain


def get_power_profile_idx(row, current_time, delta_t):
    """
    Helper function to get the current index of the power profile (i.e. how many timesteps has the EV been charging so far) \n
    INFO: this is the same as doing TOU_current_idx - TOU_start_idx

    Args:
        row: row from sessions_df
        current_time: time of optimization as a pd.datetime object
        delta_t: timestep size, in hours (e.g. 0.25 for 15-minute timesteps).
    """
    TOU_start_idx, TOU_current_idx, TOU_end_idx, N_remain = get_timestep_info(
        row, current_time, delta_t
    )
    return TOU_current_idx - TOU_start_idx


def convert_power_profile_to_df(
    power_profile: np.ndarray, start_charge_time: pd.Timestamp, delta_t: float
) -> pd.DataFrame:
    power_profile_start_time = round_up_to_nearest_timestep(start_charge_time, delta_t)
    date_index = pd.date_range(
        start=power_profile_start_time,
        periods=len(power_profile),
        freq=f"{int(delta_t * 60)}min",
    )
    power_profiles_df = pd.DataFrame({"date": date_index, "power": power_profile})
    return power_profiles_df


def get_end_charge_times(df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame, return the end charge times of each session

    Args:
        df: Must have the columns "startChargeTime" and "DurationHrs"

    """
    return df.apply(get_end_charge_time_row, axis=1)


def get_end_charge_time_row(row: pd.Series) -> pd.Timestamp:
    return (
        pd.to_datetime(row["startChargeTime"])
        + pd.to_timedelta(row["DurationHrs"], unit="h")
        # - pd.Timedelta(
        #     minutes=15
        # )  # TODO: @Sam, wy do we subtract 15 minutes here - since we are also rounding down?
    ).floor("15min")
