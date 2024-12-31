import numpy as np
import pandas as pd
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
)

from slrp_ev_ts_forecasting.default_parameters import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def apply_generate_future_session_power(
    row, raw_df_sessions: pd.DataFrame, use_all_active_sessions: bool, lookahead: int
):
    dcosId = row.name
    current_time = row["startChargeTime"]

    if use_all_active_sessions:
        data = _sum_all_active_sessions(raw_df_sessions, current_time)
    else:
        data = row["power_profiles"]

    # Truncate or pad power array to lookahead elements
    if len(data) > lookahead:
        data = data[:lookahead]
    elif len(data) < lookahead:
        data = np.pad(
            data,
            (0, lookahead - len(data)),
            mode="constant",
        )

    new_row = pd.Series(
        data=data,
        index=[f"u_{i+1}" for i in range(lookahead)],
        name=dcosId,
        dtype=float,
    )
    new_row["startChargeTime"] = current_time
    return new_row


def _sum_all_active_sessions(
    raw_df_sessions: pd.DataFrame, current_time: pd.Timestamp
) -> list:
    # We are going to replace the u features (future profile of the next session)
    # with the aggregated future power profiles of all the active sessions
    active_sessions = raw_df_sessions[
        (raw_df_sessions["startChargeTime"] <= current_time)
        & (raw_df_sessions["endChargeTime"] >= current_time)
    ]
    sum_df = pd.DataFrame()
    for _, this_session in active_sessions.iterrows():
        this_session_df = pd.DataFrame(
            data=this_session["power_profiles"], index=this_session["date"]
        )
        sum_df = pd.concat([sum_df, this_session_df], axis=1)
    sum_df = sum_df.sum(axis=1)
    sum_df = sum_df.loc[sum_df.index >= current_time]
    return sum_df.to_list()


def get_raw_df_sessions(power_df: pd.DataFrame):
    # Create a dictionary to map dcosId to their corresponding
    # power profiles in the interval data
    number_of_power_columns = power_df.filter(regex=r"power\d+").shape[1]
    session_profiles_from_interval = {}
    for i in range(1, number_of_power_columns + 1):
        dcos_column = f"dcosId{i}"
        power_column = f"power{i}"
        choice_column = f"is_choice_regular{i}"

        df_power_profiles = (
            power_df[power_df[dcos_column].notna()][
                [dcos_column, power_column, choice_column, "date"]
            ]
            .groupby(dcos_column)
            .agg(list)
        )
        df_power_profiles = df_power_profiles.rename(
            columns={
                power_column: "power_profiles",
                choice_column: "is_choice_regular",
            }
        )
        session_profiles_from_interval.update(df_power_profiles.to_dict(orient="index"))

    raw_df_sessions = pd.DataFrame(session_profiles_from_interval).T

    return raw_df_sessions


def make_artificial_sessions(
    raw_df_sessions: pd.DataFrame,
    random_start_time: bool = False,
    random_power_profile_shapes: bool = False,
    random_user_needs: bool = False,
    random_choices: bool = False,
    probability_of_scheduled_sessions: float = 0.35,
) -> pd.DataFrame:
    raw_df_sessions["choice"] = raw_df_sessions["is_choice_regular"].apply(
        lambda x: "REGULAR" if x[0] == 1 else "SCHEDULED"
    )
    raw_df_sessions["e_need"] = raw_df_sessions["power_profiles"].apply(
        lambda x: sum(x) / 4
    )
    raw_df_sessions["duration"] = raw_df_sessions["power_profiles"].apply(len)

    if random_start_time:
        raw_df_sessions["date"] = raw_df_sessions.progress_apply(
            _apply_randomize_start_time, axis=1
        )  # type: ignore

    if random_user_needs:
        raw_df_sessions["e_need"] = raw_df_sessions["e_need"].apply(
            lambda x: x * np.random.uniform(0.8, 1.2)
        )
        # TODO: need to change the dates if we allow changing the duration
        # raw_df_sessions["duration"] = raw_df_sessions["duration"].apply(
        #     lambda x: int(x * np.random.uniform(0.8, 1.2))
        # )

    # if random_choices:
    #     # TODO: need a function to generate regular profiles
    #     raw_df_sessions["choice"] = raw_df_sessions["choice"].apply(
    #         lambda x: (
    #             "REGULAR"
    #             if np.random.uniform() > probability_of_scheduled_sessions
    #             else "SCHEDULED"
    #         )
    #     )

    if random_power_profile_shapes:
        raw_df_sessions["power_profiles"] = raw_df_sessions.apply(
            lambda row: (
                generate_random_scheduled_profile(row["duration"], row["e_need"])
                if row["choice"] == "SCHEDULED"
                else row["power_profiles"]
            ),
            axis=1,
        )

    raw_df_sessions = raw_df_sessions.drop(columns=["choice", "e_need", "duration"])
    return raw_df_sessions


def _apply_randomize_start_time(
    row, max_time_shift: pd.Timedelta = pd.Timedelta(hours=3)
) -> list[pd.Timestamp]:
    random_time_shift = np.random.randint(
        -max_time_shift.total_seconds(), max_time_shift.total_seconds()  # type: ignore
    )
    # round to the closest 15 minutes (= closest 900 seconds)
    random_time_shift = pd.Timedelta(seconds=(random_time_shift // 900) * 900)
    return (pd.Series(row["date"]) + random_time_shift).to_list()


def generate_random_scheduled_profile(
    duration, e_need, min_power=0, max_power=SINGLE_EVSE_NORMALIZATION_PARAM
):
    max_power *= 1 + np.random.uniform(-0.05, 0.03)
    # Generate random numbers
    random_numbers = np.random.uniform(min_power, max_power, duration)

    # Scale the random numbers to sum to e_need
    scale_factor = e_need / np.sum(random_numbers)
    scaled_numbers = random_numbers * scale_factor

    # Ensure the scaled numbers are within the min and max bounds
    scaled_numbers = np.clip(scaled_numbers, min_power, max_power)

    # Adjust the sum to exactly match e_need
    difference = e_need - np.sum(scaled_numbers)
    for i in range(len(scaled_numbers)):
        if difference == 0:
            break
        adjustment = min(difference, max_power - scaled_numbers[i])
        scaled_numbers[i] += adjustment
        difference -= adjustment

    # randomly shuffle the power profile
    np.random.shuffle(scaled_numbers)

    # randomly slightly change the values equal to max power
    for i in range(len(scaled_numbers)):
        if scaled_numbers[i] == max_power:
            scaled_numbers[i] *= 1 + np.random.uniform(-0.01, 0.01)

    return scaled_numbers.tolist()


def revert_power_df(raw_df_sessions):
    rows = []

    for dcosId, row in raw_df_sessions.iterrows():
        for power, date, is_choice_regular in zip(
            row["power_profiles"], row["date"], row["is_choice_regular"]
        ):
            rows.append(
                {
                    "dcosId": dcosId,
                    "power": power,
                    "is_choice_regular": is_choice_regular,
                    "date": date,
                }
            )

    # Create DataFrame from collected rows
    reverted_power_df = pd.DataFrame(rows)

    return reverted_power_df


def extract_features(reverted_power_df):
    # Set index and resample
    additional_features = (
        reverted_power_df.set_index("date")
        .resample("15min")
        .agg({"dcosId": "count", "is_choice_regular": "mean", "power": "sum"})
        .reset_index()
    )

    additional_features = additional_features.rename(
        {
            "dcosId": "numberOfActiveSessions",
            "date": "startChargeTime",
            "is_choice_regular": "fractionOfRegularSessions",
        },
        axis=1,
    )
    # normalize the number of active sessions
    additional_features["numberOfActiveSessions"] /= 8

    return additional_features
