import numpy as np
import pandas as pd


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

    raw_df_sessions["startChargeTime"] = raw_df_sessions.apply(
        lambda x: x["date"][0], axis=1
    )
    raw_df_sessions["endChargeTime"] = raw_df_sessions.apply(
        lambda x: x["date"][-1], axis=1
    )
    return raw_df_sessions


def make_artificial_sessions(
    raw_df_sessions: pd.DataFrame,
    random_start_time: bool = False,
    random_power_profile_shapes: bool = False,
    random_user_needs: bool = False,
    random_choices: bool = False,
) -> pd.DataFrame:
    if random_start_time:
        raw_df_sessions["date"] = raw_df_sessions.progress_apply(
            _apply_randomize_start_time, axis=1
        )  # type: ignore
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


def extract_features(reverted_power_df, power_df):
    reverted_power_df = reverted_power_df.drop(columns=["power"])
    # Set index and resample
    additional_features = (
        reverted_power_df.set_index("date")
        .resample("15min")
        .agg({"dcosId": "count", "is_choice_regular": "mean"})
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
