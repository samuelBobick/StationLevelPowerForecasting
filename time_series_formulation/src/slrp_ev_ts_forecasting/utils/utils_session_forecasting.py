import numpy as np
import pandas as pd
from slrp_ev_data.read_new_slrpev_data import read_new_slrpev_data


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
    data = np.pad(data, (0, max(0, lookahead - len(data))), mode="constant")[:lookahead]

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


def get_raw_df_sessions(date_column: pd.Series):
    """Create a dictionary to map dcosId to their corresponding
    power profiles in the interval data"
    """

    # start by reading the power_df with all the columns
    # and filtering it to the provided date range
    power_df = read_new_slrpev_data(keep_all_columns=True)
    power_df = power_df.loc[
        (power_df["date"].dt.date >= date_column.min().date())
        & (power_df["date"].dt.date <= date_column.max().date())
    ]

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

    return additional_features
