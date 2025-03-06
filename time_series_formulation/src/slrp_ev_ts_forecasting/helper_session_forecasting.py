import numpy as np
import pandas as pd
from slrp_ev_data.feature_engineering import (
    convert_date_from_int_to_datetime,
    feature_engineering,
)
from slrp_ev_data.normalization_and_standardization import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
    get_scaling_parameters,
)
from slrp_ev_data.read_new_slrpev_data import read_new_slrpev_data

from slrp_ev_ts_forecasting.default_parameters import RANDOM_SEED, TypeScalingMode


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


def get_artificial_data(
    train_data: pd.DataFrame,
    random_start_time: bool,
    shuffle_power_profiles: bool,
    random_power_profile_shapes: bool,
    random_user_needs: bool,
    random_choices: bool,
    scaling_mode: TypeScalingMode,
    lookahead: int,
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED),
) -> tuple[pd.DataFrame, pd.DataFrame, tuple | pd.DataFrame]:
    train_data = train_data.copy()
    train_data["date"] = convert_date_from_int_to_datetime(train_data["date"])

    power_df = read_new_slrpev_data(keep_all_columns=True)

    power_df = power_df.loc[
        (power_df["date"].dt.date >= train_data["date"].min().date())
        & (power_df["date"].dt.date <= train_data["date"].max().date())
    ]

    raw_df_sessions = get_raw_df_sessions(power_df)

    artificial_raw_df_sessions = make_artificial_sessions(
        raw_df_sessions,
        random_start_time=random_start_time,
        shuffle_power_profiles=shuffle_power_profiles,
        random_power_profile_shapes=random_power_profile_shapes,
        random_user_needs=random_user_needs,
        random_choices=random_choices,
        rng=rng,
    )

    artificial_power_df = extract_features(
        revert_power_df(artificial_raw_df_sessions)
    ).drop(columns=["numberOfActiveSessions", "fractionOfRegularSessions"])

    artificial_power_df = artificial_power_df.merge(
        train_data[["date", "power", "number_of_evses_available"]],
        left_on="startChargeTime",
        right_on="date",
        how="right",
        suffixes=(None, "_initial"),
    )
    artificial_power_df.loc[artificial_power_df["startChargeTime"].isna(), "power"] = 0
    # Add back the missing values where they were in the original data
    artificial_power_df.loc[
        artificial_power_df["power_initial"].isna()
        # & (artificial_power_df["power"] == 0)
        ,
        "power",
    ] = np.nan

    artificial_power_df = artificial_power_df.drop(
        columns=["power_initial", "startChargeTime"]
    )
    artificial_power_df = artificial_power_df.dropna(subset=["power"])
    scaling_parameters = get_scaling_parameters(
        artificial_power_df,
        artificial_power_df,
        data_scaling_mode=scaling_mode,
        lookahead_15min_steps=lookahead,
        dataset="slrp-ev_new",
        retrieve_from_saved=scaling_mode in ["normalize", "standardize"],
    )
    artificial_train_data = feature_engineering(
        data_input=artificial_power_df,
        add_nans_for_missing_data=True,
        scaling_mode=scaling_mode,
        scaling_parameters=scaling_parameters,
        cols_normalization_to_skip=["number_of_evses_available"],
    )
    return artificial_train_data, artificial_raw_df_sessions, scaling_parameters


def make_artificial_sessions(
    raw_df_sessions: pd.DataFrame,
    random_start_time: bool,
    shuffle_power_profiles: bool,
    random_power_profile_shapes: bool,
    random_user_needs: bool,
    random_choices: bool,
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED),
    probability_of_scheduled_sessions: float = 0.35,
) -> pd.DataFrame:
    """Make an artificial session dataset, randomizing different things in the original
    raw_df_sessions dataset.

    Args:
        raw_df_sessions (pd.DataFrame): dataset of sessions, obtained from the function get_raw_df_sessions
        random_start_time (bool): If True, the start times will be randomly changed by +/- 3 hours
        shuffle_power_profiles (bool): If True, the power profiles will be shuffled with other similar sessions
        random_power_profile_shapes (bool): If True, the power profiles will be generated randomly
        random_user_needs (bool): If True, the energy needs will be randomly changed by +/- 20%
        random_choices (bool): If True, the choice of the session will be randomly changed
        rng(np.random.Generator): Random number generator, in case you want repeatable results
        probability_of_scheduled_sessions (float, optional): _description_. Defaults to 0.35.

    Returns:
        pd.DataFrame: _description_
    """
    raw_df_sessions = _compute_user_needs(raw_df_sessions)
    initial_raw_df_sessions = raw_df_sessions.copy()

    if random_start_time:
        raw_df_sessions["date"] = raw_df_sessions.apply(
            _apply_randomize_start_time, axis=1, rng=rng
        )  # type: ignore

    if random_user_needs:
        raw_df_sessions["e_need"] = raw_df_sessions["e_need"].apply(
            lambda x: x * rng.uniform(0.8, 1.2)
        )
        # if we randomize the duration, we need to update the date list, this
        # is done later
        raw_df_sessions["duration"] = raw_df_sessions["duration"].apply(
            lambda x: int(x * rng.uniform(0.8, 1.2))
        )

    if random_choices:
        raw_df_sessions["choice"] = raw_df_sessions["choice"].apply(
            lambda x: (
                "REGULAR"
                if rng.uniform() > probability_of_scheduled_sessions
                else "SCHEDULED"
            )
        )

    if shuffle_power_profiles and random_power_profile_shapes:
        raise ValueError(
            "Please do not set both random_power_profile_shapes and shuffle_power_profiles to True"
        )

    if shuffle_power_profiles:
        raw_df_sessions["power_profiles"] = raw_df_sessions.apply(
            _shuffle_power_profiles,
            axis=1,
            initial_raw_df_sessions=initial_raw_df_sessions,
            rng=rng,
        )

    if random_power_profile_shapes:
        raw_df_sessions["power_profiles"] = raw_df_sessions.apply(
            lambda row: (
                generate_random_scheduled_profile(row["duration"], row["e_need"], rng)
                if row["choice"] == "SCHEDULED"
                else generate_regular_profile(row["duration"], row["e_need"], rng)
            ),
            axis=1,
        )

    # If we add something after here, we need to recompute e_need and the duration
    raw_df_sessions = _compute_user_needs(raw_df_sessions)

    # shuffle power can change slightly the duration, so we need to update the date
    raw_df_sessions["date"] = raw_df_sessions.apply(_update_date_to_duration, axis=1)
    raw_df_sessions["is_choice_regular"] = raw_df_sessions.apply(
        lambda row: [row["choice"] == "REGULAR"] * row["duration"], axis=1
    )
    # assert (raw_df_sessions["date"].apply(len) == raw_df_sessions["power_profiles"].apply(len)).all()
    # assert raw_df_sessions["power_profiles"].apply(min).min() >= 0
    # assert raw_df_sessions["power_profiles"].apply(max).max() <= max(SINGLE_EVSE_NORMALIZATION_PARAM * 1.05, initial_raw_df_sessions["power_profiles"].apply(max).max())

    raw_df_sessions = raw_df_sessions.drop(columns=["choice", "e_need", "duration"])
    return raw_df_sessions


def _compute_user_needs(raw_df_sessions: pd.DataFrame) -> pd.DataFrame:
    raw_df_sessions["choice"] = raw_df_sessions["is_choice_regular"].apply(
        lambda x: "REGULAR" if x[0] == 1 else "SCHEDULED"
    )
    raw_df_sessions["e_need"] = raw_df_sessions["power_profiles"].apply(
        lambda x: sum(x) / 4
    )
    raw_df_sessions["duration"] = raw_df_sessions["power_profiles"].apply(len)
    return raw_df_sessions


def _apply_randomize_start_time(
    row: pd.Series,
    rng: np.random.Generator,
    max_time_shift: pd.Timedelta = pd.Timedelta(hours=3),
) -> list[pd.Timestamp]:
    random_time_shift = rng.integers(
        -max_time_shift.total_seconds(), max_time_shift.total_seconds()  # type: ignore
    )
    # round to the closest 15 minutes (= closest 900 seconds)
    random_time_shift = pd.Timedelta(seconds=(random_time_shift // 900) * 900)
    return (pd.Series(row["date"]) + random_time_shift).to_list()


def _update_date_to_duration(row: pd.Series) -> list[pd.Timestamp]:
    return pd.date_range(
        start=row["date"][0], periods=row["duration"], freq="15min"
    ).to_list()


def generate_random_scheduled_profile(
    duration,
    e_need,
    rng: np.random.Generator,
    min_power=0,
    max_power=SINGLE_EVSE_NORMALIZATION_PARAM,
):
    max_power *= 1 + rng.uniform(-0.05, 0.03)
    # Generate random numbers
    random_numbers = rng.uniform(min_power, max_power, duration)

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
        # randomly slightly change the values equal to max power
        scaled_numbers[i] *= 1 + rng.uniform(-0.01, 0.01)
        difference -= adjustment

    # randomly shuffle the power profile
    rng.shuffle(scaled_numbers)

    return scaled_numbers.tolist()


def generate_regular_profile(
    duration,
    e_need,
    rng: np.random.Generator,
    max_power=SINGLE_EVSE_NORMALIZATION_PARAM,
):
    max_power *= 1 + rng.uniform(-0.05, 0.03)

    profile = np.zeros(duration)

    # adjust the profile to have the right energy need
    for i in range(duration):
        if e_need == 0:
            break
        adjustment = min(e_need, max_power - profile[i])
        profile[i] += adjustment
        # randomly slightly change the values equal to max power
        profile[i] *= 1 + rng.uniform(-0.01, 0.01)
        e_need -= adjustment

    return profile


def _shuffle_power_profiles(
    row: pd.Series, initial_raw_df_sessions: pd.DataFrame, rng: np.random.Generator
) -> list:
    e_need = row["e_need"]
    duration = row["duration"]
    choice = row["choice"]
    similar_sessions = initial_raw_df_sessions[
        ((initial_raw_df_sessions["e_need"] - e_need).abs() < 2000)
        & ((initial_raw_df_sessions["duration"] - duration).abs() < 3)
        & (initial_raw_df_sessions["choice"] == choice)
    ]
    if row.name in similar_sessions.index:
        similar_sessions = similar_sessions.drop(row.name)

    if similar_sessions.empty:
        return row["power_profiles"]

    return similar_sessions.sample(frac=1, random_state=rng)["power_profiles"].iloc[0]


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
