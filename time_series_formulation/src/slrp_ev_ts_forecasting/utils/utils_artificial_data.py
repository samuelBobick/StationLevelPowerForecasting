import numpy as np
import pandas as pd
from slrp_ev_data.feature_engineering import (
    feature_engineering,
)
from slrp_ev_data.utils.data_utils import convert_date_from_int_to_datetime
from slrp_ev_data.utils.scaling_main import get_scaling_parameters
from slrp_ev_data.utils.scaling_utils import (
    SINGLE_EVSE_NORMALIZATION_PARAM,
)
from slrp_ev_ts_forecasting.default_parameters import RANDOM_SEED, TypeScalingMode
from slrp_ev_ts_forecasting.utils.utils_session_forecasting import (
    extract_features,
    revert_power_df,
)


def get_start_charge_time(row: pd.Series) -> pd.Timestamp:
    return row["date"][0]


def compute_choice(raw_df_sessions: pd.DataFrame) -> pd.Series:
    return raw_df_sessions["is_choice_regular"].apply(
        lambda x: "REGULAR" if x[0] == 1 else "SCHEDULED"
    )


def compute_energy_needs(raw_df_sessions: pd.DataFrame) -> pd.Series:
    return raw_df_sessions["power_profiles"].apply(lambda x: sum(x) / 4)


def compute_duration_timesteps(raw_df_sessions: pd.DataFrame) -> pd.Series:
    return raw_df_sessions["power_profiles"].apply(len)


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
    # multiply the energy need by 4, to go from kWh to "power by timestep"
    e_need *= 4
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
    # multiply the energy need by 4, to go from kWh to "power by timestep"
    e_need *= 4
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


def get_start_charge_time_distribution(raw_df_sessions: pd.DataFrame) -> pd.Series:
    start_charge_times = raw_df_sessions.apply(get_start_charge_time, axis=1)
    start_charge_times = start_charge_times.dt.hour + start_charge_times.dt.minute / 60

    start_charge_time_distribution = start_charge_times.value_counts().sort_index()

    return start_charge_time_distribution.reindex(np.arange(0, 24, step=0.25)).fillna(0)


def _apply_randomize_start_time(
    row: pd.Series,
    rng: np.random.Generator,
    start_charge_time_distribution: pd.Series,
    max_time_shift_hours: int = 3,
) -> list[pd.Timestamp]:
    # First, we get the distribution from which we will sample the
    # new start times
    current_hour = row["date"][0].hour + row["date"][0].minute / 60
    # max_time_shift_hours - current hour should be > 0
    max_time_shift_hours = min(max_time_shift_hours, current_hour)
    # max_time_shift_hours - current hour should be < 24
    max_time_shift_hours = min(max_time_shift_hours, 24 - current_hour)
    start_charge_time_distribution = start_charge_time_distribution.loc[
        current_hour - max_time_shift_hours : current_hour + max_time_shift_hours
    ]
    normalized_start_charge_time_distribution = (
        start_charge_time_distribution / start_charge_time_distribution.sum()
    )
    # Now we sample from the distribution
    random_time_shift = rng.choice(
        normalized_start_charge_time_distribution.index,
        p=normalized_start_charge_time_distribution.values,
    )

    random_time_shift = pd.Timedelta(hours=random_time_shift - current_hour).round(
        "15min"
    )
    return (pd.Series(row["date"]) + random_time_shift).to_list()


def make_artificial_sessions(
    raw_df_sessions: pd.DataFrame,
    random_start_time: bool,
    shuffle_power_profiles: bool,
    random_power_profile_shapes: bool,
    random_user_needs: bool,
    random_choices: bool,
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED),
    probability_of_scheduled_sessions: float = 0.3,
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
    raw_df_sessions = raw_df_sessions.copy()
    raw_df_sessions["choice"] = compute_choice(raw_df_sessions)
    raw_df_sessions["e_need"] = compute_energy_needs(raw_df_sessions)
    raw_df_sessions["duration"] = compute_duration_timesteps(raw_df_sessions)
    initial_raw_df_sessions = raw_df_sessions.copy()

    if random_start_time:
        start_charge_time_distribution = get_start_charge_time_distribution(
            raw_df_sessions
        )
        raw_df_sessions["date"] = raw_df_sessions.apply(
            _apply_randomize_start_time,
            axis=1,
            rng=rng,
            start_charge_time_distribution=start_charge_time_distribution,
        )  # type: ignore

    if random_user_needs:
        raw_df_sessions["e_need"] = raw_df_sessions["e_need"].apply(
            lambda x: x * rng.uniform(0.7, 1.3)
        )
        # if we randomize the duration, we need to update the date list, this
        # is done later
        # the minimum duration should be 2 timesteps
        raw_df_sessions["duration"] = raw_df_sessions["duration"].apply(
            lambda x: max(int(x * rng.uniform(0.7, 1.3)), 2)
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
    raw_df_sessions["e_need"] = compute_energy_needs(raw_df_sessions)
    raw_df_sessions["duration"] = compute_duration_timesteps(raw_df_sessions)

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


def get_artificial_data(
    train_data: pd.DataFrame,
    raw_df_sessions: pd.DataFrame,
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

    # we need to convert the train date column to a datetime object before
    # we can do the merge and have the same dates as the train data
    train_data = train_data.copy()
    train_data["date"] = convert_date_from_int_to_datetime(train_data["date"])
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
        scaling_mode=scaling_mode,
        lookahead_15min_steps=lookahead,
        dataset_name="slrp-ev_new",
        retrieve_from_saved=scaling_mode in ["normalize", "standardize"],
    )
    artificial_train_data = feature_engineering(
        data_input=artificial_power_df,
        add_nans_for_missing_data=True,
        scaling_mode=scaling_mode,
        scaling_parameters=scaling_parameters,
        cols_normalization_to_skip=["number_of_evses_available"],
        lookahead=lookahead,
    )
    return artificial_train_data, artificial_raw_df_sessions, scaling_parameters
