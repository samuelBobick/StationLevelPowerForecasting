from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf


def get_pacf_values(
    downsample_hours: int,
    data: pd.DataFrame,
    nb_of_days_for_pacf: int = 30,
    nb_of_steps_to_predict: int = 1,
) -> pd.DataFrame:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df = df.set_index("date")
    df = df.resample(f"{downsample_hours}h").mean()

    pacf_params = {
        "x": tuple(df["power"].values),
        "alpha": 0.05,
        "nlags": int(nb_of_days_for_pacf * 96 / (4 * downsample_hours)),
    }

    # Compute PACF values
    pacf_values = pacf_wrapper(**pacf_params)

    # Create a DataFrame to store PACF values and lags
    pacf_df = pd.DataFrame({"PACF": pacf_values[0]})

    # Convert index to timedelta to resample to 15 minutes data
    pacf_df.index = pd.to_timedelta(pacf_df.index * 15 * 4 * downsample_hours, unit="m")
    # resample to 15 minutes data
    pacf_df = pacf_df.resample("15Min").bfill()
    # Drop index so that we have lag numbers instead of time delta
    pacf_df = pacf_df.reset_index(drop=True)

    # The PACF gives us the correlation of the next value with the lags
    # However, we want to predict multiple future values. Each of the future values
    # will have the same PACF, but shifted by the number of steps to predict.
    # we have to keep the lags that are the most correlated, looking at the PACF of
    # all the values we want to predict (not only the first one)
    pacf_df["PACF"] = pacf_df["PACF"].map(abs)
    for i in range(1, nb_of_steps_to_predict):
        pacf_df["PACF"] = pacf_df["PACF"].combine(
            pacf_df["PACF"].shift(-i).fillna(0), max
        )
    return pacf_df


@lru_cache
def pacf_wrapper(x: tuple, alpha: float, nlags: int):
    # x needs to be a tuble to be hashable and used in lru_cache
    return pacf(np.array(x), alpha=alpha, nlags=nlags)


def get_threshold(
    pacf_df: pd.DataFrame, number_of_lags_to_keep: int = 96
) -> tuple[float, int]:
    """Given a DataFrame with PACF values, returns the threshold value and the index of the farthest lag"""
    df = pacf_df.sort_values(by="PACF", ascending=False).iloc[:number_of_lags_to_keep]
    threshold_value = df.iloc[-1]
    index_of_farther_lag = df.index.max()
    return threshold_value.iloc[0], index_of_farther_lag
