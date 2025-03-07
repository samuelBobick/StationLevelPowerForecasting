from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from slrp_ev_data.utils.data_utils import get_data_frequency
from slrp_ev_ts_forecasting.default_parameters import VERBOSE
from statsmodels.tsa.stattools import pacf


def get_pacf_values(
    downsample_hours: float,
    data: pd.DataFrame,
    nb_of_days_for_pacf: int,
    nb_of_steps_to_predict: int = 1,
    return_confidence_interval: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, float]:
    """Compute the Partial AutoCorrelation Function (PACF) values for the power data.

    Args:
        downsample_hours: Number of hours to downsample the data to.\
            The PACF will be computed on this downsampled data, but the final result will\
            still have a 15 minutes frequency. (e.g. 0.5, 1 or 2).
        data: Data generate by the feature engineering function of slrp_ev_data
        nb_of_days_for_pacf: Number of days to consider for the PACF (the maximum\
            lag will be 96 * nb_of_days_for_pacf)
        nb_of_steps_to_predict (optional): Number of steps ahead that we want to forecast.\
            Increasing this value means that we will keep the lags that are the most correlated\
            to all the steps to predict. If lag 4 has a high pacf for the next value (value 0),\
            it means that lag 3 has that high pacf for value 1. Defaults to 1.
        return_confidence_interval (optional): If True, the confidence interval will be returned.\

    Returns:
        DataFrame with the PACF values and the lags as index. 

    Example:
        >>> pacf_df = get_pacf_values(
                downsample_hours=1,
                data=data,
                nb_of_days_for_pacf=70,
                nb_of_steps_to_predict=96,
            )
    """
    data = data.copy()
    data = data.dropna(subset=["power"])
    # TODO: use data frequency instead of hardcoded 96, 4 and 15
    data_frequency = get_data_frequency(data)
    # TODO: change below with function from feature engineering
    data["date"] = pd.to_datetime(data["date"], unit="s")
    df_complete_intervals = get_df_complete_intervals(data, "date", data_frequency)

    total_number_of_timesteps = 0
    number_of_timesteps_with_enough_data = 0
    minimum_number_timesteps_covered_if_using_pacf = 0
    average_pacf = pd.DataFrame()
    for complete_interval in df_complete_intervals.itertuples():
        filtered_df = data.query(
            f"date >= '{complete_interval.start_complete}' and date <= '{complete_interval.end_complete}'"
        )
        length_data = filtered_df.shape[0]
        total_number_of_timesteps += length_data
        # TODO: replace this hardcoded 4 once we have the frequency

        if length_data < nb_of_days_for_pacf * 24 * 4 * 4:
            # the second 4 is a recommendation of the minimum number of timesteps to have
            # to have accurate PACF values for further lags
            # see here: https://stackoverflow.com/questions/55252492/pacf-function-in-statsmodels-tsa-stattools-gives-numbers-greater-than-1-when-usi
            # Skipping interval because not enough data points to compute pacf
            continue
        number_of_timesteps_with_enough_data += length_data
        minimum_number_timesteps_covered_if_using_pacf += (
            length_data - nb_of_days_for_pacf * 24 * 4
        )

        pacf_df = get_pacf_values_complete_interval(
            downsample_hours,
            filtered_df,
            nb_of_days_for_pacf,
            nb_of_steps_to_predict,
            False,
        )
        pacf_df = pacf_df * length_data  # for weighted average

        average_pacf = pd.concat([average_pacf, pacf_df], axis=1)  # type: ignore

    average_pacf = (
        pd.DataFrame({"PACF": average_pacf.sum(axis=1)})
        / number_of_timesteps_with_enough_data
    )
    minimum_coverage = (
        minimum_number_timesteps_covered_if_using_pacf / total_number_of_timesteps
    )
    if minimum_coverage < 0.5:
        print(
            f"WARNING: The minimum coverage of the data is {minimum_coverage*100:.0f}%. "
            "This means that the optimized lags reduce by a lot the "
            "number of samples to train the model. Please consider "
            f"decreasing nb_of_days_for_pacf (currently set to {nb_of_days_for_pacf})."
        )

    if return_confidence_interval:
        interval = compute_confidence_interval(
            total_number_of_timesteps / 4 / downsample_hours
        )
        return average_pacf, interval

    return average_pacf


def get_pacf_values_complete_interval(
    downsample_hours: float,
    data: pd.DataFrame,
    nb_of_days_for_pacf: int,
    nb_of_steps_to_predict: int = 1,
    return_confidence_interval: bool = False,
):
    df = data.copy()

    # parse date and set it as index to be able to resample data
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df = df.set_index("date")
    df = df.resample(f"{downsample_hours}h").mean()

    pacf_params = {
        "x": tuple(df["power"].values),
        "nlags": int(nb_of_days_for_pacf * 96 / (4 * downsample_hours)),
    }

    # Compute PACF values and cache the results for faster computation if function
    # is called again with the same parameters
    pacf_values = pacf_wrapper(**pacf_params)

    # Create a DataFrame to store PACF values and lags
    pacf_df = pd.DataFrame({"PACF": pacf_values})

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
    new_pacf = pacf_df["PACF"]
    for i in range(1, nb_of_steps_to_predict):
        new_pacf = new_pacf.combine(pacf_df["PACF"].shift(-i).fillna(0), max)
    pacf_df["PACF"] = new_pacf

    if return_confidence_interval:
        interval = compute_confidence_interval(df.shape[0])
        return pacf_df, interval
    return pacf_df


def compute_confidence_interval(len_df: float) -> float:
    # compute confidence interval (taken from the source code of statsmodels.tsa.stattools.pacf)
    # see also: https://en.wikipedia.org/wiki/Partial_autocorrelation_function#Autoregressive_model_identification
    varacf = 1.0 / len_df  # for all lags >=1
    confidence_level = 0.05
    interval = stats.norm.ppf(1.0 - confidence_level / 2.0) * np.sqrt(varacf)
    # for a 95% confidence interval (confidence_level = 0.05),
    # the interval is 1.96 * sqrt(varacf)
    return interval


@lru_cache
def pacf_wrapper(x: tuple, nlags: int):
    # x needs to be a tuple to be hashable and used in lru_cache
    return pacf(np.array(x), nlags=nlags)


def sort_pacf_values(
    pacf_df: pd.DataFrame, number_of_lags_to_keep: int
) -> pd.DataFrame:
    df = pacf_df.copy()
    df["Lags"] = df.index
    return (
        df.iloc[1:]  # remove first value (autocorrelation with itself)
        .sort_values(by=["PACF", "Lags"], ascending=[False, True])
        .iloc[:number_of_lags_to_keep]
    )


def get_threshold(
    pacf_df: pd.DataFrame,
    number_of_lags_to_keep: int = 96,
    interval: Optional[float] = None,
    verbose: bool = VERBOSE,
) -> tuple[float, int]:
    """Given a DataFrame with PACF values, returns the threshold value and
    the index of the farthest lag"""
    df = sort_pacf_values(pacf_df, number_of_lags_to_keep)
    threshold_value = df.iloc[-1]["PACF"]
    index_of_farther_lag = df.index.max()
    if verbose:
        plot_df_pacf(
            pacf_df, number_of_lags_to_keep=number_of_lags_to_keep, interval=interval
        )

        if interval and threshold_value < interval:
            print(
                f"WARNING: The threshold value is {threshold_value['PACF']:.2f}, "
                f"which is below the confidence interval ({interval:.2f}). "
                "This is an issue with the pacf (used to find the best lags). It means that "
                "some of the selected lags are not statistically significant. "
                "you should consider decreasing the number of lags to keep (x_dim) or increasing "
                "the number of days used to compute the pacf."
            )
    return threshold_value, index_of_farther_lag


def plot_df_pacf(
    pacf_df: pd.DataFrame,
    number_of_lags_to_keep: int,
    interval: Optional[float] = None,
    figure: Optional[go.Figure] = None,
    figure_number: int = 0,
):
    if not figure:
        fig = make_subplots(shared_xaxes=True, rows=2, cols=1, row_heights=[0.7, 0.3])
    else:
        fig = figure

    fig.add_trace(
        go.Bar(
            x=pacf_df.index,
            y=pacf_df["PACF"],
        ),
        row=2 * figure_number + 1,
        col=1,
    )
    fig.update_yaxes(title_text="PACF", row=2 * figure_number + 1, col=1)

    # update scale
    fig.update_yaxes(range=[0, 0.2], row=2 * figure_number + 1, col=1)

    # add red area for the confidence interval
    if interval:
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=pacf_df.index.max(),
            y1=interval,
            line=dict(color="red", width=2),
            fillcolor="red",
            opacity=0.1,
            # layer="below",
            row=2 * figure_number + 1,
            col=1,
        )

    # Add a horizontal bicolor line as a 1D heatmap to the second row of the figure
    selected_lags = sort_pacf_values(pacf_df, number_of_lags_to_keep)

    pacf_df["Selected"] = False
    pacf_df.loc[selected_lags.index, "Selected"] = True

    # Create a 1D heatmap for the selected lags
    heatmap_colors = [1 if selected else 0 for selected in pacf_df["Selected"]]
    fig.add_trace(
        go.Heatmap(
            z=[heatmap_colors],
            showscale=False,
            colorscale=[[0, "lightyellow"], [1, "green"]],
        ),
        row=2 * figure_number + 2,
        col=1,
    )

    if not figure:
        # Update layout
        fig.update_layout(
            title=f"PACF (top) and Selected Lags ({number_of_lags_to_keep} "
            "in green - bottom).<br>"
            "The red area represents the confidence interval (values inside "
            "the area are not statistically significant).",
            height=600,  # Total height of the figure6
        )
        fig.update_yaxes(title_text="PACF", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=1)

        fig.show()


def get_df_complete_intervals(
    df: pd.DataFrame,
    date_column: str,
    frequency: str,
) -> pd.DataFrame:
    """Returns the complete intervals. there is data for the specified start_complete \
        and end_complete dates (so use >= and <=)"""
    df_complete_intervals = df.copy()
    # find the difference between two data points
    df_complete_intervals["diff"] = df_complete_intervals[date_column].diff()

    # dates are missing if the difference is not the frequency
    df_complete_intervals = df_complete_intervals[
        df_complete_intervals["diff"] > pd.Timedelta(frequency)
    ]
    if df_complete_intervals.empty:
        # Then we have no missing interval!
        return pd.DataFrame(
            {
                "start_complete": (df[date_column].min()),
                "end_complete": df[date_column].max(),
            },
            index=[0],
        )

    # define start and end dates of missing intervals
    for i in range(df_complete_intervals.shape[0]):
        index = df_complete_intervals.index[i]
        df_complete_intervals.loc[index, "end_complete"] = (
            df_complete_intervals.loc[index, date_column]
            - df_complete_intervals.loc[index, "diff"]
        )
        if i == 0:
            df_complete_intervals.loc[index, "start_complete"] = df.iloc[0][date_column]
        else:
            df_complete_intervals.loc[index, "start_complete"] = (
                df_complete_intervals.iloc[i - 1][date_column]
            )

    # add the last missing interval
    last_row = pd.DataFrame(
        {
            "start_complete": (df_complete_intervals.iloc[-1][date_column]),
            "end_complete": df.iloc[-1][date_column],
        },
        index=[df_complete_intervals.index[-1] + 1],
    )
    df_complete_intervals = pd.concat([df_complete_intervals, last_row], axis=0)

    return df_complete_intervals
