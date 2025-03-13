from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from slrp_ev_data.utils.data_utils import convert_date_from_int_to_datetime
from slrp_ev_ts_forecasting.utils.utils_artificial_data import (
    compute_choice,
    compute_duration_timesteps,
    compute_energy_needs,
    get_start_charge_time,
)


def dataset_characteristics_visualization(
    dataset: pd.DataFrame,
    session_dataset: pd.DataFrame,
    figure: Optional[go.Figure] = None,
    dataset_name: str = "",
) -> go.Figure:
    """Create a figure with multiple subplots to visualize the dataset characteristics:
    - hourly power distribution
    - daily power distribution
    - box plot of daily peak values (weekends and weekdays)
    - box plot of daily peak timing (weekends and weekdays)
    - power distribution for a specific day (2023-01-10)
    - histogram of the start charge time of the sessions

    Args:
        dataset (pd.DataFrame): power_df like dataset
        session_dataset (pd.DataFrame): session dataset, obtained with the function \
            `utils.utils_session_forecasting.get_raw_df_sessions`
        figure (Optional[ go.Figure]): If a figure object is passed, we assume \
            that it is a subplot figure with existing plot from another dataset \
            and we will add the visualizations to the existing figure. If None, \
            we will create a new figure.
        dataset_name (str): name of the dataset to be displayed in the title

    Returns:
        go.Figure: the figure object with the visualizations
    """
    dataset = dataset.copy()
    dataset["date"] = convert_date_from_int_to_datetime(dataset["date"])
    # we need to drop the nans because some of the functions to
    # get the characteristics do not handle nans
    dataset = dataset.dropna(subset=["power"])

    if figure is None:
        figure = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=[
                "Hourly power distribution (workdays)",
                "Daily power distribution",
                "Box plot of daily peak values",
                "Box plot of daily peak timing",
                "Power distribution for 2023-01-10",
                "Fraction of regular sessions",
                "Histogram of the start charge time of the sessions",
                "Box plot of the energy need of the sessions",
                "Box plot of the duration of the sessions",
            ],
        )
        figure.update_layout(title_text="Dataset characteristics", showlegend=False)

    # get how many traces are already on the subplot 1, 1
    n_datasets_on_fig = len(figure["data"]) // 6
    # pick a color from the default color palette
    color = px.colors.qualitative.__dict__["Plotly"][n_datasets_on_fig]

    # Hourly power distribution
    dataset = add_workday_column(dataset)
    workday_dataset = dataset.loc[dataset["workday_0"]]  # Filter only workdays
    hourly_power_distribution = _compute_hourly_power_distribution(workday_dataset)
    figure.add_trace(
        go.Scatter(
            x=hourly_power_distribution.index,
            y=hourly_power_distribution.values,
            line=dict(color=color),
        ),
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Hour of the day", row=1, col=1)
    figure.update_yaxes(title_text="Average power (W or scaled)", row=1, col=1)

    # Daily power distribution
    daily_power_distribution = _compute_daily_power_distribution(dataset)
    figure.add_trace(
        go.Scatter(
            x=daily_power_distribution.index,
            y=daily_power_distribution.values,
            line=dict(color=color),
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Day of the week", row=1, col=2)
    figure.update_yaxes(title_text="Average daily power (W or scaled)", row=1, col=2)

    # Box plot of daily peak values
    peak_values = _compute_peak_values(dataset)
    workday_peaks = peak_values[(True,)]
    figure.add_trace(
        go.Box(
            y=workday_peaks.values,
            name=f"Workdays {dataset_name}",
            marker_color=color,
        ),
        row=1,
        col=3,
    )
    weekend_peaks = peak_values[(False,)]
    figure.add_trace(
        go.Box(
            y=weekend_peaks.values,
            name=f"Weekends {dataset_name}",
            marker_color=color,
        ),
        row=1,
        col=3,
    )
    figure.update_yaxes(title_text="Peak power (W or scaled)", row=1, col=3)

    # Box plot of daily peak timing
    peak_timing = _compute_peak_timing(dataset)
    workday_timing = peak_timing[(True,)]
    figure.add_trace(
        go.Box(
            y=workday_timing.values, name=f"Workdays {dataset_name}", marker_color=color
        ),
        row=2,
        col=1,
    )
    weekend_timing = peak_timing[(False,)]
    figure.add_trace(
        go.Box(
            y=weekend_timing.values, name=f"Weekends {dataset_name}", marker_color=color
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Peak timing (hour)", row=2, col=1)

    # Power distribution for a specific day (2023-01-10)
    specific_day_data = dataset[
        dataset["date"].isin(pd.date_range("2023-01-10", "2023-01-12", freq="15min"))
    ]
    figure.add_trace(
        go.Scatter(
            x=specific_day_data["date"],
            y=specific_day_data["power"],
            line=dict(color=color),
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Hour of the day", row=2, col=2)
    figure.update_yaxes(title_text="Power (W or scaled)", row=2, col=2)

    # Bar chart of the fraction of regular sessions (and the total number of sessions)
    number_of_regular_sessions = (compute_choice(session_dataset) == "REGULAR").sum()
    number_of_sessions = len(session_dataset)
    figure.add_trace(
        go.Bar(
            x=["Regular", "Total"],
            y=[number_of_regular_sessions, number_of_sessions],
            marker_color=color,
        ),
        row=2,
        col=3,
    )
    figure.update_yaxes(title_text="Number of sessions", row=2, col=3)

    # Histogram of the start charge time of the sessions
    start_charge_time = session_dataset.apply(get_start_charge_time, axis=1)
    start_charge_time = start_charge_time.dt.hour + start_charge_time.dt.minute / 60
    figure.add_trace(
        go.Histogram(x=start_charge_time, marker_color=color, opacity=0.6), row=3, col=1
    )
    figure.update_xaxes(title_text="Hour of the day", row=3, col=1)
    figure.update_yaxes(title_text="Number of sessions", row=3, col=1)

    # Box plot of the energy_need of the sessions
    energy_need = compute_energy_needs(session_dataset)
    figure.add_trace(
        go.Box(y=energy_need, name=f"Energy need {dataset_name}", marker_color=color),
        row=3,
        col=2,
    )
    figure.update_yaxes(title_text="Energy need (Wh)", row=3, col=2)

    # Box plot of the duration of the sessions
    duration = compute_duration_timesteps(session_dataset) / 4
    figure.add_trace(
        go.Box(y=duration, name=f"Duration {dataset_name}", marker_color=color),
        row=3,
        col=3,
    )
    figure.update_yaxes(title_text="Duration (hours)", row=3, col=3)

    return figure


def _compute_hourly_power_distribution(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()
    dataset["hour"] = dataset["date"].dt.hour
    return dataset.groupby("hour")["power"].mean()


def _compute_daily_power_distribution(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()
    dataset["day_of_week"] = dataset["date"].dt.day_of_week
    return dataset.groupby("day_of_week")["power"].mean()


def _compute_peak_values(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()

    dataset = add_workday_column(dataset)
    dataset["day"] = dataset["date"].dt.date
    peak_values = dataset.groupby(["workday_0", "day"])["power"].max()

    return peak_values


def _compute_peak_timing(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()

    dataset = add_workday_column(dataset)
    dataset["day"] = dataset["date"].dt.date
    peak_timing = dataset.groupby(["workday_0", "day"]).apply(
        lambda x: x["date"].dt.hour[x["power"].idxmax()]
    )

    return peak_timing


def add_workday_column(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    if "workday_0" not in dataset.columns:
        dataset["workday_0"] = dataset["date"].dt.day_of_week.isin([0, 1, 2, 3])
    else:
        dataset["workday_0"] = dataset["workday_0"].astype(bool)
    return dataset
