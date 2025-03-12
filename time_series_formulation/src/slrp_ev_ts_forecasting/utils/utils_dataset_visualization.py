from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from slrp_ev_data.utils.data_utils import convert_date_from_int_to_datetime


def dataset_characteristics_visualization(
    dataset: pd.DataFrame, figure: Optional[go.Figure] = None, dataset_name: str = ""
) -> go.Figure:
    """Create a figure with multiple subplots to visualize the dataset characteristics:
    - hourly power distribution
    - daily power distribution
    - box plot of daily peak values (weekends and weekdays)
    - box plot of daily peak timing (weekends and weekdays)
    - power distribution for a specific day (2023-01-10)

    Args:
        dataset (pd.DataFrame): power_df like dataset
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
            cols=2,
            subplot_titles=[
                "Hourly power distribution",
                "Daily power distribution",
                "Box plot of daily peak values",
                "Box plot of daily peak timing",
                "Power distribution for 2023-01-10",
                "",
            ],
        )
        figure.update_layout(title_text="Dataset characteristics", showlegend=False)

    # get how many traces are already on the subplot 1, 1
    n_datasets_on_fig = len(figure["data"]) // 6
    # pick a color from the default color palette
    color = px.colors.qualitative.__dict__["Plotly"][n_datasets_on_fig]

    # Hourly power distribution
    hourly_power_distribution = _compute_hourly_power_distribution(dataset)
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
        row=2,
        col=1,
    )
    weekend_peaks = peak_values[(False,)]
    figure.add_trace(
        go.Box(
            y=weekend_peaks.values,
            name=f"Weekends {dataset_name}",
            marker_color=color,
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Peak power (W or scaled)", row=2, col=1)

    # Box plot of daily peak timing
    peak_timing = _compute_peak_timing(dataset)
    workday_timing = peak_timing[(True,)]
    figure.add_trace(
        go.Box(
            y=workday_timing.values, name=f"Workdays {dataset_name}", marker_color=color
        ),
        row=2,
        col=2,
    )
    weekend_timing = peak_timing[(False,)]
    figure.add_trace(
        go.Box(
            y=weekend_timing.values, name=f"Weekends {dataset_name}", marker_color=color
        ),
        row=2,
        col=2,
    )
    figure.update_yaxes(title_text="Peak timing (hour)", row=2, col=2)

    # Power distribution for a specific day (2023-01-10)
    specific_day_data = dataset[
        dataset["date"].dt.date == pd.to_datetime("2023-01-10").date()
    ]
    figure.add_trace(
        go.Scatter(
            x=specific_day_data["date"],
            y=specific_day_data["power"],
            line=dict(color=color),
        ),
        row=3,
        col=1,
    )
    figure.update_xaxes(title_text="Hour of the day", row=3, col=1)
    figure.update_yaxes(title_text="Power (W or scaled)", row=3, col=1)

    return figure


def _compute_hourly_power_distribution(dataset: pd.DataFrame) -> pd.Series:
    dataset["hour"] = dataset["date"].dt.hour
    return dataset.groupby("hour")["power"].mean()


def _compute_daily_power_distribution(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()
    dataset["day_of_week"] = dataset["date"].dt.day_of_week
    return dataset.groupby("day_of_week")["power"].mean()


def _compute_peak_values(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()

    if "workday_0" not in dataset.columns:
        dataset["workday_0"] = dataset["date"].dt.day_of_week.isin([0, 1, 2, 3])
    dataset["day"] = dataset["date"].dt.date
    peak_values = dataset.groupby(["workday_0", "day"])["power"].max()

    return peak_values


def _compute_peak_timing(dataset: pd.DataFrame) -> pd.Series:
    dataset = dataset.copy()

    if "workday_0" not in dataset.columns:
        dataset["workday_0"] = dataset["date"].dt.day_of_week.isin([0, 1, 2, 3])
    dataset["day"] = dataset["date"].dt.date
    peak_timing = dataset.groupby(["workday_0", "day"]).apply(
        lambda x: x["date"].dt.hour[x["power"].idxmax()]
    )

    return peak_timing
