from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from utils.utils_time_and_indexes import (
    convert_power_profile_to_df,
    convert_time_to_index,
)


def create_tou_heatmap_trace(
    whole_time_index: pd.Series,
    TOU: np.ndarray,
    TOU_current_idx: int = 0,
    unit: Literal["kW", "W"] = "kW",
):
    """
    Creates a heatmap trace for TOU visualization in the background of plots.

    Args:
        whole_time_index (pd.Series): The time index for the plot.
        TOU (np.ndarray): The TOU price array.
        TOU_current_idx (int, optional): The starting index for TOU prices. Defaults to 0.
        unit (str, optional): The unit for the y-axis range. "kW" or "W". Defaults to "kW".

    Returns:
        go.Heatmap: A Plotly heatmap trace.
    """
    number_timesteps_in_simulation = whole_time_index.shape[0]

    if number_timesteps_in_simulation > (96 - TOU_current_idx):
        data = {
            "TOU": list(TOU[TOU_current_idx:96])
            + list(TOU[:96])
            * ((number_timesteps_in_simulation - 96 + TOU_current_idx) // 96)
            + list(TOU[: (number_timesteps_in_simulation - 96 + TOU_current_idx) % 96])
        }
    else:
        data = {
            "TOU": list(
                TOU[TOU_current_idx : TOU_current_idx + number_timesteps_in_simulation]
            )
        }
    TOU_data = pd.DataFrame(
        data=data,
        index=whole_time_index,
    )

    y_range = [0, 60_000] if unit == "W" else [0, 60]

    # Define a custom greyscale where super off-peak values are white
    custom_colorscale = [
        [0.0, "white"],  # Lowest values (super off-peak) are white
        [0.2, "lightgrey"],
        [0.5, "grey"],
        [0.8, "darkgrey"],
        [1.0, "black"],  # Highest values are black
    ]

    # if the index is in a Timestamp format, we need to add half of the timestep to the index
    if isinstance(TOU_data.index, pd.DatetimeIndex):
        x = TOU_data.index + pd.Timedelta(minutes=15 / 2)
    else:
        # Else we assume that it is float representing hours
        x = TOU_data.index + 0.25 / 2

    return go.Heatmap(
        x=x,
        y=y_range,
        z=[TOU_data["TOU"]] * 2,  # Duplicate TOU values to match the y range
        colorscale=custom_colorscale,
        showscale=False,
        zmin=TOU_data["TOU"].min(),
        zmax=TOU_data["TOU"].max() * 2,
        hovertemplate="TOU: %{z:.2f} cents/kWh<extra></extra>",
    )


def visualize_simulation_power(
    aggregate_power_profile: pd.DataFrame,
    power_profiles: dict,
    user_computed_data_for_visualization: dict,
    prices: dict,
    delta_t: float,
    TOU: np.ndarray,
) -> None:
    """
    Visualize the results of the simulation values. the x axis is time

    Args:
        aggregate_power_profile: dataframe with columns "date" and "power"
    """
    # turn dictionaries into dataframe for easier manipulation
    df_user_computed_data_for_visualization = pd.DataFrame(
        user_computed_data_for_visualization
    ).T

    user_power_profiles_dfs = pd.DataFrame()
    for user_id, user_power_profile in power_profiles.items():
        start_charge_time = df_user_computed_data_for_visualization.loc[
            user_id, "Start charge time"
        ]
        power_profiles_df = convert_power_profile_to_df(
            user_power_profile, start_charge_time, delta_t=delta_t
        )

        power_profiles_df["user_id"] = user_id
        user_power_profiles_dfs = pd.concat(
            [user_power_profiles_dfs, power_profiles_df]
        )

        df_user_computed_data_for_visualization.loc[user_id, "Energy delivered"] = (
            np.round(sum(power_profiles_df["power"] * delta_t), 2)
        )

    fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, 60])))

    # Change the background color based on the TOU prices (as bars)
    TOU_data_index = aggregate_power_profile["date"]
    # Add a heatmap for TOU values as background
    fig.add_trace(create_tou_heatmap_trace(TOU_data_index, TOU, unit="kW"))

    # Create a color map for user IDs
    user_ids = user_power_profiles_dfs["user_id"].unique()
    color_scale = plotly.colors.qualitative.Pastel1
    color_scale_darker = (
        plotly.colors.qualitative.Set1
    )  # same colors as pastel palette but darker
    length_color_scale = len(color_scale)
    colors = {
        user_id: color_scale[i % length_color_scale]
        for i, user_id in enumerate(user_ids)
    }

    # Create stacked area plots for each user
    for i, (user_id, user_power_profile_df) in enumerate(
        user_power_profiles_dfs.groupby("user_id")
    ):
        fig.add_trace(
            go.Scatter(
                x=user_power_profile_df["date"],
                y=user_power_profile_df["power"],
                mode="none",
                name=f"User {user_id}",
                line_shape="hv",
                # this parameter will make the line go horizontally first and then vertically (steps)
                stackgroup="one",
                fillcolor=colors[user_id],
            )
        )

    # Aggregate power profile
    fig.add_trace(
        go.Scatter(
            x=aggregate_power_profile["date"],
            y=aggregate_power_profile["power"],
            mode="lines",
            name="Power Profile",
            line=dict(color=color_scale_darker[1], shape="hv"),  # blue
        )
    )

    # Add price data to the second y-axis
    fig.update_layout(
        yaxis2=dict(
            title="Prices (cents/kWh)", overlaying="y", side="right", range=[0, None]
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[0] for user_id, price in prices.items()],
            mode="lines+markers",
            name="Scheduled Price",
            yaxis="y2",
            line=dict(dash="dash", color=color_scale_darker[2]),  # green
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[price[1] for user_id, price in prices.items()],
            mode="lines+markers",
            name="Regular Price",
            yaxis="y2",
            line=dict(dash="dash", color=color_scale_darker[0]),  # red
        )
    )

    # add a dot at the beginning of each session
    fig.add_trace(
        go.Scatter(
            x=df_user_computed_data_for_visualization["Start charge time"],
            y=[0] * len(df_user_computed_data_for_visualization),
            mode="markers",
            marker=dict(
                size=10,
                color=[
                    colors[user_id]
                    for user_id in df_user_computed_data_for_visualization.index
                ],
            ),
            showlegend=False,
            hovertemplate=(
                "User ID: %{customdata[0]}<br>"
                "Start Charge Time: %{x}<br>"
                "Energy needed: %{customdata[1]}<br>"
                "Energy delivered: %{customdata[2]}<br>"
                "Duration (hours): %{customdata[3]}<br>"
                "Choice: %{customdata[4]}<extra></extra>"
            ),
            customdata=np.array(
                [
                    df_user_computed_data_for_visualization.index,
                    df_user_computed_data_for_visualization["Energy needed"],
                    df_user_computed_data_for_visualization["Energy delivered"],
                    df_user_computed_data_for_visualization["Duration (hours)"],
                    df_user_computed_data_for_visualization["Choice"],
                ]
            ).T,
        )
    )

    fig.update_layout(
        title=f"Simulated Aggregate Power Profile: {min(aggregate_power_profile['date'].dt.date)} to {max(aggregate_power_profile['date'].dt.date)}",
        xaxis_title="Date",
        yaxis_title="Aggregate Station Power (kWh)",
        template="plotly_white",
    )
    fig.show()


def visualize_simulation_prices(
    session_results: pd.DataFrame,
    box_plot: bool = False,
    TOU: Optional[np.ndarray] = None,
    aggregate_power_profile: Optional[pd.DataFrame] = None,
):
    """Creates a figure with a boxplot (or just scatter) of the prices, grouped by hours.
    On the x axis, we have hours, while on the y axis we have prices.
    For each hour, there are 2 boxes, one for the scheduled prices, and one for the regular prices.
    The prices are in cents/kWh.

    Args:
        session_results (pd.DataFrame): output DataFrame of the function get_session_results
        box_plot (bool): if True, creates a box plot. If False, creates a scatter plot.
        TOU (np.ndarray, optional): Time-of-Use price array for adding background heatmap. Defaults to None.
    """
    df_session_prices = session_results[["start_time", "z_sch", "z_reg"]].copy()
    df_session_prices["hour"] = df_session_prices["start_time"].dt.hour
    df_session_prices = df_session_prices.drop(columns="start_time")

    fig = go.Figure()

    # Add TOU heatmap in the background if TOU is provided
    if TOU is not None:
        whole_time_index = pd.Series(
            data=np.arange(
                start=max(df_session_prices["hour"].min() - 1, 0),
                stop=min(df_session_prices["hour"].max() + 1, 23),
                step=0.25,
            )
        )
        TOU_current_idx = convert_time_to_index(
            pd.to_datetime("2020-01-01 00:00:00")
            + pd.Timedelta(hours=df_session_prices["hour"].min()),
            0.25,
        )
        fig.add_trace(
            create_tou_heatmap_trace(
                whole_time_index, TOU, TOU_current_idx=TOU_current_idx, unit="kW"
            )
        )

    # Add average power profile
    if aggregate_power_profile is not None:
        aggregate_power_profile["hour"] = aggregate_power_profile["date"].dt.hour
        aggregate_power_profile = aggregate_power_profile.drop(columns="date")
        average_power_profile = aggregate_power_profile.groupby("hour")["power"].mean()

        # truncate the power profile to the hours of the simulation
        average_power_profile = average_power_profile[
            average_power_profile.index.isin(
                range(
                    df_session_prices["hour"].min(), df_session_prices["hour"].max() + 1
                )
            )
        ]
        fig.add_trace(
            go.Bar(
                x=average_power_profile.index,
                y=average_power_profile,
                name="Average Power Profile",
                marker=dict(color="blue"),
                opacity=0.5,
                yaxis="y2",  # Assign to secondary y-axis
            )
        )
        fig.update_layout(
            # Update layout to include secondary y-axis
            yaxis2=dict(
                title="Average Power Profile (kW)",
                side="right",
                overlaying="y",
            )
        )

    for price_type in ["z_sch", "z_reg"]:
        color = "green" if price_type == "z_sch" else "red"
        if box_plot:
            fig.add_trace(
                go.Box(
                    x=df_session_prices["hour"],
                    y=df_session_prices[price_type],
                    name=price_type,
                    boxmean="sd",
                    marker=dict(color=color),
                    yaxis="y3",
                )
            )
        else:
            df_average_prices = df_session_prices.groupby("hour")[price_type].mean()
            fig.add_trace(
                go.Scatter(
                    x=df_average_prices.index,
                    y=df_average_prices,
                    mode="lines+markers",
                    name=price_type,
                    marker=dict(color=color),
                    line=dict(dash="dash"),
                    yaxis="y3",
                )
            )

    title = "Distribution of average prices per hour"
    # add average regular and scheduled prices in the title
    title += f"<br>    Average scheduled price: {df_session_prices['z_sch'].mean():.1f} cents/kWh"
    title += f"<br>    Average regular price: {df_session_prices['z_reg'].mean():.1f} cents/kWh"

    fig.update_layout(
        legend=dict(orientation="h"),
        title=title,
        xaxis_title="Hour",
        template="plotly_white",
        yaxis=dict(range=[30, 70], visible=False),
        yaxis3=dict(title="Average Prices (cents/kWh)", overlaying="y"),
    )

    fig.show()


def visualize_simulation_choices(session_results: pd.DataFrame):
    """Creates a figure with a bar plot. Each hour has one bar, representing the
    fraction of people that chose scheduled charging.

    Args:
        session_results (pd.DataFrame): output DataFrame of the function get_session_results
    """

    df_session_choices = session_results[["start_time", "choice"]].copy()
    df_session_choices["hour"] = df_session_choices["start_time"].dt.hour
    df_session_choices = df_session_choices.drop(columns="start_time")
    df_session_choices = (
        df_session_choices.groupby(["hour"]).value_counts(normalize=True).unstack()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_session_choices.index,
            y=df_session_choices["SCHEDULED"],
            name="choice",
            marker=dict(color="blue"),
        )
    )

    title = "Choice of charging per hour"
    # add average ratio of people choosing scheduled charging in the title, and total number of people
    title += f"<br>    Average fraction of users choosing scheduled charging: {df_session_choices['SCHEDULED'].mean():.2f}"
    title += f"<br>    Total number of users: {session_results.shape[0]}"

    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title="Fraction of users choosing scheduled charging",
        template="plotly_white",
    )

    fig.show()
