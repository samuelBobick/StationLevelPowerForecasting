from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
