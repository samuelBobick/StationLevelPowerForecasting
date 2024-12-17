import pandas as pd
import plotly.graph_objects as go
from slrp_ev_data.feature_engineering import add_missing_timesteps


def visualize_forecast(
    test_data: pd.DataFrame,
    forecast: pd.Series,
    number_of_days: int,
    forecast_dates: pd.Series,
) -> None:
    """
    Visualize the forecasted values.

    Args:
        test_data: The test data.
        forecast: The forecasted values.
        forecast_dates: The dates for the forecasted values.
    """
    fig = go.Figure()
    # resample in case there is missing values
    test_data = add_missing_timesteps(test_data)
    fig.add_trace(
        go.Scatter(
            x=test_data["date"], y=test_data["power"], mode="lines", name="Actual"
        )
    )

    # resample in case there is missing values
    forecast_data = add_missing_timesteps(
        pd.DataFrame({"date": forecast_dates, "power": forecast})
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["power"],
            mode="lines",
            name="Forecast",
        )
    )
    fig.update_layout(
        title=f"Forecast for {number_of_days} days",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    fig.show()
