import pandas as pd
import plotly.graph_objects as go


def visualize_forecast(
    test_data: pd.DataFrame, forecast: pd.DataFrame, number_of_days: int
) -> None:
    """
    Visualize the forecasted values.

    Args:
        test_data: The test data.
        forecast: The forecasted values.
        number_of_days: Number of days to forecast.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_data["date"], y=test_data["power"], mode="lines", name="Actual"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_data["date"][-len(forecast) :],
            y=forecast,
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
