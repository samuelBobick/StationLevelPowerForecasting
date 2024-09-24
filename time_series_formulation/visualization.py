import pandas as pd
import plotly.graph_objects as go


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
    fig.add_trace(
        go.Scatter(
            x=test_data["date"], y=test_data["power"], mode="lines", name="Actual"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
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
