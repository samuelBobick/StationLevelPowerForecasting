import pandas as pd
import plotly.graph_objects as go


def visualize_forecast(
    test_data: pd.DataFrame,
    df_predictions: pd.DataFrame,
    number_of_days: int,
) -> None:
    """
    Visualize the forecasted values.

    Args:
        test_data: The test data.
        forecast: The forecasted values.
        forecast_dates: The dates for the forecasted values.
    """
    fig = go.Figure()
    # First we plot the true test data that we are trying to predict
    # resample in case there is missing values
    test_data = test_data.set_index("date").resample("15min").mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=test_data["date"], y=test_data["power"], mode="lines", name="Actual"
        )
    )

    # resample in case there is missing values
    df_predictions = (
        df_predictions.set_index("date").resample("15min").mean().reset_index()
    )

    next_power_column_number = len(df_predictions.columns) - 1
    for i in range(next_power_column_number):
        fig.add_trace(
            go.Scatter(
                x=df_predictions["date"],
                y=df_predictions[f"power_{i}"],
                mode="lines",
                name=f"Forecast_{i}",
            )
        )

    fig.update_layout(
        title=f"Forecast for {number_of_days} days",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    fig.show()
