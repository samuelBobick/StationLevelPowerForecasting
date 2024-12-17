import pandas as pd
import plotly.graph_objects as go
from slrp_ev_data.feature_engineering import add_missing_timesteps


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
    test_data = add_missing_timesteps(test_data)
    fig.add_trace(
        go.Scatter(
            x=test_data["date"], y=test_data["power"], mode="lines", name="Actual"
        )
    )

    prediction_scatter_mode = "lines"
    # Below we try to detect if it is a peak prediction, in which case we should
    # plot markers instead of lines
    # For timeseries prediction, we should have values every 15 minutes, so the
    # average difference between 2 values should be around 15 minutes
    # (slightly more because of potential missing values)
    if df_predictions["date"].diff().mean() > pd.Timedelta(hours=1):
        prediction_scatter_mode = "markers"
    else:
        # resample in case there is missing values.
        # We do not need to do that if we plot markers
        df_predictions = add_missing_timesteps(df_predictions)

    next_power_column_number = len(df_predictions.columns) - 1
    if "real_power" in df_predictions.columns:
        next_power_column_number -= 1
        fig.add_trace(
            go.Scatter(
                x=df_predictions["date"],
                y=df_predictions["real_power"],
                mode="markers",
                name="Real Power",
                marker=dict(symbol="diamond"),
            )
        )

    for i in range(next_power_column_number):
        fig.add_trace(
            go.Scatter(
                x=df_predictions["date"],
                y=df_predictions[f"power_{i}"],
                mode=prediction_scatter_mode,
                name=f"Forecast_{i}",
            )
        )

    fig.update_layout(
        title=f"Forecast for {number_of_days} days",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    fig.show()
