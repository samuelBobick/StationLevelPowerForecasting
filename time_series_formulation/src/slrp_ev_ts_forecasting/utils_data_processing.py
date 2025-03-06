import pandas as pd
from slrp_ev_data.feature_engineering import (
    convert_date_from_datetime_to_int,
    reverse_feature_engineering,
)

from slrp_ev_ts_forecasting.default_parameters import TypeScalingMode


def reverse_engineer_forecast(
    df_test_example,
    df_predictions,
    scaling_mode: TypeScalingMode,
    scaling_parameters: tuple[pd.Series, pd.Series] | pd.DataFrame | None,
) -> pd.DataFrame:
    # Reverse engineer the forecast to get the original features back
    # initialize final dataframe
    df_reversed_predictions = pd.DataFrame()
    df_reversed_predictions["date"] = df_predictions["date"]

    # convert from float32 to int64
    df_predictions["date"] = convert_date_from_datetime_to_int(df_predictions["date"])

    list_columns = list(df_predictions.columns)
    for i, col_name in enumerate(list_columns):
        if col_name == "date":
            continue

        # merge_asof performs a left merge with the closest date
        df_reverse_helper = pd.merge_asof(
            df_predictions[["date", col_name]],
            df_test_example.drop(columns=["power"]),
            on="date",
            direction="nearest",
        ).rename(columns={col_name: "power"})
        df_reverse_helper = df_reverse_helper.dropna(subset=["power"])
        df_reverse_helper = reverse_feature_engineering(
            df_reverse_helper,
            scaling_mode,
            scaling_parameters,
            bypass_output_validation=True,
        )

        df_reverse_helper = df_reverse_helper[["date", "power"]]
        helper_date_mask = df_reversed_predictions["date"].isin(
            df_reverse_helper["date"]
        )
        df_reversed_predictions.loc[helper_date_mask, col_name] = df_reverse_helper[
            "power"
        ]

    return df_reversed_predictions
