from pathlib import Path

import pandas as pd


def read_new_slrpev_data(keep_all_columns=False) -> pd.DataFrame:
    """Reads the new SLRP EV data.	

    Args:
        keep_all_columns: Whether to keep all columns. This MUST be set to False to pass \
            the feature engineering validation. Defaults to False.

    Returns:
        The new SLRP EV data.
    """
    data = pd.read_csv(
        Path(__file__).parent / "data" / "power_df_2008-2406_v241220_15min.csv",
    )

    # convert date and remove timezone
    data["recordTimestamp"] = pd.to_datetime(data["recordTimestamp"])

    data["workday"] = (~data["isWeekend"]).astype("int64")

    data = data.rename(columns={"recordTimestamp": "date", "totalPower": "power"})

    if not keep_all_columns:
        data = data[["date", "power", "workday"]]
    data = data.dropna(subset=["power"])

    # remove data from the beginning because it is not very representative
    data = data.loc[data["date"] >= "2021-03-01"]

    return data
