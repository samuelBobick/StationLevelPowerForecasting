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
        Path(__file__).parent / "data" / "power_df_2008-2406_v250110_15min.csv",
        low_memory=False,
    )

    # convert date and remove timezone
    data["recordTimestamp"] = pd.to_datetime(data["recordTimestamp"])

    # data["workday"] = (~data["isWeekend"]).astype(int) # it is recomputed in feature engineering

    data = data.rename(columns={"recordTimestamp": "date", "totalPower": "power"})

    if not keep_all_columns:
        data = data[["date", "power", "number_of_evses_available"]]  # "workday"
    data = data.dropna(subset=["power"])

    # Reduce the float precision and to save memory
    data = data.astype({"number_of_evses_available": "float32"})

    # remove data from the beginning because it is not very representative
    data = data.loc[data["date"] >= "2021-01-01"]

    return data
