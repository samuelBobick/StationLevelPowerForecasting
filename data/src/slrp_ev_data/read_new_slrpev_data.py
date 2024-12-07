from pathlib import Path

import pandas as pd


def read_new_slrpev_data():
    data = pd.read_csv(
        Path(__file__).parent / "data" / "power_df_2008-2406_v241206_15min.csv",
    )

    # convert date and remove timezone
    data["recordTimestamp"] = data["recordTimestamp"] = (
        pd.to_datetime(data["recordTimestamp"], utc=True)
        .dt.tz_convert("America/Los_Angeles")
        .dt.tz_localize(None)
    )

    data["workday"] = (data["isWeekend"] == False).astype("int64")

    data = data.rename(columns={"recordTimestamp": "date", "totalPower": "power"})

    data = data[["date", "power", "workday"]]
    data = data.dropna(subset=["power"])

    return data
