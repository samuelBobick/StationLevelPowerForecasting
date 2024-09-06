import pandas as pd
import os
from pathlib import Path


def read_old_data():
    data = pd.read_csv(Path(__file__).parent / "data.csv", index_col=0)

    # convert date and time to datetime
    # data["date"] = pd.to_datetime(data["date"]).dt.tz_localize("America/Los_Angeles")
    data["date"] = pd.to_datetime(data["date"])
    data["time"] = data["time"].apply(lambda x: pd.Timedelta(hours=x))
    data["date"] = data["date"] + data["time"]
    data.drop(columns=["time"], inplace=True)

    # convert power from kW to W
    data["power"] = data["power"].round(3).apply(lambda x: x * 1000)

    return data
