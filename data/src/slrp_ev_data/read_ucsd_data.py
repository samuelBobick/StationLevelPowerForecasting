from pathlib import Path
from typing import Optional

import pandas as pd

UCSD_DATA_FOLDER = Path(__file__).parent / "data" / "UCSD_garage_datasets_PF"


def read_ucsd_data(garage_name: Optional[str] = None) -> pd.DataFrame:
    """Read UCSD data

    Args:
        garage_name: Use one of the file names \
            (without the .csv) in the folder `data/UCSD_garage_datasets_PF`. \
            Defaults to None, which means the data from all garages \
            will be read.

    Returns:
        The data in the `input_data_type.DataSchema` format.
    """
    if not garage_name:
        garage_name = "All_Garages"

    data = pd.read_csv(UCSD_DATA_FOLDER / f"{garage_name}.csv")

    # convert date and time to datetime
    # data["date"] = pd.to_datetime(data["date"]).dt.tz_localize("America/Los_Angeles")
    data["date"] = pd.to_datetime(data["date"])

    return data
