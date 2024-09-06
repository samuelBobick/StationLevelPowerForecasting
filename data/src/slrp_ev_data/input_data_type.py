import pandera as pa
import pandas as pd

DataSchema = pa.DataFrameSchema(
    {
        # for timezone aware, use: pa.Column(pd.DatetimeTZDtype(unit="ns", tz="America/Los_Angeles")),
        "date": pa.Column(pa.Timestamp),
        "power": pa.Column(
            pa.Float, pa.Check.greater_than_or_equal_to(0), nullable=True
        ),
    }
)
