import pandas as pd
import pandera as pa

DataSchema = pa.DataFrameSchema(
    {
        # for timezone aware, use: pa.Column(pd.DatetimeTZDtype(unit="ns", tz="America/Los_Angeles")),
        "date": pa.Column(pa.Timestamp),
        "power": pa.Column(
            pa.Float, pa.Check.greater_than_or_equal_to(0), nullable=True
        ),
    }
)


FeaturedEngineeredSchema = pa.DataFrameSchema(
    {
        # make sure the date is unit of seconds, not nanoseconds
        "date": pa.Column(
            pa.Int,
            pa.Check.in_range(
                min_value=pd.to_datetime("2010-01-01").timestamp(),
                max_value=pd.to_datetime("2030-01-01").timestamp(),
            ),
        ),
        "power": pa.Column(
            pa.Float, pa.Check.greater_than_or_equal_to(0), nullable=False
        ),
        "time_window": pa.Column(
            pa.Int32, pa.Check.isin(range(6)), nullable=False, required=False
        ),
    }
)
