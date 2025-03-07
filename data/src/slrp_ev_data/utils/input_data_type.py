import pandas as pd
import pandera as pa

DataSchema = pa.DataFrameSchema(
    {
        # for timezone aware, use: pa.Column(pd.DatetimeTZDtype(unit="ns", tz="America/Los_Angeles")),
        "date": pa.Column(pa.Timestamp),
        "power": pa.Column(
            pa.Float,
            [
                pa.Check.greater_than_or_equal_to(0),
                # Make sure that the data is in W (and not kW).
                # a charger has a max power of 6.6 kW,
                # so 5_000 W is a reasonable threshold
                pa.Check(
                    lambda x: x.max() > 5_000,
                    element_wise=False,
                    error="The data does not seem to be in W.",
                ),
            ],
            nullable=False,
        ),
        # The column "workday" is a binary column that indicates whether the day is a workday or not.
        # it is optional, since we actually recompute it in the feature engineering
        "workday": pa.Column(pa.Int32, nullable=False, required=False),
        "number_of_evses_available": pa.Column(pa.Float32, nullable=False),
    }
)


FeaturedEngineeredSchema = pa.DataFrameSchema(
    {
        # make sure the date is unit of seconds, not nanoseconds
        "date": pa.Column(
            pa.Int64,
            pa.Check.in_range(
                min_value=pd.to_datetime("2010-01-01").timestamp(),
                max_value=pd.to_datetime("2030-01-01").timestamp(),
            ),
        ),
        "power": pa.Column(
            pa.Float32, nullable=True
        ),  # , pa.Check.greater_than_or_equal_to(0) not useable if we standardize
        "number_of_evses_available": pa.Column(pa.Float32, nullable=True),
        "time_window": pa.Column(
            pa.Int32, pa.Check.isin(range(6)), nullable=False, required=False
        ),
        "workday_0": pa.Column(pa.Int32, nullable=False),
        "workday_96": pa.Column(pa.Int32, nullable=False, required=False),
        "workday_192": pa.Column(pa.Int32, nullable=False, required=False),
        # Add other workday windows if you do a longer horizon than 48 hours
        "Day sin": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
        "Day cos": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
        "Week sin": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
        "Week cos": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
        "Year sin": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
        "Year cos": pa.Column(pa.Float32, pa.Check.in_range(min_value=-1, max_value=1)),
    }
)
