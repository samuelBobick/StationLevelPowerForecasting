import pandas as pd
from dateutil.relativedelta import FR
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class USAcademicHolidayCalendar(AbstractHolidayCalendar):
    """
    US Academic Government Holiday Calendar based on rules specified by:
    - [UC Berkeley](https://hr.berkeley.edu/hr-network/personnel-resources/holiday-sick-vacation/holiday-schedule)
    - [UC San Diego](https://blink.ucsd.edu/HR/benefits/time-off/holidays.html)
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        Holiday(
            "Cesar Chavez Day", month=3, day=31, offset=pd.DateOffset(weekday=FR(-1))
        ),
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday(
            "Day after Thanksgiving",
            month=11,
            day=1,
            offset=pd.DateOffset(weekday=FR(4)),
        ),
        Holiday("Christmas Eve", month=12, day=24, observance=nearest_workday),
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("New Year's Eve", month=12, day=31, observance=nearest_workday),
    ]
