from typing import Literal, TypedDict

import numpy as np

# link for source of rates: https://www.pge.com/tariffs/en/rate-information/electric-rates.html#accordion-a84c67dc1e-item-69d101345a


class TypeTariff(TypedDict):
    cost_dc: float  # in cents/kW
    TOU: np.ndarray  # in cents/kWh


TypeTariffName = Literal[
    "A-10 Primary June 2023", "Original SLRP-EV", "BEV2S Secondary June 2023"
]

# TOU A-10 Primary Tariff June 2023
tou_a20p = np.ones((96,)) * 24.7  # on-peak cents/kWh
tou_a20p[:34] = 22.2  # off-peak
tou_a20p[86:] = 22.2  # 9:30pm super off-peak
tariff_A_10_primary_062023: TypeTariff = {"cost_dc": 1942, "TOU": tou_a20p}

# Original Slrp-EV Tariffs
tou_slrp_ev = np.ones((96,)) * 17.5  # off-peak cents/kWh
tou_slrp_ev[64:84] = 36.7  # 4 pm - 9 pm peak
tou_slrp_ev[36:56] = 14.9  # 9 am - 2 pm super off-peak
tariff_slrp_ev: TypeTariff = {
    "cost_dc": 500,
    "TOU": np.concatenate([tou_slrp_ev, tou_slrp_ev, tou_slrp_ev]),
}

# PGE BEV2S Secondary June 2023
tou_bev2s = np.ones((96,)) * 18.6  # off-peak cents/kWh
tou_bev2s[64:84] = 39.9  # 4 pm - 9 pm peak
tou_bev2s[36:56] = 16.3  # 9 am - 2 pm super off-peak
tariff_bev2s_secondary_062023: TypeTariff = {
    "cost_dc": 191,
    "TOU": np.concatenate([tou_bev2s, tou_bev2s, tou_bev2s]),
}

DICT_TARIFFS = {
    "A-10 Primary June 2023": tariff_A_10_primary_062023,
    "Original SLRP-EV": tariff_slrp_ev,
    "BEV2S Secondary June 2023": tariff_bev2s_secondary_062023,
}
