"""Bridge helpers between legacy artifacts and canonical MVP contracts."""

from statarb.bridges.historical import (
    HistoricalSmokeBundle,
    build_default_historical_smoke,
    build_historical_pair_bridge_from_alor_datasets,
)

__all__ = [
    "HistoricalSmokeBundle",
    "build_default_historical_smoke",
    "build_historical_pair_bridge_from_alor_datasets",
]
