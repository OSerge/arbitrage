"""Data-layer helpers for vendor-backed MVP artifacts."""

from statarb.data.alor_storage import (
    ALOR_MARKET_DATA_CONTRACT_VERSION,
    AlorDatasetManifest,
    PARQUET_PART_FILENAME,
    StoredArtifact,
    StoredDatasetBatch,
    ingest_history_payload,
    persist_raw_alor_payload,
    write_history_bars_dataset,
    write_quote_snapshot_dataset,
    write_security_snapshot_dataset,
)
from statarb.data.alor_history import LoadedAlorHistoryBarsDataset, load_history_bars_dataset

__all__ = [
    "ALOR_MARKET_DATA_CONTRACT_VERSION",
    "AlorHttpResponse",
    "AlorDatasetManifest",
    "LoadedAlorHistoryBarsDataset",
    "PARQUET_PART_FILENAME",
    "PublicHistoryFetchResult",
    "RequestsAlorHttpClient",
    "StoredArtifact",
    "StoredDatasetBatch",
    "fetch_public_history",
    "ingest_history_payload",
    "load_history_bars_dataset",
    "persist_raw_alor_payload",
    "write_history_bars_dataset",
    "write_quote_snapshot_dataset",
    "write_security_snapshot_dataset",
]


def __getattr__(name: str):
    if name in {
        "AlorHttpResponse",
        "PublicHistoryFetchResult",
        "RequestsAlorHttpClient",
        "fetch_public_history",
    }:
        from statarb.data.alor_fetch import (
            AlorHttpResponse,
            PublicHistoryFetchResult,
            RequestsAlorHttpClient,
            fetch_public_history,
        )

        exports = {
            "AlorHttpResponse": AlorHttpResponse,
            "PublicHistoryFetchResult": PublicHistoryFetchResult,
            "RequestsAlorHttpClient": RequestsAlorHttpClient,
            "fetch_public_history": fetch_public_history,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
