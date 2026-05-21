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
from statarb.data.alor_discovery import (
    DiscoveredAlorHistoryDatasetManifest,
    discover_history_dataset_manifests,
    find_latest_history_dataset_manifest,
)
from statarb.data.alor_history import LoadedAlorHistoryBarsDataset, load_history_bars_dataset
from statarb.data.alor_reference import AlorSymbolBoardEnrichment, resolve_symbol_board_enrichment

__all__ = [
    "ALOR_MARKET_DATA_CONTRACT_VERSION",
    "AlorHttpResponse",
    "AlorSymbolBoardEnrichment",
    "AlorDatasetManifest",
    "DiscoveredAlorHistoryDatasetManifest",
    "LoadedAlorHistoryBarsDataset",
    "PARQUET_PART_FILENAME",
    "PublicAvailableBoardsFetchResult",
    "PublicHistoryFetchResult",
    "PublicSecuritiesFetchResult",
    "RequestsAlorHttpClient",
    "StoredArtifact",
    "StoredDatasetBatch",
    "discover_history_dataset_manifests",
    "fetch_public_available_boards",
    "fetch_public_history",
    "fetch_public_securities",
    "find_latest_history_dataset_manifest",
    "ingest_history_payload",
    "load_history_bars_dataset",
    "persist_raw_alor_payload",
    "resolve_symbol_board_enrichment",
    "write_history_bars_dataset",
    "write_quote_snapshot_dataset",
    "write_security_snapshot_dataset",
]


def __getattr__(name: str):
    if name in {
        "AlorHttpResponse",
        "PublicAvailableBoardsFetchResult",
        "PublicHistoryFetchResult",
        "PublicSecuritiesFetchResult",
        "RequestsAlorHttpClient",
        "fetch_public_available_boards",
        "fetch_public_history",
        "fetch_public_securities",
    }:
        from statarb.data.alor_fetch import (
            AlorHttpResponse,
            PublicAvailableBoardsFetchResult,
            PublicHistoryFetchResult,
            PublicSecuritiesFetchResult,
            RequestsAlorHttpClient,
            fetch_public_available_boards,
            fetch_public_history,
            fetch_public_securities,
        )

        exports = {
            "AlorHttpResponse": AlorHttpResponse,
            "PublicAvailableBoardsFetchResult": PublicAvailableBoardsFetchResult,
            "PublicHistoryFetchResult": PublicHistoryFetchResult,
            "PublicSecuritiesFetchResult": PublicSecuritiesFetchResult,
            "RequestsAlorHttpClient": RequestsAlorHttpClient,
            "fetch_public_available_boards": fetch_public_available_boards,
            "fetch_public_history": fetch_public_history,
            "fetch_public_securities": fetch_public_securities,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
