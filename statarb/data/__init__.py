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

__all__ = [
    "ALOR_MARKET_DATA_CONTRACT_VERSION",
    "AlorDatasetManifest",
    "PARQUET_PART_FILENAME",
    "StoredArtifact",
    "StoredDatasetBatch",
    "ingest_history_payload",
    "persist_raw_alor_payload",
    "write_history_bars_dataset",
    "write_quote_snapshot_dataset",
    "write_security_snapshot_dataset",
]
"""Data domain: ingestion, reference data, schemas и dataset assembly."""
