# Domain Data Contracts

Статус: foundation

## Contract: InstrumentRef

- layer: data
- purpose: canonical instrument identity used by research, portfolio, execution, and replay.
- required-fields:
  - `symbol`
  - `exchange`
  - `instrument_type`
- invariants:
  - `symbol` is non-empty.
  - `exchange` is non-empty.
  - instrument identity is reference data, not broker-session state.

## Contract: DatasetSnapshot

- layer: data
- purpose: stable reference to a reproducible research dataset input.
- required-fields:
  - `dataset_id`
  - `dataset_kind`
  - `as_of`
  - `storage_uri`
- invariants:
  - dataset identity is immutable after publication.
  - `storage_uri` points to artifact storage, not an in-memory object.

## Notes

- `InstrumentRef` is the only mandatory executable data object for the current workstream.
- richer symbol-master fields may be added later without changing current research/execution contracts.
