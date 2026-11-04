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

- ADR 0005 adds `InstrumentSpec` and `InstrumentCatalog` as effective-dated instrument economics. See [causal futures contract](causal-futures-research.md); instrument identity remains `InstrumentRef`.

- `InstrumentRef` is the mandatory identity object; funded financial research additionally requires the historical `InstrumentSpec`/`InstrumentCatalog` described above. Read-only identity mapping does not require financial assumptions.
- richer symbol-master fields may be added later without changing current research/execution contracts.
