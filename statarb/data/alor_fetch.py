from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import requests

from statarb.adapters.alor.http_market_data import (
    AlorObjectFormat,
    HistoricalBarsRequest,
    HttpRequestShape,
)
from statarb.config import AlorContour
from statarb.data.alor_storage import StoredDatasetBatch, ingest_history_payload

DEFAULT_PUBLIC_DATA_DELAY_MINUTES = 15
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class AlorHttpResponse:
    status_code: int
    payload: Any


class AlorHttpClient(Protocol):
    def execute(self, request: HttpRequestShape, *, timeout_seconds: float) -> AlorHttpResponse: ...


class RequestsAlorHttpClient:
    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def execute(self, request: HttpRequestShape, *, timeout_seconds: float) -> AlorHttpResponse:
        response = self._session.request(
            method=request.method,
            url=request.url,
            headers=dict(request.headers),
            params=dict(request.params),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Alor returned a non-JSON response for {request.method} {request.url}."
            ) from exc
        return AlorHttpResponse(status_code=response.status_code, payload=payload)


@dataclass(frozen=True, slots=True)
class PublicHistoryFetchResult:
    request: HistoricalBarsRequest
    http_request: HttpRequestShape
    http_status_code: int
    fetched_at_utc: datetime
    batch: StoredDatasetBatch


def fetch_public_history(
    *,
    storage_root: Path,
    request: HistoricalBarsRequest,
    contour: AlorContour | str = AlorContour.LIVE,
    http_client: AlorHttpClient | None = None,
    fetched_at_utc: datetime | None = None,
    request_id: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> PublicHistoryFetchResult:
    http_request = request.to_http_request(contour=contour, access_token=None)
    response = (http_client or RequestsAlorHttpClient()).execute(
        http_request,
        timeout_seconds=timeout_seconds,
    )
    payload = _mapping_payload(response.payload)
    observed_at = _coerce_utc_datetime(fetched_at_utc or datetime.now(tz=UTC))
    batch = ingest_history_payload(
        storage_root=Path(storage_root),
        request=request,
        payload=payload,
        fetched_at_utc=observed_at,
        request_id=request_id,
        authorized=False,
        data_delay_minutes=DEFAULT_PUBLIC_DATA_DELAY_MINUTES,
        response_format=request.format,
    )
    return PublicHistoryFetchResult(
        request=request,
        http_request=http_request,
        http_status_code=response.status_code,
        fetched_at_utc=observed_at,
        batch=batch,
    )


def main(argv: list[str] | None = None, *, http_client: AlorHttpClient | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "history":
        parser.error(f"Unsupported command: {args.command!r}")

    request = HistoricalBarsRequest(
        symbol=args.symbol,
        exchange=args.exchange,
        instrument_group=args.instrument_group,
        tf=args.timeframe,
        from_time=_parse_cli_timestamp(args.from_time),
        to_time=_parse_cli_timestamp(args.to_time),
        count_back=args.count_back,
        untraded=args.untraded,
        split_adjust=not args.no_split_adjust,
        format=AlorObjectFormat.coerce(args.object_format),
    )
    result = fetch_public_history(
        storage_root=Path(args.storage_root),
        request=request,
        contour=args.contour,
        http_client=http_client,
        request_id=args.request_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "contour": AlorContour.coerce(args.contour).value,
                "dataset_id": result.batch.manifest.dataset_id,
                "manifest_uri": result.batch.manifest_uri,
                "raw_storage_uri": result.batch.raw_artifact.storage_uri,
                "row_count": result.batch.manifest.row_count,
                "storage_uri": result.batch.normalized_artifact.storage_uri,
                "symbol": request.symbol,
                "timeframe": request.tf,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m statarb.data.alor_fetch",
        description="Fetch delayed public Alor historical bars into raw+normalized MVP storage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    history = subparsers.add_parser("history", help="Fetch historical bars from public Alor HTTP API.")
    history.add_argument("--symbol", required=True)
    history.add_argument("--exchange", default="MOEX")
    history.add_argument("--instrument-group")
    history.add_argument("--timeframe", default="D")
    history.add_argument("--from", dest="from_time", required=True)
    history.add_argument("--to", dest="to_time", required=True)
    history.add_argument("--count-back", type=int)
    history.add_argument("--untraded", action="store_true")
    history.add_argument("--no-split-adjust", action="store_true")
    history.add_argument(
        "--object-format",
        default=AlorObjectFormat.HEAVY.value,
        choices=[member.value for member in AlorObjectFormat],
    )
    history.add_argument(
        "--contour",
        default=AlorContour.LIVE.value,
        choices=[member.value for member in AlorContour],
    )
    history.add_argument("--request-id")
    history.add_argument("--storage-root", default=str(_project_root()))
    history.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def _parse_cli_timestamp(value: str) -> int:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    normalized = stripped.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return int(parsed.timestamp())


def _mapping_payload(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raise ValueError("Public Alor history fetch expected a JSON object payload.")


def _coerce_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
