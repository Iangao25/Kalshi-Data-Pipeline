"""Fast public-API utilities for settled Kalshi sports markets and trades.

The functions in this module are deliberately notebook-friendly: they return
pandas DataFrames and do not read or write local datasets on their own.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import random
import re
import threading
import time as time_module
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests


DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc
FINAL_RESULTS = {"yes", "no"}


def _progress(enabled: bool, message: str, started: float | None = None) -> None:
    """Print a timestamped, immediately flushed notebook progress message."""

    if not enabled:
        return
    elapsed = ""
    if started is not None:
        elapsed = f" | elapsed {time_module.monotonic() - started:,.1f}s"
    print(f"[{datetime.now(NY_TZ):%H:%M:%S}] {message}{elapsed}", flush=True)


class KalshiAPIError(RuntimeError):
    """Raised when a Kalshi public API request cannot be completed."""


class _RateLimiter:
    """Thread-safe, evenly spaced request limiter."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._next_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time_module.monotonic()
            delay = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self._interval
        if delay:
            time_module.sleep(delay)


class KalshiPublicClient:
    """Small public REST client with pagination, pacing, and retry handling.

    ``requests_per_second=10`` is conservative relative to Kalshi's documented
    authenticated Basic read budget. Public limits are not published, so 429s
    are additionally handled with exponential backoff.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        requests_per_second: float = 10.0,
        timeout: float = 30.0,
        max_retries: int = 7,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiter = _RateLimiter(requests_per_second)
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "kalshi-sports-research/1.0"})
            self._local.session = session
        return session

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                response = self._session().get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt == self.max_retries:
                        response.raise_for_status()
                    delay = min(8.0, 0.25 * (2**attempt)) + random.uniform(0, 0.15)
                    time_module.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise KalshiAPIError(f"Expected an object from {response.url}")
                return payload
            except (requests.RequestException, ValueError, KalshiAPIError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = min(8.0, 0.25 * (2**attempt)) + random.uniform(0, 0.15)
                time_module.sleep(delay)
        raise KalshiAPIError(f"GET {url} failed after retries: {last_error}") from last_error

    def pages(
        self,
        path: str,
        item_key: str,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        query = dict(params or {})
        while True:
            payload = self.get(path, query)
            items = payload.get(item_key, [])
            if not isinstance(items, list):
                raise KalshiAPIError(f"Expected list field {item_key!r} from {path}")
            yield items
            cursor = payload.get("cursor")
            if not cursor:
                return
            query["cursor"] = cursor

    def historical_cutoff(self) -> dict[str, datetime]:
        payload = self.get("historical/cutoff")
        return {key: _parse_datetime(value) for key, value in payload.items() if value}


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        # Older Python versions only accept 3 or 6 fractional-second digits;
        # Kalshi may emit any width up to microseconds (for example, 5 digits).
        text = re.sub(
            r"\.(\d+)(?=([+-]\d{2}:\d{2})?$)",
            lambda match: "." + (match.group(1) + "000000")[:6],
            text,
        )
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _optional_datetime(value: Any) -> datetime | None:
    if value is None or value == "" or pd.isna(value):
        return None
    return _parse_datetime(value)


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.astimezone(NY_TZ).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _ny_day_bounds(
    start_date: str | date | datetime,
    end_date: str | date | datetime | None,
) -> tuple[datetime, datetime, date, date]:
    start_day = _as_date(start_date)
    if end_date is None:
        end_day = datetime.now(NY_TZ).date() - timedelta(days=1)
    else:
        end_day = _as_date(end_date)
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")
    start_utc = datetime.combine(start_day, time.min, NY_TZ).astimezone(UTC)
    end_exclusive_utc = datetime.combine(end_day + timedelta(days=1), time.min, NY_TZ).astimezone(UTC)
    return start_utc, end_exclusive_utc, start_day, end_day


def fetch_sports_series(client: KalshiPublicClient) -> pd.DataFrame:
    """Return all series whose Kalshi category is exactly ``Sports``."""

    series = client.get("series", {"category": "Sports"}).get("series", [])
    frame = pd.DataFrame(series)
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "title", "category", "tags"])
    return frame.loc[frame["category"].eq("Sports")].reset_index(drop=True)


def _market_settlement(market: Mapping[str, Any]) -> datetime | None:
    return _optional_datetime(market.get("settlement_ts"))


def fetch_historical_markets_since(
    client: KalshiPublicClient,
    start_utc: datetime,
    *,
    early_stop: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch archived markets that could have occurred on/after ``start_utc``.

    Kalshi currently exposes no archive timestamp filter; unknown query fields
    are silently ignored. The archive cursor is observed to run newest to oldest
    by settlement. With ``early_stop=True`` this function stops after the first
    full page whose settlements precede ``start_utc``. Page ordering is checked
    as it runs; if it reverses, pagination continues exhaustively.
    """

    kept: list[dict[str, Any]] = []
    previous_page_max: datetime | None = None
    ordering_ok = True
    page_number = 0
    for page in client.pages("historical/markets", "markets", {"limit": 1000}):
        page_number += 1
        settlements = [dt for market in page if (dt := _market_settlement(market))]
        page_max = max(settlements) if settlements else None
        if previous_page_max and page_max and page_max > previous_page_max + timedelta(seconds=1):
            ordering_ok = False
        if page_max:
            previous_page_max = page_max

        kept.extend(
            market
            for market in page
            if (settled := _market_settlement(market)) is None or settled >= start_utc
        )
        if page_number == 1 or page_number % 10 == 0:
            _progress(
                verbose,
                f"Historical markets: page {page_number:,}, {len(kept):,} potentially relevant rows retained",
            )
        if early_stop and ordering_ok and settlements and max(settlements) < start_utc:
            _progress(verbose, f"Historical scan reached settlements before {start_utc.isoformat()}; stopping")
            break
    _progress(verbose, f"Historical market scan complete: {len(kept):,} rows")
    return kept


def fetch_live_result_candidates(
    client: KalshiPublicClient,
    start_utc: datetime,
    *,
    refresh_since: datetime | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch live-tier rows that may now contain a Yes/No result.

    On an initial pull, finalized markets are selected by settlement time and
    non-finalized closed/determined/disputed/amended markets by close time. On
    an incremental pull, metadata updates catch new or changed determinations,
    while a settlement-time scan provides an additional finalization safeguard.
    The caller performs the definitive Yes/No result check.
    """

    queries: list[tuple[str, dict[str, Any]]] = []
    if refresh_since is None:
        queries.extend(
            [
                (
                    "finalized",
                    {
                        "status": "settled",
                        "min_settled_ts": int(start_utc.timestamp()),
                        "limit": 1000,
                    },
                ),
                (
                    "closed/non-finalized",
                    {
                        "status": "closed",
                        "min_close_ts": int(start_utc.timestamp()),
                        "limit": 1000,
                    },
                ),
            ]
        )
    else:
        queries.extend(
            [
                (
                    "recently updated",
                    {"min_updated_ts": int(refresh_since.timestamp()), "limit": 1000},
                ),
                (
                    "recently finalized",
                    {
                        "status": "settled",
                        "min_settled_ts": int(refresh_since.timestamp()),
                        "limit": 1000,
                    },
                ),
            ]
        )

    found: dict[str, dict[str, Any]] = {}
    for label, params in queries:
        query_count = 0
        for page_number, page in enumerate(client.pages("markets", "markets", params), start=1):
            query_count += len(page)
            found.update({market["ticker"]: market for market in page})
            if page_number == 1 or page_number % 10 == 0:
                _progress(
                    verbose,
                    f"Live {label} markets: page {page_number:,}, {query_count:,} rows",
                )
        _progress(verbose, f"Live {label} scan complete: {query_count:,} rows")
    _progress(verbose, f"Combined unique live result candidates: {len(found):,}")
    return list(found.values())


def _ticker_batches(tickers: Iterable[str], max_count: int = 100, max_chars: int = 6000) -> Iterator[list[str]]:
    batch: list[str] = []
    chars = 0
    for ticker in dict.fromkeys(str(t) for t in tickers if t):
        extra = len(ticker) + (1 if batch else 0)
        if batch and (len(batch) >= max_count or chars + extra > max_chars):
            yield batch
            batch, chars = [], 0
        batch.append(ticker)
        chars += extra
    if batch:
        yield batch


def fetch_markets_by_tickers(
    client: KalshiPublicClient,
    tickers: Iterable[str],
    *,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch ticker metadata from live first, then query archive for misses."""

    found: dict[str, dict[str, Any]] = {}
    requested = list(dict.fromkeys(str(t) for t in tickers if t))
    live_batches = list(_ticker_batches(requested))
    _progress(verbose, f"Resolving {len(requested):,} unique combo-leg markets in {len(live_batches):,} live batches")
    for batch_number, batch in enumerate(live_batches, start=1):
        payload = client.get("markets", {"tickers": ",".join(batch), "limit": 1000})
        found.update({market["ticker"]: market for market in payload.get("markets", [])})
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(live_batches):
            _progress(verbose, f"Combo legs, live batch {batch_number:,}/{len(live_batches):,}: {len(found):,} found")
    missing = [ticker for ticker in requested if ticker not in found]
    historical_batches = list(_ticker_batches(missing))
    if historical_batches:
        _progress(verbose, f"Checking archive for {len(missing):,} combo legs missing from live data")
    for batch_number, batch in enumerate(historical_batches, start=1):
        payload = client.get("historical/markets", {"tickers": ",".join(batch), "limit": 1000})
        found.update({market["ticker"]: market for market in payload.get("markets", [])})
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(historical_batches):
            _progress(verbose, f"Combo legs, archive batch {batch_number:,}/{len(historical_batches):,}: {len(found):,} total found")
    return found


@dataclass(frozen=True)
class _SportsSeriesMatcher:
    tickers: tuple[str, ...]

    @classmethod
    def build(cls, tickers: Iterable[str]) -> "_SportsSeriesMatcher":
        return cls(tuple(sorted(set(str(t) for t in tickers if t))))

    def match(self, event_ticker: str | None) -> str | None:
        if not event_ticker or not self.tickers:
            return None
        index = bisect_right(self.tickers, event_ticker) - 1
        if index < 0:
            return None
        candidate = self.tickers[index]
        if event_ticker == candidate or event_ticker.startswith(candidate + "-"):
            return candidate
        return None


def _sports_series_for_market(
    market: Mapping[str, Any],
    matcher: _SportsSeriesMatcher,
) -> list[str]:
    legs = market.get("mve_selected_legs") or []
    is_combo = bool(market.get("mve_collection_ticker") or legs)
    if not is_combo:
        direct = matcher.match(market.get("event_ticker"))
        return [direct] if direct else []

    # A combo qualifies only when every leg maps to a Kalshi Sports series.
    # Missing leg metadata or even one non-Sports leg rejects the combo.
    if not legs:
        return []
    matches: set[str] = set()
    for leg in legs:
        matched = matcher.match(leg.get("event_ticker"))
        if not matched:
            return []
        matches.add(matched)
    return sorted(matches)


def _resolve_occurrences(
    markets: list[dict[str, Any]],
    client: KalshiPublicClient,
    *,
    verbose: bool = False,
    occurrence_cache: dict[str, str | None] | None = None,
) -> None:
    occurrence_cache = occurrence_cache if occurrence_cache is not None else {}
    missing_combo = [
        market
        for market in markets
        if not market.get("occurrence_datetime") and market.get("mve_selected_legs")
    ]
    leg_tickers = [
        leg.get("market_ticker")
        for market in missing_combo
        for leg in market.get("mve_selected_legs") or []
        if leg.get("market_ticker") not in occurrence_cache
    ]
    _progress(verbose, f"Markets needing combo-leg occurrence fallback: {len(missing_combo):,}")
    legs = fetch_markets_by_tickers(client, leg_tickers, verbose=verbose) if leg_tickers else {}
    for ticker in set(ticker for ticker in leg_tickers if ticker):
        occurrence_cache[ticker] = legs.get(ticker, {}).get("occurrence_datetime")
    for market in markets:
        if market.get("occurrence_datetime"):
            market["occurrence_source"] = "market"
            continue
        occurrences = [
            occurrence_cache.get(leg.get("market_ticker"))
            for leg in market.get("mve_selected_legs") or []
            if occurrence_cache.get(leg.get("market_ticker"))
        ]
        if occurrences:
            market["occurrence_datetime"] = max(occurrences, key=_parse_datetime)
            market["occurrence_source"] = "latest_combo_leg"
        else:
            market["occurrence_source"] = "missing"


MARKET_COLUMNS = [
    "occurrence_date_ny",
    "ticker",
    "event_ticker",
    "is_combo",
    "sports_series_tickers",
    "title",
    "result",
    "market_volume_contracts",
    "occurrence_datetime",
    "occurrence_source",
    "open_time",
    "settlement_ts",
    "status",
    "mve_collection_ticker",
    "mve_selected_legs",
]


def _iter_historical_market_pages_since(
    client: KalshiPublicClient,
    start_utc: datetime,
    *,
    early_stop: bool,
    verbose: bool,
) -> Iterator[list[dict[str, Any]]]:
    """Yield only relevant archive rows without retaining prior pages."""

    previous_page_max: datetime | None = None
    ordering_ok = True
    retained = 0
    for page_number, page in enumerate(
        client.pages("historical/markets", "markets", {"limit": 1000}), start=1
    ):
        settlements = [dt for market in page if (dt := _market_settlement(market))]
        page_max = max(settlements) if settlements else None
        if previous_page_max and page_max and page_max > previous_page_max + timedelta(seconds=1):
            ordering_ok = False
        if page_max:
            previous_page_max = page_max
        relevant = [
            market
            for market in page
            if (settled := _market_settlement(market)) is None or settled >= start_utc
        ]
        retained += len(relevant)
        if relevant:
            yield relevant
        if page_number == 1 or page_number % 10 == 0:
            _progress(
                verbose,
                f"Historical markets: page {page_number:,}, {retained:,} potentially relevant rows streamed",
            )
        if early_stop and ordering_ok and settlements and max(settlements) < start_utc:
            _progress(verbose, f"Historical scan reached settlements before {start_utc.isoformat()}; stopping")
            break
    _progress(verbose, f"Historical market stream complete: {retained:,} relevant rows")


def _iter_live_market_pages(
    client: KalshiPublicClient,
    start_utc: datetime,
    *,
    refresh_since: datetime | None,
    verbose: bool,
) -> Iterator[list[dict[str, Any]]]:
    """Yield live candidate pages without collecting the live universe."""

    if refresh_since is None:
        queries = [
            (
                "finalized",
                {
                    "status": "settled",
                    "min_settled_ts": int(start_utc.timestamp()),
                    "limit": 1000,
                },
            ),
            (
                "closed/non-finalized",
                {
                    "status": "closed",
                    "min_close_ts": int(start_utc.timestamp()),
                    "limit": 1000,
                },
            ),
        ]
    else:
        queries = [
            (
                "recently updated",
                {"min_updated_ts": int(refresh_since.timestamp()), "limit": 1000},
            ),
            (
                "recently finalized",
                {
                    "status": "settled",
                    "min_settled_ts": int(refresh_since.timestamp()),
                    "limit": 1000,
                },
            ),
        ]
    for label, params in queries:
        count = 0
        for page_number, page in enumerate(
            client.pages("markets", "markets", params), start=1
        ):
            count += len(page)
            yield page
            if page_number == 1 or page_number % 10 == 0:
                _progress(verbose, f"Live {label}: page {page_number:,}, {count:,} rows streamed")
        _progress(verbose, f"Live {label} stream complete: {count:,} rows")


def _compact_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated-object overhead in the finished in-memory result."""

    for column in ("result", "occurrence_source", "status"):
        if column in frame:
            frame[column] = frame[column].astype("category")
    return frame


def pull_sports_markets(
    start_date: str | date | datetime,
    end_date: str | date | datetime | None = None,
    *,
    client: KalshiPublicClient | None = None,
    archive_early_stop: bool = True,
    settled_since: str | datetime | None = None,
    refresh_since: str | datetime | None = None,
    verbose: bool = True,
    include_combo_legs: bool = False,
    output_csv: str | Path | None = None,
    return_dataframe: bool = True,
    csv_flush_rows: int = 50_000,
    qualification_log_every_pages: int = 10,
) -> pd.DataFrame | None:
    """Pull resolved Sports-related markets and assign each to a New York day.

    Markets must have positive API-reported volume. Single-leg markets qualify
    when their series category is Sports. A combo qualifies only when every
    selected leg belongs to a Sports series.
    Combos without a market occurrence time use the maximum occurrence time of
    their selected legs. Markets lacking a final ``yes``/``no`` result or a
    resolvable occurrence time are omitted.

    ``refresh_since`` enables fast incremental Monday refreshes. It requests
    markets whose metadata/result changed or which finalized after that time,
    while the final inclusion window still uses occurrence time. Record the
    previous run's start time and pass it on the next run; markets that were
    unresolved then are picked up when a Yes/No result appears.
    This function returns only the newly found resolved markets in that mode;
    combine them with prior results using :func:`merge_market_refresh`.

    ``settled_since`` is retained as a backward-compatible alias for
    ``refresh_since``.

    Market pages are filtered and combo occurrences are resolved as they arrive;
    the raw live/archive universes are never retained. Nested combo-leg payloads
    are omitted by default because they can dominate memory. For very large
    results, set ``output_csv`` and ``return_dataframe=False`` to stream compact
    rows to disk with bounded memory. The output CSV is overwritten.
    ``qualification_log_every_pages`` controls how often per-page Sports
    qualification counts are printed; set it to 1 for every API page.
    """

    started = time_module.monotonic()
    client = client or KalshiPublicClient()
    start_utc, end_exclusive_utc, _, _ = _ny_day_bounds(start_date, end_date)
    if settled_since is not None and refresh_since is not None:
        raise ValueError("Pass only one of refresh_since or settled_since")
    refresh_value = refresh_since if refresh_since is not None else settled_since
    refresh_utc = _parse_datetime(refresh_value) if refresh_value is not None else None
    scan_start_utc = max(start_utc, refresh_utc) if refresh_utc is not None else start_utc
    _progress(verbose, "Fetching Kalshi Sports series", started)
    sports_series = fetch_sports_series(client)
    _progress(verbose, f"Sports series discovered: {len(sports_series):,}", started)
    matcher = _SportsSeriesMatcher.build(sports_series.get("ticker", []))

    if not return_dataframe and output_csv is None:
        raise ValueError("output_csv is required when return_dataframe=False")
    if csv_flush_rows <= 0:
        raise ValueError("csv_flush_rows must be positive")
    if qualification_log_every_pages <= 0:
        raise ValueError("qualification_log_every_pages must be positive")
    output_path = Path(output_csv) if output_csv is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=MARKET_COLUMNS).to_csv(output_path, index=False)

    _progress(verbose, f"Streaming markets from {scan_start_utc.isoformat()} and filtering each page", started)
    rows: list[dict[str, Any]] = []
    csv_buffer: list[dict[str, Any]] = []
    csv_written = output_path is not None
    leg_occurrence_cache: dict[str, str | None] = {}
    seen = set() if refresh_utc is not None else None
    qualifying_count = 0
    combo_count = 0
    pages_processed = 0

    def flush_csv() -> None:
        nonlocal csv_written
        if output_path is None or not csv_buffer:
            return
        pd.DataFrame.from_records(csv_buffer, columns=MARKET_COLUMNS).to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )
        csv_buffer.clear()
        csv_written = True

    page_streams = (
        _iter_historical_market_pages_since(
            client,
            scan_start_utc,
            early_stop=archive_early_stop,
            verbose=verbose,
        ),
        _iter_live_market_pages(
            client,
            start_utc,
            refresh_since=refresh_utc,
            verbose=verbose,
        ),
    )
    for page_stream in page_streams:
        for page in page_stream:
            pages_processed += 1
            candidates: list[dict[str, Any]] = []
            sports_by_ticker: dict[str, list[str]] = {}
            page_sports_qualified = 0
            page_single_qualified = 0
            page_combo_qualified = 0
            for market in page:
                ticker = market.get("ticker")
                if not ticker or str(market.get("result", "")).lower() not in FINAL_RESULTS:
                    continue
                if Decimal(str(market.get("volume_fp") or "0")) <= 0:
                    continue
                market_is_combo = bool(
                    market.get("mve_collection_ticker") or market.get("mve_selected_legs")
                )
                sports_matches = _sports_series_for_market(market, matcher)
                if not sports_matches:
                    continue
                if seen is not None:
                    if ticker in seen:
                        continue
                    seen.add(ticker)
                candidates.append(market)
                sports_by_ticker[ticker] = sports_matches
                page_sports_qualified += 1
                page_combo_qualified += int(market_is_combo)
                page_single_qualified += int(not market_is_combo)
            if pages_processed == 1 or pages_processed % qualification_log_every_pages == 0:
                _progress(
                    verbose,
                    (
                        f"Sports qualification page {pages_processed:,}: received={len(page):,}, "
                        f"qualified Sports={page_sports_qualified:,} "
                        f"(single={page_single_qualified:,}, combo={page_combo_qualified:,})"
                    ),
                    started,
                )
            if not candidates:
                continue
            _resolve_occurrences(
                candidates,
                client,
                verbose=False,
                occurrence_cache=leg_occurrence_cache,
            )
            for market in candidates:
                occurrence = _optional_datetime(market.get("occurrence_datetime"))
                if occurrence is None or not (start_utc <= occurrence < end_exclusive_utc):
                    continue
                is_combo = bool(market.get("mve_collection_ticker") or market.get("mve_selected_legs"))
                row = {
                    "occurrence_date_ny": occurrence.astimezone(NY_TZ).date(),
                    "ticker": market.get("ticker"),
                    "event_ticker": market.get("event_ticker"),
                    "is_combo": is_combo,
                    "sports_series_tickers": ",".join(sports_by_ticker[market["ticker"]]),
                    "title": market.get("title"),
                    "result": str(market.get("result")).lower(),
                    "market_volume_contracts": float(Decimal(str(market.get("volume_fp") or "0"))),
                    "occurrence_datetime": occurrence,
                    "occurrence_source": market.get("occurrence_source"),
                    "open_time": _optional_datetime(market.get("open_time")),
                    "settlement_ts": _optional_datetime(market.get("settlement_ts")),
                    "status": market.get("status"),
                    "mve_collection_ticker": market.get("mve_collection_ticker"),
                    "mve_selected_legs": market.get("mve_selected_legs") or [] if include_combo_legs else None,
                }
                qualifying_count += 1
                combo_count += int(is_combo)
                if return_dataframe:
                    rows.append(row)
                if output_path is not None:
                    csv_buffer.append(row)
                    if len(csv_buffer) >= csv_flush_rows:
                        flush_csv()
                        _progress(verbose, f"Streamed {qualifying_count:,} qualifying markets to {output_path}", started)
    flush_csv()
    if not return_dataframe:
        _progress(
            verbose,
            f"Pull complete: {qualifying_count:,} markets ({combo_count:,} combos) streamed to {output_path}",
            started,
        )
        return None
    if not rows:
        _progress(verbose, "Pull complete: no qualifying markets", started)
        return pd.DataFrame(columns=MARKET_COLUMNS)
    result = _compact_market_frame(
        pd.DataFrame.from_records(rows, columns=MARKET_COLUMNS)
        .sort_values(["occurrence_date_ny", "ticker"], kind="stable")
        .reset_index(drop=True)
    )
    _progress(
        verbose,
        f"Pull complete: {len(result):,} markets ({int(result['is_combo'].sum()):,} combos)",
        started,
    )
    return result


def merge_market_refresh(existing: pd.DataFrame, refresh: pd.DataFrame) -> pd.DataFrame:
    """Merge an incremental market refresh, preferring the newer API row."""

    if existing.empty:
        return refresh.copy().reset_index(drop=True)
    if refresh.empty:
        return existing.copy().reset_index(drop=True)
    combined = pd.concat([existing, refresh], ignore_index=True)
    return (
        combined.drop_duplicates("ticker", keep="last")
        .sort_values(["occurrence_date_ny", "ticker"], kind="stable")
        .reset_index(drop=True)
    )


def load_saved_markets(
    csv_path: str | Path,
    start_date: str | date | datetime,
    end_date: str | date | datetime | None = None,
    *,
    chunksize: int = 100_000,
    columns: Sequence[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a New York occurrence-date range from a saved market CSV.

    When ``end_date`` is omitted, only ``start_date`` is returned. The source
    CSV is read in chunks, so memory usage depends primarily on the matching
    output rather than the size of the complete saved file.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    start_day = _as_date(start_date)
    end_day = _as_date(end_date) if end_date is not None else start_day
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved market CSV does not exist: {path}")

    requested_columns = list(dict.fromkeys(columns)) if columns is not None else None
    read_columns = requested_columns
    drop_date_after_filter = False
    if requested_columns is not None and "occurrence_date_ny" not in requested_columns:
        read_columns = ["occurrence_date_ny", *requested_columns]
        drop_date_after_filter = True

    started = time_module.monotonic()
    _progress(
        verbose,
        f"Reading saved markets for {start_day.isoformat()} through {end_day.isoformat()} from {path}",
        started,
    )
    matches: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_matched = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunksize, usecols=read_columns), start=1
    ):
        rows_scanned += len(chunk)
        occurrence_dates = pd.to_datetime(
            chunk["occurrence_date_ny"], errors="coerce"
        ).dt.date
        selected = chunk.loc[
            occurrence_dates.ge(start_day) & occurrence_dates.le(end_day)
        ].copy()
        if not selected.empty:
            selected["occurrence_date_ny"] = occurrence_dates.loc[selected.index]
            matches.append(selected)
            rows_matched += len(selected)
        if chunk_number == 1 or chunk_number % 10 == 0:
            _progress(
                verbose,
                f"Saved-market scan: {rows_scanned:,} rows read, {rows_matched:,} matched",
                started,
            )

    if matches:
        result = pd.concat(matches, ignore_index=True)
        if drop_date_after_filter:
            result = result.drop(columns="occurrence_date_ny")
        for column in ("occurrence_datetime", "open_time", "settlement_ts"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
        if "is_combo" in result.columns and result["is_combo"].dtype == object:
            result["is_combo"] = (
                result["is_combo"].astype(str).str.lower().map({"true": True, "false": False})
            )
        sort_columns = [
            column for column in ("occurrence_date_ny", "ticker") if column in result.columns
        ]
        if sort_columns:
            result = result.sort_values(sort_columns, kind="stable")
        result = result.reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=requested_columns or pd.read_csv(path, nrows=0).columns)

    _progress(
        verbose,
        f"Saved-market load complete: {len(result):,} rows returned",
        started,
    )
    return result


def _decimal_field(record: Mapping[str, Any], fp_name: str, legacy_name: str | None = None) -> Decimal:
    value = record.get(fp_name)
    if value not in (None, ""):
        return Decimal(str(value))
    if legacy_name and record.get(legacy_name) not in (None, ""):
        return Decimal(str(record[legacy_name]))
    return Decimal("0")


def _price_dollars(record: Mapping[str, Any], side: str) -> Decimal:
    dollars = record.get(f"{side}_price_dollars")
    if dollars not in (None, ""):
        return Decimal(str(dollars))
    cents = record.get(f"{side}_price")
    return Decimal(str(cents or 0)) / Decimal("100")


def summarize_trades(
    trades: Iterable[Mapping[str, Any]],
    result: str,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Summarize public trade records for one binary market."""

    normalized_result = str(result).lower()
    if normalized_result not in FINAL_RESULTS:
        raise ValueError("result must be 'yes' or 'no'")
    total_contract = Decimal("0")
    yes_contract = Decimal("0")
    yes_dollar_volume = Decimal("0")
    no_contract = Decimal("0")
    no_dollar_volume = Decimal("0")
    yes_trade = 0
    no_trade = 0
    seen: set[str] = set()

    for trade in trades:
        trade_id = str(trade.get("trade_id") or "")
        if trade_id and trade_id in seen:
            continue
        if trade_id:
            seen.add(trade_id)
        count = _decimal_field(trade, "count_fp", "count")
        side = str(trade.get("taker_outcome_side") or trade.get("taker_side") or "").lower()
        total_contract += count
        if side == "yes":
            yes_contract += count
            yes_dollar_volume += count * _price_dollars(trade, "yes")
            yes_trade += 1
        elif side == "no":
            no_contract += count
            no_dollar_volume += count * _price_dollars(trade, "no")
            no_trade += 1

    yes_taker_pnl = (
        yes_dollar_volume
        if normalized_result == "no"
        else yes_dollar_volume - yes_contract
    )
    return {
        "ticker": ticker,
        "result": normalized_result,
        "total_contract": float(total_contract),
        "yes_contract": float(yes_contract),
        "yes_dollar_volume": float(yes_dollar_volume),
        "yes_trade": yes_trade,
        "no_contract": float(no_contract),
        "no_dollar_volume": float(no_dollar_volume),
        "no_trade": no_trade,
        "yes_taker_pnl": float(yes_taker_pnl),
    }


TRADE_SUMMARY_COLUMNS = [
    "ticker",
    "result",
    "total_contract",
    "yes_contract",
    "yes_dollar_volume",
    "yes_trade",
    "no_contract",
    "no_dollar_volume",
    "no_trade",
    "yes_taker_pnl",
]


def _records_from_markets(markets: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(markets, pd.DataFrame):
        return markets.to_dict("records")
    return [dict(item) for item in markets]


def _fetch_market_trade_summary(
    client: KalshiPublicClient,
    market: Mapping[str, Any],
    trade_cutoff: datetime,
) -> dict[str, Any]:
    ticker = str(market["ticker"])
    result = str(market["result"]).lower()
    if float(market.get("market_volume_contracts", market.get("volume_fp", 1)) or 0) == 0:
        return summarize_trades([], result, ticker)

    open_time = _optional_datetime(market.get("open_time"))
    settlement = _optional_datetime(market.get("settlement_ts"))
    use_historical = open_time is None or open_time < trade_cutoff
    use_live = settlement is None or settlement >= trade_cutoff
    paths: list[str] = []
    if use_historical:
        paths.append("historical/trades")
    if use_live:
        paths.append("markets/trades")
    if not paths:
        paths.append("historical/trades" if settlement and settlement < trade_cutoff else "markets/trades")

    def iter_trades() -> Iterator[dict[str, Any]]:
        for path in paths:
            for page in client.pages(path, "trades", {"ticker": ticker, "limit": 1000}):
                yield from page

    return summarize_trades(iter_trades(), result, ticker)


def pull_trade_summaries(
    markets: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    client: KalshiPublicClient | None = None,
    max_workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch and summarize every trade for each supplied resolved market.

    The market input must contain ``ticker`` and ``result``. Passing the output
    of :func:`pull_sports_markets` is recommended because its open/settlement
    timestamps let this function avoid unnecessary live or historical calls.
    """

    started = time_module.monotonic()
    client = client or KalshiPublicClient()
    records = _records_from_markets(markets)
    if not records:
        return pd.DataFrame(columns=TRADE_SUMMARY_COLUMNS)
    for record in records:
        if not record.get("ticker") or str(record.get("result", "")).lower() not in FINAL_RESULTS:
            raise ValueError("Every market must contain ticker and a final yes/no result")

    nonzero = sum(
        float(record.get("market_volume_contracts", record.get("volume_fp", 1)) or 0) > 0
        for record in records
    )
    _progress(
        verbose,
        f"Starting trade pull for {len(records):,} markets; {nonzero:,} have nonzero API volume; {max(1, max_workers)} workers",
        started,
    )
    cutoff = client.historical_cutoff().get("trades_created_ts")
    if cutoff is None:
        raise KalshiAPIError("Historical cutoff response lacked trades_created_ts")
    summaries: list[dict[str, Any]] = []
    report_every = max(1, len(records) // 100)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(_fetch_market_trade_summary, client, record, cutoff): record["ticker"]
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            summaries.append(future.result())
            if completed == 1 or completed % report_every == 0 or completed == len(records):
                _progress(
                    verbose,
                    f"Trade summaries complete: {completed:,}/{len(records):,} ({completed / len(records):.0%})",
                    started,
                )
    result = (
        pd.DataFrame(summaries, columns=TRADE_SUMMARY_COLUMNS)
        .sort_values("ticker", kind="stable")
        .reset_index(drop=True)
    )
    _progress(verbose, f"Trade pull complete: {len(result):,} market summaries", started)
    return result


def daily_market_summary(markets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate market counts and API-reported contract volume by NY day."""

    if markets.empty:
        return pd.DataFrame(
            columns=["occurrence_date_ny", "market_count", "single_market_count", "combo_market_count", "market_volume_contracts"]
        )
    return (
        markets.assign(single_market=~markets["is_combo"])
        .groupby("occurrence_date_ny", as_index=False)
        .agg(
            market_count=("ticker", "size"),
            single_market_count=("single_market", "sum"),
            combo_market_count=("is_combo", "sum"),
            market_volume_contracts=("market_volume_contracts", "sum"),
        )
        .sort_values("occurrence_date_ny")
        .reset_index(drop=True)
    )
