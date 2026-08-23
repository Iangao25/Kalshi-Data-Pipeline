# Kalshi Sports Market Utilities

This project contains notebook-friendly utilities for pulling resolved Kalshi
Sports markets and public trades. It uses only unauthenticated public endpoints.

## Files

- `kalshi_utils.py`: reusable API, market-discovery, occurrence, and trade-summary functions
- `kalshi_sports_analysis.ipynb`: weekly Monday workflow starting in May 2026
- `test_kalshi_utils.py`: focused calculation and metadata tests

## Important choices

- A market belongs to the New York calendar day containing Kalshi's
  `occurrence_datetime`, not its close or expiration date.
- Combo markets often omit that field. Their occurrence is the latest Kalshi
  `occurrence_datetime` among their selected legs.
- A market is sports-related when its series is categorized as Sports. A combo
  is included only when every selected leg maps to a Sports series. Mixed,
  non-Sports, or unclassifiable combos are excluded.
- Markets without a `yes` or `no` result are omitted. A populated binary result
  is accepted even when the market is determined, disputed, or amended rather
  than finalized; a later refresh can overwrite it if Kalshi changes the result.
- Markets with zero or missing `volume_fp` are discarded before Sports
  classification and combo occurrence resolution.
- The current historical-market endpoint has no settlement timestamp filter and
  silently ignores unknown filters. The utility pages the archive newest-first
  and, by default, stops after settlement dates pass the requested start. Set
  `archive_early_stop=False` for an exhaustive archive scan if Kalshi changes
  that observed ordering.
- Trade requests use page size 1,000, skip zero-volume markets, run concurrently,
  and automatically query the live endpoint, historical endpoint, or both when
  a market spans the current trade cutoff.
- Requests are paced at 10/second by default, with exponential backoff on 429
  and transient server responses. Lower `requests_per_second` if the public
  endpoint throttles your network.
- The two top-level pull functions print timestamped page counts, combo-leg
  resolution progress, percentages, and elapsed time by default. Pass
  `verbose=False` to silence them.
- Market discovery reports per-round Sports qualification counts every 10 pages
  by default. Use `qualification_log_every_pages=1` for every API page.

## Fast weekly refreshes

The initial May 2026 backfill must enumerate the relevant archive/live market
pages because Kalshi offers no occurrence-time query filter. For later Monday
runs, pass the previous run's start timestamp as `refresh_since`. This scans
markets updated or finalized since that run while still filtering the final
result by the requested occurrence-date window. Any previously unresolved
market is picked up when a Yes/No result appears, and changed determinations
replace the earlier row through `merge_market_refresh`.

## Low-memory market pulls

Market pages are filtered immediately as they arrive. The utility no longer
retains the raw live universe, archive universe, merged universe, and final
DataFrame simultaneously. Nested `mve_selected_legs` payloads are omitted from
the result by default; pass `include_combo_legs=True` only when needed.

If the qualifying output itself is too large for RAM, stream it directly to CSV:

```python
pull_sports_markets(
    "2026-05-01",
    client=client,
    output_csv="sports_markets.csv",
    return_dataframe=False,
)
```

This holds only the current API page, compact combo-occurrence cache, and a
50,000-row CSV buffer in memory. Adjust the buffer with `csv_flush_rows`.

Load a single New York occurrence day from the saved CSV without reading the
complete file into memory:

```python
markets = load_saved_markets("sports_markets.csv", "2026-08-15")
```

Or load an inclusive date range:

```python
markets = load_saved_markets(
    "sports_markets.csv",
    "2026-08-15",
    "2026-08-21",
)
```

## P&L convention

Only trades whose taker outcome is Yes enter `yes_taker_pnl`:

- result No: `yes_dollar_volume`
- result Yes: `yes_dollar_volume - yes_contract`

No-taker activity is reported separately but does not enter that P&L.
