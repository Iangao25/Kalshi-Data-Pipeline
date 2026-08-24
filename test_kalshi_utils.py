import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from kalshi_utils import (
    _SportsSeriesMatcher,
    _ny_day_bounds,
    _parse_datetime,
    _sports_series_for_market,
    fetch_markets_by_tickers,
    fetch_live_result_candidates,
    load_saved_markets,
    merge_market_refresh,
    pull_sports_markets,
    summarize_trades,
)


class CalculationTests(unittest.TestCase):
    def setUp(self):
        self.trades = [
            {
                "trade_id": "a",
                "count_fp": "10.50",
                "yes_price_dollars": "0.6000",
                "no_price_dollars": "0.4000",
                "taker_side": "yes",
            },
            {
                "trade_id": "b",
                "count_fp": "4.00",
                "yes_price_dollars": "0.3000",
                "no_price_dollars": "0.7000",
                "taker_outcome_side": "no",
            },
        ]

    def test_no_result_pnl(self):
        summary = summarize_trades(self.trades, "no", "MKT")
        self.assertEqual(summary["total_contract"], 14.5)
        self.assertEqual(summary["yes_contract"], 10.5)
        self.assertAlmostEqual(summary["yes_dollar_volume"], 6.3)
        self.assertEqual(summary["yes_trade"], 1)
        self.assertEqual(summary["no_contract"], 4.0)
        self.assertAlmostEqual(summary["no_dollar_volume"], 2.8)
        self.assertEqual(summary["no_trade"], 1)
        self.assertAlmostEqual(summary["yes_taker_pnl"], 6.3)

    def test_yes_result_pnl(self):
        summary = summarize_trades(self.trades, "yes")
        self.assertAlmostEqual(summary["yes_taker_pnl"], -4.2)

    def test_duplicate_trade_id_is_ignored(self):
        summary = summarize_trades(self.trades + [self.trades[0]], "no")
        self.assertEqual(summary["total_contract"], 14.5)


class MetadataTests(unittest.TestCase):
    def test_combo_leg_lookup_uses_matching_source_first_and_falls_back(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def get(self, path, params=None):
                self.calls.append(path)
                requested = set(params["tickers"].split(","))
                available = "HISTORICAL-LEG" if path == "historical/markets" else "LIVE-LEG"
                return {"markets": [{"ticker": available}] if available in requested else []}

        historical_first = FakeClient()
        found = fetch_markets_by_tickers(
            historical_first,
            ["HISTORICAL-LEG", "LIVE-LEG"],
            prefer_historical=True,
        )
        self.assertEqual(historical_first.calls, ["historical/markets", "markets"])
        self.assertEqual(set(found), {"HISTORICAL-LEG", "LIVE-LEG"})

        live_first = FakeClient()
        found = fetch_markets_by_tickers(
            live_first,
            ["HISTORICAL-LEG", "LIVE-LEG"],
        )
        self.assertEqual(live_first.calls, ["markets", "historical/markets"])
        self.assertEqual(set(found), {"HISTORICAL-LEG", "LIVE-LEG"})

    def test_longest_lexicographic_series_match(self):
        matcher = _SportsSeriesMatcher.build(["KXMLB", "KXMLBGAME", "KXNFLGAME"])
        self.assertEqual(matcher.match("KXMLBGAME-26AUG22NYYBOS"), "KXMLBGAME")
        self.assertIsNone(matcher.match("KXBTC15M-26AUG22"))

    def test_combo_requires_every_leg_to_be_sports(self):
        matcher = _SportsSeriesMatcher.build(["KXNFLGAME", "KXMLBGAME"])
        all_sports = {
            "mve_collection_ticker": "KXMVESPORTS",
            "mve_selected_legs": [
                {"event_ticker": "KXNFLGAME-26AUG22WASDET"},
                {"event_ticker": "KXMLBGAME-26AUG22NYYBOS"},
            ],
        }
        mixed = {
            "mve_collection_ticker": "KXMVECROSSCATEGORY",
            "mve_selected_legs": [
                {"event_ticker": "KXNFLGAME-26AUG22WASDET"},
                {"event_ticker": "KXBTC15M-26AUG221445"},
            ],
        }
        missing_legs = {"mve_collection_ticker": "KXMVESPORTS"}
        self.assertEqual(
            _sports_series_for_market(all_sports, matcher),
            ["KXMLBGAME", "KXNFLGAME"],
        )
        self.assertEqual(_sports_series_for_market(mixed, matcher), [])
        self.assertEqual(_sports_series_for_market(missing_legs, matcher), [])

    def test_new_york_day_bounds_include_dst(self):
        start, end, start_day, end_day = _ny_day_bounds("2026-05-01", "2026-05-01")
        self.assertEqual(start_day, date(2026, 5, 1))
        self.assertEqual(end_day, date(2026, 5, 1))
        self.assertEqual((end - start).total_seconds(), 86400)
        self.assertEqual(start.hour, 4)  # EDT midnight is 04:00 UTC.

    def test_variable_width_fractional_seconds(self):
        parsed = _parse_datetime("2026-06-22T23:46:04.09134Z")
        self.assertEqual(parsed.microsecond, 91340)

    def test_incremental_refresh_prefers_new_row(self):
        old = pd.DataFrame([{"ticker": "A", "occurrence_date_ny": date(2026, 5, 1), "result": "no"}])
        new = pd.DataFrame([{"ticker": "A", "occurrence_date_ny": date(2026, 5, 1), "result": "yes"}])
        merged = merge_market_refresh(old, new)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["result"], "yes")

    def test_load_saved_markets_single_day_and_range(self):
        source = pd.DataFrame(
            [
                {"occurrence_date_ny": "2026-08-01", "ticker": "A", "result": "yes"},
                {"occurrence_date_ny": "2026-08-02", "ticker": "B", "result": "no"},
                {"occurrence_date_ny": "2026-08-03", "ticker": "C", "result": "yes"},
            ]
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "markets.csv"
            source.to_csv(path, index=False)
            one_day = load_saved_markets(path, "2026-08-02", chunksize=1, verbose=False)
            date_range = load_saved_markets(
                path, "2026-08-01", "2026-08-02", chunksize=2, verbose=False
            )
        self.assertEqual(one_day["ticker"].tolist(), ["B"])
        self.assertEqual(date_range["ticker"].tolist(), ["A", "B"])

    def test_initial_live_scan_includes_finalized_and_nonfinalized(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def pages(self, path, item_key, params):
                self.calls.append(dict(params))
                yield []

        client = FakeClient()
        start, _, _, _ = _ny_day_bounds("2026-05-01", "2026-05-01")
        fetch_live_result_candidates(client, start)
        self.assertEqual({call.get("status") for call in client.calls}, {"settled", "closed"})

    def test_incremental_live_scan_uses_updates_and_settlements(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def pages(self, path, item_key, params):
                self.calls.append(dict(params))
                yield []

        client = FakeClient()
        start, _, _, _ = _ny_day_bounds("2026-05-01", "2026-05-01")
        fetch_live_result_candidates(client, start, refresh_since=start)
        self.assertTrue(any("min_updated_ts" in call for call in client.calls))
        self.assertTrue(any("min_settled_ts" in call for call in client.calls))

    def test_top_level_pull_filters_each_page(self):
        class FakeClient:
            def get(self, path, params=None):
                if path == "series":
                    return {
                        "series": [
                            {"ticker": "KXSPORT", "title": "Sport", "category": "Sports", "tags": []}
                        ]
                    }
                raise AssertionError(f"Unexpected GET {path}")

            def pages(self, path, item_key, params):
                if path == "historical/markets":
                    yield []
                elif params.get("status") == "settled":
                    yield [
                        {
                            "ticker": "KXSPORT-26MAY01-A",
                            "event_ticker": "KXSPORT-26MAY01",
                            "result": "yes",
                            "occurrence_datetime": "2026-05-01T18:00:00Z",
                            "volume_fp": "2.00",
                            "status": "finalized",
                        },
                        {
                            "ticker": "KXOTHER-26MAY01-A",
                            "event_ticker": "KXOTHER-26MAY01",
                            "result": "yes",
                            "occurrence_datetime": "2026-05-01T18:00:00Z",
                            "volume_fp": "3.00",
                        },
                        {
                            "ticker": "KXSPORT-26MAY01-ZERO",
                            "event_ticker": "KXSPORT-26MAY01",
                            "result": "yes",
                            "occurrence_datetime": "2026-05-01T18:00:00Z",
                            "volume_fp": "0.00",
                        },
                    ]
                else:
                    yield []

        frame = pull_sports_markets(
            "2026-05-01",
            "2026-05-01",
            client=FakeClient(),
            verbose=False,
        )
        self.assertEqual(frame["ticker"].tolist(), ["KXSPORT-26MAY01-A"])
        self.assertIsNone(frame.iloc[0]["mve_selected_legs"])
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "markets.csv"
            returned = pull_sports_markets(
                "2026-05-01",
                "2026-05-01",
                client=FakeClient(),
                verbose=False,
                output_csv=output,
                return_dataframe=False,
                csv_flush_rows=1,
            )
            self.assertIsNone(returned)
            self.assertEqual(pd.read_csv(output)["ticker"].tolist(), ["KXSPORT-26MAY01-A"])


if __name__ == "__main__":
    unittest.main()
