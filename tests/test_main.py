"""Tests untuk main.py — logika slicing & analyze (mock network)."""
import unittest
from unittest.mock import patch

import pandas as pd

import main as m


def fake_coins(n):
    return [{"id": f"coin{i}", "name": f"Coin{i}", "symbol": f"c{i}"}
            for i in range(n)]


class TestRankedSlicing(unittest.TestCase):
    def setUp(self):
        m._snap_cache.clear()

    def tearDown(self):
        m._snap_cache.clear()

    def test_technical_respects_limit(self):
        # _run_technical harus fetch tepat `limit` koin setelah exclude_top
        m._snap_cache[(30, 5)] = fake_coins(100)
        with patch("main.fetch_coins_ids") as mids, \
             patch("main._fetch_one") as mfo, \
             patch("main._analyze") as man:
            mids.side_effect = lambda snap: [c["id"] for c in snap]
            mfo.return_value = {"coin_id": "x", "rows": []}
            man.return_value = []
            m._run_technical(30, 5, "daily", 200, 1, True)
            ids_arg = mids.call_args[0][0]
            self.assertEqual(len(ids_arg), 30)
            self.assertEqual(ids_arg[0]["id"], "coin5")

    def test_snapshot_slice(self):
        m._snap_cache[(30, 5)] = fake_coins(100)
        with patch("main.score_snapshot") as mss:
            mss.side_effect = lambda c, n, s: c["id"]
            out = m._run_snapshot(30, 5)
            self.assertEqual(len(out), 30)
            self.assertEqual(out[0], "coin5")

    def test_ranked_fetch_size(self):
        # fetch secukupnya, bukan 250 fixed
        with patch("main.fetch_snapshot") as mfs:
            mfs.return_value = fake_coins(60)
            m._ranked(30, 5)
            want = mfs.call_args[1]["limit"]
            self.assertLessEqual(want, 100, f"fetch {want} kebanyakan")


class TestAnalyzeLookup(unittest.TestCase):
    def test_dict_lookup(self):
        ts = 1700000000000
        rows = [{"time": ts + i * 86400000, "close": 100 + i,
                 "total_volume": 1000} for i in range(70)]
        by_id = {"bitcoin": {"id": "bitcoin", "name": "Bitcoin",
                             "symbol": "btc"}}
        out = m._analyze([{"coin_id": "bitcoin", "rows": rows}], by_id, True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "Bitcoin")
        self.assertEqual(out[0].symbol, "BTC")

    def test_short_df_skipped(self):
        rows = [{"time": 1 + i, "close": 100, "total_volume": 1}
                for i in range(10)]
        out = m._analyze([{"coin_id": "x", "rows": rows}], {}, True)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
