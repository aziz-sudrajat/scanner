"""Tests untuk scanner/fetch.py — retry, validasi (mock requests)."""
import unittest
from unittest.mock import MagicMock, patch

from scanner import fetch
from scanner.fetch import CoinGeckoError
from scanner.cache import clear_cache


def resp(status=200, json_data=None, text=""):
    m = MagicMock()
    m.status_code = status
    m.text = text
    m.json.return_value = json_data
    return m


class TestGet(unittest.TestCase):
    def setUp(self):
        clear_cache()
    def test_user_agent_and_timeout(self):
        with patch("scanner.fetch.requests.get") as mg:
            mg.return_value = resp(200, [])
            fetch._get("http://x", tries=1)
            _, kw = mg.call_args
            self.assertIn("signal-scanner", kw["headers"]["User-Agent"])
            self.assertEqual(kw["timeout"], 20)

    def test_http_error_detail(self):
        with patch("scanner.fetch.requests.get") as mg:
            mg.return_value = resp(404, text="not here")
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=1)
            self.assertEqual(cm.exception.status_code, 404)
            self.assertIn("not here", str(cm.exception))

    def test_429_gives_up(self):
        import requests as _rq  # noqa
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.return_value = resp(429)
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=2)
            self.assertEqual(cm.exception.status_code, 429)

    def test_bad_json(self):
        with patch("scanner.fetch.requests.get") as mg:
            m = resp(200)
            m.json.side_effect = ValueError("no json")
            mg.return_value = m
            with self.assertRaises(CoinGeckoError):
                fetch._get("http://x", tries=1)


class TestFetchOhlcv(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _mk(self, prices):
        return {"prices": prices, "market_caps": [], "total_volumes": []}

    def test_malformed_rows_skipped(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk([[1, 100.0], "rusak", [3, "xx"],
                                        [4, 105.0]])
            out = fetch.fetch_ohlcv("c")
            self.assertEqual(len(out), 2)
            self.assertEqual(out[1]["close"], 105.0)

    def test_all_bad_raises(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk(["x", None])
            with self.assertRaises(CoinGeckoError) as cm:
                fetch.fetch_ohlcv("c")
            self.assertIn("malformed", str(cm.exception))

    def test_non_dict_raises(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = [1, 2]
            with self.assertRaises(CoinGeckoError):
                fetch.fetch_ohlcv("c")


class TestFetchCoinsIds(unittest.TestCase):
    def test_filters(self):
        snap = [{"id": "a"}, {"id": ""}, {"x": 1}, {"id": "b"}]
        self.assertEqual(fetch.fetch_coins_ids(snap), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
