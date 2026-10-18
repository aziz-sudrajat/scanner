"""Tests untuk scanner/config.py — user config override."""
import json
import os
import tempfile
import unittest

from scanner import config as cfg


class TestUserConfig(unittest.TestCase):
    def test_missing_file(self):
        self.assertEqual(cfg.load_user_config("/tidak/ada.json"), {})

    def test_corrupt_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            fh.write("{rusak")
            p = fh.name
        try:
            self.assertEqual(cfg.load_user_config(p), {})
        finally:
            os.unlink(p)

    def test_non_dict(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump([1, 2], fh)
            p = fh.name
        try:
            self.assertEqual(cfg.load_user_config(p), {})
        finally:
            os.unlink(p)

    def test_apply_only_known_uppercase(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump({"DEFAULT_LIMIT": 99, "lowercase": 1,
                       "TIDAK_ADA": 2}, fh)
            p = fh.name
        orig = cfg.DEFAULT_LIMIT
        try:
            applied = cfg.apply_user_config(p)
            self.assertEqual(applied, {"DEFAULT_LIMIT": 99})
            self.assertEqual(cfg.DEFAULT_LIMIT, 99)
        finally:
            cfg.DEFAULT_LIMIT = orig
            os.unlink(p)

    def test_defaults_intact(self):
        # tanpa file, konstanta penting ada
        for key in ("STRONG_BUY_THRESHOLD", "BUY_THRESHOLD", "SELL_THRESHOLD",
                    "STRONG_SELL_THRESHOLD", "REQUEST_TIMEOUT",
                    "REQUEST_RETRIES", "USER_AGENT"):
            self.assertIn(key, dir(cfg))


if __name__ == "__main__":
    unittest.main()
