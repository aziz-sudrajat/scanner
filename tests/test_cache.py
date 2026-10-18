"""Tests untuk scanner/cache.py — disk cache & rate limiter."""
import os
import time
import unittest

from scanner.cache import RateLimiter, cached, clear_cache


class TestRateLimiter(unittest.TestCase):
    def test_min_interval(self):
        rl = RateLimiter(0.2)
        t0 = time.monotonic()
        rl.acquire()
        rl.acquire()
        self.assertGreaterEqual(time.monotonic() - t0, 0.18)

    def test_no_wait_first(self):
        rl = RateLimiter(10.0)
        # acquire pertama tidak perlu tunggu (last=0, now-last > interval)
        waited = rl.acquire()
        self.assertEqual(waited, 0.0)


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        clear_cache()
        os.environ.pop("SCAN_NO_CACHE", None)

    def tearDown(self):
        clear_cache()
        os.environ.pop("SCAN_NO_CACHE", None)

    def test_hit(self):
        calls = []

        @cached("u-t", {"a": 1}, ttl=60)
        def f():
            calls.append(1)
            return {"x": 1}

        self.assertEqual(f(), {"x": 1})
        self.assertEqual(f(), {"x": 1})
        self.assertEqual(len(calls), 1)

    def test_bypass_env(self):
        calls = []

        @cached("u-t2", {}, ttl=60)
        def f():
            calls.append(1)
            return 1

        os.environ["SCAN_NO_CACHE"] = "1"
        f()
        f()
        self.assertEqual(len(calls), 2)

    def test_ttl_zero_always_miss(self):
        calls = []

        @cached("u-t3", {}, ttl=0)
        def f():
            calls.append(1)
            return 1

        f()
        f()
        self.assertEqual(len(calls), 2)

    def test_clear(self):
        @cached("u-t4", {}, ttl=60)
        def f():
            return 1

        f()
        n = clear_cache()
        self.assertGreaterEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
