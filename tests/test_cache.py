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

    def _age_cache_file(self, seconds_old: float):
        from scanner.cache import _cache_dir
        files = list(_cache_dir().glob("*.json"))
        self.assertEqual(len(files), 1)
        old = time.time() - seconds_old
        os.utime(files[0], (old, old))

    def test_stale_fallback(self):
        import io
        from contextlib import redirect_stderr

        @cached("u-stale", {"a": 1}, ttl=60)
        def f(fail=False):
            if fail:
                raise RuntimeError("api down")
            return {"x": 42}

        self.assertEqual(f(), {"x": 42})
        self._age_cache_file(7200)  # basi 2 jam
        buf = io.StringIO()
        with redirect_stderr(buf):
            out = f(fail=True)
        self.assertEqual(out, {"x": 42})  # data basi > crash
        self.assertIn("basi", buf.getvalue())

    def test_stale_disabled_raises(self):
        @cached("u-nostale", {}, ttl=60, allow_stale=False)
        def f():
            return {"x": 1}

        f()
        self._age_cache_file(7200)

        @cached("u-nostale", {}, ttl=60, allow_stale=False)
        def g():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            g()

    def test_no_cache_no_stale_raises(self):
        @cached("u-fresh-fail", {}, ttl=60)
        def f():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            f()


class TestCachePeek(unittest.TestCase):
    def setUp(self):
        clear_cache()
        os.environ.pop("SCAN_NO_CACHE", None)

    def tearDown(self):
        clear_cache()
        os.environ.pop("SCAN_NO_CACHE", None)

    def test_miss(self):
        hit, val = __import__("scanner.cache", fromlist=["cache_peek"]).cache_peek(
            "nope", {}, 60)
        self.assertFalse(hit)
        self.assertIsNone(val)

    def test_hit(self):
        from scanner.cache import cache_peek

        @cached("u-peek", {"a": 1}, ttl=60)
        def f():
            return [1, 2, 3]

        f()
        hit, val = cache_peek("u-peek", {"a": 1}, 60)
        self.assertTrue(hit)
        self.assertEqual(val, [1, 2, 3])

    def test_expired_is_miss(self):
        import time as _t
        from scanner.cache import cache_peek, _cache_dir

        @cached("u-peek-exp", {}, ttl=60)
        def f():
            return 1

        f()
        files = list(_cache_dir().glob("*.json"))
        self.assertEqual(len(files), 1)
        old = _t.time() - 7200
        os.utime(files[0], (old, old))
        hit, val = cache_peek("u-peek-exp", {}, 60)
        self.assertFalse(hit)
        self.assertIsNone(val)

    def test_no_cache_env(self):
        from scanner.cache import cache_peek

        @cached("u-peek-env", {}, ttl=60)
        def f():
            return 1

        f()
        os.environ["SCAN_NO_CACHE"] = "1"
        hit, _ = cache_peek("u-peek-env", {}, 60)
        self.assertFalse(hit)


if __name__ == "__main__":
    unittest.main()
