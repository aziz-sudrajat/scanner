"""Configuration and tunable defaults for the Signal Scanner.

All threshold values used in the scoring model live here so you can tune
them without touching the core logic.
"""

# --- Data fetching -----------------------------------------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
VS_CURRENCY = "usd"
CANDLE_INTERVAL = "daily"      # "daily" | "hourly"  (interval enum resmi CoinGecko)
MAX_CANDLES = 200              # berapa banyak candle per koin (daily ~ maks 3650)
FETCH_WORKERS = 1              # minimal: jaga agar tidak diejek rate limit API
REQUEST_DELAY = 10.0           # detik jeda antar request (ikuti batas free tier)


# --- Coin universe -----------------------------------------------------
DEFAULT_LIMIT = 30             # number of coins to scan (ranked by market cap)
DEFAULT_EXCLUDE_TOP = 5        # skip top N (biasanya BTC/ETH) agar lebih bersaing


# --- Scoring model (faktor 0..100; total setelah dipotong +/-100) -------
# Trend
TREND_PRICE_ABOVE_EMA200_MAX = 25     # price > EMA200
TREND_EMA_MA200_CROSS_MAX = 20        # EMA20 > EMA50 > EMA200 aligned
TREND_MOMENTUM_MAX = 20               # EMA12 > EMA24 > EMA50 momentum

# Momentum
RSI_BULLISH_MAX = 20                  # RSI di zona 30-50
RSI_OVERBOUGHT_PENALTY_MAX = -15      # RSI > 75 (overbought)
MACD_HISTORY_BULL_MAX = 15            # MACD histogram > 0
VOL_BULL_MAX = 15                     # volume > 1.5x rata-rata 20 hari
VOL_CONFIRM_MAX = 10                  # volume 1.0-1.5x

# Risk / setup
NEAR_SUPPORT_MAX = 15                 # price dalam 3% dari swing low terakhir

# Net thresholds for classification
STRONG_BUY_THRESHOLD = 35
BUY_THRESHOLD = 15
NEUTRAL_RANGE = (-15, 15)
SELL_THRESHOLD = -15
STRONG_SELL_THRESHOLD = -35

# --- Output ------------------------------------------------------------
EXPORT_CSV = True
EXPORT_JSON = False
COLUMN_WIDTH = 14
