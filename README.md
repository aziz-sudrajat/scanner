# Signal Scanner

Tools Python untuk memindai kripto secara berantai dan menghasilkan signal beli/jual
berdasarkan skor multi-faktor.

Dibuat sebagai satu paket Python bersih (`scanner/`) + entrypoint `main.py`.

## Struktur

```
~/Projects/scanner/
├── main.py                  # entrypoint CLI
├── scanner/
│   ├── __init__.py          # version
│   ├── config.py            # konfigurasi & threshold semua skor
│   ├── fetch.py             # fetch data CoinGecko (snapshot + OHLCV)
│   ├── indicators.py        # EMA / RSI / MACD / volume / support-range
│   ├── signals.py           # engine skoring (snapshot + OHLCV)
│   └── report.py            # tabel CLI + CSV + JSON
├── requirements.txt
└── README.md
```

## Cara jalan

### Mode 1 — Snapshot (direkomendasikan, cepat & andal)
Pakai data live `GET /coins/markets` (1 HTTP call). Cocok saat API dipicu
rata-rata free-tier. Skornya dari momentum 24h, aktivitas volume, volatilitas,
dan likuiditas.

```bash
python3 main.py                 # scan top 30 coin
python3 main.py --limit 50      # lebih banyak coin
python3 main.py --signal BUY    # hanya tampilkan CASH BUY
python3 main.py --export csv,json
```

### Mode 2 — Technical (lebih mendalam)
Fetch OHLCV per coin, hitung EMA200 / EMA50 / EMA20, RSI(14), MACD, volume ratio,
support-range, lalu skor multi-faktor (trend + momentum + volume + setup).

```bash
python3 main.py --mode technical --limit 10
```

> Catatan: mode technical butuh akses history dan sensitif rate-limit API. Jika
> rate-limitnya sibuk, otomatis fallback ke mode snapshot.

## Metodologi signal (kualitas)

Skor bersifat multi-faktor (bukan 1 indikator), sehingga lebih tahan noise:

1. **Trend** — price vs EMA200, alignment EMA20>EMA50>EMA200, EMA12>EMA24>EMA50
2. **Momentum** — RSI(14) di zone 30–50, MACD histogram, volume vs rata-rata 20 hari
3. **Liquidity** — ukuran cap pasar (payung kecil) & price vs support-range
4. **Risk setup** — price dekat support (bounce) / dekat resistance (breakout)

Output tiap coin: `score` (0–100), `signal` (STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL),
`confidence`, + faktor penentu (supaya bisa di-audit).

## Prerequisites

Python 3.10+ dan paket di bawah ini:

```bash
pip install -r requirements.txt
# atau pakai distro system (sudah terpasang di env ini):
# apt-get install -y python3-pandas python3-numpy python3-requests
```

## Limitations

- Data: CoinGecko public API (free, tanpa API key). Rate-limit free tier bisa
  ketat seiring cliffi. Tools punya retry + backoff 429; mode technical melakukan
  fallback ke snapshot otomatis saat rate-limit sibuk.
- Signal ini **bukan sahih (unrealiable)** — hanya *analysis* atau *tech watchlist*.
  Selalu lakukan riset sendiri sebelum bertrading.
