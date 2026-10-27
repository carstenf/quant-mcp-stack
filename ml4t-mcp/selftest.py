"""Seed demo datasets and run every recipe once (install self-test).

Usage (as the service user, same env as the server):
    python selftest.py            -> writes $ML4T_MCP_HOME/selftest.json and prints a summary
Demo data: 39 US large caps 2014-today (Yahoo), SPY, Fama-French 5 + momentum (daily), 12-1 momentum signal.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

APP = Path(__file__).resolve().parent
HOME = Path(os.environ.get("ML4T_MCP_HOME", Path.home()))
DATA = Path(os.environ.get("ML4T_MCP_DATA", HOME / "data"))
RUNS = Path(os.environ.get("ML4T_MCP_RUNS", HOME / "runs"))
WORK = Path(os.environ.get("ML4T_MCP_WORKSPACE", HOME / "workspace"))
TICKERS = ["AAPL", "ABBV", "ADBE", "AMZN", "BA", "BAC", "C", "CAT", "COST", "CRM", "CSCO", "CVX", "DIS", "GOOGL",
           "GS", "HD", "HON", "INTC", "JNJ", "JPM", "KO", "MA", "META", "MMM", "MRK", "MS", "MSFT", "NFLX", "NVDA",
           "ORCL", "PEP", "PG", "QCOM", "TXN", "UNH", "V", "WFC", "WMT", "XOM"]


def _panel(df: pl.DataFrame) -> pl.DataFrame:
    ren = {k: v for k, v in (("symbol", "asset"), ("ticker", "asset"), ("date", "timestamp")) if k in df.columns}
    df = df.rename(ren)
    if df.schema["timestamp"] == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    if getattr(df.schema["timestamp"], "time_zone", None):
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    return (df.select("timestamp", pl.col("asset").cast(pl.Utf8), "open", "high", "low", "close", "volume")
              .drop_nulls(["close"]).unique(["timestamp", "asset"], keep="last").sort(["timestamp", "asset"]))


def seed() -> dict:
    from ml4t.data.providers import FamaFrenchProvider, YahooFinanceProvider
    for d in ("prices", "signals", "factors"):
        (DATA / d).mkdir(parents=True, exist_ok=True)
    end = dt.date.today().isoformat()
    y = YahooFinanceProvider()
    px = _panel(y.fetch_batch_ohlcv(TICKERS, "2014-01-01", end, "daily"))
    px.write_parquet(DATA / "prices" / "test-largecap40.parquet")
    _panel(y.fetch_batch_ohlcv(["SPY"], "2014-01-01", end, "daily")).write_parquet(DATA / "prices" / "spy.parquet")
    ff = FamaFrenchProvider()
    ff.fetch_factors("ff5", start="2014-01-01", end=end, frequency="daily").write_parquet(DATA / "factors" / "ff5_daily.parquet")
    ff.fetch_factors("mom", start="2014-01-01", end=end, frequency="daily").write_parquet(DATA / "factors" / "mom_daily.parquet")
    lr = pl.col("close").log().diff().over("asset")
    sig = (px.select("timestamp", "asset", "close").with_columns(lr.alias("_lr"))
             .with_columns(((pl.col("close").shift(21) / pl.col("close").shift(252)) - 1).over("asset").alias("test_mom_12_1"),
                           (pl.col("_lr").rolling_std(63).over("asset") * math.sqrt(252)).alias("vol"))
             .select("timestamp", "asset", "test_mom_12_1", "vol").drop_nulls())
    sig.write_parquet(DATA / "signals" / "test_mom_12_1.parquet")
    return {"prices_rows": px.height, "assets": px["asset"].n_unique(), "signal_rows": sig.height}


def run_recipes() -> dict:
    sys.path.insert(0, str(APP))
    from recipes import RECIPES
    WORK.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ML4T_DATA=str(DATA), ML4T_RUNS=str(RUNS), ML4T_WORKSPACE=str(WORK), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(APP), os.environ.get("PYTHONPATH")])))
    out = {}
    for name in ("momentum_signal", "rank_backtest", "analyze_signal", "deflated_sharpe", "factor_model", "portfolio_stats"):
        script = WORK / f"_selftest_{name}.py"
        script.write_text(RECIPES[name]["code"])
        p = subprocess.run([sys.executable, str(script)], cwd=WORK, env=env, capture_output=True, text=True, timeout=900)
        out[name] = {"ok": p.returncode == 0, "tail": (p.stdout if p.returncode == 0 else p.stderr)[-400:]}
    demo = DATA / "signals" / "demo_mom.parquet"
    if demo.exists():
        demo.unlink()
    return out


if __name__ == "__main__":
    res = {"time": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        res["seed"] = seed()
    except Exception as e:  # network problems etc.
        res["seed_error"] = f"{type(e).__name__}: {e}"
    res["recipes"] = run_recipes()
    res["all_ok"] = "seed_error" not in res and all(v["ok"] for v in res["recipes"].values())
    (HOME / "selftest.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"all_ok": res["all_ok"], **{k: v["ok"] for k, v in res["recipes"].items()}}))
