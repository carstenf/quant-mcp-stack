"""ML4T MCP server — equity backtesting and signal research on top of Stefan Jansen's ML4T libraries.

Libraries: ml4t-data (acquisition), ml4t-backtest (event-driven engine), ml4t-diagnostic (signal
analysis à la Alphalens, deflated Sharpe, tearsheets).

Data layout (all owned by the service user `ml4t-mcp`):
    DATA_DIR/prices/*.parquet        timestamp, asset, open, high, low, close, volume
    DATA_DIR/signals/*.parquet       timestamp, asset, <one or more numeric signal columns>
    DATA_DIR/fundamentals/*.parquet  provider-specific long format (period_end, filed_at, line_item, value)
    DATA_DIR/factors/*.parquet       timestamp + factor columns
    RUNS_DIR/<run_id>/               config.json, metrics.json, trades/equity parquet, tearsheet.html

Timing convention: a signal row stamped at bar t must only use information known at the close of t.
The engine fills orders according to the preset (default `backtrader`: next-bar open).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as imd
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from mcp.server.fastmcp import FastMCP

HOME = Path(os.environ.get("ML4T_MCP_HOME", Path.home() / "ml4t"))
DATA_DIR = Path(os.environ.get("ML4T_MCP_DATA", HOME / "data"))
RUNS_DIR = Path(os.environ.get("ML4T_MCP_RUNS", HOME / "runs"))
WORK_DIR = Path(os.environ.get("ML4T_MCP_WORKSPACE", HOME / "workspace"))
APP_DIR = Path(__file__).resolve().parent
ROOTS = {"data": DATA_DIR, "runs": RUNS_DIR, "workspace": WORK_DIR}
PRESETS = ("backtrader", "zipline", "vectorbt", "lean", "realistic")
MAX_TEXT = 40_000

log = logging.getLogger("ml4t-mcp")


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    name = name.strip().removesuffix(".parquet")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", name):
        raise ValueError(f"invalid name {name!r}: use letters, digits, . _ - (max 100)")
    return name


def _resolve(path: str, kind: str | None = None, must_exist: bool = True) -> Path:
    """Resolve a dataset reference.

    Accepted: 'prices/sp500', 'sp500' (with kind given), 'data/prices/sp500.parquet', absolute paths inside
    one of the service roots. Everything outside the roots is rejected.
    """
    p = path.strip()
    if not p:
        raise ValueError("empty path")
    candidates: list[Path] = []
    if p.startswith("/"):
        candidates.append(Path(p))
    else:
        first = p.split("/", 1)[0]
        if first in ROOTS and "/" in p:
            candidates.append(ROOTS[first] / p.split("/", 1)[1])
        if kind and "/" not in p:
            candidates.append(DATA_DIR / kind / p)
        candidates.append(DATA_DIR / p)
    out: list[Path] = []
    for c in candidates:
        for cc in (c, c.with_name(c.name + ".parquet")) if c.suffix == "" else (c,):
            r = cc.resolve()
            if not any(r.is_relative_to(root.resolve()) for root in ROOTS.values()):
                continue
            out.append(r)
    if not out:
        raise ValueError(f"path {path!r} is outside the allowed roots (data/runs/workspace)")
    if not must_exist:
        return out[0]
    for r in out:
        if r.exists():
            return r
    raise FileNotFoundError(f"dataset not found: {path!r} (tried {', '.join(str(x) for x in out)})")


def _rel(p: Path) -> str:
    for name, root in ROOTS.items():
        try:
            return f"{name}/{p.resolve().relative_to(root.resolve())}"
        except ValueError:
            continue
    return str(p)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(x, (dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def _normalize_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Canonical panel columns: timestamp (Datetime), asset (Utf8)."""
    ren = {}
    for src, dst in (("symbol", "asset"), ("ticker", "asset"), ("date", "timestamp")):
        if src in df.columns and dst not in df.columns:
            ren[src] = dst
    df = df.rename(ren)
    if "timestamp" not in df.columns or "asset" not in df.columns:
        raise ValueError(f"panel needs timestamp/date and asset/symbol/ticker columns, got {df.columns}")
    if df.schema["timestamp"] == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    elif df.schema["timestamp"] == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
    tz = getattr(df.schema["timestamp"], "time_zone", None)
    if tz:
        df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    return df.with_columns(pl.col("asset").cast(pl.Utf8)).sort(["timestamp", "asset"])


def _load_panel(path: str, kind: str, start: str | None = None, end: str | None = None,
                assets: list[str] | None = None) -> pl.DataFrame:
    p = _resolve(path, kind)
    df = _normalize_panel(pl.read_parquet(p))
    if start:
        df = df.filter(pl.col("timestamp") >= dt.datetime.fromisoformat(start))
    if end:
        df = df.filter(pl.col("timestamp") <= dt.datetime.fromisoformat(end))
    if assets:
        df = df.filter(pl.col("asset").is_in(assets))
    if df.is_empty():
        raise ValueError(f"{kind} dataset {path!r} is empty after filtering")
    return df


def _write_dataset(df: pl.DataFrame, kind: str, name: str, overwrite: bool) -> Path:
    name = _safe_name(name)
    out = DATA_DIR / kind / f"{name}.parquet"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{_rel(out)} exists; pass overwrite=true or another name")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")
    df.write_parquet(tmp)
    tmp.replace(out)
    return out


def _panel_summary(df: pl.DataFrame) -> dict:
    info: dict[str, Any] = {"rows": df.height, "columns": df.columns}
    if "timestamp" in df.columns:
        info["start"] = df["timestamp"].min()
        info["end"] = df["timestamp"].max()
    if "asset" in df.columns:
        info["assets"] = df["asset"].n_unique()
    return _jsonable(info)


def _apply_universe(df: pl.DataFrame, universe: str | None, start: str | None, end: str | None) -> pl.DataFrame:
    """Keep only (timestamp, asset) rows that are members of a point-in-time universe file."""
    if not universe:
        return df
    u = _load_panel(universe, "signals", start, end).select("timestamp", "asset")
    out = df.join(u, on=["timestamp", "asset"], how="semi")
    if out.is_empty():
        raise ValueError(f"no overlap between signals and universe {universe!r}")
    return out


def _pad_delisted(px: pl.DataFrame) -> pl.DataFrame:
    """Forward-fill assets whose price history ends before the panel end with flat bars (volume 0)."""
    end = px["timestamp"].max()
    last = px.group_by("asset").agg(pl.col("timestamp").max().alias("_last"))
    dead = last.filter(pl.col("_last") < end)
    if dead.is_empty():
        return px
    days = px.select("timestamp").unique()
    lastbar = px.join(dead, left_on=["asset", "timestamp"], right_on=["asset", "_last"], how="inner")
    pad = (dead.join(days, how="cross").filter(pl.col("timestamp") > pl.col("_last"))
           .join(lastbar.drop("timestamp").with_columns(pl.col("asset")), on="asset", how="inner"))
    c = pl.col("close")
    pad = pad.with_columns(c.alias("open"), c.alias("high"), c.alias("low"), pl.lit(0.0).alias("volume"))
    pad = pad.select(px.columns)
    return pl.concat([px, pad.cast(px.schema)]).sort(["timestamp", "asset"])


def _err(e: Exception) -> dict:
    log.warning("tool error: %s\n%s", e, traceback.format_exc())
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}


KEY_METRICS = (
    "initial_cash", "final_value", "total_return_pct", "cagr", "volatility", "sharpe", "sortino", "calmar",
    "max_drawdown_pct", "num_trades", "win_rate", "profit_factor", "payoff_ratio", "expectancy",
    "total_commission", "total_slippage", "avg_cost_drag", "avg_turnover", "avg_open_positions",
    "max_open_positions", "unique_symbols_traded", "num_rejected_orders", "skipped_bars",
)


def _metrics_of(result) -> dict:
    out = {}
    for k in KEY_METRICS:
        try:
            out[k] = result.get(k) if hasattr(result, "get") else result[k]
        except Exception:
            pass
    return _jsonable(out)


def _new_run_id(label: str | None) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_safe_name(label)}" if label else stamp


# ----------------------------------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------------------------------

def build_server(host: str, port: int) -> FastMCP:
    mcp = FastMCP(
        "ml4t",
        instructions=(
            "Equity backtesting and factor research with the ML4T libraries (ml4t-data, ml4t-backtest, "
            "ml4t-diagnostic, ml4t-engineer, ml4t-models). The libraries ARE the toolbox: this server only adds data "
            "storage, a run store, charts and an execution sandbox. Before writing any analysis or backtest code, "
            "LOOK UP what the libraries already offer: list_library_functions(query=...) searches ~2000 public "
            "classes/functions, describe_function(name) gives signature + docstring, get_recipe() gives tested "
            "templates (rank_backtest, momentum_signal, analyze_signal, deflated_sharpe, factor_model, "
            "portfolio_stats). Then run the code with run_python (`from mcp_helpers import load_panel, pad_delisted, "
            "save_backtest_run, save_signal_run` links results to list_runs/get_run/plot_*). Write own maths only "
            "for genuine gaps and say so. Conventions: datasets are parquet panels (timestamp, asset, ...) under "
            "data/prices, data/signals, data/fundamentals, data/factors; signals must be point-in-time (a row at t "
            "only uses data public at t's close; fundamentals by filing date); BacktestConfig() fills NEXT_BAR at "
            "OPEN with NO costs, so set commission/slippage explicitly (from_preset brings its own costs); count "
            "every tried variant for the Deflated Sharpe. If the env var TRADEBLOCKS_DATA is set (see run_python env), "
            "write TradeBlocks CSVs there and import them with the TradeBlocks import_csv tool."
        ),
        host=host,
        port=port,
        stateless_http=False,
    )

    # -- info / datasets -------------------------------------------------------------------------------

    @mcp.tool()
    def info() -> dict:
        """Server status: library versions, directories, available presets and dataset counts."""
        vers = {}
        for pkg in ("ml4t-backtest", "ml4t-diagnostic", "ml4t-data", "ml4t-engineer", "ml4t-models",
                    "ml4t-specs", "polars", "mcp"):
            try:
                vers[pkg] = imd.version(pkg)
            except imd.PackageNotFoundError:
                vers[pkg] = None
        counts = {d.name: len(list(d.glob("*.parquet"))) for d in sorted(DATA_DIR.iterdir())
                  if d.is_dir() and not d.name.startswith("_")} if DATA_DIR.exists() else {}
        runs = len([p for p in RUNS_DIR.iterdir() if p.is_dir()]) if RUNS_DIR.exists() else 0
        return {"ok": True, "versions": vers, "dirs": {k: str(v) for k, v in ROOTS.items()},
                "presets": PRESETS, "datasets": counts, "runs": runs}

    @mcp.tool()
    def list_datasets(kind: str | None = None) -> dict:
        """List parquet datasets (optionally only one kind: prices, signals, fundamentals, factors, ...).

        Returns size, row count, columns, date range and number of assets per file (lazy scan, schema only
        for large files).
        """
        try:
            base = DATA_DIR / _safe_name(kind) if kind else DATA_DIR
            items = []
            for p in sorted(base.rglob("*.parquet")):
                if any(part.startswith("_") for part in p.relative_to(DATA_DIR).parts):
                    continue
                entry: dict[str, Any] = {"path": _rel(p), "size_mb": round(p.stat().st_size / 1e6, 2)}
                try:
                    lf = pl.scan_parquet(p)
                    cols = lf.collect_schema().names()
                    entry["columns"] = cols[:40]
                    agg = [pl.len().alias("rows")]
                    tcol = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
                    acol = next((c for c in ("asset", "symbol", "ticker") if c in cols), None)
                    if tcol:
                        agg += [pl.col(tcol).min().alias("start"), pl.col(tcol).max().alias("end")]
                    if acol:
                        agg.append(pl.col(acol).n_unique().alias("assets"))
                    entry.update(lf.select(agg).collect().to_dicts()[0])
                except Exception as e:
                    entry["error"] = str(e)
                items.append(entry)
            return _jsonable({"ok": True, "datasets": items})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def describe_dataset(path: str, n: int = 5, asset: str | None = None) -> dict:
        """Schema, first n rows (optionally for one asset) and numeric summary statistics of a dataset.

        path: e.g. 'prices/sp500' or 'signals/mom_12_1' ('.parquet' optional).
        """
        try:
            p = _resolve(path)
            df = pl.read_parquet(p)
            if asset:
                acol = next((c for c in ("asset", "symbol", "ticker") if c in df.columns), None)
                if acol:
                    df = df.filter(pl.col(acol) == asset)
            num = [c for c, t in df.schema.items() if t.is_numeric()]
            stats = {}
            if num:
                d = df.select(num[:30]).describe()
                stats = {row["statistic"]: {k: v for k, v in row.items() if k != "statistic"}
                         for row in d.to_dicts()}
            return _jsonable({"ok": True, "path": _rel(p), "schema": {k: str(v) for k, v in df.schema.items()},
                              "rows": df.height, "head": df.head(min(n, 50)).to_dicts(), "stats": stats})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def delete_dataset(path: str) -> dict:
        """Delete one dataset file under data/ (not runs)."""
        try:
            p = _resolve(path)
            if not p.is_relative_to(DATA_DIR.resolve()):
                raise ValueError("only files under data/ can be deleted")
            p.unlink()
            return {"ok": True, "deleted": _rel(p)}
        except Exception as e:
            return _err(e)

    # -- acquisition -----------------------------------------------------------------------------------

    @mcp.tool()
    def fetch_prices(symbols: list[str], start: str, end: str, name: str, provider: str = "yahoo",
                     frequency: str = "daily", overwrite: bool = False) -> dict:
        """Download OHLCV bars and save them as data/prices/<name>.parquet (columns timestamp, asset, OHLCV).

        provider: 'yahoo' (free, split/dividend-adjusted closes; no delisted tickers → survivorship bias).
        Other ml4t-data providers need API keys and are not configured yet.
        dates: 'YYYY-MM-DD'.
        """
        try:
            if provider != "yahoo":
                raise ValueError("only provider='yahoo' is configured on this server")
            if not symbols or len(symbols) > 1500:
                raise ValueError("1..1500 symbols per call")
            from ml4t.data.providers.yahoo import YahooFinanceProvider
            t0 = time.time()
            df = YahooFinanceProvider().fetch_batch_ohlcv([s.upper().strip() for s in symbols], start, end, frequency)
            df = _normalize_panel(df).select(["timestamp", "asset", "open", "high", "low", "close", "volume"])
            df = df.drop_nulls(["close"]).unique(["timestamp", "asset"], keep="last").sort(["timestamp", "asset"])
            out = _write_dataset(df, "prices", name, overwrite)
            missing = sorted(set(s.upper() for s in symbols) - set(df["asset"].unique().to_list()))
            return _jsonable({"ok": True, "path": _rel(out), **_panel_summary(df), "missing_assets": missing,
                              "seconds": round(time.time() - t0, 1),
                              "note": "Yahoo closes are adjusted; universe contains only today's tickers (survivorship bias)."})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def fetch_factors(name: str, dataset: str = "ff3", frequency: str = "daily", start: str | None = None,
                      end: str | None = None, overwrite: bool = False) -> dict:
        """Download Fama-French factor returns (dataset 'ff3', 'ff5', 'mom', ...; frequency daily|weekly|monthly)
        to data/factors/<name>.parquet. Returns are in decimal form."""
        try:
            from ml4t.data.providers.fama_french import FamaFrenchProvider
            df = FamaFrenchProvider().fetch_factors(dataset, start=start, end=end, frequency=frequency)
            out = _write_dataset(df, "factors", name, overwrite)
            return _jsonable({"ok": True, "path": _rel(out), **_panel_summary(df)})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def fetch_fundamentals(symbols: list[str], name: str, statement: str = "income", period: str = "annual",
                           overwrite: bool = False) -> dict:
        """Download financial statements via Yahoo (statement income|balance|cashflow, period annual|quarterly)
        to data/fundamentals/<name>.parquet (long format: symbol, period_end, line_item, value, filed_at).

        WARNING: Yahoo only has the latest few periods and no filing dates — NOT point-in-time, do not use
        for historical backtests. For PIT research use Sharadar SF1 (AR dimensions, datekey) or SEC data.
        """
        try:
            from ml4t.data.providers.yahoo import YahooFinanceProvider
            y = YahooFinanceProvider()
            frames, failed = [], {}
            for s in symbols[:200]:
                try:
                    frames.append(y.fetch_financials(s.upper(), statement, period))
                except Exception as e:
                    failed[s] = str(e)[:200]
            if not frames:
                raise ValueError(f"no data returned: {failed}")
            df = pl.concat(frames, how="diagonal_relaxed")
            out = _write_dataset(df, "fundamentals", name, overwrite)
            return _jsonable({"ok": True, "path": _rel(out), "rows": df.height, "line_items": df["line_item"].n_unique(),
                              "failed": failed, "point_in_time": False})
        except Exception as e:
            return _err(e)

    # -- runs ------------------------------------------------------------------------------------------

    @mcp.tool()
    def list_runs(limit: int = 30) -> dict:
        """Most recent backtest/analysis runs with headline metrics."""
        try:
            runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)[:max(1, min(limit, 200))]
            out = []
            for r in runs:
                e: dict[str, Any] = {"run_id": r.name}
                mf = r / "metrics.json"
                if mf.exists():
                    m = json.loads(mf.read_text())
                    e.update({k: m.get(k) for k in ("cagr", "sharpe", "max_drawdown_pct", "num_trades")})
                e["tearsheet"] = (r / "tearsheet.html").exists()
                out.append(e)
            return _jsonable({"ok": True, "runs": out})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def get_run(run_id: str, trades: int = 20) -> dict:
        """Parameters, full metrics, yearly returns and the last `trades` closed trades of a run."""
        try:
            rd = RUNS_DIR / _safe_name(run_id)
            if not rd.is_dir():
                raise FileNotFoundError(run_id)
            out: dict[str, Any] = {"ok": True, "run_id": run_id, "files": sorted(p.name for p in rd.iterdir())}
            for fn in ("params.json", "metrics.json"):
                if (rd / fn).exists():
                    out[fn.removesuffix(".json")] = json.loads((rd / fn).read_text())
            if (rd / "equity.parquet").exists():
                eq = pl.read_parquet(rd / "equity.parquet")
                yr = (eq.group_by(pl.col("timestamp").dt.year().alias("year"), maintain_order=True)
                      .agg(pl.col("equity").first().alias("first"), pl.col("equity").last().alias("last"),
                           pl.col("drawdown").min().alias("max_dd")))
                prev = None
                yearly = []
                for row in yr.to_dicts():
                    base = prev if prev is not None else row["first"]
                    yearly.append({"year": row["year"], "return_pct": (row["last"] / base - 1) * 100,
                                   "max_dd_pct": row["max_dd"] * 100})
                    prev = row["last"]
                out["yearly"] = yearly
            if (rd / "trades.parquet").exists() and trades > 0:
                tr = pl.read_parquet(rd / "trades.parquet")
                cols = [c for c in ("symbol", "entry_time", "exit_time", "entry_price", "exit_price", "quantity",
                                       "pnl", "pnl_percent", "exit_reason") if c in tr.columns]
                out["trade_count"] = tr.height
                out["last_trades"] = tr.select(cols).tail(min(trades, 200)).to_dicts()
            return _jsonable(out)
        except Exception as e:
            return _err(e)

    # -- library discovery ------------------------------------------------------------------------------

    _catalog: dict[str, Any] = {}

    def _load_catalog() -> dict:
        vers = "-".join(f"{p}{imd.version(p)}" for p in ("ml4t-backtest", "ml4t-diagnostic", "ml4t-data",
                                                          "ml4t-engineer", "ml4t-models", "ml4t-specs")
                        if _has(p))
        if _catalog.get("key") == vers:
            return _catalog["data"]
        cache = DATA_DIR / "_cache" / f"api_catalog-{hashlib.sha1(vers.encode()).hexdigest()[:12]}.json"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.run([sys.executable, str(APP_DIR / "api_catalog.py"), "build", str(cache)],
                               cwd=WORK_DIR, capture_output=True, text=True, timeout=600)
            if p.returncode != 0:
                raise RuntimeError(f"catalog build failed: {p.stderr[-1500:]}")
        _catalog.update(key=vers, data=json.loads(cache.read_text()))
        return _catalog["data"]

    def _has(pkg: str) -> bool:
        try:
            imd.version(pkg)
            return True
        except imd.PackageNotFoundError:
            return False

    @mcp.tool()
    def list_library_functions(query: str | None = None, package: str | None = None,
                               kind: str | None = None, top_level_only: bool = False, limit: int = 60) -> dict:
        """Search the public API of the installed ML4T libraries (classes + functions, one-line docs, import line).

        Call this BEFORE writing analysis/backtest code. query: words matched against name and docstring (all
        words must match, case-insensitive; e.g. 'sharpe', 'walk forward', 'factor model', 'stop loss',
        'rebalance', 'triple barrier'). package: backtest | diagnostic | data | engineer | models | specs.
        kind: class | function. top_level_only: only names exported by the package root. Without query: counts
        per module plus the top-level exports. Then use describe_function for details.
        """
        try:
            cat = _load_catalog()
            items = cat["items"]
            sel = []
            words = [w.lower() for w in (query or "").split() if w]
            for q, it in items.items():
                if package and not q.startswith(f"ml4t.{package}."):
                    continue
                if kind and it["kind"] != kind:
                    continue
                if top_level_only and not it.get("top_level"):
                    continue
                hay = f"{q} {it['doc']}".lower().replace("_", " ")
                if words and not all(w in hay or w in q.lower() for w in words):
                    continue
                sel.append((q, it))
            if not words:
                from collections import Counter
                mods = Counter(".".join(q.split(".")[:3]) for q, _ in sel)
                tops = [{"name": q, "kind": it["kind"], "doc": it["doc"], "import": it["import"]}
                        for q, it in sel if it.get("top_level")]
                return _jsonable({"ok": True, "versions": cat["versions"], "total": len(sel),
                                  "modules": dict(mods.most_common(80)), "top_level": tops[:max(1, min(limit, 400))]})
            # rank: names containing all words first, then exported names
            def _rank(x):
                leaf = x[0].rsplit(".", 1)[-1].lower()
                return (not all(w in leaf for w in words), not any(w in leaf for w in words),
                        not x[1].get("top_level"), x[1]["kind"] != "function", len(x[0]))
            sel.sort(key=_rank)
            out = [{"name": q, "kind": it["kind"], "doc": it["doc"], "import": it["import"]}
                   for q, it in sel[:max(1, min(limit, 300))]]
            return _jsonable({"ok": True, "versions": cat["versions"], "matches": len(sel), "items": out})
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def describe_function(name: str) -> dict:
        """Signature, full docstring and (for classes) fields and public methods of an ML4T object.

        name: dotted path as returned by list_library_functions, e.g.
        'ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio' or 'ml4t.backtest.BacktestConfig'. Modules work
        too (lists their members).
        """
        try:
            if not re.fullmatch(r"ml4t(\.[A-Za-z_][A-Za-z0-9_]*)+", name.strip()):
                raise ValueError("name must be a dotted ml4t.* path")
            p = subprocess.run([sys.executable, str(APP_DIR / "api_catalog.py"), "describe", name.strip()],
                               cwd=WORK_DIR, capture_output=True, text=True, timeout=180)
            if p.returncode != 0:
                raise RuntimeError(p.stderr[-1500:])
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def get_recipe(name: str | None = None) -> dict:
        """Tested run_python templates that use the ML4T libraries with this server's conventions.

        Without name: list of recipes with descriptions. With name: the code (adapt dataset names/parameters,
        keep the conventions). Available: rank_backtest, momentum_signal, analyze_signal, deflated_sharpe,
        factor_model, portfolio_stats.
        """
        try:
            from recipes import RECIPES
            if not name:
                return {"ok": True, "recipes": {k: v["description"] for k, v in RECIPES.items()}}
            if name not in RECIPES:
                raise KeyError(f"unknown recipe {name!r}; available: {sorted(RECIPES)}")
            return {"ok": True, "name": name, **RECIPES[name]}
        except Exception as e:
            return _err(e)

    # -- escape hatch -----------------------------------------------------------------------------------

    @mcp.tool()
    def run_python(code: str, timeout_s: int = 300) -> dict:
        """Run arbitrary Python in the server's venv (ml4t-backtest, ml4t-diagnostic, ml4t-data, polars,
        pandas, numpy available) as a subprocess with cwd=workspace/.

        This is where backtests, signals and statistics are done - with the library functions (find them via
        list_library_functions / describe_function, start from get_recipe). `from mcp_helpers import load_panel,
        pad_delisted, save_backtest_run, save_signal_run` connects results to the run store and plot tools.
        Signals: write parquet to os.environ['ML4T_DATA']+'/signals/<name>.parquet'. Env: ML4T_DATA, ML4T_RUNS,
        ML4T_WORKSPACE. Returns exit code and truncated stdout/stderr. Print compact results.
        """
        try:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            script = WORK_DIR / f"_run_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.py"
            script.write_text(code)
            env = dict(os.environ, ML4T_DATA=str(DATA_DIR), ML4T_RUNS=str(RUNS_DIR),
                       ML4T_WORKSPACE=str(WORK_DIR), PYTHONUNBUFFERED="1",
                       PYTHONPATH=os.pathsep.join(filter(None, [str(APP_DIR), os.environ.get("PYTHONPATH")])))
            t0 = time.time()
            p = subprocess.run([sys.executable, str(script)], cwd=WORK_DIR, env=env, capture_output=True,
                               text=True, timeout=max(5, min(timeout_s, 1800)))

            def cut(s: str) -> str:
                return s if len(s) <= MAX_TEXT else "...[truncated]...\n" + s[-MAX_TEXT:]

            return {"ok": p.returncode == 0, "exit_code": p.returncode, "seconds": round(time.time() - t0, 1),
                    "stdout": cut(p.stdout), "stderr": cut(p.stderr), "script": _rel(script)}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout_s} s"}
        except Exception as e:
            return _err(e)

    if (APP_DIR / "sharadar_tools.py").exists():  # optional: licensed Sharadar bulk data
        import sharadar_tools
        sharadar_tools.register(mcp, DATA_DIR=DATA_DIR, _load_panel=_load_panel, _write_dataset=_write_dataset,
                                _rel=_rel, _jsonable=_jsonable, _err=_err, _safe_name=_safe_name,
                                _panel_summary=_panel_summary)
    import plots
    plots.register(mcp, RUNS_DIR=RUNS_DIR, _load_panel=_load_panel, _safe_name=_safe_name, _rel=_rel,
                   _jsonable=_jsonable, _err=_err)
    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("ML4T_MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ML4T_MCP_PORT", "3250")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for d in (DATA_DIR, RUNS_DIR, WORK_DIR):
        d.mkdir(parents=True, exist_ok=True)
    build_server(args.host, args.port).run(transport=os.environ.get("ML4T_MCP_TRANSPORT", "streamable-http"))


if __name__ == "__main__":
    main()
