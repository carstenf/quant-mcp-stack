"""Helpers for code run via the ml4t MCP `run_python` tool.

They connect your own ML4T code to the server's run store, so `list_runs`, `get_run`, `plot_equity`,
`plot_signal_spreads` and `plot_quantile_returns` work on results you produced yourself.

    from mcp_helpers import load_panel, pad_delisted, save_backtest_run, save_signal_run
"""
from __future__ import annotations

import json
from typing import Any

from server import (RUNS_DIR, _jsonable, _load_panel, _metrics_of, _new_run_id, _pad_delisted,  # noqa: F401
                    _rel)


def load_panel(path: str, kind: str = "prices", start: str | None = None, end: str | None = None,
               assets: list[str] | None = None):
    """Load a dataset ('prices/sp500', 'signals/x', ...) as a polars panel with timestamp/asset columns."""
    return _load_panel(path, kind, start, end, assets)


def pad_delisted(prices):
    """Keep delisted names at their last close (flat bars, volume 0) so held positions can still be exited."""
    return _pad_delisted(prices)


def save_backtest_run(result, label: str, params: dict | None = None, config=None, tearsheet: bool = True) -> str:
    """Store an ml4t.backtest BacktestResult under runs/<run_id>/ (metrics, equity, trades, fills, tear sheet).

    Returns the run_id (usable in get_run / plot_equity). Pass `config` (the BacktestConfig) and `params`
    (your strategy parameters) so the run is reproducible.
    """
    run_id = _new_run_id(label)
    rd = RUNS_DIR / run_id
    rd.mkdir(parents=True)
    metrics = _metrics_of(result)
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (rd / "params.json").write_text(json.dumps(_jsonable(params or {}), indent=2))
    if config is not None:
        try:
            (rd / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
        except Exception:
            pass
    result.to_equity_dataframe().write_parquet(rd / "equity.parquet")
    result.to_trades_dataframe().write_parquet(rd / "trades.parquet")
    try:
        result.to_fills_dataframe().write_parquet(rd / "fills.parquet")
    except Exception:
        pass
    if tearsheet:
        try:
            from ml4t.diagnostic.integration.backtest_profile import analyze_backtest_result
            from ml4t.diagnostic.visualization.backtest.tearsheet import generate_backtest_tearsheet
            generate_backtest_tearsheet(profile=analyze_backtest_result(result), output_path=str(rd / "tearsheet.html"),
                                        title=label, subtitle=run_id)
        except Exception as e:  # tear sheet is optional
            (rd / "tearsheet_error.txt").write_text(str(e)[:2000])
    print(f"saved run {run_id} -> {_rel(rd)}")
    return run_id


def save_signal_run(result, signal_col: str, meta: dict | None = None, tearsheet: bool = False) -> str:
    """Store an ml4t.diagnostic.analyze_signal result as runs/signal-.../signal.json (input for
    plot_signal_spreads / plot_quantile_returns). Returns the run_id."""
    d = result.to_dict()
    keep = {k: d.get(k) for k in ("ic", "ic_std", "ic_t_stat", "ic_p_value", "ic_ir", "ic_positive_pct",
                                  "quantile_returns", "spread", "spread_t_stat", "spread_p_value",
                                  "monotonicity", "turnover", "half_life", "n_assets", "n_dates", "date_range")}
    run_id = _new_run_id(f"signal-{signal_col}")
    rd = RUNS_DIR / run_id
    rd.mkdir(parents=True)
    out: dict[str, Any] = {"ok": True, "run_id": run_id, "signal_col": signal_col, **keep, **(meta or {})}
    try:
        out["summary"] = result.summary()
    except Exception:
        pass
    (rd / "signal.json").write_text(json.dumps(_jsonable(out), indent=2))
    if tearsheet:
        try:
            result.to_tear_sheet(signal_col).save_html(str(rd / "tearsheet.html"))
        except Exception as e:
            (rd / "tearsheet_error.txt").write_text(str(e)[:2000])
    print(f"saved signal run {run_id} -> {_rel(rd)}")
    return run_id
