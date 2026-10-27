"""Pick-list backtest on the ML4T engine + diagnostics + TradeBlocks export.

Usage (inside an environment with ml4t-backtest, ml4t-diagnostic, ml4t-data, polars, pandas, yfinance):
    python picklist_backtest.py --picks picks.csv --prices prices.parquet --out outdir \
        [--holds 1,2,4,8,12,26] [--cost-bps 20] [--exclude MU,SNDK] [--benchmark ^GSPC]

picks.csv   : columns week (YYYY-MM-DD, the trading week's first day), ticker, rank (1 = best)
prices      : parquet panel timestamp, asset, open, high, low, close, volume (daily, adjusted),
              must contain all pick tickers and the benchmark ticker.
Execution   : the list for week W is known before W's open -> target weights are set on the last
              bar before W and filled at W's open (ML4T NEXT_BAR / OPEN). Weekly rotation = Monday
              open to next Monday open. Holding H weeks = H staggered tranches (each buy-and-hold,
              1/H of capital), summed.
Outputs     : <out>/<variant>/equity.parquet, trades.parquet, tb_tradelog.csv, tb_dailylog.csv,
              <out>/summary.json (ml4t-diagnostic metrics, DSR, MinTRL, White's reality check).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl
from ml4t.backtest import BacktestConfig, RebalanceConfig, Strategy, TargetWeightExecutor, run_backtest
from ml4t.backtest.config import CommissionType, MissingPricePolicy, ShareType, SlippageType

TB_COLS = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price", "Date Closed",
           "Time Closed", "Avg. Closing Cost", "Reason For Close", "P/L", "No. of Contracts", "Funds at Close",
           "Margin Req.", "Strategy", "Opening Commissions + Fees", "Closing Commissions + Fees", "P/L %"]


def load(picks_path: str, prices_path: str, benchmark: str):
    pk = pd.read_csv(picks_path)
    pk["week"] = pd.to_datetime(pk["week"])
    pk["ticker"] = pk["ticker"].astype(str).str.replace(".", "-", regex=False)
    pk = pk.sort_values(["week", "rank"]).drop_duplicates(["week", "ticker"])
    px = pl.read_parquet(prices_path).with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))
    cal = sorted(set(px.filter(pl.col("asset") == benchmark)["timestamp"].to_list()))
    weeks = [(w, g["ticker"].tolist()) for w, g in pk.groupby("week")]
    # keep only weeks whose following week-open exists in the price data
    last_bar = cal[-1]
    weeks = [w for i, w in enumerate(weeks) if i + 1 < len(weeks) and weeks[i + 1][0] <= last_bar]
    return pk, px, cal, weeks


def prev_bar(cal, d):
    return max(t for t in cal if t < d)


def config(cost_bps_per_side: float, cash: float) -> BacktestConfig:
    c = BacktestConfig()  # NEXT_BAR execution at OPEN, no costs by default
    c.missing_price_policy = MissingPricePolicy.USE_LAST
    c.initial_cash = float(cash)
    c.share_type = ShareType.FRACTIONAL
    if cost_bps_per_side > 0:  # split evenly into commission and slippage
        c.commission_type = CommissionType.PERCENTAGE
        c.commission_rate = cost_bps_per_side / 2 / 1e4
        c.slippage_type = SlippageType.PERCENTAGE
        c.slippage_rate = cost_bps_per_side / 2 / 1e4
    return c


def make_strategy(schedule: dict, gross: float = 0.98) -> Strategy:
    class PickList(Strategy):
        def __init__(self):
            self.ex = TargetWeightExecutor(RebalanceConfig(allow_short=False, max_single_weight=1.0, min_weight_change=0.0))

        def on_data(self, timestamp, data, context, broker):
            if timestamp not in schedule:
                return
            names = [a for a in schedule[timestamp] if a in data]
            w = {a: gross / len(names) for a in names} if names else {}
            held = broker.get_positions()
            for a in (list(held.keys()) if isinstance(held, dict) else [getattr(p, "asset", p) for p in held]):
                if a not in w:
                    if a in data:
                        w[a] = 0.0
                    else:
                        try:
                            broker.close_position(a)
                        except Exception:
                            pass
            self.ex.execute(w, data, broker, timestamp=timestamp)
    return PickList()


def run_variant(px, cal, weeks, hold: int, cost: float, exclude: set, capital: float = 100_000.0):
    sig_ts = [prev_bar(cal, w) for w, _ in weeks]
    end_ts = prev_bar(cal, weeks[-1][0] + pd.Timedelta(days=7))
    assets = sorted({a for _, l in weeks for a in l})
    pxs = px.filter(pl.col("asset").is_in(assets) & (pl.col("timestamp") >= sig_ts[0])
                    & (pl.col("timestamp") <= end_ts + pd.Timedelta(days=4)))
    eqs, trs = [], []
    for o in range(hold):
        sched = {}
        for i, (ts, (_, l)) in enumerate(zip(sig_ts, weeks)):
            if i == 0 or (i >= o and (i - o) % hold == 0):
                sched[ts] = [a for a in l if a not in exclude]
        sched[end_ts] = []  # liquidate at the open after the last week
        res = run_backtest(pxs, make_strategy(sched), signals=None, config=config(cost, capital / hold))
        e = res.to_equity_dataframe().to_pandas()
        eqs.append(e.set_index(e.columns[0])["equity"])
        trs.append(res.to_trades_dataframe().to_pandas().assign(tranche=o))
    return pd.concat(eqs, axis=1).ffill().sum(axis=1), pd.concat(trs, ignore_index=True)


def export_tradeblocks(eq: pd.Series, tr: pd.DataFrame, strategy: str, folder: str):
    tr = tr[tr["exit_time"].notna()].copy()
    rows = []
    for _, r in tr.iterrows():
        et, xt = pd.to_datetime(r.entry_time), pd.to_datetime(r.exit_time)
        notional = r.entry_price * r.quantity
        rows.append([et.strftime("%Y-%m-%d"), "09:30:00", round(r.entry_price, 4), f"{r.quantity:.4f} {r.symbol} STK",
                     round(notional, 2), round(r.exit_price, 4), xt.strftime("%Y-%m-%d"), "09:30:00",
                     round(r.exit_price, 4), "Rebalance", round(r.pnl, 2), 1, round(float(eq.asof(xt)), 2),
                     round(notional, 2), strategy, 0, 0, round(r.pnl_percent * 100, 3)])
    pd.DataFrame(rows, columns=TB_COLS).sort_values("Date Opened").to_csv(f"{folder}/tb_tradelog.csv", index=False)
    dl = pd.DataFrame({"Date": eq.index.strftime("%Y-%m-%d"), "Net Liquidity": eq.round(2).values})
    dl["P/L"] = dl["Net Liquidity"].diff().fillna(0).round(2)
    dl["Drawdown %"] = ((eq / eq.cummax() - 1) * 100).round(3).values
    dl.to_csv(f"{folder}/tb_dailylog.csv", index=False)


def _flat(obj) -> dict:
    d = obj.to_dict() if hasattr(obj, "to_dict") else (vars(obj) if hasattr(obj, "__dict__") else dict(obj))
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()
            if isinstance(v, (int, float, str, np.floating, np.integer)) or v is None}


def diagnostics(returns: dict, bench: pd.Series, rebalance_step: int = 5) -> dict:
    from ml4t.diagnostic.evaluation import (PortfolioAnalysis, compute_backtest_uncertainty,
                                            compute_paired_uncertainty, compute_selection_adjustment)
    from ml4t.diagnostic.evaluation.stats import compute_min_trl, deflated_sharpe_ratio, whites_reality_check
    out = {}
    for n, r in returns.items():
        s = PortfolioAnalysis(r.values, benchmark=bench.values, periods_per_year=252).compute_summary_stats()
        out[n] = _flat(s)
        # block-bootstrap confidence intervals (Sharpe, Sortino, CAGR, MDD, Calmar) + PSR p-value
        out[n]["uncertainty"] = _flat(compute_backtest_uncertainty(r.values, rebalance_step=rebalance_step))
        # paired bootstrap vs benchmark: Sharpe/return/MDD difference and information ratio with CI, p-value
        out[n]["vs_benchmark"] = _flat(compute_paired_uncertainty(r.values, bench.values, rebalance_step=rebalance_step))
    # selection bias over the whole cohort of variants (DSR of the leader, expected max Sharpe, MinTRL)
    out["_selection_adjustment"] = _flat(compute_selection_adjustment({k: v.values for k, v in returns.items()}))
    ex = [r.values - bench.values for r in returns.values()]
    d = deflated_sharpe_ratio(ex, frequency="daily")
    out["_dsr_excess_vs_benchmark"] = {"probability": float(d.probability), "p_value": float(d.p_value), "n_trials": int(d.n_trials)}
    first = next(iter(returns))
    m = compute_min_trl(returns[first].values - bench.values, frequency="daily")
    out["_min_trl_first_variant"] = {"min_trl_days": float(m.min_trl), "years": float(m.min_trl_years),
                                    "adequate": bool(m.has_adequate_sample)}
    w = whites_reality_check(bench.values, np.column_stack([r.values for r in returns.values()]), bootstrap_samples=2000)
    out["_whites_reality_check"] = {"p_value": float(w["p_value"] if isinstance(w, dict) else w.p_value)}
    return out


def holding_choice(summary: dict, holds: list, cost_bps: float, n_weeks: int) -> dict:
    """Pick the reliable hold with the highest net CAGR; break-even cost per side of each hold vs the shortest.

    A hold is 'reliable' only if the sample covers at least 4 full holding periods (n_weeks >= 4 * hold)."""
    rows = {}
    for h in holds:
        g, n = summary.get(f"{h}w-gross", {}).get("annual_return"), summary.get(f"{h}w-net", {}).get("annual_return")
        if g is None or n is None:
            continue
        rows[h] = {"gross_cagr": g, "net_cagr": n, "cost_drag_per_bp": (g - n) / cost_bps if cost_bps else 0.0,
                   "full_periods": round(n_weeks / h, 1), "reliable": n_weeks >= 4 * h}
    if not rows:
        return {}
    base = min(rows)
    for h, r in rows.items():
        if h == base:
            r["breakeven_cost_bps_vs_shortest"] = None
            continue
        gap = rows[base]["gross_cagr"] - r["gross_cagr"]            # gross advantage of the shortest hold
        drag = rows[base]["cost_drag_per_bp"] - r["cost_drag_per_bp"]  # extra cost of the shortest hold per bp
        r["breakeven_cost_bps_vs_shortest"] = (gap / drag) if drag > 0 and gap > 0 else (0.0 if gap <= 0 else None)
    ok = [h for h in rows if rows[h]["reliable"]] or list(rows)
    best = max(ok, key=lambda h: rows[h]["net_cagr"])
    return {"assumed_cost_bps_per_side": cost_bps, "weeks_in_sample": n_weeks, "best_net_hold_weeks": best,
            "rule": "highest net CAGR among reliable holds (>= 4 full holding periods)", "by_hold": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holds", default="1,2,4,8,12,26", help="holding periods in weeks (test all, do not pre-select)")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="cost per side in bps (commission+slippage)")
    ap.add_argument("--exclude", default="", help="comma-separated tickers for an ex-theme variant")
    ap.add_argument("--benchmark", default="^GSPC")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _, px, cal, weeks = load(a.picks, a.prices, a.benchmark)
    excl = {t.strip() for t in a.exclude.split(",") if t.strip()}
    variants = [(f"{h}w-gross", h, 0.0, set()) for h in map(int, a.holds.split(","))]
    variants += [(f"{h}w-net", h, a.cost_bps, set()) for h in map(int, a.holds.split(","))]
    if excl:
        variants.append(("1w-ex-theme-gross", 1, 0.0, excl))
    bench_px = px.filter(pl.col("asset") == a.benchmark).to_pandas().set_index("timestamp")["close"]
    returns = {}
    for name, h, cost, ex in variants:
        eq, tr = run_variant(px, cal, weeks, h, cost, ex)
        d = f"{a.out}/{name}"
        os.makedirs(d, exist_ok=True)
        eq.to_frame("equity").to_parquet(f"{d}/equity.parquet")
        tr.to_parquet(f"{d}/trades.parquet")
        export_tradeblocks(eq, tr, f"PickList {name}", d)
        returns[name] = eq.pct_change().fillna(0)
    idx = returns[variants[0][0]].index[1:]
    returns = {k: v.reindex(idx).fillna(0) for k, v in returns.items()}
    bench = bench_px.pct_change().reindex(idx).fillna(0)
    summary = diagnostics(returns, bench)
    summary["_holding_period_choice"] = holding_choice(summary, [int(h) for h in a.holds.split(",")], a.cost_bps,
                                                       len(weeks))
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: (v.get("annual_return"), v.get("sharpe_ratio"), v.get("max_drawdown")) if not k.startswith("_") else v
                      for k, v in summary.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
