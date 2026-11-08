"""Pick-list backtest on the ML4T engine + diagnostics + TradeBlocks export.

Built for third-party pick lists whose history may be filtered: data checks, survivorship handling,
publication split, entry-delay test, cost grid, unknown-trial DSR sensitivity.

Usage (inside an environment with ml4t-backtest, ml4t-diagnostic, polars, pandas):
    python picklist_backtest.py --picks picks.csv --prices prices.parquet --out outdir \
        [--holds 1,2,4,8,12,26] [--cost-bps 20] [--cost-grid 10,50] [--delays 1,2,5] \
        [--exclude MU,SNDK] [--benchmark SPY] [--published-from 2024-03-01] \
        [--delist-map delist.csv] [--delist-return 0] [--vendor-trials 1,10,100,1000]

picks.csv   : columns week (YYYY-MM-DD, the trading week's first day), ticker, rank (1 = best)
prices      : parquet panel timestamp, asset, open, high, low, close, volume (daily, split AND dividend
              adjusted, i.e. total return), incl. delisted names and the benchmark (use a total-return
              series such as adjusted SPY, not the price index ^GSPC).
delist-map  : optional csv asset,return - terminal return applied on the bar after an asset's last price
              (e.g. -1.0 bankruptcy, actual premium for an acquisition). Assets not in the map get
              --delist-return. A name counts as delisted if its last bar is >= 5 bars before the panel end.
Execution   : the list for week W is known before W's open -> target weights are set on the last bar
              before W (+ delay bars) and filled at the next open (ML4T NEXT_BAR / OPEN). Weekly
              rotation = Monday open to next Monday open. Holding H weeks = H staggered tranches (each
              buy-and-hold, 1/H of capital), summed. Names without a price on the signal bar stay in
              CASH (their weight is not redistributed to the survivors).
Outputs     : <out>/data_checks.json, <out>/<variant>/{equity.parquet, trades.parquet, tb_tradelog.csv,
              tb_dailylog.csv}, <out>/summary.json.
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
DELIST_GAP_BARS = 5
RECORD_TYPES = {  # name -> (evidence class, can reach grade A)
    "audited": ("real money, verifiable statements", True),
    "self-reported": ("real money, vendor's own figures only - counted as vendor paper until statements are seen", False),
    "paper-independent": ("signals recorded by an independent party (you, a colleague) as they arrived", True),
    "paper-vendor": ("paper / model portfolio published by the vendor with timestamps", False),
    "unknown": ("unknown or hypothetical, e.g. produced by a backtest", False),
}
REJECTED = [0]  # rejected engine orders of the current variant (should be 0; otherwise the tape is not as intended)


# ----------------------------------------------------------------------------------------------- data
def load(picks_path: str, prices_path: str, benchmark: str):
    raw = pd.read_csv(picks_path)
    raw["week"] = pd.to_datetime(raw["week"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    dup_rows = int(raw.duplicated(["week", "ticker"]).sum())
    pk = raw.sort_values(["week", "rank"]).drop_duplicates(["week", "ticker"])
    px = pl.read_parquet(prices_path).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone(None))
    cal = sorted(set(px.filter(pl.col("asset") == benchmark)["timestamp"].to_list()))
    if not cal:
        raise SystemExit(f"benchmark {benchmark} not in price panel")
    weeks_all = [(w, g["ticker"].tolist()) for w, g in pk.groupby("week")]
    # a week is usable if the following Monday open (week + 7 days) is inside the price data
    weeks = [w for w in weeks_all if w[0] + pd.Timedelta(days=7) <= cal[-1]]
    checks = data_checks(raw, pk, px, cal, weeks_all, weeks, dup_rows, benchmark)
    return pk, px, cal, weeks, checks


def data_checks(raw, pk, px, cal, weeks_all, weeks, dup_rows, benchmark) -> dict:
    """Signs of an incomplete or edited history. Report all of them; none is proof of filtering."""
    sizes = pk.groupby("week").size()
    wk = sorted(sizes.index)
    gaps = [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(wk, wk[1:]) if (b - a).days > 9]
    lists = [frozenset(l) for _, l in weeks_all]
    same_as_prev = [weeks_all[i][0].strftime("%Y-%m-%d") for i in range(1, len(lists)) if lists[i] == lists[i - 1]]
    first = px.group_by("asset").agg(pl.col("timestamp").min().alias("first"), pl.col("timestamp").max().alias("last"))
    span = {r["asset"]: (r["first"], r["last"]) for r in first.iter_rows(named=True)}
    tickers = sorted(pk["ticker"].unique())
    no_price = [t for t in tickers if t not in span]
    end_cut = cal[-DELIST_GAP_BARS] if len(cal) > DELIST_GAP_BARS else cal[-1]
    delisted = sorted(t for t in tickers if t in span and span[t][1] < end_cut)
    first_pick = pk.groupby("ticker")["week"].min()
    last_pick = pk.groupby("ticker")["week"].max()
    pre_listing = sorted(t for t in tickers if t in span and span[t][0] > first_pick[t])
    after_delist = sorted(t for t in delisted if last_pick[t] > span[t][1])
    weekday = pk["week"].dt.day_of_week.value_counts().to_dict()
    turnover = [1 - len(lists[i] & lists[i - 1]) / max(len(lists[i]), 1) for i in range(1, len(lists))]
    return {
        "weeks_in_file": len(weeks_all), "weeks_usable": len(weeks),
        "first_week": wk[0].strftime("%Y-%m-%d"), "last_week": wk[-1].strftime("%Y-%m-%d"),
        "list_size": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max()),
                      "weeks_off_median": int((sizes != sizes.median()).sum())},
        "duplicate_rows": dup_rows, "missing_weeks": gaps, "identical_to_previous_week": same_as_prev,
        "week_start_weekdays": {int(k): int(v) for k, v in weekday.items()},
        "list_turnover_per_week_median": float(np.median(turnover)) if turnover else None,
        "tickers": len(tickers), "tickers_without_prices": no_price,
        "share_picks_without_prices": float(pk["ticker"].isin(no_price).mean()),
        "delisted_in_sample": delisted, "picked_before_first_price": pre_listing,
        "picked_after_last_price": after_delist, "benchmark": benchmark,
    }


def apply_delisting(px: pl.DataFrame, cal: list, delisted: list, delist_map: dict, default_ret: float):
    """Append one terminal bar after the last price (terminal return), then keep the name flat to the end
    so held positions can be exited by the engine (volume 0 -> flags illiquidity)."""
    out = [px]
    for a in delisted:
        g = px.filter(pl.col("asset") == a).sort("timestamp")
        last = g.row(-1, named=True)
        r = float(delist_map.get(a, default_ret))
        lvl = max(last["close"] * (1 + r), 1e-6)
        after = [t for t in cal if t > last["timestamp"]]
        if not after:
            continue
        rows = [{"timestamp": t, "asset": a, "open": lvl, "high": lvl, "low": lvl, "close": lvl, "volume": 0.0}
                for t in after]
        out.append(pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))))
    cols = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
    return pl.concat([d.select(cols).with_columns(pl.col("volume").cast(pl.Float64),
                                                  *[pl.col(c).cast(pl.Float64) for c in cols[2:6]]) for d in out])


def prev_bar(cal, d):
    return max(t for t in cal if t < d)


# --------------------------------------------------------------------------------------------- engine
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
            listed = schedule[timestamp]
            # weight per name uses the FULL list: names without a price stay in cash (no survivor reweighting)
            w = {a: gross / len(listed) for a in listed if a in data} if listed else {}
            # names leaving the list are closed with close_position (exact quantity). Selling them via a
            # target weight of 0 lets the executor size the sell from the order request instead of the filled
            # position -> "position reversal" rejections, positions kept by accident, costs understated.
            held = broker.get_positions()
            for a in (list(held.keys()) if isinstance(held, dict) else [getattr(p, "asset", p) for p in held]):
                if a not in w:
                    try:
                        broker.close_position(a)
                    except Exception:
                        pass
            self.ex.execute(w, data, broker, timestamp=timestamp)
    return PickList()


def run_variant(px, cal, weeks, hold: int, cost: float, exclude: set, delay: int = 0, capital: float = 100_000.0):
    idx = {t: i for i, t in enumerate(cal)}
    sig_ts = [cal[min(idx[prev_bar(cal, w)] + delay, len(cal) - 1)] for w, _ in weeks]
    end_ts = cal[min(idx[prev_bar(cal, weeks[-1][0] + pd.Timedelta(days=7))] + delay, len(cal) - 2)]
    assets = sorted({a for _, l in weeks for a in l})
    pxs = px.filter(pl.col("asset").is_in(assets) & (pl.col("timestamp") >= sig_ts[0])
                    & (pl.col("timestamp") <= end_ts + pd.Timedelta(days=4)))
    eqs, trs = [], []
    for o in range(hold):
        sched = {}
        for i, (ts, (_, l)) in enumerate(zip(sig_ts, weeks)):
            # every tranche starts fully invested in week 0 (ramp-up), then rotates every `hold` weeks
            if i == 0 or (i >= o and (i - o) % hold == 0):
                sched[ts] = [a for a in l if a not in exclude]
        sched[end_ts] = []  # liquidate at the open after the last week
        res = run_backtest(pxs, make_strategy(sched), signals=None, config=config(cost, capital / hold))
        e = res.to_equity_dataframe().to_pandas()
        eqs.append(e.set_index(e.columns[0])["equity"])
        trs.append(res.to_trades_dataframe().to_pandas().assign(tranche=o))
        try:
            REJECTED[0] += len(res.to_rejected_orders_dataframe())
        except Exception:
            pass
    return pd.concat(eqs, axis=1).ffill().sum(axis=1), pd.concat(trs, ignore_index=True)


def turnover(eq: pd.Series, tr: pd.DataFrame) -> float:
    """Annualised one-way turnover = traded notional / 2 / mean equity / years."""
    if tr.empty:
        return 0.0
    notional = (tr["entry_price"].abs() * tr["quantity"].abs()).sum()
    notional += (tr["exit_price"].fillna(0).abs() * tr["quantity"].abs()).sum()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return float(notional / 2 / eq.mean() / years)


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


# ------------------------------------------------------------------------------------------- per pick
def pick_stats(px: pl.DataFrame, cal: list, weeks: list, exclude: set) -> dict:
    """Per-pick Monday-open to next-Monday-open returns: win rate, payoff, share of a 3:1 target,
    top contributors, liquidity and a participation-based capacity estimate."""
    p = px.select("timestamp", "asset", "open", "close", "volume").to_pandas()
    op = p.pivot(index="timestamp", columns="asset", values="open")
    dv = (p.assign(dv=p["close"] * p["volume"]).pivot(index="timestamp", columns="asset", values="dv")
          .rolling(20, min_periods=5).median())
    rows = []
    for w, l in weeks:
        t0 = min(t for t in cal if t >= w)
        t1 = min((t for t in cal if t >= w + pd.Timedelta(days=7)), default=None)
        if t1 is None:
            continue
        sig = prev_bar(cal, w)
        for a in l:
            if a in exclude or a not in op.columns:
                continue
            o0, o1 = op.at[t0, a] if t0 in op.index else np.nan, op.at[t1, a] if t1 in op.index else np.nan
            if np.isfinite(o0) and np.isfinite(o1) and o0 > 0:
                rows.append((w, a, o1 / o0 - 1, dv.at[sig, a] if sig in dv.index else np.nan))
    df = pd.DataFrame(rows, columns=["week", "ticker", "ret", "dollar_vol"])
    if df.empty:
        return {}
    wins, losses = df.loc[df.ret > 0, "ret"], df.loc[df.ret < 0, "ret"]
    payoff = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None
    n_per_week = df.groupby("week").size().median()
    contrib = (df.groupby("ticker")["ret"].sum() / n_per_week).sort_values(ascending=False)
    total = float(df.groupby("week")["ret"].mean().sum())
    adv = df["dollar_vol"].dropna()

    def cap(part, q):
        return float(part * n_per_week * adv.quantile(q)) if len(adv) else None
    return {
        "n_picks": int(len(df)), "win_rate": float((df.ret > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": payoff, "share_of_3to1_target": (payoff / 3) if payoff else None,
        "sum_weekly_portfolio_returns": total,
        "top10_contributors": {k: float(v) for k, v in contrib.head(10).items()},
        "top10_share_of_sum": float(contrib.head(10).sum() / total) if total else None,
        "liquidity": {"median_dollar_volume": float(adv.median()) if len(adv) else None,
                      "share_below_5m": float((adv < 5e6).mean()) if len(adv) else None,
                      "share_below_1m": float((adv < 1e6).mean()) if len(adv) else None,
                      "share_zero_volume": float((df["dollar_vol"].fillna(0) == 0).mean())},
        # weekly full rotation: each rebalance trades ~AUM/N per name -> AUM <= participation * N * ADV
        "capacity_aum_usd": {"1pct_adv_p10_name": cap(0.01, 0.10), "1pct_adv_median_name": cap(0.01, 0.50),
                             "5pct_adv_p10_name": cap(0.05, 0.10)},
    }


# ---------------------------------------------------------------------------------------- diagnostics
def _flat(obj) -> dict:
    d = obj.to_dict() if hasattr(obj, "to_dict") else (vars(obj) if hasattr(obj, "__dict__") else dict(obj))
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()
            if isinstance(v, (int, float, str, np.floating, np.integer)) or v is None}


def stats(r: pd.Series, b: pd.Series, full: bool = True, step: int = 5) -> dict:
    from ml4t.diagnostic.evaluation import PortfolioAnalysis, compute_backtest_uncertainty, compute_paired_uncertainty
    out = _flat(PortfolioAnalysis(r.values, benchmark=b.values, periods_per_year=252).compute_summary_stats())
    if full:
        out["uncertainty"] = _flat(compute_backtest_uncertainty(r.values, rebalance_step=step))
        out["vs_benchmark"] = _flat(compute_paired_uncertainty(r.values, b.values, rebalance_step=step))
    return out


def diagnostics(returns: dict, bench: pd.Series, trial_names: list) -> dict:
    from ml4t.diagnostic.evaluation import compute_selection_adjustment
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio, whites_reality_check
    out = {n: stats(r, bench, full=not n.startswith(("cost", "delay"))) for n, r in returns.items()}
    trials = {k: returns[k] for k in trial_names}
    if len(trials) > 1:
        out["_selection_adjustment"] = _flat(compute_selection_adjustment({k: v.values for k, v in trials.items()}))
    ex = [r.values - bench.values for r in trials.values()]
    d = deflated_sharpe_ratio(ex, frequency="daily")
    out["_dsr_excess_vs_benchmark"] = {"probability": float(d.probability), "p_value": float(d.p_value),
                                       "n_trials": int(d.n_trials), "trials": trial_names,
                                       "note": "n_trials = 1 -> this is the PSR (no selection among variants)" if len(trials) == 1 else ""}
    w = whites_reality_check(bench.values, np.column_stack([r.values for r in trials.values()]), bootstrap_samples=2000, random_state=42)
    out["_whites_reality_check"] = {"p_value": float(w["p_value"] if isinstance(w, dict) else w.p_value)}
    return out


def vendor_trials(returns: dict, bench: pd.Series, leader: str, ns: list) -> dict:
    """Unknown number of variants the vendor tried before publishing: PSR of the leader's excess return
    against the expected max Sharpe of N zero-skill strategies (estimator variance ~ 1/T, native frequency)."""
    from ml4t.diagnostic.evaluation.stats import compute_expected_max_sharpe, deflated_sharpe_ratio
    x = (returns[leader] - bench).values

    def prob(n):
        sr0 = 0.0 if n <= 1 else compute_expected_max_sharpe(n, 1.0 / len(x))
        r = deflated_sharpe_ratio(x, frequency="daily", benchmark_sharpe=sr0)
        return sr0, float(r.probability), float(r.p_value)
    sens = {str(n): dict(zip(["sr0_daily", "probability", "p_value"], prob(n))) for n in ns}
    grid = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    first_fail = next((n for n in grid if prob(n)[1] < 0.95), None)
    return {"leader": leader, "by_assumed_trials": sens,
            "significant_up_to_trials": (None if first_fail == 1 else
                                         grid[grid.index(first_fail) - 1] if first_fail else ">10000"),
            "note": "excess vs benchmark; probability >= 0.95 = significant; None = not significant even for 1 trial"}


def holding_choice(summary: dict, holds: list, cost_bps: float, n_weeks: int) -> dict:
    """Pick the reliable hold with the highest net CAGR; break-even cost per side of each hold vs the shortest.

    A hold is 'reliable' only if the sample covers at least 4 full holding periods (n_weeks >= 4 * hold)."""
    rows = {}
    for h in holds:
        g, n = summary.get(f"{h}w-gross", {}).get("annual_return"), summary.get(f"{h}w-net", {}).get("annual_return")
        if g is None or n is None:
            continue
        rows[h] = {"gross_cagr": g, "net_cagr": n, "cost_drag_per_bp": (g - n) / cost_bps if cost_bps else 0.0,
                   "turnover": summary.get(f"{h}w-net", {}).get("turnover"),
                   "full_periods": round(n_weeks / h, 1), "reliable": n_weeks >= 4 * h}
    if not rows:
        return {}
    base = min(rows)
    for h, r in rows.items():
        if h == base:
            r["breakeven_cost_bps_vs_shortest"] = None
            continue
        gap = rows[base]["gross_cagr"] - r["gross_cagr"]
        drag = rows[base]["cost_drag_per_bp"] - r["cost_drag_per_bp"]
        r["breakeven_cost_bps_vs_shortest"] = (gap / drag) if drag > 0 and gap > 0 else (0.0 if gap <= 0 else None)
        # cost per side at which this hold's net CAGR is zero (linear in the measured drag)
    for r in rows.values():
        r["breakeven_cost_bps_to_zero"] = (r["gross_cagr"] / r["cost_drag_per_bp"]) if r["cost_drag_per_bp"] > 0 else None
    ok = [h for h in rows if rows[h]["reliable"]] or list(rows)
    best = max(ok, key=lambda h: rows[h]["net_cagr"])
    return {"assumed_cost_bps_per_side": cost_bps, "weeks_in_sample": n_weeks, "best_net_hold_weeks": best,
            "rule": "highest net CAGR among reliable holds (>= 4 full holding periods)", "by_hold": rows}


def plateau_holds(summary: dict, holds: list, n_weeks: int, tol: float = 0.25, alt_tol: float = 0.50) -> dict:
    """Plateau rule on the 1-D hold grid (metric: net CAGR, reliable holds only, grid order = holds order).

    Stated before choosing: a hold belongs to the plateau if its net CAGR is within tol * |best| of the best;
    regions are runs of adjacent holds; the centre is the middle actual hold of the widest region
    (even count: the inner of the two middle holds, i.e. the one farther from a grid boundary; tie: the shorter hold).
    Repeated with alt_tol; rule_dependent = the centre changes."""
    vals = [(h, summary.get(f"{h}w-net", {}).get("annual_return")) for h in holds if n_weeks >= 4 * h]
    vals = [(h, v) for h, v in vals if v is not None]
    if len(vals) < 3:
        return {"note": "fewer than 3 reliable holds - no neighbourhood to judge", "holds": [h for h, _ in vals]}
    hs = [h for h, _ in vals]
    best_h, best_v = max(vals, key=lambda x: x[1])

    def run(t):
        thr = best_v - t * abs(best_v)
        runs, cur = [], []
        for h, v in vals:
            if v >= thr:
                cur.append(h)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        widest = max(runs, key=lambda r: (len(r), best_h in r))
        if len(widest) % 2:
            centre = widest[len(widest) // 2]
        else:
            a_, b_ = widest[len(widest) // 2 - 1], widest[len(widest) // 2]
            da, db = min(hs.index(a_), len(hs) - 1 - hs.index(a_)), min(hs.index(b_), len(hs) - 1 - hs.index(b_))
            centre = a_ if (da > db or (da == db and a_ < b_)) else b_
        return {"tolerance": t, "threshold_net_cagr": thr, "regions": runs, "widest_region": widest, "centre_hold_weeks": centre}
    main, alt = run(tol), run(alt_tol)
    i = hs.index(best_h)
    lead_region = next(r for r in main["regions"] if best_h in r)
    flags = []
    if i in (0, len(hs) - 1):
        flags.append("leader on grid boundary")
    if main["widest_region"] == hs:
        flags.append("flat grid: all reliable holds within tolerance - the hold choice matters little; centre is arbitrary")
    if len(lead_region) <= 2:
        flags.append(f"leader region has only {len(lead_region)} hold(s)")
    for j in (i - 1, i + 1):
        if 0 <= j < len(hs) and best_v - vals[j][1] > 2 * tol * abs(best_v):
            flags.append(f"cliff next to leader ({hs[j]}w)")
    return {"metric": "net CAGR", "best_hold_weeks": best_h, "best_net_cagr": best_v, "by_hold": dict(vals),
            "main": main, "alt": alt, "rule_dependent": main["centre_hold_weeks"] != alt["centre_hold_weeks"],
            "leader_region": lead_region, "flags": flags,
            "note": "a grid winner is a lead, not a finding; compare leader vs centre with the paired test"}


def paired_leader_centre(returns: dict, leader: str, centre: str, step: int = 5) -> dict:
    """Paired block-bootstrap of leader minus centre (same ml4t-diagnostic function as the benchmark test).
    Not adjusted for having selected both holds from the grid."""
    from ml4t.diagnostic.evaluation import compute_paired_uncertainty
    if leader == centre:
        return {"skipped": "leader and plateau centre coincide - no second arm"}
    out = _flat(compute_paired_uncertainty(returns[leader].values, returns[centre].values, rebalance_step=step))
    out.update({"leader": leader, "centre": centre, "note": "not adjusted for selecting both holds from the grid"})
    return out


def day_dependence(r: pd.Series, bench: pd.Series) -> dict:
    """Share of the total log return from the best 1/5/10 days, the result without them, and per-year returns."""
    lr = np.log1p(r.values)
    tot = float(lr.sum())
    order = np.argsort(lr)[::-1]
    yrs = len(r) / 252.0
    out = {"total_return": float(np.expm1(tot)), "cagr": float(np.expm1(tot / yrs)) if yrs > 0 else None}
    for k in (1, 5, 10):
        top = float(lr[order[:k]].sum())
        rest = tot - top
        out[f"best_{k}_days"] = {"dates": [str(r.index[j].date()) for j in order[:k]],
                                 "share_of_log_return": (top / tot) if tot else None,
                                 "total_return_without": float(np.expm1(rest)),
                                 "cagr_without": float(np.expm1(rest / yrs)) if yrs > 0 else None}
    yr = pd.Series(lr, index=r.index).groupby(r.index.year).sum()
    by = pd.Series(np.log1p(bench.values), index=bench.index).groupby(bench.index.year).sum()
    out["by_year"] = {str(y): {"return": float(np.expm1(yr[y])), "benchmark": float(np.expm1(by.get(y, 0.0)))} for y in yr.index}
    return out


def min_improvement_check(summary: dict, leader: str, centre: str | None, bench_cagr: float | None, mi_pp: float | None) -> dict:
    """Compare differences with the minimum improvement (net CAGR, percentage points) stated before looking."""
    if mi_pp is None:
        return {"stated": None, "note": "no minimum improvement was stated before looking (--min-improvement-pp) - report this"}
    lc = summary.get(leader, {}).get("annual_return")
    res = {"stated_pp": mi_pp}
    if bench_cagr is not None and lc is not None:
        d = (lc - bench_cagr) * 100
        res["leader_vs_benchmark_pp"] = d
        res["leader_vs_benchmark_meets"] = d >= mi_pp
    if centre and centre != leader:
        cc = summary.get(centre, {}).get("annual_return")
        if cc is not None and lc is not None:
            d = (lc - cc) * 100
            res["leader_vs_centre_pp"] = d
            res["leader_vs_centre_meets"] = d >= mi_pp
    return res


def split_stats(returns: dict, bench: pd.Series, names: list, date: str) -> dict:
    """Before vs after the first verifiable publication. Only 'after' is genuine out-of-sample."""
    t = pd.Timestamp(date)
    out = {}
    for n in names:
        r = returns[n]
        pre, post = r[r.index < t], r[r.index >= t]
        out[n] = {"pre": stats(pre, bench.reindex(pre.index), full=False) if len(pre) > 20 else None,
                  "post": stats(post, bench.reindex(post.index), full=len(post) > 60) if len(post) > 20 else None,
                  "days_pre": int(len(pre)), "days_post": int(len(post))}
    return {"published_from": date, "variants": out}


def evidence_grade(summary: dict, returns: dict, bench: pd.Series, variant: str, a, conservative: str | None) -> dict:
    """How much the record can prove. A = out-of-sample record nobody could filter (audited real money or
    independently recorded signals), long and significant enough on its own; B = some genuine out-of-sample
    evidence, not yet sufficient; C = in-sample only. Records without real fills are graded on the
    conservative cost level."""
    from ml4t.diagnostic.evaluation.stats import compute_min_trl
    rt = a.record_type
    paper = rt != "audited"
    used = conservative if (paper and conservative in returns) else variant
    out = {"record_type": rt, "meaning": RECORD_TYPES[rt][0], "variant_used": used,
           "cost_basis": "conservative cost level (no real fills)" if used != variant else "baseline cost",
           "published_from": a.published_from or None, "assumed_vendor_trials": a.assumed_vendor_trials}
    vt = summary.get("_vendor_trials_sensitivity", {}).get("by_assumed_trials", {}).get(str(a.assumed_vendor_trials), {})
    out["psr_at_assumed_trials"] = vt.get("probability")
    if rt == "unknown" or not a.published_from:
        out.update(grade="C", reason="no verifiable out-of-sample period (unknown record or no start date)")
        return out
    r = returns[used]
    post = r[r.index >= pd.Timestamp(a.published_from)]
    b = bench.reindex(post.index).fillna(0)
    x = (post - b).values
    if len(x) < 60:
        out.update(grade="C", reason=f"only {len(x)} out-of-sample days")
        return out
    from ml4t.diagnostic.evaluation import compute_paired_uncertainty
    m = compute_min_trl(x, frequency="daily")
    p = _flat(compute_paired_uncertainty(post.values, b.values, rebalance_step=5)).get("p_value")
    out.update(oos_days=int(len(x)), min_trl_days=float(m.min_trl), oos_paired_p=p)
    ok = (len(x) >= m.min_trl and p is not None and p < 0.05 and (out["psr_at_assumed_trials"] or 0) >= 0.95)
    if ok and RECORD_TYPES[rt][1]:
        out.update(grade="A", reason="unfilterable out-of-sample record, long and significant enough")
    elif ok:
        out.update(grade="B", reason="long and significant, but recorded/reported by the vendor - get statements "
                                     "or an independent record")
    else:
        out.update(grade="B", reason="genuine out-of-sample evidence, but shorter than MinTRL or not significant "
                                     "at the assumed vendor trials")
    return out


def warnings(summary: dict, checks: dict) -> list:
    w = []
    for k, v in summary.items():
        main = k.endswith("w-net") and not k.startswith(("cost", "delay")) and "ex-theme" not in k
        if main and isinstance(v, dict) and (v.get("sharpe_ratio") or 0) > 2:
            w.append(f"{k}: daily Sharpe {v['sharpe_ratio']:.2f} > 2 - check fill timing / look-ahead before trusting it")
    if checks["share_picks_without_prices"] > 0.02:
        w.append(f"{checks['share_picks_without_prices']:.1%} of picks have no prices (held as cash) - survivorship risk")
    if checks["missing_weeks"]:
        w.append(f"{len(checks['missing_weeks'])} missing weeks in the list - missing weeks can be removed bad weeks")
    if checks["picked_before_first_price"]:
        w.append("tickers picked before their first price bar: " + ", ".join(checks["picked_before_first_price"][:10]))
    return w


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holds", default="1,2,4,8,12,26", help="holding periods in weeks (test all, do not pre-select)")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="baseline cost per side in bps (commission+slippage)")
    ap.add_argument("--cost-grid", default="10,50", help="extra cost levels per side (bps) for every hold")
    ap.add_argument("--delays", default="1,2,5", help="entry delays in trading days for the 1-week net variant")
    ap.add_argument("--exclude", default="", help="comma-separated tickers for an ex-theme variant")
    ap.add_argument("--benchmark", default="SPY", help="total-return benchmark in the price panel")
    ap.add_argument("--published-from", default="", help="first verifiable publication date of the list (YYYY-MM-DD)")
    ap.add_argument("--delist-map", default="", help="csv asset,return of terminal delisting returns")
    ap.add_argument("--delist-return", type=float, default=0.0, help="terminal return for delisted names not in the map")
    ap.add_argument("--vendor-trials", default="1,10,100,1000", help="grid of assumed numbers of vendor variants")
    ap.add_argument("--assumed-vendor-trials", type=int, default=100,
                    help="vendor variants from the intake (lists/strategies the vendor runs); unknown -> 100")
    ap.add_argument("--record-type", default="unknown", choices=list(RECORD_TYPES),
                    help="how the track record was produced (see RECORD_TYPES)")
    ap.add_argument("--extra-trials", default="", help="names of further variants already looked at (counted for DSR)")
    ap.add_argument("--traded-hold", type=int, default=0,
                    help="holding period (weeks) that was fixed/traded BEFORE the results were seen (e.g. the paper-traded "
                         "rule). It becomes the leader and the only trial: no selection -> DSR = PSR; other holds are "
                         "robustness checks only. 0 = holds were compared and the best is taken (DSR over all holds)")
    ap.add_argument("--min-improvement-pp", type=float, default=None,
                    help="smallest net-CAGR difference (percentage points) that would matter for trading, stated BEFORE "
                         "looking; differences below it count as no improvement")
    ap.add_argument("--plateau-tol", type=float, default=0.25, help="plateau tolerance (share of |best net CAGR|)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    holds = [int(h) for h in a.holds.split(",") if h.strip()]
    grid = [float(c) for c in a.cost_grid.split(",") if c.strip()]
    delays = [int(d) for d in a.delays.split(",") if d.strip()]
    excl = {t.strip().upper() for t in a.exclude.split(",") if t.strip()}

    _, px, cal, weeks, checks = load(a.picks, a.prices, a.benchmark)
    dmap = {}
    if a.delist_map:
        dm = pd.read_csv(a.delist_map)
        dmap = dict(zip(dm.iloc[:, 0].astype(str).str.upper(), dm.iloc[:, 1].astype(float)))
    checks["delist_terminal_returns"] = {t: dmap.get(t, a.delist_return) for t in checks["delisted_in_sample"]}
    px = apply_delisting(px, cal, checks["delisted_in_sample"], dmap, a.delist_return)
    json.dump(checks, open(f"{a.out}/data_checks.json", "w"), indent=2, default=str)

    variants = [(f"{h}w-gross", h, 0.0, set(), 0) for h in holds]
    variants += [(f"{h}w-net", h, a.cost_bps, set(), 0) for h in holds]
    variants += [(f"cost{int(c)}-{h}w-net", h, c, set(), 0) for c in grid for h in holds if c != a.cost_bps]
    variants += [(f"delay{d}-1w-net", 1, a.cost_bps, set(), d) for d in delays]
    if excl:
        variants.append(("1w-ex-theme-gross", 1, 0.0, excl, 0))
        variants += [(f"{h}w-ex-theme-net", h, a.cost_bps, excl, 0) for h in holds]

    bench_px = px.filter(pl.col("asset") == a.benchmark).to_pandas().set_index("timestamp")["close"]
    returns, tos = {}, {}
    rejected = {}
    for name, h, cost, ex, dl in variants:
        REJECTED[0] = 0
        eq, tr = run_variant(px, cal, weeks, h, cost, ex, delay=dl)
        d = f"{a.out}/{name}"
        os.makedirs(d, exist_ok=True)
        eq.to_frame("equity").to_parquet(f"{d}/equity.parquet")
        tr.to_parquet(f"{d}/trades.parquet")
        export_tradeblocks(eq, tr, f"PickList {name}", d)
        returns[name] = eq.pct_change().fillna(0)
        tos[name] = turnover(eq, tr)
        rejected[name] = REJECTED[0]
        print(f"done {name}", flush=True)
    idx = returns[variants[0][0]].index[1:]
    returns = {k: v.reindex(idx).fillna(0) for k, v in returns.items()}
    bench = bench_px.pct_change().reindex(idx).fillna(0)

    base_trials = [f"{a.traded_hold}w-net"] if a.traded_hold else [f"{h}w-net" for h in holds]
    trial_names = base_trials + [t for t in a.extra_trials.split(",") if t.strip() in returns]
    summary = diagnostics(returns, bench, trial_names)
    for k, t in tos.items():
        summary[k]["turnover"] = t
    summary["_holding_period_choice"] = holding_choice(summary, holds, a.cost_bps, len(weeks))
    best = a.traded_hold or summary["_holding_period_choice"].get("best_net_hold_weeks", holds[0])
    summary["_holding_period_choice"]["leader"] = {"hold_weeks": best, "pre_specified": bool(a.traded_hold)}
    pla = plateau_holds(summary, holds, len(weeks), tol=a.plateau_tol, alt_tol=min(1.0, 2 * a.plateau_tol))
    summary["_plateau_holds"] = pla
    grid_best = pla.get("best_hold_weeks")
    centre = pla.get("main", {}).get("centre_hold_weeks")
    if grid_best and centre:
        summary["_paired_leader_vs_centre"] = paired_leader_centre(returns, f"{grid_best}w-net", f"{centre}w-net")
    summary["_day_dependence"] = day_dependence(returns[f"{best}w-net"], bench)
    bcagr = float(np.expm1(np.log1p(bench.values).sum() / (len(bench) / 252.0)))
    summary["_min_improvement"] = min_improvement_check(summary, f"{best}w-net", f"{centre}w-net" if centre else None,
                                                        bcagr, a.min_improvement_pp)
    from ml4t.diagnostic.evaluation.stats import compute_min_trl
    m = compute_min_trl(returns[f"{best}w-net"].values - bench.values, frequency="daily")
    summary["_vendor_trials_sensitivity"] = vendor_trials(returns, bench, f"{best}w-net",
                                                          sorted({int(n) for n in a.vendor_trials.split(",")} | {a.assumed_vendor_trials}))
    summary["_min_trl_best_net_hold"] = {"variant": f"{best}w-net", "min_trl_days": float(m.min_trl),
                                         "years": float(m.min_trl_years), "adequate": bool(m.has_adequate_sample)}
    summary["_cost_grid"] = {f"{h}w": {str(int(c)): summary.get(f"cost{int(c)}-{h}w-net" if c != a.cost_bps else f"{h}w-net", {})
                                       .get("annual_return") for c in sorted(set(grid + [a.cost_bps]))} for h in holds}
    summary["_cost_grid"]["gross"] = {f"{h}w": summary[f"{h}w-gross"].get("annual_return") for h in holds}
    summary["_entry_delay"] = {"0": summary["1w-net"].get("annual_return"),
                               **{str(d): summary[f"delay{d}-1w-net"].get("annual_return") for d in delays}} \
        if "1w-net" in summary else {}
    if a.published_from:
        summary["_publication_split"] = split_stats(returns, bench, [f"{h}w-net" for h in holds], a.published_from)
    cons = f"cost{int(max(grid))}-{best}w-net" if grid and max(grid) > a.cost_bps else None
    summary["_evidence_grade"] = evidence_grade(summary, returns, bench, f"{best}w-net", a, cons)
    summary["_pick_stats"] = pick_stats(px, cal, weeks, set())
    if excl:
        summary["_pick_stats_ex_theme"] = pick_stats(px, cal, weeks, excl)
    summary["_rejected_orders"] = rejected
    summary["_warnings"] = warnings(summary, checks)
    if any(rejected.values()):
        summary["_warnings"].append(f"engine rejected orders in {sum(1 for v in rejected.values() if v)} variants - "
                                    "inspect before trusting the numbers")
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: (v.get("annual_return"), v.get("sharpe_ratio"), v.get("max_drawdown"))
                      for k, v in summary.items() if not k.startswith("_")}, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ["_evidence_grade", "_holding_period_choice", "_plateau_holds", "_paired_leader_vs_centre", "_min_improvement", "_vendor_trials_sensitivity", "_entry_delay",
                                              "_warnings"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
