"""Tested code templates for run_python. They replace the former wrapper tools (compute_signal, analyze_signal,
run_backtest, deflated_sharpe): use the ML4T libraries directly, change what you need, keep the conventions."""

RECIPES: dict[str, dict[str, str]] = {}

RECIPES["rank_backtest"] = {
    "description": "Cross-sectional rank backtest on ml4t-backtest: every k bars hold the top N by a signal, equal "
                   "weight, explicit costs, next-bar-open fills, delisted names exit at last price; stored as a run.",
    "code": '''
import polars as pl
from ml4t.backtest import BacktestConfig, RebalanceConfig, Strategy, TargetWeightExecutor, run_backtest
from ml4t.backtest.config import CommissionType, MissingPricePolicy, ShareType, SlippageType
from mcp_helpers import load_panel, pad_delisted, save_backtest_run

PRICES, SIGNALS, COL = "prices/test-largecap40", "signals/test_mom_12_1", "test_mom_12_1"
TOP_N, EVERY, GROSS, COST_BPS_PER_SIDE = 10, 21, 0.98, 20      # costs: half commission, half slippage

px = pad_delisted(load_panel(PRICES, "prices"))
sig = load_panel(SIGNALS, "signals").select("timestamp", "asset", COL)
px = px.filter(pl.col("asset").is_in(sig["asset"].unique().to_list()))

cfg = BacktestConfig()                 # default: NEXT_BAR execution at OPEN, no costs -> set them explicitly
cfg.missing_price_policy = MissingPricePolicy.USE_LAST
cfg.initial_cash = 100_000.0
cfg.share_type = ShareType.FRACTIONAL
cfg.commission_type, cfg.commission_rate = CommissionType.PERCENTAGE, COST_BPS_PER_SIDE / 2 / 1e4
cfg.slippage_type, cfg.slippage_rate = SlippageType.PERCENTAGE, COST_BPS_PER_SIDE / 2 / 1e4
# NOTE: BacktestConfig.from_preset(...) brings its own costs; changing fields afterwards may not override them.

class TopN(Strategy):
    def __init__(self):
        self.ex = TargetWeightExecutor(RebalanceConfig(allow_short=False, max_single_weight=1.0, min_weight_change=0.0))
        self.bar = 0
    def on_data(self, timestamp, data, context, broker):
        self.bar += 1
        if (self.bar - 1) % EVERY:
            return
        # signal at bar t uses information up to t's close; the order fills at t+1 open (point-in-time safe)
        s = {a: b["signals"][COL] for a, b in data.items() if (b.get("signals") or {}).get(COL) is not None}
        top = sorted(s, key=s.get, reverse=True)[:TOP_N]
        w = {a: GROSS / len(top) for a in top} if top else {}
        held = broker.get_positions()
        for a in (list(held) if isinstance(held, dict) else [getattr(p, "asset", p) for p in held]):
            if a not in w:
                if a in data:
                    w[a] = 0.0
                else:
                    broker.close_position(a)
        self.ex.execute(w, data, broker, timestamp=timestamp)

res = run_backtest(px, TopN(), signals=sig, config=cfg)
run_id = save_backtest_run(res, f"top{TOP_N}-{COL}", params=dict(prices=PRICES, signals=SIGNALS, top_n=TOP_N,
                           every=EVERY, cost_bps_per_side=COST_BPS_PER_SIDE), config=cfg)
print({k: res.get(k) for k in ("cagr", "sharpe", "max_drawdown_pct", "num_trades", "win_rate", "payoff_ratio")})
'''}

RECIPES["momentum_signal"] = {
    "description": "Point-in-time price signals with polars (12-1 momentum, short-term reversal, realized vol) saved "
                   "to data/signals/<name>.parquet.",
    "code": '''
import math, os
import polars as pl
from mcp_helpers import load_panel

px = load_panel("prices/test-largecap40", "prices").select("timestamp", "asset", "close")
lr = pl.col("close").log().diff().over("asset")
sig = (px.with_columns(lr.alias("_lr"))
         .with_columns(
             ((pl.col("close").shift(21) / pl.col("close").shift(252)) - 1).over("asset").alias("mom_12_1"),
             (-(pl.col("close") / pl.col("close").shift(5) - 1)).over("asset").alias("rev_5"),
             (pl.col("_lr").rolling_std(63).over("asset") * math.sqrt(252)).alias("vol"))
         .select("timestamp", "asset", "mom_12_1", "rev_5", "vol")
         .drop_nulls())
out = os.environ["ML4T_DATA"] + "/signals/demo_mom.parquet"
sig.write_parquet(out); print(out, sig.shape)
'''}

RECIPES["analyze_signal"] = {
    "description": "Alphalens-style IC / quantile analysis with ml4t.diagnostic.analyze_signal; stored as a signal run "
                   "for plot_signal_spreads / plot_quantile_returns.",
    "code": '''
import polars as pl
from ml4t.diagnostic import analyze_signal
from mcp_helpers import load_panel, save_signal_run

COL = "test_mom_12_1"
sig = load_panel("signals/test_mom_12_1", "signals")
px = load_panel("prices/test-largecap40", "prices")
factor = sig.select(pl.col("timestamp").dt.date().alias("date"), "asset", pl.col(COL).cast(pl.Float64).alias("factor")).drop_nulls()
prices = px.select(pl.col("timestamp").dt.date().alias("date"), "asset", pl.col("close").alias("price"))
res = analyze_signal(factor, prices, periods=(1, 5, 21), quantiles=5, ic_method="spearman", min_assets=10, filter_zscore=3.0)
print(res.summary())
save_signal_run(res, COL, meta={"signals": "signals/test_mom_12_1", "prices": "prices/test-largecap40",
                                "quantiles": 5, "periods": [1, 5, 21]})
'''}

RECIPES["deflated_sharpe"] = {
    "description": "Multiple-testing checks over ALL tried runs: Deflated Sharpe, MinTRL, White's reality check (vs "
                   "a benchmark), PBO if you have a matrix of variants.",
    "code": '''
import os
import numpy as np, polars as pl
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio, compute_min_trl, whites_reality_check

R = os.environ["ML4T_RUNS"]
RUN_IDS = sorted((r for r in os.listdir(R) if os.path.exists(f"{R}/{r}/equity.parquet")),
                 key=lambda r: os.path.getmtime(f"{R}/{r}/equity.parquet"))[-3:]   # put EVERY variant you tried here
rets = [pl.read_parquet(f"{os.environ['ML4T_RUNS']}/{r}/equity.parquet")["return"].fill_null(0).to_numpy()[1:]
        for r in RUN_IDS if os.path.exists(f"{os.environ['ML4T_RUNS']}/{r}/equity.parquet")]
n = min(len(r) for r in rets); rets = [r[-n:] for r in rets]
d = deflated_sharpe_ratio(rets, frequency="daily")
print("DSR probability", round(d.probability, 3), "p", round(d.p_value, 4), "trials", d.n_trials)
m = compute_min_trl(rets[0], frequency="daily")
print("MinTRL years", round(m.min_trl_years, 2), "adequate", m.has_adequate_sample)
bench = np.zeros(n)                                                   # or the benchmark's daily returns
w = whites_reality_check(bench, np.column_stack(rets), bootstrap_samples=2000)
print("White RC p", w["p_value"] if isinstance(w, dict) else w.p_value)
'''}

RECIPES["factor_model"] = {
    "description": "Fama-French 5 + momentum regression with HAC (Newey-West) errors via compute_factor_model on a "
                   "run's daily returns (factors from fetch_factors).",
    "code": '''
import os
import polars as pl
from ml4t.diagnostic.evaluation import FactorData, compute_factor_model

R = os.environ["ML4T_RUNS"]
RUN = max((r for r in os.listdir(R) if os.path.exists(f"{R}/{r}/equity.parquet")),
          key=lambda r: os.path.getmtime(f"{R}/{r}/equity.parquet"))   # newest run - or put your run_id
eq = pl.read_parquet(f"{os.environ['ML4T_RUNS']}/{RUN}/equity.parquet").select(
    pl.col("timestamp").cast(pl.Date).alias("timestamp"), pl.col("return").fill_null(0))
ff = pl.read_parquet(os.environ["ML4T_DATA"] + "/factors/ff5_daily.parquet")
mom = pl.read_parquet(os.environ["ML4T_DATA"] + "/factors/mom_daily.parquet")
f = ff.join(mom, on="timestamp").with_columns(pl.col("timestamp").cast(pl.Date))
df = eq.join(f, on="timestamp").sort("timestamp")   # factors only cover the dates fetched via fetch_factors
print("observations", df.height, df["timestamp"].min(), df["timestamp"].max())
names = [c for c in f.columns if c not in ("timestamp", "RF")]
fd = FactorData(returns=df.select(["timestamp"] + names), rf_rate=df["RF"], factor_names=names)  # rf as polars Series
r = compute_factor_model(df["return"].to_numpy(), fd, hac=True)   # HAC = Newey-West standard errors
print(r.summary() if hasattr(r, "summary") else r)
'''}

RECIPES["portfolio_stats"] = {
    "description": "Performance summary vs a benchmark with ml4t.diagnostic PortfolioAnalysis (CAGR, vol, Sharpe, "
                   "Sortino, Calmar, MDD, alpha/beta, IR, capture ratios).",
    "code": '''
import os
import polars as pl
from ml4t.diagnostic.evaluation import PortfolioAnalysis

R = os.environ["ML4T_RUNS"]
RUN = max((r for r in os.listdir(R) if os.path.exists(f"{R}/{r}/equity.parquet")),
          key=lambda r: os.path.getmtime(f"{R}/{r}/equity.parquet"))   # newest run - or put your run_id
eq = pl.read_parquet(f"{os.environ['ML4T_RUNS']}/{RUN}/equity.parquet").select("timestamp", "return")
spy = (pl.read_parquet(os.environ["ML4T_DATA"] + "/prices/spy.parquet").sort("timestamp")
         .select("timestamp", pl.col("close").pct_change().alias("bench")))
df = eq.join(spy, on="timestamp").fill_null(0)
s = PortfolioAnalysis(df["return"].to_numpy(), benchmark=df["bench"].to_numpy(), periods_per_year=252).compute_summary_stats()
print(s)
'''}
