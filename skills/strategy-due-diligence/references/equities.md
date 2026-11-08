# Equities adapter (stock pick lists -> ML4T -> ml4t-diagnostic -> TradeBlocks)

Read the core [SKILL.md](../SKILL.md) first. This file holds sections 1 and 3-6 and the equity-specific parts of the golden rule, requirements and intake.

## Tools (golden rule, equities)
- Backtest = **ml4t-backtest** engine (via `scripts/picklist_backtest.py`). Never hand-roll return simulations.
- Metrics, bootstrap confidence intervals, benchmark comparison, Deflated/Probabilistic Sharpe, expected max Sharpe, MinTRL, reality check, factor models, signal/IC analysis = **ml4t-diagnostic**. Data = **ml4t-data** / the ML4T MCP data tools.
- Before writing any analysis code, look it up: ML4T MCP `list_library_functions(query=...)`, `describe_function(name)`, `get_recipe()`. Only if nothing fits, write own code and say so in the report.
- Robustness on the realised equity/trade tape (edge decay, rolling/period metrics, walk-forward, Monte Carlo, stress, regime, block comparison, backtest vs. actual) = **TradeBlocks MCP**.
- Own code ONLY for real gaps: parsing the pick file, random-portfolio and baseline tests against a universe, concentration, charts, the report.

## Requirements (both MCP servers must be connected)
- **ML4T MCP server** with `run_python`, `job_start`, `fetch_prices`, `fetch_factors`, `list_library_functions`, `describe_function`, `get_recipe` (ml4t-backtest >= 0.2, ml4t-diagnostic >= 0.1.9, ml4t-data >= 0.3). All code runs through `run_python` (short) or `job_start` (the full backtest, several minutes). Datasets live under `data/prices`, `data/factors`. Survivorship-free US prices (e.g. Sharadar `sharadar_prices` / `sharadar_universe` tools) are strongly preferred if the server offers them.
- **TradeBlocks MCP server** (Docker image `romeo345/tradeblocks-mcp` or `npx tradeblocks-mcp <data-dir>`). The exported CSVs must be copied to a path the TradeBlocks server can read (its data dir). On a [quant-mcp-stack](https://github.com/carstenf/quant-mcp-stack) server both run on one host: write the CSVs from `run_python` to `$TRADEBLOCKS_DATA/<block>/` and pass `/data/<block>/tb_tradelog.csv` (the path inside the TradeBlocks container) to `import_csv`.
- Setup of both servers from scratch: see `SETUP.md` in that repo.
- If a server is missing: stop, say which one, and point to its setup; do not substitute own code.
- Network for downloads (Yahoo Finance, Kenneth French Data Library, GitHub raw). If blocked: say which domain and ask for data files; never invent data.

## Intake additions (equities)
- Benchmarks: S&P 500 total return (adjusted SPY, not the price index ^GSPC) always; add Nasdaq 100 (adjusted QQQ) and the dominant sector/country ETF.
- Costs: baseline 0,2 % per side (commission + slippage) plus a gross run and a cost grid (default 0,1 % / 0,5 %); raise the baseline for small caps, OTC ADRs and low dollar volume.

### How the intake answers drive the evaluation
| Intake answer | Parameter | Effect |
|---|---|---|
| Track-record type | `--record-type` | evidence grade: only `audited` and `paper-independent` can reach A; `self-reported` counts as `paper-vendor`; `unknown` = in-sample only, verdict at most Watch unless the tradability audit (section 8) succeeds |
| First verifiable publication / live start | `--published-from` | in-sample vs. out-of-sample split; MinTRL and the paired benchmark test on the out-of-sample part decide |
| Number of vendor lists/strategies | `--assumed-vendor-trials` | multiple-testing hurdle: PSR of the excess return against the expected max Sharpe of that many zero-skill strategies |
| Rules and list timing | engine timing, `--delays` | execution model; look-ahead check |
| Costs, liquidity, real fills | `--cost-bps`, `--cost-grid` | net results, break-even cost, capacity; without real fills the evidence grade uses the conservative (highest) cost level of the grid |
| Holding period actually traded/recorded | `--traded-hold` | that rule is the only trial: PSR instead of DSR over the holds; other holds = robustness only |
| Benchmark | `--benchmark` | paired test, factor alpha |

## 1. Data
Prices: use the best curated source available (e.g. the user's Sharadar build: point-in-time, delisted names, total-return OHLC) before free sources such as Yahoo. If a share of the picks is missing in that source (often OTC ADRs), ask whether to redistribute their weight or hold cash, and quantify the effect: run the list once with the old source on the full list, once with the old source restricted to the covered names, once with the new source - this splits the change into universe effect and price-source effect (report it). Normalise ticker spelling (BRK.B vs BRK-B) between picks and price panel and check `tickers_without_prices` is empty.
- Pick file -> `picks.csv` with columns `week` (first trading day of the week, YYYY-MM-DD), `ticker` (Yahoo/US format, '.'->'-'), `rank`. Excel or other files received from the user: keep a parquet copy of the raw file next to the run.
- Prices: daily OHLCV panel (timestamp, asset, open, high, low, close, volume), **split- and dividend-adjusted consistently for open and close** (a split-day jump in opens is a data error), incl. benchmark tickers.
  - **Survivorship:** use a source that includes delisted names (e.g. Sharadar SEP for US-listed stocks); Yahoo only for names it lacks (OTC ADRs, foreign lines), and say per ticker which source was used. Yahoo has no delisted names.
  - **Delisting returns:** for every name whose prices end inside the sample, determine the reason (e.g. Sharadar actions: acquisition, bankruptcy, delisted) and build `delist.csv` (asset, terminal return): acquisition = offer premium if known else 0, bankruptcy/liquidation = -1,0, other = 0. Unknown reason -> `--delist-return` (default 0) and add a stress run with -1,0.
- Factors: Fama-French 5 + momentum, daily, via MCP `fetch_factors` (datasets ff5, mom) or ml4t-data `FamaFrenchProvider`. Note the last factor date.
- Universe for the random-portfolio and baseline tests: **point-in-time** membership (e.g. `sharadar_universe` / historical index constituents). Today's constituents from Wikipedia are outcome-dependent (survivors) - only as a last resort, and say so.

## 3. Backtest on the ML4T engine
Use `scripts/picklist_backtest.py` from this skill's folder (if the skill was installed without its folder, fetch it from `https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/skills/strategy-due-diligence/scripts/picklist_backtest.py`), write it and `picks.csv` into the run folder, then start it with `job_start`:
`python picklist_backtest.py --picks picks.csv --prices <panel>.parquet --out out --holds 1,2,4,8,12,26 --cost-bps 20 --cost-grid 10,50 --delays 1,2,5 --benchmark SPY --exclude <theme tickers> --published-from <date> --delist-map delist.csv --record-type <audited|self-reported|paper-independent|paper-vendor|unknown> --assumed-vendor-trials <N> --min-improvement-pp <pp> [--traded-hold <weeks>] [--plateau-tol 0.25]`
- `--min-improvement-pp`: the smallest net-CAGR difference (percentage points) that would matter, written down in the intake BEFORE any result is seen (core section 2, item 8). If it was not stated before looking, leave it out: the summary then records that it was missing - say so in the report, do not invent one afterwards.
- Execution: weights set on the last bar before the pick week, filled NEXT_BAR at OPEN (= Monday open), equal weight over the FULL list, 98 % gross, fractional shares. Names without a price stay in cash (no reweighting to survivors). Names leaving the list are closed with `close_position` (exact quantity; a target weight of 0 can produce rejected "position reversal" sells). `_rejected_orders` must be 0 for every variant, otherwise the tape is not what was intended - check before using any number. Delisted names get the terminal return on the bar after their last price, then stay flat until sold.
- Holding H weeks = H staggered buy-and-hold tranches (1/H capital each), summed. All tranches start invested in week 0 (ramp-up), so for long holds the first H weeks over-weight the first lists - mention it for H >= 12.
- Outputs per variant: equity.parquet, trades.parquet, TradeBlocks `tb_tradelog.csv` + `tb_dailylog.csv`; `summary.json` with
  - per variant: PortfolioAnalysis stats (CAGR, vol, Sharpe, Sortino, Calmar, MDD, alpha/beta, IR, up/down capture), annualised turnover, `uncertainty` (block-bootstrap 95 % CIs + PSR p-value), `vs_benchmark` (paired bootstrap strategy minus benchmark: Sharpe, CAGR and MDD difference, information ratio, each with CI; p-value; probability the strategy wins),
  - evidence: `_evidence_grade`; cohort: `_selection_adjustment`, `_dsr_excess_vs_benchmark` (DSR on excess returns over the net hold variants), `_whites_reality_check`, `_min_trl_best_net_hold`, `_vendor_trials_sensitivity`,
  - `_holding_period_choice`, `_cost_grid` (net CAGR per hold and cost level, gross row), `_entry_delay`, `_publication_split`,
  - `_plateau_holds` (core section 2, item 7, on the hold grid by net CAGR: regions, centre, alternative tolerance, `rule_dependent`, flags such as boundary leader, one- or two-hold region, cliff), `_paired_leader_vs_centre` (paired block bootstrap grid-best minus plateau-centre hold, ml4t-diagnostic `compute_paired_uncertainty`; not adjusted for selection), `_day_dependence` (leader: share of the log return from the best 1/5/10 days, result without them, per-year returns vs. benchmark; core item 9), `_min_improvement` (differences vs. the stated minimum),
  - `_pick_stats` (+ `_pick_stats_ex_theme`): per-pick win rate, avg win/loss, payoff ratio, share of a 3:1 target, top-10 contributors and their share, liquidity, capacity estimate,
  - `_warnings` (daily Sharpe > 2 -> check fill timing first; picks without prices; gaps; picks before first price).
- Exit-rule tests (stop loss, take profit, trailing stop) use the engine's `StopLoss`, `TakeProfit`, `TrailingStop` rules, not hand-made price checks.
- Count variants honestly. **Was the rule fixed before the results were seen?** Ask in the intake which holding period was actually traded/recorded. If one was (e.g. a colleague paper-traded the 1-week rotation), pass `--traded-hold <weeks>`: it is the leader and the only trial, so the DSR collapses to the PSR (no selection to deflate) and the other holds are robustness checks only - say so in the report and do not show a DSR. Only if holds were compared and the best is taken (no traded rule), the DSR counts all net hold variants. Pass any further variant you looked at with `--extra-trials`. Cost-grid, delay and ex-theme runs are sensitivity checks, not candidates. The unknown number of vendor variants (`_vendor_trials_sensitivity`) still applies either way.
- Holding-period choice (no preferred horizon): always run all holds. Recommend the hold with the highest NET CAGR at the assumed costs among reliable holds (>= 4 full holding periods). For each longer hold report the break-even cost per side vs. the shortest hold and the cost at which net CAGR reaches zero. Recommend a longer hold only if it is within ~1 pp net or costs above break-even are realistic, and say so. If the best net hold is flagged by `_plateau_holds` (spike, boundary, cliff) or `_paired_leader_vs_centre` does not separate it from the centre, prefer the plateau centre and report both.

## 4. Diagnostics with ml4t-diagnostic
- Factor model: `compute_factor_model(returns, FactorData(...), hac=True)` on daily returns AND on weekly (Monday-to-Monday) returns. Foreign/ADR-heavy lists have asynchronous closes: report both; only claim a style tilt that is significant in both.
- Signal ranking: `analyze_signal` (rank IC, quantiles) with the rank as signal, or rank-IC per week.
- Optional: `compute_pbo` when many parameter variants were scanned; `compute_return_attribution` / `compute_rolling_exposures`; `WalkForwardCV` / `CombinatorialCV` when parameters are fitted.
- Look-ahead: check publication timing (intake, section 2). If you compute a signal yourself, run `audit_lookahead` on the signal function.

## 5. Robustness with TradeBlocks
`import_csv` (tradelog + dailyLogPath, plBasis net_includes_fees) per variant, then:
- `get_statistics` (marked-curve Sharpe/Sortino/Calmar with T-bill rf, CAGR, MDD, time in drawdown).
- `analyze_edge_decay` (period trends with p-values, rolling metrics, walk-forward efficiency). Use its quarterly trend p-values and walk-forward IS/OOS Sharpe; ignore the Monte-Carlo regimeComparison absolute values (margin-based for stock tapes).
- `run_monte_carlo` with `resampleMethod: marked-daily` (stationary block) for 1-year return and drawdown quantiles.
- `get_period_returns` / `get_performance_charts` for monthly tables; `stress_test` / `analyze_regime_performance` if the period covers stress or VIX data is available; `compare_blocks` for variants; `compare_backtest_to_actual` / `analyze_discrepancies` against vendor returns (section 2).
- TradeBlocks "trades" are engine round trips (names held across weeks are merged); per-pick win rate and payoff come from `_pick_stats` (its share of a 3:1 target is reported only for Minervini-style trade strategies).

## 6. Gaps filled with own code
- **Baseline (Jansen ch. 16):** equal-weight portfolio of the point-in-time universe with the same Monday-open protocol and costs. Separates "the universe did well" from selection skill; report strategy minus baseline.
- **Random-portfolio test:** 10,000 runs, each week N random names from the same point-in-time universe (plain and sector-matched), same timing; p = share of runs with CAGR >= strategy; also per half, post-publication only, and ex-theme.
- **Concentration:** contribution of the dominant theme and top tickers to excess return (start from `_pick_stats`), Herfindahl index and effective number of contributors, sector/country shares; trimming top/bottom 1/5/10 % of pick returns at weekly-portfolio level.
- **Costs and capacity (Jansen ch. 16/18):** cost grid from `_cost_grid` (conservative / baseline / optimistic); safety margin = gross excess return / round-trip cost per turnover (>= 2,5 robust, < 1,5 likely unprofitable); for small caps use a square-root impact estimate instead of a flat rate; capacity from `_pick_stats.capacity_aum_usd` (AUM at 1 % / 5 % of median 20-day dollar volume); share of names < 5 M / < 1 M USD, OTC ADRs.
