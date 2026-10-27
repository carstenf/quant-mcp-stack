# Skills

## strategy-due-diligence

Institutional-style due diligence of a stock-picking strategy from a weekly pick/trade list: backtest on the
ML4T engine, statistics with ml4t-diagnostic (bootstrap CIs, paired benchmark test, Deflated Sharpe, factor
model), robustness with TradeBlocks (edge decay, walk-forward, Monte Carlo), PDF report.

Requires the **ML4T** and **TradeBlocks** connectors from this repo (see `../SETUP.md`).

### Install in claude.ai (works on a phone)

1. Download [`strategy-due-diligence.zip`](https://raw.githubusercontent.com/carstenf/quant-mcp-stack/main/skills/strategy-due-diligence.zip)
   and save it to Files. Do not unzip it.
2. claude.ai -> Settings -> Capabilities -> Skills (claude.ai/customize/skills) -> **+** -> Upload a skill -> pick the zip.
3. In a chat with both connectors enabled, attach a pick list (columns week, ticker, rank) and ask for a due diligence.

Updating: delete the old skill, upload the new zip. The folder `strategy-due-diligence/` is the same content unzipped.
