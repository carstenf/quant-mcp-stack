"""Chart tools for the ml4t MCP server (house style /opt/skills/_shared/plotstyle.py, PNG returned as image)."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import numpy as np
import polars as pl
from mcp.server.fastmcp import Image

STYLE_DIR = os.environ.get("ML4T_MCP_STYLE_DIR", "/opt/skills/_shared")  # plotstyle.py (house chart style)


def _ps():
    if STYLE_DIR not in sys.path:
        sys.path.insert(0, STYLE_DIR)
    import plotstyle as ps  # noqa: PLC0415

    ps.reset_colors()
    return ps


def _de(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _stats(v: np.ndarray, days: int) -> dict:
    r = np.diff(v) / v[:-1]
    years = max(days / 365.25, 1e-9)
    sd = r.std()
    return {"cagr": (v[-1] / v[0]) ** (1 / years) - 1,
            "vol": sd * math.sqrt(252),
            "sharpe": (r.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan"),
            "max_dd": float((v / np.maximum.accumulate(v) - 1).min()),
            "end_value": float(v[-1])}


def _log_ticks(lo: float, hi: float) -> list[float]:
    out = []
    for e in range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1):
        for m in (1, 2, 5):
            t = m * 10 ** e
            if lo <= t <= hi:
                out.append(t)
    if len(out) < 3:  # narrow range: evenly spaced in log
        out = list(np.geomspace(lo, hi, 5))
    return out


def _money(v: float, _=None) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.3g} Mio".replace(".", ",")
    if v >= 1e3:
        return f"{v / 1e3:.3g}k".replace(".", ",")
    return f"{v:.3g}".replace(".", ",")


def register(mcp, *, RUNS_DIR: Path, _load_panel: Callable, _safe_name: Callable, _rel: Callable,
             _jsonable: Callable, _err: Callable) -> None:

    def _finish_png(ps, fig, ax, out: Path, title: str, sub: str, src: str, keys=None) -> Image:
        out.parent.mkdir(parents=True, exist_ok=True)
        ps.finish(fig, ax, title, subtitle=sub, source=src, out=str(out), keys=keys)
        return Image(data=out.read_bytes(), format="png")

    @mcp.tool()
    def plot_equity(run_ids: list[str], labels: list[str] | None = None, benchmark: str | None = None,
                    benchmark_asset: str | None = None, benchmark_label: str | None = None, log: bool = True,
                    title: str | None = None, mode: str = "dark") -> Any:
        """Equity curve(s) of one or more backtest runs (max 5), optionally vs. a benchmark, as PNG image.

        All series start at the first run's initial capital on the first common day. benchmark: a prices
        dataset (e.g. 'spy' from fetch_prices), benchmark_asset picks the ticker if the dataset has several.
        log=true for a logarithmic y-axis. Default title states the CAGR takeaway; subtitle MaxDD + Sharpe.
        House style (dark by default, mode='light' on request). Saved as runs/<first run>/equity.png.
        Returns the image and the statistics per series.
        """
        try:
            if not run_ids or len(run_ids) > 5:
                raise ValueError("1..5 run_ids")
            series: list[tuple[str, pl.DataFrame]] = []
            costs = []
            for i, rid in enumerate(run_ids):
                rd = RUNS_DIR / _safe_name(rid)
                eq = pl.read_parquet(rd / "equity.parquet").select("timestamp", pl.col("equity").alias("v"))
                lab = (labels[i] if labels and i < len(labels) else rid.split("-", 2)[-1])
                series.append((lab, eq))
                pp = rd / "params.json"
                if pp.exists():
                    p = json.loads(pp.read_text())
                    costs.append(p.get("preset"))
            if benchmark:
                b = _load_panel(benchmark, "prices")
                asset = benchmark_asset or b["asset"][0]
                b = b.filter(pl.col("asset") == asset).select("timestamp", pl.col("close").alias("v"))
                series.append((benchmark_label or asset, b))
            # common dates: intersection of all series
            days = series[0][1].select("timestamp")
            for _, df in series[1:]:
                days = days.join(df.select("timestamp"), on="timestamp", how="inner")
            days = days.sort("timestamp")
            if days.height < 2:
                raise ValueError("series do not overlap in time")
            cap0 = float(series[0][1].sort("timestamp")["v"][0])
            ps = _ps()
            ps.apply(mode)
            fig, ax = ps.figure("portrait")
            stats, keys, allv = {}, [], []
            d0, d1 = days["timestamp"][0], days["timestamp"][-1]
            for lab, df in series:
                s = days.join(df, on="timestamp", how="left").sort("timestamp").with_columns(pl.col("v").fill_null(strategy="forward"))
                v = s["v"].to_numpy().astype(float)
                v = v / v[0] * cap0
                stats[lab] = _stats(v, (d1 - d0).days)
                c = ps.color(lab)
                ax.plot(s["timestamp"].to_list(), v, color=c)
                keys.append((lab, c))
                allv.append(v)
            lo, hi = float(min(x.min() for x in allv)), float(max(x.max() for x in allv))
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(FixedLocator(_log_ticks(lo * 0.95, hi * 1.05)))
                ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(_money))
            ps.ylabel(ax, "Depotwert, USD" + (" (log)" if log else ""))
            labs = [k for k, _ in series]
            s0 = stats[labs[0]]
            if title is None:
                if len(labs) == 1:
                    title = f"{labs[0]}: {_de(s0['cagr'] * 100)} % p.a."
                else:
                    title = f"{labs[0]}: {_de(s0['cagr'] * 100)} % p.a. statt {_de(stats[labs[1]]['cagr'] * 100)} %"
            if len(title) > 42:
                title = title[:41].rstrip() + "…"
            dd = " vs. ".join(f"{stats[l]['max_dd'] * 100:.0f} %" for l in labs[:3])
            sh = " vs. ".join(_de(stats[l]["sharpe"], 2) for l in labs[:3])
            sub = f"Start {_money(cap0)} $, nach Kosten\nMaxDD {dd}, Sharpe {sh}"
            src = f"ml4t-backtest, Preset {'/'.join(sorted(set(c for c in costs if c))) or '-'}\n{d0.date()} bis {d1.date()}"
            if benchmark:
                src = f"Benchmark {labs[-1]} aus {benchmark}; " + src
            out = RUNS_DIR / _safe_name(run_ids[0]) / "equity.png"
            img = _finish_png(ps, fig, ax, out, title, sub, src, keys)
            info = _jsonable({"ok": True, "file": _rel(out), "stats": stats, "period": [d0, d1]})
            return [img, json.dumps(info)]
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def plot_signal_spreads(run_ids: list[str], labels: list[str] | None = None, period: str | None = None,
                            title: str | None = None, mode: str = "dark") -> Any:
        """Compare factor analyses (run ids from analyze_signal): horizontal bars of the top-bottom quantile
        spread, annualized (period return × 252/period), blue = positive, red = negative, sorted; IC in the label.

        period: forward period to show, e.g. '21D' (default: longest one analysed). Saved as
        runs/<first run>/spreads.png.
        """
        try:
            if not run_ids or len(run_ids) > 10:
                raise ValueError("1..10 run_ids")
            rows = []
            meta = None
            for i, rid in enumerate(run_ids):
                rd = RUNS_DIR / _safe_name(rid)
                d = json.loads((rd / "signal.json").read_text())
                per = period or max(d.get("spread", {}), key=lambda k: int(re.sub(r"\D", "", k) or 0))
                n = int(re.sub(r"\D", "", per) or 1)
                sp = d["spread"][per] * 252 / n * 100
                ic = d["ic"][per]
                lab = labels[i] if labels and i < len(labels) else d.get("signal_col", rid)
                rows.append((lab, sp, ic))
                meta = meta or (per, d.get("date_range"), d.get("universe"), d.get("quantiles"))
            rows.sort(key=lambda r: r[1])
            ps = _ps()
            ps.apply(mode)
            pos, mid, neg = ps.diverging()
            fig, ax = ps.figure("portrait")
            y = list(range(len(rows)))
            vals = [r[1] for r in rows]
            lim = max(1.0, max(abs(v) for v in vals) * 1.35)
            ax.barh(y, vals, color=[pos if v > 0 else neg for v in vals], height=0.45)
            ax.axvline(0, color=mid, lw=1)
            ax.set_ylim(-0.5, len(rows) - 0.1)
            ax.set_xlim(-lim, lim)
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.xaxis.grid(True)
            ax.yaxis.grid(False)
            for yi, (lab, v, ic) in zip(y, rows):
                ax.text(-lim * 0.97, yi + 0.34, f"{lab}  (IC {_de(ic, 3)})", va="bottom", ha="left",
                        color=ps.ink(), fontsize=14)
                ax.text(v + (lim * 0.02 if v > 0 else -lim * 0.02), yi, ("+" if v > 0 else "") + f"{_de(v)} %",
                        va="center", ha="left" if v > 0 else "right", color=ps.ink(), fontsize=14)
            per, dr, univ, q = meta
            best = [r[0] for r in reversed(rows) if r[1] > 0][:2]
            if title is None:
                t2 = " und ".join(best) + " wirken"
                title = (t2 if len(t2) <= 42 else f"{best[0]} wirkt am stärksten") if best else "Kein Faktor wirkt"
            if len(title) > 42:
                title = title[:41].rstrip() + "…"
            qn = q or 5
            sub = f"Spread Q{qn} − Q1 in % p.a. (aus {per}-Renditen)\nUniversum: {univ or 'gesamtes Signal-Panel'}"
            src = f"ml4t-diagnostic analyze_signal, ohne Kosten\n{dr[0][:10] if dr else ''} bis {dr[1][:10] if dr else ''}"
            out = RUNS_DIR / _safe_name(run_ids[0]) / "spreads.png"
            img = _finish_png(ps, fig, ax, out, title, sub, src)
            info = _jsonable({"ok": True, "file": _rel(out), "period": per,
                              "rows": [{"label": a, "spread_pct_pa": b, "ic": c} for a, b, c in rows]})
            return [img, json.dumps(info)]
        except Exception as e:
            return _err(e)

    @mcp.tool()
    def plot_quantile_returns(run_id: str, periods: list[str] | None = None, demean: bool = True,
                              label: str | None = None, title: str | None = None, mode: str = "dark") -> Any:
        """Alphalens-style quantile / decile bar chart of one analyze_signal run (use quantiles=10 there for
        deciles), as PNG image.

        One panel per forward period (default: all analysed, max 3), values annualized in % p.a. (period
        return × 252/period). demean=true (default) shows each quantile relative to the quantile average (blue
        = above, red = below); demean=false shows raw mean returns. Saved as runs/<run_id>/quantiles.png.
        """
        try:
            rd = RUNS_DIR / _safe_name(run_id)
            d = json.loads((rd / "signal.json").read_text())
            qr = d["quantile_returns"]
            pers = periods or sorted(qr, key=lambda k: int(re.sub(r"\D", "", k) or 0))
            pers = [p for p in pers if p in qr][:3]
            if not pers:
                raise ValueError(f"periods not in run (available: {list(qr)})")
            ps = _ps()
            ps.apply(mode)
            pos, mid, neg = ps.diverging()
            fig, axs = ps.figure("portrait", nrows=len(pers), sharex=False)
            axs = list(axs) if hasattr(axs, "__len__") else [axs]
            period_stats = {}
            for ax, per in zip(axs, pers):
                n = int(re.sub(r"\D", "", per) or 1)
                ks = sorted(qr[per], key=int)
                vals = np.array([qr[per][k] for k in ks]) * 252 / n * 100
                if demean:
                    vals = vals - vals.mean()
                    cols = [pos if v >= 0 else neg for v in vals]
                else:
                    cols = ps.color("Rendite")
                x = [int(k) for k in ks]
                ax.bar(x, vals, color=cols, width=0.7)
                ax.axhline(0, color=mid, lw=1)
                ax.set_xticks(x)
                ps.ylabel(ax, f"{per}, % p.a." + (" vs. Ø" if demean else ""))
                ax.xaxis.grid(False)
                period_stats[per] = [(int(k), round(float(v), 3)) for k, v in zip(ks, vals)]
            name = label or d.get("signal_col", run_id)
            qname = "Dezile" if len(ks) == 10 else ("Quintile" if len(ks) == 5 else f"{len(ks)} Quantile")
            if title is None:
                v = [b for _, b in period_stats[pers[-1]]]
                k = max(1, len(v) // 5)
                lo_, hi_, mid_ = np.mean(v[:k]), np.mean(v[-k:]), np.mean(v[k:-k]) if len(v) > 2 * k else 0.0
                if mid_ > max(lo_, hi_):
                    pat = "Mitte am besten"
                elif hi_ > lo_:
                    pat = "hohe Werte gewinnen"
                else:
                    pat = "niedrige Werte gewinnen"
                title = f"{name}: {pat}"
                if len(title) > 42:
                    title = f"{qname} {name}"[:42]
            dr = d.get("date_range") or ["", ""]
            sub = (f"{qname} je Tag ({len(ks)} = höchster Signalwert), Ø Forward-Rendite p.a.\n"
                   f"Universum: {d.get('universe') or 'gesamtes Panel'}, {d.get('n_assets')} Titel")
            src = f"ml4t-diagnostic analyze_signal, ohne Kosten\n{dr[0][:10]} bis {dr[1][:10]}"
            out = rd / "quantiles.png"
            img = _finish_png(ps, fig, axs if len(axs) > 1 else axs[0], out, title, sub, src)
            info = _jsonable({"ok": True, "file": _rel(out), "demean": demean, "pct_pa": period_stats})
            return [img, json.dumps(info)]
        except Exception as e:
            return _err(e)


import re  # noqa: E402  (used inside plot_signal_spreads)
