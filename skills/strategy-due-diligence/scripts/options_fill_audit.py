#!/usr/bin/env python3
"""Fill audit for an options trade log (Option Omega export format) against cached minute NBBO quotes.

Reads quotes read-only from a TradeBlocks market database (table option_quote_minutes: underlying, date,
ticker (OCC), time HH:MM, bid, ask, mid) - no market-data provider call, so it works when the provider
plan does not allow single-contract queries. Sample = N random trades (fixed seed) + every trade on the
K worst days by total P/L.

Per trade: entry credit/debit reported vs. mid and natural (sell at bid, buy at ask) at the entry minute;
exit cost reported vs. mid and natural at the exit minute (skipped for expired trades). Flags:
  entry_better_than_mid   reported entry better than mid by more than --tol per spread
  exit_outside_natural    reported exit cost worse than the natural price (+tol) - possible bad quote/fill
  missing_quotes          a leg has no usable quote at that minute

Usage: options_fill_audit.py --trades tradelog.csv --db market.duckdb --out audit/ [--root SPX] [--n 30]
       [--worst-days 10] [--seed 42] [--tol 0.05]
"""
from __future__ import annotations
import argparse, json, os, re
import datetime as dt
import duckdb, numpy as np, pandas as pd

MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
LEG = re.compile(r"(\d+)\s+(\w{3})\s+(\d+)\s+([\d.]+)\s+([CP])\s+(STO|BTO|STC|BTC)\s+([\d.]+)")


def parse_legs(s: str, opened: dt.date):
    out = []
    for q, mon, day, strike, cp, side, px in LEG.findall(s):
        m = MON[mon]
        y = opened.year + (1 if m < opened.month else 0)
        out.append({"qty": int(q), "exp": dt.date(y, m, int(day)), "strike": float(strike), "cp": cp,
                    "sell": side in ("STO", "STC"), "px": float(px)})
    return out


def occ(root, exp, cp, strike):
    return f"{root}{exp:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def quote(con, root, d, tick, hhmm):
    r = con.execute("select bid, ask from option_quote_minutes where underlying=? and date=? and ticker=? and time=?",
                    [root, d, tick, hhmm]).fetchone()
    if not r or r[0] is None or r[1] is None or r[1] <= 0 or r[1] < r[0]:
        return None
    return float(r[0]), float(r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True); ap.add_argument("--db", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="SPX"); ap.add_argument("--n", type=int, default=30); ap.add_argument("--worst-days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.trades)
    t["opened"] = pd.to_datetime(t["Date Opened"]).dt.date
    worst = t.groupby("opened")["P/L"].sum().nsmallest(a.worst_days).index
    rnd = t.sample(n=min(a.n, len(t)), random_state=a.seed)
    sample = pd.concat([rnd.assign(why="random"), t[t["opened"].isin(worst)].assign(why="worst_day")]).drop_duplicates(
        subset=["Date Opened", "Time Opened", "Legs"])
    con = duckdb.connect(a.db, read_only=True)
    rows = []
    for _, r in sample.iterrows():
        legs = parse_legs(r["Legs"], r["opened"])
        if not legs:
            continue
        sign = lambda L: 1 if L["sell"] else -1
        rep_entry = sum(sign(L) * L["px"] * L["qty"] for L in legs)            # credit > 0
        ent_min = str(r["Time Opened"])[:5]
        qs = [quote(con, a.root, r["opened"], occ(a.root, L["exp"], L["cp"], L["strike"]), ent_min) for L in legs]
        row = {"date": str(r["opened"]), "time_opened": ent_min, "why": r["why"], "legs": r["Legs"], "pl": float(r["P/L"]),
               "reason": r["Reason For Close"], "reported_entry": rep_entry}
        if any(q is None for q in qs):
            row["missing_quotes"] = True
        else:
            mid = sum(sign(L) * (q[0] + q[1]) / 2 * L["qty"] for L, q in zip(legs, qs))
            nat = sum((q[0] if L["sell"] else -q[1]) * L["qty"] for L, q in zip(legs, qs))
            row.update(entry_mid=mid, entry_natural=nat, entry_minus_mid=rep_entry - mid,
                       entry_better_than_mid=rep_entry - mid > a.tol)
        if str(r["Reason For Close"]).lower() != "expired" and pd.notna(r.get("Date Closed")):
            cd = pd.to_datetime(r["Date Closed"]).date(); cm = str(r["Time Closed"])[:5]
            qx = [quote(con, a.root, cd, occ(a.root, L["exp"], L["cp"], L["strike"]), cm) for L in legs]
            rep_exit = float(r["Avg. Closing Cost"]) / 100.0 / max(1, int(r["No. of Contracts"]))  # debit per spread, per share
            row["reported_exit"] = rep_exit
            if any(q is None for q in qx):
                row["missing_exit_quotes"] = True
            else:
                xmid = sum(sign(L) * (q[0] + q[1]) / 2 * L["qty"] for L, q in zip(legs, qx))
                xnat = sum((q[1] if L["sell"] else -q[0]) * L["qty"] for L, q in zip(legs, qx))
                row.update(exit_mid=xmid, exit_natural=xnat, exit_minus_natural=rep_exit - xnat,
                           exit_outside_natural=rep_exit - xnat > a.tol)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}/fill_audit_rows.csv", index=False)
    def share(c): return float(df[c].mean()) if c in df and df[c].notna().any() else None
    summ = {"trades_audited": len(df), "random": int((df.why == "random").sum()), "worst_day": int((df.why == "worst_day").sum()),
            "entry_quotes_found": int(df.get("entry_mid", pd.Series(dtype=float)).notna().sum()),
            "mean_entry_minus_mid_per_share": share("entry_minus_mid"),
            "share_entry_better_than_mid": share("entry_better_than_mid"),
            "mean_entry_mid_minus_natural_per_share": float((df["entry_mid"] - df["entry_natural"]).mean()) if "entry_mid" in df else None,
            "exits_checked": int(df.get("exit_mid", pd.Series(dtype=float)).notna().sum()),
            "share_exit_outside_natural": share("exit_outside_natural"),
            "exit_outside_natural_rows": df[df.get("exit_outside_natural", False) == True][["date", "time_opened", "legs", "reported_exit", "exit_natural", "pl"]].to_dict("records") if "exit_outside_natural" in df else [],
            "tolerance_per_share": a.tol, "seed": a.seed, "quotes": "cached minute NBBO, read-only"}
    json.dump(summ, open(f"{a.out}/fill_audit_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "exit_outside_natural_rows"}, indent=1, default=str))
    print("exit_outside_natural:", len(summ["exit_outside_natural_rows"]))


if __name__ == "__main__":
    main()
