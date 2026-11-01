"""Correlate Goldmine trade outcomes with market regime (2025-01-01 -> 2026-10-30).

For every Goldmine trade entry, looks up the 5-class regime (from the trained
RegimePredictor on XAUUSD M15) that was active at entry time, then reports
win rate, net P/L and avg P/L per regime, plus which regime drives wins vs losses.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from regime_predictor import RegimePredictor


TRADE_LOG = Path(__file__).resolve().parent / "GMStandalone_backtest_trades.csv"
M15_CSV = Path("C:/Users/MYCkey98/Documents/Reversion/data/raw/XAUUSDm_M15_202401012300_202610302045.csv")


def load_m15():
    df = pd.read_csv(M15_CSV, sep="\t")
    df = df.rename(columns={
        "<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close",
        "<TICKVOL>": "tickvol", "<SPREAD>": "spread",
    })
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("time").sort_index()
    df["spread"] = df["spread"] / 100.0  # points -> price units (rough)
    return df[["open", "high", "low", "close", "tickvol", "spread"]]


def main():
    predictor = RegimePredictor()
    if not predictor.load_model():
        sys.exit("could not load regime model")

    print("Loading M15 history ...")
    df15 = load_m15()
    print(f"  {len(df15):,} bars {df15.index.min()} -> {df15.index.max()}")

    print("Running regime prediction over history ...")
    hist = predictor.predict_history(df15)
    if hist is None:
        sys.exit("regime history prediction failed")
    # regime per M15 bar (3-class) + state name
    regime = hist[["regime_now_3class", "state_name", "score"]].copy()
    regime.index = pd.DatetimeIndex(regime.index)

    print("Loading Goldmine trades ...")
    T = pd.read_csv(TRADE_LOG, parse_dates=["entry", "exit"])
    T = T.sort_values("entry").reset_index(drop=True)
    T = T[(T.entry.dt.year >= 2025)]  # 2025-2026 window
    print(f"  {len(T)} trades in 2025-2026 window")

    # Map each trade entry to the regime active at entry (latest M15 bar at or before entry)
    reg_idx = regime.index
    entr_regime = []
    entr_score = []
    for et in T.entry:
        pos = reg_idx.searchsorted(et, side="right") - 1
        if pos < 0:
            pos = 0
        entr_regime.append(regime.iloc[pos]["regime_now_3class"])
        entr_score.append(regime.iloc[pos]["score"])
    T["regime_at_entry"] = entr_regime
    T["score_at_entry"] = entr_score
    T["win"] = T.pl > 0

    # ---- aggregate by regime ----
    g = T.groupby("regime_at_entry").agg(
        trades=("pl", "size"),
        wins=("win", "sum"),
        winrate=("win", "mean"),
        net=("pl", "sum"),
        avg=("pl", "mean"),
        median=("pl", "median"),
        worst=("pl", "min"),
        best=("pl", "max"),
    )
    g["winrate"] = (g["winrate"] * 100).round(1)
    g = g.sort_values("net", ascending=False)
    print("\n=== Trade outcome by regime at entry (2025-01-01 -> 2026-10-30) ===")
    print(g.round(2).to_string())

    total_trades = len(T)
    total_net = T.pl.sum()
    total_wr = T.win.mean() * 100
    print(f"\nOVERALL: {total_trades} trades | net ${total_net:,.0f} | win rate {total_wr:.1f}%")

    # contribution: share of net P/L per regime vs share of trades
    g["net_share_pct"] = (g.net / total_net * 100).round(1)
    g["trade_share_pct"] = (g.trades / total_trades * 100).round(1)
    print("\n=== Regime contribution (net share vs trade share) ===")
    print(g[["trades", "trade_share_pct", "net", "net_share_pct", "winrate", "avg"]].round(2).to_string())

    # best / worst regime
    best = g.loc[g.net.idxmax()]
    worst = g.loc[g.net.idxmin()]
    print(f"\nBEST regime:  {best.name}  -> net ${best.net:,.0f}, {best.winrate:.1f}% WR, {best.trades} trades")
    print(f"WORST regime: {worst.name} -> net ${worst.net:,.0f}, {worst.winrate:.1f}% WR, {worst.trades} trades")

    # direction x regime
    print("\n=== By regime x direction ===")
    gd = T.groupby(["regime_at_entry", "dir"]).agg(
        trades=("pl", "size"), wins=("win", "sum"), net=("pl", "sum"), avg=("pl", "mean"),
    )
    gd["winrate"] = (gd.wins / gd.trades * 100).round(1)
    print(gd.round(2).to_string())

    # exit reason x regime
    print("\n=== Exit reason x regime (counts) ===")
    print(pd.crosstab(T.regime_at_entry, T.reason).to_string())

    # save enriched trade log
    out = Path(__file__).resolve().parent / "GMStandalone_trades_with_regime.csv"
    T.to_csv(out, index=False)
    print(f"\nEnriched trade log written to {out}")


if __name__ == "__main__":
    main()
