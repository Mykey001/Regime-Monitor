"""Goldmine / MoneyMaker EA (MoneyMaker.mq5 v1.30, settings of 'moneymaker current.set') as ONE Python file.

Everything comes from the MetaTrader 5 terminal (the `MetaTrader5` package) - no CSV exports needed:
    live      trades the strategy through the terminal (same signal, filters, 3-leg reverse basket, boost,
              daily target / loss limit and 2.5 h time stop as the EA)
    backtest  downloads M1 history from the terminal (cached on disk) and replays the strategy with the replica
              that was validated against the MT5 Strategy Tester (443 of 445 trades matched, P/L corr 0.998)
    fetch     only downloads / updates the cached M1 history

Self-contained: it imports nothing from the rest of goldmine_py and does not touch the running bot or the
Streamlit app. It uses its own magic number (789012005) and its own files (GMStandalone_*), so it can sit
next to the EA (789012003) and live.py (789012004) without closing each other's trades.

Defaults = the EA's live settings: 3 x 0.10 lot, target $100, daily loss limit $1,000, 2.5 h time stop, boost
x2 (lot and target) when the ATR regime has been stable >= 200 min, not Friday and RSI extreme bars <= 3.
`--compound` switches to the bot's balance-linked sizing (0.10 lot and $100 target per $1,000, loss limit 80 %).

Usage (Windows, MT5 terminal open and logged in, Algo Trading enabled):
    pip install MetaTrader5 numpy pandas numba
    python goldmine_standalone.py backtest --from 2025-01-01
    python goldmine_standalone.py live --dry-run        # logs every decision, never sends orders - start here
    python goldmine_standalone.py live                  # trades the DEMO account (--allow-live for a real one)
"""
import argparse, csv, glob, json, logging, os, sys, time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:                      # backtests from a cached history work without the package
    mt5 = None

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                      # still correct, just slow (a full 5-year backtest takes minutes)
    HAVE_NUMBA = False

    def njit(*a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) else (lambda f: f)

HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("goldmine_standalone")


# =====================================================================================================
# Settings
# =====================================================================================================
@dataclass
class Config:
    symbol: str = "XAUUSDm"
    # RSI signal (M5), checked on each new M3 bar
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    # Sizing - fixed (EA): `lot` per leg x `legs`. Compounding: lot from the day-start balance
    lot: float = 0.1
    legs: int = 3
    compounding: bool = False
    lot_per_1000: float = 0.10
    target_per_lot: float = 1000.0          # daily target $ = lot x this (0.1 lot -> $100)
    daily_loss_pct: float = 0.80            # compounding only: loss limit = this x day-start balance
    daily_target: float = 100.0             # fixed sizing
    daily_loss_limit: float = 1000.0        # fixed sizing
    contract_size: float = 100.0            # oz per 1.00 lot (XAUUSD)
    slippage_points: int = 30
    magic: int = 789012005
    # ATR regime filter (M15)
    atr_period: int = 14
    atr_avg_period: int = 20
    atr_ratio_threshold: float = 1.1
    min_regime_persistence_minutes: float = 60
    # RSI extreme-bars filter
    rsi_extreme_history: int = 15
    rsi_extreme_max: int = 7
    # Trend / EMA / POC filters
    swing_bars: int = 5
    ema_period: int = 50                    # H1
    poc_lookback_bars: int = 24             # H1
    poc_update_minutes: int = 5
    poc_bins: int = 150
    # Days Mon..Fri (Sat/Sun never trade)
    trade_days: tuple = (True, True, True, True, True)
    # Max trade duration
    use_max_duration: bool = True
    max_trade_hours: float = 2.5
    # Dynamic sizing (boost)
    boost: bool = True
    boost_multiplier: float = 2.0
    boost_require_min_persistence: bool = True
    boost_min_persistence_minutes: float = 200
    boost_excluded_dow: int = 5             # MQL convention 0=Sun..6=Sat -> Friday
    boost_exclude_day: bool = True
    boost_max_rsi_extreme_bars: int = 3
    boost_scale_loss_limit: bool = False
    # Broker-side emergency SL/TP in points (0 = off, as in the .set file)
    emergency_sl_points: int = 0
    emergency_tp_points: int = 0
    trade_send_retries: int = 2
    poll_seconds: float = 0.25
    # Files (next to this script unless absolute)
    trade_log: str = "GMStandalone_TradeLog.csv"
    daily_log: str = "GMStandalone_DailyLog.csv"
    state_file: str = "GMStandalone_state.json"
    log_file: str = "goldmine_standalone.log"
    cache_file: str = "GMStandalone_M1_{symbol}.npz"


def path_of(f):
    return f if os.path.isabs(f) else os.path.join(HERE, f)


def base_sizing(cfg, balance, step=0.01, vmin=0.01, vmax=1e9):
    """(lot per leg, daily target $, daily loss limit $) before any boost, for a day that starts at `balance`."""
    if cfg.compounding:
        lot = np.floor(balance / 1000.0 * cfg.lot_per_1000 / step + 1e-9) * step
        lot = round(min(max(lot, vmin), vmax), 2)
        return lot, round(lot * cfg.target_per_lot, 2), round(cfg.daily_loss_pct * balance, 2)
    return cfg.lot, cfg.daily_target, cfg.daily_loss_limit


def boosted(cfg, lot, target, loss, step=0.01, vmin=0.01, vmax=1e9):
    m = cfg.boost_multiplier
    l2 = round(min(max(round(lot * m / step) * step, vmin), vmax), 2)
    return l2, target * m, loss * (m if cfg.boost_scale_loss_limit else 1.0)


def boost_ok(persist_min, dow_mql, xbars, cfg):
    """IsFavorableLowDrawdownCondition. dow_mql: 0=Sun..6=Sat."""
    if not cfg.boost:
        return False
    if cfg.boost_require_min_persistence and persist_min < cfg.boost_min_persistence_minutes:
        return False
    if cfg.boost_exclude_day and dow_mql == cfg.boost_excluded_dow:
        return False
    if cfg.boost_max_rsi_extreme_bars > 0 and xbars > cfg.boost_max_rsi_extreme_bars:
        return False
    return True


# =====================================================================================================
# Signal maths for the LIVE path (bar arrays oldest -> newest, last element = forming bar, exactly what
# copy_rates_from_pos returns). Checked 100 % identical to the replica below on 3,116 historical signals.
# =====================================================================================================
def rsi_wilder(close, period=14):
    """iRSI (Wilder smoothing, SMA seed)."""
    d = np.diff(close, prepend=close[0])
    up, dn = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return out
    ag, al = up[1:period + 1].mean(), dn[1:period + 1].mean()
    for k in range(period, len(close)):
        if k > period:
            ag = (ag * (period - 1) + up[k]) / period
            al = (al * (period - 1) + dn[k]) / period
        out[k] = 100.0 - 100.0 / (1.0 + ag / (al if al > 0 else 1e-12))
    return out


def rsi_signal(close_m5, cfg):
    """CheckRSISignal: +1 = raw BUY (cross up through oversold), -1 = raw SELL, 0 = none."""
    r = rsi_wilder(close_m5, cfg.rsi_period)
    cur, prev = r[-1], r[-2]
    if prev < cfg.rsi_oversold and cur >= cfg.rsi_oversold:
        return 1, cur, r
    if prev > cfg.rsi_overbought and cur <= cfg.rsi_overbought:
        return -1, cur, r
    return 0, cur, r


def rsi_extreme_bars(rsi, for_buy, cfg):
    """GetRSIExtremeBars: consecutive closed bars (from bar[1] back) already beyond the level."""
    thr = cfg.rsi_oversold if for_buy else cfg.rsi_overbought
    n = 0
    for q in range(1, cfg.rsi_extreme_history):
        v = rsi[-1 - q]
        if (v <= thr) if for_buy else (v >= thr):
            n += 1
        else:
            break
    return n


def atr_ratio_live(h, l, c, period=14, avg_period=20):
    """UpdateATRRegimeState: SMA-ATR of the forming bar / mean of the previous avg_period ATR values."""
    cp = np.r_[c[0], c[:-1]]
    tr = np.maximum(h, cp) - np.minimum(l, cp)
    cs = np.r_[0.0, np.cumsum(tr)]
    atr = (cs[period:] - cs[:-period]) / period
    if len(atr) < avg_period + 1:
        return np.nan
    avg = atr[-1 - avg_period:-1].mean()
    return atr[-1] / avg if avg > 0 else np.nan


def ema_last(close_h1, period=50):
    a = 2.0 / (period + 1)
    e = close_h1[0]
    for x in close_h1[1:]:
        e += a * (x - e)
    return e


def poc_live(h, l, tv, bins=150):
    """CalculatePOC over the given bars (last = forming)."""
    hi, lo = h.max(), l.min()
    if hi <= lo:
        return 0.0
    w = (hi - lo) / bins
    vol = np.zeros(bins)
    for bh, bl, bv in zip(h, l, tv):
        s = max(int((bl - lo) / w), 0)
        e = min(int((bh - lo) / w), bins - 1)
        span = e - s + 1
        vol[s:e + 1] += bv / span if span > 0 else bv
    return lo + int(np.argmax(vol)) * w + w / 2.0


def trend_live(h, l, c, swing=5, nb=50):
    """AnalyzeTrend (with the EA's own swing-count conventions). +1 BULLISH, -1 BEARISH, 0 NEUTRAL."""
    if len(c) < nb:
        return 0
    hs, ls, cl = h[::-1][:nb], l[::-1][:nb], c[::-1][:nb]        # [0] = forming bar
    bull = int(np.sum(cl[1:10] > cl[2:11])); bear = 9 - bull
    SH, SL = [], []
    for j in range(swing, nb - swing):
        if all(hs[j] > hs[j - q] and hs[j] > hs[j + q] for q in range(1, swing + 1)):
            SH.append(hs[j])
        if all(ls[j] < ls[j - q] and ls[j] < ls[j + q] for q in range(1, swing + 1)):
            SL.append(ls[j])
    hh = sum(SH[q - 1] > SH[q] for q in range(1, len(SH)))
    ll = sum(SL[q - 1] > SL[q] for q in range(1, len(SL)))
    if (bull > bear and hh >= ll) or bull >= 7:
        return 1
    if (bear > bull and ll >= hh) or bear >= 7:
        return -1
    return 0


# =====================================================================================================
# M1 history from the MT5 terminal (cached on disk)
# =====================================================================================================
FIELDS = ("time", "open", "high", "low", "close", "tick_volume", "spread")


def mt5_start(terminal=None):
    if mt5 is None:
        sys.exit("The MetaTrader5 package is missing: pip install MetaTrader5")
    ok = mt5.initialize(terminal) if terminal else mt5.initialize()
    if not ok:
        sys.exit(f"MT5 initialize failed: {mt5.last_error()} - open and log in to the terminal first")


def _rates_range(symbol, a, b):
    """M1 bars in [a, b) from the terminal, fetched in quarter-year pieces (a terminal can cap one call)."""
    out, x = [], a
    while x < b:
        y = min(x + timedelta(days=92), b)
        r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, x, y)
        if r is not None and len(r):
            out.append(r)
        x = y
    return np.concatenate(out) if out else None


def _arrays(r):
    return {k: np.asarray(r[k]) for k in FIELDS}


def _merge(*parts):
    cat = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    order = np.argsort(cat["time"], kind="stable")
    cat = {k: v[order] for k, v in cat.items()}
    keep = np.r_[cat["time"][1:] != cat["time"][:-1], True]      # duplicate minute: keep the later copy
    return {k: v[keep] for k, v in cat.items()}


def load_history(cfg, start, end=None, update=True, refresh=False, terminal=None):
    """dict ts/tm/o/h/l/c/tv/sp covering [start, now] (every finished minute), updated from the terminal.
    `sp` is the spread in price units. With update=False only the cache is used (no terminal needed)."""
    cache = path_of(cfg.cache_file.format(symbol=cfg.symbol))
    have, point = None, None
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        have = {k: z[k] for k in FIELDS}; point = float(z["point"])
    if update:
        mt5_start(terminal)
        if not mt5.symbol_select(cfg.symbol, True):
            sys.exit(f"{cfg.symbol} is not available in the terminal: {mt5.last_error()}")
        point = float(mt5.symbol_info(cfg.symbol).point)
        tick = mt5.symbol_info_tick(cfg.symbol)
        now_min = (int(tick.time) // 60) * 60 if tick else int(time.time()) // 60 * 60
        utc = lambda s: datetime.fromtimestamp(int(s), tz=timezone.utc)          # noqa: E731
        want = int(start.replace(tzinfo=timezone.utc).timestamp())
        pieces = [have] if have else []
        if have is None or int(have["time"][0]) > want + 86400 * 4:               # need earlier history
            hi = int(have["time"][0]) if have else now_min + 86400
            log.info("downloading M1 %s from %s ...", cfg.symbol, utc(want).date())
            r = _rates_range(cfg.symbol, utc(want), utc(hi))
            if r is not None:
                pieces.append(_arrays(r))
        if have:                                                                   # newer bars
            lo = int(have["time"][-1]) + 60
            r = _rates_range(cfg.symbol, utc(lo), utc(now_min + 86400))
            if r is not None:
                pieces.append(_arrays(r))
        if not pieces:
            sys.exit(f"the terminal returned no M1 bars for {cfg.symbol} ({mt5.last_error()})")
        have = _merge(*pieces)
        have = {k: v[have["time"] < now_min] for k, v in have.items()}            # drop the minute still forming
        np.savez(cache, point=point, **have)
        log.info("M1 history: %s bars, %s -> %s (cached in %s)", f"{len(have['time']):,}",
                 utc(have["time"][0]).date(), utc(have["time"][-1]).date(), os.path.basename(cache))
        mt5.shutdown()
    if have is None:
        sys.exit("no cached history yet - run once with the terminal open (drop --no-update)")
    return to_data(have, point)


def to_data(a, point):
    t = np.asarray(a["time"]).astype(np.int64)
    ts = t.astype("datetime64[s]").astype("datetime64[ns]")
    return dict(ts=ts, tm=t // 60, o=np.asarray(a["open"], float), h=np.asarray(a["high"], float),
                l=np.asarray(a["low"], float), c=np.asarray(a["close"], float),
                tv=np.asarray(a["tick_volume"], float), sp=np.asarray(a["spread"], float) * point)


def load_csv(path, point=0.001):
    """Optional: an MT5 'Export bars' M1 file (tab separated), instead of the terminal."""
    m = pd.read_csv(path, sep="\t", header=0, names=["d", "t", "o", "h", "l", "c", "tv", "v", "sp"])
    ts = pd.to_datetime(m.d + " " + m.t, format="%Y.%m.%d %H:%M:%S")
    return dict(ts=ts.values, tm=ts.values.astype("datetime64[m]").astype(np.int64), o=m.o.values, h=m.h.values,
                l=m.l.values, c=m.c.values, tv=m.tv.values.astype(np.float64), sp=m.sp.values * point)


# =====================================================================================================
# Backtest replica (vectorised indicators + signal list + sequential simulator), validated vs MT5 tester.
# Check-time convention: the EA acts on the first tick of each new M3 bar; on M1 data that tick is the OPEN
# of the first M1 bar of the M3 bar, and every "forming bar" is the M1 bars before it in the same higher-
# timeframe bar plus that open price.
# =====================================================================================================
def make_params(cfg):
    return dict(rsi_period=cfg.rsi_period, os=cfg.rsi_oversold, ob=cfg.rsi_overbought, atr_period=cfg.atr_period,
                atr_avg=cfg.atr_avg_period, atr_thr=cfg.atr_ratio_threshold,
                regime_min=cfg.min_regime_persistence_minutes, xbars_hist=cfg.rsi_extreme_history,
                xbars_max=cfg.rsi_extreme_max, ema_period=cfg.ema_period, poc_bars=cfg.poc_lookback_bars,
                poc_bins=cfg.poc_bins, poc_every=cfg.poc_update_minutes, swing=cfg.swing_bars)


@njit(cache=True)
def _agg(tm, o, h, l, c, tv, tf):
    n = len(tm)
    gid = tm // tf
    gidx = np.empty(n, np.int64)
    ph = np.empty(n); pl = np.empty(n); ptv = np.empty(n)
    G = 0
    for i in range(n):
        if i == 0 or gid[i] != gid[i - 1]:
            G += 1
    O = np.empty(G); H = np.empty(G); L = np.empty(G); C = np.empty(G); TV = np.empty(G); T0 = np.empty(G, np.int64)
    g = -1
    for i in range(n):
        if i == 0 or gid[i] != gid[i - 1]:
            g += 1
            O[g] = o[i]; H[g] = h[i]; L[g] = l[i]; TV[g] = 0.0; T0[g] = gid[i] * tf
            ph[i] = np.nan; pl[i] = np.nan; ptv[i] = 0.0
        else:
            ph[i] = H[g]; pl[i] = L[g]; ptv[i] = TV[g]
            if h[i] > H[g]: H[g] = h[i]
            if l[i] < L[g]: L[g] = l[i]
        C[g] = c[i]; TV[g] += tv[i]; gidx[i] = g
    return O, H, L, C, TV, T0, gidx, ph, pl, ptv


def agg(d, tf):
    O, H, L, C, TV, T0, gidx, ph, pl, ptv = _agg(d["tm"], d["o"], d["h"], d["l"], d["c"], d["tv"], tf)
    return dict(O=O, H=H, L=L, C=C, TV=TV, T0=T0, gidx=gidx, ph=ph, pl=pl, ptv=ptv)


def wilder_rsi(C, n):
    d = np.diff(C, prepend=C[0]); up = np.maximum(d, 0); dn = np.maximum(-d, 0)
    ag = np.full(len(C), np.nan); al = np.full(len(C), np.nan)
    ag[n] = up[1:n + 1].mean(); al[n] = dn[1:n + 1].mean()
    for k in range(n + 1, len(C)):
        ag[k] = (ag[k - 1] * (n - 1) + up[k]) / n; al[k] = (al[k - 1] * (n - 1) + dn[k]) / n
    rsi = 100 - 100 / (1 + ag / np.where(al == 0, 1e-12, al))
    return rsi, ag, al


@njit(cache=True)
def _poc(H, L, TV, k, h0, l0, tv0, nbars, bins):
    hi = h0; lo = l0
    for j in range(1, nbars):
        if k - j < 0: break
        if H[k - j] > hi: hi = H[k - j]
        if L[k - j] < lo: lo = L[k - j]
    if hi <= lo: return 0.0
    w = (hi - lo) / bins; vol = np.zeros(bins)
    for j in range(nbars):
        if j == 0:
            bh, bl, bv = h0, l0, tv0
        else:
            if k - j < 0: break
            bh, bl, bv = H[k - j], L[k - j], TV[k - j]
        s = int((bl - lo) / w); e = int((bh - lo) / w)
        if s < 0: s = 0
        if e >= bins: e = bins - 1
        span = e - s + 1
        per = bv / span if span > 0 else bv
        for b in range(s, e + 1): vol[b] += per
    mb = 0; mv = -1.0
    for b in range(bins):
        if vol[b] > mv: mv = vol[b]; mb = b
    return lo + mb * w + w / 2.0


@njit(cache=True)
def _regime_changed(elev, tm):
    n = len(tm); out = np.zeros(n, np.int64); last = tm[0]
    for i in range(1, n):
        if elev[i] != elev[i - 1]: last = tm[i]
        out[i] = last
    return out


@njit(cache=True)
def _poc_series(tm, o, kh, ph, pl, ptv, H, L, TV, k3, C3, nbars, bins, every):
    n = len(tm); poc = np.zeros(n); bias = np.zeros(n, np.int8)
    lastupd = -10 ** 9; cur = 0.0; b = 0
    for i in range(n):
        if tm[i] - lastupd >= every or cur == 0:
            hh = o[i] if np.isnan(ph[i]) else max(ph[i], o[i])
            ll = o[i] if np.isnan(pl[i]) else min(pl[i], o[i])
            v = _poc(H, L, TV, kh[i], hh, ll, ptv[i] + 1, nbars, bins)
            if v > 0: cur = v; lastupd = tm[i]
        if i > 0 and k3[i] != k3[i - 1] and cur > 0:
            cp = C3[k3[i] - 1]
            if cp > cur: b = 1
            elif cp < cur: b = -1
        poc[i] = cur; bias[i] = b
    return poc, bias


def indicators(d, P):
    o = d["o"]; n = len(o); tm = d["tm"]
    m3, m5, m15, h1 = agg(d, 3), agg(d, 5), agg(d, 15), agg(d, 60)
    # RSI on M5, forming bar uses the current price
    rp = P["rsi_period"]
    r5, ag, al = wilder_rsi(m5["C"], rp)
    k = m5["gidx"]; prevC = m5["C"][k - 1]
    ch = o - prevC
    ag0 = (ag[k - 1] * (rp - 1) + np.maximum(ch, 0)) / rp; al0 = (al[k - 1] * (rp - 1) + np.maximum(-ch, 0)) / rp
    rsi0 = 100 - 100 / (1 + ag0 / np.where(al0 == 0, 1e-12, al0)); rsi1 = r5[k - 1]
    # ATR (SMA) on M15 + ratio vs the mean of the prior atr_avg completed ATR values
    ap, aa = P["atr_period"], P["atr_avg"]
    C, H, L = m15["C"], m15["H"], m15["L"]
    Cp = np.r_[C[0], C[:-1]]
    TR = np.maximum(H, Cp) - np.minimum(L, Cp)
    cs = np.r_[0, np.cumsum(TR)]
    ATRc = np.full(len(TR), np.nan); ATRc[ap - 1:] = (cs[ap:] - cs[:-ap]) / ap
    ca = np.r_[0, np.nancumsum(np.nan_to_num(ATRc))]
    k15 = m15["gidx"]
    h0 = np.fmax(m15["ph"], o); l0 = np.fmin(m15["pl"], o)
    cprev = C[k15 - 1]
    TR0 = np.maximum(h0, cprev) - np.minimum(l0, cprev)
    sumn = cs[k15] - cs[np.maximum(k15 - (ap - 1), 0)]
    ATR0 = (TR0 + sumn) / ap
    avg = (ca[k15] - ca[np.maximum(k15 - aa, 0)]) / aa
    ratio = np.where(k15 >= ap + aa + 6, ATR0 / avg, np.nan)
    elev = ratio > P["atr_thr"]
    persist = (tm - _regime_changed(elev, tm)).astype(float)
    # H1 EMA with the forming bar
    Ch = h1["C"]; a = 2 / (P["ema_period"] + 1); E = np.empty(len(Ch)); E[0] = Ch[0]
    for j in range(1, len(Ch)): E[j] = E[j - 1] + a * (Ch[j] - E[j - 1])
    kh = h1["gidx"]; ema0 = E[kh - 1] + a * (o - E[kh - 1])
    # POC (H1, incl. forming bar) recomputed every few minutes; bias updated on each new M3 bar
    k3 = m3["gidx"]
    poc, bias = _poc_series(tm, o, kh, h1["ph"], h1["pl"], h1["ptv"], h1["H"], h1["L"], h1["TV"], k3, m3["C"],
                            P["poc_bars"], P["poc_bins"], P["poc_every"])
    newm3 = np.r_[True, k3[1:] != k3[:-1]]
    return dict(m3=m3, m5=m5, m15=m15, h1=h1, r5=r5, rsi0=rsi0, rsi1=rsi1, ratio=ratio, elev=elev, persist=persist,
                ema0=ema0, poc=poc, bias=bias, newm3=newm3, atr0=ATR0)


def trend(bars, i, o, swing=5, nb=50):
    """AnalyzeTrend replica; bar 0 = forming bar at minute i."""
    k = bars["gidx"][i]
    if k < nb: return 0
    hs = np.r_[max(o[i], bars["ph"][i]) if not np.isnan(bars["ph"][i]) else o[i], bars["H"][k - 1:k - nb:-1]]
    ls = np.r_[min(o[i], bars["pl"][i]) if not np.isnan(bars["pl"][i]) else o[i], bars["L"][k - 1:k - nb:-1]]
    cl = np.r_[o[i], bars["C"][k - 1:k - nb:-1]]
    bull = int(np.sum(cl[1:10] > cl[2:11])); bear = 9 - bull
    SH, SL = [], []
    for j in range(swing, nb - swing):
        if all(hs[j] > hs[j - q] and hs[j] > hs[j + q] for q in range(1, swing + 1)): SH.append(hs[j])
        if all(ls[j] < ls[j - q] and ls[j] < ls[j + q] for q in range(1, swing + 1)): SL.append(ls[j])
    hh = sum(SH[q - 1] > SH[q] for q in range(1, len(SH)))
    ll = sum(SL[q - 1] > SL[q] for q in range(1, len(SL)))
    if (bull > bear and hh >= ll) or bull >= 7: return 1
    if (bear > bull and ll >= hh) or bear >= 7: return -1
    return 0


def signals(d, I, P, cfg, start=0):
    """Every RSI cross at an M3 check with the verdict of each filter. `start` = first bar index to scan."""
    o, tm = d["o"], d["tm"]; rows = []
    idx = np.flatnonzero(I["newm3"]); idx = idx[idx >= start]
    r0, r1 = I["rsi0"][idx], I["rsi1"][idx]
    buy = (r1 < P["os"]) & (r0 >= P["os"]); sell = (r1 > P["ob"]) & (r0 <= P["ob"])
    r5 = I["r5"]; k5 = I["m5"]["gidx"]
    for i in idx[buy | sell]:
        sig = 1 if I["rsi1"][i] < P["os"] else -1
        k = k5[i]; thr = P["os"] if sig == 1 else P["ob"]; xb = 0
        for q in range(1, P["xbars_hist"]):
            v = r5[k - q]
            if (v <= thr) if sig == 1 else (v >= thr): xb += 1
            else: break
        t3 = trend(I["m3"], i, o, P["swing"]); t5 = trend(I["m5"], i, o, P["swing"])
        rev = -sig
        f_atr = (not I["elev"][i]) or I["persist"][i] >= P["regime_min"]
        f_xb = xb <= P["xbars_max"]
        f_tr = (t3 >= 0) if sig == 1 else (t3 <= 0)
        f_ema = (o[i] > I["ema0"][i]) if rev == 1 else (o[i] < I["ema0"][i])
        f_poc = I["bias"][i] == rev
        dow = int(pd.Timestamp(d["ts"][i]).dayofweek + 1) % 7            # MQL: 0 = Sunday
        rows.append(dict(i=i, tm=tm[i], sig=sig, dir=rev, rsi=I["rsi0"][i], xbars=xb, atr_ratio=I["ratio"][i],
                         elev=bool(I["elev"][i]), persist=I["persist"][i], t3=t3, t5=t5, f_atr=f_atr, f_xb=f_xb,
                         f_tr=f_tr, f_ema=f_ema, f_poc=f_poc, dow=dow,
                         boost=boost_ok(I["persist"][i], dow, xb, cfg)))
    if not rows:
        return pd.DataFrame()
    S = pd.DataFrame(rows)
    S["ok"] = S.f_atr & S.f_xb & S.f_tr & S.f_ema & S.f_poc
    S["ts"] = pd.to_datetime(S.tm, unit="m")
    return S


def _path(o, h, l, c):
    """Intrabar order heuristic: bullish bar o->l->h->c, bearish o->h->l->c."""
    return (o, l, h, c) if c >= o else (o, h, l, c)


def simulate(d, S, cfg, start=None, end=None, balance=1000.0, spread_k=1.0):
    """Sequential EA simulator on M1 bars. Baskets are closed at the daily target (closed + floating P/L), at the
    daily loss limit or by the time stop; after a target or loss-limit exit nothing more trades that day.
    Prices are bid; buys pay the spread on entry, sells on exit. A basket still open at the end of the data is
    not counted."""
    o, h, l, c, tm = d["o"], d["h"], d["l"], d["c"], d["tm"]
    sp = d["sp"] * spread_k; n = len(o); day = tm // 1440
    allowed = np.array([False, *cfg.trade_days, False])                  # index = MQL day of week 0..6
    C = S[S.ok & allowed[S.dow.values]].sort_values("i")
    ci = C.i.values; cboost = C.boost.values; cdir = C.dir.values
    oz_per_lot = cfg.legs * cfg.contract_size
    ts_min = cfg.max_trade_hours * 60 if cfg.use_max_duration else float("inf")
    bal = balance; day_bal = bal
    lot0, tgt0, loss0 = base_sizing(cfg, day_bal)
    lo_i = 0 if start is None else np.searchsorted(tm, start)
    hi_i = n if end is None else np.searchsorted(tm, end)
    trades = []; curday = -1; closed = 0.0; limit = False; bk = None; blown = False

    def new_day():
        nonlocal day_bal, lot0, tgt0, loss0
        day_bal = bal
        lot0, tgt0, loss0 = base_sizing(cfg, day_bal)

    def pl_of(px_bid, b, j):                                              # $/oz at bid price px
        return (px_bid - b["e"]) if b["dir"] == 1 else (b["e"] - px_bid - sp[j])

    i = lo_i
    while i < hi_i:
        if bk is None:
            q = np.searchsorted(ci, i)
            if q >= len(ci) or ci[q] >= hi_i: break
            j = ci[q]
            if day[j] != curday: curday = day[j]; closed = 0.0; limit = False; new_day()
            if cfg.compounding and bal <= 0: blown = True; break
            if limit:
                i = np.searchsorted(day, curday + 1); continue
            b = bool(cboost[q])
            lot, tgt, loss = boosted(cfg, lot0, tgt0, loss0) if b else (lot0, tgt0, loss0)
            dr = int(cdir[q]); e = o[j] + sp[j] if dr == 1 else o[j]
            bk = dict(dir=dr, e=e, lot=lot, oz=lot * oz_per_lot, t0=tm[j], i0=j, boost=b, target=tgt, loss=loss,
                      mae=0.0, mfe=0.0)
            i = j
        if day[i] != curday:                                             # midnight with a basket open
            curday = day[i]; closed = 0.0; limit = False; new_day()
            if bk is not None:
                bk["target"] = tgt0; bk["loss"] = loss0                  # the EA resets target / loss limit at midnight
        done = None
        for k_, px in enumerate(_path(o[i], h[i], l[i], c[i])):
            v = pl_of(px, bk, i); bk["mae"] = min(bk["mae"], v); bk["mfe"] = max(bk["mfe"], v)
            need = (bk["target"] - closed) / bk["oz"]; lossv = (-bk["loss"] - closed) / bk["oz"]
            if v >= need:
                done = (need if k_ > 0 else v, "target"); break
            if v <= lossv:
                done = (lossv if k_ > 0 else v, "loss limit"); break
            if k_ == 0 and tm[i] - bk["t0"] >= ts_min:
                done = (v, "time stop"); break
        if done is not None:
            v, why = done; pnl = v * bk["oz"]
            closed += pnl
            if why != "time stop": limit = True
            bal += pnl
            trades.append(dict(entry=pd.Timestamp(bk["t0"], unit="m"), dir=bk["dir"], boost=bk["boost"], lot=bk["lot"],
                               oz=bk["oz"], entry_px=bk["e"], exit=pd.Timestamp(tm[i], unit="m"), reason=why,
                               pl=pnl, mae=bk["mae"] * bk["oz"], mfe=bk["mfe"] * bk["oz"], dur_min=int(tm[i] - bk["t0"]),
                               balance=bal))
            bk = None
        i += 1
    T = pd.DataFrame(trades); T.attrs["blown"] = blown
    return T


def report(T, balance, out):
    if T.empty:
        print("no trades in this period"); return
    eq = balance + T.pl.cumsum(); dd = (eq - eq.cummax()).min()
    days = T.groupby(T.entry.dt.normalize()).pl.sum()
    print(f"\ntrades {len(T)} | net ${T.pl.sum():,.0f} | per trade ${T.pl.mean():.2f} | win {np.mean(T.pl > 0):.0%} | "
          f"worst ${T.pl.min():,.0f} | boosted {int(T.boost.sum())}")
    print(f"loss-limit exits {(T.reason == 'loss limit').sum()} | time stops {(T.reason == 'time stop').sum()} | "
          f"targets {(T.reason == 'target').sum()} | trading days {len(days)} | worst day ${days.min():,.0f}")
    print(f"max closed-trade drawdown ${dd:,.0f} | end balance ${eq.iloc[-1]:,.0f}" +
          ("  *** ACCOUNT WIPED OUT ***" if T.attrs.get("blown") else ""))
    print(T.groupby(T.entry.dt.year).pl.agg(trades="size", per_trade="mean", net="sum").round(1).to_string())
    T.to_csv(out, index=False); print(f"trades written to {out}")


def cmd_backtest(a, cfg):
    start = pd.Timestamp(a.start) if a.start else pd.Timestamp("2021-08-01")
    if a.csv:
        d = load_csv(a.csv)
    else:
        d = load_history(cfg, (start - pd.Timedelta(days=a.warmup_days)).to_pydatetime(), update=not a.no_update,
                         refresh=a.refresh, terminal=a.terminal)
    P = make_params(cfg)
    if not HAVE_NUMBA:
        log.warning("numba is not installed - this will be slow (pip install numba)")
    t0 = time.time(); I = indicators(d, P)
    to_min = lambda s: int(pd.Timestamp(s).value // 60e9) if s else None   # noqa: E731
    s_min, e_min = to_min(a.start), to_min(a.end)
    S = signals(d, I, P, cfg, start=int(np.searchsorted(d["tm"], s_min)) if s_min else 0)
    if S.empty:
        sys.exit("no signals in the data")
    T = simulate(d, S, cfg, start=s_min, end=e_min, balance=a.balance, spread_k=a.spread_k)
    sizing = (f"compounding {cfg.lot_per_1000:.2f} lot/$1k" if cfg.compounding else f"fixed 3 x {cfg.lot:.2f}")
    print(f"{cfg.symbol} {pd.Timestamp(d['ts'][0]).date()} -> {pd.Timestamp(d['ts'][-1]).date()} | sizing {sizing}"
          f"{' + boost' if cfg.boost else ''} | start balance ${a.balance:,.0f} | {len(S)} signals, {int(S.ok.sum())} pass all "
          f"filters | {time.time() - t0:.0f}s")
    report(T, a.balance, a.out)


# =====================================================================================================
# Live trading through the MT5 terminal (mirrors OnTick of the EA)
# =====================================================================================================
TRADE_HEADER = ["EntryTime", "CloseTime", "DurationSec", "Symbol", "OriginalSignal", "Direction", "Hour", "DayOfWeek",
                "RSI_AtSignal", "RSI_ExtremeBars", "ATR_Ratio", "ATR_RegimeElevated", "ATR_PersistenceMin",
                "Trend1", "Trend2", "Trend3", "PriceAtEntry", "EMA_Value", "EMA_DistPoints", "POC_Bias", "POC_Price",
                "DailyPL_AtEntry", "Equity_AtEntry", "Balance_AtEntry", "ConsecWinsBefore", "ConsecLossesBefore",
                "MaxDD_AllTime_AtEntry", "MaxDD_AllTime_Pct_AtEntry", "MaxFloatingProfit_MFE", "MaxFloatingDrawdown_MAE",
                "ClosePL", "CloseReason", "TradeNumber", "BoostActive", "LotSizeUsed", "DailyTargetUsed", "DailyLossLimitUsed"]
DAILY_HEADER = ["Date", "StartBalance", "EndEquity", "DailyPL", "MaxDD_Daily_Value", "MaxDD_Daily_Pct", "TradeCount",
                "WinCount", "LossCount", "WinRatePct", "LimitReached", "LimitReason", "BoostActiveToday"]
TREND_NAME = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
SIDE = {1: "BUY", -1: "SELL"}


def srv(ts):
    """MT5 timestamps are server wall-clock time encoded as epoch seconds."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def fmt(ts):
    return srv(ts).strftime("%Y.%m.%d %H:%M:%S")


class Bot:
    def __init__(self, cfg, dry_run):
        self.cfg, self.dry = cfg, dry_run
        self.TF = {1: mt5.TIMEFRAME_M1, 3: mt5.TIMEFRAME_M3, 5: mt5.TIMEFRAME_M5, 15: mt5.TIMEFRAME_M15,
                   60: mt5.TIMEFRAME_H1}
        self.OK = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL}
        self.RETRY = {mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED, mt5.TRADE_RETCODE_PRICE_OFF,
                      mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_CONNECTION, mt5.TRADE_RETCODE_REJECT}
        self.info = mt5.symbol_info(cfg.symbol)
        self.point = self.info.point
        fm = self.info.filling_mode
        self.filling = mt5.ORDER_FILLING_FOK if fm & 1 else (mt5.ORDER_FILLING_IOC if fm & 2 else mt5.ORDER_FILLING_RETURN)
        self.last_m3 = None; self.poc_price = 0.0; self.poc_bias = 0; self.poc_time = 0; self.poc_m3 = None
        self.ratio = float("nan"); self.daily_pl = 0.0
        self.now = mt5.symbol_info_tick(cfg.symbol).time
        self.closed_cache = (None, 0.0, 0)
        self.s = self.load_state()

    # ------------------------------------------------------------------ sizing
    def sizing(self, balance, boost=False):
        i = self.info
        lot, tgt, loss = base_sizing(self.cfg, balance, i.volume_step or 0.01, i.volume_min, i.volume_max)
        if boost:
            lot, tgt, loss = boosted(self.cfg, lot, tgt, loss, i.volume_step or 0.01, i.volume_min, i.volume_max)
        return lot, round(tgt, 2), round(loss, 2)

    # ------------------------------------------------------------------ state (restart-safe)
    def default_state(self, now):
        acc = mt5.account_info()
        lot, target, loss = self.sizing(acc.balance)
        return dict(day=now // 86400, limit=False, limit_reason="", lot=lot, target=target, loss_limit=loss,
                    boost=False, regime_elev=False, regime_changed=now, last_signal=0, wins=0, losses=0,
                    d_trades=0, d_wins=0, d_losses=0, trade_no=0, start_balance=acc.balance, eq_peak=acc.equity,
                    dd=0.0, dd_pct=0.0, eq_peak_day=acc.equity, dd_day=0.0, dd_day_pct=0.0, basket=None, saved_at=now)

    def load_state(self):
        now = self.now
        s = self.default_state(now)
        try:
            with open(path_of(self.cfg.state_file)) as f:
                s.update(json.load(f))
            if now - s.get("saved_at", 0) > 300:       # bot was off: the regime timer is unknown, restart it like the EA
                s["regime_changed"] = now
            log.info("state restored from %s", self.cfg.state_file)
        except FileNotFoundError:
            log.info("no saved state, starting fresh")
        return s

    def save(self, now):
        self.s["saved_at"] = now
        tmp = path_of(self.cfg.state_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.s, f)
        os.replace(tmp, path_of(self.cfg.state_file))

    # ------------------------------------------------------------------ market data
    def rates(self, tf, n, start=0):
        r = mt5.copy_rates_from_pos(self.cfg.symbol, self.TF[tf], start, n)
        if r is None or len(r) == 0:
            raise RuntimeError(f"no rates for TF {tf}: {mt5.last_error()}")
        return r

    def my_positions(self):
        ps = mt5.positions_get(symbol=self.cfg.symbol) or ()
        return [p for p in ps if p.magic == self.cfg.magic and p.comment.startswith("Reverse")]

    # ------------------------------------------------------------------ per-tick pipeline (OnTick)
    def on_tick(self, tick):
        now = self.now = tick.time
        self.reconcile(now)
        self.update_regime(now)
        self.update_poc(now)
        self.update_drawdowns()
        self.check_new_day(now)
        if self.update_daily_pl(now):
            return
        b = self.s["basket"]
        if self.cfg.use_max_duration and b and now - b["open_time"] >= self.cfg.max_trade_hours * 3600:
            hrs = (now - b["open_time"]) / 3600
            log.info("max trade duration reached (%.2fh)", hrs)
            self.close_basket(now, f"Max trade duration reached ({hrs:.2f}h)")
        m3 = int(self.rates(3, 1)[0]["time"])
        if m3 == self.last_m3:
            return
        first = self.last_m3 is None
        self.last_m3 = m3
        if first:
            return                  # act only on genuine new M3 bars (the EA also fires on its very first tick)
        if not self.my_positions() and self.day_allowed(now):
            self.try_signal(tick, now)

    def day_allowed(self, now):
        dow = srv(now).weekday()      # Mon=0
        return dow < 5 and self.cfg.trade_days[dow]

    # ------------------------------------------------------------------ positions / restart safety
    def reconcile(self, now):
        pos = self.my_positions(); b = self.s["basket"]
        live = {p.ticket for p in pos}
        if b is None and pos:
            b = self.s["basket"] = dict(tickets=sorted(live), dir=1 if pos[0].type == mt5.POSITION_TYPE_BUY else -1,
                                        open_time=min(p.time for p in pos), mfe=0.0, mae=0.0, snap=None)
            log.warning("adopted %d open position(s) %s - managing them as the current basket", len(pos), b["tickets"])
            self.save(now); return
        if b is None:
            return
        tracked = set(b["tickets"])
        for p in pos:                   # the EA's orphan scan: a Reverse position we do not track is closed
            if p.ticket not in tracked:
                log.warning("untracked position %s with our magic - closing it", p.ticket)
                self.close_position(p.ticket)
        if not (tracked & live):
            log.warning("basket %s closed outside the bot", b["tickets"])
            self.finish_basket(now, "Closed outside the bot (manual/stop-out)")
        elif tracked - live:
            b["tickets"] = sorted(tracked & live)

    # ------------------------------------------------------------------ ATR regime & POC
    def update_regime(self, now):
        c = self.cfg
        r = self.rates(15, c.atr_period + c.atr_avg_period + 6)
        ratio = atr_ratio_live(r["high"], r["low"], r["close"], c.atr_period, c.atr_avg_period)
        if np.isnan(ratio):
            return
        self.ratio = ratio
        elev = bool(ratio > c.atr_ratio_threshold)
        if elev != self.s["regime_elev"]:
            self.s["regime_elev"] = elev; self.s["regime_changed"] = now
            log.info("ATR regime changed: %s (ratio %.3f)", "ELEVATED" if elev else "NORMAL", ratio)
            self.save(now)

    def persist_min(self, now):
        return (now - self.s["regime_changed"]) / 60.0

    def update_poc(self, now):
        c = self.cfg
        if now - self.poc_time >= c.poc_update_minutes * 60 or self.poc_price == 0:
            h1 = self.rates(60, c.poc_lookback_bars)
            v = poc_live(h1["high"], h1["low"], h1["tick_volume"].astype(float), c.poc_bins)
            if v > 0:
                self.poc_price, self.poc_time = v, now
        m3 = int(self.rates(3, 1)[0]["time"])
        if m3 != self.poc_m3 and self.poc_price > 0:
            self.poc_m3 = m3
            prev = float(self.rates(3, 1, start=1)[0]["close"])
            if prev > self.poc_price: self.poc_bias = 1
            elif prev < self.poc_price: self.poc_bias = -1

    # ------------------------------------------------------------------ daily limits
    def check_new_day(self, now):
        if now // 86400 == self.s["day"]:
            return
        self.write_daily_row()
        acc = mt5.account_info()
        lot, target, loss = self.sizing(acc.balance)
        self.s.update(day=now // 86400, limit=False, limit_reason="", lot=lot, target=target, loss_limit=loss, boost=False,
                      d_trades=0, d_wins=0, d_losses=0, start_balance=acc.balance, eq_peak_day=acc.equity,
                      dd_day=0.0, dd_day_pct=0.0)
        log.info("=== new trading day %s, balance %.2f -> lot %d x %.2f, target $%.2f, loss limit $%.2f ===",
                 srv(now).date(), acc.balance, self.cfg.legs, lot, target, loss)
        self.save(now)

    def closed_today(self, now):
        tickets = tuple(sorted(p.ticket for p in self.my_positions()))
        key, val, at = self.closed_cache
        if key == tickets and now - at < 10:
            return val
        day0 = (now // 86400) * 86400
        deals = mt5.history_deals_get(srv(day0 - 86400), srv(now + 86400)) or ()
        val = sum(dl.profit + dl.swap + dl.commission for dl in deals
                  if dl.magic == self.cfg.magic and dl.symbol == self.cfg.symbol and dl.time >= day0
                  and dl.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT))
        self.closed_cache = (tickets, val, now)
        return val

    def floating(self):
        return sum(p.profit + p.swap for p in self.my_positions())

    def update_daily_pl(self, now):
        """Returns True when the day is finished (the EA's early return)."""
        if self.s["limit"]:
            return True
        self.daily_pl = self.closed_today(now) + self.floating()
        if self.daily_pl >= self.s["target"]:
            self.s["limit"] = True
            self.s["limit_reason"] = f"Daily Profit Target Reached: ${self.daily_pl:.2f}" + (
                f" [BOOSTED target ${self.s['target']:.2f}]" if self.s["boost"] else "")
            log.info("daily profit target reached: %.2f", self.daily_pl)
            if self.s["basket"]: self.close_basket(now, "Daily profit target reached")
            self.save(now)
        if self.daily_pl <= -self.s["loss_limit"]:
            self.s["limit"] = True
            self.s["limit_reason"] = f"Daily Loss Limit Reached: ${self.daily_pl:.2f}"
            log.warning("daily loss limit reached: %.2f", self.daily_pl)
            if self.s["basket"]: self.close_basket(now, "Daily loss limit reached")
            self.save(now)
        return self.s["limit"]

    # ------------------------------------------------------------------ drawdown & excursion tracking
    def update_drawdowns(self):
        acc = mt5.account_info(); eq = acc.equity; s = self.s
        for pk, dv, dp in (("eq_peak", "dd", "dd_pct"), ("eq_peak_day", "dd_day", "dd_day_pct")):
            s[pk] = max(s[pk], eq); dd = s[pk] - eq
            if dd > s[dv]:
                s[dv] = dd; s[dp] = dd / s[pk] * 100 if s[pk] > 0 else 0
        b = s["basket"]
        if b:
            f = self.floating(); b["mfe"] = max(b["mfe"], f); b["mae"] = min(b["mae"], f)

    # ------------------------------------------------------------------ signal & filters
    def try_signal(self, tick, now):
        c = self.cfg
        if now - self.s["last_signal"] < 60:
            return
        m5 = self.rates(5, 1000)
        sig, rsi_cur, rsi = rsi_signal(m5["close"], c)
        if sig == 0:
            return
        self.s["last_signal"] = now
        persist = self.persist_min(now)
        if self.s["regime_elev"] and persist < c.min_regime_persistence_minutes:
            log.info("signal %s blocked by ATR filter (ratio %.3f, regime age %.0f min)", SIDE[sig], self.ratio, persist); return
        xb = rsi_extreme_bars(rsi, sig == 1, c)
        if xb > c.rsi_extreme_max:
            log.info("signal %s blocked by RSI extreme bars (%d)", SIDE[sig], xb); return
        m3, m1 = self.rates(3, 60), self.rates(1, 60)
        t1 = trend_live(m1["high"], m1["low"], m1["close"], c.swing_bars)
        t3 = trend_live(m3["high"], m3["low"], m3["close"], c.swing_bars)
        t5 = trend_live(m5["high"][-60:], m5["low"][-60:], m5["close"][-60:], c.swing_bars)
        if not (t3 >= 0 if sig == 1 else t3 <= 0):
            log.info("signal %s blocked by M3 trend filter", SIDE[sig]); return
        direction = -sig
        ema = ema_last(self.rates(60, 1000)["close"], c.ema_period)
        if not (tick.bid > ema if direction == 1 else tick.bid < ema):
            log.info("reverse %s blocked by H1 EMA gate", SIDE[direction]); return
        if self.poc_bias != direction:
            log.info("reverse %s blocked by POC filter", SIDE[direction]); return
        dow_mql = (srv(now).weekday() + 1) % 7
        acc = mt5.account_info()
        boost = boost_ok(persist, dow_mql, xb, c)
        lot, target, loss = self.sizing(self.s["start_balance"], boost)     # re-decided at every entry, like the EA
        self.s.update(lot=lot, target=target, loss_limit=loss, boost=boost)
        entry_px = tick.ask if direction == 1 else tick.bid
        snap = dict(orig=SIDE[sig], dir=SIDE[direction], hour=srv(now).hour, dow=dow_mql, rsi=rsi_cur, xb=xb,
                    ratio=self.ratio, elev=self.s["regime_elev"], persist=persist, t1=TREND_NAME[t1], t2=TREND_NAME[t5],
                    t3=TREND_NAME[t3], price=entry_px, ema=ema, ema_pts=(entry_px - ema) / self.point,
                    poc_bias=TREND_NAME[self.poc_bias], poc=self.poc_price, daily_pl=self.daily_pl,
                    equity=acc.equity, balance=acc.balance, wins=self.s["wins"], losses=self.s["losses"],
                    dd=self.s["dd"], dd_pct=self.s["dd_pct"], boost=boost, lot=lot, target=target, loss_limit=loss)
        log.info("=== SIGNAL: RSI %s -> reverse %s | %d x %.2f lot, target $%.2f, loss limit $%.2f%s ===", SIDE[sig],
                 SIDE[direction], c.legs, lot, target, loss, " [BOOSTED]" if boost else "")
        if self.dry:
            log.info("DRY-RUN: would open %s basket at %.3f", SIDE[direction], entry_px); self.save(now); return
        self.open_basket(direction, snap, now)

    # ------------------------------------------------------------------ orders
    def send(self, req):
        for attempt in range(self.cfg.trade_send_retries + 1):
            r = mt5.order_send(req)
            if r is not None and (r.retcode in self.OK or r.order > 0 or r.deal > 0):
                return r
            code = None if r is None else r.retcode
            log.error("order failed (attempt %d): %s %s", attempt + 1, code, mt5.last_error() if r is None else r.comment)
            if code not in self.RETRY:
                return None
            time.sleep(0.2)
        return None

    def emergency(self, direction, px):
        c = self.cfg; sl = tp = 0.0
        stops = self.info.trade_stops_level
        if c.emergency_sl_points > 0:
            d = max(c.emergency_sl_points, stops + 1) * self.point
            sl = round(px - d if direction == 1 else px + d, self.info.digits)
        if c.emergency_tp_points > 0:
            d = max(c.emergency_tp_points, stops + 1) * self.point
            tp = round(px + d if direction == 1 else px - d, self.info.digits)
        return sl, tp

    def open_basket(self, direction, snap, now):
        for n in range(1, self.cfg.legs + 1):
            t = mt5.symbol_info_tick(self.cfg.symbol)
            px = t.ask if direction == 1 else t.bid
            sl, tp = self.emergency(direction, px)
            req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.cfg.symbol, volume=self.s["lot"],
                       type=mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL, price=px, sl=sl, tp=tp,
                       deviation=self.cfg.slippage_points, magic=self.cfg.magic, comment=f"Reverse{n}",
                       type_time=mt5.ORDER_TIME_GTC, type_filling=self.filling)
            self.send(req); time.sleep(0.1)
        time.sleep(0.3)
        pos = self.my_positions()                       # trust the terminal, not the return codes
        if len(pos) < self.cfg.legs:
            log.warning("only %d of %d legs opened - closing them", len(pos), self.cfg.legs)
            for p in pos: self.close_position(p.ticket)
            return
        self.s["basket"] = dict(tickets=sorted(p.ticket for p in pos), dir=direction,
                                open_time=min(p.time for p in pos), mfe=0.0, mae=0.0, snap=snap)
        self.s["trade_no"] += 1
        log.info("basket #%d opened: %s %s", self.s["trade_no"], SIDE[direction], self.s["basket"]["tickets"])
        self.save(now)

    def close_position(self, ticket):
        for _ in range(self.cfg.trade_send_retries + 1):
            ps = mt5.positions_get(ticket=ticket)
            if not ps:
                return True
            p = ps[0]; t = mt5.symbol_info_tick(p.symbol)
            req = dict(action=mt5.TRADE_ACTION_DEAL, position=ticket, symbol=p.symbol, volume=p.volume,
                       type=mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                       price=t.bid if p.type == mt5.POSITION_TYPE_BUY else t.ask, deviation=self.cfg.slippage_points,
                       magic=self.cfg.magic, type_time=mt5.ORDER_TIME_GTC, type_filling=self.filling)
            mt5.order_send(req); time.sleep(0.2)
            if not mt5.positions_get(ticket=ticket):
                return True
        log.error("could not close position %s", ticket)
        return False

    def close_basket(self, now, reason):
        b = self.s["basket"]
        log.info("closing basket %s: %s", b["tickets"], reason)
        for t in b["tickets"]:
            self.close_position(t)
        self.finish_basket(now, reason)

    def finish_basket(self, now, reason):
        b = self.s["basket"]; time.sleep(0.3)
        pl = 0.0
        for t in b["tickets"]:
            for dl in mt5.history_deals_get(position=t) or ():
                if dl.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                    pl += dl.profit + dl.swap + dl.commission
        self.write_trade_row(b, now, pl, reason)
        s = self.s; s["d_trades"] += 1
        if pl > 0: s["d_wins"] += 1; s["wins"] += 1; s["losses"] = 0
        elif pl < 0: s["d_losses"] += 1; s["losses"] += 1; s["wins"] = 0
        s["basket"] = None; self.closed_cache = (None, 0.0, 0)
        log.info("basket closed: P/L %.2f (%s)", pl, reason)
        self.save(now)

    # ------------------------------------------------------------------ CSV logs (same columns as the EA)
    def _append(self, fname, header, row):
        p = path_of(fname); new = not os.path.exists(p)
        if not new:                                   # a file with different columns is kept aside, not mixed
            with open(p, newline="") as f:
                first = f.readline().strip()
            if first != ",".join(header):
                os.replace(p, p.replace(".csv", f"_old_{int(time.time())}.csv")); new = True
        with open(p, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(header)
            w.writerow(row)

    def write_trade_row(self, b, now, pl, reason):
        sn = b.get("snap") or {}
        g = lambda k, d="": sn.get(k, d)             # noqa: E731
        self._append(self.cfg.trade_log, TRADE_HEADER, [
            fmt(b["open_time"]), fmt(now), int(now - b["open_time"]), self.cfg.symbol, g("orig", "ADOPTED"),
            g("dir", SIDE[b["dir"]]), g("hour"), g("dow"), f"{g('rsi', 0):.2f}", g("xb"), f"{g('ratio', 0):.3f}",
            str(g("elev", False)).upper(), f"{g('persist', 0):.1f}", g("t1"), g("t2"), g("t3"), g("price"),
            g("ema"), f"{g('ema_pts', 0):.1f}", g("poc_bias"), g("poc"), f"{g('daily_pl', 0):.2f}",
            f"{g('equity', 0):.2f}", f"{g('balance', 0):.2f}", g("wins"), g("losses"), f"{g('dd', 0):.2f}",
            f"{g('dd_pct', 0):.2f}", f"{b['mfe']:.2f}", f"{b['mae']:.2f}", f"{pl:.2f}", reason, self.s["trade_no"],
            str(g("boost", False)).upper(), f"{g('lot', 0):.2f}", f"{g('target', 0):.2f}", f"{g('loss_limit', 0):.2f}"])

    def write_daily_row(self, reason=None):
        s = self.s; acc = mt5.account_info()
        wr = s["d_wins"] / s["d_trades"] * 100 if s["d_trades"] else 0.0
        reason = reason or (s["limit_reason"] if s["limit"] else "Day ended (no limit hit)")
        self._append(self.cfg.daily_log, DAILY_HEADER, [
            srv(s["day"] * 86400).strftime("%Y.%m.%d"), f"{s['start_balance']:.2f}", f"{acc.equity:.2f}",
            f"{self.daily_pl:.2f}", f"{s['dd_day']:.2f}", f"{s['dd_day_pct']:.2f}", s["d_trades"],
            s["d_wins"], s["d_losses"], f"{wr:.1f}", str(s["limit"]).upper(), reason, str(s["boost"]).upper()])


def cmd_live(a, cfg):
    mt5_start(a.terminal)
    acc = mt5.account_info()
    if acc is None:
        sys.exit("no account logged in to the terminal")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not a.allow_live and not a.dry_run:
        sys.exit(f"account {acc.login} is not a demo account - refusing to trade (pass --allow-live to override)")
    if not mt5.symbol_select(cfg.symbol, True):
        sys.exit(f"symbol {cfg.symbol} not available: {mt5.last_error()}")
    if not a.dry_run and not mt5.terminal_info().trade_allowed:
        sys.exit("Algo Trading is disabled in the terminal - enable it (toolbar button) and restart")
    log.info("connected: account %s (%s) server %s | %s | sizing %s%s | magic %d | %s", acc.login,
             "DEMO" if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "REAL", acc.server, cfg.symbol,
             f"compounding {cfg.lot_per_1000:.2f} lot per $1,000" if cfg.compounding else f"fixed {cfg.legs} x {cfg.lot:.2f}",
             " + boost" if cfg.boost else "", cfg.magic, "DRY-RUN" if a.dry_run else "TRADING")
    bot = Bot(cfg, a.dry_run); last = None
    try:
        while True:
            tick = mt5.symbol_info_tick(cfg.symbol)
            if tick is None:
                log.warning("no tick (%s) - reconnecting", mt5.last_error()); time.sleep(5)
                mt5.shutdown(); (mt5.initialize(a.terminal) if a.terminal else mt5.initialize()); continue
            if tick.time_msc != last:
                last = tick.time_msc
                try:
                    bot.on_tick(tick)
                except RuntimeError as e:
                    log.warning("skipped tick: %s", e)
            time.sleep(cfg.poll_seconds)
    except KeyboardInterrupt:
        log.info("stopping (Ctrl+C) - open positions stay open and are re-adopted on the next start")
        bot.write_daily_row("BOT_STOPPED"); bot.save(bot.now)
    finally:
        mt5.shutdown()


def cmd_fetch(a, cfg):
    d = load_history(cfg, pd.Timestamp(a.start or "2021-08-01").to_pydatetime(), update=True, refresh=a.refresh,
                     terminal=a.terminal)
    print(f"{len(d['tm']):,} M1 bars, {pd.Timestamp(d['ts'][0])} -> {pd.Timestamp(d['ts'][-1])}")


# =====================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol", help="default XAUUSDm")
    common.add_argument("--terminal", help="path to terminal64.exe if more than one MT5 is installed")
    common.add_argument("--lot", type=float, help="fixed lot per leg (daily target = lot x $1,000 unless --target)")
    common.add_argument("--target", type=float, help="daily profit target $ (fixed sizing)")
    common.add_argument("--loss-limit", type=float, help="daily loss limit $ (fixed sizing)")
    common.add_argument("--compound", action="store_true", help="size from the day-start balance (0.10 lot / $1,000), boost off")
    common.add_argument("--no-boost", action="store_true", help="turn the EA's x2 boost off")
    common.add_argument("--magic", type=int)
    lv = sub.add_parser("live", parents=[common], help="trade through the terminal")
    lv.add_argument("--dry-run", action="store_true", help="log signals and decisions, never send orders")
    lv.add_argument("--allow-live", action="store_true", help="permit trading on a real (non-demo) account")
    bt = sub.add_parser("backtest", parents=[common], help="backtest on M1 history pulled from the terminal")
    bt.add_argument("--from", dest="start", help="first trading day, e.g. 2025-01-01 (default: all history)")
    bt.add_argument("--to", dest="end", help="last day (exclusive)")
    bt.add_argument("--balance", type=float, default=1000.0, help="starting balance (matters for --compound)")
    bt.add_argument("--spread-k", type=float, default=1.0, help="multiply the recorded spread (stress test)")
    bt.add_argument("--warmup-days", type=int, default=60, help="history fetched before --from for the indicators")
    bt.add_argument("--no-update", action="store_true", help="use the cached history only, do not open the terminal")
    bt.add_argument("--refresh", action="store_true", help="throw the cache away and download again")
    bt.add_argument("--csv", help="use an MT5 'Export bars' M1 file instead of the terminal")
    bt.add_argument("--out", default="GMStandalone_backtest_trades.csv")
    ft = sub.add_parser("fetch", parents=[common], help="download / update the cached M1 history")
    ft.add_argument("--from", dest="start"); ft.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    if a.symbol: cfg.symbol = a.symbol
    if a.magic: cfg.magic = a.magic
    if a.lot:
        cfg.lot = a.lot; cfg.daily_target = a.lot * cfg.target_per_lot
    if a.target: cfg.daily_target = a.target
    if a.loss_limit: cfg.daily_loss_limit = a.loss_limit
    if a.compound: cfg.compounding = True; cfg.boost = False
    if a.no_boost: cfg.boost = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(path_of(cfg.log_file))] if a.cmd == "live" else []))
    {"live": cmd_live, "backtest": cmd_backtest, "fetch": cmd_fetch}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
