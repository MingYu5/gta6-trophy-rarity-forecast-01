"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Two regimes, chosen by what the case's history shows.

1. Moving history (the final case: days of reads with real dynamics).
   For each trophy, fit a line of logit(rate) on log(1 + hours since the
   first read) over the last 24 hours of reads -- the current regime, not
   the launch-night one -- shrink the slope toward the cross-trophy median
   slope, and read the line at the target.  The fitted drift from the last
   rate is damped by D = span / (span + extrapolation): a trend measured
   over one day does not run unchanged for four more, growth saturates, so
   only part of the drift is kept.  The spread combines the fit's residual
   and leverage with a horizon term that widens like sqrt(days), because the
   trend itself is the uncertain part over a long horizon.

2. Flat history (the practice cases: every read identical).  The page is a
   stale snapshot and Exophase reveals its accumulated syncs in jumps, so a
   structural prior is used instead: a trophy's rate is holders / players,
   synced players accumulate roughly linearly in time, so the count the
   next refresh reveals is the first read's count times the target's time
   axis, anchored at the first read (the refresh reveals the whole stale
   period's syncs, not just the horizon's worth); holders are assumed to
   grow by a small prior factor over the horizon.  The spread is wide,
   because the refresh size is genuinely uncertain.

A trophy every tracked player already holds stays at ~100%.

Standard library only, no network, deterministic: the same case always
writes the same bytes.  Reads every *.in in the input directory (the
grader runs one case at a time) and writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants ---
# Window for the trend fit: the last WINDOW_HOURS of reads.
WINDOW_HOURS = 24.0
# Weight given to a trophy's own fitted slope (rest goes to the
# cross-trophy median slope).
OWN_SLOPE_WEIGHT = 0.7
# Horizon spread: the logit spread grows by HORIZON_SIGMA per sqrt(day).
HORIZON_SIGMA = 0.45
# Floor / cap on the total logit spread, and the irreducible base term.
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.0
SIGMA_BASE = 0.15
# Flat-history prior: synced players accumulate roughly linearly in time
# since the recording began (see the module docstring).
BETA_ANCHOR = 1.0
# Log-player uncertainty when the history is flat.
SIGMA_LOGP_FLAT = 0.8
# Prior holder growth over the horizon when the history is flat, as a
# fraction of current holders, scaled by sqrt(horizon / 1.5 h), capped.
HOLDER_PRIOR_RATE = 0.3
HOLDER_PRIOR_CAP = 1.5
# Log-holder uncertainty when the history is flat.
SIGMA_LOGH_FLAT = 0.4
# A trophy at (essentially) 100% is forecast as staying there, with a small
# hedge on the low quantile in case a newcomer lacks it.
FULL_RATE = 99.99
FULL_Q05 = 99.0
FULL_Q25 = 99.9


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_std, n, x_mean, sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, n, mx, sxx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1) if n > 2 else 0.0
    return a, b, math.sqrt(var), n, mx, sxx


def quantiles(median_pct: float, sigma: float):
    """Five quantiles around a median percent with a logit spread."""
    lg = logit(median_pct)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
    qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    t0 = parse_time(history[0]["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_h = case["horizon_minutes"] / 60.0

    times = [parse_time(read["at"]) for read in history]
    xs = [math.log1p(max((t - t0).total_seconds() / 3600.0, 0.0)) for t in times]
    x_last = xs[-1]
    x_target = math.log1p(max((target - t0).total_seconds() / 3600.0, 0.0))
    players = [read["players_tracked"] for read in history]
    p_last = players[-1]

    # Last non-null rate per trophy.
    last_rates = []
    for i in range(n_tr):
        r = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                r = read["rates"][i]
                break
        last_rates.append(r if r is not None else 0.0)

    # Does anything in the history move at all?
    moves = len(set(players)) >= 3
    if not moves:
        for i in range(n_tr):
            seen = {read["rates"][i] for read in history if read["rates"][i] is not None}
            if len(seen) >= 3:
                moves = True
                break

    if not moves:
        return _forecast_flat(case, name, history, trophies, last_rates,
                              xs, x_last, x_target, players, p_last, horizon_h)

    # --- moving history: direct logit trend per trophy, last 24 h ---
    window_start = times[-1].timestamp() - WINDOW_HOURS * 3600
    widx = [j for j, t in enumerate(times) if t.timestamp() >= window_start]
    if len(widx) < 3:
        widx = list(range(len(history)))
    span = xs[widx[-1]] - xs[widx[0]] if len(widx) > 1 else 1.0
    damp = span / (span + max(x_target - x_last, 0.0)) if span > 0 else 0.0
    horizon_term = HORIZON_SIGMA * math.sqrt(max(horizon_days, 0.0))

    slopes = []
    fits = {}
    for i in range(n_tr):
        pts = [(xs[j], logit(history[j]["rates"][i])) for j in widx
               if history[j]["rates"][i] is not None]
        if len(pts) >= 3 and len({round(v, 9) for _, v in pts}) >= 3:
            a, b, resid, n, mx, sxx = ols([p[0] for p in pts], [p[1] for p in pts])
            fits[i] = (a, b, resid, n, mx, sxx)
            slopes.append(b)
    median_slope = sorted(slopes)[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= FULL_RATE:
            rows.append({"id": trophy["id"],
                         "q": [FULL_Q05, FULL_Q25, FULL_RATE, FULL_RATE, FULL_RATE]})
            continue
        if i in fits:
            a, b, resid, n, mx, sxx = fits[i]
            b_use = OWN_SLOPE_WEIGHT * b + (1 - OWN_SLOPE_WEIGHT) * median_slope
            drift = (a + b_use * x_target) - logit(r_last)
            med = inv_logit(logit(r_last) + damp * drift)
            lev = 1 + 1 / n + (x_target - mx) ** 2 / sxx if sxx > 0 else 1.0
            sig_fit = min(1.2, max(0.25, resid * math.sqrt(lev)))
            sigma = math.sqrt(sig_fit ** 2 + horizon_term ** 2 + SIGMA_BASE ** 2)
        else:
            # This trophy did not move in the window: stay at its last rate
            # with a wide band.
            med = r_last
            sigma = math.sqrt(0.25 ** 2 + horizon_term ** 2 + SIGMA_BASE ** 2)
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def _forecast_flat(case, name, history, trophies, last_rates,
                   xs, x_last, x_target, players, p_last, horizon_h):
    """Flat history: structural holders/players prior (see module docstring)."""
    logp = [math.log(max(p, 1)) for p in players]
    # The count the next refresh reveals, anchored at the first read.
    logp_star = logp[0] + BETA_ANCHOR * x_target
    logp_star = max(logp_star, logp[-1])
    logp_star = min(logp_star, logp[-1] + math.log(50))
    p_star = math.exp(logp_star)

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= FULL_RATE:
            rows.append({"id": trophy["id"],
                         "q": [FULL_Q05, FULL_Q25, FULL_RATE, FULL_RATE, FULL_RATE]})
            continue
        h_last = max(r_last / 100.0 * p_last, 0.5)
        growth = min(HOLDER_PRIOR_RATE * math.sqrt(max(horizon_h, 0.0) / 1.5),
                     HOLDER_PRIOR_CAP)
        h_star = h_last * (1 + growth)
        med = 100.0 * h_star / p_star
        sigma = math.sqrt(SIGMA_LOGP_FLAT ** 2 + SIGMA_LOGH_FLAT ** 2
                          + SIGMA_BASE ** 2)
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
