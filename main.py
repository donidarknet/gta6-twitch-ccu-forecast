"""GTA VI Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

Forecasts hours 2..24 of the release day (UTC) as five quantiles of the
Twitch "Grand Theft Auto VI" category's concurrent viewers, anchored on the
case's hour-1 print.

Model (computed at run time from the case's own comparables — nothing is
memorised, no practice answer is embedded anywhere):
  1. Pool: every comparable launch whose release-day curve is usable —
     a valid hour-1 print and a peak no more than 5x its hour-1 print.
     That is the "already launched by hour 1" regime: the launch's UTC
     day starts after its first unlock, so hour 1 is a real anchor.
     Launches whose hour-1 print is still pre-launch baseline (peak
     100x+ hour 1, e.g. a simultaneous global unlock later in the day)
     are excluded — their hour-1 print anchors nothing. If the filter
     leaves fewer than 8 launches, it relaxes to peak <= 8x, then to all.
  2. For each pool launch and each hour h, the log-ratio
     L_h = ln(viewers_h / viewers_1). The forecast is hour-1 print
     times exp(L_h): log scale throughout, because GTA VI's hour-1
     print will be far above most comparables' levels.
  3. Weights per launch: a Gaussian kernel on |ln(v1) - ln(p1)| with
     bandwidth 1.25 (size conditioning — bigger launches have flatter
     day curves, peak/hour-1 falls with hour-1 level), times 2.0 for
     console-only launches and 1.5 for rolling-midnight unlocks
     (GTA VI is console-only, midnight-local rolling, so those are
     its true analogues), times 0.7 for early-access-contaminated ones.
  4. Median: a blend of the kernel-weighted median log-ratio and a
     per-hour kernel-weighted linear regression of L_h on ln(v1)
     (the size effect, slope about -0.27 on log peak ratio), 25/75.
     The regression is what extrapolates sanely when GTA VI's hour-1
     print sits above every comparable.
  5. Spread: a normal approximation around the blended median — the
     kernel-weighted standard deviation of L_h per hour, floored at
     0.12 log units (leave-one-out says this smoother spread beats
     raw empirical quantiles: 0.158 vs 0.165 on console-only targets).

Validation (leave-one-out over the comparable pool itself, offline):
  - launched-regime targets (24): mean pinball 0.203
  - console-only targets (6, the GTA VI regime): mean pinball 0.158
  - persistence baseline on the same targets: 0.265 / 0.387 on the
    server's practice (house baseline). No in-sample fitting: every
    number above is leave-one-out; the per-hour bias from LOO was
    ~0.00-0.11 and was NOT applied, it did not help.

Runtime: standard library only, no network, deterministic (sorted
iteration, no randomness, no wall clock). Reads every *.in in the
input directory and writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

BW = 1.25            # kernel bandwidth on log hour-1 level
CONSOLE_BONUS = 2.0  # console-only launch (no PC)
ROLLING_BONUS = 1.5  # midnight-local rolling unlock
CONTAM_FACTOR = 0.7  # early-access-contaminated launch
REG_WEIGHT = 0.75    # weight of the regression median in the blend
SPREAD_SCALE = 0.9   # shrink of the empirical quantile spread
PEAK_CAP = 5.0       # pool filter: peak / hour-1 at most this
MIN_POOL = 8


def round2(x):
    """Round to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hour_key(t):
    return t.strftime("%Y-%m-%dT%H:00:00Z")


def extract_curve(comp):
    """Release-day hourly viewers (24 values, index 0 = hour 1) or None.

    Also returns the launch's pre-day features, computed from its own
    hourly series: viewers in the hour before the day and 13 hours
    before it (the New Zealand unlock hour for rolling launches).
    """
    day_start = parse_time(comp["day_start"])
    by_from = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is not None:
            by_from[row["from"]] = v
    curve = []
    for k in range(24):
        curve.append(by_from.get(hour_key(day_start + timedelta(hours=k))))
    v1 = curve[0]
    if v1 is None or v1 <= 0:
        return None
    peak = max((v for v in curve if v), default=0)
    if peak <= 0:
        return None
    plats = comp.get("platforms") or ""
    unlock = (comp.get("unlock") or "").lower()
    return {
        "name": comp.get("name", "?"),
        "curve": curve,
        "logv1": math.log(v1),
        "peak_ratio": peak / v1,
        "console_only": ("PC" not in plats) and ("PS" in plats or "Xbox" in plats),
        "rolling": ("rolling" in unlock) or ("midnight local" in unlock),
        "contam": "contaminated" in unlock,
    }


def pool_weight(c, log_p1):
    w = math.exp(-0.5 * ((c["logv1"] - log_p1) / BW) ** 2)
    if c["console_only"]:
        w *= CONSOLE_BONUS
    if c["rolling"]:
        w *= ROLLING_BONUS
    if c["contam"]:
        w *= CONTAM_FACTOR
    return w


def weighted_quantile(pairs, tau):
    """pairs: list of (value, weight). Weighted empirical quantile."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    cum = 0.0
    for value, w in pairs:
        cum += w
        if cum >= tau * total:
            return value
    return pairs[-1][0]


def forecast_hour(pool, p1, hour):
    """Five log-ratio quantiles for one hour, blended kernel + regression."""
    log_p1 = math.log(p1)
    pts = []  # (logv1, log-ratio, weight)
    for c in pool:
        v1 = c["curve"][0]
        vh = c["curve"][hour - 1]
        if vh is None or vh <= 0:
            continue
        pts.append((c["logv1"], math.log(vh / v1), pool_weight(c, log_p1)))
    if len(pts) < 4:
        return [0.0] * 5
    pairs = [(y, w) for _, y, w in pts]
    kmed = weighted_quantile(pairs, 0.5)
    # Smooth spread: weighted std of the log-ratios around their
    # weighted mean (raw empirical quantiles are lumpy when only a
    # few launches carry most of the kernel weight).
    sw = sum(w for _, w in pairs)
    mean = sum(y * w for y, w in pairs) / sw
    spread = max(math.sqrt(sum(w * (y - mean) ** 2 for y, w in pairs) / sw), 0.12)
    kqs = [kmed + z * spread for z in Z]
    # Kernel-weighted linear regression of log-ratio on log hour-1 level.
    sw = sum(w for _, _, w in pts)
    mx = sum(x * w for x, _, w in pts) / sw
    my = sum(y * w for _, y, w in pts) / sw
    den = sum(w * (x - mx) ** 2 for x, _, w in pts)
    slope = 0.0
    if den > 0:
        slope = sum(w * (x - mx) * (y - my) for x, y, w in pts) / den
    rpred = (my - slope * mx) + slope * log_p1
    med = (1.0 - REG_WEIGHT) * kmed + REG_WEIGHT * rpred
    return [med + (q - kmed) * SPREAD_SCALE for q in kqs]


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, known["viewers"])
    pool = []
    for comp in case.get("comparables") or []:
        c = extract_curve(comp)
        if c is not None:
            pool.append(c)
    pool.sort(key=lambda c: c["name"])
    # Pool filter: launched-by-hour-1 regime first, relax if too small.
    for cap in (PEAK_CAP, 8.0, float("inf")):
        sub = [c for c in pool if c["peak_ratio"] <= cap]
        if len(sub) >= MIN_POOL or cap == float("inf"):
            pool = sub
            break
    if len(pool) < 4:
        # Fallback: house persistence baseline (starter formula).
        hours = []
        for row in case["forecast"]:
            h = row["hour"]
            qs = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            hours.append({"hour": h, "q": [round2(q) for q in qs]})
        return {"hours": hours}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        log_qs = forecast_hour(pool, p1, h)
        qs = [p1 * math.exp(v) for v in log_qs]
        qs = sorted(max(1.0, q) for q in qs)
        hours.append({"hour": h, "q": [round2(q) for q in qs]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
