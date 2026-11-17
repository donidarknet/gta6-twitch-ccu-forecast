# GTA VI Twitch CCU forecaster

Forecasts hours 2–24 of Grand Theft Auto VI's release day (2026-11-19 UTC)
as five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch category's
concurrent viewers, anchored on the hour-1 print. Entry for make.fun
`gta6-twitch-ccu-forecast-01`. One file, `main.py` (Python 3.13, standard
library only, offline, deterministic).

## Model

Computed at run time from the case's own comparables — no practice
answer is embedded anywhere, nothing is memorised:

1. **Pool.** Every comparable launch with a valid hour-1 print whose
   release-day peak is at most 5x its hour-1 print: the "already launched
   by hour 1" regime. Launches whose hour-1 print is still pre-launch
   baseline (peak 100x+ hour 1, e.g. a simultaneous global unlock later
   in the day) are excluded — their hour-1 print anchors nothing.
   The filter relaxes (8x, then all) if fewer than 8 launches remain.
2. **Log ratios.** For each pool launch and hour `h`,
   `L_h = ln(viewers_h / viewers_1)`. The forecast is the hour-1 print
   times `exp(L_h)`: log scale throughout, because GTA VI's hour-1
   print will sit far above most comparables' levels.
3. **Weights.** Gaussian kernel on `|ln(v1) - ln(p1)|`, bandwidth 1.25
   (size conditioning — bigger launches have flatter day curves),
   times 2.0 for console-only launches and 1.5 for rolling-midnight
   unlocks (GTA VI is console-only, midnight-local rolling), times 0.7
   for early-access-contaminated launches.
4. **Median.** A 25/75 blend of the kernel-weighted median log-ratio
   and a per-hour kernel-weighted linear regression of `L_h` on
   `ln(v1)` (the size effect: log peak/hour-1 falls with hour-1 level,
   slope about -0.27). The regression is what extrapolates sanely when
   GTA VI's hour-1 print sits above every comparable.
5. **Spread.** Normal approximation around the blended median: the
   kernel-weighted standard deviation of `L_h` per hour, floored at
   0.12 log units, scaled by 0.9.

## Validation (leave-one-out, over the comparable pool itself)

| Targets | Mean pinball loss |
|---|---|
| Launched-regime launches (24) | **0.203** |
| Console-only launches (6, the GTA VI regime) | **0.158** |
| Persistence baseline, same targets | 0.265 |

No in-sample fitting: every number above is leave-one-out. A per-hour
median-bias correction from LOO residuals was tried and did not help,
so it is not applied. Practice lookup / memorised practice answers
were rejected outright — they score ~0.000 on practice and cannot
transfer to the unseen final case.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `*.out` per case:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Deterministic (two runs byte-identical), ~0.04 s per case.
