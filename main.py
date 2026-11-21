"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model: a trophy's rate is earners / players_tracked, and the two move for
different reasons, so they are forecast separately and combined:

  * players_tracked(t): log-linear in log-hours since release, fitted on
    the distinct refresh points of the history (consecutive identical reads
    are one refresh level, not new information), anchored at the last read.
  * earners_i(t) = rate_i(t) * players(t) / 100: log-linear in log-hours
    since release per trophy, slope shrunk toward the cross-trophy median
    slope, anchored at the trophy's last observed earners.
  * rate_target = 100 * earners_target / players_target, clamped.

Both fits use only the recent window of refresh points (the launch rush's
slope is not the next days' slope) and their forward slopes are damped
with horizon, because launch growth keeps decelerating; a whole-history
fit overshoots a saturating curve badly (measured on the synthetic
backtest in tests/validate.py).

Exophase refreshes a game's counts on its own schedule, hours apart, so a
history can be perfectly flat for hours and then jump. The spread therefore
combines a random-walk term, the fits' residual uncertainty, a refresh
timing term, and (when the history shows no movement at all) a refresh
prior: a refresh dilutes rates because newly tracked players start with
about zero trophies, and it dilutes rare trophies hardest.

Standard library only. Deterministic: no randomness, no dict-order or
clock dependence, files processed in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a rate's logit move, per square-root day. Smaller
# than the house baseline's 0.5: the trend fit already removes the drift
# the baseline's walk has to cover, and the backtest's coverage sits above
# 0.9 even at 0.3.
SIGMA_RW_PER_SQRT_DAY = 0.3
# Floor / cap on the total logit spread of any trophy.
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.5
# Refresh timing uncertainty, in days: the answer is the page as of the
# last refresh before the target, and refreshes are hours apart.
REFRESH_TIMING_DAYS = 0.15
# Fit uncertainty assumed when a trophy has too few points for residuals.
SIGMA_FIT_DEFAULT = 0.5
# Spread used when the history shows no movement at all (flat reads).
SIGMA_FLAT = 0.9
# Cap on the players multiplier over the horizon, per day of horizon.
PLAYERS_GROWTH_CAP_PER_DAY = 8.0
# Trend fits use only the recent window of distinct refresh points: early
# launch growth is far steeper than what the next days look like, and a
# fit over the whole history overshoots a saturating curve badly.
RECENT_WINDOW_HOURS = 36.0
# Minimum distinct points for a recent-window fit; fewer falls back to
# all points.
RECENT_MIN_POINTS = 4
# Forward slopes decay with horizon (growth keeps decelerating): a slope
# is multiplied by 1 / (1 + SLOPE_DAMP * horizon_days). Tuned on the
# synthetic backtest (tests/validate.py) across seeds and two launch
# regimes: 1.0 is near-best on the mix and on the final's 4.5-day shape,
# keeping about a fifth of the fitted trend at that horizon.
SLOPE_DAMP = 1.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (slope, intercept, resid_std, n, xbar, sxx)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, ybar, 0.0, n, xbar, 0.0
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    if n > 2:
        resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid_std = math.sqrt(resid / (n - 2))
    else:
        resid_std = 0.0
    return slope, intercept, resid_std, n, xbar, sxx


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def distinct_points(history):
    """Collapse consecutive identical reads into one refresh point each.

    A read is new information only when the players count or some rate
    changed; a run of identical reads is one stale page, not a trend.
    Returns a list of (epoch_seconds, players, rates_tuple).
    """
    points = []
    prev_key = None
    for read in history:
        players = read.get("players_tracked")
        rates = tuple(read.get("rates") or [])
        key = (players, rates)
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read["at"])
        except (KeyError, ValueError, TypeError):
            continue
        points.append((t, players, rates))
    return points


def flat_prior_shift(rate):
    """Median logit shift a refresh is expected to cause, by current rate.

    Newly tracked players start with about zero trophies, so a refresh
    dilutes every rate; common (early) trophies are re-earned fast by the
    new cohort, rare (late) ones are not, so the dilution grows as the
    rate falls. Calibrated on the launch-night refresh the competition
    documents (638 -> 3,868 players tracked, nearly every rate fell, the
    typical trophy's odds down several-fold).
    """
    if rate >= 50.0:
        return -0.15
    if rate >= 10.0:
        return -0.5
    if rate >= 1.0:
        return -1.2
    return -1.5


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target_t = parse_time(case["target"])

    # Time base: hours since the release day began, in log space. Growth
    # early in a launch is fast and then decelerates; log-time linearises it.
    release = case.get("game", {}).get("release_date")
    try:
        release_t = parse_time(release + "T00:00:00Z") if release else None
    except (ValueError, TypeError):
        release_t = None
    if release_t is None:
        release_t = parse_time(history[0]["at"]) if history else target_t

    def x_of(t):
        hours = max(0.0, (t - release_t) / 3600.0)
        return math.log1p(hours)

    x_target = x_of(target_t)

    if not history:
        # Nothing to learn from: a wide, low forecast for every trophy.
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * SIGMA_CAP)) for z in Z]})
        return {"trophies": rows}

    points = distinct_points(history)
    last_t, last_players, last_rates = points[-1]
    if not last_players or last_players <= 0:
        # The last read lacks a usable players count: fall back to the
        # latest read that has one, so the anchor never crashes.
        for (_t, p, _r) in reversed(points):
            if p and p > 0:
                last_players = p
                break
    if not last_players or last_players <= 0:
        last_players = 1
    x_last = x_of(last_t)
    dx = x_target - x_last

    # ---- players_tracked ----
    # Fit on the recent window (growth decelerates; the whole history's
    # slope is the launch rush, not the next days), blended with the full
    # fit, which is steadier for the smooth players series.
    p_times = [(x_of(t), math.log(p)) for (t, p, _r) in points if p and p > 0]
    if len(p_times) >= 2:
        xs = [p[0] for p in p_times]
        ys = [p[1] for p in p_times]
        slope_all, _ia, resid_p, _n, xbar_p, sxx_p = ols(xs, ys)
        recent = [(x, y) for (x, y) in p_times if x >= x_of(last_t - RECENT_WINDOW_HOURS * 3600)]
        if len(recent) >= RECENT_MIN_POINTS:
            slope_rec, _ir, _rr, _nr, _xb, _sx = ols([r[0] for r in recent], [r[1] for r in recent])
            slope_p = 0.5 * (slope_all + slope_rec)
        else:
            slope_p = slope_all
        slope_p *= 1.0 / (1.0 + SLOPE_DAMP * max(horizon_days, 0.0))
        log_players_target = math.log(last_players) + slope_p * dx
        cap = math.log(last_players) + math.log(PLAYERS_GROWTH_CAP_PER_DAY) * max(horizon_days, 0.0)
        log_players_target = min(log_players_target, cap)
        log_players_target = max(log_players_target, math.log(last_players))
        players_target = math.exp(log_players_target)
        # Players-fit uncertainty, shared by every trophy's rate.
        if len(p_times) > 2 and sxx_p > 0:
            lev = 1.0 + 1.0 / len(p_times) + (x_target - xbar_p) ** 2 / sxx_p
            sigma_players = min(1.0, resid_p * math.sqrt(lev))
        else:
            sigma_players = SIGMA_FIT_DEFAULT
        # Daily drift of log players near the target, for the timing term.
        hours_target = max(0.0, (target_t - release_t) / 3600.0)
        dx_per_day = 24.0 / (1.0 + hours_target)
        drift_players = abs(slope_p) * dx_per_day
    else:
        players_target = last_players
        slope_p = 0.0
        sigma_players = SIGMA_FIT_DEFAULT
        drift_players = 0.0

    # ---- per-trophy earners ----
    # First pass: raw per-trophy slopes on the recent window, for the
    # cross-trophy median slope used to shrink noisy trophies.
    n_trophies = len(trophies)
    raw_slopes = []
    trophy_series = []
    recent_x_min = x_of(last_t - RECENT_WINDOW_HOURS * 3600)
    for i in range(n_trophies):
        series = []
        for (t, p, rates) in points:
            if i < len(rates) and rates[i] is not None and p and p > 0:
                earners = rates[i] / 100.0 * p
                series.append((x_of(t), math.log(earners + 0.5), earners, t))
        trophy_series.append(series)
        fit_series = [s for s in series if s[0] >= recent_x_min]
        if len(fit_series) < RECENT_MIN_POINTS:
            fit_series = series
        if len(fit_series) >= 3:
            slope_i, _ii, _ri, _ni, _xb, _sx = ols([s[0] for s in fit_series], [s[1] for s in fit_series])
            raw_slopes.append(slope_i)
    median_slope = median(raw_slopes) if raw_slopes else 0.0

    flat = len(points) < 2
    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    hours_target = max(0.0, (target_t - release_t) / 3600.0)
    dx_per_day = 24.0 / (1.0 + hours_target)
    damp = 1.0 / (1.0 + SLOPE_DAMP * max(horizon_days, 0.0))

    rows = []
    for i, trophy in enumerate(trophies):
        series = trophy_series[i]
        # Last observed rate for this trophy (may predate the last read).
        last_rate = None
        for (_t, _p, rates) in reversed(points):
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 0.0

        if flat or len(series) < 2:
            # No movement to learn a trend from: persistence shifted by
            # the refresh prior, with a wide spread.
            shift = flat_prior_shift(last_rate)
            med_logit = logit(last_rate) + shift
            sigma = math.sqrt(SIGMA_FLAT ** 2 + sigma_rw ** 2)
        else:
            fit_series = [s for s in series if s[0] >= recent_x_min]
            if len(fit_series) < RECENT_MIN_POINTS:
                fit_series = series
            xs = [s[0] for s in fit_series]
            ys = [s[1] for s in fit_series]
            slope_i, _ii, resid_i, n_i, xbar_i, sxx_i = ols(xs, ys)
            # Shrink the noisy per-trophy slope toward the median trophy,
            # then damp it for the horizon: growth keeps decelerating.
            w = n_i / (n_i + 3.0)
            slope = (w * slope_i + (1.0 - w) * median_slope) * damp
            # Anchor at the last observed earners, not the fitted level.
            x_anchor, y_anchor, _e, _t = series[-1]
            log_e_target = y_anchor + slope * (x_target - x_anchor)
            earners_target = math.exp(log_e_target) - 0.5
            earners_target = min(max(earners_target, 0.0), players_target)
            med_rate = 100.0 * earners_target / players_target if players_target > 0 else last_rate
            med_logit = logit(med_rate)
            # Spread: random walk + earners fit + players fit + refresh timing.
            if n_i > 2 and sxx_i > 0:
                lev = 1.0 + 1.0 / n_i + (x_target - xbar_i) ** 2 / sxx_i
                sigma_fit = min(1.0, resid_i * math.sqrt(lev))
            else:
                sigma_fit = SIGMA_FIT_DEFAULT
            drift_rate = abs(slope * dx_per_day - drift_players)
            sigma_refresh = math.sqrt(0.15 ** 2 + (drift_rate * REFRESH_TIMING_DAYS) ** 2)
            sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_players ** 2 + sigma_refresh ** 2)

        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        qs = [round4(inv_logit(med_logit + z * sigma)) for z in Z]
        # Rounding can only break monotonicity by a hair; enforce it.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        qs = [min(100, max(0, q)) for q in qs]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
