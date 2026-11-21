"""Local validation for the trophy rarity forecaster (not part of the entry).

Runs four checks against main.py in the repo root:
  1. practice format / runtime / determinism (p00, p01 from the kit)
  2. practice scenario scoring: the real refreshed answers are not in the
     kit, so three plausible refresh scenarios are built from the totals
     the competition documents (3,868 players, 4,276 trophies earned) and
     our loss is compared with the starter's under each
  3. synthetic multi-day backtest: a generated 7-day recording with
     stepwise Exophase refreshes, cases cut at several checkpoints and
     horizons (6h, 24h, 4.5d), scored against the starter
  4. edge cases: 0% / 100% trophies, very short history, null rates

Usage: python3 tests/validate.py   (from the repo root)
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import main as model  # noqa: E402

KIT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-tools/kit/"
    "gta6-trophy-rarity-forecast-01-kit"
)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball_loss(forecast, actual_rates):
    """Mean pinball loss on the logit scale, exactly as the grader scores."""
    total = 0.0
    n = 0
    by_id = {row["id"]: row["q"] for row in forecast["trophies"]}
    for tid, y in actual_rates.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, by_id[tid]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n


def coverage(forecast, actual_rates):
    inside = 0
    by_id = {row["id"]: row["q"] for row in forecast["trophies"]}
    for tid, y in actual_rates.items():
        q = by_id[tid]
        if q[0] <= y <= q[4]:
            inside += 1
    return inside / len(actual_rates)


def starter_forecast(case):
    """The house baseline, re-implemented here for comparison only."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        rows.append(
            {
                "id": trophy["id"],
                "q": [model.round4(model.inv_logit(lg + z * spread)) for z in model.Z],
            }
        )
    return {"trophies": rows}


# ---------------------------------------------------------------- check 1
def check_practice_format():
    print("== 1. practice format / runtime / determinism ==")
    out1, out2 = "/tmp/trof-out1", "/tmp/trof-out2"
    env = dict(os.environ, MAKEFUN_INPUT_DIR=f"{KIT}/input")
    for out in (out1, out2):
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        env["MAKEFUN_OUTPUT_DIR"] = out
        subprocess.run([sys.executable, f"{ROOT}/main.py"], env=env, check=True)
        print(f"   run into {out}: {time.time() - t0:.2f}s")
    for name in ("p00", "p01"):
        a = open(f"{out1}/{name}.out", "rb").read()
        b = open(f"{out2}/{name}.out", "rb").read()
        assert a == b, f"{name} not deterministic"
        assert len(a) <= 65536, f"{name} too large: {len(a)}"
        d = json.loads(a)
        assert set(d) == {"trophies"} and len(d["trophies"]) == 55
        for row in d["trophies"]:
            q = row["q"]
            assert len(q) == 5 and all(0 <= x <= 100 for x in q)
            assert all(q[i] <= q[i + 1] for i in range(4))
        print(f"   {name}.out valid, {len(a)} bytes, byte-identical across runs")


# ---------------------------------------------------------------- check 2
def refresh_scenarios(case):
    """Plausible refreshed answers for a practice case.

    Documented totals after the refresh: 3,868 players tracked, 4,276
    trophies earned (from 638 players / 943 earned). Three ways the
    3,333 newly earned trophies could spread over the trophies:
      A: uniform earner growth (every trophy's earners x4.53)
      B: concentrated on early trophies (new earners ~ rate^2, capped at
         the new players; the 100% trophy stays 100%)
      C: middle (new earners ~ rate^1)
    """
    last = case["history"][-1]
    p_old = last["players_tracked"]
    e_old_total = last["trophies_earned"]
    p_new, e_new_total = 3868, 4276
    rates = last["rates"]
    earners_old = [r / 100.0 * p_old for r in rates]
    new_players = p_new - p_old
    new_earners_total = e_new_total - e_old_total

    out = {}
    # A: uniform growth
    f = e_new_total / e_old_total
    out["A uniform x4.53"] = {
        t["id"]: min(100.0, earners_old[i] * f / p_new * 100.0)
        for i, t in enumerate(case["trophies"])
    }
    # B and C: weighted spread of the new earners
    for name, power in (("B rate^2 (concentrated)", 2.0), ("C rate^1 (middle)", 1.0)):
        weights = [(r / 100.0) ** power for r in rates]
        # The 100% trophy: every new player earns it.
        alloc = [0.0] * len(rates)
        remaining = float(new_earners_total)
        for i, r in enumerate(rates):
            if r >= 100.0:
                alloc[i] = float(new_players)
                remaining -= new_players
        wsum = sum(weights[i] for i in range(len(rates)) if rates[i] < 100.0)
        for i in range(len(rates)):
            if rates[i] < 100.0:
                alloc[i] = remaining * weights[i] / wsum
        out[name] = {
            t["id"]: min(100.0, (earners_old[i] + alloc[i]) / p_new * 100.0)
            for i, t in enumerate(case["trophies"])
        }
    return out


def check_practice_scenarios():
    print("== 2. practice scenario scoring (vs starter) ==")
    for name in ("p00", "p01"):
        case = json.load(open(f"{KIT}/input/{name}.in"))
        ours = model.forecast_case(case)
        base = starter_forecast(case)
        for scen, actual in refresh_scenarios(case).items():
            lo, lb = pinball_loss(ours, actual), pinball_loss(base, actual)
            co, cb = coverage(ours, actual), coverage(base, actual)
            print(
                f"   {name} {scen}: ours {lo:.3f} (cov {co:.2f})  "
                f"starter {lb:.3f} (cov {cb:.2f})"
            )


# ---------------------------------------------------------------- check 3
def synthetic_recording(seed=7, days=7, regime="saturating"):
    """Generate a plausible 7-day Exophase recording.

    True players grow in log-time; true rates saturate per trophy with a
    per-trophy timescale (early trophies fast, late slow). What the
    recorder sees is stepwise: the page refreshes every ~4h and holds the
    previous refresh's values between refreshes; rates print to 2 decimals.

    Two regimes, because the real launch's shape is unknown:
      saturating: players level off near 400k by day 3 (persistence-friendly)
      growing: players still grow ~6x over the final horizon (power law),
               trophies saturate slower (a 40-hour game)
    """
    rng = random.Random(seed)
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    n_trophies = 55
    # Eventual rates: one 99.9%, a spread of early/mid/late trophies.
    eventual = [99.9]
    for _ in range(n_trophies - 1):
        eventual.append(10 ** rng.uniform(-1.0, 1.7))  # 0.1% .. 50%
    # Timescale in hours: common trophies fast, rare trophies slow.
    taus = [2.0]
    tau_hi = 2.3 if regime == "saturating" else 2.6
    for r in eventual[1:]:
        taus.append(10 ** rng.uniform(0.7, tau_hi) * (1.0 if r > 5 else 2.0))

    def true_players(hours):
        x = math.log1p(hours)
        if regime == "saturating":
            # Logistic in log-hours: 600 at launch, ~400k by day 7.
            return 600.0 + 400000.0 / (1.0 + math.exp(-(x - 3.4) * 2.2))
        # Power law, still growing hard at day 7.
        return min(2000000.0, 600.0 * (1.0 + hours) ** 1.5)

    def true_rate(i, hours):
        sat = 1.0 - math.exp(-hours / taus[i])
        return eventual[i] * sat

    reads = []
    refresh_every = timedelta(hours=4)
    next_refresh = release
    shown_players, shown_rates = None, None
    t = release - timedelta(minutes=10)
    end = release + timedelta(days=days)
    slot = 0
    while t < end:
        t += timedelta(minutes=10)
        slot += 1
        hours = (t - release).total_seconds() / 3600.0
        if t >= next_refresh:
            shown_players = true_players(hours)
            shown_rates = [true_rate(i, hours) for i in range(n_trophies)]
            next_refresh += refresh_every
        rates = [round(r, 2) for r in shown_rates]
        reads.append(
            {
                "slot": ts(t),
                "at": ts(t),
                "players_tracked": int(shown_players),
                "trophies_earned": int(sum(r / 100.0 * shown_players for r in rates)),
                "rates": rates,
            }
        )
    trophies = [
        {
            "id": i + 1,
            "name": f"Trophy {i + 1}",
            "description": "",
            "tier": "bronze",
            "points": 15,
            "secret": False,
        }
        for i in range(n_trophies)
    ]
    return release, trophies, reads, true_players, true_rate


def cut_case(release, trophies, reads, checkpoint_hours, horizon_hours):
    at = release + timedelta(hours=checkpoint_hours)
    target = at + timedelta(hours=horizon_hours)
    history = [r for r in reads if datetime.fromisoformat(r["at"].replace("Z", "+00:00")) < at]
    return {
        "id": f"syn-{checkpoint_hours}h-{horizon_hours}h",
        "at": ts(at),
        "horizon_minutes": int(horizon_hours * 60),
        "target": ts(target),
        "levels": LEVELS,
        "game": {
            "name": "Grand Theft Auto VI",
            "platform": "PS5",
            "release_date": "2026-11-19",
            "source": "https://example.invalid/",
        },
        "trophies": trophies,
        "history": history,
        "notes": "synthetic backtest",
    }


def actual_at(reads, target_dt):
    """First read at/after the target (the answering read)."""
    for r in reads:
        if datetime.fromisoformat(r["at"].replace("Z", "+00:00")) >= target_dt:
            return {i + 1: v for i, v in enumerate(r["rates"])}
    return None


def check_synthetic():
    print("== 3. synthetic multi-day backtest (vs starter) ==")
    cuts = [
        (12, 6),
        (24, 6),
        (48, 24),
        (48, 108),  # the final's shape: 2 days in, 4.5 days out
        (36, 72),
        (24, 48),
    ]
    grand_o = grand_b = n_cuts = 0
    for seed in (7, 11, 23):
        for regime in ("saturating", "growing"):
            release, trophies, reads, _tp, _tr = synthetic_recording(seed=seed, regime=regime)
            tot_o = tot_b = 0.0
            for cp, hz in cuts:
                case = cut_case(release, trophies, reads, cp, hz)
                target_dt = release + timedelta(hours=cp + hz)
                actual = actual_at(reads, target_dt)
                ours = model.forecast_case(case)
                base = starter_forecast(case)
                lo, lb = pinball_loss(ours, actual), pinball_loss(base, actual)
                co, cb = coverage(ours, actual), coverage(base, actual)
                tot_o += lo
                tot_b += lb
                print(
                    f"   seed {seed} {regime} +{cp}h/{hz}h: ours {lo:.3f} (cov {co:.2f})  "
                    f"starter {lb:.3f} (cov {cb:.2f})"
                )
            grand_o += tot_o
            grand_b += tot_b
            n_cuts += len(cuts)
            print(
                f"   seed {seed} {regime} MEAN: ours {tot_o / len(cuts):.3f}  "
                f"starter {tot_b / len(cuts):.3f}"
            )
    print(f"   GRAND MEAN: ours {grand_o / n_cuts:.3f}  starter {grand_b / n_cuts:.3f}")


# ---------------------------------------------------------------- check 4
def check_edge_cases():
    print("== 4. edge cases ==")
    base_trophies = [
        {"id": 1, "name": "A", "description": "", "tier": "platinum", "points": 300, "secret": False},
        {"id": 2, "name": "B", "description": "", "tier": "bronze", "points": 15, "secret": False},
        {"id": 3, "name": "C", "description": "", "tier": "bronze", "points": 15, "secret": False},
    ]

    def mkcase(history, horizon=1440):
        return {
            "id": "edge",
            "at": "2026-11-20T00:00:00Z",
            "horizon_minutes": horizon,
            "target": "2026-11-21T00:00:00Z",
            "levels": LEVELS,
            "game": {"name": "G", "platform": "PS5", "release_date": "2026-11-19", "source": ""},
            "trophies": base_trophies,
            "history": history,
            "notes": "",
        }

    def read(at, players, rates, earned=10):
        return {"slot": at, "at": at, "players_tracked": players, "trophies_earned": earned, "rates": rates}

    # 0% and 100% trophies, with movement
    hist = [
        read("2026-11-19T00:00:00Z", 100, [0.0, 100.0, 5.0]),
        read("2026-11-19T06:00:00Z", 200, [0.5, 100.0, 4.0]),
        read("2026-11-19T12:00:00Z", 400, [1.0, 99.5, 3.5]),
        read("2026-11-19T18:00:00Z", 800, [2.0, 99.0, 3.0]),
    ]
    out = model.forecast_case(mkcase(hist))
    print("   0%/100% trophies:", out["trophies"])
    for row in out["trophies"]:
        q = row["q"]
        assert all(0 <= x <= 100 for x in q) and all(q[i] <= q[i + 1] for i in range(4))

    # Very short history (2 reads, then 1 read)
    out = model.forecast_case(mkcase(hist[:2]))
    print("   2-read history:", out["trophies"][0])
    out = model.forecast_case(mkcase(hist[:1]))
    print("   1-read history:", out["trophies"][0])

    # Null rates scattered through
    hist_null = [
        read("2026-11-19T00:00:00Z", 100, [None, 100.0, 5.0]),
        read("2026-11-19T06:00:00Z", 200, [0.5, None, 4.0]),
        read("2026-11-19T12:00:00Z", 400, [1.0, 99.5, None]),
        read("2026-11-19T18:00:00Z", 800, [2.0, 99.0, 3.0]),
    ]
    out = model.forecast_case(mkcase(hist_null))
    print("   null rates:", out["trophies"])
    for row in out["trophies"]:
        q = row["q"]
        assert all(math.isfinite(x) for x in q)

    # Empty history
    out = model.forecast_case(mkcase([]))
    print("   empty history:", out["trophies"][0])
    print("   all edge cases valid")


if __name__ == "__main__":
    check_practice_format()
    check_practice_scenarios()
    check_synthetic()
    check_edge_cases()
    print("ALL CHECKS DONE")
