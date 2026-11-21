# GTA VI trophy rarity forecast

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast every GTA VI
trophy's Exophase earn rate at 2026-11-26T00:00Z as five quantiles per
trophy, scored by pinball loss on the logit scale.

## Model (`main.py`, Python 3.13 stdlib only)

A trophy's rate is `earners / players_tracked`, and the two move for
different reasons, so they are forecast separately and combined:

- **Players tracked**: log-linear in log-hours since release, fitted on the
  history's distinct refresh points, anchored at the last read. Exophase
  refreshes a game's counts on its own schedule (hours apart), so
  consecutive identical reads collapse into one refresh level — a flat run
  is one stale page, not a trend.
- **Earners per trophy** (`rate x players / 100`): log-linear in log-hours
  since release per trophy, slope shrunk toward the cross-trophy median
  slope, anchored at the trophy's last observed earners.
- **Rate at target** = `100 x earners_target / players_target`, clamped to
  `[0.01, 99.99]` before the logit.
- **Recent window + horizon damping**: both fits use only the last 36 hours
  of refresh points (the launch rush's slope is not the next days' slope),
  and forward slopes are multiplied by `1 / (1 + horizon_days)` — launch
  growth keeps decelerating, and a whole-history fit overshoots a
  saturating curve badly (measured on the synthetic backtest).
- **Spread** (logit sigma) combines a random-walk term (`0.3` per
  square-root day), the fits' residual uncertainty, a refresh-timing term
  (the answer is the page as of the last refresh before the target), and a
  floor of `0.25` / cap of `1.5`.
- **Flat history**: when the history shows no movement at all (as the
  practice cases do), the median is persistence shifted by a refresh prior
  — a refresh dilutes rates because newly tracked players start with about
  zero trophies, and it dilutes rare trophies hardest (logit shift -0.15
  for rates >= 50%, -0.5 for 10-50%, -1.2 for 1-10%, -1.5 below 1%) —
  with a wide spread.

Deterministic: no randomness, files processed in name order.

## Local validation (`tests/validate.py`)

- Practice format / runtime / determinism: both practice cases valid,
  2,771 bytes each (limit 65,536), ~0.1 s per run (limit 8 s),
  byte-identical across runs.
- Practice refresh scenarios (built from the documented totals — 3,868
  players, 4,276 trophies earned after the refresh): under the two
  realistic scenarios, loss **0.20 vs the starter's 1.40-1.42** (the
  documented baseline loss is ~1.51). Under a uniform-growth scenario
  (ruled out by that documented 1.51), 0.74 vs 0.34.
- Synthetic multi-day backtest (generated 7-day recordings with stepwise
  refreshes, 3 seeds x 2 launch regimes x 6 checkpoint/horizon cuts):
  grand mean loss **0.280 vs the starter's 0.319**; on the final's shape
  (48 h checkpoint, 108 h horizon) it wins in all 6 seed/regime combos
  (0.27-0.35 vs 0.33-0.42). Mean 90% coverage 0.97.
- Edge cases: 0% / 100% trophies, 1-2 read histories, null rates, empty
  history — all produce valid output.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 tests/validate.py   # full local validation
```

## License

MIT — see LICENSE.
