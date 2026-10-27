# Durable labeled position bank — schema & usage (2026-07-21)

A REUSABLE labeled position corpus sampled from the whole saved-game archive
(`selfplay/games/*/game_*.jsonl` = **33,683 games / ~96k unique positions**). Built once, accumulates across
sessions, and is the shared source for every eval-fit (KS recalibration now; pawn-overvaluation lever next).
Built by `diagnostics/build_position_bank.py`; SF18 truth added by `diagnostics/add_sf18_labels.py` (curated
subset). Output: `diagnostics/ks_sets/position_bank.csv`.

## Schema (one row per position)
| column | meaning |
| --- | --- |
| `fen` | position |
| `src` | game-dir it was sampled from |
| `phase_score` | engine phase (0 opening .. 128 endgame; LOW=midgame per `phase-score-convention`) |
| `our_total` | our static eval, WHITE-POV pawns (`-ev_breakdown.total/1000`) |
| `our_ks` | our king_safety term, WHITE-POV pawns |
| `sf11_total` | SF11-static total, WHITE-POV pawns (`eval_vs_sf11.SF11Eval`) |
| `sf11_ks` | SF11-static "King safety" term, WHITE-POV pawns |
| `geo_class` | lightweight `ks` (king under ring pressure / check) vs `quiet` |
| `kzone_w/b` | king-ring enemy-attacker pressure per side (density) |
| `in_check_w/b` | is that king in check |
| `sf18` | SF18-SEARCH truth, WHITE-POV pawns — blank until `add_sf18_labels.py` fills a subset |

## Tier selection (drawn from the bank by label, NOT hand-built)
- **broken / target** (must START firing): `abs(sf11_ks) >= 1.0 AND abs(our_ks) < 0.3` — SF sees king danger we
  zero. **281 of 1852** in the N=2000 sample. **MUST be SF18-filtered first** (SF11-static is contaminated on
  sharp positions; keep only where SF18-search agrees in direction — the SF11⊥SF18 exclusion rule).
- **working** (must NOT change): `abs(sf11_ks) >= 0.5 AND abs(our_ks - sf11_ks) < 0.5` — we already agree. **82**.
- **control-crowded-safe** (blow-up guard): high `kzone_*` density but small `abs(sf18)` — needs SF18 labels.
- **control-calm** (must stay ~0): `geo_class == quiet` and small `abs(sf11_ks)`. ~1600.
Each tier is phase-bucketed (opening/midgame/endgame/advanced-endgame) so the fit is validated in every phase.

## Rebuild / extend
`pyrun diagnostics/build_position_bank.py [N=3000 SEED=7 PER_GAME=3]` — deterministic (seeded stride), stratified
across phase buckets. Scale N for a bigger bank; SEED for a disjoint draw. Guards: skips terminal positions and
SF11-unparsable evals.

## IMPORTANT caveat
`sf11_ks` is SF11-STATIC — a diagnostic aid, contaminated on sharp positions (see `sts-wac-tag-artifact` /
SF18-truth lesson). Any TARGET set derived from it must be SF18-validated before it drives a fit. `sf18` is truth.
