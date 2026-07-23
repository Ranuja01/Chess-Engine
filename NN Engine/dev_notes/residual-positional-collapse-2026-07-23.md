# Residual positional-collapse analysis — picking the next target (2026-07-23)

After the passer V3 win (positional class 59→53/seed), mined the v3 residual positional collapses
(`positional_collapse_dossier.py` pointed at family=v3, seeds 0/1/2 — via new `DOSSIER_FAMILY`/`DOSSIER_SEEDS`
defaults) for the per-term over-read vs SF11/SF18.

## Aggregate over-read by clean-comparable term (mean, oriented to the collapsing side)
| term | mean over-read | note |
|---|---|---|
| **Space/central** | **+0.62** | BIGGEST — the next target |
| **Imbalance (OvD)** | **+0.32** | second — the known OvD realizability lead |
| KingSafety | +0.10 | small (KS is handled) |
| Threats | +0.09 | small |
| **Passed** | **+0.07** | TINY — passer fix VALIDATED (was the dominant driver pre-V3) |

## Reads
- **Passers are no longer a top driver** (+0.07) — independent confirmation the V3 passer work removed passers
  as the dominant positional-collapse cause. The −10% positional game win is real and passer-specific.
- **NEXT TARGET = Space/central over-read** (+0.62, ~2× the next term). Matches the memory eval-map (Space ~9×
  SF: 0.46 vs 0.05). It is a **detector-efficacy** problem: we credit central/space control UNCONDITIONALLY;
  SF conditions it (safe squares, pawns behind, actually-usable). Secondary = **Imbalance/OvD** (+0.32, the c5
  realizability over-read).
- Representative FENs are sharp middlegames where we are wildly optimistic (our +20.6 vs SF +6.5; +10.8 vs +3.0)
  — general over-reading of our own position. Multi-term (placement/PST, capgains, KS also pile on in specific
  positions), so JOINT tuning matters — but Space is the single biggest CLEAN lever.

## Plan (user's round-based method)
1. Curate a SPACE corpus (space-over-read collapse FENs + guards where space IS real), SF18-labelled, phased.
2. Detector work: make the space/central credit CONDITIONAL on usability (SF-style: safe/behind-pawns/
   reachable), gated default-off + byte-id, verified two-sided on the corpus (over-read down, real-space held).
3. Fold the space knobs into the JOINT tune with passer(V3) + KS + OvD + material on the mixed corpus, win%-space
   (NOT cp), guard every family — no regressions.
4. Games verdict (KS-model categorical: space/positional class down, others not resurface) → lazy-accept → ship.

Knobs in play: `CENTER_INNER_MULT`/`CENTER_OUTER_MULT` (=200/150), `SCALE_CENTRAL`; the central_score machinery
in cpp_bitboard.cpp (update_global_central_scores). Detector conditioning is the new work.
