# SESSION HANDOFF — 2026-10-13 (covers 10-10 → 10-13)

HANDOFF — this phase **shipped twice** after a week of nothing: the structural retune (gen 2) and round 2 with explicit piece
values (gen 3), ≈ +25-30 Elo external over the 10-04 engine (self-play ≈ +80). Round 3 split the instruments and was not
shipped; a diagnostic is running. POT groundwork is done and the design brief is ready for the owner's discussion.

## ▶️ FIRST ACTIONS
1. `overnight_runner.sh ps` — expect queue #63 (R3A split diagnostic, log `E:\chess_data\q63_r3a_split.log`) running or done.
2. Fingerprints: v1 **250 / 35,310,778 / 3.784** · v2 shipped (gen 3) **259 / 47,869,841 / 3.927** · `V2_SHIP_GEN=2` 256 / 47,473,313
   · `V2_SHIP_GEN=1` 254 / 50,622,239.
3. Read #63's result (below), then `dev_notes/POT-DESIGN-BRIEF-2026-10-13.md` before the POT discussion.

## ▶️ UPDATE 2026-10-15 (supersedes "NOT SHIPPED — round 3" below)
- R3A CLOSED: #63 halves values-only +3 / cells-only +4 on SF18 (self-play +10 / +12); #64 R3A whole on seeds 224/225 −3 ⇒
  4-seed pooled ≈ −8 ± 10 (self-play +21) ⇒ not shipped; structural retune done (+22 → +9 → ~0 external).
- Judge re-anchor (#65): gen 3 vs SF18 @1000 54.0% · @1500 37.3% · @2000 28.3% ⇒ POT gates at SF18 @~1100.
- SPSA (#65, 2 cold-start runs): only `KS_V2_HALF` replicates (772 → 814 / 822) ⇒ candidate ≈ 818, needs the games gate;
  KS MAX / EG_PCT and winnability BASE / SP / OCB did not replicate (at optimum).
- Data for POT: EGX endgame enrichment (#66, 60k + 58,929 d10 rows) and the trajectory set (#67, ~39k) — POT brief appendix B.
- Nothing running; machine idle. Next: POT discussion (brief + appendices A/B), then the KS_V2_HALF gate if wanted.

## SHIPS (all three legs of the owner rule: self-play ≥ +2σ · SF18 > −1σ · Mediocre ≥ 200 g not negative)
| gen | what | self-play | SF18 @1000 | Mediocre | commit |
|---|---|---|---|---|---|
| 2 (10-11) | structural retune, hybrid λ 0.1 | +56.6 ± 17.9 | +22 ± 14 | +10 ± 30 | c193973 |
| 3 (10-12) | round 2: λ 0.03 + V2_VAL piece values + PST pin | +22.8 ± 17.9 | +9 ± 14 | +7.3 pts | 13b293e |
Compiled tables: `ship_tables_v3.h` (gen 2), `ship_tables_v4.h` (gen 3) via `_ship_tables_gen.py`; `V2_SHIP_GEN` selects
(preset default 3). Config §1 of EVAL-V2-CURRENT-CONFIG.md, OPTIMIZATION_LOG, memory `baseline-fingerprints-register`.

## NOT SHIPPED — round 3 (R3A)
Winnability-aware fit + 4 caps + GAUNT slice (games vs SF18/Mediocre), λ 0.03 on gen 3: self-play +20.9 ± 17.9, SF18 pooled
−14 ± 14 (−1σ) ⇒ split. The fit drove the caps to ~0 (they push toward draws; our eval is UNDER-confident). Twin without GAUNT:
self-play +12.9. **Running: #63** splits R3A's δ into values-only / cells-only arms (SF18 seeds 224/225 + self-play seed 226).
Prediction: the SF18 loss sits in values-only (queen eg −68 → −104 overshoot). Judge drift: gen 3 scores 54.7% vs SF18 @1000 ⇒
re-anchor (more SF nodes) before the next big external read.

## METHOD (what now works, and the traps found)
- Corpus `E:\chess_data\retune\corpus_v1.csv` (956,670; by-game split via a recovered game map) and `corpus_v2.csv` (+100k GAUNT).
- `_retune_fit.py EXPERIMENT=hybrid`: static (quiet SF18 labels) + depth target (our d10 of the CURRENT ship) — static-only
  worsens depth (memory `static-only-fit-worsens-the-depth-target`). Flags: VAL/PIN, WIN_AWARE, CAPS, FEAT/CORPUS/D10_GLOB.
- `_retune_export.py MODE=tables|verify`: closure + REALISED gain (gen 2 ~85% → WIN_AWARE 94%). Traps: KD cells fractional
  (write full precision); per-file isolated-pawn cells must be file-tied; never MODE=export/closure (_texel_kauf_fit import
  side effects); value knobs written as SHIP base + δ.
- Bench (#55, 10-12): static gap to SF11 shrank ~40%; ahead on OWN-PLAY middlegames / pure endgames (home field), behind on
  EG-leaning + decisive; off-distribution SF11 still ahead. Effective-value ratios moved toward SF18 search.

## POT (owner discussion pending — brief ready)
`POT-DESIGN-BRIEF-2026-10-13.md` (+ appendix A, endgame signatures): static eval under-confident in every endgame type, search
fixes most EXCEPT pawn endings; POT's DIRECTION half adds nothing (PX/KFL own it), MAGNITUDE (complexity) static EG −5.9% /
depth −1.5%; SF11's recipes still beat us on fortress/drawing patterns (KBPsK, KNPKB/KBPKN, KRPKR/KRPPKRP), we beat SF11 on
conversion endings. Decisions for the owner: cause split (pawn-ending SEARCH rules first?), SF's bounded complexity form,
mg potential still parked?, which SF endgame recipes to port.

## OWNER CALLS this phase
Quick experimental understanding over a-priori choices (weights experiment) · ship gen 2 and gen 3 (values arm for structure)
· piece-value knobs OK'd · corpus breadth incl. "whacky" positions for a general evaluator; the EXPLORE/EXPLOIT generator waits
for the full retune after POT · bench checkpoints after retune / POT / search transition · concurrency 4 for timed runs OK ·
busy ~10-13/14 (interview): run autonomously, bring only short decisions.

## STATE
Branch NN-ENgine, ~40 commits since 10-10, none pushed (push only on say-so). Queues #50-#63 in `selfplay/`. Data on
`E:\chess_data\retune` and `E:\chess_data\bench101{1,3}`.
