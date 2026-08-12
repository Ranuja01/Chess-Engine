# Collapse fix — Campaign A1 (existing conditional hooks) + overnight run — 2026-07-04 night

Continues `collapse-diagnosis-2026-07-04.md` (read its CORRECTION banner: the collapse is FIXABLE
conditional-eval over-read, not an NNUE boundary). Plan: `~/.claude/plans/handoff-collapse-fix-eventual-allen.md`.

## Infrastructure (this session)
- `mine_overreads.py` now emits a `game` column (bench regenerated: SAME 2,172 positions, 1099/1073 W/B).
- `diagnostics/bench_split.py` — deterministic 70/30 split BY GAME (md5 of tag/game):
  train 1,510 pos / 553 games, holdout 662 pos / 238 games.
- `bench_gate.py` takes an optional bench-csv arg; new dispatcher sub `bench_gate <tag> [bench_csv] [KNOB=v ...]`.
- `spsa` sub concurrency 6 → 4 (worker-cap rail).
- All engine code untouched — NO rebuild; byte-id unaffected.

## A1 hook ladder (STATIC sign-flips vs SF, TRAIN split; baseline 982/1510 = 65.0%, mean|gap| 2.84p)
| config | flips | % | mean\|gap\| |
|---|---|---|---|
| baseline | 982 | 65.0% | 2.84 |
| MOD_PIECES_DEFEND=32 | 969 | 64.2% | 2.80 |
| MOD_PIECES_DEFEND=64 | 951 | 63.0% | 2.76 |
| MOD_PIECES_DEFEND=128 | 932 | 61.7% | 2.73 |
| MOD_PIECES_DEFEND=256 | 916 | 60.7% | 2.70 |
| MOD_PIECES_DEFEND=512 | 908 | 60.1% | 2.69 |
| REALIZ_MAT_K=128 PHASE_K=128 | 982 | 65.0% | 2.84 (DEAD — imbalance credit isn't the flip carrier; midgame phase_score≈0) |
| REALIZ_MAT_K=256 MAT_THRESH=2000 | 982 | 65.0% | 2.84 (DEAD) |
| MOD_PIECES_CONTROL=64 | 826 | 54.7% | 2.47 |
| **MOD_PIECES_CONTROL=128** | **824** | **54.6%** | **2.45** (saturates ≥64 — MOD_FLOOR=128 caps the damp) |
| MOD_PIECES_CONTROL=256 | 824 | 54.6% | 2.44 |
| ctl128 + def128 (C1) | 766 | 50.7% | 2.37 (additive) |
| C1 + MOD_FLOOR=64 (C2) | 625 | 41.4% | 2.22 |
| C1 + MOD_FLOOR=32 (C3) | 559 | 37.0% | 2.17 |
| C1 + MOD_FLOOR=0 | 488 | 32.3% | 2.14 (full zeroing allowed — high collateral risk, not carried) |

## Held-out confirmation (baseline 434/662 = 65.6%)
- C1 (MOD_PIECES_CONTROL=128 MOD_PIECES_DEFEND=128): **328 = 49.5%** (train 50.7% — generalizes)
- C2 (+MOD_FLOOR=64): **271 = 40.9%** (train 41.4%)
- C3 (+MOD_FLOOR=32): **241 = 36.4%** (train 37.0%)
Game-level split, near-identical train/holdout rates ⇒ NOT overfit (expected — 2-3 knobs only).

## Reading
- The flip carrier is the PLACEMENT claim (`pieces`), exactly the illusory-activity diagnosis: damp it when the
  favoured side lacks a control edge (CONTROL) or is net out-attacked (DEFEND). REALIZ/imbalance is not involved.
- MOD_PIECES_DEFEND history (collapse-campaign.md 2026-06-29): cluster-positive, generalizing, but move-match
  showed placement collateral; move-match was ruled the WRONG gate. Tonight's gates are WAC/STS-held + aggregate
  SPRT — the correct adjudicators.
- KS-family hooks not laddered: KING_SAFETY_MAG=0 at defaults (MOD_KS_* modulate a zero term), and cranked static
  KS was already shown not to move this bench.

## Overnight sequence (autonomous)
1. Pre-flight byte-id `wac` (expect 245/39,146,294).
2. WAC + STS under C1/C2/C3 (small WAC dip OK, STS crater = reject).
3. Best survivor → lightning SPRT `gate '<cfg>' <label> 1200 5` overnight.
4. If none survive → Campaign B `spsa spsa_search_rfp.json 25 200 fast` instead.
NOTHING ships/commits unattended; results recorded here.

## Results (filled as the night progresses)
- Pre-flight byte-id: **PASS** 245/300 solved, 39,146,294 nodes, EBF 3.671 (exact) — clean build.
- Baseline (no knobs): WAC 245/300, STS 1512/3000 (50.4%).
- **C1** (ctl128 def128): WAC **238** (−7, small dip), STS **1507** (50.2%, −5 = HELD). Flips train 50.7% / holdout 49.5%. → PASSES gate.
- **C2** (+MOD_FLOOR=64): STS **1420 (47.3%) = −92 = CRATER → REJECTED.** Aggressive placement-zeroing buys big
  bench-flip reduction (40.9% holdout) but costs ~3% positional — the placement collateral the 2026-06-29 history
  predicted. Rejected on the STS gate.
- **C3** (+MOD_FLOOR=32): STS **1250 (41.7%) = −262 = deep CRATER → REJECTED.**
- **SURVIVOR = C1** (`MOD_PIECES_CONTROL=128 MOD_PIECES_DEFEND=128`): best flip-reduction-per-collateral
  (bench flips 65%→50% train + holdout, STS held −5, WAC −7). Both MOD_FLOOR variants rejected — the FLOOR is the
  collateral dial, and this bench's flip-cut past C1 costs positional strength faster than it cuts flips. C1 keeps
  the default MOD_FLOOR=128.
- **SPRT (DONE, task bqubnzqdr): C1 REJECTED, ~−202 Elo.** H0 accepted at game 250: **+47 −178 =25 (23.8%),
  elo ≈ −202.2 ± 50.6, LLR −3.056** (bound −2.944). Verdict: `selfplay/games/sprt_c1damp/sprt.json`.

## ⚠️ KEY RESULT — bench sign-flip reduction is ANTI-PREDICTIVE of strength here
C1 cut static sign-flips vs SF from **65% → 50%** (train AND holdout, game-split, not overfit) and held STS
(−5) and WAC (−7) — every static/bench gate said "clean, generalizing." Yet in real lightning games it loses
**~200 Elo.** The bench metric is not merely a weak proxy (cf. [[eval-accuracy-payoff-is-pruning]] "SF-parity
accuracy barely helps move-choice") — for the placement-damping mechanism it is *actively misleading*.

**Interpretation:** the collapse bench is the ~6% tail of sharp/over-read positions. Damping the placement
(`pieces`) claim toward SF on that tail also fires across the other ~94% of ordinary midgame play, where the
placement term is load-bearing for good move choice. STS (a curated d10 positional suite) barely feels it, but
full games at time control are dominated by that broad collateral → catastrophic. This is the placement-collateral
the 2026-06-29 `MOD_PIECES_DEFEND` history flagged — now quantified as a **hard game-loss**, not the "small STS
collateral" it looked like at fixed depth. The two floor-variants (C2/C3) that cut flips further were already
STS-rejected; C1 was the mildest survivor and still craters.

**Disconfirmed (do not re-litigate unattended):** reducing `overread_bench.csv` static sign-flips via the existing
placement-conditioning hooks (`MOD_PIECES_CONTROL` / `MOD_PIECES_DEFEND` / `MOD_FLOOR`) as a *strength* lever. The
bench remains a valid DIAGNOSTIC of where static eval disagrees with SF, but flip-count is NOT a ship gate — only
the SPRT is. Any future conditional-damp must be validated in games directly, and must NOT bleed into ordinary
midgame placement (the collateral is the killer). This strengthens the case that the collapse tail is dynamic
(NNUE territory) rather than cheaply reachable by damping an existing static term.

## Campaign B (safe overnight, launched after C1 rejection)
Build unchanged since the byte-id pre-flight (no rebuild all night) → SPSA is byte-id-safe. Launched:
`spsa spsa_search_rfp.json 25 200 fast spsa_rfp` (search cluster around the shipped RFP defaults; conc 4; fast d4
lane; 25 iters × 200 games). Aggregate, self-ratifying, zero overfit risk. Result: (pending — task bvuxr72ds; confirmed running iter 1
plus/minus A/B at conc 4 LIGHTNING). NOTE: dispatcher lane arg must be `search` (not `fast` — spsa.py --lane
accepts only search|eval; first launch errored on that and was corrected).
On completion: read the SPSA-winning config, ratify with `gate '<winner>' spsa_win sprt_spsa_rfp 1200 5`, and
record — SHIP DECISION IS THE USER'S. If SPSA is flat (likely — combo1+RFP may be near-optimal per the audit),
record "search tapped."

**RESULT — killed after 6/25 iters (user call, during the daytime pivot to fixed-nodes); FLAT → search knobs
confirmed tapped (2nd confirmation).** θ never moved off the shipped defaults across 6 iterations (final
`spsa_rfp_state.json` iter 6: RFP_MARGIN 1473.6 vs 1500, RFP_MAX_DEPTH 5.81 vs 6, LMP_BASE 1.14 vs 1,
LMP_MAX_DEPTH 4.89 vs 5, NULLMOVE_EXTRA 1.81 vs 2, HISTORY_LMR_SCALE 2.11 vs 2, VERIFY_MARGIN 16283 vs 16000,
ATTACK_OPEN_MULT 4.81 vs 5); per-iter plus-arm scores 0.40–0.52 (coin-flips → no visible gradient). Nothing
shipped, nothing committed. Symmetry caveat (Fable): "SPSA-flat → tapped" already failed once this week (RFP
shipped +73 post-"tapped") → this means "no gradient at this power over THESE knobs," not "no search Elo exists";
new search Elo comes from new MECHANISMS (ProbCut unbuilt), not sweeps. Superseded by the fixed-nodes pivot
(`fable-audit-2026-07-04.md`).
