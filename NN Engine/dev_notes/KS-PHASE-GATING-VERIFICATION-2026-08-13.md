# KS phase-gating verification — 2026-08-13 (cold source read)

Claim under test: "The attack-unit king-safety term is MIDGAME-ONLY — not computed at all in the
endgame, so we have effectively no king-safety signal once the position simplifies."

Verdict: **CONFIRMED for the engine's own phase definition (phase_score <= 64), with two important
corrections**: (1) the "~28 non-pawn material" cutoff is not a material threshold at all — the gate is
on PHASE (Q=4/R=2/minor=1), which diverges from material value, and (2) the endgame is not signal-free:
a king-PROXIMITY signal (attackingLayer) and a blended shelter sliver survive, but neither reads
`KING_SAFETY_MAG` nor any KS detector.

All line numbers: `cpp_bitboard.cpp` unless noted.

---

## 1. Is evaluate_king_safety midgame-only? — CONFIRMED

Brace trace of `placement_and_piece_eval`:

- 7185-7196: `phase_score <= 64 -> isEndGame = false`; `65..96` and `>96 -> isEndGame = true`.
- 7203: `if (!isEndGame){` — opens the midgame branch.
- 7635: `	}else{` (comment 7634 "Else the game is in endgame phase") — the matching else.
- 7894: `	}` — closes the endgame branch (the `advanced_endgame_eval` gate at 7884-7893 sits just
  inside it, matching the UB comment at 7149-7151).

A mechanical brace-depth scan from 7203 returns to depth 0 first at 7894; the only depth-1 `}else{`
between is 7635. The KS block at 7568-7583 — `evaluate_king_safety` call at 7574, light-path
`king_safety_score` at 7581 — lies between 7203 and 7635, i.e. **inside the midgame branch only**.
Neither `evaluate_king_safety` nor `king_safety_score` has any other call site (grep: consumers of
`KING_SAFETY_MAG` = line 5797 only; call sites of the two functions = 7574/7581/5747).

So: `evaluate_king_safety` executes **iff phase_score <= 64**, gated further by
`ks_active = ENABLE_KS_REPLACE_LT || KING_SAFETY_MAG != 0` (7572). Defaults
`ENABLE_KS_REPLACE_LT = true` (search_engine.h:1170), `KING_SAFETY_MAG = 3000` (search_engine.h:1165)
=> active whenever the midgame branch runs.

## 2. The exact cutoff — it is a PHASE cutoff, not a material cutoff

- `MAX_PHASE = 24` (cpp_bitboard.h:101). `phase = 4*Q + 2*R + 1*(B|N)` popcounts (7178-7181).
- `phase_score = 128*(24-phase)/24` (7183, integer division).
- KS on ⇔ `phase_score <= 64` ⇔ `128*(24-phase)/24 <= 64` ⇔ **phase >= 12**.
  Check: phase=12 → 128·12/24 = 64 → ON. phase=11 → floor(128·13/24) = 69 → OFF.

Minimum non-pawn material (N=B=3, R=5, Q=9) reaching phase 12 — maximize phase-per-point
(Q: 4/9 ≈ 0.44 > R: 2/5 = 0.40 > minor: 1/3 ≈ 0.33):

| census | phase | material |
|---|---|---|
| 3Q | 12 | **27** (absolute minimum, freak) |
| 2Q + 2R | 12 | **28** (realistic minimum) |
| Q + 4R | 12 | 29 |
| 12 minors | 12 | 36 |

But the converse breaks the "~28" framing: **11 minors = 33 material, phase 11 → KS OFF**, while
2Q+2R = 28 → ON; and 2Q+R+minor = 26 material, phase 11 → OFF. There is **no material value at which
KS is guaranteed on**below 36; the gate is the phase mix, not the material sum. The "~28" claim is a
fair heuristic for normal piece mixes and its arithmetic checks out (exact min 27), but it is wrong as
a threshold statement.

**KS_PHASE_ZERO=104 is DEAD CODE in this engine.** `king_safety_score` early-outs at
`phase_score >= KS_PHASE_ZERO` (5724; default 104, search_engine.h:1375), but both of its call sites
(5747 via `evaluate_king_safety` at 7574, and 7581) are inside the midgame branch, so phase_score is
always <= 64 < 104. The taper (rebuild_ks_tables, 425-437; KS_PHASE_FULL=48, search_engine.h:1374)
does bind: full 256/256 at ps<=48, then t = 256·(104-ps)/56 → at ps=64 t=182 (**71%**). So the live KS
weight ramps 100%→71% over ps 49-64 and then **cliffs to 0 at ps=65** — the designed smooth fade to
ps=104 never happens.

## 3. Endgame king-safety signal — attack-unit KS: NONE; king-flavoured signals: some

In the endgame branch (7635-7894):

- No `evaluate_king_safety`, no `king_safety_score` (see #1).
- No latent_threat either: the only `get_latent_threat_score` eval call is 7549-7561, midgame branch,
  and it is additionally gated `!ENABLE_KS_REPLACE_LT` — with the default `true` it is skipped even in
  the midgame. **At defaults the engine has exactly one king-danger term, and it is midgame-only.**
- What DOES run in the endgame and is king-flavoured:
  - **attackingLayer proximity** — `setAttackingLayer(10, true)` (7643) builds a heatmap with a
    king-2-ring boost around BOTH kings in all phases (`kinc = increment * KS_ZONE_ATTACK_PCT/100`,
    9241; ring loops 9258-9314 / 9334+; only the open-square/pawn-shield refinements are
    `!isEndGame`, 9277-9285). Default KS_ZONE_ATTACK_PCT=50 (search_engine.h:1337) → kinc=5 per ring
    visit in the endgame. Every `evaluate_*_endgame` reads this per attacked square, and
    `evaluate_kings_endgame` charges `attackingLayer[0][x][y]` + half of `[1]` around the king
    (4700-4701). This is a *proximity/placement* credit — no attackers/defenders/weak-square/check
    model — but it is a nonzero "pieces near the enemy king score more" endgame signal.
  - **Blended shelter sliver, ps 65-69 only**: the endgame kings loop (7739-7777) blends
    `evaluate_kings_midgame` (shelter credits, 3112-3123) with `evaluate_kings_endgame` for
    PHASE_BLEND_LO=40 < ps < 70 (search_engine.h:1073-1074); inside the endgame branch that window is
    ps 65-69, midgame weight (70-ps)/30 ≤ 5/30 ≈ 17%.
  - `advanced_endgame_eval` (7884-7893) — king-driving/conversion logic, not danger.
  - `is_practically_drawn` returns 0 outright (7637-7638) for drawn material.

None of these read `KING_SAFETY_MAG`, `king_safety_danger`, or any KS_* detector.

## 4. Detector consumers — KS-local ONLY, CONFIRMED

Full-repo grep for the five detector knobs (cpp/h): every read is inside `king_safety_danger`:

- `KS_PIN_MODE` — 5342 (own_pinned mask, also reused at 5500 for FLANK_MODE=2).
- `KS_SQC_MODE` — 5376 (contested_zone verdict via `ks_sqc_breaks`).
- `KS_WEAK_VAL_MODE` — 5441 (weak_val_sum vs weak_squares in `units`).
- `KS_DEFAWARE_MODE` — 5453-5478 (attacker reweight).
- `KS_FLANK_MODE` — 5484-5505 (flank breadth).

`king_safety_danger` is called only from `king_safety_score` (5726-5727), which is called only from
`evaluate_king_safety` (5747) and the light path (7581) — all midgame-branch. The detectors touch
**nothing** in setAttackingLayer, OvD accumulators (whiteOffensive/DefensiveScore are *read* by
MOD_KS_CONTROL at 5761-5762, data flows IN, never out), central_score, or passers.

**⇒ Changing KS detectors cannot directly change any eval computed in the endgame branch.**

## 5. Reconciling the "endgame deltas" — mechanisms ranked

(a) **Labeler mismatch — REAL, and almost certainly the dominant cause for the regret phase-split.**
`_build_game_regret_set.py:58-60`: `phase_of` buckets by TOTAL piece count including kings and pawns —
`"endgame" if pc < 14`. Example: 13 units = 2K + 2Q + 2R + 2B + 5P → phase = 4+4+2+2+1+1 = 14 ≥ 12 →
phase_score = floor(128·10/24) = 53 → engine MIDGAME, KS **fully live** (taper 256/256 at ps≤48;
at 53, t = 256·51/56 = 233 ≈ 91%). The labeler's "endgame" bucket demonstrably contains KS-on
positions. (`_build_variant_regret_set.py:91-93` has the same labeler.)

(b) **Shared global caches — REAL for any multi-position run in one process.** All caches — TT
`searchEvalCache`, `evalCacheNew`, `quiesceEvalCache`, `moveGenCache`, history / moveFrequency /
killers / counter-moves — are file-scope globals, shared across `ChessAI` instances, never cleared
(documented at dev_notes/HANDOFF.md:189; `tactical_test.py:run_one` at line 136 constructs a fresh
`ChessAI` per FEN but stays in one process). A KS config alters searches at high-material positions
early in the suite → different TT entries, history and killer contents → different move ordering and
cutoffs at later low-material positions whose static evals are KS-independent. This is the mechanism
for the MAG-ruler moving npm≤12 positions: at npm≤12 the maximum phase is 5 (Q+B → ps=101), so the
root position is deep in the endgame branch and `evaluate_king_safety` is not reachable from statics.

(e) **Within-search phase crossing via promotion — real but marginal.** phase is recomputed per eval
node; each promotion adds +4 phase, so from an npm≤12 root (max phase 5) a line with two extra queens
on the board reaches phase ≥ 13 and executes KS at those nodes. Possible in deep/qsearch promotion
races, but requires both promotions simultaneously on the board — a rare subtree, not a bulk effect.

(c) **Detectors feeding endgame-active terms — REFUTED** (see #4).

(d) **A missed all-phase king term — REFUTED as an explanation.** The endgame king-flavoured signals
(#3: attackingLayer proximity, ps 65-69 shelter blend, advanced_endgame_eval) are real but read
neither `KING_SAFETY_MAG` nor any detector, so they cannot produce KS-config-dependent deltas.

## Corrections to the claim

1. The claim's threshold framing ("off below ~28 non-pawn material") is wrong in kind: the gate is
   phase ≥ 12, which is non-monotonic in material (OFF at 33 = 11 minors; ON at 27-28 queen-heavy).
2. "No king-safety signal once simplified" overstates slightly: a king-proximity attackingLayer signal
   (at 50%) and a ≤17% shelter blend at ps 65-69 survive — but no danger MODEL (attackers/weak
   squares/checks) exists past ps=64, and the live term cliffs from 71% weight to 0 at ps 64→65.
3. `KS_PHASE_ZERO=104` never fires — the phase fade the taper implements is truncated by the branch
   gate at 64; the effective KS phase profile is 100% (ps≤48) → 71% (ps=64) → 0 (ps≥65).
