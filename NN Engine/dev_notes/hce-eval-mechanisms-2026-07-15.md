# HCE Eval MECHANISMS (not features) — mined 2026-07-15 (Fable)

Scope: mechanisms/methodology portable to our engine under the load-bearing-optimism constraint.
Sources: Weiss `src/evaluate.c` + `src/history.h` (TerjeKir/weiss master), Ethereal `src/evaluate.c` + `src/tuner.c/.h` (AndyGrant/Ethereal master), chessprogramming.org "Static Evaluation Correction History", SF11 `evaluate.cpp` (via our sf11_eval_reference.md background).

---

## 1) Correction history — variants across engines

Definition: online table(s) that learn the ERROR (searchScore − staticEval) keyed by cheap position features, then add the learned correction back onto static eval. This is a de-biaser, not a feature: it can only pull eval toward what search actually returns — including pulling optimism DOWN.

### Variant table (from CPW + Weiss code)
| Key | Index source | Origin |
|---|---|---|
| Pawn | stm + pawn-structure hash (== our planned pawn-hash key) | Caissa (Oct 2023) |
| Material | stm + material-config hash | Caissa |
| Threats | stm + hash of capturable-pieces bitboard | Motor |
| Minor piece | stm + K/N/B locations hash | Sirius |
| Major piece | stm + K/R/Q locations hash | Sirius |
| Non-pawn | per-color hash of all non-pawn pieces | Starzix |
| Last move / continuation | prev 1–2 ply moves (piece×to) | ice4 / Motor |

### Weiss concrete implementation (history.h — the cleanest reference)
- Tables: `pawnCorrHistory[stm][idx]`, `minorCorrHistory[stm][idx]`, `majorCorrHistory[stm][idx]`, `nonPawnCorrHistory[color][stm][idx]`, plus contCorr keyed off prior moves.
- Bonus: `CLAMP((score - eval) * depth / 4, -172, 289)` — note the ASYMMETRIC clamp: larger headroom upward than downward in Weiss's sign convention; tunable.
- Update: gravity form `*entry += bonus - *entry * abs(bonus) / div;` (div ≈ 1651 pawn, 1222 major, per-table) — self-decaying EMA, no explicit aging pass.
- Application: weighted blend `c = 5868*pawnEntry + 7217*minorEntry + 4416*majorEntry + ...` then `correction = c / 131072`, added to raw static eval, result clamped inside mate bounds.
- Update gating (standard across engines): only update when the node's best score is a usable bound vs static eval (skip fail-highs below eval / fail-lows above eval, skip in-check and mate scores) — this keeps the table learning genuine eval error, not search noise.
- CPW note: corrected eval is also used as a **complexity proxy** — |correction| large ⇒ static eval unreliable ⇒ widen RFP margin / reduce less in LMR. This is exactly "make the eval that gates pruning more reliable" and doubles as an eval-trust signal.

### Fit to OUR engine (absolute Black-positive, non-negamax)
- Everyone else stores corrections in stm-relative space. Ours is simpler: eval is absolute, search scores are absolute ⇒ learn `err = searchScoreAbs − staticEvalAbs` directly, keyed by (stm, structure-hash). Keep stm in the key (the same structure has different error depending on who moves) but NO sign gymnastics on the value.
- Pawn-key variant piggybacks on the planned pawn-hash (one hash, two consumers).
- Start with pawn + non-pawn (or minor+major) two-table blend; contCorr later.

## 2) Lazy / tapered / incremental eval mechanics
- **Taper** — Ethereal: phase = `4*Q + 2*R + 1*(N|B)` in [0,24]; `eval = (MG*phase + EG*(24-phase)*scaleFactor/SCALE_NORMAL) / 24` — note the scale factor multiplies ONLY the EG half (drawishness is an endgame property). Weiss identical shape with 0..128 phase (matches our phase_score convention). We already taper; the EG-only scaling hook is the takeaway (see §4).
- **Packed score** — both engines pack mg/eg into one int (`MakeScore/ScoreMG/ScoreEG`), accumulate once, split at the end. Pure speed mechanics; relevant if we ever restructure, not a strength lever.
- **Pawn(-king) cache** — Weiss `ProbePawnCache()` keyed by `pawnKey`; Ethereal caches pawn+king-safety-relevant structure (`getCachedPawnKingEval`). Confirms our pawn-hash plan; Ethereal's twist: include king squares in the key so king-shelter terms cache too. Both DISABLE the cache under the tuner (traces must be complete) — remember for our Texel harness.
- **Lazy/early-out** — Weiss's only early-out is the endgame-table probe (`EndgameTable[EndgameIndex(materialKey)]` → specialized KPK-style evaluators, return immediately). Ethereal's "lazy" is NNUE-vs-classical routing — N/A. Neither does margin-based lazy-eval aborts at master; our LATENT_THREAT/CAPTURE_GAINS lazy-skip idea has no clean donor here (and it's byte-id-risky per lane-2 notes).

## 3) Texel tuning methodology (Ethereal tuner.c/.h — the reference HCE tuner)
- **What is tuned**: ALL 904 existing weights (856 linear, 44 king-safety, 4 complexity), mg and eg independently (`TVector[NTERMS][PHASE_NB]`). Retuning existing weights ≠ feature-add.
- **Labels**: pure game OUTCOMES `[1.0]/[0.5]/[0.0]` from a FENS file — no search-eval mixing at all in Ethereal. 42.5M positions. (Modern practice elsewhere blends outcome with search score; Ethereal shows pure-outcome works at this data scale.)
- **Loss**: MSE of `result − sigmoid(K, staticEval)`, `sigmoid = 1/(1+exp(-K*E/400))`.
- **K**: grid search with progressive refinement (`computeOptimalK`, KPRECISION=10, each pass `step /= 10`) — fit K to the CURRENT eval before tuning, so scale is calibrated not fought.
- **Optimizer**: **AdaGrad** (not plain SGD/Adam): `adagrad += g²; param += g * rate/sqrt(1e-8+adagrad)`; global rate 0.10, decay `rate /= LRDROPRATE` every 250 epochs; mini-batch 16,384.
- **Nonlinear terms**: king-safety is quadratic in eval, so its gradient is chain-ruled with max-gating: `grad += (base/360)*(max(bsafety,0)*bcoeff − max(wsafety,0)*wcoeff)` — i.e., they linearize AROUND the current safety accumulator instead of pretending the term is linear. Directly relevant if we ever tune our KS/latent-threat aggregates.
- **Anti-overfit**: honestly, almost none explicit — no L1/L2, no visible validation split. Their protection is DATA VOLUME (42M positions ≫ 904 params) + mini-batching + LR decay. Lesson for our tapped outcome-Texel campaign: the fix is more/better-distributed positions (self-play at multiple TCs, dedup, quiet-only), not a cleverer loss.
- **TRACE mechanics**: eval terms increment a coefficient struct `T` under `if (TRACE)` so each position becomes a sparse linear feature vector; gradients are exact, one eval per position per epoch is avoided (coefficients cached). If we retune, build this trace path.

## 4) Whole-eval conditioning / scaling / drawishness (non-feature, often optimism-SUBTRACTING)
- **Endgame scale factor** (Weiss `ScaleFactor()`, Ethereal `evaluateScaleFactor()`): multiply the EG component by factor/normal BEFORE taper. Triggers: opposite-colored bishops (Weiss returns 64 or 96 of 128; Ethereal `SCALE_OCB_BISHOPS_ONLY` etc.), lone-minor ⇒ near-`SCALE_DRAW`, lone queen vs pieces, and — key — **stronger side's pawn count** (few pawns ⇒ quadratic scale-down; Weiss also checks pawns spread across both flanks). This is a pure optimism-SUBTRACTOR: it only shrinks the winning side's score toward draw.
- **Rule-50 damping** (SF11): `v = v * (100 − rule50_count) / 100` applied to the final eval. Whole-eval, monotone shrink, two lines of code.
- **SF11 initiative/complexity as a CLAMP not a bonus**: SF11's initiative term is bounded so it can only pull eval toward zero for the side that can't make progress (`max(complexity, −|eg|)`-style). Ethereal's version: `complexity`, returned as `MakeScore(0, sign*MAX(ScoreEG(complexity), −abs(eg)))` — the max(…, −|eg|) guarantees the adjustment never flips the sign of the eval, only shrinks it. The MECHANISM (sign-preserving shrink toward 0 keyed on "can the winning side actually make progress": total pawns, pawns on both flanks, pawn-only endgame) is a drawishness detector, not a feature bonus. But note: as usually tuned it also ADDS for high-pawn positions — porting only the negative/shrink half keeps it optimism-safe.
- **Tempo**: flat ~18-20cp side-to-move bonus (Weiss `+ Tempo`, Ethereal `Tempo + …`). We know this one; it's the one "feature" both add unconditionally — mentioned for completeness only.

## 5) Portability table

| Mechanism | Verdict vs (a) absolute eval / (b) optimism constraint | Notes |
|---|---|---|
| Correction history (pawn+nonpawn keys, gravity update) | **PORT** | (a) simpler for us — absolute err, no stm sign-flip; (b) de-biaser, pulls optimism DOWN where search disagrees. Piggybacks pawn-hash. |
| |correction| as eval-trust signal for RFP/LMR margins | **PORT** (phase 2) | Non-feature; directly addresses "eval that gates pruning". Our prune-verification harness can validate it by mechanism. |
| EG scale factor (OCB / few-pawns / lone-minor) | **PORT** | Multiplicative shrink toward draw only; cannot add optimism. Cheap (popcounts). Applies to EG half pre-taper. |
| Rule-50 damping | **PORT** | Trivial, whole-eval, shrink-only. |
| Complexity/initiative — shrink half only (`max(c, −|eg|)`, c ≤ 0 branch) | **ADAPT** | Sign-preserving shrink is safe; the POSITIVE half is a disguised feature-add ⇒ port shrink-only. |
| Texel retune of EXISTING weights (AdaGrad, sigmoid-K, outcome labels, 40M+ positions) | **ADAPT** | Calibration not feature-add. Our prior campaign was data-starved, not method-starved. Needs TRACE path + cache-off tuning mode. |
| KS chain-rule gradient for nonlinear aggregates | **ADAPT** | Only if/when we retune KS/latent-threat; methodology note. |
| Pawn(-king) cache | **PORT** (already planned) | Ethereal twist: king squares in key. Speed lane, regression-immune. |
| Packed mg/eg score | SKIP (for now) | Restructure cost ≫ benefit at our NPS bottlenecks (PAWNS 22%). |
| Endgame-table early-out (KPK etc.) | SKIP | We have advanced_endgame_eval; adding recognizers = feature lane. |
| Tempo bonus | **SKIP** | Flat feature-add; exactly the refuted lane. |
| Threat/mobility/complexity POSITIVE terms | **SKIP** | Disguised feature-adds; refuted by our own gauntlets. |
| Search-eval-blended tuning labels | SKIP | Ethereal proves pure outcome suffices; blending imports search's own biases. |

## 6) Top 3 genuinely-not-feature-adds, each with an offline test

**A. Correction history (pawn-key + non-pawn-key, gravity update, absolute-space).**
Build: two tables `corr[stm][16384]`, bonus `clamp((searchAbs−staticAbs)*depth/4, −L, +L)`, gravity update, blend/apply clamped; gate updates to usable bounds, no-check, no-mate.
Offline test: (1) log |staticEval − d8-search score| per position on the failure corpus BEFORE/AFTER (the direct target metric — corrhist must shrink it, especially on the over-push positions where our static agrees with SF11 but search must save us); (2) WAC/STS fixed-depth must not regress; (3) equal-node gauntlet vs SF18@400 (external calibrated venue, baseline 51%). Kill-criterion: if |err| doesn't shrink on held-out positions, the keys are wrong — try minor/major split before abandoning.

**B. Endgame scale-factor + rule-50 damping (whole-eval shrink-only bundle).**
Build: `scaleEG()` — OCB, stronger-side pawn count quadratic, lone-minor; multiply EG component pre-taper; plus final `v*(100−rule50)/100`.
Offline test: (1) draw-adjudicated-loss rate in the gauntlet PGNs (positions we over-pressed drawish endgames); (2) |static − deep| on endgame-phase slices of the failure corpus; (3) equal-node gauntlet. This is the one lever the load-bearing-optimism result actually PREDICTS should help: it removes optimism instead of adding accuracy.

**C. Full-eval Texel RE-calibration (existing weights only, Ethereal recipe).**
Build: TRACE coefficient path over our existing terms, sigmoid-K grid fit, AdaGrad mini-batch on ≥10-20M outcome-labeled self-play positions (multi-TC, quiet-only, dedup) — no new terms, weights only, optionally with shrink-toward-current-weights prior (our one improvement over Ethereal's zero regularization) to bound how far any single weight moves.
Offline test: (1) held-out sigmoid loss (make the validation split Ethereal skips); (2) move-agreement (argmax) vs SF11 per sf11-texel-scale-invariance method — NOT magnitude match; (3) equal-node gauntlet as sole ship gate.

**Single most promising**: **A — correction history**. It attacks our exact diagnosed failure (search-vs-static disagreement on over-push positions), is structurally incapable of being a feature-add, fits the absolute-eval architecture MORE naturally than it fits negamax engines, reuses the planned pawn-hash, and its |correction| by-product feeds the RFP/LMR-margin reliability goal that the prune-verification harness can validate by mechanism.
