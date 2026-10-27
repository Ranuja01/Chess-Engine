# Passer re-architecture spec — additive form + candidate-realizability, on the RIGHT instrument (2026-08-22)

**One-line:** the additive machinery already exists in the code, parked at 0; the real work is (a) reopening it on the de-biased instrument where it was never judged, and (b) adding the ONE missing input — a realizability dock for *candidate* passers — that makes the SF detection extension safe. This is a **re-shape of the existing owner** (like the OvD feeder-partition), NOT a 5th valuation mechanism.

## 0. Read first — the failure law this must obey
Memory `passer-law-multiplicative-vs-additive-and-valuation-graveyard`: passer VALUATION is a graveyard (4 mechanisms dead + "no 5th"); errors are two-sided so any magnitude knob that closes corpus `under_fire` lifts the over-read guard equally; corpus + STS are anti-correlated with Elo; no master off-switch ⇒ fractional ablations. **This spec is not a valuation add — it changes the FUNCTIONAL FORM of the existing term and adds a detection-conditioned realizability input. Judge it on the de-biased regret set + diverse games, NEVER corpus `under_fire` or STS.**

## 1. The three DISTINCT failure modes (do not conflate — they need different fixes)
- **Mode 1 — near-promotion ceiling too low.** A clean 7th-rank passer maxes ~1.56 pawns (`passed/endgame_pawn_rank_bonus[·]` × `PASSER_MAG_SCALE`), so up a rook it never reads as catastrophic (the odds-game losses). This is an **absolute-magnitude** problem = the graveyard (raising it flattens; `passer_danger` is a double-count). **OUT OF SCOPE here** — do not chase it.
- **Mode 2 — R-collapse under-read.** `val = mag × R/256` with `R` collapsing to ~0 on a contested passer prices it *below an ordinary pawn* (code comment `cpp_bitboard.cpp:6659-6666`; the 08-18 `passed_pawn_support −4.55` live loss). Fix = **additive base floor** (already built: `PASSER_RESID_PCT`, `ENABLE_PASSER_ORD_FLOOR`, `PASSER_RFLOOR_R5/R6`).
- **Mode 3 — candidate-passer over-credit.** The SF detection extension (`ENABLE_PASSER_DETECT_SF`, built 2026-08-22) flags candidate passers (stopper is a capturable/pushable pawn), but `passer_realizability_R` (`:6519`) docks only enemy PIECE attackers / blockade / rear / king — **nothing docks the un-won lever/push**, so a candidate is priced like a clean passer. Measured: detection-alone = **+0.126 aggregate / +0.684 endgame** regret on the de-biased UHO set (worse). Fix = **a candidate-conditioned realizability dock** (NEW input to R, keyed on the detection predicate).

## 2. Current architecture (grounded in code)
`evaluate_passers` (`cpp_bitboard.cpp:6630`): per passer,
```
mag  = phaseblend(passed_mid[rank], endgame[rank]) * PASSER_MAG_SCALE/100
R    = min(passer_realizability_R(sq,white), PASSER_R_CAP)          // :6642
base = mag * PASSER_RESID_PCT/100                                   // :6658  (0 => pure multiplicative)
if ENABLE_PASSER_ORD_FLOOR: base = min(max(base, default_rank[rank]), mag)   // :6667
val  = base + (mag - base) * R/256                                  // :6671  THE dial: base share is additive, rest R-gated
```
`passer_realizability_R` (`:6519`): `R=256`, minus `passer_block_quality`, minus graded path-contest (net enemy−own attacker popcounts, stop-square weighted `PASSER_CONTEST_STOP`/`PASSER_CONTEST_PATH`), plus/minus rear heavy-piece (`PASSER_REAR_ENEMY/OWN`), plus king-proximity (`PASSER_KING_FAR/HELP`), `clamp(0, PASSER_R_MAX=384)`. **It never sees the stopper pawn** — only pieces.
So: `PASSER_RESID_PCT` IS the multiplicative(0)↔additive(100) dial, and it + `ORD_FLOOR` + `RFLOOR` are the parked Mode-2 fixes.

## 3. Where we went wrong (so we do better this time)
From the failure autopsy (memory + PAWN_MODEL refutation record):
1. **Judged the additive machinery on the WRONG instrument.** `PASSER_RESID_PCT`/`ORD_FLOOR`/`RFLOOR` were parked on **corpus (−0.6/worse) and STS (ORD_FLOOR −82)** — both anti-correlated with Elo. **Never** run on the de-biased UHO regret set or diverse games. Their "null" is a wrong-instrument null, not a games null → reopenable.
2. **The one residual form actually tried was inverted.** The blockade-scaled residual `(BLOCK_MAX−blk)` gave the base to UNBLOCKED passers and withheld it from blockaded ones — backwards from SF (comment `:6655-6657`). The FLAT residual (`PASSER_RESID_PCT` as-is) is the SF-correct form and is the one to test.
3. **Magnitude tuning flattens** (Mode-1 chasing): `PASSER_MAG_SCALE=150` → STS −54; fitted tables → −15 Elo. Two-sided error. Don't.
4. **No master off-switch ⇒ fractional ablations** (`SCALE_PASSED_PAWN`=18%, `PASSER_MAG_SCALE`=32%). Screen the FULL config, not a single knob.
5. **Detection never co-designed with realizability** — we just proved (Mode 3) that adding detection without the candidate-R dock over-credits. Detection + pricing are ONE definition.
6. **Manufactured instruments wrong 2.6×; `br_pt_pawns` vs SF `Passed` apples-to-oranges** — validate on real positions only.

## 4. The re-architecture (how we improve, as ONE symmetric definition)
Three coupled changes, each byte-identical at default, screened together:

**(A) Additive base — reopen Mode-2 fix.** Sweep `PASSER_RESID_PCT ∈ {25, 40, 60}` (flat, unscaled — the SF-correct form) and separately `ENABLE_PASSER_ORD_FLOOR=1` (the aggregate-neutral floor). These grant the rank bonus as a FLOOR so `R` can never zero a real passer. Symmetric by construction (both colours, opposite sign).

**(B) Candidate realizability dock — NEW input, the Mode-3 fix that makes detection safe.** When `ENABLE_PASSER_DETECT_SF` flags a pawn passed *only via a candidate case* (it has a stopper pawn), dock `R` by a `PASSER_CANDIDATE_DOCK` reflecting the un-won lever/push (the pawn is passed CONDITIONAL on winning that pawn battle). Implementation: pass a `is_candidate` bit from `getPPIncrement` into `passer_realizability_R` (or dock at the `evaluate_passers` call site), so a candidate starts lower than a clean passer. This is a DISTINCT detector (the stopper-pawn relationship), not a re-scale of existing R terms — OvD-clean.

**(C) Detection** — `ENABLE_PASSER_DETECT_SF=1` (built, byte-id) supplies the candidate passers that (B) then prices correctly and (A) floors.

The single-definition property: (A)+(B) are checked identically for both sides ⇒ one correct definition fixes over-credit our ghosts, under-fear their runners, under-push our edge, over-defend phantoms — together.

## 5. What this explicitly does NOT do
- NOT raise the near-promotion magnitude ceiling (Mode 1 = graveyard).
- NOT a new valuation term — it reshapes the existing `val` form and adds one realizability input.
- NOT judged on corpus `under_fire` or STS (anti-correlated). Those may MOVE; ignore them as arbiters.

## 6. Validation ladder (in order)
1. **Byte-identity at default** — all new knobs off/0 ⇒ `243 / 31,764,817 / EBF 3.729 / STS 1703`.
2. **Colour-symmetry ship gate** — `_eval_symmetry.py N=800` with the full on-config (mirror = −eval).
3. **De-biased regret** — `_ks_phase_split.py` new mode (base vs the A+B+C bundle) on `game_regret_set_uho.csv`, read the phase/npm bands. TARGET: the detection-extension's +0.68 endgame regret goes to ≤0 (B fixes the over-credit) while advanced-passer bands improve. This is the Elo-bearing screen.
4. **Detection co-design check** — confirm candidate passers are priced BELOW clean passers of the same rank (a probe via `g_passer_probe`/`priced_passer`).
5. **Offense sanity** — passer_movematch on real positions: does the engine pick more SF-best push/support/clearance moves.
6. **Diverse games decide** — UHO + varied seed, ~24h, ALONE, then BUNDLE with the parked KS-endgame candidate (distinct detectors, confirmed safe). Per `resolve-builds-within-24h-keep-moving`: inconclusive at 24h ⇒ soft-null, pivot.

## 7. Knobs (all default byte-id)
`ENABLE_PASSER_DETECT_SF` (built), `PASSER_RESID_PCT` (0; sweep 25/40/60), `ENABLE_PASSER_ORD_FLOOR` (0/1), `PASSER_CANDIDATE_DOCK` (NEW, default 0 = byte-id), plus the existing `PASSER_R_CAP`/`PASSER_CONTEST_*` as context (do NOT sweep as magnitude). Screen the FULL config together (no-master-off-switch law).

## 8. Bottom line
The additive form is not a new idea to build — it's `PASSER_RESID_PCT`, already in the code, parked because it was judged on anti-correlated instruments. The genuinely new, small piece is the **candidate-realizability dock** that makes the SF detection extension safe (fixing the measured +0.68 endgame over-credit). Build (B), reopen (A) on the de-biased regret set with (C) on, screen as one symmetric definition, and let diverse games — not corpus — decide.
