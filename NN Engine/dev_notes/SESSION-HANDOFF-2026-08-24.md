# SESSION HANDOFF 2026-08-24 — SEE-captures SHIPPED (+18); KS/passer eval arc = 3 nulls; pivot to SEARCH/corrhist

## ⏱️ READ FIRST — state in one screen
Long session (08-22→08-24). **One real ship: SEE-captures (+18.4 Elo).** Then a full KS + passer eval arc that produced three clean **practical nulls** in diverse games — the proxy (D7 regret) again did not convert to Elo. The forward value is two well-scoped lanes, and the **recommendation is to switch from eval-term work to SEARCH (correction history), using the pawn-loss post-mortems to aim it.**

- **SHIPPED: SEE-capture pruning + C1 fixes** (`f831263`). New default fingerprint **243 / 31,764,817 / EBF 3.729 / STS 1703**. +18.4/2400g diverse UHO, CI[+6,+31]. Stands.
- **KS/passer arc — ALL PRACTICAL NULLS in diverse UHO games:**
  - Bundle (passer+KS): **−4.8 / 1600g**.
  - KS-solo (`KS_EXTEND_EG=1 KS_NO_QUEEN=20 KS_EG_MAT_GATE=1 KS_EG_MAT_LO=12 KS_EG_MAT_HI=20 KS_EG_MAT_FLOOR=25`): **−6.5 / 800g** (additive-KS ~0-for-11; closed).
  - Passer-solo (`ENABLE_PASSER_DETECT_SF=1 PASSER_CANDIDATE_DOCK=64`): **+2.6 / 1600g** (seg1 spiked +16.5 = regression-to-mean noise; settled to a below-floor null).
- **RECOMMENDATION for the new window: pivot to SEARCH (corrhist), not more eval terms** — see §4. Eval degeneracy beat every candidate all session; corrhist is the one lever *orthogonal* to it.

## 1. What shipped (the win)
SEE-capture pruning (`SEE_PRUNE_CAPTURES=1 SEE_PRUNE_CAPTURE_MARGIN=1000`) + C1 correctness fixes (`ENABLE_TT_FLAG_FIX=1 ENABLE_NULL_MATE_CLAMP=1`), flipped to default, committed `f831263`. Nodes −13%. Fixed-depth WAC/STS dropped (243/1703) — EXPECTED for a node-reducer; the +18 is at fixed TIME. Baseline register updated ([[baseline-fingerprints-register]]).

## 2. The passer work — built, banked, null (do NOT re-run as-is)
Built this session (all gated, **byte-id at default**, symmetry-clean, UNCOMMITTED):
- `ENABLE_PASSER_DETECT_SF` — extends `getPPIncrement` detection to SF15.1's candidate-passer cases (lever/leverPush+phalanx/blocked+rank5+safe-support). Closes the measured 8.6%/8.1% rank-5/6 detection gap. **Detection-ALONE over-credits (+0.68 endgame regret)** because candidates flow through the multiplicative R that doesn't dock the un-won lever.
- `PASSER_CANDIDATE_DOCK` — docks R for candidate passers (the un-won-lever signal, a DISTINCT detector). Fixes the over-credit: detection+dock = −0.015..−0.029 aggregate regret, endgame over-credit crushed.
- A **capgains candidate-fix**: `capture_gains` zeroes the speculative passer rank-bonus for a candidate passer (stopper-bound). This fixed a colour-symmetry bug the ship gate caught (root cause = capgains turn-asymmetry reading `pawn_rank_bonuses`, found by the new `diagnostics/_passer_sym_probe.py`). Result: candidate is symmetry-CLEANER than baseline (9<11 violations).
- **Games verdict: practical null** (+2.6/1600g). The code is correct and clean but doesn't convert. Banked gated; available, not shipped.
- ⚠️ `ENABLE_PASSER_ORD_FLOOR` CONFLICTS with the dock (re-lifts candidates) — do not combine.
- Full spec + the multiplicative→additive analysis: `dev_notes/PASSER-REARCHITECTURE-SPEC-2026-08-22.md`. The passer failure law: [[passer-law-multiplicative-vs-additive-and-valuation-graveyard]] (valuation is a graveyard; only DETECTION + the DEFENSIVE side were untested — detection is now tested/null, defensive side still open, see §3).

## 3. FORWARD LANE A — pawn-structure-danger eval (game-proven, but it's eval)
The owner's real concern, from **two odds-game losses from winning positions** where the opponent built winning passers. Reframe (owner): it's a **search-eval interaction** — search reaches pre-passed/passed positions and if the eval mis-judges the pawn-structure danger, search walks into the losing line. Fix the eval's judgment of the *trajectory* (both sides, symmetric — one correct definition fixes over-credit-ours / under-fear-theirs / under-push-ours / over-defend-phantoms together) and search never allows it.
- This is DISTINCT from the null detection+dock candidate (that was clean passers priced too low = the near-promotion CEILING, "Mode 1", which we deferred as the magnitude graveyard).
- **START: `_pgn_walk` the two loss PGNs** (they're in the 08-23/24 conversation; owner can re-paste) to pin the exact term + swing magnitude. Judge on THOSE positions + games, NEVER corpus `under_fire` (anti-correlated).
- ⚠️ CAUTION: this is still eval-term work, which nulled 3-for-3 this session. See the §4 recommendation.

## 4. FORWARD LANE B — SEARCH re-audit (RECOMMENDED next) — "we use these features WRONG"
Full reference: `dev_notes/SEARCH-INFRA-FLAVORS-2026-08-23.md` + memory pointer. Owner reframe: we HAVE most features but they underperform while strong engines gain ⇒ we use them wrong (location/tuning/premature-close). Reference engine = **Weiss** (only modern HCE engine with a 2024 search).
- **★ TOP LEAD: correction history as STATIC-EVAL correction.** `ENABLE_CORR_HIST=0` exists; our past null was **corrhist-in-QSEARCH** ([[corrhist-qsearch-harmful-and-qcache-masks-eval]]) — the VALUABLE form corrects the static eval before it feeds RFP/futility/stand-pat, keyed by pawn-structure hash. **It is ORTHOGONAL to the eval degeneracy that flattened every eval candidate** (corrects net output per position-class, doesn't touch collinear terms), reference-proven on pure HCE (Weiss), and targets our diagnosed static-eval-error problem. Verify our impl's location; build the static-eval-feeding-pruning form (pawn-keyed first); screen + games.
- Then: conthist 1-2ply, threat-indexed main history + malus, pawn-king eval cache (cuts our 65-84% eval cost), capture-hist→SEE-prune (synergizes with shipped SEE-captures).
- Aspiration windows: implementation is CORRECT (audited — no α/β bug), just wide (`ASPIRATION_DELTA=500`); tune via **fixed-NODE-budget screen (deterministic) → fixed-TIME games** (search-efficiency method; fixed-depth is blind, single-run fixed-time is machine-noisy).
- NON-levers confirmed: SIMD (NNUE-only), movegen (our eval dominates node cost).

### THE RECOMMENDATION (owner-endorsed direction, 2026-08-24)
**Pivot to SEARCH/corrhist, open with the `_pgn_walk` post-mortems as the bridge.** Rationale: eval-term work is 3-for-3 null this session and fights the degeneracy; corrhist is the one orthogonal, reference-proven, diagnosis-matched lever. If the post-mortems show the losses are pawn-structure static-eval misjudgments (likely), a **pawn-structure-keyed corrhist is the UNIFIED fix** — it addresses the passer/pawn-danger concern THROUGH the search lane, dodging the degeneracy. Order: (1) `_pgn_walk` 2 losses → (2) corrhist static-eval form → (3) pawn-danger eval-terms only if corrhist can't reach it.

## 5. Gyatso + HCE→NNUE (context, not action)
Gyatso (~3300 CCRL) = **NNUE, not an eval flavor** — useful only as a gauntlet opponent (grab a release binary). HCE→NNUE path (owner's endgame): Viridithas template (train first net on own HCE self-play), `bullet` trainer, ~768→N perspective net. ⚠️ Field says you do NOT need 2800-3000 HCE first — 2200-2500 labels leapfrog; self-play data we're banking is the asset. Details in the SEARCH-INFRA-FLAVORS dev-note §3.

## 6. UNCOMMITTED / housekeeping (owner-approval)
All UNCOMMITTED, gated/byte-id at default (nothing changes the shipped engine):
- Passer code: `cpp_bitboard.cpp` (detect_sf + candidate_dock + capgains-candidate-fix), `search_engine.{h,cpp}` (flags). `diagnostics/_passer_sym_probe.py`, `_build_game_regret_set.py` GAMES_GLOB, `_ks_phase_split.py` PDETECT/EGCAND/CAPGD/BUNDLE modes.
- Dev-notes: PASSER-REARCHITECTURE-SPEC, SEARCH-INFRA-FLAVORS, this handoff.
- **Housekeeping queue:** commit the above; diagnostics prune (~50 scripts, audit done — the prune list is in the 08-22 conversation / re-runnable); **MEMORY.md compaction (21.6KB, over the 17.1KB target — genuinely needed)**; resolve `KS_BATTERY` phantom (declared/registered/unwired).
- New de-biased instrument built: `diagnostics/ks_sets/game_regret_set_uho.csv` (6000 SF18-labeled UHO positions) — use for eval screening, NEVER corpus under_fire.

## 7. Disciplines (unchanged)
GAMES decide on DIVERSE UHO + varied seed, segmented (~400g), pooled, run ALONE, ONE core-loading job (no `nohup` — orphans → OOM; use tracked run_in_background, [[nohup-orphans-a-games-job-and-relaunch-double-cores-oom]]). ~24h per candidate, inconclusive = soft-null → pivot ([[resolve-builds-within-24h-keep-moving]]). Judge only CLOSED segments (mid-segment swings ±40 — over-called twice this session). Byte-id every build; colour-symmetry ship gate for eval. Proxy anti-correlated with Elo — screen cheap, games arbitrate. No commits/ships without owner confirm.
