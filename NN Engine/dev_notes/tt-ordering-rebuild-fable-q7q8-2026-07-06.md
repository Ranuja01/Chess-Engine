# TT rebuild + move-ordering campaign (Fable Q7/Q8, 2026-07-06) — with OUR-CODE verification annotations

One campaign (TT feeds the ordering fix). **Verification status per claim marked ✓/⚠️** — user directive: don't blindly agree.
Companion to [[sf-source-evolution-bank]], [[search-soundness-fable-q3-2026-07-06]].

## Q7 — TT REBUILD
### Consolidation — merge two of four, keep two; DO NOT copy SF's 10-byte int16 entry
⚠️→✓ **VERIFIED our scale forbids int16.** Current `TTEntry` (cache_management.h:69): `{key(8), int score, int depth,
TTFlag flag, int alpha=-9999999, int beta=9999999, bool valid}` — **dead alpha/beta CONFIRMED, no move/eval/aging field.**
Scores are millipawn (pawn=1000, sentinels ±9,999,999, SEE least_value=999,999,999) → **int16 (±32767) can't hold winning/
mate scores.** So SF's value16/eval16 is WRONG for us. SPEC: 16-byte entry `{key16(2), move(3: from,to,promo), genBound8(1),
depth8(1), pad(1), value32(4), eval32(4)}`, **4-way cluster = one 64-byte cache line** (gives TT_WAYS real semantics; keeps
one-line-per-probe). `evalCacheNew` → absorbed into eval32 (one probe serves cutoff + static eval). `moveGenCache` → KEEP as
a generation-cost memo (also serves is_checkmate/stalemate); its hash-move role migrates to move16. `quiesceEvalCache` → keep
in v1, merge later (depth 0/−1 sentinels).

### Non-negamax bounds — we're in the EASY case (verify one sub-claim)
✓ Load-bearing: our scores are FIXED-ORIENTATION (absolute, flipped ONCE by side_to_play), so bound semantics are a property
of the WINDOW not the mover: `v ≤ alpha_orig` = UPPER, `v ≥ beta_orig` = LOWER, else EXACT — orientation never enters →
**min and max nodes use byte-identical TT code, NO sign-flipping** (unlike negamax). Hazards that ARE real: (a) code drift
across minimizer/maximizer/pre_minimizer → fix with ONE shared `tt_probe()`/`tt_store(value,alpha_orig,beta_orig,depth,move,
staticEval)` (= migration Stage 0). (b) eval32 stored in the SAME flipped space as value32; route reads through one accessor
(we shipped the static_eval_for_improving raw-vs-flipped bug once). (c) side_to_play constancy — process-per-side guarantees
it; add a cheap "clear TT if side_to_play changes in-process" guard for GUI/analysis. (d) mate scores — keep NOT storing in
v1. (e) pre_minimizer uses the same shared store with its honest depth.
⚠️ TO-VERIFY before Stage 1: Fable says "use_tt_entry never READS alpha/beta so dropping them is byte-identical." The struct
STORES them (addToSearchEvalCache passes alpha/beta). Confirm they're never READ in a live path before assuming byte-id.

### Replacement/aging + 5-stage migration (each stage its own checkpoint)
SF rule: 5-bit gen in genBound8, `gen += 8` per get_engine_move; `relative_age = (CYCLE + curGen − entryGen) & 0xF8`; victim
= cluster entry minimizing `depth8 − relative_age`; on save overwrite same-key when `newDepth > oldDepth − 4` OR new bound
EXACT; preserve stored move when new save has none.
- **Stage 0** shared probe/store refactor — zero behavior change → **checkpoint byte-id 245.** (Kills drift.)
- **Stage 1** repack (drop dead alpha/beta, 16B layout, still direct-mapped, no aging, move16 present-unused) — byte-id since
  use_tt_entry doesn't read alpha/beta (⚠️verify) → **byte-id.**
- **Stage 2** 4-way clusters + depth-preferred victim, gen frozen — behavioral → **checkpoint STS/WAC held, nodes flat-down,
  quick node_ab ≥ 0.**
- **Stage 3** aging on — KEY: single-position benches CAN'T see this (one search = one gen → byte-id to Stage 2 on WAC/STS by
  construction). **Checkpoint must be GAME-shaped:** node_ab (small +) + per-move node-count trace across one full game (working
  aging = nodes-per-move stops creeping up in the late middlegame as the table silts).
- **Stage 4** move16 live — store best move at the two node-local accept points (cutoff, best-raise) keyed by the NODE's own
  hash (⚠️ real work: current stores are CHILD-keyed from the parent), order it index-0 behind key+pseudo-legality. →
  **full node_ab gate.** Gateway to IIR + singular (separate campaigns).
- **Stage 5** eval32 live, then optional qsearch-cache merge.

### eval32 read/write (the payoff, not the packing)
WRITE: every tt_store includes the node static eval whenever computed (incl. fail-low + no-move stores; SF saves eval-only
entries). Store RAW static eval, flipped-space. READ: node entry → on TT hit `staticEval = entry.eval32` → **shipped RFP +
futility-in-LMR run with ZERO eval calls at hit nodes** (THE payoff). ttValue-refines-eval (free accuracy): if entry LOWERBOUND
and value32 > staticEval, use value32 as working eval for RFP/improving (UPPER/< mirror at min nodes). Feed g_evalStack[cur_depth]
from eval32 → makes re-testing ENABLE_IMPROVING (post sign-fix) ~free + enables hindsight/opponentWorsening.

## Q8 — ORDERING + the frozen cache
### Tail re-sort — why FULL re-sort regressed, and the correct scope
✓ Mechanism sound: cached front = EVIDENCE (updateMoveCacheForBetaCutoff = "this exact position cut off with this move");
history = a noisy PRIOR (mid-search high-variance, polluted by transient killers + root-PV moveFrequency bonus). Full re-sort
OVERWRITES evidence with prior → worse ordering avg + churns which moves get LMR-reduced visit-to-visit → destabilizes the
reductions STS measures at fixed depth. **The STS regression was the CORRECT verdict on the WRONG scope.** CORRECT scope:
- WHAT: re-score ONLY the never-tried quiet TAIL (behind the proven prefix: promoted cutoff move, killers, tried moves).
  Leave captures + evidence prefix untouched. The buried refutation IS in the never-cut tail (phantom-loop definition).
- WHERE: extend beyond `lci > 2` to FAIL-LOW (all-)nodes — add a "last visit returned fail-low" bit to the cache entry.
  Phantom-supporting nodes are all-nodes whose refutation never surfaced. Cutoff-happy nodes don't need it (96% FMC).
- HOW OFTEN: once per ID iteration per node (tag entry with iteration#, refresh tail only when stale) — stability WITHIN an
  iteration (no reduction churn, protects STS) + freshness ACROSS iterations (restores the self-correction the frozen cache
  disabled). `partial_sort` top-K (~8) of the tail, not a full sort.
- KPI: gate on PHANTOM RATE (fraction with search−static > 1.5p — now measurable) + node_ab, NOT STS (2 causal steps away).

### Staged MovePicker — the structural fix (middle path, not a rip-out)
Insight: SF has NO move-order cache — every visit re-scores quiets from live history — and STAGING is what makes that
affordable (at 96% FMC most visits end at TT-move/good-capture, never touching the quiet list). So staged picking is the
architecture in which the frozen-order problem CANNOT EXIST. MIDDLE PATH: keep moveGenCache as an UNSORTED move memo (still
amortizes the 16-pass generation — fold in the blockers/checkers hoist + single-pass gen while there), layer a picker on top:
(1) TT-move verified, no list touch; (2) captures by score + SEE split; (3) killers/counter; (4) score quiet tail from LIVE
history at that moment w/ partial selection. Then Q8.1's re-sort becomes a deletable transitional patch. Non-negamax fine
(ordering orientation-free; history side-indexed).

### SEE split + sort cutoff — ⚠️ VERIFY the SEE-fix precondition FIRST
✓ Our structure (good captures skip SEE / unclear SEE-tested / SEE-negative below quiets, bad-last) matches SF GOOD/BAD_CAPTURE.
Refinements: SF threshold is `see_ge(m, −55·score/1024)` not ≥0 (tolerate slightly-losing captures of valuable victims — pure
≥0 misclassifies sound sacs, mildly phantom-relevant) — one sweep. `−3000·depth` partial-sort cutoff is SF-unit-bound → port
the CONCEPT (sort depth-proportionally more of the tail deeper), calibrate the constant to OUR history spread once staged.
⚠️⚠️ **PRECONDITION — Fable says "fix SEE first (June one-sided-attacker-refresh + stale-square_values LVA bug)." STATUS
UNRESOLVED:** the bug was empirically confirmed June-11 (self-check `see_test.cpp` exists), BUT we carry `ENABLE_SEE_FIX=1` +
`ENABLE_SEE_INCREMENTAL=1` default-on and the square_values positional adjusts are commented out — **so it may ALREADY be
fixed and Fable's precondition is stale.** ACTION: RE-RUN the June SEE self-check (assert see==ref_see) BEFORE treating
"fix SEE" as a task. Don't tune the split/threshold on a possibly-still-2%-wrong SEE — but also don't re-fix an already-fixed bug.

## SEQUENCING (Fable, adjusted for our verification)
Q7 Stage 0-2 (byte-id-or-near prep) land THIS WEEK alongside a SEE re-check → Q8.1 tail re-sort = the fast phantom patch (gate
phantom-rate + node_ab) → Q7 Stage 4 (TT move) feeds Q8.2 picker + later singular. Every stage its own checkpoint.

## Q9/Q10 — architecture + blind spots (verified/annotated)
**⚠️ d3/d4 null-move gate is NOT an asymmetry — VERIFIED, Fable ERRED.** Root (alpha_beta, cur_depth 0) = max, calls
`minimizer` for children (1608) → strict alternation → **minimizer only at ODD cur_depths, maximizer only at EVEN.** Gates:
minimizer null-move `cur_depth≥3` (MINI=3, line 2840), maximizer `cur_depth≥4` (MAXI=4, line 3328). Since maximizer has NO
node at odd depth 3, `maxi≥4 ≡ maxi≥3` → the config = uniform "null-move for every node at depth ≥3" = SYMMETRIC. The
asymmetric constants ENCODE symmetric behavior via parity; NOT a bug. **⇒ Fable's whole "asymmetry bug ledger" needs
PER-ITEM parity verification before any counts as a bug / Stage-A task** (improving-sign was independently confirmed real
June; futility-gate + PVS-window items = UNVERIFIED, spot-check each vs the parity structure first).

**Non-negamax: direct Elo cost ≈ 0 (min/max ≡ negamax mathematically); realized cost = maintainability + the port-tax
(mechanisms built timidly / deferred because 2-3× — e.g. singular).** REFACTOR = "negamax-LITE" in 2 stages, NOT true
negamax (true negamax forfeits our free sign-less TT bounds + churns the whole score space): **Stage A** symmetrize the REAL
asymmetries in place (one at a time, bench each; deliverable = mirror A/A test green, into the permanent bench signature);
**Stage B** fold to one `template<int Sign>` direction-parameterized search fn, byte-identical vs Stage A (mechanical once A
makes them true mirrors; the risk lives in A, sliced into benchable pieces). Do Stage A NOW, fold BEFORE TT-move/singular.
Snag: minimizer's `cur_depth==1` special branch (second-level lists, fed by pre_minimizer) is a TRUE asymmetry, can't fold —
dies with pre_minimizer.
**pre_minimizer:** NOT a wart (its depth−1 root re-search = razor-fuel + ply-2 ordering lists + TT warming + root-tail order;
its removal measured +32% confounded). But every product gets a cheaper replacement at the TT-move stage (TT-move+IIR =
root/ply-2 ordering+warm seed; razor→"prev-iteration real score else skip"; cur_depth==1 branch → normal path). **DELETE as
the closing step of the TT-move campaign** (node_ab before/after) → every port 3×→2×; Stage B → 2×→1×. End state: one search
fn, one qsearch, one TT interface.

**TIME MANAGEMENT (Q10 blind spot) — DEFERRED by user (absolute strength first).** We have NONE sophisticated (static
per-depth `MOVE_TIMES`; hard `TIME_LIMIT` mid-iteration discards partial work = burns 30-50% of budget on a thrown-away last
iteration). Real: naive-vs-good TM = 30-80+ Elo at fast TC; lives OUTSIDE the search core (no sign-mirroring, no port-stack
interaction). PARKED until absolute strength is up; then order = don't-start-what-you-can't-finish → instability-aware
spending (bestMoveChanges/fail-low panic = a TIME-SIDE phantom mitigation) → game-clock allocation. Fable code-confirmed the gap.

**RUNNER-UP BLIND SPOTS (absolute-strength-relevant, KEEP):** (a) **TC-scaling validation** — whole roadmap adjudicated at
lightning/fixed-node but goal = 3000 @ STANDARD; reductions/extensions FLIP SIGN with depth → spot-check top 2-3 shipped
gains at one slower TC overnight before declaring shipped. (b) **KPK bitbase** (~200 lines, exact won/drawn every K+P-vs-K
leaf) = standard 2700→3000 endgame-conversion patch, cheap, byte-auditable, no NNUE (extends our is_practically_drawn cliff
handling). (c) **`make regress` suite** = mirror A/A search test + eval-mirror probe + SEE self-check as ONE build gate → no
fixed bug class silently returns (an afternoon, protects everything built this month). Full Q9/Q10 in-chat 2026-07-06.
