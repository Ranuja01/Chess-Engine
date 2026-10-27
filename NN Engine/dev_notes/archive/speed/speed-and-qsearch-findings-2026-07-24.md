# Speed lane measured + the qsearch question (2026-07-24)

## Throughput vs the reference engines (measured on this machine, single-thread `bench`)
| engine | NPS | vs ours |
|---|---|---|
| SF11 (classical) | **2,527,826** | 5.1× faster |
| SF15, NNUE off (classical) | **1,326,973** | 2.7× faster |
| ours (d12 stratified midgame) | ~497,000 | — |

**Our static eval ≈ 5,900 cycles/call** (derived from the `EVAL_PROFILE` build: `CAPTURE_GAINS` 766 cyc/call at
13.0% share ⇒ 766/0.13; cross-checks against PAWNS 27.3% ÷ 126 cyc/call ≈ 12.8 pawn calls). At ~3.5-4.5 GHz that
puts eval at **~65-84% of our per-node cost**, i.e. **all search machinery (movegen/make-unmake/TT/search logic)
is only ~15-35%**. Our eval ALONE costs more than SF11's ENTIRE node (~1,400 cyc) — the NPS gap is essentially
all eval, with no throughput worth reclaiming on the search side.

## Eval-cost profile — no hotspot exists
PAWNS 27.3% (126 cyc/call) · ROOKS 15.2% (247) · BISHOPS 14.1% (279) · **CAPTURE_GAINS 13.0% (766)** ·
QUEENS 8.5% (287) · KNIGHTS 7.2% · KING_SAFETY 5.7% (364) · rest <4%.
Cost is SPREAD — the signature of an eval that is expensive by design (many moderate terms), not one with a
hot spot. Nothing can be cut for a big win.

## Speed levers MEASURED and REJECTED (do not re-litigate without new evidence)
- **Pawn-hash eval cache: NO-GO.** `PROF_PAWN_PPINC` (the passed-span analysis = the cacheable part) is
  **30 cyc/call of PAWNS' 126** ⇒ only ~24% of pawn eval ⇒ **~6.5% of eval**. Not worth fable's full
  dependency/risk list (per-pawn structural scalars because of the `min(225)` saturation coupling, per-colour
  lookup-count arrays, partial-mask replay, blend-zone double mutation, config-generation invalidation, 11
  silent-corruption risks). **The pawn KEY (`generatePawnKey`, cache_management.h:521) still exists** — so
  PawnHistory / KS-shelter caching remain separately available; only the eval-content cache is dead.
- **Lazy capgains: NO-GO.** Probe (`CAPG_LAZY_PROBE`, 13.5M calls in one WAC): tension=0 in **15.2%** of calls
  and those are 100% negligible; ~36% of ALL calls yield |cg|<50. But wall-clock says the whole term is worth
  only **+1.4% NPS** (see below), so even a PERFECT skip-oracle is worth ~0.5%. The 766 cyc/call is misleading:
  capgains fires ONCE per eval while PAWNS fires ~13×.

## ★ Capgains ablation — it is a PRUNING-SAFETY device, not a qsearch substitute
Added a byte-identical guard: `SCALE_CAPTURE_GAINS=0` now SKIPS the simulation (zero-weighted term ⇒ don't
compute). Byte-id 248/44,038,704 preserved at any nonzero scale.

| config | WAC | nodes | NPS |
|---|---|---|---|
| baseline | 248 | 44.0M | 482,827 |
| `QSTANDPAT_EVAL_MODE=2` (drop capgains+passed-support+latent+adv-endgame **at qsearch leaves only**) | 235 (−13) | 49.3M (**+12%**) | 487,156 (+0.9%) |
| `SCALE_CAPTURE_GAINS=0` (capgains off **everywhere**) | 233 (−15) | 64.0M (**+45%**) | 489,548 (**+1.4%**) |

1. **Capgains' value is concentrated at INTERIOR nodes** (dropping it everywhere = +45% nodes vs +12% for four
   terms at leaves) — pruning safety for RFP/futility/null-move: [[eval-accuracy-payoff-is-pruning]].
2. ⚠️ **CORRECTION (fable audit):** I first read the mode-2 result as "the code's ≈0-at-quiescent-leaves bet is
   false / our leaves aren't quiet." **That was wrong.** Mode 2 *"removed the oracle while leaving the
   oracle-dependent prunes in place"* — the −13 solves is the signature of the UNSOUND DELTA PRUNE below, not
   evidence about leaf quietness. Re-test only AFTER fixing the prune.

## ★★ ROOT CAUSE FOUND (fable qsearch audit) — our delta pruning is UNSOUND, capgains MASKS it
Ours is **node-level** (`search_engine.cpp:5383-5384`, minimizer mirror 5392-5393):
```cpp
if (static_eval > alpha) alpha = static_eval;
if (ENABLE_QDELTA && static_eval < alpha - Config::DELTA_MARGIN)
    return static_eval;            // skips ALL captures at this node
```
SF is **per-move and CREDITS THE VICTIM FIRST** (SF11 `search.cpp:1447,1473-1492`; SF15.1 `:1478,1519-1534`):
```cpp
futilityBase = bestValue + 154;
futilityValue = futilityBase + PieceValue[EG][pos.piece_on(to_sq(move))];
if (futilityValue <= alpha) continue;
```
SF's ~150cp margin sits ON TOP of the captured piece's value; **ours REPLACES it** ⇒ a node 1.5 pawns below
alpha skips EVERY capture, **including a hanging queen** — unsound by up to ~7500mp per node. It only "works"
because stand-pat already banks the pending capture material via `approximate_capture_gains`. **That is why
removing capgains costs 15 solves / +45% nodes: it is plugging this hole, not duplicating qsearch.**

Other defects found: **`MAX_QDEPTH` cap is tested BEFORE the in-check branch** (`SE:5277`) ⇒ an in-check node at
q-depth 10 returns a raw static eval of an in-check position (mate-blind), and it is **cached with no
truncation tag** (`SE:6387`). We search **quiet checks at ALL 10 q-plies** (SF: first q-ply only,
`DEPTH_QS_CHECKS=0`) via a full board-copy + `is_check` per quiet move — expensive, but plausibly part of why
WAC is high, so judge on games not WAC. SEE filter is per-SQUARE not per-move (admits individually-losing
captures); no TT move in qsearch (QCache stores score+flag only).

## ★ VERIFICATION ROUND (2026-07-24 later) — theory CONFIRMED, but my first "fix" was INERT
Two errors made and caught; record them so nobody repeats either.
1. **Wrong env name.** Ran `wac ... QDELTA=0`, but the registration is `env_flag("ENABLE_QDELTA", true)` — the
   `[toggles]` dump label (`QDELTA`) is DISPLAY-ONLY. The knob was **silently ignored**, the run reproduced the
   baseline byte-for-byte, and I briefly concluded "delta pruning is dead code." **It is not.** See
   [[env-knob-name-verify]]. **Rule: a byte-identical A/B usually means the knob was IGNORED, not that the
   feature is inert.**
2. **`ENABLE_QDELTA_PERMOVE` (my per-capture futility) NEVER FIRES.** `ENABLE_QDELTA_PERMOVE=1` produces
   **exactly** the `ENABLE_QDELTA=0` result (249 / 47,387,950) ⇒ it only disables the old prune via its guard
   and contributes nothing; margin-independence at DELTA_MARGIN 1500/700/300 was the tell. Left GATED OFF
   (default false, byte-id 248/44,038,704 verified). Needs a fire-counter before any further claim.

**Correctly-labelled results (WAC, fixed depth 10, baseline 248 / 44,038,704):**
| delta | stand-pat | WAC | nodes |
|---|---|---|---|
| ON | full | 248 | 44.0M |
| **OFF** (`ENABLE_QDELTA=0`) | full | **249** | 47.4M |
| ON | **light** (`QSTANDPAT_EVAL_MODE=2`) | **235 (−13)** | 49.3M |
| **OFF** | **light** | **248 (−1)** | 55.4M |

⇒ **The −13 penalty appears ONLY when delta pruning is ON.** That is exactly fable's mechanism: the delta test
`static_eval < alpha − DELTA_MARGIN` is calibrated to a stand-pat that capgains INFLATES; strip capgains and
stand-pat drops, so delta over-prunes and loses 13 solves. **Theory CONFIRMED — demonstrated by disabling delta,
not by my broken replacement.** Also established: **delta pruning is ACTIVE and cheap-ish — it saves ~7.6% nodes
for 1 WAC solve.**

## ★ PRUNING / ORDERING INVENTORY (each flag turned OFF; WAC d10, baseline 248 / 44,038,704)
| mechanism OFF | WAC | nodes | effect of having it ON | nodes saved per solve lost |
|---|---|---|---|---|
| `ENABLE_LMR=0` | 263 (+15) | 289.5M | **6.58× fewer nodes**, −15 solves | **16.4M** |
| `ENABLE_LMP=0` | 249 (+1) | 78.3M | 1.78× fewer, −1 solve | **34.3M** ← best ratio |
| `ENABLE_NULLMOVE=0` | 243 (−5) | 65.7M | 1.49× fewer **AND +5 solves** | FREE |
| `ENABLE_RAZORING=0` | 252 (+4) | 55.4M | 1.26× fewer, −4 solves | 2.85M ← **worst ratio** |
| `ENABLE_FUTILITY=0` | 246 (−2) | 48.5M | 1.10× fewer **AND +2 solves** | FREE |
| `ENABLE_QDELTA=0` | 249 (+1) | 47.4M | 1.08× fewer, −1 solve | 3.4M |
| `ENABLE_STATSCORE_LMR=0` | 236 (−12) | 37.2M | +12 solves for +18% nodes | (accuracy buy) |
| `ENABLE_HISTORY_LMR=0` | 244 (−4) | 40.0M | +4 solves for +10% nodes | (accuracy buy) |
| `ENABLE_LAZY_RESORT=0` | 248 (=) | **40.7M** | ⚠️ **+8% nodes for ZERO solves** | pure cost @ d10 |
| `ENABLE_CONT_HIST=0` | 248 | **44,038,704 IDENTICAL** | ⚠️ **INERT — never fires** | none |

**Reads:**
- **LMP is our most efficient pruner** (34.3M nodes/solve), **LMR the most powerful** (6.58×) and 2nd most
  efficient. LMR's −15 solves is essentially the WHOLE accuracy cost of the pruning stack ⇒ *making LMR
  reductions safer is the single highest-value search lever* (= the roadmap's ordering × prune-push pair, now
  with a measured target).
- **Null-move and futility are FREE** — fewer nodes AND better accuracy. No trade.
- **Razoring is the worst-value mechanism** (4 solves for only 1.26×) — retune-or-remove candidate; its margins
  may be another "calibrated for the old eval" case.
- **`ENABLE_LAZY_RESORT` costs 8% nodes for nothing at d10** (it shipped as part of the lazy-hybrid+LMP bundle
  measured in GAMES, so re-judge in games before touching — but flag it).
- ⚠️ **`ENABLE_CONT_HIST` is INERT** (byte-identical). Name verified (`env_flag("ENABLE_CONT_HIST")`, :1222), so
  this is NOT the mistyped-knob trap. Its ONLY functional use is `search_engine.cpp:665`: for a tier-0 quiet,
  cancel the extra LMR reduction if `counterMoveHeuristics[...] >= CONT_HIST_LMR_THRESH (2000)`. Byte-identity
  ⇒ **that threshold is never reached** — the direct consequence of the known **history UNDER-FILL**
  (`counterMoveHeuristics[2][4096][4096]` sparse + `DECAY_INTERVAL=35000` halving ~7×/search ⇒ mostly 0).
  **RESOLVED — it is UNREACHABLE DEAD CODE, not under-fill.** `CONT_HIST_LMR_THRESH` at 500/100/20 is
  byte-identical too (so not a tuning issue). The reduction helper returns inside the
  `ENABLE_STATSCORE_LMR` block (`search_engine.cpp:643`); `int tier = ...` and the cont-hist rescue start at
  **:646, AFTER that return**. With `ENABLE_STATSCORE_LMR=1` (shipped default, +23 Elo) the legacy tier path is
  never reached. **Proof:** with `ENABLE_STATSCORE_LMR=0`, cont-hist ON = 236/37,224,400 vs OFF =
  242/38,922,936 — it bites only in the legacy regime. (And it is a BAD mechanism even there: ON costs 6 solves
  for 4.6% nodes — worse than razoring.) **Nothing was lost:** statScore already sums 1-ply and 2-ply
  continuation history via `STATSCORE_CONT1_W`/`STATSCORE_CONT2_W` (:624-630). ⇒ **Do NOT chase cont-hist as a
  lever.** The whole legacy tier path (`lmr_hist_tier`, `HISTORY_LMR_CAP`, `HISTORY_LMR_MORE_CAP`,
  `ENABLE_CONT_HIST`, `CONT_HIST_LMR_THRESH`) is vestigial under the default config — it READS as live and
  misleads code review. Candidate for deletion (per CLAUDE.md's delete-dead-code rule).
  Note `ENABLE_HISTORY_LMR` gates the WHOLE helper (both paths) — that is why it still changes results.

**Perspective:** LMR alone is worth 6.58× in nodes; the ENTIRE eval-speed lane measured today offered
single-digit percentages. EBF/pruning — not throughput — is where our leverage is.

## FIX LADDER (ranked by fable; each gated + byte-id)
1. **Per-capture futility** replacing the node-level delta: `futilityBase = static_eval + DELTA_MARGIN`, then in
   the move loop skip only if `futilityBase + value_of(captured) <= alpha` (`get_value_at`, `cpp_bitboard.h:1478`
   already exists); keep a node-level prune ONLY at ≈queen value (~10000mp).
   **Confirm:** re-run `QSTANDPAT_EVAL_MODE=2` WAC — the −13 should shrink substantially.
2. **Move the qdepth cap below the in-check test** (or exempt in-check) + add a `g_qcap_hits` counter at SE:5277.
3. **THEN retest dropping capgains at qsearch stand-pat.** If #1 removes the hole capgains was plugging, we may
   get BOTH a sounder search AND a cheaper eval — the "have both" outcome.
4. Quiet-check depth limiting (`ENABLE_QCHECK_DEPTH0` / `ENABLE_QCHECK_MASK`, both already built, default off).
   ✅ **DONE 2026-07-28 and it was the big one — see `SESSION-HANDOFF-2026-07-28.md`.** These two are a
   MATCHED PAIR and neither worked alone. The simulate path's check test was broken (`update_state` takes
   `turn` BY VALUE ⇒ `is_check` asked whether the MOVER was in check ⇒ **qsearch had NEVER searched a single
   quiet check**, counter-verified at 0/300 WAC). ⇒ the "full board-copy + `is_check` per quiet move at every
   q-ply" cost noted in this document was **being paid for nothing**. Fixed pair (`MASK=1 DEPTH0=1`):
   **253 WAC / 49.3M nodes, and +42.6 ±25.6 Elo over 977 games at equal time.**
   ⚠️ **Item 3 below is now ANSWERED and NEGATIVE:** capgains is NOT redundant once qsearch sees checks —
   capgains-off costs −8 solves without quiet checks and −7 with them. No "have both" outcome; capgains stays.
   ⚠️ Discovered checks (`ENABLE_QCHECK_FULL`, detector verified correct) are ALSO negative — one discovery
   spawns many near-duplicate checks and floods qsearch.
5. Minor: per-move SEE, qsearch TT move / `ENABLE_QSEE_RESORT`. Check `qfmc` (SE:2013) first.
6. QCache depth/truncation tag (moot if #2 lands).

## PRIOR OPEN QUESTION (now answered)
Capgains simulates CAPTURE SEQUENCES — work qsearch also does. Some overlap is legitimate (qsearch searches
captures/checks; it structurally CANNOT see quiet moves that create threats, which is what `latent_threat` and
part of capgains encode). But the capture-sequence overlap is real and suspicious: a crude static approximation
should not be worth 15 WAC solves if qsearch were fully resolving exchanges.
**Audit targets:** how often qsearch hits `MAX_QDEPTH=10` (truncation ⇒ unresolved tactics); whether
`QDELTA`/`DELTA_MARGIN=1500` prune too aggressively; qsearch move-ordering/SEE quality; and how our qsearch
compares to SF11/SF15/Ethereal. If qsearch truncates often that is a STRUCTURAL defect — fixing it would both
gain strength AND reduce our dependence on the expensive capgains term.

## Strategic implication
4× NPS ≈ **+1.1 plies** at EBF 3.68, whereas EBF 3.68→2.5 ≈ **+5 plies**. Combined with "no hotspot exists",
**throughput is NOT our lever — EBF/pruning is.** Closing SF's 5× gap would require redesigning what the eval
computes per node, which trades away the accuracy de-king just proved valuable.
