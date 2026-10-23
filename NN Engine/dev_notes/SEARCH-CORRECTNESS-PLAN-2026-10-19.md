# SEARCH CORRECTNESS — plan for owner review (2026-10-19; search transition step 3)

The search is NOT negamax: `maximizer` and `minimizer` (search_engine.cpp) are two hand-written mirror halves, and the root is
always the maximizer (minimizer nodes = odd plies, maximizer = even plies ≥ 2). A read-only audit diffed the two halves
(`get_score_for_minimizer` 4524-4936 vs `get_score_for_maximizer` 4938-5368, the node bodies, qSearch, root). Every item below
marked VERIFIED was re-read in the code after the audit (line numbers as of 10-19).

## A. Real asymmetries (VERIFIED)
| # | where | what | likely effect |
|---|---|---|---|
| A1 | 4883 & 5300 (identical text in both helpers) | LMR re-search / VERIFY rule `(alpha < score < beta) ‖ (VERIFY && do_lmr && score ≤ alpha && alpha − score < margin)` copied verbatim into BOTH halves. At non-PV nodes (β = α + 1) the first clause can never fire ⇒ no standard "reduced move beat the bound → full-depth re-search". MAX: a reduced fail-HIGH is accepted as a cutoff unverified. MIN: `score ≤ alpha` is the minimizer's CUTOFF side — it verifies its OWN optimistic results. The root does it correctly (≈4321). | biased toward the root side's view; misses refutations hidden by reductions — fits "our search misses the opponent's resource" (owner game 7's pending break, 10-19 loss 25.Rxe5) |
| A2 | 4816, 4859 | the MINIMIZER's PVS / LMR scout uses `(alpha, alpha+1)`; a min node should scout at `(beta−1, beta)` (MAX at 5233 / 5276 correct) | at PV min nodes nearly every move gets a full-window re-search ⇒ PVS / LMR collapse there — pure efficiency loss |
| A3 | 4585 vs 5002 | full-window branch at remaining depth 1 on the MINIMIZER only (`i == 0 ‖ rd == 1`) ⇒ at rd 1 the minimizer skips LMP / history / SEE pruning / LMR / VERIFY while the maximizer prunes its frontier quiets to ~2 | opponent's frontier searched fully, ours pruned — another root-side bias, plus nodes |
| A4 | 6565 / 6580 | MAXIMIZER null move searches `(alpha, alpha+1)` but cuts on `≥ beta` (MIN 5881 correct) | at PV max nodes the null search tests the wrong bound and almost never cuts — wasted nodes, sound |

## B. Bookkeeping bugs (VERIFIED)
| # | where | what | effect |
|---|---|---|---|
| B1 | make_move 3291 | halfmove clock reset tests the pawn bit AFTER update_state moved the pawn ⇒ only captures reset it in search | false 50-move draws when root clock + capture-free plies ≥ 100: in long endgames pawn pushes (the way to progress) score as draws. Root clock is correct (python-chess / UCI). Bench byte-identical (clock ≈ 0) |
| B2 | repetition key (make_move 3306, cache_management.h:370) | `position_count` keyed by placement + side to move — no castling / ep; and pre-root game positions count once ⇒ 1 prior + 1 in search = draw (SF needs 3-fold for pre-root positions) | false repetition draws (Ke1-f1-e1 losing rights, ep, shuffles that recur from the game) |
## C. Intentional / harmless (checked)
NULLMOVE_CURDEPTH MINI 3 / MAXI 4 = ply-parity artifact (each side skips its first interior ply; MAXI=3 byte-identical, h:3133) ·
futility `cur_depth > 1` on the min side only (always true at max nodes; guards ply 1) · RFP / ProbCut / singular / improving /
qSearch stand-pat & delta / mate & stalemate returns: correct mirrors · mates never TT-stored ⇒ no mate-ply bug (efficiency cost
only) · 4 live debug `std::cout` blocks on hard-coded positions (5485, 5609, 6080, 6823) — harmless (the binary redirects cout).

## D. How to fix and measure (each fix = its own knob, default = TODAY'S behaviour ⇒ byte-identical until switched)
☠️ Record: search changes are ANTAGONISTIC, not additive (2×2 first); "productive bugs" exist — tuning (VERIFY_MARGIN, LMP, LMR,
RFP margins) absorbed today's behaviour, so a correct fix can LOSE Elo until the margins re-sweep (memory
search-value-bugs-and-the-productive-bug-pattern, search-changes-are-antagonistic-not-additive). Hence: knobs, then games.
1. **Cheap & safe first — B1, B2, A4** (bookkeeping + a wrong bound): bench should be (near-)identical; unit tests: B1 a FEN with
   clock ≈ 90 where a pawn push wins; B2 the Ke1-f1-e1 rights case. Games: one standard self-play match each (node-limited) —
   expect ~0; ship on "not harmful".
2. **The big three — A1, A2, A3** (coupled: A1 and A2 both live in the re-search logic): knobs `SEARCH_FIX_RESEARCH`,
   `SEARCH_FIX_MIN_SCOUT`, `SEARCH_FIX_MIN_RD1`. Measure: (i) bench nodes + WAC solves per knob (A2 / A3 should cut nodes; A1
   likely raises them and WAC solves); (ii) the triage DEPTH / PRUNING cases (`_case_triage.py`) — do the missed resources appear?
   (iii) GAMES at EQUAL TIME through the binary + fastchess (fixed nodes would hide A2 / A3's speed effect): each fix alone vs
   gen 4, then a 2×2 of A1 × A2, then all three; then a VERIFY_MARGIN / LMP re-check for the winning combination.
3. **Pre-registered predictions:** A2 −5…−20% nodes at fixed depth, best moves mostly unchanged, timed Elo +0…+15; A3 −5…−15%
   nodes, Elo −10…+15 (may have been tuned in); A1 +5…+25% nodes, WAC +1…+4, Elo −10…+20 before margin retune; B1 / B2 / A4
   ≈ 0 ± noise.
4. Then the remaining correctness items: the null-move fix re-read (−1.4 ± 17.9 node-limited), why v2's tree blows up on some
   positions (shallowest d8 vs v1 d10 at ~1 s), the resign return, the redundant Cython_Chess init.

## Owner decisions needed
1. Order OK (cheap three first, then A1-A3 with a 2×2)?
2. Equal-TIME games through the binary as the main instrument for A2 / A3 (speed effects), node-limited only as a secondary?
3. Keep the old behaviour behind knobs (reproducible) — yes by default, unless you prefer plain replacement once a fix ships.

## E. IMPLEMENTED 2026-10-20 00:40 (knobs, all default 0 ⇒ byte-identical: Cython 258 / 47,689,413 · 250 / 35,310,778 ✓)
`FIX_HALFMOVE` (B1) · `FIX_MAX_NULL` (A4) · `FIX_MIN_RD1` (A3) · `FIX_MIN_SCOUT` (A2) · `FIX_RESEARCH` (A1) · `FIX_REP_KEY` (B2, knob
declared, NOT implemented yet — 10 count sites with different state variables; needs its own unit test). Binary option
`Knob_<NAME>` sets any env knob; fix binary `build_uci/chess_engine_uci_fix` (`OUT=` in build_uci.sh).
WAC d10 bench, each alone (fingerprint + tactical veto only — memory wac-at-d10-does-not-discriminate-strength):

    none        258  47.69M        A3 FIX_MIN_RD1     243 (−15)  −20.1%   ☠️ tactical veto — likely a PRODUCTIVE asymmetry:
    B1          258  identical                                             try the OTHER direction (full window at rd 1 BOTH sides)
    A4          254 (−4)  +0.2%    A2 FIX_MIN_SCOUT   250 (−8)   −12.6%   (pred −5…−20 ✓)
    ALL five    254 (−4)  +1.6%    A1 FIX_RESEARCH    259 (+1)   +33.6%   (pred +5…+25 nodes, WAC +1…+4)

**B2 `FIX_REP_KEY` IMPLEMENTED 2026-10-20 (code only; not yet built — queue #88 held the box).** Bit flags: `1` = every count
site keys with `repetition_key()` (cache_management.h: placement ⊕ castling ⊕ ep, the move-gen cache mixer) — root history
(initialize_engine / set_current_state / UCI `pos::rep_key`), make_move (child's rights), unmake_move (uncounts with the child's
rights before the pop), `is_repetition` (now takes the node's state; 7 call sites); `2` = SF's pre-root rule: at exactly
REPETITION_THRESHOLD occurrences a draw only if none is from the game before the root (`g_pre_root_counts`, snapshotted in
get_engine_move); `3` = both. Unit test + registered predictions: `selfplay/_test_rep_key.sh` (false draw Q vs R with lost
rights; true-repeat control; bench at 0 / 3).

## 🧭 TRANSITION SCOPE (owner-agreed 2026-10-22, after the five search audits in `research-2026-10-22/`)
Filter: an item belongs to the TRANSITION only if it is a correctness bug, a result-identical speed gain, the time manager, or
eval-coupled. Everything else is SEARCH V2. Each new correctness item starts with a COUNTER (fire rate), then is judged where it fires.
- DONE: infra (binary, fastchess, baselines; 10-22 Knobs fix) · correctness gen 5 (B3, C1) + the empty-move crash fix · pawn-ending
  rules re-read (#90).
- LEFT: speed programme (profile on v2 → move-gen cache size, KS-A, unscoped late-eg eval, pawn hash, pin-aware movegen) · step 6 table
  footprint · features (corrhist — consensus form per CORRHIST-REFERENCE-AUDIT, threats eval+search, TIME MANAGER, POT-as-search) ·
  margins re-sweep LAST → bench checkpoint #3. Odds and ends: NULL_NPM_GUARD mode 2, v2 tree blow-ups, owner-game triage cases.
- ADDED BY THE AUDITS: mate-loss gate on move-loop pruning · VERIFY downgrade defect · endgame LMR bypass (`reduced_search_depth`
  phase ≥ 117 → `DEPTH_REDUCTION[depth_limit]`; pawn-ending suspect) · killer-splice swap + half-applied gravity · mate-distance pruning,
  mate-beats-50-move, TT never stores mates/draws · qsearch generates all moves keeps captures (speed, result-identical) · WASTED
  UNFINISHED ITERATIONS (est. 30-55% of move time, unmeasured) + depth-1 ID start (time manager).
- SEARCH V2 (not now): null reform, improving, LMR v2 with A1/A2, IIR, singular/multicut, ProbCut, new history features, staged picker
  + node-entry TT (then delete the move-gen cache), TT buckets, negamax port.
- ORDERING PRINCIPLE (discussed with the owner 2026-10-22): (1) result-identical changes ship immediately — they cannot disturb the
  tuned optimum; (2) FOUNDATION changes that move the regime everything is tuned to (time use, TT / table sizes, depth reached,
  correctness fixes) go EARLY, each with an equal-time harm check against the current optimum — deferring them would mean tuning
  twice; one that loses only because surrounding constants were fitted without it is bundled with the retune (the A2 pattern), not
  shipped alone; (3) additive features (corrhist, threats-in-search) in the middle, on the settled foundation; (4) the margins /
  constants retune LAST. The current optimum is the comparison point, not something to protect.
- STRUCTURE (owner 2026-10-22): the transition works in the CURRENT search (`search_engine.cpp`), preferring SHARED components
  (movegen, eval, caches, time manager, corrhist tables) — they carry over; fixes inside the max/min halves only for real bugs (they
  carry over only as knowledge). SEARCH V2 = likely a NEW file (`search_v2.cpp`, runtime-selected, sharing movegen/eval/caches) — a
  negamax parity port then consensus-form features; the old search becomes the frozen control. Details (parity vs redesign, root
  architecture) are discussed when the arc starts. Result-identical speed work applies to v1 too (fp_v1 = a free second identity
  check); tree-changing items go in the v2 `shipped` block so v1 stays frozen.
- Re-runs after the void-match discovery: gen 5 vs gen 4 harm check (#92) and, optionally, the PE_PASSER_EXT venue. Not A2/A1/B1/A4/B2.

**SPEED PROGRAMME STEP 0 — FIRST NODE-LEVEL PROFILE ON v2 (gen 5, 2026-10-22; `E:\chess_data\profile_v2_g5*.log`).** Midgame 40 FENs
d12, shares of SEARCH_ROOT cycles (rdtsc scopes inflate what they wrap ⇒ upper bounds):
QGEN (qsearch move-list build) **32.4%** (6.87M calls, ~17.9k cyc) · MOVEGEN 35.1% (9.80M; nested inside the two builders; of it ISSAFE
11.9%, PSEUDO 5.8%, SCORE 4.9%, SORT 4.9%) · MG_CACHE (main-search list build) 10.8% (4.30M) · V2_EVAL 21.1% (KS 7.2%) · MAKEUNMAKE 3.9%
· TT_PROBE 1.3% · REPETITION 0.5% · unscoped ~30%. WAC d10: same shape (MOVEGEN 32.9%, eval 17.9%).
⇒ (1) qsearch generates + legality-checks + scores + sorts EVERY legal move to keep the noisy ones (buildNoisyMoveList) — a third of
all search time: a captures/promotions-only generator, result-identical (same list, same order), is the first speed change (est.
~1.25-1.3× NPS). (2) inferred: 11.2M list builds vs 9.8M generations ⇒ the move-gen cache serves only ~12% of builds, for ~8%
probe/copy overhead + ~384 MB (step 6). (3) eval is ~21% now (memory said 11.5% — the shipped gens grew it). (4) repetition: no target.

**⚡ SPEED TALLY (result-identical changes, batched; games once the total nears ~+30%):**
| # | change | identity proof | speed (v2) |
|---|---|---|---|
| S1 | qsearch skips quiet moves at defaults (the default quiet-check test asked about the mover's own king — never true) | quiet_checks_added 0 over both benches; bench v2 47,689,586 / v1 35,310,778 identical | ≈ +4.6% (2 alternating UCI benches) |
| S2 | `QGEN_NOISY_ONLY` (default 1): on a cache miss qsearch generates only captures + promotion pushes + en passant (`generateLegalMovesReordered(..., noisy_only)`) | `QGEN_VERIFY`: 0 mismatches in 5,398,832 (v2) + 2,514,985 (v1) lists; benches identical both arms | +3.7% / +18.6% by pair (box not idle) ⇒ ≈ +10%, imprecise |
| S3 | `HugeVector` (HugePageAllocator): TT, TT-move table, q-cache, eval cache, move-gen cache slots 2 MB-aligned + MADV_HUGEPAGE (Linux; plain allocation elsewhere) | benches identical both arms; AnonHugePages 1.2 GB in use | +5.6% / +4.5% (2 consistent pairs) |
| S4 | `REP_STACK` (default 1): search-path positions on a fixed key stack (push/pop) instead of position_count insert/erase; map = pre-root history; count = map + path, scan stops at the last capture/pawn move | benches identical (v2, v1, FIX_REP_KEY=3 map vs stack); self-play 120 games stack on/off, 0 divergences | +4.2% / +2.7% |
**Combined S1-S3 vs gen 5 this morning (3 alternating pinned pairs): 500k → 610k bench nps ≈ +22% (bench time 95.3 → 78.2 s)**; with
S4 ≈ +26% on the bench. The bench spends ~19% clearing tables per position (`clearSearchTables`), which real play does not ⇒ the
real-play gain is likely ≈ +30%. ⏭️ The batch's equal-time games: new binary vs this morning's (`build_uci/chess_engine_uci_cnt` =
gen 5 + inert counters), st=1, ~400 g — Elo here is pure speed (identical search).
**perf (sampling, real play, 10-22):** with the clear-on-side-change artifact removed (the harness plays both sides in one process,
so `clearAllCaches` + `clearSearchTables` ran every move — fastchess engines play one side), real-play shares: TT access
`accessSearchEvalCache` ~17% · eval ~27% (`ks_units` ~10%) · pseudo-legal gen ~10% · repetition map ~5.6% (now S4) · qsearch body
~6.5%. ⚠️ This perf build cannot resolve symbols itself — symbolise with addr2line (scratchpad `perf_sym.py`). The rdtsc ISSAFE
"12%" was mostly scope overhead (141M scope entries): is_safe is ALREADY pin-aware (python-chess form), so "pin-aware movegen" is
not a lever; the 16-pass structure is (~5-8%). Next candidates: KS cost in eval, 16-pass collapse, TT density (fold g_ttMoveTable). Logged defect: `isNearGameEnd` read uninitialised in `generateLegalMovesReordered`'s
normal-endgame branch (quiet-pass order) — a tree change to fix, so it waits.

**QUEUE #92 — THE REAL GEN 5 HARM CHECK (2026-10-22, `_queue_gen5_check.sh`):** gen 5 vs gen 4 (`option.Knobs=V2_SHIP_GEN=4`), same
binary, st=1, 400 g UHO, seed 921. Guards: gen 5 bench 258 / 47,689,586, gen 4 via Knobs 258 / 47,689,413, knob-channel pre-flight 0
warnings + echo. **Result +2.6 ± 29.4 (50.4%), 0 unresponsive, 0 option warnings** — registered ≈ 0 (harm < −20): HELD. Gen 5 stands
on correctness with no harm shown. (Partial reads swung +37 @162 g → −8 @306 g → +2.6 @400: the noise the void matches measured.)

## ☠️☠️ 2026-10-22 — EVERY FASTCHESS `Knob_` MATCH WAS GEN-vs-GEN (VOID)
fastchess drops options the engine does not ADVERTISE in its `uci` list ("Warning; X doesn't have option Knob_…", printed every
game, 5,897 times). The binary never advertised `Knob_<NAME>` ⇒ the knob was NEVER SENT; both sides ran the identical engine.
Proven with a stdin-logging wrapper: fastchess sent `setoption name OwnBook` only. (My 10-21 "knob verified LIVE" test typed
`setoption` by hand — it proved the engine, not the channel.)
**VOID (identical engines):** fix_A2 +49 ± 37 · fix_A1 +31 ± 39 · fix_A2_confirm −10 ± 32 · fix_A2A1_vs_A2 −26 ± 42 · fix_bundle_c1
−3 ± 31 · fix_repb2 −12 ± 42 · pxext_eg +12 ± 19 (280 g, stopped on a stall) · pxext_uho +28 ± 41.
⇒ A2 / A1 / the bundle / FIX_REP_KEY bit 2 / PE_PASSER_EXT have NO game evidence at all. The "A2 first screen inflated, fresh seeds
pulled it back" story was wrong — both were nulls. Gen 5 (B3 + C1) was shipped on correctness; its "no harm shown (−3 ± 31)" line is
void ⇒ gen 5 still needs a real harm check.
**VALID:** v2 vs v1 (`option.EvalArm=0`, advertised), all SF matches, the SF11 ladder, the SF18 autopsy, and every `_fix_firecount.py`
fire count / move test (env knobs, and the divergence proves they were live).
★ **Free instrument:** the eight void matches = ~2,080 games of IDENTICAL engines: weighted mean ≈ +5.8 Elo (memory: harness null
+4.6), and one 200-game null read **+49 ± 37**. ⇒ a 200-game fastchess screen cannot separate a real ±40 from nothing.
**FIX (2026-10-22):** declared option `Knobs` (comma-separated NAME=VALUE, e.g. `option.Knobs=FIX_MIN_SCOUT=1,FIX_RESEARCH=1`); the
engine echoes `info string knobs …` at init so every match log records what each side ran. ☠️ Every knob queue must check (a) zero
"doesn't have option" warnings and (b) the echo, before reading a result.

**QUEUE #90 — SF11 PAWN-ENDING RULES RE-READ ON GEN 5 (2026-10-21, `_queue_pe_rules.sh`):** each rule alone vs gen 5, self-play
30k nodes, 150 eg_critical + 150 kp_dense starts, SF18 d16 move test at the first divergence (logs `E:\chess_data\pe_rules_*.log`):

    rule              diverge eg / kp   median ply   better / worse   mean Δ (all)        kp mean Δ
    PE_PASSER_EXT      131 / 97            4           21 / 14       +0.012 ± 0.011      +0.002
    NULL_NPM_GUARD     137 / 68            1-10        24 / 31       −0.014 ± 0.012      −0.015
    PE_PRUNE_GATE      139 / 105           3-6         15 / 26       −0.016 ± 0.010      −0.029 ± 0.014

All fire in most endgame games within a few plies (so this read is close to each rule's own effect, little butterfly). 0 crashes in
~1,800 games. Registered predictions: PASSER_EXT "kp better > worse" — direction held only on eg (kp 4 / 3, ≈ 0); "NULL_NPM / PRUNE_GATE
rarely diverge" — WRONG (they act deep in the tree wherever exchanges leave pawn-only nodes). Verdicts: PE_PRUNE_GATE CLOSED for now
(−2σ on kp); NULL_NPM_GUARD PARKED (leans −; untested guess: our null-move verification already covers zugzwang) — ⚠️ CORRECTION
(pruning audit 10-22): queue #90 ran MODE 1, which REPLACES the legacy count rule (newly allowing null in some low-material endings);
the pure SF ADDITION is MODE 2 (legacy OR npm==0) and was never run ⇒ the park covers mode 1 only; PE_PASSER_EXT the
candidate ⇒ equal-time venue match from eg_critical starts. ⚠️ fixed-node move test ≈ equal work, not equal time.

**QUEUE #89 + FIRE COUNTS (2026-10-21).** Bundle B1 + A4 + B3 + B2 bit 1 + C1 (`FIX_QS_CHECK`, new) vs gen 4, st=1, 400 g: **−3 ± 31**.
FIX_REP_KEY=3 vs that bundle, 200 g: −12 ± 42 (bit 2 stays OFF). ⚠️ A4 is a GENERAL change, wrongly bundled with the narrow ones.
Fire counts (`_fix_firecount.py`, counter binary `build_uci/chess_engine_uci_cnt`, FIX_COUNT_LOG; self-play at 30k nodes, fixes
B1 + B2b1 + B3 + C1 OFF vs ON, 30 starts per set):

    set              diverge  median ply  result changed   B3 drawexit games   C1 qsmate games   B1 fifty games
    eg_critical      10/30        11           4                 16                  0                 4
    kp_dense          3/30        22           2                  4                  8                 1
    UHO (general)    11/30        43           3                 11                  9                 0

⇒ NOT narrow: they change ~1/3 of general games (late) and ~10% of results everywhere, so the 400-g general match resolves only
~40 changed results. Next instrument: the MOVE TEST at the divergence points (SF18 d16 judges the off vs on move;
`_fix_firecount.py SF=… OUT=E:\chess_data\fix_divergences.csv`), registered: fixed move better more often than worse, mean Δ > 0.

**CRASH BUG (pre-existing, all knobs 0):** the search makes a null move (0→0) and `update_state` throws ("push() expects move to
be pseudo-legal") → SIGABRT. Repro: start `8/3p3P/5pPK/2p3pp/1P1P1p1P/3P1k2/pP2pP2/8 w - - 0 1`, moves `h7h8q a2a1q g6g7 a1b2`,
self-play 30k nodes/move (`_fix_firecount.py`). Two queens, promotion race — suspect a stale/empty Move from a killer / TT /
root second_moves slot. Not seen in any fastchess match since restart=on (no illegal / forfeit terminations).
**ROOT CAUSE + FIX (2026-10-21, gdb on a -O1 -g build):** the throw is get_engine_move's FINAL make_move of `move`, still the
empty `Move(0,0,0)`. A node limit (qSearch sets time_up at NODE_LIMIT) or the clock landing INSIDE the first depth-3 pass makes
alpha_beta return before any root move list exists (the 10-19 guard covered a UCI stop there, not this). These multi-queen races
cost ~50k nodes for depth 3 (30k crashes, 100k searches to d5). Fix: if `move` is empty after the search, play the first ordered
legal move. Unconditional (changes only would-be crashes); bench 258 / 47,689,413 identical; both repros return a legal move.
Real-play risk it removes: a very low clock in a multi-queen endgame (60+0.6 near the end).
⏭️ FOR THE TIME-MANAGER UPGRADE (owner 10-21: "we'd need to provide a move in those low time cases"): iterative deepening
starts at DEPTH 3 (`depth_limit = 3` in get_engine_move); SF and most engines start at 1, so a SEARCHED move exists within
microseconds. Starting at 1-2 would make low-clock moves come from a real (shallow) search instead of the unsearched ordering
fallback. Changes every node count (fingerprint) and interacts with time allocation ⇒ judge with the time manager, equal time. Second repro: start
`8/p3P3/PP3PP1/2p5/3P2K1/ppp2PP1/2pp2p1/6k1 b - - 0 1` + `d2d1q b6a7 g1f2 a7a8q g2g1q g4f5 g1g3 g6g7 g3h3 f5g6 a3a2`.
**MOVE TEST (150 starts per set, 112 first-divergence points, SF18 d16):** fixed move better 13 / worse 14 / ~equal 85, mean Δ
−0.020 ± 0.014 — registered prediction FAILED; no broad benefit. ⚠️ first divergence ≠ where a fix acted (later moves diverge
from a changed TT/history — butterfly; fits 76% ~equal). Exact-trigger judging (log the FENs where B3 / C1 fire) not done.
Owner decision pending: ship B3 + C1 as pure correctness (demonstrated failures, no harm shown); B1, B2 bit 1, A4 stay off.
**CRASH WATCH:**
every harness must survive an engine death and report its position (the fire-count harness now does); more may exist.

**B2 BUILT + TESTED; B3 FOUND (2026-10-21, `build_uci/chess_engine_uci_b2`, `selfplay/_test_rep_key.sh`).** Test position: Q vs R
with Black's rights lost by Ke8-f8 / Qa1-b1-a1. ⚠️ My registered "Black is losing" was WRONG — ...Rh1+ skewers K and wins Qa1, Black
is WINNING (+2.6); the draw / no-draw predictions all held.
- **B3 (new) `FIX_DRAW_EXIT`:** the iterative-deepening loop `break`s on `is_draw && score == 0` (get_engine_move). At all-knobs-0
  the engine stopped at **depth 3 after 45 nodes** and played the "draw" Ke8, never seeing ...Rh1+ winning the queen — with a FALSE
  repetition (B2) AND with a genuine single repetition (control). `FIX_DRAW_EXIT=1`: Rh1+ (+259) in both; **bench byte-identical
  258 / 47,689,413** (no WAC position hits the exit). Cleanest fix of the set.
- **B2 `FIX_REP_KEY`:** bit 1 fixes the false draw, keeps the genuine one (as registered); bit 2 (SF pre-root 3-fold) also
  un-draws the genuine single repeat. WAC d10 (veto only): =1 **257** / 47,596,347 · =2 **253** / 47,525,587 · =3 **251** /
  47,905,411 ⇒ bit 2 costs WAC; judge it in games separately from bit 1.
- All knobs 0 on the new binary: 258 / 47,689,413 ✓.

**QUEUE #88 RESULTS (2026-10-20/21 night, fresh seeds 881-885, st = 1 s, concurrency 3, restart=on, 0 unresponsive):**
- **A2 confirmation: −10 ± 32 / 400 g** (48.5%). Pooled with #86 (+49 / 200) ≈ **+10 ± 25 / 600 — NOT confirmed**; the first screen
  inflated. Knob verified LIVE (QGD d9 435k → 257k nodes). ⇒ A2 stays OFF; base for the bundle = plain gen 4. The fix × reductions
  2×2 is a search-v2 question (owner 10-20).
- **A2+A1 vs A2: −26 ± 42 / 200 g** — A1 shows nothing on top of A2 either (A1 alone vs base: +31 ± 39, one seed, unconfirmed).
- **SF11 time ladder (v2 gen 4 @ 1 s):** SF11 @ 10 ms **−171 ± 49** (27.3%) · @ 30 ms **−512** (5.0%) · @ 100 ms **−727** (1.5%) ⇒
  today's break-even is BELOW 10 ms of SF11 time against our 1 s (≥ 100× time). This is the "playable with SF11" yardstick.
- **SF18 @ 5 ms loss autopsy** (`_loss_autopsy.py`, 174 lost games, SF18 d14 WDL, re-search 10 s; `E:\chess_data\autopsy\`): 172 single
  decisive blunders (median drop 0.34 expected score from a median 0.45, i.e. from a level game), 2 drifts, 0 lost from the start;
  decisive move median **10 moves into the UHO start** (quartiles 6 / 16) ⇒ early middlegame. Phase: middlegame 136 · late mg 27 ·
  endgame 10 · pawn ending 1. Kind: **positional 154** / tactical 20. Cause: **eval 108 / search 66** (10 s still plays a losing move
  vs finds an SF-acceptable one); on the 64 games whose 1 s replay reproduces the game move: eval 40 / search 24 — same split.
  `saw_it`: our own score fell ≥ 1 pawn only AFTER the move in 60; in 114 our score never showed the drop within 3 moves.
  ⚠️ 110 / 174 replays at 1 s did not reproduce the game move (timing + cold TT) ⇒ per-game cause labels are noisy; the split is stable.

**PLAN FOR 2026-10-21 (owner-agreed 10-20 evening; queue #88 = A2 confirm 400 g · A2+A1 vs A2 · SF11 ladder 10/30/100 ms ·
SF18 loss autopsy, logs `E:\chess_data\q88_fc_night3.log`, `E:\chess_data\autopsy_sf18_5ms.log`):**
1. Read #88. Build B2 to `build_uci/chess_engine_uci_fix` + rebuild Cython; `_test_rep_key.sh`; both fingerprints at knobs 0.
2. Add the missed defect as a knob: **qsearch horizon return precedes the in-check test** (`qDepth >= MAX_QDEPTH` returns before
   `currently_in_check`; preplan "VERIFIED 2026-10-09" item 2 — dropped from §A/§B by mistake). Then the cheap bundle
   B1 + A4 + B2 + this, equal time ~400 g, on base gen 4 + A2 (if A2 confirms) else gen 4.
3. Re-read `NULL_NPM_GUARD` / `PE_PRUNE_GATE` / `PE_PASSER_EXT` at EQUAL TIME on the fixed base, K+P-heavy openings. Their 10-15
   reads (#69 bias, #71 −1.4 ± 17.9 node-limited) ran with B1 (false 50-move draws in pawn endings) + B2 live ⇒ unresolved.
4. Background, no timed games: corrhist Phase-0 offline on v2's residual; STS at a second equal-node budget (100k).
5. IIR on the fixed base (v1-era +7.7 ± 17.7, never resolved). Then the other v1-era / pre-A2 closures (null eval gate + R,
   improving, singular).
6. **TABLE FOOTPRINT (owner 10-22: "reduce not just the move-order/score caches but also tables like history" — a transition-phase
   step, part of the speed programme).** Engineering, judged on RAM + NPS + equal-time "not harmful", never on byte-identity:
   (a) move-gen cache (~384 MB + a heap list per slot; the fastchess memory-growth source; move-ordering audit 10-22 §4: its one
   real asset is "own last cutoff move first", which a node-entry TT move replaces in the negamax port); (b) TT density — fold
   `g_ttMoveTable` (16.7M × Move, separate array) into `TTEntry` if padding allows (memory tt-footprint-is-a-throughput-tax-we-pay-now;
   resizing instead CHANGES RESULTS — re-baseline); (c) HISTORY tables — piece×to keys `CONT_HIST_PIECE_KEY` (114× smaller, games
   −0.6 ± 23.1 / 1,200 = Elo-neutral ⇒ adopt on engineering merit, preplan §1 #15, ordering audit §8) and `CAPTURE_HIST_VICTIM`.
   ⚠️ any re-key/resize moves the fingerprint ⇒ a ship with a new gen + register row.
   **Owner 10-22 — (c) MOVES TO SEARCH V2:** re-keying history changes what it learns (samples per slot, fill rate), and the bonus /
   malus / gravity limits, statScore divisor and ordering thresholds were fitted to the current layout — a re-key alone may raise
   nodes / EBF without being bad. So history layout + update formulas + move ordering are refit TOGETHER in search v2 (ordering
   audit stage C 2×2s), judged by equal-time games after the refit, never by nodes alone. The transition keeps only
   result-identical footprint work: huge pages (S3), the dead `g_ttMoveTable` deleted (S5, 10-22: no write or read site anywhere;
   ~64 MB + one 16.7M-entry fill per clear), TT density if `TTEntry` allows. Note: `counterMoveHeuristics[2][4096][4096]` = 134 MB
   and is most of each clear's cost — the search-v2 re-key target.
   **FOOTPRINT PLAN (owner 10-22: "reduce to that of a regular engine"; machine = Ryzen 7 5700X3D, 8C/16T, 96 MB L3, WSL 31 GB):**
   one process = **1.23 GB resident** at startup (+~1 MB per search, +134 MB once counterMoveHeuristics is written) vs ~30-60 MB for a
   regular engine. Timed matches are CORE-bound (~3 games); the payoff is wider node-limited runs, safer timed runs, and SPEED (an
   L3-sized TT on a 96 MB-L3 chip). **Phase A (result-identical, coded 10-22, unbuilt):** S5 delete dead `g_ttMoveTable`; runtime
   table sizing — masks `g_tt_mask / g_eval_mask / g_q_mask / g_mg_mask` + `initTableSizes()` (knobs TT_LOG2, EVAL_LOG2, QCACHE_LOG2,
   MGCACHE_LOG2; unset = today's sizes ⇒ byte-identical; v1 runs never set them ⇒ frozen). First test: v2 bench at EVAL_LOG2=20 —
   identical nodes would prove v2's eval cache has no side-effect coupling (v1's does: memory
   eval-global-side-effects-are-skipped-by-a-cache-hit). **Phase B (results change ⇒ games, owner go):** a UCI `Hash` option; TT
   16/32/64/128 MB with caches scaled, measured for speed AND equal-time strength (st=1 and 60+0.6); target ~100-200 MB/process.
   Logged caveat: initialize_engine counts the first root position BEFORE the knob environment is parsed ⇒ with FIX_REP_KEY bit 1 the
   Cython path's first root key is unmixed (bit 1 is off by default — no current effect).
Open owner questions (preplan §7, unanswered): threats T1 @250k · SPSA nightly budget · negamax parity vs redesign · root
architecture · Ethereal/Weiss as judges.

Queue #85 (chained after #84): equal-time screens st=1, knob on vs off in the same binary, 200 g each — A2 then A1.
A3 both directions (WAC d10): baseline asymmetric 258 / 47.69M · `FIX_MIN_RD1=1` (drop min's rd-1 full window) 243 / −20.1% ·
`FIX_MIN_RD1=2` (full window at rd 1 on BOTH sides) 254 / +28.0% ⇒ the asymmetric baseline is best at fixed depth — a PRODUCTIVE
asymmetry. Only equal-time games can say whether =1's node saving buys back its tactical loss.

## F. FIRST EQUAL-TIME SCREENS (2026-10-20, queue #86; fix knob ON vs OFF in the same binary, st = 1 s, 200 games, UHO pairs)
- **A2 `FIX_MIN_SCOUT`: +49 Elo (95% +12 … +87)**, +95 −67 =38 (57.0%), equal time (0.957 / 0.961 s), equal depth (12.2 / 12.3)
  ⇒ the gain is search CORRECTNESS at the opponent's PV nodes, not extra depth. Strongest candidate.
- **A1 `FIX_RESEARCH`: +31 Elo (95% −8 … +71)**, +87 −69 =44 (54.5%), equal time, depth 11.9 / 11.9 — positive, not yet resolved.
- Both are FIRST screens (one seed, 200 g; big early numbers shrink — memory sprt-point-estimates-inflate-at-the-bound-they-stop-on,
  gate-new-candidates-on-fresh-seeds-not-ship-seeds). NEXT: confirmation on fresh seeds (A2 ≥ 400 g), then A2 × A1 (2×2 — search
  changes are antagonistic, not additive), then the cheap three (B1, A4) as a bundle; A3 both directions only at equal time.
