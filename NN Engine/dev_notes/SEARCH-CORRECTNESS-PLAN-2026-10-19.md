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
