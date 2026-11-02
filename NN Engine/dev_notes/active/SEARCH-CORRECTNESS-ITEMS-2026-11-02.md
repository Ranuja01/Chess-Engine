# Search correctness items still open — plan for owner approval (2026-11-02)

Scope: the correctness and cheap-gating items that the 10-22 search audits (`research-2026-10-22/`) added to the transition, plus the
already-built fix knobs whose games were void (the 10-19 … 10-22 `Knob_` matches). Re-checked on the gen 6.4 source today: the speed
batches since 10-22 were result-identical, so every behaviour below is still present. The `isNearGameEnd` read is no longer a defect:
the live generator initialises it (result-identical), and the copy in `generateLegalMovesReordered1` is never called.

**Rules for every item** (SEARCH-CORRECTNESS-PLAN §D, audit §5): one knob, default 0 = byte-identical (gen 6.4 `bench` 36,991,901);
a counter on the taken branch read before any games; an SF18 d16 move test at the EXACT trigger positions (not the first divergence —
the 10-21 lesson: 76% of first divergences were butterfly); games at equal time. Correctness items ship on **not harmful**: games
≥ −1σ and the move test not negative. Disjoint narrow items may share one harm match (owner 11-01 bundling rule) because each is read
on its own counter and move test.

## 1. Built, never judged validly (knob only, no code)

| item | knob | defect | record |
|---|---|---|---|
| B1 | `FIX_HALFMOVE=1` | the in-search halfmove clock resets only on captures (the pawn test runs after the pawn moved), so pawn pushes in long endgames can score as 50-move draws | fired in 4/30 eg_critical, 1/30 kp, 0/30 UHO games (#89 counts); games void |
| B2 bit 1 | `FIX_REP_KEY=1` | repetition counts keyed by placement + side to move only: positions differing in castling rights or ep count as repeats | unit test passes (`_test_rep_key.sh`); games void. Bit 2 (SF's pre-root 3-fold) −12 ± 42 / 200 g, unresolved — judged only where it fires |
| A4 | `FIX_MAX_NULL=1` | the maximizer's null move searches (α, α+1) but cuts on ≥ β, so at PV max nodes it almost never cuts (the minimizer is correct) | general change (wasted nodes, sound); games void |
| G5 | `NULL_NPM_GUARD=2` | SF's rule ADDED to ours: no null move when the mover has no non-pawn material. #90 parked MODE 1, which also drops our count rule; mode 2 never ran | mode 1 move test −0.014 ± 0.012 (not this form) |

## 2. Need code (small; each its own knob)

| item | defect | references | counter (what it counts) |
|---|---|---|---|
| MDP | no mate-distance pruning | 8/8 | nodes where the mate bounds alone would cut (needs the plies-from-root frame) |
| Mate before 50 | a mate delivered on the 100th half-move scores as a draw | 6/8 | 50-move draw returns where the side to move is mated (pairs with B1) |
| Mate gate | move-loop pruning (LMP, futility, SEE, history) is not gated on `best > −MATE_BOUND`; RFP and null already are | 9/9 | prunes taken while the node's best is a mated score |
| Killer splice | on a move-gen cache hit, killers / counter are placed by `iter_swap`, not inserted: the best quiet is demoted, and when `firstNonCapture` is 0 the killers land AHEAD of every capture | consensus: insert at the first quiet | cache hits where a killer / counter is swapped ahead of a capture, or demotes the previous first quiet |
| TT PV cutoffs | TT cutoffs are taken at PV nodes | 7/8 forbid them | TT cutoffs taken with β − α > 1 |
| TT rule 50 | TT cutoffs are taken when the halfmove clock is near 100 | SF and others | TT cutoffs with clock ≥ 90 |
| TT mates | mates and draws are never stored | consensus stores both, mates ply-adjusted | only after MDP ships |

## 3. Not here (SEARCH V2)

- A1 / A2 / A3 and the VERIFY downgrade: the reductions audit found the bound, the reduction scale and VERIFY are ONE mechanism (VERIFY
  is the hidden second look that makes the reduction magnitude work). They are replaced together as LMR v2, not fixed one at a time.
- The half-applied history gravity (presearch and node-exit writes unbounded, bonus ~0.6% of the maximum): it changes every history
  magnitude, so it belongs with the history redesign and its retune.

## 4. Order and predictions (registered before any run)

**Step 0 — counters (no behaviour change).** A counter build on gen 6.4 (bench must stay 36,991,901), self-play at fixed nodes from
eg_critical, kp_dense and UHO starts (`_fix_firecount.py` pattern), logging the trigger FENs for the move test. Predictions: MDP,
mate gate, mate-before-50, rule-50 guard each fire in < 5% of UHO games; killer splice fires on ≥ 2% of cache hits (audit); TT PV
cutoffs are a large share of PV-node visits (no prediction on the sign of their effect).

**Step 0 result (queue #136 + an adjudicated re-run, 11-02).** The counter build (`-DFIXCNT_BUILD`, compiled out otherwise) and the
plain build of the same source both bench 36,991,901. Self-play at 30k nodes/move, 30 starts per set (seed 7), one engine process
per game (caches cold at each move). Headline numbers resign at |score| ≥ 1000 cp for 6 plies, as the matches do
(`_fix_firecount.py ADJ=1000`); #136 itself played every game out, in brackets. Trigger FENs were NOT logged; each fix build
logs its own for the move test.

| item | UHO | kp_dense | eg_critical | prediction |
|---|---|---|---|---|
| MDP: nodes where the mate bounds alone cut | 2.6% of nodes, 30/30 games (4.0%) | 7.2% (12.9%) | 2.3% (3.3%) | < 5% of UHO games ✗ |
| Mate gate: prunes with the best still mated or unset | 0.22% of prunes, 30/30 | 0.35%, 30/30 | 0.07%, 27/30 | < 5% of UHO games ✗ |
| Mate before 50: 50-move draws scored on a checkmate | 0 | 0 | 20, in 2/30 games | < 5% of UHO games ✓ |
| Rule-50 guard: TT cutoffs at clock ≥ 90 | 0 | 0 | 2.4% of TT cutoffs, 7/30 | < 5% of UHO games ✓ |
| Killer splice: swaps that displace a capture | 2.7% of cache hits | 2.8% | 0.8% | ≥ 2% of hits ✓ (UHO, kp) |
| TT cutoffs with an open window (β − α > 1) | 22% of TT cutoffs | 26% | 20% | a visible share ✓ |

What it changes (proposed; owner to approve with the rest):
- **MDP and the mate gate are general, not narrow.** Both fire in every game. MDP fires where a mate found in one line bounds the
  siblings' subtrees (middlegame tactics too: adjudication removed only ~40% of its UHO count). It is a node saver of up to ~2.6%
  on UHO and ~7% in pawn endings, Elo-neutral alone (memory node-savings-below-35-percent), so it is judged as a harm match plus the
  bench (nodes down, WAC solved held). The mate gate turns ~0.2% of prunes back into searched moves. The two act in the same
  mated subtrees, so they are not bundled: MDP first, the mate gate on top of it.
- **The narrow set is mate-before-50 + the rule-50 guard**: both fire only in long endgames (eg_critical), both are the 50-move
  rule's handling, as B1 is. Proposed: after B1 is judged, the three as one 50-move item on endgame starts.
- **TT PV cutoffs move to SEARCH V2.** In this move loop only reducible quiet moves are scouted with a null window; first moves,
  captures, checks and protected killers inherit the parent's full window (`get_score_for_*`, the `i == 0` / `do_lmr` branches).
  So open-window nodes are far more common than PV nodes in a PVS engine, and forbidding their TT cutoffs would drop a fifth of
  all TT cutoffs, which is not what the references do. It belongs with the PVS move-loop design.
- The killer splice keeps its own match (unchanged).

**Step 1 — the four built knobs**, each 1,000 g vs gen 6.4 at 10+0.1:
- B1 and B2 bit 1 together (narrow, disjoint, both draw bookkeeping). Registered: −10 … +10; ships if ≥ −1σ and the move test at their
  triggers is not negative.
- A4 alone (general, changes node counts). Registered: −5 … +15.
- `NULL_NPM_GUARD=2` alone, on the transition starts (it fires in pawn-only-mover endings). Registered: −10 … +10.

**Step 2 — the code items**, each with its counter and move test, then one harm match for the narrow set (MDP + mate-before-50 + mate
gate + rule-50 guard; registered −10 … +10). The killer splice and TT PV cutoffs change ordering / the tree broadly, so each gets its
own match (registered −10 … +15 and −10 … +10). TT mates only after MDP ships.

**Cost.** Step 0 ≈ 1 h of box time, no timed games. Step 1 ≈ 8 h (three matches). Step 2 ≈ a day of code and counters plus three
matches. Expected Elo: small — these are correctness items, valued for removing wrong behaviour without harm; the search arc's
larger forecast (+50 … +150) rests on the LMR / pruning / ordering redesign that comes after them.
