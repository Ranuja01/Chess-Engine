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

*11-02 afternoon (idle box, owner away):* A4 and G5 running in queue #137 (A4 on UHO seed 1361, G5 on transition.epd seed 1362;
production PGO binary both sides through `option.Knobs`; fingerprints: A4 bench 248 / 37,936,030, G5 leaves the bench unchanged
but doubles the d14 nodes on a 7-pawn ending, so it is live). Results are measurements only; shipping either is the owner's call.
B1 + B2 bit 1 waits for a venue choice: B1 fires only in long endgames (4/30 eg_critical, 0/30 UHO), so a UHO match cannot judge it.
**Read (#137):** A4 +8.3 ± 19.4 on UHO (in its band) · G5 +15.0 ± 12.8 on the transition starts (above its band, z 2.29). Both pass
not-harmful; G5 gets a fresh-seed replicate before it is quoted as a gain. B1 + B2 bit 1 runs on eg_critical.epd in #138
(owner 11-02 approved the plan; venue = the set B1 fires in).

**Step 2 — the code items**, each with its counter and move test, then one harm match for the narrow set (MDP + mate-before-50 + mate
gate + rule-50 guard; registered −10 … +10). The killer splice and TT PV cutoffs change ordering / the tree broadly, so each gets its
own match (registered −10 … +15 and −10 … +10). TT mates only after MDP ships.

**Step 2 build checks (queue #138 part A, 11-02 evening).** Owner 11-02 approved the plan ("as long as they are as efficient
and correct as can be"). Knobs `FIX_MDP`, `FIX_MATE_GATE`, `FIX_MATE_50`, `FIX_TT_R50`, `FIX_KILLER_SPLICE` (all 0 = today),
binary `build_uci/chess_engine_uci_fix` (not PGO). With no knob: bench 249 / 36,991,901 = gen 6.4 ✓.

| knob alone | bench solved | nodes | Δ nodes | bench time | games diverged eg / kp / UHO (of 30) | results changed |
|---|---|---|---|---|---|---|
| none | 249 | 36,991,901 | — | 37.2-37.6 s warm (first run 39.4 s cold) | — | — |
| FIX_MDP | **253** | 42,158,317 | **+14.0%** | 42.0 s | 28 / 17 / 30 | 18 / 6 / 12 |
| FIX_MATE_GATE | **253** | 37,742,103 | +2.0% | 37.3 s | 7 / 7 / 27 | 4 / 2 / 11 |
| FIX_MATE_50 | 249 | identical | 0 | 37.6 s | 0 / 0 / 0 | 0 |
| FIX_TT_R50 | 249 | identical | 0 | 37.2 s | 7 / 0 / 0 (median ply 118) | 0 |
| FIX_KILLER_SPLICE | 247 | 36,874,208 | −0.3% | 37.1 s | 27 / 20 / 30 | 20 / 6 / 16 |

0 crashes in 900 games. FIX_MATE_50 unit check (`7k/8/6K1/8/8/8/8/R7 w - - 99 120`, d8): without the knob `cp 0`, Rf1 (Ra8# on
the 100th half-move read as a draw); with it `mate 1`, Ra8 ✓; at clock 0 both mate 1.
- ☠️ **MDP is not a node saver here (registration refuted on the bench):** +14% nodes at fixed depth, +4 solved. Cause (read, not
  yet measured): the TT never stores a score beyond ±9,000,000 (`addToSearchEvalCache`), and an MDP cut returns a mate-range bound,
  so its answer is never remembered; the same searches used to return ordinary fail-soft scores that WERE stored (most MDP cuts
  are scouts built on the window sentinels, e.g. (−9999998, −9999997)). MDP and "store TT mates" are therefore coupled: judge MDP
  alone in games (mdp_uho, tonight), and if it is flat or negative, re-test it together with ply-adjusted mate storage.
- **Per position (11-02 evening, `_depth_nodes_probe.py PAIRED=1`, d10, each position a fresh game, 300 each):**

  | knob | WAC total · median ratio | WAC, mate score seen / not | UHO total · median ratio |
  |---|---|---|---|
  | FIX_MDP | +13.7% · 0.999 (10 positions carry 58%) | +62.5% (76) / +10.4% (224) | **−0.7%** · 0.999 |
  | FIX_MATE_GATE | +2.2% · 1.000 (10 carry 69%) | +47.0% / −1.1% | +2.3% · 1.000 (10 carry 73%) |
  | FIX_KILLER_SPLICE | −1.2% · 0.998 | −1.4% / −1.1% | **−4.5%** · 0.958 |

  So MDP's bench increase is specific to the tactical, mate-heavy set and a few positions in it; on normal openings it is
  neutral. The killer fix is a broad node saving in normal positions (median −4.2%). Per-position ratios spread ±15% (IQR) for
  every arm, the usual tree-change scatter; totals and medians are the read.
- The mate gate costs 2% nodes for +4 solved, as expected of a gate that searches where it used to prune.
- FIX_MATE_50 never fired in 90 games with resign adjudication: correct by the unit check, practically inert in matches.

**Step 2 games (queue #139, 11-02 night → 11-03; full table in OPTIMIZATION_LOG).** All pass not-harmful: B1 + B2 bit 1 +8.2 on
eg_critical, MDP −4.2, killer splice −7.3 (edge of −1σ), mate gate +3.8, all on UHO; the candidate with all of them (plus A4, G5,
MATE_50) 0.0 ± 19.2. G5's replicate −1.0 ± 12.4, pooled +6.9 (−2.0 … +15.9) over 2,000 g: not a gain. Benches: the candidate
costs +10.6% WAC bench nodes and +6.7% d10 WAC (+65% on mate positions), −2.6% d10 UHO; **without MDP** it is −0.1% / +0.5% /
−6.5%, 252 solved. So MDP is held out of gen 6.5 and returns only with ply-adjusted TT mate storage (the coupling read above).
Confirm match for candidate-without-MDP: queue #140, +8.7 ± 18.7. **SHIPPED as gen 6.5 on 11-03** (owner-confirmed; bench
252 / 36,960,934). Still open after the ship: the 50-move item (MATE_50 already in, plus
FIX_TT_R50) on eg starts, and MDP + TT mates as one item. Outside-opponent read (queue #143, OPTIMIZATION_LOG 11-04): SF18 @ 1,100
nodes, paired 500 g, gen 6.5 − gen 6.4 = +0.80 pp ± 4.82 (95%) — not harmful against an outside engine either.

**FIX_TT_R50 on its own venue (queue #144, 11-04).** Venue: the 464 eg_critical starts (no ep square) with clocks set to halfmove
80 / move 40, so the 100-half-move horizon is inside every search. Fires there: at 30k nodes/move it changed 15 of 40 games
(divergence median ply 4, min 2), 8 results, 0 crashes; preflight 0 option warnings, the knob echoed, 132 position commands
carried the 80/40 counters. Match, `FIX_TT_R50=1` vs production, 10+0.1, no draw adjudication, 5 × 200 (seeds 14411-14415):
**+1.7 Elo (95% −7.9 … +11.4)**, +210 −205 =585, pentanomial [10, 60, 353, 69, 8]; chunks 0.0 / +5.2 / 0.0 / −12.2 / +15.7;
0 disconnects, 0 forfeits, no crash dumps. Real play: depth 15.4 vs 15.7, 2.44M vs 2.40M nps (endgames). Registered "ship only at
≥ +2σ, expected 0 … +15": **+0.35σ — not shipped by its registered bar.** It is not harmful where it fires (the correctness-item
bar would pass it); whether to adopt it on those grounds is the owner's call. Either way it is no Elo source, and on this venue
58.5% of games are drawn and 71% of pairs split 1-1: the 50-move horizon rarely decides a game even where it is inside every search.

**Scoping for the last open item, TT mates + MDP (read 11-04 morning, no code yet).**
- **The score frame is absolute game ply, not distance from the root.** A mated node returns `±(9999999 − state_history.size())`
  with the mated position on top of the history (`search_engine.cpp` 6095 / 7065 / 8234), so one value means "mate delivered at game
  ply G" from every node. A stored mate is wrong only when the same position is reached at a different ply (a transposition through
  a longer or shorter path), so the fix is the usual one in this frame: store `s + g` for a win and `s − g` for a loss, where g is
  the keyed position's ply (`state_history.size()` with it on top); read back `s' − g` / `s' + g`. The stored value is then the
  distance from that position, as consensus engines keep it.
- **g is already passed to every store.** All ~30 `addToSearchEvalCache` calls give `state_history.size()` as `num_plies` (unused
  today), and at each one checked the keyed position is on top (root 4558-4582, the helpers 5081 / 5513 read `state_history.back()`
  after the caller's `make_move`). Each site gets checked once more when the code goes in.
- **Reads that need the conversion (11 live):** `use_tt_entry` (one function, 7 calls — add a ply argument next to the 50-move
  clock it already takes), the null-move TT reuse (6466, 7177), the singular `ttScore` (6584, 7298; singular is off), the root-level
  exact read (8367) and the shadow probe's `tts` (1165, diagnostics only). Move-only reads (9933, 9974, the cutoff move writes)
  need nothing. The stored `alpha` / `beta` fields are read only in commented-out code.
- **What must not be stored:** the non-improving sentinels `±9999999` (pruned or futile moves, the window ends) sit just outside
  the mate range (a real mate has g ≥ 1). Today the `|s| ≥ 9,000,000` filter drops them with the mates. The new filter keeps
  `9,000,000 ≤ |s| < 9,999,999` and still drops the sentinels.
- **Out of scope:** the qsearch cache (it skips mates too; qsearch mates are rare: `qsmate` 0-202 per 40 games) and draws
  (`ENABLE_TT_STORE_DRAW` already exists; storing them is path-dependent through repetitions — a separate question).
- **Plan:** knob `FIX_TT_MATE` (0 = today), a round-trip unit check (store at one ply, probe at another, the expected mate
  distance), bench + d10 WAC / UHO node counts for `FIX_TT_MATE` alone and with `FIX_MDP` (the registration to beat: MDP alone
  +14% WAC bench nodes, because its mate-range answers were never remembered), a fire-count gate, then one match of the pair
  vs production. Correctness bar (not harmful), as for the rest of this list.

**Cost.** Step 0 ≈ 1 h of box time, no timed games. Step 1 ≈ 8 h (three matches). Step 2 ≈ a day of code and counters plus three
matches. Expected Elo: small — these are correctness items, valued for removing wrong behaviour without harm; the search arc's
larger forecast (+50 … +150) rests on the LMR / pruning / ordering redesign that comes after them.
