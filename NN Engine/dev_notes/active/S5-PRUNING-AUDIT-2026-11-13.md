# S5 audit: our pruning, reductions, extensions and history against the giants (2026-11-13)

Created 2026-11-13 for stage S5 of `SEARCH-V2-PLAN-2026-11-10.md` §3. Read-only: nothing was built or run, so every
statement about behaviour is from the code as written, and consequences that were reasoned rather than read are marked
INFERRED. The giants' side is in `reference/pruning-history-giants-2026-11-13.md` (called "the reference" below).

Scope: the interior at the shipped preset, which is `SEARCH_ARM = 1` (`search_engine.cpp:2761`), so `negamax` and
`score_move` in `search_v2.cpp`. The root, the pre-search and ply 1 are in `current/ROOT_SEARCH_MODEL.md`. Line numbers
are as read today; the symbol named beside each is the stable reference.

How to judge anything proposed here: `current/JUDGING-SEARCH-IDEAS.md`. Nothing in this file recommends removing a
part.

---

## 1. What we have, technique by technique

Runtime defaults were checked in both places: the header value and the shipped block (`search_engine.cpp:2723-2762`).
"rem" is `depth_limit - cur_depth`, the node's remaining depth. Our unit is the millipawn (pawn 1000).

### 1.1 Summary table

| technique | exists | on at shipped defaults | our form | SF19's form (reference §1) |
|---|---|---|---|---|
| Reverse futility | yes | yes (`ENABLE_RFP`, `RFP_MARGIN 600` shipped, header 1500) | null-window nodes, rem 1..6, `eval - 600 * rem >= beta`, returns the eval (`RFP_RETURN_BLEND 0`) | `!ttPv`, depth < 19, margin grows with depth, improving and correction terms, blended return |
| Razoring (interior) | not in `negamax` | — | root razoring only (ROOT_SEARCH_MODEL) | `eval < alpha - 482 * d^2` |
| Null move | yes | yes | no eval gate, no node-type gate, absolute target depth from a table keyed on the iteration depth | cut nodes, static-eval gate, `R = 7 + d/3 + excess` |
| IIR | yes | **off** (`ENABLE_IIR`) | keyed on a move-cache miss, rem >= 6, every node type | PV / cut nodes without a TT move, off the old PV |
| ProbCut | yes | **off** (`ENABLE_PROBCUT`) | `beta + 2200`, first 3 captures with per-square SEE >= 0, rem >= 5 | SEE threshold `probCutBeta - staticEval`, qsearch first, TT write |
| LMP | yes | yes | `i >= 1 + rem^2`, rem <= 5, quiets that are LMR-eligible | `(3 + d^2) / (2 - improving)`, no depth cap |
| Quiet futility | yes | yes | the **child's** static eval + `{450, 650, 950}` at rem 2 / 3 / 4 | the **parent's** static eval + `119 * lmrDepth + ...`, lmrDepth < 12 |
| History pruning | yes | **off** (`ENABLE_HIST_PRUNE`) | butterfly `< -512 * rem`, rem <= 4 | continuation 1 + 2 + pawn `< -4136 * d` |
| SEE pruning, quiets | yes | **off** (`ENABLE_SEE_PRUNE`) | recapture gain above 0 after the move, rem <= 3 | `-23 * lmrDepth^2` |
| SEE pruning, captures | yes | yes (`SEE_PRUNE_CAPTURES`) | `see < -1000`, rem <= 3, flat margin | `-(177 * d + captHist * 34 / 1024)` |
| Capture futility | no | — | — | `staticEval + 234 + 247 * lmrDepth + victim + captHist` |
| LMR | yes | yes (`LMR_SHAPE 1`) | log product, saturated at the clamp; quiets only | log product in 1/1024 ply with about fifteen terms; all moves |
| LMR history term | yes | yes (`ENABLE_STATSCORE_LMR`) | `(main + cont1) / 683`, clamp +-2 plies, extra reduce-more | `statScore * 439 / 4096` in 1/1024 ply |
| improving | yes | **off** (`ENABLE_IMPROVING`) | one ply more when not improving, cheap eval, near the leaves | scales the base by 197/512, also in LMP, RFP, null, ProbCut |
| Re-search rule | yes | yes (`VERIFY_MARGIN 16000`) | near-miss fail lows re-searched 2 plies short, full window; differs by side | fail highs re-searched, depth adjusted by the result |
| Check extension | yes | yes (`CHECK_EXTENSION 3`) | one ply per checking move, 3 per path, dropped for a losing check | none |
| Singular extension | yes | **off** (`ENABLE_SINGULAR`) | `+1` only; no multi-cut, no negative extension | `+1..+3`, multi-cut, `-3` |
| Passed-pawn extension / exemption | yes | off (`PE_PASSER_EXT 0`, `ENABLE_PASSER_PRUNE_EXEMPT`) | — | none (Triumviratus reduces passer pushes less) |
| Killer / counter protection | yes | yes (`PROTECT_KILLERS`, index <= 8) | not reduced, not LMP-pruned | no killers at all |

### 1.2 Reverse futility (`negamax`, `search_v2.cpp:403-433`)

Gate: `ENABLE_RFP`, not in check, `beta - alpha == 1`, rem in `[RFP_MIN_DEPTH 1, RFP_MAX_DEPTH 6]`, beta not a mate
score. Test: `s * rfp_eval - RFP_MARGIN * rem >= beta`. The eval is the full eval plus corrections (both correction
tables are off). This is also the lazy-eval site `LAZY_SITE_RFP` (`:411-422`).

Differences from SF19: no `ttPv` or TT-move condition (we have neither), linear margin without improving, depth cap 6
against 19, and the raw eval is returned. `RFP_RETURN_BLEND` exists and is 0.

**The node's static eval exists only here.** It is computed when the RFP gate could fire (null window and rem <= 6),
or when `ENABLE_NULL_EVAL_GATE` is on (it is off) (`:403-407`). At PV nodes and at rem > 6 the node has no static
eval. Everything in SF19 that reads the node's eval (improving, the null gate, parent futility, the alpha-against-eval
LMR term) has nothing to read at those nodes today.

### 1.3 Null move (`negamax`, `search_v2.cpp:435-496`)

Gate (`:437-438`): `cur_depth >= 4` for the root side and `>= 3` for the other (`NULLMOVE_CURDEPTH_MAXI / _MINI`),
`depth_limit >= 5` (the iteration's limit, not rem), the last move was not a capture and not a null move, not in
check, `!isUnsafeForNullMovePruning` (with `NULL_NPM_GUARD 2` shipped).

Not in the gate: any eval test (`ENABLE_NULL_EVAL_GATE` off), any node-type test (it runs at PV nodes), any minimum
remaining depth, and the singular exclusion state.

Depth (`:447-464`): `reduced_depth = depth_reduction(depth_limit) - NULLMOVE_EXTRA 2 - (depth_limit >= 10)`. This is
an **absolute target depth** read from `DEPTH_REDUCTION`, indexed by the iteration's limit. In the `LONG_FORMAT`
preset the table is `d - 1` up to 9, `8` at 10, `d - 2` for 11 to 13 and **12 for every depth from 14 up**
(`search_engine.h:194-215`). The null child then has `reduced_depth - cur_depth - 1` plies left.

INFERRED consequences, to confirm with a counter before relying on them:
- At iteration depth 10 the target is 5, so a null move at ply 3 gets 1 ply and at ply 4 or deeper goes straight to
  the q-search. At depth 12 the target is 7. From depth 14 up the target is 9 at every iteration.
- So our null move is "pass, then q-search" at most nodes, and its effective R grows with the node's ply instead of
  with its remaining depth. SF19's `R = 7 + d/3` is also a q-search at remaining depth up to about 10, so the depth is
  less different than it looks; **the gates are the real difference**: we try it at every node type and whatever the
  eval says, and pay a q-search each time it fails.
- The other presets' tables were not read.

TT shortcut for the null child: an EXACT entry of sufficient depth only (`:465-473`). Window: `(beta - 1, beta)` with
`FIX_MAX_NULL 1` shipped (`:481`). No verification search at any depth. `ENABLE_NULL_MATE_CLAMP` is false at the
shipped preset (`search_engine.cpp:2114`, `:2775`; the header says true: the runtime default wins).

At the shipped `LMR_SHAPE 1`, `DEPTH_REDUCTION` no longer feeds LMR (`reduced_search_depth`, `search_engine.cpp:10295-
10313`, returns before the table-based line). INFERRED: in the interior the null move is the table's only reader, so
the table can be re-fitted for the null move alone.

### 1.4 IIR (`search_v2.cpp:498-501`), off

`depth_limit -= 1` when not in check, rem >= `IIR_MIN_DEPTH 6`, and the position has no cached move list
(`moveGenCacheHasMoves`). The trigger is "first visit", our stand-in for SF19's "no TT move"; it has no node-type gate.
Memory `iir-is-the-venue-correct-node-saver` holds its record.

### 1.5 ProbCut (`search_v2.cpp:573-607`), off

`probcut_beta = beta + PROBCUT_MARGIN 2200`; walks the node's own move list and takes the first
`PROBCUT_CANDIDATES 3` captures or promotions with `see(move.to_square, ...) >= 0`; searches each at
`depth_limit - 3` on a null window; returns the raw score. Differences from SF19: no SEE threshold tied to
`probCutBeta - staticEval`, no q-search before the reduced search, no TT write, no `improving`, the per-square exchange
value (MOVEGEN-AUDIT's per-square SEE issue), and no skip of the excluded move.

### 1.6 The move loop: order of tests in `score_move` (`search_v2.cpp:869-1165`)

The move is already made when these run (unless `SV2_PRUNE_EARLY`, off, decides LMP first through `lmp_before_make`).

1. Repetition / fifty-move draw of the child (`:884`).
2. Check extension (`:893-908`), §1.9.
3. Child TT probe and cut (`:911-923`). **We have no node-entry TT cutoff; the parent probes the child's entry.**
4. Move 0, or rem 1 on the non-root side (A3): full-window search and return (`:927-935`).
5. `lmr_eligible`: `i != 0`, quiet, not a promotion, does not give check, node not in check (`:939`). Guards: PV
   (off), killer / counter at `i <= 8` (on), capture chain (off), TT depth (off), passer (off), the OTV no-reduce
   window (off) (`:940-976`). The result is `do_lmr`.
6. LMP, history pruning, SEE pruning of quiets: all require `do_lmr` (`:988-1017`).
7. SEE pruning of captures (`:1019-1027`).
8. Scout window (`:1031-1032`), A2.
9. Inside `do_lmr`: futility on the child's eval (`:1037-1059`), then the reduction and its adjustments
   (`:1060-1093`), then the reduced scout.
10. Re-search rules (`:1139-1163`), A1 / VERIFY.

Because pruning hangs off `do_lmr`, **a killer or counter move in the first nine is exempt from LMP, history pruning
and futility as well as from LMR**, and captures, promotions and checks are never reduced. SF19 reduces and prunes all
of those with their own terms.

### 1.7 LMP, futility, history and SEE pruning

- **LMP** (`:988-999`): `i >= LMP_BASE 1 + LMP_SCALE 1 * rem^2`, rem 1..`LMP_MAX_DEPTH 5`. Thresholds 2, 5, 10, 17, 26.
  SF19 without improving: 2, 3, 6, 9, 14; with improving: 4, 7, 12, 19, 28. Ours sits near SF19's improving arm at
  every node and has no improving split and a depth cap. `i` is the index in the cached list, counting captures. The
  pruned move returns `-MATE` and the loop continues with the next move; SF19 instead stops generating quiets.
- **Futility** (`:1037-1059`): rem 2..4, `depth_limit >= 5`, not ply 1 on the non-root side, not a pin defender.
  Computes the **child's full static eval** and prunes when `eval + {450, 650, 950}[rem - 2] < alpha`
  (`FUTILITY_MARGINS_EFF`, `search_engine.cpp:480`). This is lazy-eval site `LAZY_SITE_FUT`. Cost: one eval per late
  quiet at those depths, pruned or not, where SF19 reads one parent eval for all of them. The pruned move's eval is
  kept as `best_early_eval`, the node's value if everything is pruned (`:818-819`).
- **History pruning** (`:1001-1008`), off: butterfly only, `< -HIST_PRUNE_COEF 512 * rem`, rem <= 4.
- **SEE, quiets** (`:1010-1017`), off: `see(to, ...)` on the position after the move `> SEE_PRUNE_MARGIN 0`.
- **SEE, captures** (`:1019-1027`), on: `< -1000`, rem 1..3, not when the node is in check or the move extends. It uses
  the per-move exchange value only with `SEE_PER_MOVE & 4`; the shipped value is 3, so **this site still reads the
  per-square value** (`:1020-1021`).
- `FIX_MATE_GATE 1` (shipped) switches all of these off while every move so far is mated (`:980-984`).

### 1.8 LMR as built

Base, `reduced_search_depth` with `LMR_SHAPE 1` (`search_engine.cpp:10295-10313`, table at `:2087-2097`):

```
reductions[x]    = (LMR_PRODUCT_K 2480 / 100) * ln(x)      (0 for x < 2)
red              = reductions[rem] * reductions[max(i - 2, 1)] / LMR_PRODUCT_DIV 64
red              = red * 2.0 / phase_scale                  (1.5 .. 2.25 by material)
red              = clamp(red, 0, rem - 2)
target           = depth_limit - red                        (absolute depth; min 2)
```

That is `9.61 * ln(rem) * ln(i - 2)` plies before the clamp. For `i >= 4` it exceeds `rem - 2` at every remaining depth
up to about 22 (INFERRED from the constants; memory `lmr-product-schedule-is-fully-saturated-against-the-rem-2-clamp`
measured the same on 09-05). So in practice:

- `i = 1, 2, 3`: no base reduction (`ln(1) = 0`).
- `i >= 4`: the child gets **one ply**, whatever the node's depth.
- Light endgames (phase score >= 117) return the table value, not the product (`:10289-10294`, `FIX_LMR_EG 0`).
  INFERRED: with `LONG_FORMAT`'s table that is a target of `depth_limit - 1` or `- 2`, so those positions barely
  reduce.

Adjustments (`search_v2.cpp:1061-1093`):

- **statScore** (`history_lmr_delta`, `search_engine.cpp:1791-1821`): `statScore = main + cont1` (`cont2` is off);
  `delta = clamp(statScore / STATSCORE_DIVISOR 683, -2, +2)` plies. A negative delta gets a further
  `-min(rem / HISTORY_LMR_SCALE 2, 2)`. The target is then clamped to `[2, depth_limit]`, an absolute lower bound of 2,
  not `cur_depth + 1`. INFERRED: with the base already at one ply, any negative delta sends the child straight to the
  q-search (`cur_depth >= depth_limit`, `search_v2.cpp:376-380`), and a positive delta gives it 2 or 3 plies.
- improving (off), capture chain (off), `LMR_MIN_REM` / `LMR_REM_FLOOR_PCT` (off).

So the shipped reduction has three levels for a late quiet: q-search, 1 ply, or 2 to 3 plies, chosen by history. The
log product only decides that moves 1 to 3 have no base reduction (the history delta still applies to them, so a
move 1 to 3 with negative statScore loses 1 to 4 plies).

**The re-search rule carries the real depth** (`:1139-1163`). After the scout:

| side | result of the reduced scout | action at defaults |
|---|---|---|
| root side (maximizer) | inside the window | full-depth re-search |
| | fails low by less than `VERIFY_MARGIN 16000` (16 pawns) | re-search at `depth_limit - VERIFY_RESEARCH_REDUCTION 2`, **full window** |
| | fails high (`>= beta`) | **accepted**, even from a one-ply scout (A1; `FIX_RESEARCH 0`) |
| other side (minimizer) | its own cutoff, within 16 pawns | re-searched 2 plies short (it verifies its cutoffs) |
| | a fail low for it | accepted unverified |

INFERRED: 16 pawns covers nearly every fail low, so nearly every reduced move of the root side is searched twice: a
1-ply scout, then `depth - 2`. The effective reduction for a move that fails low is therefore 2 plies flat, plus the
scout's cost; the scout's main job is to let a fail high cut at once. This matches SEARCH-V2-PLAN's note that "the
bound, the reduction scale and VERIFY are ONE mechanism".

Against SF19: SF19 re-searches fail highs and trusts fail lows; our root side does the reverse, and the two sides
differ from each other (A1). SF19's reduction varies smoothly from 0 to many plies by depth, move number, node type and
history; ours is flat. SF19's scout is always a null window at alpha; our non-root side scouts at `(beta - 1, beta)` of
its own frame (A2, `:1031`, `FIX_MIN_SCOUT 0`), and the VERIFY re-search uses the full window at PV nodes.

### 1.9 Extensions

- **Check extension** (`:890-908`): every move that gives check extends one ply, at most `CHECK_EXTENSION 3` per
  path, unless `see(to) > SEE_EXTEND_MARGIN 300` on the position after the move (a losing check). It shares its
  per-path counter with the passer extension. Extended moves are also outside LMR, LMP and SEE pruning. SF19 and
  Reckless have no check extension; they reduce checks less. SF11 has one.
- **Singular** (`search_v2.cpp:609-629`, `:654-673`), off: gate `rem >= SINGULAR_MIN_DEPTH 6`, the node's TT entry has
  a move, is a fail-high bound, has depth `>= rem - 3` and a non-mate score, node not in check, fewer than
  `SINGULAR_MAX_EXT 8` on the path. `sb = tt_score - SINGULAR_MARGIN 2 * rem` (2 millipawns per ply: SF11's number on
  SF11's scale, never converted to ours). Exclusion search at `rem / 2`. Outcome: `+1` or nothing. No multi-cut, no
  negative extension, no double extension.
- The TT move it needs is written only at a cutoff, straight onto the node's existing entry, and only when
  `ENABLE_SINGULAR` or `ENABLE_TT_MOVE` is on (`:785-790`); `store_node_tt` does nothing unless `ENABLE_NODE_TT`
  (`search_engine.cpp:5512`). Memory `node-tt-dependency-starved-singular-and-ttmove` is the record.

### 1.10 Known asymmetries, live in the port

From memory `search-correctness-asymmetries-2026-10`, each a `RootSide` branch in `search_v2.cpp`:

| id | what | where | shipped |
|---|---|---|---|
| A1 | the re-search rule is not mirrored | `:1141-1147` | `FIX_RESEARCH 0` |
| A2 | the non-root side scouts at the other bound | `:1031` | `FIX_MIN_SCOUT 0` |
| A3 | full-window search of every move at rem 1, non-root side only | `:927` | `FIX_MIN_RD1 0` |
| A4 | null-move window | `:481` | `FIX_MAX_NULL 1` (fixed) |
| — | null-move ply gate 4 / 3 by side | `:436` | by design, tuned with the pre-search |
| — | futility skips ply 1 on the non-root side | `:1037` | — |

A2 and A1 have **no valid game evidence** (the 10-22 note in that memory voids every game number for them).

---

## 2. Our history tables

| table | index | written | read |
|---|---|---|---|
| `historyHeuristics` | `[side][from][to]` | cutoff bonus and malus; plus an unbounded add at node exit (below) | ordering; statScore; LMP exemption (off); history pruning (off) |
| `counterMoveHeuristics` | `[side][prev from x to][from x to]` (4096 x 4096, 134 MB per the comment at `cache_management.h:316`) | cutoff bonus and malus | ordering; statScore |
| `contHist2` | same shape, the move two plies back | only with `ENABLE_CONT_HIST_2PLY` (off) | statScore, ordering when on |
| `pieceContHist`, `counterMoveHeuristicsP`, `contHist2P` | piece x to keys | off (`ENABLE_PIECE_CONTHIST`, `CONT_HIST_PIECE_KEY 0`) | — |
| `threatHist` | `[side][from attacked][to attacked][from][to]` | off (`ENABLE_THREAT_HIST`) | — |
| `captureHistory` | `[side][from][to]`; victim key off (`CAPTURE_HIST_VICTIM 0`) | capture cutoff bonus and malus | capture ordering |
| killers (2 per ply), `counterMoves[from][to]` | | quiet cutoff | ordering; LMR / LMP protection |
| `moveFrequency` | | (not read here) | ordering |
| correction (`pawnCorrHist`, `CORR2`) | pawn key | off | RFP eval |

Update at a quiet cutoff, `quiet_cutoff_history` (`search_v2.cpp:157-210`):

```
b = rem * rem * HISTORY_BONUS_SCALE 100 / 100
hist_update(h, d):  d clamped to +-MAX_HISTORY 16384;  h += d - h * |d| / MAX_HISTORY     (cache_management.h:1513)
```

The cutoff move gets `+b` in the butterfly and 1-ply continuation tables; every quiet searched before it gets
`-b / MALUS_DIV 1`, the same size for each. Captures: the same `b` and malus in `captureHistory`
(`capture_cutoff_history`, `:213-232`). All tables are also halved on a node count: every `DECAY_INTERVAL 35000` nodes
for butterfly and capture, every 16th such interval for the continuation tables
(`increment_node_count_with_decay`, `search_engine.cpp:10101-10123`).

Against SF19 (reference §2):

| aspect | ours | SF19 |
|---|---|---|
| bonus size | `rem^2`, uncapped (100 at rem 10, 400 at rem 20) against a limit of 16384 | linear in depth, capped (`min(133 d - 81, 1487)` against 7183) |
| malus | equal for every tried quiet | larger than the bonus, decaying by 921/1024 per move tried |
| continuation key | from x to of both moves | piece x to of both, split by in-check and capture |
| continuation plies | 1 (2 built, off) | 1 to 6 written, weighted; agreement between plies scales the update |
| capture key | from, to | piece, to, victim type |
| extra tables | none on | pawn-structure, low-ply, TT-move, correction |
| bound | gravity **and** periodic halving | gravity only |
| writes besides the cutoff | one unbounded add (below) | TT cutoff, eval change, fail-low credit to the prior move, re-search confirmation |
| killers / counter move | yes, and they gate LMR / LMP | none |

Two aging mechanisms run together (gravity and halving). Memory `gravity-is-built-and-statscore-decalibrates-it` is the
record of how that couples to `STATSCORE_DIVISOR`.

---

## 3. Defects and hazards seen

**D1. An unbounded history write that also credits fail-low nodes** (`search_v2.cpp:827-831`). When the loop ends
without a cutoff and the best move is quiet, `historyHeuristics[...] += rem`, outside `hist_update`, so it bypasses the
gravity bound; and it runs whether or not the move raised alpha, so at an all-node the least-bad move is rewarded. SF19
updates only when a move beat alpha (`search.cpp:1570`). SEARCH-V2-PLAN already lists "five unbounded
`historyHeuristics +=` sites"; this is the one in the v2 interior. Any change is a results change, so it needs a knob.

**D2. The singular exclusion search is not isolated** (latent, `ENABLE_SINGULAR` off). MOVEGEN-AUDIT C1 covers the
shared move buffer. Beyond that, inside the exclusion call:
- the null move runs (`:437-438` has no exclusion test; SF19 gates it, `search.cpp:1011`);
- IIR can shorten it (`:499`);
- ProbCut does not skip the excluded move (`:580-606`), so with both on, the TT move itself can cut the search that
  was meant to exclude it;
- a cutoff promotes the exclusion search's move to the front of this node's cached list
  (`updateMoveCacheForBetaCutoff`, `:791-796`) and writes killers and history (`:797-800`), none gated on `excluding`;
- the TT-move write (`:785`) and `store_node_tt` (`search_engine.cpp:5516-5518`) are correctly gated.

**D3. Turning singular, ProbCut or TT-move ordering on turns lazy generation off** (`search_v2.cpp:508-509`). With
`SV2_LAZY_GEN` in play, a test of any of the three is also a speed change. `SV2_LAZY_GEN` is 0 today, so this is a
hazard for the S3 / S5 overlap, not a live defect.

**D4. `SINGULAR_MARGIN 2` is in SF11's units** (`search_engine.h:2910`). On our scale it is 2 millipawns per ply, far
tighter than SF11's `2 * depth` on a 128-213 pawn. INFERRED: the "singular is neutral" verdict was measured with a
margin about five to eight times tighter than its model, as well as with D2.

**D5. The capture SEE prune reads the per-square value at the shipped `SEE_PER_MOVE 3`** (`:1020-1021`). Bit 4 exists
for it. Not a bug; an unshipped part of the same idea.

**D6. Header and runtime disagree for `ENABLE_NULL_MATE_CLAMP`** (header true, start-up read defaults false,
`search_engine.cpp:2114`, `:2775`). The engine is correct per CLAUDE.md (gen 6.6's bundle was not kept); the header
comment misleads a reader.

**D7. The near-miss rule (VERIFY) fires on moves that were not reduced** (`score_move`'s re-search rule in `search_v2.cpp`;
the same `do_lmr` test in v1's maximizer and minimizer, `search_engine.cpp`). The condition is `do_lmr && fails low by less
than VERIFY_MARGIN`, not `was_reduced && ...`. A do_lmr move that took no reduction (moves 1-3 without a negative history
delta) was already searched at full depth; VERIFY then searches it again at `depth - VERIFY_RESEARCH_REDUCTION`, 2 plies
SHALLOWER, and that score replaces the full-depth one (and goes into the TT at the shallower depth). Measured (#233): 21-27%
of the root side's VERIFY re-searches are of this kind (487k at WAC depth 10). The same shape affects a move reduced by
exactly 1 ply (12% of reduced scouts): its re-search is 1 ply shallower than its scout. The header's rationale for VERIFY
("the buried winning capture that history-LMR over-reduces") is about reduced moves. Possibly a productive bug (the shallower
score may land above alpha where the full-depth one did not, an optimism for the root side): test it behind a new knob (re-search only
when the re-search is deeper than the first search), never as a silent fix; retune `VERIFY_MARGIN` with it. Belongs to
Stack D.

---

## 4. Dependencies: what leans on our own systems

Owner's rule: never rip a part out alone. Each row says what a change must carry with it.

| our system | what in the pruning set leans on it | changes together | retunes together |
|---|---|---|---|
| **Move cache, promote-to-front order** | The index `i` in LMP, the LMR base and killer protection is the position in the cached list, not in a fresh sort. IIR's trigger is a cache miss. The promoted cutoff move stands in for a TT move. | A staged picker or a TT move changes what `i` means, so LMP's thresholds, the "moves 1-3 unreduced" rule and `PROTECT_MAX_IDX` move with it. IIR needs a second trigger (no TT move) beside the cache one. Singular needs the exclusion call kept out of the cache (D2). | `LMP_BASE / LMP_SCALE`, `PROTECT_MAX_IDX`, `IIR_MIN_DEPTH` |
| **Root pre-search** | It writes history, killers, counter moves and the cache before the main search, so the interior starts warm. The null gates at plies 3 / 4 were tuned with it present. | A root without the pre-search changes the history the first plies read; a low-ply history table is the giants' answer to the same need. The null gates and `PLY1_NODE_PRUNE` belong to the root arc's stacks (ROOT_SEARCH_MODEL §5), not to S5 alone. | `NULLMOVE_CURDEPTH_*`, `HISTORY_BONUS_SCALE` |
| **Null move on an iteration-depth table** | `DEPTH_REDUCTION` is, in the interior, read only by the null move (§1.3). The gate has no eval test, so RFP is the only eval-based node cut. | An eval gate on the null move needs a node eval at every node (S4). A remaining-depth R replaces the table for the null move; the table stays for `LMR_SHAPE 0`. | `NULLMOVE_EXTRA`, `RFP_MARGIN` (the two overlap: both cut nodes that stand above beta) |
| **Lazy eval** (`LAZY_V2 7`) | Two of its three sites are pruning sites: RFP (`LAZY_SITE_RFP`) and the child's futility eval (`LAZY_SITE_FUT`). A lazy exit returns a bound, not a value. | Moving futility to the parent's eval removes the futility site. `improving`, the null eval gate and any "alpha minus eval" LMR term need a real value at the node, so they cannot read a lazy bound; either the node eval is exact, or those readers accept only the sign. Any change to `RFP_MARGIN` or the futility margins moves the lazy thresholds. | watch list W1-W5 in `current/LAZY-EVAL.md` §7 after every change here |
| **Node eval only at null-window nodes with rem <= 6** | improving (off) uses its own cheap eval stack; nothing else has an eval at PV nodes or deep nodes. | S4's node-entry eval is the prerequisite for SF19's forms of futility, improving, null gate and hindsight. Cost: memory `eval-is-a-third-of-node-cost-not-half`. | every margin that reads it |
| **TT probed by the parent, no node move** | No node-entry TT cut, no `ttPv`, no TT-move terms in LMR, no `ttCapture`; singular and TT-move ordering starve without the move. | `ENABLE_NODE_TT` (or the cutoff write at `:785`) is a prerequisite of singular, of an SF-style IIR and of the TT-move LMR terms. | — |
| **statScore reads the tables it is calibrated to** | `STATSCORE_DIVISOR 683` was fitted to today's bonus size, gravity and halving. | Any change to the bonus formula, the malus, the keys or the aging changes the distribution statScore reads. | `STATSCORE_DIVISOR / OFFSET / CLAMP`, `HISTORY_LMR_SCALE` |
| **VERIFY + scout bound + saturated base** | One mechanism (§1.8). The margins and `VERIFY_RESEARCH_REDUCTION` were tuned around A1 and A2. | An unsaturated base, a symmetric scout and an SF-style re-search rule only make sense as one stack; each alone has already been screened and lost or stayed unreadable. | `VERIFY_MARGIN`, `VERIFY_RESEARCH_REDUCTION`, `LMR_PRODUCT_DIV`, LMP, futility margins |
| **Killers and counter moves** | They exempt moves from LMR and all `do_lmr` pruning, and shipped at +12.4 Elo. | A history-only scheme (the giants have no killers) would have to replace that protection with a history term of equal reach; keep both behind switches. | `PROTECT_MAX_IDX` |
| **Check extension** | Checks are extended and never reduced. | "Reduce checks less" (Reckless `-955`, Triumviratus `LmrGives`) is the alternative; it only exists once captures and checks can be reduced at all. | `CHECK_EXTENSION`, `SEE_EXTEND_MARGIN` |

---

## 5. Options for S5 as stacks

Ordered smallest and most reversible first. Every knob named as "new" would default to off and byte-identical. Our
record says search parts are antagonistic, so each stack is tested whole first, then split only if it wins. None of
these removes an existing part.

**Stack 0. Counters only (no behaviour change).**
Port the LMR and prune counters to `search_v2.cpp` (SW4 in the plan) and add: the child's remaining depth after
reduction (histogram), the share of reduced scouts that VERIFY re-searches by side, the null child's remaining depth
at entry, and null tries against null cuts by node type.
- Prerequisite: none.
- Why first: §1.3 and §1.8 rest on INFERRED arithmetic; one fixed-depth run confirms or corrects both.
- Cheap test: WAC + quiet corpus at depth 10 and 12, counters only, identical node counts.

**Stack A. History hygiene.**
Knobs: a new switch for D1 (route the node-exit write through `hist_update`, and separately require that the move
raised alpha); `ENABLE_HISTORY_DECAY 0` against 1 (one aging mechanism instead of two); a decaying malus (new,
SF19's `921/1024` per tried move); a capped linear bonus as an alternative to `rem^2` (new).
- Prerequisite: none.
- Retune with it: `STATSCORE_DIVISOR` (re-derive with `ENABLE_STATSCORE_PROFILE`).
- Cheap test: first-move cutoff rate (`g_fh_first / g_fh_total`) and nodes at fixed depth, then games.

**Stack B. History keys and reach** (on top of A).
Knobs that exist: `CONT_HIST_PIECE_KEY`, `ENABLE_CONT_HIST_2PLY` with `CONT2_GRAVITY_DIV`, `ENABLE_THREAT_HIST`,
`CAPTURE_HIST_VICTIM` (measured -4% nodes on 08-28). New: continuation at plies 4 and 6; a fail-low credit to the
opponent's previous quiet move; a re-search confirmation bonus.
- Prerequisite: A's bounded writes. The 2-ply table's "loss" was at equal weight and fixed depth, not a games verdict.
- Retune with it: `STATSCORE_*` weights and divisor.
- Cheap test: same as A, plus the `statScore ~ 0` bucket share (memory `history-sparsity-is-coverage-not-architecture`).

**Stack C. History-driven shallow pruning** (the consumers B feeds).
Knobs that exist: `ENABLE_HIST_PRUNE` (re-keyed on continuation history, new), `ENABLE_SEE_PRUNE` with a margin that
grows with depth (new form), `SEE_PER_MOVE` bit 4, `ENABLE_LMP_HIST_EXEMPT`.
- Prerequisite: B, since butterfly-only history pruning is what the record shows failing. `guard-rescues-aggression-
  but-knobs-are-chaotic` is the warning: test the pair (prune + exemption), not the parts.
- Cheap test: prune counts by `rd` (the `SV2_MOVE_PROBE` counters exist), then the low-depth regret ruler.

**Stack D. LMR v2: one mechanism, replaced as one.**
Knobs: `FIX_MIN_SCOUT 1` + `FIX_RESEARCH 1` + `FIX_MIN_RD1 1` (symmetric halves), an unsaturated base (raise
`LMR_PRODUCT_DIV` or a new fractional schedule in 1/1024 ply), `VERIFY_MARGIN` reduced or 0 with SF19's
"re-search on a fail high, deeper or shallower by the result" (new), the statScore term rescaled to fractional plies.
- Prerequisite: Stack 0's numbers; the plan already assigns "LMR v2 replaces A1-A3 together" to S5.
- Retune with it: LMP, the futility margins, `RFP_MARGIN` (memory `eval-accuracy-payoff-is-pruning`: re-sweep
  combined).
- Cheap test: a 2 x 2 at fixed depth (today's bound and rule against the symmetric ones, saturated against
  unsaturated), read on time to depth and on the SF18 move test, before any games.
- Later additions once it stands: captures and checks reducible with their own terms, the causal reduction (reference
  §5.2; needs only the child's cutoff move), cut-node and all-node terms (memory
  `cutnode-allnode-does-not-separate-wrong-lmr-reductions` says the first read was negative here, so they come last).

**Stack E. Node eval consumers** (after S4's node-entry eval).
Knobs: `ENABLE_IMPROVING` on the node eval, `ENABLE_NULL_EVAL_GATE`, `ENABLE_NULLMOVE_EVAL_R`, futility on the
parent's eval (new), `RFP_RETURN_BLEND`, RFP beyond rem 6, a remaining-depth null R (new).
- Prerequisite: S4, and a decision on lazy bounds (§4).
- Retune with it: `RFP_MARGIN`, `NULLMOVE_EXTRA`, the lazy thresholds; re-run LAZY-EVAL's W list.
- Cheap test: null tries and cuts by node type from Stack 0; evals per node before and after.

**Stack F. TT move, IIR, singular.**
Knobs: `ENABLE_NODE_TT` or `ENABLE_TT_MOVE`, `ENABLE_IIR` with a no-TT-move trigger (new), `ENABLE_SINGULAR` with the
D2 gates (new), a converted `SINGULAR_MARGIN` (D4), then multi-cut and the negative extension (new).
- Prerequisite: the exclusion search isolated. Two designs are in the reference §1.5: SF19's (a list owned by the
  call, plus gates) and Reckless's (run the test before the node builds its list, which avoids MOVEGEN-AUDIT C1
  without a second buffer; the cache promotion still needs its gate).
- Cheap test: the counters already there (`g_sing_eligible`, `g_sing_gatepass`, `g_sing_fire`) at fixed depth: if few
  nodes have a TT move at sufficient depth, the stack is starved and waits for the TT work. Then MOVEGEN-AUDIT's
  buffer-compare debug check.
- Check extension against singular: both extend forcing lines; test with `CHECK_EXTENSION` at 3 and lower as a 2 x 2.

**Stack G. ProbCut in SF19's form.**
Knobs: `ENABLE_PROBCUT` with an SEE threshold from the node eval (new), a q-search before the reduced search (new), a
TT write, the excluded-move skip (D2).
- Prerequisite: E's node eval, the per-move exchange value, and F's gates if both are on.
- Cheap test: ProbCut tries, cuts and the nodes spent in failed tries at fixed depth.

Order of dependence: 0 -> A -> B -> C; 0 -> D; S4 -> E -> G; node TT -> F. A / B and D touch disjoint code and could
share a games slot only under `bundle-to-save-box-time-only-when-the-parts-stay-readable`.

---

## 6. Open questions for the owner

1. **Stack 0 first?** Two central claims here (the null move is mostly a q-search, and LMR is "one ply, then
   `depth - 2` through VERIFY") are arithmetic from the constants. Is a counters-only run the first S5 step?
2. **D1**: should the node-exit history write be treated as a defect to switch off behind a knob, or as an owner idea
   (credit for the best move of a node that did not cut) to keep and bound?
3. **Killers and counter moves** have no counterpart in SF19, Reckless or Triumviratus, and they shipped here at
   +12.4. Keep them as a fixed part of v2's history design, with the continuation tables added beside them?
4. **Check extension**: keep it as ours and test singular beside it, or also build "reduce checks less" as the
   traditional arm?
5. **Singular isolation**: SF19's design or Reckless's "test before the list"? The second fits the move cache with
   less change; the first is what a staged picker would give for free.
6. **The null move's place**: its gates belong partly to the root arc (plies 3 / 4, `PLY1_NODE_PRUNE`) and partly to
   S4-S5 (eval gate, remaining-depth R). One owner for both, or split as above?
7. **HistGrad and the causal reduction** are a rival engine's own ideas. Under "strength while staying distinctive",
   are they references to learn from only, or candidates to build behind switches? The causal reduction is close in
   spirit to capture chains and to the owner's refutation ideas.
8. **Lazy eval against `improving`**: accept an exact node eval wherever improving is read (cost), or let improving
   read only the sign of a lazy bound?
9. **Order**: history first (A-B-C, small and reversible) or LMR v2 first (D, the larger expected effect and the
   plan's named S5 item)?

## 6b. Stack 0 results (queue #232, 2026-11-14; counters only, node counts unchanged)

Depth 10 on WAC / STS300 / quiet (100) / v2 game set (100), and WAC at depth 12, for today's root (S0) and H2 (S2).
Bench with `ENABLE_LMR_COUNTERS=1` stayed 249 / 37,279,382; S2's nodes match H2's #230 ratios on every set.
Log: `E:\chess_data\q232.log`; stderr per run in `E:\chess_data\r4a\`.

| claim (§) | measured | verdict |
|---|---|---|
| LMR leaves the child one ply, whatever the depth (§1.8) | Child's remaining depth = 1 for 87% of LMR-eligible moves at d10 and d12 (WAC); at least 83% of the REDUCED scouts (bound: even if every unreduced move were in that bucket). Plies removed: 1 (12%), 2 (18%), 3 (33%), 4 (22%), 5 (12%), 6+ (3%) of reduced scouts at d10 WAC, i.e. the reduction is "whatever leaves one ply" | **CONFIRMED** |
| VERIFY re-searches nearly every reduced fail low of the root side (§1.8) | Root side: 86.4% (WAC d10), 80.4% (WAC d12), 97.1% (STS), 95.5% (quiet), 96.3% (game) of reduced scouts re-searched at `depth - 2`. Full-depth (PVS) re-searches 0.1-0.2% | **CONFIRMED**: the root side's real LMR is a flat 2-ply reduction plus a 1-ply scout, and its reduced fail highs are accepted from the 1-ply scout (A1) |
| (new) the other side | 91.2% (WAC) / 95.8% (STS) of its reduced scouts are accepted from the 1-ply scout; 8.2% / 3.5% PVS re-search; VERIFY 0.5% / 0.7% | The opponent's late replies are judged at one ply; ours at `depth - 2` |
| (new) reduced scouts by side | root side 2.9x (WAC), 2.3x (STS), 1.6x (game) the other side's | asymmetric; not explained yet |
| The null move is "pass, then q-search" (§1.3) | Null child's remaining depth 0 for 93.3% (WAC d10), 92.8% (STS), 91.4% (WAC d12) of tries; >= 3 plies: 0 at d10, 7.4k of 4.5M at d12 | **CONFIRMED** |
| The null move runs at PV nodes (§1.3) | 26% of tries at PV nodes (WAC d10), cut rate 32% there vs 48% at null-window nodes | **CONFIRMED** |
| (H2) same shape without the pre-search | S2: child rem 1 ~86%; root-side VERIFY 84.8% (WAC) / 96.1% (STS); null rem 0 ~93% | the interior is unchanged by the root |

**Follow-up counters (queue #233, 11-14; `[sv2_verify]`; node counts again equal #232's on every run):**

| | S0 WAC | S0 STS | S0 quiet | S0 game | S2 WAC | S2 game |
|---|---|---|---|---|---|---|
| nodes under VERIFY re-searches (outermost), share of all nodes | 23.6% | 29.8% | 28.4% | 24.4% | 23.0% | 24.7% |
| VERIFY re-searches of moves NOT reduced (root side), share of the root side's VERIFY re-searches | 20.8% (487k) | 21.9% | 21.4% | 24.0% | 24.4% | 26.9% |
| root-side reduced scouts that fail high, kept from the 1-ply scout | 1,777 (0.08%) | 4,658 | 1,020 | 1,645 | 1,574 | 1,531 |

So the root side's reduced scouts split: about 86% (WAC) to 97% (STS) re-searched at `depth - 2`; the rest fail low by
more than 16 pawns and are dropped; fail highs kept unverified are under 0.1%. **A1's accepted fail highs are numerically
small.** The cost sits in VERIFY: **a quarter to 30% of every search's nodes**, on both roots. The 1-ply scout's only real job
is dropping moves that lose more than 16 pawns.

**D7 (new, measured): VERIFY also re-searches moves that were never reduced** (see §3).

## 7. Not verified

- ~~Nothing was run: every "INFERRED" above, in particular the null child's depth (§1.3), the saturation of the LMR base
  and the share of scouts VERIFY re-searches (§1.8).~~ All three measured in §6b (Stack 0, 11-14) and confirmed.
- `DEPTH_REDUCTION` was read for `LONG_FORMAT` only; which preset the UCI binary plays under was not checked.
- The ordering side (how each table is weighted in `move_gen.h`, `moveFrequency`, `THREAT_ORDER_W`) was not re-read;
  MOVEGEN-AUDIT covers it.
- `isUnsafeForNullMovePruning` under `NULL_NPM_GUARD 2`, `decayHistoryHeuristics` and `relevant_pin` were not opened.
- The q-search's pruning (delta, per-move delta, the SEE filter) is outside this read.
- The v1 interior (`SEARCH_ARM 0`) was not compared line by line with the port; the port's identity run
  (`_sv2_identity.sh`) is the evidence that they match.
