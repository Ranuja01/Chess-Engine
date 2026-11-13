# S4 audit: our node-entry static eval and its consumers (2026-11-13)

Created 2026-11-13 for search v2 stage S4 (`SEARCH-V2-PLAN-2026-11-10.md` §3). A read of source only: nothing was
compiled or run, so every claim about behaviour is from the code as written. Statements marked INFERRED were not
traced to the end. The reference engines' side is in `dev_notes/reference/node-eval-giants-2026-11-13.md` ("giants
doc" below). How any option here is to be judged: `dev_notes/current/JUDGING-SEARCH-IDEAS.md`.

Scope: the interior that runs at the shipped default, `sv2::negamax` / `sv2::score_move` / `sv2::qsearch` in
`search_v2.cpp` (`SEARCH_ARM 1`, set by the shipped preset, `search_engine.cpp:2761`). The older maximizer / minimizer
pair in `search_engine.cpp` has the same sites (S1 was a one-for-one port) and is not described separately.
Line numbers are as read on 2026-11-13; the symbol beside each is the stable reference.

---

## 1. What the interior computes at node entry today

### 1.1 The node-entry eval (`rfp_eval`, `rfp_raw_eval`)

`sv2::negamax`, `search_v2.cpp:399-423`.

- **Computed only when a consumer wants it** (`rfp_want_eval`, :403-406): not in check, and either reverse futility is
  about to be tested (`ENABLE_RFP`, a null window `beta - alpha == 1`, remaining depth in
  `[RFP_MIN_DEPTH, RFP_MAX_DEPTH]`) or `ENABLE_NULL_EVAL_GATE` is on.
- Runtime values: `ENABLE_RFP true` (`search_engine.h:2863`), `RFP_MIN_DEPTH 1`, `RFP_MAX_DEPTH 6` (:2880-2881; no
  preset changes them), `ENABLE_NULL_EVAL_GATE false` (:2892; no preset changes it).
- **So at the shipped default a node has a static eval of its own only if it is a non-PV node, not in check, with
  remaining depth 1 to 6.** PV nodes, nodes with remaining depth 7 or more, and nodes in check have none.
- The value: `eval_by_mode(RFP_EVAL_MODE = 0, ..)` = `get_board_evaluation`, the full eval through the eval cache
  (`search_engine.cpp:1772-1789`), in the **root frame** (root side positive); `s * rfp_eval` is the side-to-move
  value. The corrections (`ENABLE_CORR_HIST`, `CORR2`, both off) are read first and added (:413-420).
- **Where it is kept: a local variable.** Not in a per-ply stack; not in the TT (`TTEntry` has no eval field,
  `cache_management.h:113-121`; `store_node_tt` takes none, `search_v2.cpp:857`). A child cannot read its parent's
  eval. The eval cache is the only place a second request for the same position finds it.
- No TT-value-as-better-eval step.

### 1.2 Consumers of the node-entry eval

| consumer | condition and formula | on at ship? | where |
|---|---|---|---|
| reverse futility | null window, bounds not mate, remaining depth 1-6: `s * rfp_eval - RFP_MARGIN * rd >= beta` → return `s * rfp_eval` (the raw eval; `RFP_RETURN_BLEND 0`) | yes; `RFP_MARGIN` **600** from the preset (`search_engine.cpp:2732`; header 1500, gens before 6.4 1000) | `search_v2.cpp:425-433` |
| null-move eval gate | `!ENABLE_NULL_EVAL_GATE \|\| s * rfp_eval >= beta` | **no** | :438 |
| null-move reduction by eval | `extra_R = min((s * rfp_eval - beta) / NULLMOVE_R_DIV, NULLMOVE_R_CAP)` | **no** (`ENABLE_NULLMOVE_EVAL_R false`, `search_engine.h:3042`) | :451-456 |
| OTV reference | computes the eval itself if absent | no (`ENABLE_OTV false`) | :568-571, :734-735 |
| correction update at node exit | reuses `rfp_raw_eval`, else evaluates again | no (corrections off) | :832-855 |

At the shipped default the node-entry eval has **one** consumer: reverse futility.

### 1.3 The other static-eval sites (not the node-entry eval)

| site | what is evaluated | formula | where |
|---|---|---|---|
| futility of a child | the **child** position, after the move is made | late, LMR-eligible move, `(RootSide \|\| cur_depth > 1) && rd <= 4 && rd > 1 && depth_limit >= 5`, no relevant pin: `early_score + FUTILITY_MARGINS_EFF[rd - 1] < alpha` → the move returns `early_score`. Margins `{200, 450, 650, 950}` × `FUTILITY_MARGIN_SCALE 100` (`search_engine.h:34`, :234), so 450 / 650 / 950 at rd 2 / 3 / 4; index 0 (200) is never read here because `rd > 1` | `sv2::score_move`, `search_v2.cpp:1037-1058` |
| qsearch stand-pat | the q-node | fail-hard: `static_eval >= beta` → return `static_eval`; raises alpha; node-level delta only when `ENABLE_QDELTA_PERMOVE` is off (it is on, `search_engine.h:2858`); per-capture `static_eval + pm_margin + victim <= alpha` | `sv2::qsearch`, :1262-1306 |
| horizon | the leaf when qsearch is off or at the q-depth cap | plain eval | :380, :1207-1212 |
| `g_evalStack` (improving) | the node, **cheap surrogate** | §2 | :394-397 |

### 1.4 The null move today (no eval involved)

`search_v2.cpp:435-496`. Conditions: `ENABLE_NULLMOVE`, `cur_depth >= NULLMOVE_CURDEPTH_MAXI (4)` for the root side's
nodes or `NULLMOVE_CURDEPTH_MINI (3)` for the other side's (`search_engine.h:3188-3189`), `depth_limit >= 5`, the
last move not a capture and not a null move, not in check, `isUnsafeForNullMovePruning` false. The reduction comes
from the **iteration** depth (`depth_reduction(depth_limit) - NULLMOVE_EXTRA (2)`, one more at `depth_limit >= 10`),
not from the eval or the remaining depth. The null move is tried whether the static eval is above beta or far below.

### 1.5 Late-move pruning and LMR today

- LMP: `rd in [1, LMP_MAX_DEPTH (5)] && i >= LMP_BASE (1) + LMP_SCALE (1) * rd * rd`, history exemption
  (`search_v2.cpp:987-999`, `search_engine.h:1374-1377`). No eval, no improving.
- LMR: `reduced_search_depth(depth_limit, cur_depth, pin, i, state)` plus the history delta; the improving step only
  under `ENABLE_IMPROVING` (:1060-1079).
- No razoring in the interior (the word does not occur in `search_v2.cpp`; root razoring is the root's,
  `ROOT_SEARCH_MODEL.md`). ProbCut and singular exist, both off (`search_engine.h:2899`, :2909).

---

## 2. Improving: exists, off

- Knobs: `ENABLE_IMPROVING false`, `IMPROVING_EVAL_WINDOW 6`, `IMPROVING_CHEAP true`, `IMPROVING_REDUCTION 1`,
  `IMPROVING_DELTA_MARGIN 0` (`search_engine.h:1354-1366`); env read `search_engine.cpp:3196`; no preset sets them.
- Signal: `g_evalStack[cur_depth]`, filled at node entry only under the flag, only within 6 plies of the leaf, and
  `NO_STATIC_EVAL` in check (`search_v2.cpp:394-397`). With `IMPROVING_CHEAP` the value is `cheap_eval` (material +
  PST), not the eval the node's pruning uses (`static_eval_for_improving`, `search_engine.cpp:1727-1767`).
- Definition (`search_v2.cpp:1070-1079`): root-frame values, so
  `RootSide ? ev[d] > ev[d-2] - MARGIN : ev[d] < ev[d-2] + MARGIN`; **default true** when either value is missing
  (in check, outside the window, `cur_depth < 2`). SF19's default in the same cases is false; SF11's is true.
- **One consumer**: not improving → `reduced_depth -= IMPROVING_REDUCTION` in LMR. Reverse futility, LMP, the null
  move and ProbCut do not read it.
- Record (memory `improving-heuristic-shelved`): shelved 06-11 on a sign bug; fixed and re-tested 06-13
  (+0.095 ply, −15 ± 26 Elo over 577 games); cheap-eval revival 06-21, tuned to `REDUCTION 2 / DELTA_MARGIN 150`
  (+125 STS), then **−7 ± 20 over 1,658 games** (06-22). All of it v1 era (`EVAL_ARM=0`), LMR-only, before the
  diagnostic-contamination fix of 08-14 for the STS numbers. Owner 10-30: a good option for the search arc once a
  node-entry eval is affordable. By the rule in `JUDGING-SEARCH-IDEAS.md` this is an unresolved null on a different
  engine, not a refutation of improving as the references use it (five consumers, on the real eval).
- ⚠️ Under v2 the cheap path is v1's `cheap_eval` (memory `the-fifteen-nulls-headline-mixed-three-classes`): the
  trend would be read on v1's material and PST values while the node prunes on v2's eval.

---

## 3. Correction history: state

Two implementations, both off, both kept.

| | `ENABLE_CORR_HIST` (phase 1, v1 era) | `CORR2` (consensus form, 10-27) |
|---|---|---|
| tables | `pawnCorrHist[maxbit][pawn key]`, an integer EMA of `best - raw eval`, `CORR_SHIFT 6`, `CORR_MAX 2000`, applied `× 192 / 256` (`search_engine.cpp:500-519`, `search_engine.h:3005-3008`) | three `int16` tables `[real side to move][16384]`: pawn key, white non-pawn key, black non-pawn key; limit 1024 (`search_engine.cpp:521-591`) |
| read | added to `rfp_eval`; with `ENABLE_CORRHIST_QSEARCH` also stand-pat and horizon | `(CORR2_WP 350 × pawn + CORR2_WN 400 × (white + black)) / 1024` millipawns, to `rfp_eval`, and with `CORR2_QS 1` to stand-pat and horizon (`search_engine.h:3010-3022`) |
| update | at node exit | at node exit: residual against the **corrected** eval, `bonus = clamp(resid × max(1, rd) / CORR2_BDIV 38, ±256)`, gravity; a bound guard refuses a fail high that does not exceed the eval and a fail low that is not below it (`corr2_update`, :574-591) |

**Why parked** (`dev_notes/current/EVAL-V2-PARKED-REGISTER.md`, search-phase table, row "Corrhist"; OPTIMIZATION_LOG
10-27 and 11-10):
- 10+0.1 against gen 6.2: full weights −8.7 ± 18.4, half −2.4 ± 19.6, quarter +7.6 ± 20.0, re-run +3.5 ± 19.2;
  pooled ≈ +5.5 ± 14 over 2,000 games.
- Re-sweep on gen 7.1 (#189): quarter weights −0.0 ± 13.5 (2,000 g, nps −2.9%); with `RFP_MARGIN 500` +6.9 ± 14.5
  (1,750 g, nps −2.2%). SF18 at equal nodes +3.8 / +1.5 pp, Mediocre +3.5 / +4.5 pp. Not harmful anywhere, +2σ nowhere.
- The register's own diagnosis: "It has only two consumers here (reverse futility, the qsearch stand-pat)".
- Earlier v1 record: the 08-24 offline gate found the residual predictable but equally by every key (memory
  `corrhist-signal-is-position-local-not-structural`); the 07-30 qsearch extension was live and harmful with the
  q-cache masking it (memory `corrhist-qsearch-harmful-and-qcache-masks-eval`). Both at `EVAL_ARM=0`.

**Trigger** (same row): search v2, once its node-entry static eval feeds the standard consumers (null-move gate,
improving, futility, LMR, ProbCut): test `CORR2` on vs off behind its switch on that search, from the quarter weights
with a weight sweep, and with that search's margins tuned with it on. Also in the final retune. S4 is that trigger.

**Lazy eval is ready for it**: every lazy site reads the correction first and shifts its bound by it
(`search_v2.cpp:413-416`, :1265-1268; LAZY-EVAL.md W4, fixed 11-09).

### 3.1 What differs from SF19's correction history (giants doc §1.4)

| # | ours (`CORR2`) | SF19 | consequence |
|---|---|---|---|
| C1 | **The update is reached only by nodes that did not cut.** The beta cutoff returns from inside the move loop (`search_v2.cpp:757-803`); the correction update sits after the loop (:827-856). | Updated at the end of every node, fail highs included (`search.cpp:1629-1638`); a fail high with a quiet best move is the "eval too low" teacher. | In side-to-move terms our tables are taught "eval too high" by every fail-low node, and "eval too low" only by the few nodes that finish inside the window. INFERRED consequence, not measured: the learned correction leans negative for the side to move, which would act like a slightly wider reverse-futility and stand-pat margin on both sides, and would fit the result that the smallest weights tested best. The bound guard in `corr2_update` is written for both directions, so the asymmetry is in where the call sits, not in the rule. |
| C2 | At a fail-low node `best_move` is the highest-scoring move (fail-soft), and the update is skipped when that move is a capture (:827-829). | No best move at a fail low, so every fail low not in check updates. | Fewer "eval too high" lessons than SF19 takes, from an arbitrary subset. Small. |
| C3 | Three tables, 16,384 slots each, per thread. | Five sources (pawn, minor, non-pawn ×2, continuation ×2), 65,536-class shared array. | No continuation term: SF19's only move-keyed (not position-keyed) source is absent. The 08-24 finding (every position key predicts the same) says the missing *kind* is the more interesting one. |
| C4 | Weights 350 / 400 / 400 mp at saturation (≈ 1.15 pawns max), fail high and fail low weigh the same. | ≈ 541 units max of which pawn ≈ 120; fail lows weigh 18 against 12. | Scale differs; weights were swept only by a common factor (full / half / quarter). |
| C5 | Two consumers of the value, none of its size. | The value in every consumer of `ss->staticEval`; its size in reverse futility, LMR and singular. | The register's diagnosis. |
| C6 | The correction is not stored anywhere stale: read fresh at each site. ✅ The q-cache masking of 07-30 is avoided for `rfp_eval`; the stand-pat value returned on a cutoff is the corrected one (:1272-1277) and is cached by the q-cache when that is on (INFERRED from the 07-30 record; the q-cache store was not re-read). | TT stores the raw eval. | Re-check before a `CORR2_QS` verdict. |

C1 is a defect against the stated design (`search_engine.h:3014-3016`: "a fail-high only teaches 'eval too low'" can
almost never happen). It is cheap to count before anything else: a sign histogram of the bonus beside the existing
`CORR2CNT` line (`search_engine.cpp:4856-4857`).

---

## 4. Lazy eval: sites and gates

Full description: `dev_notes/current/LAZY-EVAL.md` §1-5. On at ship: `LAZY_V2 7`, `LAZY_V2_KEEP 1`, `LAZY_V2_GATE 3`
(`search_engine.cpp:2746`); header defaults 0 (`search_engine.h:3473-3523`).

| bit | site | request | where |
|---|---|---|---|
| 1 | qsearch stand-pat | `lazy_request(root_side, s * beta - corrections, LAZY_SITE_QS)`; not at PV windows (`LAZY_V2_GUARDS & 2`) | `search_v2.cpp:1262-1274` |
| 2 | reverse futility's node-entry eval | `lazy_request(RootSide, s * (beta + RFP_MARGIN * rd) - corrections, LAZY_SITE_RFP)`; armed exactly where reverse futility is tested | :409-422 |
| 4 | futility of a child | `lazy_request(!RootSide, s * (alpha - fut_margin - 1), LAZY_SITE_FUT)`; not at PV windows | :1039-1046 |

Checked by reading: the direction and bound of all three requests agree with the test each site then makes, for both
values of `RootSide`. `g_lazy_nocut` (a lazy return the site did not cut with) is counted at the two sites that can
tell (:421, :1273).

Properties S4 must keep in view:
- **A lazy value is a bound, never a value.** It is not cached, and the node that received it cuts at once. So a lazy
  return can never be the eval a stack slot keeps for the node's children: a node that goes on to search always holds
  a full eval (when it holds one at all).
- **The estimate needs the parent's positional part** (`g_lazy_ply`, recorded by every full eval on the path,
  `search_engine.cpp:1535-1572`). A parent that computed no eval (PV node, remaining depth ≥ 7, in check) leaves no
  record unless something else evaluated it, and its children fall to the cheap path with the wider margin
  (`LAZY_V2_M_CHEAP 3000` against `LAZY_V2_M 1500`). Parent known at depth 10: 82.5% of asks (LAZY-EVAL.md §2).
- `LAZY_V2_KEEP` saves and restores the node's record around the null search, which evaluates in the same history
  slot (`search_v2.cpp:440-441`, :488-489).
- Watch item W1 (the equal-node accuracy cost, level at +0.13 ± 2.49 pp on 11-10) is tied to today's fire rates.

---

## 5. Asymmetric or missing against SF19

Numbered for reference; "A" = architecture, "M" = missing consumer or shape.

| # | ours | SF19 (giants doc §1) |
|---|---|---|
| A1 | The node's eval exists only at non-PV nodes, not in check, remaining depth 1-6. | At every node out of check. |
| A2 | A local variable; children cannot read it. The eval cache is the only store. | `ss->staticEval`; raw value in the TT. |
| A3 | No TT value as a better eval. | `eval` = TT value when its bound allows. |
| A4 | **Futility of a move is decided on the child's eval, after the move is made** (make, child state, eval, unmake). | On the parent's eval, before the move. No make, no eval. |
| A5 | The null move is not gated on the eval. Its gates are ply from the root (≥ 3 / 4), iteration depth ≥ 5, "last move not a capture". Its reduction follows the iteration depth. | Gate `staticEval >= beta - 13 d - 47 improving + 365` at expected cut nodes; `R = 7 + d / 3 + (staticEval - beta) / 256`. |
| A6 | The two sides' null-move ply gates differ (`MAXI 4`, `MINI 3`); futility of a child is off at ply 1 for the non-root side (`RootSide \|\| cur_depth > 1`). Inherited from the min / max split. | One rule for both sides. |
| M1 | Reverse futility: linear 600 mp per ply, remaining depth ≤ 6, no improving, no opponent-worsening, no TT-move condition, returns the raw eval. | Depth < 19, margin per ply 25-85 units, minus 2.7 plies' worth when improving, only without a quiet TT move, returns a blend toward beta. |
| M2 | LMP without improving. | Threshold halved when not improving. |
| M3 | LMR: improving optional, off, on a surrogate eval; no `alpha - eval` term. | Base reduction × 1.38 when not improving; `r += 3 * clamp(alpha - eval, -64, 96)`. |
| M4 | No interior razoring. | `eval < alpha - 482 d²` → qsearch. |
| M5 | No hindsight depth step; no quiet-history lesson from the eval change. | Both, from `(ss-1)->staticEval`. |
| M6 | No \|correction\| signal. | Reverse futility, LMR, singular. |
| M7 | Stand-pat is fail-hard on the raw value; no TT value as stand-pat. | Blend toward beta; TT value when allowed. |
| M8 | Correction learning at non-cut nodes only (§3.1 C1). | Every node. |

Ours with no counterpart there, to keep (owner's rule): the child-eval futility (it sees what the move does, which a
parent-eval margin can only bound), lazy eval's three sites and its parent-positional estimate, the passer and pin
exemptions on pruning, the capture-chain signals.

---

## 6. Dependencies (never rip a part out alone)

### 6.1 What was tuned around today's design

| value | tuned when, against what | what it silently assumes |
|---|---|---|
| `RFP_MARGIN` 600 | gen 6.4 (10-28/29), jointly with threats in the eval: 600 alone −27.5, with threats mg 50 +73.0 (#115); plateau check 800 / 400 (#116). `search_engine.cpp:2688-2691` | The raw eval; no improving (so 600 is an average over improving and non-improving nodes); remaining depth ≤ 6; returns the raw eval; the null move behind it ungated; lazy eval not yet present (lazy's reverse-futility site was added later and fires at `beta + 600 rd + 1500`). |
| `RFP_MAX_DEPTH` 6, `RFP_MIN_DEPTH` 1 | v1 era | It is also what limits where a node eval exists (A1). Widening the range and "eval at every node" are the same change at non-PV nodes. |
| `FUTILITY_MARGINS` 450 / 650 / 950 | v1 era, child-eval form | Margins for an eval taken **after** the move: they need not cover the move's own gain. A parent-eval futility needs a larger, differently shaped margin (it must cover what a quiet move can add). Not transferable. |
| `NULLMOVE_CURDEPTH_MAXI 4` / `MINI 3`, `depth_limit >= 5`, `NULLMOVE_EXTRA 2`, "not after a capture" | v1 era; the per-side split is the min / max legacy | They stand in for an eval gate: they limit how often a hopeless null move is tried. With an eval gate they may become redundant or over-restrictive (a null move at plies 1-2 is what SF does). `ROOT_SEARCH_MODEL.md` and S3's `PLY1_NODE_PRUNE` already touch ply 1. |
| `LMP_BASE 1`, `LMP_SCALE 1`, `LMP_MAX_DEPTH 5` | 07-02, with capture-gains conditions | One threshold for improving and non-improving nodes; `1 + rd²` is already far tighter than SF19's non-improving `(3 + d²) / 2` at small depth only by the base. |
| `LAZY_V2_M 1500`, `M_CHEAP 3000`, the gate and guards | 11-07 to 11-09, on today's ask mix and "parent known" rate | §4. Any change to which nodes hold a full eval changes the parent-known rate and so the fire and miss rates. |
| `CORR2_WP / WN` quarter weights | 10-27, 11-10, with two consumers and one-sided learning | §3.1. |
| the eval itself (gen 7 tables) | fitted at d10 play depth with today's pruning | memory `retune-depth-follows-play-depth`. |

### 6.2 What a stack-stored eval with improving changes, consumer by consumer

"Stack eval" below = a full static eval at every node out of check, kept per ply, with nothing else changed.

| consumer | with the stack eval alone | with improving wired in | retune with |
|---|---|---|---|
| reverse futility | unchanged where it fires today; available at remaining depth > 6 if the range is widened | a per-node margin instead of an average; the SF shape removes margin at improving nodes, which **tightens** an already tight 600 | `RFP_MARGIN`, `RFP_MAX_DEPTH`, the return blend; `CORR2` weights if on |
| null move | the eval gate becomes free (today it would cost an eval at every node that lacks one) | gate slope by improving | the ply gates, `NULLMOVE_EXTRA`, the reduction's shape; ply-1 pruning from S3 |
| futility of a child | the child's eval, once computed here, is the child's own node-entry eval if the child is searched (today that second read is an eval-cache hit); the parent's eval is available as a first, cheaper test | - | `FUTILITY_MARGINS`; a parent-form margin is a new parameter family |
| LMP | - | threshold by improving | `LMP_BASE / SCALE` |
| LMR | - | today's `ENABLE_IMPROVING` step, on the real eval instead of the surrogate | `IMPROVING_REDUCTION / DELTA_MARGIN`; the history delta |
| lazy eval | more parents with a positional record → more asks on the narrow margin → more fires; the reverse-futility site keeps its place; at nodes where no bound is being tested the eval must be full | - | `LAZY_V2_M`, `M_CHEAP`; W1 re-run (already S4's gate) |
| correction history | one read per node instead of one per site; every consumer inherits it | the correction largely cancels in an eval difference two plies apart when the keys do not change (memory `corrhist-qsearch-harmful-and-qcache-masks-eval`, unverified argument) | weights, with the margins |
| node count / bench | `get_board_evaluation` takes the node counter (`num_iterations`) and, per memory `improving-heuristic-shelved`, increments it; extra evals would then move the bench node count without changing a move. Not re-verified here. | | the fingerprint register |
| time | the cost side. Two things reduce it: children of late moves at remaining depth 2-4 are already evaluated by the futility site, and most interior nodes with remaining depth ≤ 6 already evaluate. The new evals are at PV nodes and at remaining depth ≥ 7, which are a small share of nodes (INFERRED from the tree's shape; count it, option 0). | | - |

### 6.3 Groups that move together

1. **Reverse-futility group**: `RFP_MARGIN`, `RFP_MAX_DEPTH`, improving in the margin, the return blend, `CORR2`
   weights, lazy's reverse-futility scale (`LAZY_V2_S_RFP`). The 11-10 re-sweep already showed `CORR2` and
   `RFP_MARGIN` interact (500 with the correction read better than 600 with it).
2. **Null-move group**: the eval gate, the ply gates (3 / 4), `depth_limit >= 5`, the reduction, the capture
   condition, and S3's ply-1 null move.
3. **Move-pruning group**: LMP's threshold, futility's form and margins, history and SEE pruning (S5), all sharing the
   "late quiet move at small remaining depth" population. Memory `search-changes-are-antagonistic-not-additive`
   applies: 2×2 before bundling.
4. **Lazy group**: any change in which nodes are fully evaluated; W1.
5. **Eval scale**: every margin is in absolute millipawns of today's eval; the after-search retune changes them all.

---

## 7. Options for S4, smallest and reversible first

Each is a knob beside today's design, default off and byte-identical, and nothing is removed. Names are proposals.
"Screen" means a fixed-depth or counter read under the Cython path; verdicts are games (`JUDGING-SEARCH-IDEAS.md`).

### Option 0. Counters first (no behaviour change)

One diagnostic build or guarded counters, read on the bench set and on game-like positions:

| counter | answers |
|---|---|
| interior nodes by (PV / non-PV) × remaining depth × in check, and how many hold an eval today; eval-cache hit rate of the node-entry eval | the true cost of A1 (how many new full evals "eval at every node" adds) |
| null moves tried, split by `s * eval >= beta` or not, and the fail-high rate of each class (needs the eval at those nodes: count under option 1) | what an eval gate would save and what it would lose |
| reverse-futility tests and fires split by would-be improving (real eval, two plies back) | whether improving separates safe cuts from unsafe ones before any margin is touched; pair with the prune-verification log (`ENABLE_PRUNE_LOG`) for the error rate per class |
| child-futility prunes: how many would a parent-eval test at margin X have predicted before the make | the size of option 5 |
| `CORR2` bonus sign histogram and the share of updates by node type | C1, directly |

Depends on: nothing. Cheap first test: it is the test.

### Option 1. The stack eval, no consumer changed (`NODE_EVAL_STACK`)

Compute the full eval at every node out of check, store it per ply (the `Stack` struct of the plan's §2, or
`g_evalStack` under a new mode beside `IMPROVING_CHEAP`), feed reverse futility from it. Nothing else reads it yet.

- Expected: the same moves at fixed depth with lazy eval off (the eval is a pure function of the position); with lazy
  on, the parent-known rate rises, so not identical. The bench node count may move through the node counter (§6.2).
- Screen: moves identical on the bench with `LAZY_V2=0`; time to depth and nps against today; the lazy fire / miss
  counters with it on.
- Depends on: option 0's first counter (if the added evals are a few percent, proceed; if not, see the two cheaper
  variants: store only what is computed anyway and treat a missing parent as "unknown", or fill the gaps with the
  eval's cheap part plus the parent's positional part, which the lazy machinery already produces).
- In-check rule to choose and record: SF19 copies the eval of two plies back (improving false); SF11 leaves it empty
  and defaults true; ours today defaults true.

### Option 2. Improving on the real eval, one consumer per bit (`IMPROVING_SRC`, `IMPROVING_USE`)

`IMPROVING_SRC`: 0 = today's cheap surrogate, 1 = the stack eval. `IMPROVING_USE` bits: 1 = LMR (today's step),
2 = reverse-futility margin, 4 = LMP threshold, 8 = null-move gate slope (needs option 3). Opponent-worsening as a
later bit.

- Screen: option 0's split tells which consumer has a signal to use. Then fixed-depth: nodes to depth and move
  agreement with the base per bit.
- Depends on: option 1. Each bit needs its group's margin re-swept (§6.3): bit 2 with `RFP_MARGIN` (start by keeping
  600 for non-improving nodes and widening, not narrowing, the other class, so the first test cannot be a pure
  loosening), bit 4 with `LMP_BASE`.
- The v1 null (−7 ± 20, LMR only, surrogate eval) is bit 1 on source 0; it stays registered as it is.

### Option 3. The null-move eval gate with a margin (`ENABLE_NULL_EVAL_GATE` + `NULL_GATE_BASE`, `NULL_GATE_SLOPE`)

Today's knob is `eval >= beta` flat. Add the references' shape, `eval >= beta + BASE - SLOPE * rd`, default
reproducing today's flat form; keep the ply gates as they are.

- Earlier read of the flat gate: `node_ab` +11.3 ± 40 (memory `statscore-lmr-shipped`), v1 era, unresolved.
- Screen: option 0's null-move split gives the saved searches and the lost cutoffs before any game.
- Depends on: option 1 (otherwise the gate adds an eval at every node without one). Then, as a 2×2 with the gate, the
  ply gates relaxed (`NULLMOVE_CURDEPTH_*`), since they were standing in for it.

### Option 4. Correction history on the stack eval (`CORR2` as it is, plus `CORR2_LEARN_CUT`)

- `CORR2_LEARN_CUT` (default 0): also call the update at the beta cutoff, quiet cutoff move only (C1). This is a
  correctness item against the knob's own description; it can be tested on today's search before S4, by counters
  (sign balance, mean |correction|) and then the same games as #189.
- With option 1, apply the correction once where the stack eval is set, so every consumer added later inherits it
  (the register's trigger). Keep the q-cache question (C6) in the test plan.
- Later, each behind its own knob: a continuation table (C3); |correction| as an uncertainty signal in reverse
  futility and LMR (S5's list); a material-band salt on the pawn key (`CORR2_BAND`, Triumviratus's CorrPhase, giants
  doc §4.1). The band salt has a specific reason here: the remaining eval gap is endgame-structural (memory
  `the-remaining-eval-gap-is-endgame-structural`), and CorrPhase's measured gain came from positions of 9 pieces or
  fewer. Its risk is the one Triumviratus hit: finer keys starve slots, and our tables are 16,384 slots.
- Depends on: C1's counter first. Weights re-swept with the reverse-futility group.

### Option 5. A parent-eval futility test before the make (`FUTILITY_PARENT`)

Before `make_move`, for the same move population as today's child futility: `parent eval + margin(rd) <= alpha` →
skip, with its own margin table. Today's child-eval test stays behind it for the moves that pass.

- Why: A4. It removes a make, a state copy, an eval and an unmake per pruned move (S3 step 0 found 56-63% of the
  moves made are pruned straight away; `SV2_PRUNE_EARLY` already moved LMP before the make).
- Screen: option 0's fourth counter (how many of today's child-futility prunes a parent test predicts at each
  margin, and how many it would add that the child test does not make).
- Depends on: option 1 (the parent's eval at remaining depth 2-4 exists today only at non-PV nodes). Lazy eval's
  futility site sees fewer asks, so W1 again.

### Option 6. Reverse futility's remaining shape knobs

Already present: `RFP_RETURN_BLEND` (0), `RFP_MAX_DEPTH`. Possible beside them: a quadratic or capped margin, no cut
when the cached best move is quiet (needs S3's TT move). Only after options 1-2, as part of the reverse-futility
group's sweep.

### Option 7. Eval in the TT and the TT value as a better eval

A field in the TT entry for the raw eval, and `eval = tt value` when its bound allows. Larger: it changes the entry
layout (S2 / S3 territory) and the eval cache's role. Deferred until the staged move source settles the entry.

### Not in S4 (S5's list, noted so they are not lost)

Hindsight depth steps, the eval-change lesson for quiet history, opponent-worsening, the `alpha - eval` LMR term,
interior razoring, the stand-pat blend, continuous improvement (Reckless) in place of the flag.

### Order suggested by dependency only

0 → (4's `CORR2_LEARN_CUT` in parallel, it needs nothing) → 1 → 2 and 3 as separate arms, then their 2×2 → 4 on top →
5. Each arm complete and tuned in its group before any comparison between arms (`JUDGING-SEARCH-IDEAS.md`).

---

## 8. Defects and oddities seen

| # | what | where | weight |
|---|---|---|---|
| D1 | Correction history learns only at nodes that do not cut (§3.1 C1). Off at ship, so no effect on play today; it bears on every `CORR2` game result on record. | `search_v2.cpp:757-803` against :827-856; the older pair the same way (`search_engine.cpp:7575-7601` sits after its loop; INFERRED for its cutoff path, not re-read) | medium, for the corrhist verdicts |
| D2 | A fail-low node skips the correction update when its highest-scoring move is a capture (C2). | `search_v2.cpp:827-832` | small |
| D3 | `ENABLE_IMPROVING` with `IMPROVING_CHEAP` (the default pairing) reads v1's `cheap_eval` inside a v2 search. Off at ship. | `search_engine.cpp:1733-1739` | small; matters the day improving is re-tested |
| D4 | `FUTILITY_MARGINS[0]` (200) is unreachable from the interior's futility site (`rd > 1`). Not a bug; a dead table entry to know about before tuning it. | `search_v2.cpp:1037-1040` | note |
| D5 | Improving's default when a value is missing is true, and values are missing in check and beyond 6 plies from the leaf, so under the flag the step is silently off over most of a deep tree. | `search_v2.cpp:394-397`, :1073-1076 | note; bears on reading the v1 nulls |

No defect was found in the shipped path's node-entry eval, reverse futility, or the three lazy requests.

## 9. Not verified

- Nothing was run: no counter in §7 has a number yet, and every "INFERRED" above is a reading.
- Whether `get_board_evaluation` increments the node counter today (taken from memory `improving-heuristic-shelved`).
- The q-cache's store of a corrected stand-pat value under `CORR2_QS` (C6).
- The older maximizer / minimizer pair beyond the lines cited; `PLY1_NODE_PRUNE`'s ply-1 eval
  (`search_engine.cpp:6687-6706`) was seen in a listing only.
- OPTIMIZATION_LOG entries were taken from the parked register's summary, not re-read.

## 10. Open questions for the owner

1. **Eval at every node, or only where it is computed anyway?** Option 1 in full costs evals at PV nodes and at
   remaining depth ≥ 7; the partial form costs nothing and leaves improving undefined there. Decide after option 0's
   count, or state a cost ceiling now (for example "up to N% time to depth")?
2. **In-check rule for improving**: SF19's (copy two plies back, improving false), SF11's (default true), or ours
   today (default true)? It decides how hard nodes after a check are pruned.
3. **`CORR2_LEARN_CUT` (D1)**: treat as a correctness fix to switch on inside `CORR2` before its S4 test, or as a
   separate arm so the two on-record results (10-27, 11-10) stay comparable?
4. **Child-eval futility** is ours and distinctive. Is a parent-eval pre-test in front of it (option 5) acceptable as
   "beside", given that it changes which moves reach the child test?
5. **The null-move ply gates (3 / 4, per side)**: keep the per-side split as an owner design, or allow the 2×2 of
   option 3 to test one common gate?
6. **Improving's first consumer**: start with the bit the counters favour, or with LMR on the real eval to close the
   v1 null on equal terms first?
7. **Material-band salt for `CORR2`** (CorrPhase): worth a slot in S4's corrhist test, or parked to the final retune
   with a trigger?
8. **W1 budget**: every option from 1 on moves lazy eval's fire rates. Re-run W1 once per option, or once at the end
   of S4 as the plan's gate says?
