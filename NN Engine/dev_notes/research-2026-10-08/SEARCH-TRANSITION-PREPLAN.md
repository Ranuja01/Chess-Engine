# SEARCH TRANSITION — RESEARCH PRE-PLAN (2026-10-08/09)

**Status:** input for discussion with the owner, NOT a decision. Read-only research; nothing was built, run or
edited. Every claim is tagged **[E]** = EVIDENCE (file:line, doc, memory, URL) or **[S]** = SPECULATION / my
inference. Line numbers are from the tree as of 2026-10-09; the overnight game gate was running, so no fingerprint
was re-verified here.

**Framing (owner, 10-08, memory `long-term-goal-3000-single-core-hce-via-search-v2`):** goal ~3000 CCRL single-core
HCE; path = "search v2" run with v2's method (reference audit → instruments first → one feature at a time → fair
gates; SPSA for search params); the search becomes NEGAMAX; uniqueness lives in play style, never code style.
**First rule:** re-audit every CLOSED search item against the PREMISE it was closed under — all ~35 arms + 52
configs were swept at `EVAL_ARM=0` (v1: 3.7× costlier per call, colour-asymmetric, noisier).

---

## 0. The facts the plan stands on

| fact | source |
|---|---|
| v2 eval = **11.5%** of node cost (v1 32.8%); movegen **41.1%**; `MG_ISSAFE` 2.08× `MG_PSEUDO` ⇒ pin-aware legal movegen is the indicated fix | [E] memory `the-nps-gap-is-mostly-not-eval` |
| NPS ~450k (LIGHTNING mean) vs SF11 2.53M (5.1×); SF15-classical 1.33M | [E] `speed-and-qsearch-findings-2026-07-24.md:4-8`, INSTRUMENT-MAP §D |
| At EQUAL NODES (249,014) SF11 reaches **d15** where we reach **d10**; nodes-to-d10 26,265 vs 249,014 (**9.48×**); marginal EBF 1.452 vs **1.914** | [E] memory `the-sf11-gap-is-two-thirds-node-efficiency` |
| STS300 @ equal nodes: v2 **1838** · v1 1752 · SF11 2374 · SF15.1c 2492 | [E] REFERENCE-BENCH-LADDER:356-357 |
| v2 ≈ 2265-2340 CCRL (Mediocre anchor 45%); v1 was ≈15% on the same anchor | [E] REFERENCE-BENCH-LADDER:377-382 |
| Pawn endings are a SEARCH gap: static bias ours −22 / SF11 −16 (we are **better** on all 26 own-play rows, 0.87×), but d10 search bias ours **−8.5** vs SF11 **−1.0** (ours d14 −5.8), 313 depth rows | [E] REFERENCE-BENCH-LADDER:366-373 |
| Threats: real d10 −7.8%, self-play @50k **+32 ± 9**, SF18 @1000 **−5 ± 9** (2,000 paired games, 4 seeds). Hypotheses: node budget · symmetric threats mislead stand-pat/pruning · self-play exploitation | [E] TEXEL-C3 §20a lines 1049-1057; SESSION-HANDOFF-2026-10-07 "ORDER RESHAPED" |
| Search changes are ANTI-additive (two 2×2s; 10 margin combos; IIR×guard; IIR×R=2) | [E] memories `search-changes-are-antagonistic-not-additive`, `margin-sweep-52-configs…`, `iir-is-the-venue-correct-node-saver` |
| A truer eval buys pruning headroom: RFP 1500→400 costs ours −207 STS, SF11 **+78**; futility same shape; root razor no differential (keys on presearch scores, not eval) | [E] memory `a-truer-eval-buys-pruning-headroom-the-crank-result` (measured on **v1**) |
| Speed/node bar: ±0.5 ply is Elo-neutral ⇒ a node saving or NPS gain must be ≈35% to be measurable in benches; the only search win that shipped INCREASED nodes (`LMR_SHAPE` +20.7 Elo, +8% nodes) | [E] memories `half-a-ply-is-elo-neutral-the-speed-bar`, `node-savings-below-35-percent…` |
| Printed EBF is `pow(cumulative nodes incl. presearch/qsearch/TT-hits, 1/exit_depth)` — not comparable to anything | [E] memory `ebf-metric-is-not-comparable` |
| Harness null is +4.6 Elo, not 0; SPRT estimates inflate at the bound they stop on; read pooled seeds only | [E] MEMORY.md §📏 |

---

## 1. PREMISE-AUDIT TABLE — every search feature that is OFF or was "closed"

Legend for "premise still holds?": **CHANGED** = the closing premise was v1-specific (eval cost/noise/asymmetry) or
the mechanism was dead when measured ⇒ reopen · **HOLDS** = closed on a mechanism or on a fair games test that
v2 does not disturb · **PARTIAL** = re-test only in a specific form. Priority: **H/M/L/—**.

Current defaults from `search_engine.h` unless stated. Our search is `minimizer()`/`maximizer()`/`pre_minimizer()`
/`qSearch(is_maximizing)` with an absolute Black-positive eval flipped once by `Config::side_to_play`
(`search_engine.cpp:9398-9399`) [E].

| # | item (knob, default) | what it does (ours) | premise it was closed under (record) | premise under v2 | re-test priority |
|---|---|---|---|---|---|
| 1 | **IIR** `ENABLE_IIR=false`, `IIR_MIN_DEPTH=6` (h:2831) | depth−1 on a movegen-cache MISS at rem ≥ 6 (cpp:5757/6435) — our analog of `!ttMove` | Best candidate of the arc: quiet −16% nodes, +1 ply @1s, STS +22 (v1). Games **+7.7 ± 17.7 / 2,400** — unresolved; parked "needs 4,000 alone" (SESSION-HANDOFF-2026-09-05:16, -09-09-B:106). Anti-additive with `PROTECT_TT_DEPTH` and `R=2` | **CHANGED** (measured on v1; never resolved); SF16+/Ethereal/Weiss all have IIR, Weiss splits PV d≥3 / cut d≥8, Ethereal d≥7 PV-or-cut with `ttDepth+4<depth` [E WebFetch] | **H** — first games arm on v2; run ALONE, 4,000 UHO, both instruments |
| 2 | **Correction history** `ENABLE_CORR_HIST=false` (h:2862), `CORR_SHIFT=6, CORR_MAX=2000, CORR_W=192/256` (h:2918-2921) | ONE pawn-key table × `maxbit`, integer EMA of `(best − raw static)`, applied only to `rfp_static_eval` (cpp:5643/6328) and optionally qsearch (`ENABLE_CORRHIST_QSEARCH`, cpp:7770/7876); update at node exit when best move is a non-capture, no bound guards, no depth weight (cpp:360-373, 6162-6183) | Closed 08-24 on 615k records: residual is POSITION-LOCAL; all 5 key families identical (+8.9…9.1%); keyed table worse than one global slot (sample starvation, median 5 updates/slot) (`SEARCH-SWEEP-2026-08-25.md` §1; memory `corrhist-signal-is-position-local…`). Earlier: qsearch extension −49 STS (v1) | **CHANGED in part**: the residual being analysed was **v1's** (3.7× costlier, asymmetric, over-reading capgains) — v2's residual has never been logged. The starvation finding is structural to the keying and likely survives. Our form is ALSO not the giants' form (see §3.2) | **M-H** — re-run the Phase-0 `corrlog`/`corrsignal` gate on v2 FIRST (cheap, no games); build the multi-table form only if a key family separates from GLOBAL |
| 3 | **Singular extension** `ENABLE_SINGULAR=false`, `SINGULAR_MARGIN=2` (mp!), `MIN_DEPTH=6` (h:2822-2825) | SF11-form exclusion re-search at rem/2 on the TT move; needs `ENABLE_NODE_TT` to be fed | 07-09 gauntlet neutral (sign-flipped with seed) — but 08-25 found `ENABLE_NODE_TT=0` **starved it** ("every prior singular measurement was of a dead mechanism"); once fed: WAC +9 at fixed depth, **−1 at equal nodes**, STS −56, +11.2% nodes ⇒ "WAC mirage" (SEARCH-SWEEP §2). **Never games-tested while live** | **CHANGED** (dead when gauntleted; v1 when benched). SF11 credits singular ~70 Elo (:1036); Ethereal/Weiss add double extensions + negative extension + multicut | **H** (in the negamax rewrite, with NODE_TT; margin in SF units ≈ 2 cp/ply ⇒ ours ~20 mp × depth, not 2) |
| 4 | **ProbCut** `ENABLE_PROBCUT=false`, `MARGIN=2200`, `MIN_DEPTH=5`, `CANDIDATES=3` (h:2812-2816; cpp:5776-5814) | non-PV, rem ≥ 5, ≤3 SEE≥0 captures/promos, child at rem−4, no qsearch pre-verify | Closed early ("inherits phantom scores") under v1 before Kaufman/CAPG_PIN/de-king; memory `ordering-and-reduction-idea-backlog` flags the closure as suspect and the margin ~1.5× too wide (SF11 `beta+189−45·improving` ≈ 1.5 pawns; Ethereal `beta+100`; Weiss `beta+200`) | **CHANGED** (self-verifying by search, so eval-noise premise never applied; no fire counter was ever read) | **M** — fire counter → margin sweep 1200-2200 → quiet-corpus nodes + STS@eq-nodes → games. Add SF's qsearch pre-verification step |
| 5 | **Improving** `ENABLE_IMPROVING=false`, `IMPROVING_CHEAP=true` (h:1297-1309) | +1 LMR ply when stm's static eval not rising vs 2 plies back; cheap material+PST surrogate | Shelved 3× (06-11 sign bug; 06-13 deeper-not-stronger; 06-21/22 cheap R2/M150 **−7 ± 20 / 1,658 games**) — all v1, full-eval variant cost too much per node (memory `improving-heuristic-shelved`) | **CHANGED**: v2 eval is 3.7× cheaper per call and we already compute a node-entry eval for RFP at non-PV rem 1-6 (cpp:5633-5644). In SF11 `improving` feeds futility margin, LMP count, reduction, null-move gate, ProbCut margin (:828-835, :852, :895, :1002, :1008); Ethereal/Weiss likewise | **H** — but as the giants use it (a modulator of LMP/futility/RFP/LMR), tuned jointly by SPSA, never as a lone +1 ply |
| 6 | **Static move ordering** `ENABLE_STATIC_ORDER=false`, `HIST_MAX=0` (h:2993-3007) | placement-table delta for quiets with \|history\| ≤ HIST_MAX | Bench +9 WAC/+34 STS; **games +0.4 ± 20 / 1,600** (SEARCH-SWEEP §4); antagonistic with `LMP_BASE=0` and with `NODE_TT` | **PARTIAL**: measured with v1's PST; v2's Texel-fitted tapered PST is a better static signal. The anti-additivity with index-keyed pruning is a mechanism and stands | **M** — screen FMC% and quiet nodes on v2; games only if the screen moves >bench noise; never bundled with LMP changes |
| 7 | **Cheap stand-pat** `QSTANDPAT_EVAL_MODE=0` (h:2778) | 1 = material+PST, 2 = full minus heavy dynamic terms at qsearch stand-pat | Closed 06-27: mode 1 −235 STS, mode 2 −258 STS at equal time — stand-pat IS the leaf eval (memory `lighteval-standpat-is-leaf`); the 07-24 −13 WAC was the unsound node-level delta prune (now bypassed by `ENABLE_QDELTA_PERMOVE=true`, h:2786) | **HOLDS on accuracy; the COST side shrank**: with v2 at 11.5% the whole eval is worth 1.13× NPS, so there is almost nothing to buy. Dead by arithmetic | **L** — do not re-run; record the arithmetic |
| 8 | **Cheap futility eval** `FUTILITY_EVAL_MODE=0` (h:2777) | our futility evaluates the CHILD after the move (cpp:4591-4610), margins {200,450,650,950} mp for rem 1-4 | 08-25: mode 1/2 = 241/236 WAC, STS 1575 — closed; `FUTILITY_MARGIN_SCALE` 100 optimal, non-monotonic (SEARCH-SWEEP §5) | **PARTIAL**: same cost arithmetic as #7 (little to gain); the SF form (PARENT eval + margin + history gate, :1017-1024) is a different mechanism that saves the child eval entirely and prunes before `make_move` | **L** as an eval-mode; **M** as "adopt the parent-eval form in negamax v2 and re-fit margins" |
| 9 | **RFP** `ENABLE_RFP=true`, `MARGIN=1500/ply`, rem 1-6, `RFP_RETURN_BLEND=0` (h:2791-2804) | returns raw static eval | Shipped +73 Elo (v1). Margin sweep: the ONLY monotone margin lever (1000 → −13% nodes, STS 1744) (memory `margin-sweep…`). `RFP_RETURN_BLEND` (SF16-18 pull-toward-bound form): **no record found** ⇒ never tested [E grep] | **CHANGED**: margin is pinned by eval noise; v2 is less noisy ⇒ re-crank (the `crank_sweep.sh` protocol) | **H** — part of the margin re-sweep; also test BLEND 50/67 |
| 10 | **Null-move** `ENABLE_NULLMOVE=true`, `NULLMOVE_EXTRA=2`, `NULL_EVAL_GATE=false`, `NULLMOVE_EVAL_R=false`, `NULLMOVE_PROGRESSIVE=false` (h:262-264, 2805, 2941, 3087-3088) | fires at cur_depth ≥ 3/4, `depth_limit ≥ 5`, not after capture/null/check; R = `DEPTH_REDUCTION[depth_limit] − 2 − (depth_limit≥10)` (ITERATION-keyed); material guard = `isUnsafeForNullMovePruning` (cpp:8509-8551): queenless & <7 pieces (ALL pieces incl. pawns) or with queen & <4 ⇒ no null | Eval gate: gauntlet neutral, sign-unstable (SESSION-HANDOFF-07-08:358-375); eval-scaled R: 07-08 campaign (v1); PROGRESSIVE byte-identical (dead knob) | **CHANGED** for the two eval-reading knobs (they read the static eval). SF11: `eval ≥ beta && staticEval ≥ beta − 32·depth + 292 − 30·improving` gate, `R = (854+68·depth)/258 + min((eval−beta)/192, 3)`, verification at depth ≥ 13, `non_pawn_material(us)` guard (:838-886). Our material guard is **type-blind**: K+7P passes it, K+Q+2 fails it [E] | **H** — adopt SF's form in v2 (eval gate + depth/eval-scaled R + npm guard); re-test gate+R on the current search as a pair |
| 11 | **LMP** `ENABLE_LMP=true`, `BASE=1`, `SCALE=1`, `MAX_DEPTH=5` (h:1317-1320; cpp:4535-4550) | `i ≥ 1 + rd²` for rem 1-5, quiets only (`do_lmr`) | `LMP_BASE=0` games +2.6 ± 40 (null); `LMP_HIST_EXEMPT` loses; SF form `(3+d²)/(2−improving)` has no `improving` here | **PARTIAL**: shape ported; the `improving` half missing (#5) | **M** — joint SPSA with #5 |
| 12 | **SEE pruning of quiets** `ENABLE_SEE_PRUNE=false` (h:1493-1495) | prune quiet whose moved piece is capturable (post-move SEE) at rem ≤ 3 | "parked, accuracy loss" −7.1% nodes, 240 WAC; `seep_d5/d7` cut 33-63% nodes for −315/−483 STS (margin sweep, v1). Note SEE is **per-square, not per-move** (`speed-and-qsearch…:73`) | **PARTIAL**: SF11 (~20 Elo) and Ethereal (~42 Elo incl. captures) key it on `lmrDepth` with a quadratic margin (`-(32−min(lmrDepth,18))·lmrDepth²`, :1027), not a flat margin at rem ≤ 3 | **M** — only after a per-move SEE; port the lmrDepth-keyed form |
| 13 | **SEE capture pruning** `SEE_PRUNE_CAPTURES=true`, margin 1000 (h:1501-1502) | skip captures with SEE < −1 pawn at rem ≤ 3 | Shipped in the +18 bundle (08-22) | HOLDS; SF11 `−194·depth` (:1030) scales with depth — ours flat | **L** — SPSA the margin/depth later |
| 14 | **History pruning** `ENABLE_HIST_PRUNE=false`, `COEF=512`, `MAX_DEPTH=4` (h:1331-1333) | skip late quiets with very negative butterfly history | No games record found; Weiss `histScore < −1024·depth` at lmrDepth<3; SF11 countermove-history pruning `contHist < CounterMovePruneThreshold` (:1010-1014, ~20 Elo) | **UNMEASURED** | **M** — in v2 with piece×to cont-hist |
| 15 | **Cont-hist 2-ply** `ENABLE_CONT_HIST_2PLY=false` (h:1541); `CONT_HIST_PIECE_KEY=0`, `CAPTURE_HIST_VICTIM=0` (h:1547-1548) | 2-ply continuation; piece×to re-keys | 2PLY loses at d10 (v1); re-keys **−0.6 ± 23.1 / 1,200 games** = flat but 114× smaller tables | **HOLDS as Elo**; adopt the re-keyed forms as the v2 default on engineering merit (memory `node-savings-below-35…`) | **L** (Elo) / **adopt** (engineering) |
| 16 | **Threat-conditioned history** `ENABLE_THREAT_HIST=false` (h:2936) | butterfly split by attacked from/to | MIRAGE (+5% nodes, STS 1593), v1 | **PARTIAL**: v2's threat detector exists; SF uses threatened-piece info in LMR/ordering since SF15 | **L-M** — after threats-in-search (§3.3) |
| 17 | **Check ordering bonus** `ENABLE_CHECK_ORDER=false` (h:1553) | flat quiet-check bonus 6000 | mirage profile (v1) | HOLDS (ordering is already 87.5% FMC) | **L** |
| 18 | **TT move ordering / node TT** `ENABLE_TT_MOVE=false`, `ENABLE_NODE_TT=false` (h:1559, 2977) | self-store at node exit with the best move | TT move −3 WAC vs its control; redundant with `promoteMoveToFront` (91.9% FMC); NODE_TT +1.43% nodes "for nothing" | **HOLDS as a lever; CHANGED as PLUMBING** — singular, IIR-SF-form, and `ttPv`/`ttCapture` reductions all need a node-entry probe (memory `tt-and-cache-architecture`: child-probe architecture) | **adopt** in negamax v2 (node-entry probe is the standard shape); not a solo arm |
| 19 | **Protect PV / TT-depth** `PROTECT_PV=false`, `PROTECT_TT_DEPTH=false` (h:333, 352) | never reduce at PV / at TT-covered depth | PV: +151% nodes at AUC 0.75 marker; TT-depth: destroys IIR's saving | HOLDS (mechanism) — SF reduces LESS at ttPv (`r −= 2`, :1137), never "protect" | **L**; re-express as `ttPv r−=k` in v2 |
| 20 | **Cutnode/allnode** (absent; probe only) | SF's `cutNode r += 2` (:1155) | AUC 0.5678 vs move_index 0.8151 for wrong LMR; closed for ~10 lines of probe (memory `cutnode-allnode…`) | HOLDS as LMR lever — ⚠️ the probe was PVS parity, not a threaded flag incl. null/ProbCut flips | **L** as a solo arm; **free** in negamax v2 (one bool parameter) — include and let SPSA size it |
| 21 | **Root LMR / root table / presearch-off** (`ENABLE_ROOT_LMR`, `ENABLE_ROOT_TABLE`, `ENABLE_ROOT_PRESEARCH`) | the giants' root vs our presearch + root razor | Presearch removal via root LMR **−25.0 ± 19.6 at equal nodes** (fair test, 09-03); presearch spends 38.5% of nodes for a 2:1 return; provenance-tiered sort reaches node parity presearch-off (memory `presearch-economics…`) | **HOLDS**: a fair games test exists. ⚠️ The untested bundle `ROOT_TABLE=1 ROOT_LMR=1 EXEMPT_BEST=1 PRESEARCH=0 RAZOR=0` remains untested | **L now**; **design question** for negamax v2 (§4.4) |
| 22 | **Root razoring** `ENABLE_ROOT_RAZOR=true` (h:1413) | abandon low root moves on presearch scores | wrong-razor 17.65% = iteration instability, margin not the lever (SEARCH-SWEEP §6b); crank shows no eval differential | HOLDS | — |
| 23 | **LMR_REMDEPTH** (h:311) | re-index the old table on remaining depth | closed on mechanism (clamp-saturated); superseded by `LMR_SHAPE=1` product keyed on remaining depth (h:3216-3218, shipped +20.7) | HOLDS | — |
| 24 | **LMR fine terms** (SF11 :1126-1191) | `ttPv −2`, `(ss-1)->moveCount>14 −1`, `singularLMR −2`, `ttCapture +1`, `cutNode +2`, escape-capture −2, statScore/16384, capture r+1 late | none built except statScore (`STATSCORE_DIVISOR=683`, shipped +23) and PROTECT_KILLERS (+12.4) | **UNMEASURED** | **M** — the natural SPSA set once negamax v2 has `cutNode`, `ttPv`, `ttCapture` |
| 25 | **Aspiration** `ASPIRATION_DELTA=500` (h:3240) | root window | DELTA=750 null (step-shaped equal-work artifact); fail rate 51% | HOLDS; SF's `averageScore` centering untested (memory `root-techniques…`) | **L** |
| 26 | **Verification re-search** `VERIFY_RESEARCH_REDUCTION=2` (h:3105) | null-move verification depth | plateau-checked both directions (+17 LIGHTNING) | HOLDS | — |
| 27 | **OTV** `ENABLE_OTV=false` (h:3118-3123) | re-search a child whose score overshoots the node's static eval | shelved −17.8 (07-12, v1); "structurally blocked" for the KS failure mode | **PARTIAL** (reads the static eval); no giant has it | **L** — v2's phantom rate is unmeasured; measure before any retry |
| 28 | **Passer prune exemption** `ENABLE_PASSER_PRUNE_EXEMPT=false`, `ADV=5` (h:1521-1522) | exempt advanced pawn pushes from LMP/LMR | FAILED gate: rank-proxy over-fires (825k fires, +12% nodes, STS −3.8%); fix = true-passed mask (`STRENGTH_BACKLOG.md:26`, OPTIMIZATION_LOG:1235) | **CHANGED by the data**: pawn endings −8.5 vs −1.0 at d10 is the first measured pawn-ending search gap; SF11's rule is `killer[0] && advanced_pawn_push && pawn_passed` (:1078-1081) — far narrower than ours | **H** — rebuild in SF11's narrow form (§3.1) |
| 29 | **Check extension** `CHECK_EXTENSION=3`, `SEE_EXTEND_MARGIN=300` (h:3152-3163) | extend checks, SEE-filtered, cap 3/path | validated (+43.9 bundle) | HOLDS; SF11 extends check only if discovered or SEE≥0 (:1073-1075) and has no path cap | — |
| 30 | **Stand-pat seed / fail-soft fixes** `ENABLE_QSTANDPAT_SEED=false`, `ENABLE_TT_FLAG_FIX=true`, `ENABLE_NULL_MATE_CLAMP=true` (h:276, 272, 267) | SF-form qsearch seeding; TT flag vs searched window; mate clamp | seed −0.13 depth; C1 (flag fix + clamp) +6 / 790g null — "productive bugs", search tuned around them (memory `search-value-bugs…`) | **CHANGED by the rewrite**: a negamax v2 is a fresh tuning, so the correct forms are free | **adopt** in v2 |
| 31 | **qsearch horizon before in-check test** (cpp:7766 `qDepth >= MAX_QDEPTH` returns a raw eval BEFORE the `currently_in_check` branch at :7789-7791) | mate-blind horizon return, cached | flagged 07-24 as defect #2 ("move the qdepth cap below the in-check test"); **still present** [E] | correctness | **adopt** in v2 (SF has no qsearch depth cap; it stops generating checks after the first q-ply) |
| 32 | **Draw randomisation / mate-distance pruning** | SF `value_draw = VALUE_DRAW ± 1 by nodes` (:91-93); Ethereal mate-distance pruning | we return a flat 0 (cpp:7745) and `REPETITION_THRESHOLD=2`; no mate-distance pruning found [E grep] | n/a | **L** — cheap to add in v2 |
| 33 | **Pruning gates** `bestValue > MATED_IN_MAX` and `non_pawn_material(us)` on ALL shallow pruning (SF11 :997-999) | ours: none | never considered | n/a | **H** (part of §3.1) |

**Closed-and-stays-closed (do not reopen without a new mechanism):** corrhist keying in its single-table form
(#2, the starvation half), presearch removal via index-keyed root LMR (#21), root-razor margins (#22), LMR_REMDEPTH
(#23), PROTECT_PV as "never reduce" (#19), three movegen micro-fixes and staged/lazy generation (memory
`movegen-is-36-percent…`: "don't hand-optimize against `-Ofast -flto`"), pawn-king eval cache (3% of total),
capgains lazy skip, aspiration DELTA (#25), cheap stand-pat (#7, by arithmetic).

---

## 2. THE TRANSITION PLAN — in order, with instruments and gates

Order follows the owner's 10-08 "ORDER RESHAPED" list (SESSION-HANDOFF-2026-10-07): SF11 pawn-ending rules → corr
hist → threats + its search interaction → cheap stand-pat / static-ordering re-tests → maybe NPS. §4 argues where
the negamax rewrite sits relative to these.

### 2.0 Instruments to have in hand BEFORE step 1 (no games, ~1 day)
- **Fingerprint re-verify** both arms (`254 / 50,622,239 / 4.029`; v1 `250 / 35,310,778 / 3.784`).
- **Fair node-efficiency instrument** (§5) — needed so every step below is judged on the same ruler.
- **Pawn-ending corpus split**: the 313 depth rows behind "−8.5 vs −1.0" (`_pe_search_bias.py`, REFERENCE-BENCH-LADDER:371)
  as TRAIN, plus a held-out K+P set (`kp_stress_*`, `selfplay/kp_fens.txt`, `kp_dense_fens.txt` exist [E git status])
  — the "understanding vs memorisation" discipline the owner set for eval.
- **Register predictions** for every arm before running (MEMORY.md "NEVER CALL A DIRECTION ON PARTIAL DATA").

### 2.1 SF11 pawn-ending rules → our code

**What SF11 does [E `stockfish-11-linux/src/search.cpp`]:**
1. `:846` null move requires `pos.non_pawn_material(us)` — no null move when the mover has only pawns.
2. `:997-999` Step 13 "pruning at shallow depth" (LMP, countermove pruning, parent futility, quiet SEE prune,
   capture SEE prune) is gated on `pos.non_pawn_material(us) && bestValue > VALUE_MATED_IN_MAX_PLY`.
3. `:1078-1081` passed-pawn push extension: `move == ss->killers[0] && advanced_pawn_push(move) && pawn_passed(us, to)` ⇒ +1.
4. `:1083-1086` last-capture extension: captured piece > pawn and `non_pawn_material() <= 2·RookValueMg` ⇒ +1.
5. `:1472-1475` qsearch futility exempts `advanced_pawn_push` moves (pawn to rank ≥ 6 relative).
6. Also `:1117-1124`: captures are reduced only if `moveCountPruning || staticEval + captured <= alpha || cutNode || ttHitAverage low`.

Ethereal: only the NMP guard `boardHasNonPawnMaterial(board, turn)` [E WebFetch]. Weiss: NMP `nonPawnCount[stm] > (depth > 8)`
and LMR `r += nonPawnCount[opponent] < 2` (reduce MORE in endgames) [E WebFetch]. ⇒ rule 1 is universal (3/3);
rules 2-5 are SF11-specific (invariant SF11→SF18? — verify in the local SF15/17 trees before porting, per
`adopt-reference-methods-only-if-universally-superior`).

**Mapping to our code [E]:**
| SF11 rule | our site | gap |
|---|---|---|
| 1. npm guard on null | `isUnsafeForNullMovePruning` cpp:8509-8551 counts `popcount(occupied_colour[turn]) − 1` — pawns included, type-blind | K+7P (pieceNum 7, queenless) → null move ALLOWED; K+Q+N+B (pieceNum 3) → DISALLOWED. Fix: `(occupied_colour[turn] & ~pawns & ~kings) == 0 ⇒ unsafe`; keep or drop the count rule — test both |
| 2. npm gate on shallow pruning | LMP cpp:4535-4550 / 4940; futility 4591-4610 / 4998; SEE-capture 4579-4585; hist/see prune blocks — none gated on material or on `best > mated` | add one `bool pawns_only = …` per node and AND it into `do_lmr`-side prunes (not LMR itself) |
| 3. passed-push extension | `ENABLE_PASSER_PRUNE_EXEMPT` (h:1521) is an EXEMPTION from LMP/LMR on a rank proxy, not an extension; over-fired | rebuild as SF11's: killer-slot-0 AND true-passed (`passed_span` mask, STRENGTH_BACKLOG:26 has the plan) AND rank ≥ 6 ⇒ `extend` (we already have the `extend` path for checks, `g_check_extensions` cap) |
| 4. last-capture ext in low material | none | small; include in the same arm or skip |
| 5. qsearch futility exempt advanced push | per-move futility cpp:7929-7955 exempts promotions/ep only | add `!advanced_pawn_push` to the exemption |

**Why the record says this should bite [E]:** we are statically BETTER than SF11 in pawn endings (0.87×) yet
d10-worse (−8.5 vs −1.0) and d14 still −5.8 ⇒ the search is discarding pawn-ending truth; null move in zugzwang-rich
pawn endings and LMP/futility on pawn pushes are the textbook causes [S, but it is the standard explanation and
exactly what SF11's three gates target].

**Instruments & gates for 2.1:**
- Static mechanism check (no games): fire counters per rule on the 313-row corpus; the d10 search bias on the
  pawn-ending rows (`_pe_search_bias.py`) BEFORE/AFTER, prediction: −8.5 → nearer −1.0; held-out K+P rows must move
  the same way (generalisation).
- Cost: quiet-corpus median nodes (`depth_nps_bench --n 60 MAX_DEPTH=10`) and `depth@1s` — these rules ADD nodes in
  pawn endings only; the all-position cost should be ≈0 (prediction: < +2% quiet nodes).
- STS@249,014 equal nodes (sanity), `_eval_symmetry`-style colour mirror of the SEARCH on K+P FENs (the new gates
  must be colour-blind).
- Games: pawn-ending-heavy start set (K+P stress FENs as openings) for a sharp read + the standard UHO
  self-play and SF18 @1000 for "no harm". Ship rule unchanged (~2σ combined).
- 2×2 with corr hist (step 2.2) before any bundle — anti-additivity law.

### 2.2 Correction history — our built version vs the giants

**Ours [E cpp:354-373, 5633-5644, 6162-6183; h:2859-2921]:** one table `pawnCorrHist[maxbit][pawnKey & (CORR_SIZE−1)]`
of an integer EMA (`e += (best − static − e) >> 6`, clamp ±2000 mp), applied × 0.75 ONLY to `rfp_static_eval` (RFP +
null-eval gate) and optionally to qsearch stand-pat/horizon; updated at node exit when the best move is a
non-capture, non-mate, not in check; **no fail-high/fail-low guards, no depth weighting, raw eval cached in
`rfp_raw_eval`** (so the TT-compounding trap is avoided at that one site).

**SF16 (first version, one table) [E `stockfish_16/src/search.cpp:67-72, 1342-1350`]:** `v += cv·|cv| / 12475`
(QUADRATIC application); update guard `!inCheck && (!bestMove || !capture(bestMove)) && !(bestValue ≥ beta &&
bestValue ≤ staticEval) && !(!bestMove && bestValue ≥ staticEval)`; bonus `clamp((best − static)·depth/8, ±LIMIT/4)`
via a gravity-style `<<` update (not a plain EMA).

**SF17 (five tables) [E `stockfish_17/src/search.cpp:85-97, 128-152, 1494-1502`]:** `7685·pawn + 7495·minor +
9144·(nonPawnW + nonPawnB) + 6469·cont(ss−2)[piece][to]`, `/131072`, LINEAR; guards `(best < static && best < beta)
|| (best > static && bestMove)`; bonus `(best − static)·depth/8` clamped; weights per table 111/146/162/143 ÷128;
applied once where `ss->staticEval` is assigned in BOTH search (:812, :822) and qsearch (:1593, :1606); TT stores the
UNADJUSTED eval (:1486).

**Weiss [E WebFetch history.h]:** pawn + minor + major + nonPawn[colour] + continuation at offsets 2..7;
`c/131072` with weights 5868/7217/4416/7025/4060…2901; bonus `clamp((score − eval)·depth/4, −172, 289)`; applied
`× (256 − rule50)/256` when rule50 > 7 [E WebFetch search.c]. **Ethereal master has no correction history** [E
WebFetch — none surfaced].

**Nuances we must not miss (the ones that differ from our build):**
1. **Where it is applied.** The giants correct the ONE `ss->staticEval` every consumer reads (RFP, razoring, null
   gate, futility, LMP-improving, stand-pat). Ours reaches RFP only. Our 07-30 qsearch extension HURT (−49 STS) —
   but with a single coarse table on v1 (memory `corrhist-qsearch-harmful…`); SF/Weiss apply it in qsearch fine.
2. **Keying breadth + gravity update.** Five independent partitions, each a `<<` gravity update (bonus −
   entry·|bonus|/LIMIT), not an EMA. Our Phase-0 result "all keys identical ⇒ position-local" was measured on v1's
   residual with an EMA replay. **Re-run the offline gate on v2** (the `corrlog`/`corrsignal` tooling exists;
   `generateMinorKey`/`generateMajorKey`/`generateNonPawnKey` exist in `cache_management.h` per SEARCH-SWEEP §11).
   Add the continuation key (we have `g_searchStack`).
3. **Update guards.** We update on every non-capture best move; the giants refuse fail-highs below static
   (`best ≥ beta && best ≤ static`) and fail-lows without a move. Note our 08-24 replay found Weiss-style guards
   WORSE (+1.76 vs +3.21) — on v1; keep the comparison arm.
4. **Depth weighting** (`·depth/8`, `/4`): our replay found it worse (slows learning on starved slots) — same caveat.
5. **Frame.** In negamax the correction is stm-relative and keyed `[us]`; ours is root-relative keyed by `maxbit`.
   The negamax rewrite makes the giants' form the natural one (§4).
6. **TT must store the UNADJUSTED eval** (SF17:1486). Our TT has no eval field at all; the q-cache keyed by
   zobrist froze corrected values in 07-30 (the bug that confounded the qsearch arm).
7. **Rule50 damping** (Weiss) — we have no rule50 scaling anywhere (grep: only the ≥100 draw test, cpp:4400 etc.).

**Instruments & gates for 2.2:** Phase-0 offline (`corrsignal` on v2 logs: GLOBAL arm + random-bucket arm +
ORACLE vs REPLAY, kill criteria as 08-24) → if a key separates: build the multi-table gravity form behind one flag →
quiet nodes + STS@eq-nodes + `_pe_search_bias` rows (corrhist is a candidate explanation for the pawn-ending
bias too) → games both instruments. Prediction to register: on v2 the GLOBAL slot still beats single-pawn keying;
the 5-table sum may clear it. ⚠️ Eval-side interaction: the owner's plan prices DYNAMIC terms INSIDE the new search
because corrhist may absorb part of what they add — so corrhist must be settled BEFORE the dynamic-lane retune.

### 2.3 Threats in search — tests

**Where it stands [E C3 §20a:1049-1057]:** real d10 −7.8% (better), self-play @50k +32 ± 9, SF18 @1000 −5 ± 9
(2,000 paired, 4 seeds). Three hypotheses on record. Knobs exist: `THREAT_V2_PCT` + 5 leg flags (h:1241-1255).

**Mechanism behind hypothesis 2 [S — standard theory, not measured here]:** a static, SIDE-SYMMETRIC threats term
credits (a) hanging/attacked pieces of the side NOT to move — which qsearch will cash anyway when captures are
searched, so at stand-pat they are double-counted with what the capture search returns — and (b) threats AGAINST
the side to move, which the stm usually parries with its next quiet move (one hanging piece ≈ nothing; two ≈ a
piece). SF's eval is stm-relative with `Tempo` and its threats term is still symmetric, but SF's search reads the
eval mostly at NON-PV nodes gated by `improving`/margins fitted jointly with it. Our RFP/futility margins were
fitted WITHOUT threats; turning threats on moves the eval distribution they were pinned to — exactly the
anti-additivity shape.

**Test ladder (cheap → expensive), each with a registered prediction:**
- **T1 (owner's first test): self-play at 250k nodes** (vs the 50k that gave +32). Node-limited ⇒ safe while the
  owner plays. Prediction per hypothesis 1: the gain fades toward ~0 by 250k. If +32 persists ⇒ hypothesis 1 dead,
  3 (exploitation of the ship's blind spot) or 2 live.
- **T2: SF18 @1000, 2 more fresh seeds** (the 4-seed −5 ± 9 was stopped early); pooled only.
- **T3: stand-pat ablation** — needs a one-line knob: threats weight at qsearch stand-pat/horizon (an
  `eval_by_mode`-style site switch, e.g. `THREAT_V2_QS_PCT`), so threats are ON in main-search statics (RFP,
  futility, corrhist) and OFF/halved at qsearch leaves. If T3 recovers the SF18 read ⇒ hypothesis 2.
- **T4: side-to-move-aware variant** — split the term into `T_vs_stm` and `T_vs_opp` with separate percentages;
  first setting to try: `T_vs_opp` at ~50% (qsearch cashes the rest), `T_vs_stm` full only when ≥ 2 targets
  (cannot save both) else ~50%. ⚠️ A stm-aware eval term breaks the "eval is a function of the position" caching
  assumption only if stm is not in the key — our zobrist includes turn, fine; `_eval_symmetry` mirror swaps turn,
  fine. ⚠️ In the Black-positive absolute frame this is NOT a tempo term (EVAL-V2-SLICE1-TEMPO-DESIGN.md:119 —
  "not negamax") — it must be written as a stm-conditioned weight, not a sign flip.
- **T5: pruning interaction** — threats ON + re-crank RFP/futility (the `crank_sweep.sh` protocol): if threats
  tolerate a harder crank than the ship (SF11 did +78 where we did −207), the term is accurate and the margins are
  the problem.
- **T6: threats as ORDERING only** — `ENABLE_THREAT_HIST`-style, or a static "moves a threatened piece / makes a
  safe threat" ordering bonus for history-0 quiets (memory `ordering-and-reduction-idea-backlog` #1). Cheap screen:
  FMC% + quiet nodes.
- **Gate:** both instruments ~2σ combined, pooled seeds, fresh baselines; also the d10 `MODE=dualread` re-search
  (memory `root-delta-depth-proxy-is-biased-against-dynamic-terms`: never close a dynamic term on the proxy).

### 2.4 Cheap stand-pat / static-ordering re-tests
- `QSTANDPAT_EVAL_MODE` — **do not run** (§1 #7): the whole v2 eval is 1.13× NPS; stand-pat is the leaf eval;
  the −235 STS accuracy cost cannot be bought back by <13% speed. Write the arithmetic into the record and close.
- `FUTILITY_EVAL_MODE` — same arithmetic; instead test the **parent-eval form** (SF11 :1016-1024) in negamax v2,
  which removes the child eval call entirely and adds the history gate. Judge at fixed time + STS (child-eval
  futility is more accurate per fire; the trade is nodes vs calls).
- `ENABLE_STATIC_ORDER` — screen on v2 (FMC%, quiet nodes, STS@eq-nodes) at `HIST_MAX=1`; games only if the
  screen beats bench noise; keep LMP fixed during the test; prediction: bench +, games ≈ 0 (as before) unless v2's
  PST changes the picture.
- `ENABLE_NULL_EVAL_GATE` + `ENABLE_NULLMOVE_EVAL_R` as a PAIR (SF's form) on v2 — both read the static eval ⇒
  premise changed; quiet nodes + STS + `depth@1s`, then games.
- `RFP_RETURN_BLEND` 50/67 — never tested; one cheap bench pass.

---

## 3. NEGAMAX MIGRATION PLAN

### 3.1 What changes structurally [E]
- Eval frame: `get_board_evaluation` returns an absolute Black-positive millipawn total, negated once when
  `Config::side_to_play` (cpp:9398-9399; `eval_v2.cpp` is Black-positive per SESSION-HANDOFF-10-07 "What this is").
  Negamax needs **side-to-move-relative**: `eval_stm = (turn == BLACK) ? total : −total` (verify the `turn` encoding:
  `occupied_colour[1] = white`, h:3274; `turn==true` is White per `is_check(current_state.turn, …)` usage — confirm
  against `initialize_engine`). The `side_to_play` flip disappears from the search; it survives only in the UI/
  reporting layer.
- Four searchers (`alpha_beta` root, `minimizer`, `maximizer`, `pre_minimizer`, `qSearch(is_maximizing)`) → one
  `search(alpha, beta, depth, cutNode)` + one `qsearch`. The root loop (presearch + root razor) is a design choice
  (§3.4).
- Mate scores: ours `±9999999 ∓ moveNum` with `moveNum = state_history.size()` (cpp:7805-7816) — ply from GAME
  start, not from root; negamax convention `MATE − ply_from_root`, `value_to_tt/value_from_tt` adjust by ply.
- TT: today fixed-orientation absolute bounds, "the EASY case" (memory `tt-and-cache-architecture`); in negamax
  bounds are stm-relative (standard) and the child-probe architecture becomes a node-entry probe. The 07-28 review's
  "do NOT unify to negamax — maximal risk, zero measured payoff" (search-architecture-fable-review:70-73) is now
  overridden by the owner's call; its depth off-by-one trap (:80-83) still applies: all stores/probes use the
  PARENT frame's remaining depth.
- History tables indexed by `turn` stay; corrhist `maxbit` → `[us]`; killers per ply unchanged; `g_searchStack`/
  cont-hist unchanged.

### 3.2 Verification ladder (each step byte-reproducible, no games)
1. **Perft** — no perft exists in `NN Engine/` [E grep]; python-chess is already a dependency (`selfplay/
   engine_server.py`, `tournament.py`, …). Build `diagnostics/_perft.py`: our movegen (via the Cython entry or a
   small C++ hook) vs `chess.Board` on the standard suite (startpos d1-6, Kiwipete d1-5, positions 3-6 of the CPW
   set, plus 200 random game FENs incl. ep/castling/promotion). Movegen is unchanged by the negamax switch, so this is
   the baseline that later protects the **pin-aware movegen** (§4) — write it once, keep it.
2. **Fixed-depth equivalence with pruning OFF** — old vs new at d1-6 on ~300 FENs (WAC + quiet corpus) with
   `ENABLE_NULLMOVE=LMR=LMP=FUTILITY=RFP=RAZORING=ROOT_PRESEARCH=ROOT_RAZOR=0`, `CHECK_EXTENSION=0`, TT and all
   caches OFF, qsearch delta OFF, `MAX_QDEPTH` equal: root score (sign-mapped) and best move must be IDENTICAL, and
   node counts (make_move calls, §5) equal. Then qsearch-only equivalence separately (the fail-hard stand-pat at
   cpp:7881-7901 and the ±mate `best` init must be reproduced or deliberately replaced — decide and record).
3. **Leaf sign test** — 10k random positions: `eval_stm(pos) == −eval_stm(pos with turn flipped)` when no ep/
   castling difference; and `search_d1(pos) == −search_d1(null-moved pos)` where legal.
4. **Colour-mirror of the SEARCH** — extend `_eval_symmetry.py` to the search: `bestmove(mirror(pos)) ==
   mirror(bestmove(pos))` and equal scores at fixed depth with caches cleared, on the 4,000-position set (v2 eval is
   0/4000 clean, so any asymmetry is the search's). The old search cannot be assumed to pass; the new one must.
5. **Feature-by-feature re-enable** — turn the old features back on one at a time in BOTH searches and require
   identical trees where the semantics are identical (TT off), diff where they are not and explain the diff.
6. **Fingerprint + fixed-time games** old-vs-new at parity config (equal nodes AND equal time, `node_ab` +
   `tournament`), expecting ≈0 at equal nodes and ≥0 at equal time (one function, fewer copies).

### 3.3 Sequencing vs the transition (recommendation, for discussion)
- **Option A — transition first on the old search, negamax later.** Pros: pawn-ending gates are ~20 lines on
  3 sites; corrhist is built. Cons: every item is written TWICE (min+max+pre+q) and then ported; colour-asymmetry
  risk per item; the dynamic-lane retune would be priced on a search about to be replaced.
- **Option B — negamax parity port first, then the transition on it.** Pros: one implementation of everything;
  node-entry TT probe, `cutNode`, `improving`, stm-relative corrhist come for free; the giants' forms are
  copy-fit. Cons: weeks with no Elo; parity must be proven (3.2) before anything new.
- **Recommended hybrid [S]:** do 2.1 (the three npm/passed-pawn gates) NOW on the old search — tiny, symmetric by
  construction, measurable on `_pe_search_bias`, and they teach us the pawn-ending instrument. Run the corrhist
  Phase-0 gate on v2 (offline, no search change). Start the negamax parity port in parallel as a separate file
  (`search_v2.cpp`) behind `SEARCH_V2=1`, with 3.2 as its gate. Everything else (corrhist multi-table, threats T3-T6,
  parent-eval futility, singular/ProbCut/IIR retests, SPSA) lands on negamax only. The owner's rule "see one
  subsystem through" argues for B over A once the eval arc's structural retune is done.

### 3.4 Root architecture decision inside the port
Keep presearch + root razor as the FIRST port (it is measured: removing it costs +36-42% nodes and the fair root-LMR
replacement lost −25 ± 19.6 at equal nodes). Port the root loop as-is. The giants' root (sorted by score, every move
searched, late root moves REDUCED, `averageScore`-centred aspiration) becomes a later A/B with ALL its supports
(root table coverage 100%, exempt-best, sorted) — the bundle the record says was never tested. Uniqueness is not at
stake here (owner: code style is not the uniqueness) — the question is only which root wins at equal time.

---

## 4. SPEED PLAN (profile-first)

**Bar [E]:** a speed change must be ≳35% NPS to be visible in benches; owner's rule of thumb 2× speed ≈ +70-100 Elo
here. The SF11 gap is 5.1× NPS × 9.48× nodes-to-depth — they compound.

**Step 0 — profile on v2, both workloads, with scopes that answer the fix question [E tooling exists]:**
`build_profile` → `evalprofile` (WAC) + `evalprofile_mid` (quiet d12). Existing scopes: `V2_EVAL` (6 sub-scopes),
`MOVEGEN` with `MG_PSEUDO`/`MG_ISSAFE`/`MG_SCORE`/`MG_SORT`, `MAKEUNMAKE`, `TT_PROBE`, `SEARCH_ROOT`. Add (compile-time
only, never a runtime flag): `MG_CACHE` (moveGenCache probe/store + the per-position move-list copy), `REPETITION`
(the `unordered_map` find/insert per make/unmake — cpp:7743 `is_repetition` on EVERY qsearch node [E]),
`STATE_COPY` (`BoardState current_state = state_history.back()` per node, cpp:7778 — a 13-word struct copy [E
h:3265-3285]), `QCACHE`, `SEE`. ⚠️ `__rdtsc` scopes inflate what they wrap (memory `the-nps-gap…`); read shares as
upper bounds; ~26% of v2 eval is unscoped.
Then read the generated ASM of the top scope before any hand fix (the 08-25 lesson).

**Step 1 — pin-aware legal move generation (the indicated algorithmic fix) [E memory `the-nps-gap…`: ISSAFE 2.08×
PSEUDO].** Building blocks exist: `pin_mask` (cpp_bitboard.h:1048), `slider_blockers` (cpp_bitboard.cpp:5372,
9093), `generateEvasions` (move_gen.h:579). Design (the SF/Ethereal shape): per node compute `checkers`, `pinned`
(blockers of our king), king-danger squares once; generate non-king moves with `from ∈ pinned ⇒ to ∈ line(king,
from)`, king moves filtered by an attacked-square test on the occupancy minus the king, ep with the two
discovered-check cases, evasions when in check — delete per-move `is_safe()` (move_gen.h:202/221/274) and the
16 mask-pair passes (13.8-14.7 passes/node). ⚠️ Correctness catch from the record: final order ties break by
generation order ⇒ the tree moves unless the tiebreak reproduces the old sequence (memory `movegen-is-36-percent…`
"CORRECTNESS CATCH"). Gate: **perft identical** (3.2.1), then IDENTICAL node counts/solves/STS with only NPS moving
(node-identical speed work composes exactly — the one class exempt from anti-additivity), else explain the tree move.
Expected: the 08-25 "~30% NPS" estimate is **refuted by implication** in the record (direct-emit showed the copy
traffic ~1%); honest prior = unknown until Step 0's `MG_ISSAFE` is re-read on v2 as a share of TOTAL. [S] If it is
still ~20-25% of total, deleting it is ~1.3× — at the bar, worth it only inside the rewrite.

**Step 2 — the structural costs the giants do not pay [S until profiled]:**
- `std::vector<BoardState> state_history` + `std::unordered_map position_count` for repetition: SF/Ethereal/Weiss
  keep an undo stack and scan a ply-indexed key array for repetitions (O(plies since irreversible)). A hash-map
  insert/erase per make/unmake and a `find` per qsearch node is plausibly several % — profile.
- `moveGenCache` (384 MB + heap) storing ordered move lists per position: the giants regenerate (cheap with pin-aware
  gen) and order lazily with a staged picker (TT move → good captures → killers → quiets by history → bad captures).
  Our cache is part of the baseline's node economy (24% hit rate, memory `movegen-is-36-percent…`), so it cannot be
  dropped on speed alone — measure its hit value under pin-aware gen. ~1.2 GB of resident tables (memory
  `tt-and-cache-architecture`) also hurts cache locality (`tt-footprint-is-a-throughput-tax-we-pay-now`).
- Staged/lazy generation: sized at a 5-8% NPS ceiling in 08-25 — but that sizing EXCLUDED pin-aware gen and assumed
  the move-gen cache stays. Re-size after Step 1.
- Eval: 11.5% — NOT a lever (memory). Pawn hash ≈ 1.7% of node cost — do not build for Elo. Incremental
  material/PST would matter only for `improving`/corrhist-everywhere evals; v2 is cheap enough.
- TT: node-entry single probe with an `eval` field (SF form) replaces per-child probes + separate `evalCacheNew`;
  the q-cache gets a generation/age field (the real gap vs SF, memory `corrhist-qsearch…`).
- Already done: PEXT sliding attacks (OPTIMIZATION_LOG:1305), `-Ofast -march=native -flto`, SEE incremental.

**What Ethereal/Weiss do for speed [E WebFetch + general knowledge, S where unmarked]:** staged move picker with
SEE-filtered noisy stage; pseudo-legal generation with legality checked only for the moves actually tried (Ethereal
`if legal` inside the loop); incremental make/unmake with a `Undo` stack; `pawnCache` (Weiss) / pawn-king hash
(Ethereal) for the eval; TT prefetch (`prefetch(TT.first_entry(key_after(move)))`, SF11 :1096/:1505); no per-node
heap allocation. Ethereal ~1.5-3M NPS single-thread with a far costlier HCE than v2 — so the gap is in the
make/unmake/movegen/TT path, consistent with our profile.

**Order:** Step 0 → pin-aware gen (with perft) → repetition/undo stack → TT/eval-field consolidation → re-size staged
gen. Each gated node-identical where possible, else fixed-time + STS. Do this INSIDE the negamax port where the
move loop is rewritten anyway (one rewrite of the move loop, not two).

---

## 5. A FAIR NODE-EFFICIENCY INSTRUMENT (ours vs SF11/SF15c/Ethereal/Weiss)

**Why the current numbers are not comparable [E]:** our `num_iterations` counts presearch, aspiration re-searches,
qsearch and TT-hit bookkeeping (`use_tt_entry` increments it) and the printed EBF divides by the loop's exit depth
(memory `ebf-metric-is-not-comparable`); `NODES` excludes the q-tree while `qnodes` is separate (INSTRUMENT-MAP §D);
`--depth` is not equal work across engines (SF's d10 is a far deeper effective search); WAC node counts reversed sign
on 4/6 configs vs the quiet corpus.

**Design — `node_efficiency_bench` (one script, both sides):**
1. **Node definition = `make_move` calls** (main + qsearch, excluding presearch? — NO: include everything the engine
   does to produce the move, but ALSO report the split). SF counts `nodes` in `do_move`; Ethereal/Weiss likewise.
   Add a compile-time-excluded counter at our `make_move`, report `nodes_total`, `nodes_main`, `qnodes`,
   `nodes_presearch`, `tt_hits` separately. ⚠️ Do NOT change `num_iterations` (byte-id fingerprints and
   `NODE_LIMIT` calibration depend on it).
2. **Corpus = the 60-FEN stratified quiet midgame corpus** `depth_nps_bench` uses (seed 1234), extended with the
   313 pawn-ending rows and the K+P held-out set as separate strata (`_sf_node_bench.py` already runs SF on the
   identical FENs [E memory `the-sf11-gap…`]).
3. **Metrics, each with its reference column:**
   - **Depth reached at N nodes** for N ∈ {100k, 250k, 1M} (three budgets — the step-shape caveat); report median
     and the per-position distribution, not the mean.
   - **Nodes to reach depth d** for d ∈ {8, 10, 12} — informative but NOT equal work (extension/reduction regimes
     differ); label it so.
   - **Real marginal EBF** = `N(d)/N(d−1)` from per-iteration node logs (`info depth` lines for SF; a per-iteration
     print for us), median over positions, d 8-12.
   - **STS300 at equal nodes** (`_sts_reference.py --nodes 249014` + ours at `NODE_LIMIT=249014`) — the accuracy per
     node reading; and at a second budget (100k) to defeat the step artifact.
   - **Time-to-depth at fixed NPS**: `nodes_to_d10 / NPS` — the number that actually predicts play; quote with
     the NPS that produced it (LIGHTNING mean 450k; LONG_FORMAT ~386k).
   - **Structure stats** that explain a change: first-move-cutoff %, qnodes share, presearch share, null-move
     cutoff rate, LMR re-search rate, TT hit rate — ours only (SF exposes none), used to diagnose, never to judge.
4. **Protocol:** both engines single-thread, same FENs, same hash size, TT cleared per position, 3 reps for timed
   numbers on an idle box (timed benches span 14.6% NPS run-to-run, INSTRUMENT-MAP §D), nodes/depth numbers are
   deterministic. Ethereal/Weiss as UCI binaries (fetch + build under WSL) extend the ladder beyond SF11 — they are the
   3000-class HCE targets the owner named.
5. **How it lies:** depth is not comparable across engines with different extension policies (our `CHECK_EXTENSION`
   inflates `depth_limit`; SF's singular extends too) — the honest cross-engine columns are STS@N and depth@N on the
   SAME N; "nodes to depth d" is for OUR before/after only. Our qsearch has a depth cap (MAX_QDEPTH=10) and SF's does
   not; both are "nodes" either way.

---

## 6. NUANCES IN THE GIANTS' SEARCHES WE MAY BE MISSING

Ranked by (how many of SF11 / Ethereal / Weiss have it) × (absence here). [E] for the reference side; "ours" is
from the code read above.

1. **`improving` as a universal modulator** (3/3): futility margin, LMP count, LMR, null-move gate, ProbCut margin,
   RFP (Ethereal `BetaMargin·max(0, depth − improving)`, Weiss `77·(depth − improving)`). Ours: absent (shelved as a
   lone +1 ply on v1). Needs the node-entry static eval at every non-check node.
2. **One node-entry `staticEval`, stored in the TT, corrected once** (3/3 store eval in TT; SF/Weiss correct it;
   SF11 uses `ttValue` as a better eval when the bound allows, :803-806). Ours: eval computed conditionally for RFP
   only; TT has no eval field; qsearch recomputes.
3. **Null move: eval-gated, depth+eval-scaled R, npm guard, verification at high depth, `(ss−1)->statScore`
   guard** (3/3 for gate+R+npm). Ours: no gate, flat iteration-depth R, type-blind material guard, verification via
   `VERIFY_RESEARCH_REDUCTION`.
4. **Pruning gates `non_pawn_material(us)` and `bestValue > mated`** (SF11 on all shallow pruning; Ethereal
   `best > −TBWIN_IN_MAX` on LMP/futility/SEE). Ours: none ⇒ we prune when already mated and in pawn endings.
5. **Parent-eval futility with a HISTORY gate** (SF11 `staticEval + 235 + 172·lmrDepth ≤ alpha && hist sum < 25000`;
   Ethereal `FutilityPruningHistoryLimit[improving]`). Ours: child-eval, no history gate, margins 0.2-0.95 pawn.
6. **Countermove-/continuation-history pruning** (SF11 ~20 Elo :1010-1014; Ethereal "Continuation Pruning ~10";
   Weiss history pruning). Ours: `ENABLE_HIST_PRUNE` off, butterfly only.
7. **Quiet SEE pruning keyed on `lmrDepth` with a quadratic margin** (SF11 :1027; Ethereal with `− hist/128`;
   Weiss `−73·depth`). Ours: off, flat, per-square SEE.
8. **Singular extension + multicut + (Ethereal/Weiss) double and NEGATIVE extensions**; `singularLMR` reduce-less.
   Ours: built single extension only, never games-tested live.
9. **LMR fine terms**: `ttPv −2` (SF11), `cutNode +2`, `ttCapture +1`, escape-capture −2, opponent moveCount >14,
   `(ss−1)->statScore` comparison, `ttHitAverage` (SF11 only — a running "how predictable is this region"
   statistic feeding reductions, :1129); Ethereal `R += !PvNode + !improving`, `R −= hist/6167`, noisy LMR
   `3 − hist/4952`; Weiss `r += nonPawnCount[opp] < 2`, `r += 2·cutnode`, `r −= improving`. Ours: statScore + killer
   protection + product schedule.
10. **Extensions beyond checks**: passed-pawn push (SF11), last capture in low material (SF11), castling (SF11),
    check only if SEE≥0 or discovered (SF11 — we SEE-filter too). Ours: checks only, capped at 3/path.
11. **qsearch**: TT probe + TT move in qsearch (3/3); stand-pat seeds `best` (3/3); per-move futility with victim
    credit (ours now via `QDELTA_PERMOVE`); SEE-prune non-check captures (`!see_ge(move)`, SF11 :1501); evasion
    pruning in check (:1495-1498); checks only at the first q-ply (`DEPTH_QS_CHECKS`; ours `QCHECK_DEPTH0`); no
    depth cap (ours MAX_QDEPTH=10 with the horizon-before-check-test defect). Ethereal: delta prune vs
    `moveBestCaseValue`, short-circuit `eval + pessimism > beta`.
12. **Root**: no root pruning (3/3); every root move searched, late ones reduced; re-sort by score every iteration
    (SF11 :462/:506); aspiration centred on `previousScore` (SF11) / `averageScore` (SF16+); best-move-stability
    time management. Ours: presearch + root razor (fair-tested as better for us); `best_move` taken on `score >
    best_score` without `score > alpha` on a fail-low pass (memory `presearch-economics…` "STILL OPEN at :3517";
    knob `ENABLE_ROOT_BEST_REQUIRES_ALPHA=false` h:1470).
13. **TT**: generation/aging + bucket replacement + `ttPv` flag + eval field (3/3). Ours: direct-mapped, never
    cleared, no age, no eval, child-probe.
14. **Correction history** (SF16.1+, Weiss; not Ethereal): see §2.2.
15. **Piece×to continuation history at (ss−1),(ss−2),(ss−4),(ss−6)** (SF11 :943-945); `lowPlyHistory` (SF12+);
    capture history by [piece][to][captured]. Ours: from×to cont1/cont2, capture history by [side][from][to]
    (victim re-key flat).
16. **Draw-score noise** `VALUE_DRAW ± 1 by node parity` (SF11 :91-93) to avoid 3-fold blindness; **mate-distance
    pruning** (Ethereal step 3); **rule50 eval scaling** (SF in `evaluate`, Weiss in corrhist). Ours: none found.
17. **Weiss `TTScoreIsMoreInformative`** (trust a TT bound only when it says more than the current window) and
    SF11's "use ttValue as eval when the bound allows" — cheap, 2-line ideas.

---

## 7. OPEN QUESTIONS FOR THE OWNER

1. **Sequencing:** hybrid (§3.3) — pawn-ending gates + corrhist Phase-0 on the old search now, negamax parity port in
   parallel, everything else on negamax — or strictly "negamax first"? How long a no-Elo window is acceptable?
2. **Parity or redesign in the port?** A pure parity port (identical trees with pruning off, then feature-by-feature)
   is the safe verification path but reproduces our fail-hard stand-pat, ±mate `best` init, iteration-keyed null R,
   type-blind null guard, horizon-before-check qsearch cap. Which of these do we deliberately NOT reproduce (and
   accept a tree diff at step 3.2.5)?
3. **Root architecture:** port presearch + root razor as-is (measured best for us) and A/B the giants' root later
   with all three supports — agreed? Is the presearch part of the engine's identity you want kept, or purely a
   measured choice?
4. **Pawn-ending rules — SF11-narrow or ours-wide?** SF11's passed-push extension requires the move to be
   `killers[0]`; our old exemption fired on every advanced push and over-fired. Start narrow (SF11) and widen only on
   the pawn-ending corpus?
5. **Threats:** confirm T1 (self-play @250k) first; may I add the one-line `THREAT_V2_QS_PCT` site switch for T3 and
   the stm-aware split for T4 (both byte-identical at default) once the game slots are free?
6. **Corrhist:** re-run the offline Phase-0 gate on v2 BEFORE building the five-table form — agreed? If the v2
   residual is also position-local, do we still want corrhist (the giants' ~10-20 Elo) or do we skip it?
7. **Node-efficiency instrument:** `make_move`-call counting as the cross-engine node definition, and
   Ethereal/Weiss built under WSL as ladder rungs — do you want them as judges too, or SF11/SF15c only?
8. **SPSA budget:** search params are to be tuned by GAMES (your call). What nightly budget (games/night, node- or
   time-limited) can the search arc count on, given the 4-slot limit and your 9pm-midnight play window?
9. **Speed inside the port:** pin-aware movegen + undo stack + node-entry TT are move-loop rewrites. Do them inside
   the negamax port (one rewrite) or as a separate, node-identical phase afterwards (cleaner attribution)?
10. **Mediocre's source** (promised for the NPS phase) — when, and is it the Java v0.5?
11. **Instrument for "critical positions"** (INSTRUMENT-MAP §F: n_crit 27/43) — the search arc will again be judged
    on averages; do you want a criticality-enriched search corpus built before the margin re-sweep?

---

## Appendix — evidence index (files read)
- Code: `search_engine.h` (Config knobs: :222-2060, :2770-3007, :3087-3240, :3265-3285), `search_engine.cpp`
  (corrhist :354-373; LMP/futility/LMR :4530-4630; node entry/RFP/null :5625-5669, :6320-6420; IIR/ProbCut
  :5757-5814; corrhist update :6156-6187; qSearch :7740-8000; null guard :8509-8551; `reduced_search_depth` :8553+;
  eval flip :9393-9399), `cpp_bitboard.h` (:778, :987-1008, :1048-1087), `cpp_bitboard.cpp` (:5366-5387, :9081-9146),
  `move_gen.h` (:175-274, :579, :658, :997, :1166).
- References: SF11 `search.cpp` :60-93, :780-1210, :1440-1519; SF16 :64-72, :1330-1354; SF17 :80-160, :1480-1507;
  Ethereal `src/search.c` and Weiss `src/search.c` + `src/history.h` via raw.githubusercontent.com (summaries).
- Record: SESSION-HANDOFF-2026-10-07, SEARCH-SWEEP-2026-08-25, speed-and-qsearch-findings-2026-07-24,
  REFERENCE-BENCH-LADDER (:330-385), TEXEL-C3 §20a (:1049-1068), INSTRUMENT-MAP §D/§F, DIAGNOSTICS-TOOLKIT,
  ENGINE-ORIENTATION, OPTIMIZATION_LOG (headers; :1235), STRENGTH_BACKLOG:26, search-architecture-fable-review-07-28
  (:55-94), EVAL-V2-INVENTORY-09-25 (:108-111), SESSION-HANDOFF-09-05:16 / -09-09-B:106, SESSION-HANDOFF-07-08
  (:358-394), collapse-search-efficiency-diagnosis-07-12:29.
- Memory: MEMORY.md, KNOWLEDGE-MAP §Search, long-term-goal…, search-changes-are-antagonistic…,
  eval-accuracy-payoff-is-pruning, a-truer-eval-buys-pruning-headroom…, the-nps-gap-is-mostly-not-eval,
  the-sf11-gap-is-two-thirds-node-efficiency, movegen-is-36-percent…, iir-is-the-venue-correct-node-saver,
  half-a-ply…, node-savings-below-35…, ebf-metric-is-not-comparable, presearch-economics…,
  root-techniques-all-depend…, presearch-and-root-razoring…, margin-sweep-52-configs…, corrhist-signal-is-position-local…,
  corrhist-qsearch-harmful…, corrhist-pawn-structure-reopens-pawn-hash, singular-banked, improving-heuristic-shelved,
  lighteval-standpat-is-leaf, ordering-and-reduction-idea-backlog, sf-source-paths-and-pruning-shapes,
  cutnode-allnode…, search-value-bugs…, tt-and-cache-architecture, root-delta-depth-proxy-is-biased-against-dynamic-terms.

---
## VERIFIED 2026-10-09 (code read; for the transition — nothing changed yet)
1. **Null-move guard counts pawns as pieces** — `isUnsafeForNullMovePruning` (search_engine.cpp ~8509): `pieceNum` = all non-king
   pieces incl. pawns; unsafe only below 7 (below 4 with a queen) ⇒ a PAWN-ONLY side with ≥ 7 pawns is null-moved (zugzwang). SF11
   disables null move whenever the mover has no non-pawn material (search.cpp:846). Fix = a `non_pawn_material(us)==0` gate.
2. **qsearch horizon return precedes the in-check test** — `qDepth >= MAX_QDEPTH` returns the static eval (search_engine.cpp ~7766)
   before `currently_in_check` is computed (~7789) ⇒ a mated position at q-depth 10 is scored as a normal position (the 07-24
   mate-blind defect, still live; rare). Fix = test check first (or never horizon-return while in check).
Both change search behaviour and the fingerprint ⇒ gate each like any arm, first items of the transition.
