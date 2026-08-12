# Session handoff — overnight 2026-07-27/28

---
# ★ PART 2 (2026-07-28 evening): the node-cost investigation

## The headline: our time-to-depth gap is NODE COUNT, not eval speed
`diagnostics/depth_race_vs_sf.py` — identical FENs, both engines, one process per depth (our Config knobs
latch at extension init, so varying MAX_DEPTH inside one process silently measures the same depth ten times;
an earlier version of the script did exactly that and the flat node counts were the tell).

| d | our nodes | SF nodes | node × | our ms | SF ms | time × | nps × |
|---|---|---|---|---|---|---|---|
| 5 | 5,530 | 382 | 14.5 | 36.8 | 1.0 | 36.8 | 2.5 |
| 8 | 41,715 | 774 | 53.9 | 139.2 | 1.0 | 139.2 | 2.6 |
| **10** | **133,090** | **2,160** | **61.6** | 481.7 | 3.0 | 160.6 | **2.6** |

**61.6 × 2.6 ≈ 160 — the decomposition closes.** Node count dominates; per-node cost is only 2.6×.
⇒ **The earlier claim "the gap is essentially all evaluation cost" is WRONG.**
Per-ply growth is close to SF's (ours 1.72, SF 1.50); we start **17.6× higher at shallow depth**.
**It is a fixed-overhead problem, not a growth-rate problem.**
⚠️ SF times sit at timer resolution (1.0 ms) below d9 — the NODE ratios are the trustworthy column.
⚠️ SF18 is NNUE, so its NPS is *lower* than an HCE's would be: the 2.6× per-node gap flatters us.

## REAL EBF ≈ 1.7/ply — the printed 3.934 is an artifact
`ENABLE_ITER_LOG` + `diagnostics/iter_ebf.py` (median per-iteration node ratios, WAC):
single-ply ratios alternate (odd-even effect), so use the 2-ply geometric mean → **≈1.7/ply**.
See memory `ebf-metric-is-not-comparable`. **Every EBF-based conclusion before this is void**, including
"EBF is dominated by the pre-search". ⚠️ Low EBF does NOT mean the search is healthy — see the node race.

## Node composition (`[node_split]`, before/after delta on the shared counter)
| component | nodes | share |
|---|---|---|
| **root pre-search** | **16,051,978** | **41.3%** |
| qsearch | 11,804,531 | 30.4% |
| total | 38,840,709 | — |
⚠️ qnodes and presearch OVERLAP (qsearch inside the pre-search counts in both); the pre-search figure is the
solid one. It also re-runs on every aspiration widening.

## Results this session
| config | verdict |
|---|---|
| `LMR_EXTRA=2` | **≈ −55 Elo** (243 games, interval excludes 0) ⇒ NO-GO. Also **disconfirms "we are under-reduced"** |
| `MAX_QDEPTH` 8/6/4 | **DOMINATED** — fewer solves AND *more* nodes (245/39.1M, 246/39.4M, 245/40.1M vs 249/38.8M). Noisier leaves cost the main search more than qsearch saves. Not a trade; closed, no games |
| `ENABLE_NODE_TT` | Built. byte-id exact. **stores=3,240,958 vs the old cutoff path's 36** ⇒ the defect model was right. Alone: 246 solves / +2.8% nodes |
| `ENABLE_SINGULAR` + node-TT | eligible **49.6M → 159.3M (3.2×)** but **fire only +11.6%** (2.13M → 2.38M), fire/elig 4.30% → 1.49%, +9.9% nodes. Mechanism confirmed, payoff thin. Possible cause: `SINGULAR_MIN_DEPTH=6` was tuned when entries came only from deep revisits |
| **`ENABLE_NODE_TT=1 ENABLE_SINGULAR=1` (GAMES)** | ☠️ **≈ −28 Elo** — SPRT `sprt_nodett_sing`, **+407 −503 =288 over 1198 games**, LLR −1.999 (ran to the 1200 cap, never hit the −2.94 bound). Margin ±23 ⇒ interval excludes zero. **NO-GO.** Consistent with the mechanism: ~10% more nodes bought a 12% increase in extensions. Our tightest measurement of the session |

## ★ SEARCH LANE LEDGER — 0 for 13 on 2026-07-28
quiet checks **−42** · check filters **dominated** · discovered checks **−6 solves** · capgains-redundancy
**disconfirmed** · PST ordering **null** · `LMR_EXTRA=2` **−55** · `MAX_QDEPTH` 8/6/4 **dominated** ·
`ENABLE_IMPROVING` **−5 solves** · `ENABLE_TT_MOVE` **redundant** (moveGenCache promotion already does it) ·
`ENABLE_NODE_TT`+singular **−28**. **Zero Elo from search.**
Against: **5 eval ships = +83.4 Elo**, and ~18pp of CLASSICAL headroom at equal depth (NNUE worth only 2.7pp
over SF15-classical). ⇒ **Recommendation: next real work goes to EVAL.** Bank the pre-search prefix-share
number via the three byte-identical Stage-0 counters, then leave search alone.
| `ENABLE_IMPROVING=1` | −5 solves, node-neutral ⇒ clean negative (as wired; the SF11 futility-MARGIN port is untested) |

## ⚠️ `ROOT_PRESEARCH_REDUCTION` is NOT a valid experiment
The pre-search's output **overrides** previous-iteration data. Reducing its depth produces data *shallower*
than what the previous iteration already has, while keeping it in charge — so the dial strictly degrades the
ordering source rather than trading accuracy for nodes. Any reduction must come AFTER a quality-aware merge.
Plan: `dev_notes/presearch-replacement-plan-2026-07-28.md` (design only, nothing built).

---


## ★ THE HEADLINE: qsearch has NEVER searched a quiet check

`buildNoisyMoveList`'s quiet-check test is broken, and the cause is a signature:

```
inline void update_state(..., uint64_t& castling_rights, int& ep_square, int promotion_type, bool turn)
```

**`turn` is passed BY VALUE** while every board mask beside it is a reference. The caller's `turn` is
therefore never flipped, so the following `is_check(turn, ...)` asks *"is the MOVER in check after their own
move"* — false for every legal move.

**Measured, not inferred:** a counter on the taken branch (`g_q_quiet_checks_added`, incremented only where a
quiet check is actually pushed) reads **0 across all 300 WAC positions** under the shipped default.

### The two flags were a matched PAIR, and neither ever worked
| config | WAC | nodes | EBF | reading |
|---|---|---|---|---|
| base | 249 | 38,840,709 | 3.934 | quiet checks: **none admitted** |
| `ENABLE_QCHECK_MASK=1` | 239 | **740,471,970** (19×) | 6.492 | detection FIXED, but at all 10 q-plies ⇒ ruinous |
| `ENABLE_QCHECK_DEPTH0=1` | 249 | 38,840,709 | 3.934 | byte-identical — drops an EMPTY set |
| **`MASK=1 DEPTH0=1`** | **253 (+4)** | 49,309,135 (**+27%**) | 4.202 | **the SF shape: quiet checks at the first q-ply only** |

`ENABLE_QCHECK_MASK` supplies working detection (bitboard test vs `enemy_king`, misses discovered checks —
the standard tradeoff); `ENABLE_QCHECK_DEPTH0` confines it to q-ply 0. **Alone, one explodes and one is a
no-op; the audit tested them individually and drew the wrong conclusion from each.** The pair had never been
run in this engine's history until tonight.

### ☠️☠️ GAME RESULT: the config is ≈ **−42 Elo. IT LOSES. DO NOT SHIP.**

🚨 **SIGN ERROR, CORRECTED.** `tournament.py` aggregates W/L/D from **p1's** perspective
(`score = (W + 0.5D)/n`, `elo = elo_from_score(score)`), and the `tournament` runner sub sets
**`--p1-label base --p1-config ""`** ⇒ **the printed Elo is BASE's, not the candidate's.**

| block | games | BASE score | BASE Elo | ⇒ candidate |
|---|---|---|---|---|
| `qcheck_pair_0727` | 487 | 54.5% | +31.5 ±36.3 | **−31.5** |
| `qcheck_pair_0728b` | 490 | 57.7% | +53.6 ±36.1 | **−53.6** |
| **pooled** | **977** | **56.1%** | **≈ +42.6 ±25.6** | **≈ −42.6** |

**Independently confirmed by SPRT**, which labels p1 = the CANDIDATE and so needs no interpretation:
`gate 'ENABLE_QCHECK_MASK=1 ENABLE_QCHECK_DEPTH0=1'` → **+44 −72 =38, elo ≈ −64 at 154 games**, LLR heading
to reject. Stopped early (it was also holding ~17.8 GB at concurrency 6).

**Mechanism is coherent:** +27% nodes at equal TIME costs depth, and the tactical gain does not cover it.
The +4 WAC at fixed depth was exactly the **node-increasing acceptance** that fixed depth flatters —
`node-saving-changes-need-fixed-time` warns about this and the warning was quoted and then ignored.

⇒ **`ENABLE_QCHECK_MASK` + `ENABLE_QCHECK_DEPTH0` stay DEFAULT OFF.** The quiet-check BUG is still real
(qsearch genuinely never searched one, counter-verified 0/300) — but **fixing it is not worth its price at
equal time.** Any future attempt must make checks far cheaper before it can pay.

🚨 **ALSO CORRECTED: `ENABLE_QCUT` = +6.8 ±28.8 (769g), NOT −6.8** — same inversion, still n.s.
✅ **AUDIT DONE — the historical ledger is CORRECT; the earlier "everything may be inverted" alarm is
RETRACTED.** `diagnostics/audit_tournament_signs.py` (new) re-derives every stored run from its own
`tournament.json`, which records `p1_label`/`p2_label`:

| run | base score | reported | verdict |
|---|---|---|---|
| `qdelta_pm1500` | 42.3% | qdelta +53.8 | ✅ |
| `malus_night` | 48.4% | malus +10.9 | ✅ |
| `spc_night` | 48.6% | SEE_PRUNE +9.5 | ✅ |
| `evalarc_2026_07_25` | 61.8% | arc +83.4 | ✅ (p2 was the REVERT ⇒ base holds the feature) |

Earlier sessions negated correctly. **Only the 07-28 readings were wrong — a reading error, not a systemic
defect.** ⚠️ The audit tool's `SIGN FLIPPED` marker assumes p2 is the new feature, which is FALSE for
revert-style tests; it flags rows for inspection, not conclusions.
**Prefer `gate` (SPRT) when the sign matters: its p1 IS the candidate.**

⚠️ Note the +27% nodes: this survived a **timed** venue, which is the honest test for a node-increasing
config (fixed depth flatters those — `node-saving-changes-need-fixed-time`). That it stayed positive at
equal time is the encouraging part, and it is the first candidate since qdelta to do so.

### Trying to buy the nodes back: `ENABLE_QCHECK_SAFE` — NO-GO, and instructive
Qsearch admits captures only at `see() >= 0`; quiet checks were the one category admitted unconditionally,
so a safety filter looked free. It is not. (`see()` itself is unusable here — keyed on the SQUARE, starting
from `get_value_at(to_square)` = 0 for a quiet destination, so it cannot know which piece is moving. The
filter uses bitboard tests instead, no board copy, preserving the mask path's advantage.)

| config | solves | nodes | vs base | checks rejected |
|---|---|---|---|---|
| base | 249 | 38,840,709 | — | — |
| **paired, unfiltered** | **253** | 49,309,135 | **+4 / +27%** | — |
| `QCHECK_SAFE_LEVEL=1` (enemy pawn attacks dest, non-pawn mover) | 251 | 43,637,502 | +2 / +12% | 10% |
| `QCHECK_SAFE_LEVEL=2` (+ attacked and undefended) | 248 | 40,395,689 | −1 / +4% | 56% |

**★ A perfectly monotone trade: every node reclaimed costs tactical yield in proportion.** ⇒ the +27% is
**real work, not waste** — if it were junk, some filter would have removed it cheaply. Level 2 fails hardest
because **a check that hangs its piece is frequently a sacrifice, i.e. the entire point of the line**; it
rejects 56% of checks and lands BELOW base.

**Recommendation: keep the UNFILTERED config.** It has 977 games behind it; L1's edge is speculative, small,
and would need its own ~1,000 games to separate from it. Both knobs are gated default-off and byte-id clean
(249 / 38,840,709 verified after every build), so L1 remains available if we ever want to squeeze.

Why it still deserves attention despite WAC's discredited record: this is a **soundness/mechanism fix**, the
same class as the per-move qdelta prune (+53.8 Elo), not a tuning knob. The last two things WAC liked
(QCUT +5, malus) both measured ~0 in games.

---

## ★ DISCOVERED CHECKS (`ENABLE_QCHECK_FULL`) — correct code, NO-GO result

`ENABLE_QCHECK_MASK` sees only DIRECT checks (moving piece attacks the king) and is structurally blind to
discoveries. `moveGivesCheckFast()` closes that gap without a board copy: relocate the mover's bit inside its
own type mask (`(mask & ~from_bb) | to_bb`), then one `attackersMask()` query against the enemy king. Catches
direct AND discovered. (Castling checks by the ROOK remain a known exclusion — only the king's from/to are
modelled; the mask path shares this gap.)

| config | solves | nodes |
|---|---|---|
| base | 249 | 38,840,709 |
| **paired, mask-only** | **253** | 49,309,135 |
| `ENABLE_QCHECK_FULL=1` | **247** | 46,239,909 |

**The detector is CORRECT: `missed=0`** (two-sided counter — zero direct checks the mask found that the full
test missed, i.e. a strict superset) and it found **107,982 real discoveries, 5.8% of admitted checks**.
So discoveries exist in quantity, and including them still LOSES 6 solves and lands below base.

**★ Likely mechanism — combinatorial fan-out.** A discovered check fires wherever the discovering piece goes,
so ONE discovery yields MANY near-duplicate checking moves (often a dozen destinations delivering the same
check), most of them pointless. Direct checks do not fan out this way. Qsearch drowns in redundant branches.
⇒ If ever revisited, admit at most ONE representative per discovery (e.g. best by placement), not all of them.

⚠️ Caveat: full is node-CUTTING relative to mask (46.2M vs 49.3M), and fixed depth is biased against
node-cutters (`node-saving-changes-need-fixed-time`), so this is a slightly pessimistic reading. Not pursued:
the gap is 6 solves and the mask config already has 977 games behind it.

## ★ CAPGAINS vs QUIET CHECKS — hypothesis tested and DISCONFIRMED

Standing puzzle: why does interior-node capgains exist at all if qsearch resolves tactics? Natural
hypothesis once the quiet-check bug was found — **capgains was compensating for a check-blind qsearch**, so
it should become redundant now qsearch sees checks. Tested 2×2 (fixed depth 10, `SCALE_CAPTURE_GAINS=0`):

| | capgains ON | capgains OFF | penalty |
|---|---|---|---|
| no quiet checks | 249 / 38.84M | 241 / 59.82M | **−8 solves, +54% nodes** |
| quiet checks | 253 / 49.31M | 246 / 65.50M | **−7 solves, +33% nodes** |

**The accuracy penalty is essentially unchanged (−8 → −7) ⇒ NOT redundant.** The two mechanisms are largely
independent. The node penalty does soften (relative +54%→+33%, absolute 21.0M→16.2M), so quiet checks absorb
some of capgains' work, but nowhere near enough to retire it.

⇒ **Capgains stays. The "why is capgains needed" question remains OPEN** — check-blindness is not the answer.
⇒ This also closes the hoped-for node-recovery route: we cannot pay for the quiet-check nodes by dropping
capgains.
⚠️ Note the recorded inventory figure (capgains off = −15 WAC / +45% nodes) has drifted to −8 / +54% — the
eval arc moved the baseline. **Re-measure inventory numbers before reasoning from them.**

## ★★ THE TAIL SCREEN IS FALSIFIED — the phase's instrument is gone

Full detail: `dev_notes/tail-screen-falsified-2026-07-27.md`.

| config | mean | **>20%** | **p99** | n |
|---|---|---|---|---|
| base | 25.25 | **1.0%** | **20.1** | 9823 |
| qdelta OFF (**−53.8 Elo**) | **25.03** | **1.0%** | **19.9** | 9838 |

The tail is **flat across a 54-Elo gap**, and the **mean is INVERTED** (qdelta-OFF looks better). Control
passed (`[qdelta_permove] seen=0 fires=0` ⇒ the prune really was off). The 07-26 separation (0.5%/0.8%) was
n=1477 ≈ 7 vs 12 events — inside its own ±10 Poisson bound. **It was noise.**

⇒ **Every metric we own fails to predict Elo.** The RULER (fixed nodes) is still right and still
deterministic; the STATISTIC is dead. **Games are the only instrument.** Pick candidates by cross-engine
prior warrant (`sf-schedule-portability-heuristic`), not by local measurement.

**Corroborating game result:** `ENABLE_QCUT` = **−6.8 ±28.8 Elo (769 games)**. Its +5 WAC did not convert.

---

## `ENABLE_TT_MOVE` — broken on BOTH sides, now half-fixed

- **Reader:** read `g_ttMoveTable`, which has **zero write sites** in the engine (declaration + 2 reads only).
  Superseded by the `TTEntry::move` store on 07-08; the dead table was left behind. → rewired to read
  `accessSearchEvalCache()`, promoting **within the quiet region** (captures untouched — the absolute-front
  hoist is what the 07-03 audit blamed). `TT_MOVE_POLICY=2` keeps the old shape for comparison.
- **Writer:** the only two `TTEntry::move` writes are at the beta-cutoff sites and were gated on
  **`ENABLE_SINGULAR` (default OFF)**; none of the ~21 `addToSearchEvalCache()` call sites pass a move.
  → store now gated on `ENABLE_SINGULAR || ENABLE_TT_MOVE`.
- ✅ **VERIFIED and PARKED (07-28):** it now runs — and fires **36 times in a whole 300-position WAC**,
  giving **243 solves (−6) / 38,509,916 nodes**. **Third defect:** the cutoff-site store probes
  `accessSearchEvalCache()` and writes only `if (nodeEntry != nullptr)`, but at a beta cutoff the node has
  **not stored its own entry yet**, so the probe returns null on first visit and the move is dropped; only
  re-visited nodes record one. (The SF move-preservation rule in `addToSearchEvalCache` is fine — not a
  clobbering problem.) **Correct wiring = pass the cutoff move through that function's unused
  `Move move = Move()` parameter at the node's own score store; no caller passes it (~21 sites).**
  **PARKED:** the moveGenCache cutoff-promotion already fills this role, the 07-03 audit found TT-move
  COMPETES rather than adds, and −6 solves off 36 promotions is within normal chaotic swing — not clean
  evidence either way.
- ⇒ The memory claim that TT-move was "tested and DEGRADED" is **retracted** — it was never executed.

---

## ★ The lesson, third occurrence this week

**Byte-identical output with a gate OFF proves the gate is off — NOT that the feature works when ON.**
And a feature can be *reachable*, *fire billions of times*, and still be a no-op because it guards an empty
set. Only a counter on the **taken** branch distinguishes these. Confirmed sound: the fingerprint IS
sensitive to qsearch (`MAX_QDEPTH=4` → 245 solves / 40.1M nodes / EBF 3.842), so the
"byte-identical ⇒ never ran" rule survives — it just doesn't mean "harmless".

---

## STATE

- HEAD still `029f619`. **Nothing committed.** Working tree: `search_engine.cpp/.h` (TT_MOVE rewire + store
  gating, `TT_MOVE_POLICY`, counters `g_qcheck_d0_skipped` / `g_tt_move_promotions` /
  `g_q_quiet_checks_added`), plus `diagnostics/show_counters.py` (new).
- **Byte-id verified twice after both builds: 249 / 38,840,709 / EBF 3.934.** All new code inert with gates off.
- ⚠️ `/tmp` is wiped aggressively — counter `.err` files vanish between calls. **Chain the run and the counter
  read in ONE command** (`bash '<runner>' wac tag KNOBS; bash '<runner>' pyrun diagnostics/show_counters.py ...`).

## NEXT (ordered)

1. **Read `qcheck_pair_0727`** (the overnight tournament) — the only Elo-relevant number pending.
2. **Verify `[tt_move] promotions>0`**, then bench TT_MOVE (WAC/nodes) and, if live, consider games.
3. **Decide the quiet-check fix properly.** The mask path is a workaround; the *root* fix is `update_state`
   taking `bool& turn`, or the call site passing `!turn`. Gate it, byte-id it, and prefer the root fix so the
   simulate path (which catches discovered checks the mask misses) becomes usable.
4. **Instrument strategy** — with no screen and ±29-37 Elo per tournament against a +5-20 Elo queue,
   sequential testing (SPRT with a stopping rule) probably buys more per hour than another metric attempt.
5. **PST/attack-map ordering tiebreaker** — deliberately NOT built; design still open for discussion.
6. ProbCut (best cross-engine warrant); corrhist re-siting (gate on the pawn-structure signal test).
