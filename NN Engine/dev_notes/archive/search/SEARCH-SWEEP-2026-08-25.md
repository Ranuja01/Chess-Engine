# Search sweep + corrhist closure (2026-08-24 → 08-25)

Entry state: `SESSION-HANDOFF-2026-08-24.md` recommended pivoting to SEARCH/corrhist after the KS+passer arc
produced 3 diverse-game nulls. Baseline throughout: **`243 / 31,764,817 / EBF 3.729 / STS 1703`**
(verified byte-identical after every edit below; nothing shipped, nothing committed).

---

## 1. CORRHIST — LANE CLOSED (all three pre-registered kill criteria fired)

Built the offline gate first (`corrlog` / `corrsignal` runner subs, `_corrlog_capture.py`, rewritten
`diagnostics/corrhist_signal.py`) and captured **615,162 `[CORRLOG]` records** over 200 corpus positions at
`CORRHIST_LOG_STRIDE=1`.

| criterion | threshold | measured |
|---|---|---|
| REPLAY NET vs GLOBAL (best over weight sweep) | ≥ +5% | **−0.46%** |
| ORACLE NET vs GLOBAL | ≥ +10% | **+8.85%** |
| some key family clears GLOBAL | — | all five identical |

**The finding: the residual is POSITION-LOCAL, not STRUCTURAL.** All five key families score within 0.22pp
(pawn +8.90 · minor +8.93 · major +8.98 · nonPawnW +9.01 · nonPawnB +9.12). Independent structural
information would differ; identical scores mean every key is merely a proxy for "which position am I in."
Combining keys is *worse* (pawn+minor+major +4.22) through coverage collapse 93.9% → 47.9%.
⇒ **multi-keying dies with it; the whole corrhist family closes, not just the pawn table.**

**The live table is beaten by NO KEYING AT ALL.** A single-slot running offset wins at every weight
(at the 192/256 default: keyed +6.46% vs global +8.23%). Cause = **sample starvation**: 80.8% of the table
touched but median **5 updates/slot**, only 6.2% reaching the ~64 a shift-6 EMA needs. Scaling the corpus 10×
moved p50 from 2 → 5 while distinct keys grew proportionally ⇒ "just log more" is refuted.
⇒ This explains the old flat STS 1630-vs-1629: the table was injecting noise where it should have corrected.

Also measured NEGATIVE: Weiss bound guards (+1.76 vs +3.21 unguarded), depth-weighted update (+2.23 vs
+3.21). Faster EMA helps monotonically (shift 4 +4.55 → shift 8 +1.80) — the signature of under-sampling.

★ **METHOD (reusable): THE BASELINE IS THE WHOLE BALLGAME.** A shrunk per-key mean absorbs the GLOBAL bias,
so on skewed data ANY partition "reduces MAE" using no structure. Always add a single-bucket GLOBAL arm and a
POSITION-INDEPENDENT random arm. The script's pre-existing "control" was `hash((pawn_key*2654435761)&0xffff)`
— a COARSENED PAWN KEY, i.e. not a control at all. ★ **ORACLE vs REPLAY**: an upper bound can look healthy
(+8.85%) while the actual update rule cannot reach it; the gap IS the diagnosis.

---

## 2. `ENABLE_NODE_TT=0` was STARVING two features (the productive find)

`store_node_tt()` opens with `if (!Config::ENABLE_NODE_TT) return;` and is the **only** store that fills
`TTEntry::move` under the node's **own** key — every other store runs in the PARENT's frame, where the child's
best move is out of scope (its own comment says consumers are "starved by construction"). Default is 0, so:

- **`ENABLE_TT_MOVE`** measured `promotions=0` on a full d10 search, byte-identical to baseline. (This flag had
  now looked dead through TWO fixes: it first read `g_ttMoveTable`, which has **no write site anywhere**;
  repointed at `TTEntry::move`, which is itself never written unless NODE_TT is on.)
- **`ENABLE_SINGULAR`** with NODE_TT=1 fires **2,095,753** times (`eligible=134,377,496 gatepass=3,809,812`).
  Every prior singular measurement was of a dead mechanism.

**Both still lose once fed.** Singular: +9 WAC at fixed depth (252) collapses to **−1 at equal nodes** with
STS −56 ⇒ a WAC MIRAGE, the whole gain was its +11.2% extra nodes. TT-move ordering: −3 WAC vs its correct
control (NODE_TT alone), matching the 07-03 ordering audit; our history/killer/counter ordering already
reaches 83-86% first-move-cutoff, so the TT move is largely redundant.

★ **A FEATURE FLAG IS NOT A FEATURE — find the STORE site, not the read site.** Byte-identical-when-ON is the
tell. ★ **MEASURE NODE-INCREASERS AT EQUAL WORK** (`NODE_LIMIT=N`): fixed depth is biased FOR extensions
exactly as it is biased AGAINST node-reducers, and it killed singular in one run.

---

## 3. ☠️ SEARCH CHANGES ARE ANTAGONISTIC, NOT ADDITIVE (two 2×2s, both failed)

Owner proposal: single candidates land ~+7 Elo needing ~13,000 games; stack several into a bundle big enough
to resolve in 24h. Arithmetic sound, premise false.

**2×2 #1 — static ordering × `LMP_BASE=0`:** corner **245**, worse than EITHER part (252, 249); the −6.3% node
saving vanishes entirely (32.61M ≈ static-alone). Mechanism: static ordering's value IS the tail (its
first-move-cutoff barely moves, 83.77 → 84.42%) and LMP deletes that tail ⇒ they compete for one resource.
**Better tail ordering does not earn harder tail pruning — it makes the tail MORE worth keeping.** This
refutes the "ordering buys INDEX-KEYED pruning" thesis that `OPTIMIZATION_LOG` (07-30) flagged as untested.

**2×2 #2 — static ordering × `ENABLE_NODE_TT`** (chosen because the subsystems are unrelated): corner
**STS 1687 — BELOW the 1703 baseline**, though both parts are STS-positive alone (+34, +38).

⇒ **Independence of SUBSYSTEM ≠ independence of EFFECT.** Hypothesis: the search's tuned constants (LMR/LMP
index thresholds, RFP/futility margins, reduction schedules) are **co-adapted to the current move-ordering
distribution**; one change is partly absorbed, two move the distribution beyond where the constants fit. This
is the search-side analog of eval degeneracy. ★ **ALWAYS RUN THE 2×2 BEFORE BUNDLING** (~2 runs, the solo
cells are usually already measured). ⚠️ UNTESTED escape: re-tune the constants FOR the combined regime.

---

## 4. Static ordering — bench-positive, GAMES NULL (and an over-call to learn from)

`ENABLE_STATIC_ORDER=1 STATIC_ORDER_HIST_MAX=1` (NOT the default 0, which applies only where `|history| <= 0`
and is near-inert). A **re-screen of a parked positive**, not a new idea: built 07-28, mis-called a failure
07-29, retracted 07-30 (+8 STS, the only non-negative arm that session), queued for re-screen 08-15.
☠️ Distinct from the separate broad "PST ordering" null (23M fires, +0.02pp FMC).

Bench: **WAC 252 (+9) · STS 1737 (+34) · 214 vs 212 at equal node budget (using FEWER nodes) · +2.7% nodes.**
The only arm all session positive on all three. Headroom: 1.4 eligible quiets per main node, **140,692 fires
per search**, 64.8% non-zero ⇒ broad, not fragile.

| segment | seed | result | Elo |
|---|---|---|---|
| s1 | 7 | +151 −145 =104 | +5.2 ± 40 |
| s2 | 23 | +158 −148 =94 | +8.7 ± 40 |
| s3 | 59 | +160 −145 =95 | +13.0 ± 40 |
| s4b | 83 | +130 −159 =111 | **−25.2 ± 40** |
| **POOLED** | — | **+599 −597 =404 / 1600 (50.06%)** | **≈ +0.4, CI ±20 — PRACTICAL NULL** |

☠️ **After 3 positive segments this was called "+9.0, 3-for-3, the most consistent evidence any search
candidate has produced," with an extrapolated ~8,400-game SPRT acceptance.** The 4th ran −25.2. Pooled LLR is
**−0.22** (+0.061 +0.134 +0.229 −0.640), drifting toward REJECT, so the acceptance extrapolation was invalid —
it projected a trend from the favourable subset. The 12.5%-by-chance figure for 3 same-sign samples was quoted
and then ignored. Same error as the 08-18 "+23/390g" call a clean segment reversed.
★ **3 same-sign 400g segments is a coin-flip artifact, not evidence. LLR is ADDITIVE — quote the RUNNING SUM
over ALL segments, never a slope.**
⚠️ The bench gains were real and reproducible and **did not convert** — proxy anti-correlation landing on the
one candidate we had become confident about.

---

## 5. Everything else screened (all vs baseline 243 / 1703)

| arm | WAC | nodes | STS | verdict |
|---|---|---|---|---|
| `ENABLE_NODE_TT=1` | 245 | +1.3% | **1741 (+38)** | neutral at equal work (211 vs 212) |
| `NODE_TT`+`SINGULAR` | 252 | +11.2% | 1647 | MIRAGE (−1 at equal work) |
| `LMP_BASE=0` | 249 | **−6.3%** | 1653 | games **+2.6 ± 40** — node-reducer precedent did NOT transfer |
| `ENABLE_THREAT_HIST=1 SHIFT=3` | 245 | +5.0% | 1593 | MIRAGE |
| `ENABLE_THREAT_HIST=1` (SHIFT=0) | 240 | +3.3% | — | loses |
| `ENABLE_CONT_HIST_2PLY=1` | 242 | +1.5% | — | loses (header already recorded this) |
| `ENABLE_IMPROVING=1` | 239 | +2.5% | — | loses |
| `ENABLE_LMP_HIST_EXEMPT=1` | 241 | +1.2% | — | loses |
| `ENABLE_SEE_PRUNE=1` | 240 | −7.1% | — | parked (node-reducer, accuracy loss) |
| `ENABLE_CHECK_ORDER=1 BONUS=1500` | 245 | +4.6% | — | mirage profile |
| `FUTILITY_EVAL_MODE=1` / `=2` | 241 / 236 | −1.9% / +4.2% | 1575 / — | both closed |
| `FUTILITY_MARGIN_SCALE` 50/150/200 | 240/241/241 | all worse | — | **owner's 100 is OPTIMAL** |

**Futility is HEALTHY, not dormant** (fires 86-88% of the moves it examines; new `[futility]` counters).
Its eval-cost lane is closed: `get_board_evaluation` is `searchEvalCache`-backed, so "extra" evals are mostly
PROBES and a cheap surrogate forfeits cache population (`MODE=1` = −0.7% NPS). Our futility evaluates the
**CHILD** (more accurate and costlier than SF's parent-eval rule) ⇒ a parent-eval rewrite is a TRADE with no
speed to reclaim. Margins are non-monotonic (150 worse than both 100 and 200) ⇒ step-shaped, don't re-sweep.

---

## 6. ☠️ RAZORING — lead raised, then LARGELY REFUTED (see §6b for the closure)

From the `Config` header (search_engine.h ~243-252): `RAZOR_BASE_FIRST=750 RAZOR_FLOOR_FIRST=200
RAZOR_BASE=300 RAZOR_FLOOR=100 RAZOR_DECAY_PCT=75`, with the comment:
> *"These are absolute millipawn margins, so they are coupled to eval magnitude — **they were calibrated
> before the de-king change shrank attack-position evals.**"*

And the existing `[razor_audit]` diagnostic, from two incidental depth-10 runs:
```
iters_after_razor=12  winner_was_razored=5  pct=41.7%  avg_depth_past_razor=6.2
iters_after_razor=19  winner_was_razored=5  pct=26.3%  avg_depth_past_razor=6.2
```
⇒ **in 26-42% of iterations following a razor, the move that eventually won had been razored away.**

This is categorically better than anything screened above: an **ACTIVE, default-ON mechanism** with
constants its own comment flags as stale, a built-in diagnostic, a measured defect, and a clear fix direction
(rescale the absolute margins to post-de-king eval magnitude). Same shape as the NODE_TT find — the only
genuinely productive discovery of the sweep. ⚠️ n=2 positions; **first step is widening the measurement across
the WAC suite** to get a real wrong-razor rate before touching any constant.

Second lead from the same read: **`ENABLE_LMR_REMDEPTH`** (off) — LMR keys on ITERATION depth, not remaining
depth, so near the horizon a constant reduction wipes the child into qsearch; the header cites measured
wrong-reduction rates of **4.1% at L6 / 6.2% at L8** vs ~1% at L1-L5, with the fix already built and gated off.

⚠️ Constraint on both: `STATSCORE_OFFSET/DIVISOR` (0 / 683) are "one atomic unit WITH gravity" and are
invalidated by any change to history scoring — they must be re-derived if the history tables move.

---

## 6b. Razoring + LMR_REMDEPTH — BOTH CLOSED
**Razoring.** Real wrong-razor rate is **17.65%** (604/3422 across WAC), not the 26-42% the n=2 sample showed
— measuring before touching a constant was correct. Razoring EARNS its nodes: `ENABLE_ROOT_RAZOR=0` costs
**+23.9% nodes** for +4 solves, with **STS −56**. ☠️ **The MARGIN is not the controlling variable**: doubling
`RAZOR_BASE`/`FLOOR` moved the rate only 17.65 → 15.81% and fires −1.8%, so the razor fires on gaps far
beyond either threshold and the "stale absolute margins" hypothesis yields NO lever. ★ Reinterpretation: the
17.65% are **iteration-to-iteration SEARCH INSTABILITY** cases (a move's score jumping between iterations),
not calibration cases. Any future attempt should target the instability (re-verify a razored root move whose
score is stale), not the margins.

**`ENABLE_LMR_REMDEPTH`.** Re-index = 244 solves / **+13.1% nodes** (pure de-aggression, and de-aggression is
on record as paying nothing: `LMR_MIN_REM=4` = WAC +1 / STS −8; `LMR_EXTRA=2` ≈ −55 Elo). The dial meant to
enable the corner test is **CLAMP-SATURATED**:
`red = (rem - DEPTH_REDUCTION[rem]) * SCALE / 100; red = clamp(red, 0, max(rem-2,0));`
SCALE=150 is **byte-identical** to 100 (already at the cap); only SCALE<100 is reachable (50 → 249 solves /
+36.9% nodes). ⇒ the corner needs raising the `rem-2` clamp — **but that clamp IS the fix** ("a reduction may
never consume the child's last ply"). Closed on mechanism, not measurement.
★ **A knob can be correctly REGISTERED and correctly WIRED and still be inert because a downstream CLAMP eats
it. Sweep BOTH directions before concluding a knob is live.**

---

## 7. 🔥 MOVEGEN — the share is real, the lane is closed
Profiled with the built-in `EVAL_PROFILE` counters (`build_profile` → new `evalprofile` / `evalprofile_mid`
runner subs, which sum the per-search blocks; PROF counters RESET per `get_engine_move`).

| block | WAC tactical | midgame corpus |
|---|---|---|
| eval terms | 52.8% | 56.1% |
| **MOVEGEN** | **36.0%** | **35.3%** |
| makeunmake | 7.6% | 6.4% |
| TT probe | 3.7% | 2.1% |
| `MG_ISSAFE` / `MG_PSEUDO` (of total) | 12.6% / 6.5% | 10.7% / 5.1% |

⇒ `SEARCH-INFRA-FLAVORS` §0 calls movegen a **"confirmed non-lever ... a few %"** on the premise that
"eval = 65-84% of node cost". **Both halves are wrong**: eval is 53-56%, movegen is ~35%, confirmed on BOTH
workloads. `generateLegalMovesPre` runs **13.8-14.7× per node** (once per mask pair) and per-move `is_safe`
costs ~2× the generation it filters.

☠️☠️ **BUT THREE TARGETED FIXES RETURNED NOTHING** (all verified NODE-IDENTICAL, `243 / 31,764,817 / 3.729`):

| change | result |
|---|---|
| attack-cache memo (`attacks_mask` per node) | **−1.7%** NPS (−3.6% with `thread_local`) |
| direct-emit (generate into caller's arrays, compact in place) | ~0-3%, inside noise |
| `is_safe` fast path (avoid the 16-param call for non-king/non-ep moves) | **−1.9%** |

★★★★ **WHY — DON'T HAND-OPTIMIZE AGAINST `-Ofast -march=native -flto`.** `is_safe` is `inline`, its args are
loop-invariant, and the mask-pair passes live in ONE translation unit, so the compiler had **already** hoisted
the marshalling and kept `attacks_mask` in registers. Each "optimization" removed an imaginary cost while
**adding a `Config::` branch the compiler cannot fold away**. ⇒ **A PROF cycle share tells you where TIME
goes, NOT what is REMOVABLE.** Only honest routes left: ALGORITHMIC (pin-aware legal generation, which removes
work rather than rearranging it) or read the generated ASM first.
🧹 All three were **REMOVED, not left gated** — each sat on a per-piece / per-move / per-call path where a dead
flag check is pure cost. Kept instead: `PROF_MG_PSEUDO`/`PROF_MG_ISSAFE` (compiled out without `-DEVAL_PROFILE`)
and `FUTILITY_MARGINS_EFF` (scale folded once at init, removing an integer DIVIDE from a ~13M-times path).

☠️ Also SIZED-AND-REJECTED the same day, both via probes already in the code:
- **pawn-king eval cache** — `PAWN_PPINC` is only 5.9% of eval ≈ **3% of total**; the doc ranked it lever #4.
  ⚠️ `PAWN_ATKLOOP` reads 0 calls, so the pawn-eval PURITY question (doc says pure, our memory says impure via
  `attackingLayer`/blockers/phase) is still UNRESOLVED.
- **capgains lazy skip** — `CAPG_LAZY_PROBE`: `t0=15.9% (tiny 100%)`. capgains is 9.5% of total, so a
  VALUE-PRESERVING skip caps at ~1.5%, less in practice (tension is a BYPRODUCT of the function, so the real
  gate must be a cheaper "no captures at all" pre-test ⊂ t0). The tempting 3.5% version is a DAMPING = the
  `CAPG_*` family whose best arm is **−65 Elo**. Rejected.

## 8. ⚠️ TWO INSTRUMENT FAILURES THAT COST REAL TIME
**(a) The "26% NPS regression" that wasn't.** Owner flagged NPS should be ~450k; `wac_speed` read 366k. I
theorised node-mix drift and bisected the SEE-captures ship (which buys back only +3.6%) before checking the
obvious: the historical figure came from **`depth_nps_bench.py` on a stratified MIDGAME corpus**, whose
docstring explicitly skips tactical shots as *"not representative of game speed"*. Measured on the ORIGINAL
instrument: **median 450,201** — owner exactly right, no regression. Same-instrument drift since 07-28 is only
**−9%**, paid for by five shipped Elo gains. ★★★★ **MATCH THE INSTRUMENT BEFORE DIAGNOSING A REGRESSION.**

**(b) `NODE_LIMIT` equal-work is STEP-SHAPED.** `ASPIRATION_DELTA=750` produced the best equal-work reading of
the session (**220 vs 212 at 100k, using 2.5% fewer nodes**), backed by a mechanism measured BEFORE the sweep
(`[aspiration]` counters = **248 windows / 126 fails ≈ 51%**, far above the doc's healthy 10-30%, so widening
was the predicted fix — and the doc's "too wide ⇒ rarely fails" premise is simply WRONG for us). Then its own
prediction failed:

| budget | baseline | `DELTA=750` | gap |
|---|---|---|---|
| 50k | 194 | 194 | 0 |
| 100k | 212 | 220 | +8 |
| 200k | 234 | 230 | −4 |
| fixed depth | 243 | 242 | −1 (+2.4% nodes) · STS 1706 vs 1703 (flat) |

⇒ **the +8 was ONE LUCKY CAP.** Truncation is a step function per position, so sliding the budget flips a
handful of positions and moves the count several points. The metric is perfectly DETERMINISTIC — which makes
it feel authoritative — but carries **±8 solves of cap-placement artifact** at a single budget.
★★★★ **ALWAYS SWEEP 2-3 BUDGETS; a single-budget equal-work delta under ~10 solves means nothing.** This
undercuts (without reversing) the single-budget equal-work calls in §2 and §4 — singular still dies on
STS −56 / +11.2% nodes, static ordering was settled by 1600 games.
✅ **Aspiration = NULL**, but fix the doc's fail-rate premise if it is ever revisited.

## 9. 🎮 THE TWO ODDS LOSSES — walked at last; the passer premise is REFUTED
Queued since 08-24. `_pgn_walk.py SIDE=black DEPTH=12` on both. **Our static eval is FLAT across the entire
collapse in both games** while SF swings ~1300cp:

| game | SF trajectory (White-POV) | our static eval (Black-positive) |
|---|---|---|
| knight odds | −398 → **+891** | +4.39 → +6.42 |
| rook odds | −463 → **+817** | +9.62 → +16.36 → +13.05 |

Decisive plies:
- **19...Kc8** `Rn1r2nr/1kp1q1pp/3bp3/1P1p1b2/Q2P4/8/4BPPP/2B2RK1 b - - 1 18` → `ours +5.60, KS −2.70` while
  SF has **White +8.9** (~14 pawn gap), with Ra8+Qa4+b5 = a mating attack. KS fires but is swamped by ~+8.3.
- **22...Qxc6** `r3kb1r/pPpRpppp/2P3q1/8/Q3b3/1P2B3/P3KPPP/8 b kq - 0 21` → `ours +13.05, KS +1.62` while SF
  has **White +2.3** (~15 pawn gap), with b7/c6 promoting (`bxa8=Q+` next). ☠️ KS reports Black's king SAFER.

★ **Diagnosis: MATERIAL OVER-VALUATION SWAMPING THE REFUTATION.** The refuting mechanism differs (mating
attack vs promoting passers) but the cause is identical — nothing damps the material term. The owner's
contemporaneous "lack of passer danger response" read was right about game 2's *mechanism* but not the cause;
passer detection would not have saved either game.
⚠️⚠️ **REGIME-SPECIFIC — DO NOT REDIRECT THE EVAL PROGRAM ON THIS.** Normal-corpus mean|gap| vs SF11 for
Material is **1.125 pawns**; here it is **10-15 pawns**, an order of magnitude worse ⇒ extreme material
imbalance is far outside the range the eval was fitted on (cf. the dormant `material-edge-overvaluation`).
Two samples from the one regime our eval handles worst are not evidence about general playing strength.
▶️ If the odds losses are wanted as their own goal, the machinery exists and is disabled (`REALIZ_MAT_K`,
`REALIZ_MAT_THRESH`, `REALIZ_PHASE_K`, the `MOD_*` modulators) — but "exists and is off" is not evidence it
works, and that family's best measured arm is −65 Elo.
★★★ **A CONTEMPORANEOUS READ OF A LOSS IS A HYPOTHESIS, NOT A FINDING** — it survived ~10 days as the premise
for a whole lane and the first actual walk refuted it in one run.

## 10. ▶️ WHERE TO GO NEXT
**Nothing shipped in two days; ~two dozen arms closed.** The cheap-screen well is demonstrably dry — every
default-off flag and every active constant we could sweep is either null, mirage, clamp-inert, or below the
measurement floor. What we gained is a much sharper map of WHY each closed, plus eight reusable method rules.

Remaining leads, ranked, with the mechanism that justifies each:

> ☠️☠️ **RANKING SUPERSEDED LATER ON 08-25 — leads 1 AND 2 ARE NOW CLOSED. See §12.** Both were closed
> CHEAPLY (no C++ shipped, ~6 runs), and lead 1 was already closed IN THE RECORD before I ranked it, which
> is the failure this document should be read for. **Start at §12, not here.**

1. ~~**Passer DETECTION + the DEFENSIVE/enemy-passer side**~~ ☠️ **CLOSED — see §12.1.** Both halves were
   already BUILT, and detection had already run **1600 games (+2.6, below-floor null)** per
   `SESSION-HANDOFF-2026-08-24.md` §2. The "enemy side was NEVER BUILT" premise was simply false.
2. ~~**Pin-aware legal generation**~~ ☠️ **movegen CLOSED — see §12.2.** The 4th idea (staged/lazy gen) was
   sized at ≈10% of search CEILING; pin-aware's own "~30% NPS" estimate is refuted by implication.
3. **The active-untuned knob inventory** — ▶️ **NOW THE TOP LEAD.** Only 2 of ~15 groups checked (futility
   margins ALREADY OPTIMAL; razoring dead). Screen as a **PORTFOLIO** rather than one 24h SPRT per +5: these
   are largely NOT co-adapted search constants, and the anti-additivity law is scoped to those — result-
   identical and correctness/symmetry classes have both composed cleanly before. 2×2 the top pair before
   bundling; gated on the ±8 equal-work caveat (§8b).
4. **Search instability at the root** (§6b) — the razor's 17.65% wrong-razors are score-jump cases; a
   stale-score re-verification is a different shape from anything tried. Speculative.

☠️ **Do NOT re-attempt without reading the record**: corrhist (any keying), singular, TT-move ordering,
threat-hist, conthist-2ply, improving, LMP variants, futility eval-mode/margins, razoring margins,
`LMR_REMDEPTH`, aspiration DELTA, the three movegen micro-optimizations, pawn-king cache, capgains skip.

## 11. Housekeeping
- New/changed, all default byte-identical: Stage-0 corrhist eval reuse (reuses the raw node-entry eval at the
  update site instead of recomputing), `[futility]` + `[corrhist]` counters (behind `ENABLE_PRUNE_LOG`),
  `FUTILITY_MARGIN_SCALE` + `FUTILITY_MARGINS_EFF` (precomputed at init — removed an integer DIVIDE from the
  hot gate), `generateMinorKey`/`generateMajorKey`/`generateNonPawnKey` (cache_management.h), extended
  `corrhist_log`, `PROF_MG_PSEUDO`/`PROF_MG_ISSAFE`, and runner subs `corrlog` · `corrsignal` · `probe` ·
  `razoraudit` · `evalprofile` · `evalprofile_mid`, plus `diagnostics/_corrlog_capture.py` and a rewritten
  `corrhist_signal.py` (its old "random control" was a COARSENED PAWN KEY — not a control at all).
- 🧹 **REMOVED rather than left gated**: the three movegen experiments (§7). A dead `Config::` flag on a
  per-piece/per-move path is pure cost. ★ **A probe or experiment on a hot path must be compile-time-excluded
  or hoisted out of the loop — never a runtime flag check.**
- ✅ Final verification after cleanup: **`243 / 31,764,817 / EBF 3.729 / STS 1703`**, all four suites.
  Nothing committed, nothing shipped, every new flag defaults off.
- ☠️ **Never edit `overnight_runner.sh` while a job launched from it is running** — bash reads scripts lazily
  by byte offset, so the running instance dies with a bogus syntax error at an unrelated line.
- ☠️ **Machine suspend voids fixed-TIME segments** (owner is remote): clocks distort for in-flight games.
  Discard any straddling segment, relaunch on a fresh seed under a NEW TAG, and TaskStop the stalled job first.
- ⏸️ Still queued and never run: `_pgn_walk` of `diagnostics/_odds_knight_loss.pgn` and `_odds_rook_loss.pgn`.

---

## 12. ▶️ LATER ON 08-25 — §10's TOP TWO LEADS BOTH CLOSED (~6 runs, no C++ shipped)

### 12.1 Passer detection + defensive side — CLOSED, and it was ALREADY closed in the record
Ran the evidence-first step §10 itself demanded. In order:

| check | result |
|---|---|
| detection gap on **game** data (`passer_detector_diff.py`, `game_regret_set.csv`) | **2,003 SF-only passers, 0 ours-only** ⇒ strict subset, **11.2%** of SF's passers missed, concentrated rank 5 (6.6%) / rank 6 (7.2%) |
| move-relevance of the missed pawns (`_passer_miss_profile.py`, NEW) | SF's best move touches a missed passer's square/path **27.2%** — against a **28.0% CONTROL** (already-detected passers). **NO EXCESS.** |
| criticality conditioning | mild rise 10.6%→18.1% benign→moderate, then **DOWN to 12.1% in the CRIT band** (n=520) ⇒ no concentration |
| owner split | 51.1% side-to-move / 57.8% opponent / only 8.9% both ⇒ one-sided per position (so no within-position cancellation) |
| symmetry gate, `PASSER_ENEMY_CREDIT_PCT=100` | 11 violations / 1.4% colour — **identical to baseline**; no defect |
| `_move_change_arms.py ARM=PASSER_ENEMY_CREDIT_PCT=100` | **6.0% / 4.2% ≥10cp** (control `ENABLE_THREATS=0` = 13.2% / 9.2%) |

☠️ **Then I checked the record instead of the header, and it inverted the lane.**
- The **DEFENSIVE side is fully BUILT** in `boost_pieces_for_supporting_passed_pawns` — it walks the enemy
  passer's promotion path accumulating blockade + path-attack credit for the defender. It is correctly
  signed, and its two colour clamps DO mirror (`min(-pawn_rank_bonuses[r],·)` / `max(·)` are proper twins
  because `pawn_rank_bonus` is negative for White pawns and positive for Black — **verified, not assumed;
  my initial "asymmetry bug" suspicion was wrong**).
- It is **deliberately zeroed**, and that zeroing is "Gap-P P1", a component of the **6-knob collapse bundle
  that shipped +38.7 ±27 Elo over 875 games (2026-06-27)** — the notes call it the bundle's TOP contributor,
  nearly parked on its −106 STS. `external-play-gaps.md:93`: *"=0 fully removes the wrong-sign."*
  `passed-pawn-subsystem-map`: *"path-attack EXISTS but as independent re-credit; defender half disabled"*
  ⇒ it re-credits an obstruction `getPPIncrement` has ALREADY docked = collinear double-count.
  ⇒ **Turning it back on is an UN-FIX, not an untested lane.**
- **DETECTION was built on 08-22** (`ENABLE_PASSER_DETECT_SF` + `PASSER_CANDIDATE_DOCK`) and already ran
  **1600 games = +2.6, a below-floor null** — recorded in `SESSION-HANDOFF-2026-08-24.md` §2, the document
  §10 was written from.

★★★★ **THE LESSON, and it is about this document**: I ranked as the #1 lead something my own prior handoff
had already spent 1600 games closing, because §10 was written from a summary rather than a grep. **THE
HEADER IS NOT THE RECORD applies to MY OWN documents.** Grep `dev_notes` + memory before RANKING a lead,
not just before implementing one.
★★ **A large move-change rate is a resolution bound, NOT a direction.** 6.0% is among the biggest any arm
produced across the whole sweep — and it belongs to a change known to lose ~38 Elo.

✅ **By-product: the first calibration of the D7 regret ruler against a known-Elo-signed change**
(`_ks_regret_score.py`, extended with an env `ARMS=` override so it need not be forked per subsystem):

| arm | HELD regret | Δ | HELDcrit (**n=60**) | Δ |
|---|---|---|---|---|
| base | 2.5971 | — | 3.3280 | — |
| `credit50` | 2.6273 | +0.030 | 3.2816 | −0.046 |
| `credit100` | 2.6959 | +0.099 | 2.4410 | **−0.887** |

**Aggregate PASSES** — monotone, correct sign vs 875 games, a dose-response (still only n=1 change: a pass,
not a proof). **The CRITICALITY SPLIT FAILS** — largest magnitude on the table, pointing at the −38 Elo
change, on **n_crit = 60**. ⇒ **QUOTE `n_crit` BESIDE EVERY CRITICALITY READ; never adjudicate at n≈60.**
Same family as the `NODE_LIMIT` step function and the 3-same-sign-segments streak: **deterministic ≠ powered.**

### 12.2 Movegen — the 4th idea (STAGED/LAZY generation) sized and closed
87.5% of cutoff nodes cut on move 0, so generating the full list everywhere looked mostly wasted.
Instrumented actual consumption with `#ifdef EVAL_PROFILE` counters (compile-excluded, never runtime-gated)
at the cache-MISS generation site and both main move loops; surfaced via `probe`. Node-identical
(243 / 31,764,817):

```
gen_calls=2,061,579   gen_moves=42,413,263    (20.6 generated per generating node)
loop_nodes=2,712,870  seen_moves=33,016,044   (12.2 examined per looping node)
```

★ **I PREDICTED "consumption well under half" AND WAS FALSIFIED** — the prediction was registered before the
run, which is the only reason it could be. Raw `seen/gen` = **77.9%**; netting out the 651,291 nodes that
looped on a **cache hit** without generating (24% hit rate) gives ≈**59% consumption / 41% waste**.
Ceiling = 0.41 × `MG_GEN` 24% ≈ **10% of search**, before staging's own overhead and before the quiets-only
restriction ⇒ realistically 5-8% NPS ≈ 5-8 Elo, **below the floor for a major refactor of the ordering path.**
- ★★★ **THE MOVE-GEN CACHE IS PART OF THE BASELINE, NOT PART OF THE PRIZE** — omitting it from the
  denominator inflates the lever by ~19pp.
- ⚠️ **Unreconciled**: 87.5% first-move cutoffs should imply far more waste than 41%, implying cut-nodes
  disproportionately hit the cache and never pay generation. Needs a per-node join; can only move the prize
  DOWN, so the closure is robust.
- ⚠️ Does not *formally* close pin-aware generation, but its "~30% more NPS" estimate is refuted by
  implication (`MG_ISSAFE` wraps the copy loop, and direct-emit already showed that traffic is ~1%).

### 12.3 Where that leaves the program
▶️ **The active-untuned knob inventory (~13 of ~15 groups) is now the top lead**, screened as a **portfolio**.
Owner's framing, which the record supports: *lots of little wins can still add up*. The anti-additivity law
(§3) is scoped to **co-adapted search constants** — it was never measured on **result-identical speed work**
(nodes pinned ⇒ composes exactly) or **correctness/symmetry fixes** (the 7-fix bundle composed free). So the
right shape is: cheap screen across all groups → **2×2 the top pair before bundling** → ONE tournament on the
survivors, rather than a 24h SPRT per +5 Elo.
🧰 New this pass: `diagnostics/_passer_miss_profile.py` (missed-passer profile, WITH its control) and an
env `ARMS=` override on `_ks_regret_score.py` (extends the canonical tool instead of forking it).
