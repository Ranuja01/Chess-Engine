# Session handoff — 2026-07-26/27

## ★ TWO SHIPS (committed, verified)

| commit | what | evidence |
|---|---|---|
| `e02219e` | **de-king** — `KS_ZONE_ATTACK_PCT` 100→50 | 3 seeds × 200g, score 40.1→47.5% (up ALL seeds), positional collapses −22% ⇒ ≈ +50 Elo |
| `029f619` | **per-move qsearch futility** — `ENABLE_QDELTA_PERMOVE=1`, `QDELTA_PERMOVE_MARGIN=1500` | **456 games, +53.8 ±37.5 Elo**, colour-symmetric |
| `2487145` | guard fix + prune instrumentation | byte-identical |
| `8ca657d` | gated scaffolding | byte-identical |

**BYTE-ID REFERENCE: WAC 249 / 38,840,709 / STS 1662.** `ENABLE_QDELTA_PERMOVE=0` reproduces the old
248 / 44,038,704 exactly.

The qsearch prune replaced a **node-level** delta prune that skipped EVERY capture at a node (including a hanging
queen) with SF's per-move form crediting the victim: `static_eval + margin + victim <= alpha`. It is the most
CONSERVED schedule across SF11→SF18.

---

## ★★★ THE MEASUREMENT CRISIS AND ITS RESOLUTION (the session's main output)

Nothing we owned predicted Elo. Resolved in two parts.

### 1. The RULER: use fixed NODES
| ruler | deterministic | fair to node-savers |
|---|---|---|
| fixed DEPTH (`wac`/`sts` d10) | ✅ | ❌ savings handed back |
| fixed TIME (LIGHTNING) | ❌ **σ≈1.0** | ✅ |
| **fixed NODES** (`NODE_LIMIT=250000 PRESET=LONG_FORMAT MAX_DEPTH=64`) | ✅ **exact** | ✅ |

Two runs of one config at fixed nodes were **byte-identical**. At fixed time the same config gave 23.46 / 23.18 /
22.30 — and the qdelta comparison **flipped sign** between runs. ⇒ **all fixed-TIME cploss comparisons are void**,
including the "wide corpus gets qdelta backwards" claim (retracted — it was noise).
`node_ab` already used NODE_LIMIT for exactly this reason; we never applied it to cploss.

### 2. The STATISTIC: use the TAIL, not the mean
Default vs qdelta-OFF (a known **+53.8 Elo** gap), deterministic:

| | mean | >5% | >10% | **>20%** | **p99** |
|---|---|---|---|---|---|
| default | **23.82** | 14.6% | 5.1% | **0.5%** | **17.0** |
| qdelta OFF | **23.82** | 14.1% | 5.1% | **0.8%** | **18.4** |

**Identical means.** qdelta-OFF is slightly BETTER on small errors and clearly WORSE on catastrophes; they cancel.
⇒ **Elo is TAIL-driven; every metric we had (STS, WAC solves, cploss mean) is MEAN-like.** Explains the whole
week: STS +107/+93 ⇒ ~0 Elo; WAC +1 solve for +54 Elo.
**SELECT ON `rate>20%` and p99.** ⚠️ ~100 tail events ⇒ ±10 Poisson: resolves qdelta-class differences, NOT
+5-vs-+15 Elo. Games remain the arbiter; the screen decides WHICH candidates deserve games.

---

## Tooling built (all in `diagnostics/` + runner)
- **`build_cploss_wide.py`** — mines `games/*/tournament.pgn` (annotated `{ base +0.49/d10 }`) AND
  `games/*/game_*.jsonl`. **77,432 games / 5.1M decision points / 4.29M unique.** Tiers come from the ENGINE'S OWN
  eval trace, deliberately NOT from cploss (else selection inflates the tier and every new config gains by
  regression to the mean). Graded: `general / inaccuracy(≥0.4p fall) / mistake(≥1.0) / blunder(≥2.5) /
  found(≥0.75 RISE)`, 2000 each → `selfplay/tune_data/cploss_corpus_wide.csv`.
- **`cploss_corpus` runner sub** — `<corpus> [max_depth=64] [judge_depth=12] [sample=0] [shard] [seed=12345]
  [KNOBS]`. Defaults to the TIMED ruler; **pass `PRESET=LONG_FORMAT NODE_LIMIT=250000` for the correct one.**
- **`cploss_frozen.py`** — added `--sample` (fresh random subset; **seed FIXED by default so configs are
  comparable** — rotate deliberately BETWEEN campaigns), `--dump` (per-position losses), per-stratum + overall
  TAIL stats. SF judge cache is keyed by FEN@depth and **config-independent** ⇒ pay once per corpus.

---

## ★ FIVE UNREACHABLE FEATURES FOUND THIS WEEK (one was worth +54 Elo)
| feature | why it could never fire |
|---|---|
| `ENABLE_CONT_HIST` | helper returns before its code under the shipped statScore default |
| `ENABLE_QDELTA_PERMOVE` | guard tested `promotion == 0`; **move_gen pushes 1 for non-promotions** (use `<= 1`) |
| `ENABLE_HIST_PRUNE` | needs NEGATIVE history, which needs malus (off) |
| **`ENABLE_TT_MOVE`** | byte-identical when enabled — **memory claims it was "tested and DEGRADED", impossible for code that never runs** |
| **`ENABLE_QCHECK_DEPTH0`** | byte-identical when enabled |

**★ DETECTION RULE: byte-identical output across a knob change = the code never ran.** Identical node counts
across a swept *margin* is the same tell. **Never claim a prune's behaviour without a fire counter.**

---

## Reachability audit (15 flags, fixed-depth WAC vs base 249 / 38,840,709)
| flag | WAC | nodes |
|---|---|---|
| **`ENABLE_QCUT`** | **254 (+5)** | 39.64M |
| **`ENABLE_OTV`** | **253 (+4)** | 42.98M (+11%) |
| `ENABLE_PIECE_CONTHIST` | 249 (=) | 40.97M |
| `ENABLE_LMP_HIST_EXEMPT` | 248 | 38.78M |
| `QSEE_RESORT` / `IIR` / `THREAT_HIST` | 247 | — |
| `SINGULAR` / `CHECK_ORDER` / `THREATS` | 246 | — |
| `CAPTURE_HIST` | 243 | — |
| `CONT_HIST_2PLY` | 241 | — |
| **`ENABLE_PROBCUT`** | **237 (−12)** | 38.30M |
| `TT_MOVE`, `QCHECK_DEPTH0` | **INERT** | — |

⚠️ Fixed-depth is biased against node-CUTTING configs; only ProbCut and LMP_HIST_EXEMPT cut nodes, slightly.

---

## Baseline on the graded corpus (fixed nodes, n=9823)
```
OVERALL      25.25   >5%=15.7% >10%=5.2% >20%=1.0%  p95=10.2 p99=20.1
  blunder    22.56   >20%=1.0%     found      23.76  >20%=1.1%
  general    26.32   >20%=0.9%     inaccuracy 27.77  >20%=0.9%
  mistake    25.70   >20%=1.3%
```
**★ NO TIER IS TAIL-RICH** (0.9–1.3% everywhere) ⇒ no sampling shortcut; only VOLUME buys tail resolution.
n=9823 gives ~98 tail events, which is adequate. The tiers ARE diagnostically useful for means.

**★ ERROR PROFILE (new, and it inverts intuition):** the `blunder` tier has the **LOWEST** mean loss (22.56) and
`inaccuracy` the **HIGHEST** (27.77). **We recover where we once blundered badly and bleed persistently in quiet
subtle positions** — the positional weakness as a measurement, not an impression.

**Screened on the graded corpus (fixed nodes, n≈9830):**

| config | mean | `>20%` | p99 | games |
|---|---|---|---|---|
| base | 25.25 | 1.0% | 20.1 | — |
| **malus** | 24.53 | **0.9% (flat)** | 19.7 | **+10.9 ±36.6 n.s.** ✅ consistent |
| **QCUT** | 24.57 | **1.1% (flat)** | 20.6 | *predicted flat* |

**★ THE CONTROL WORKED: malus has a FLAT tail and games already said flat.** Malus and QCUT have near-identical
profiles (both improve the mean ~0.7, both leave catastrophes untouched) ⇒ **QCUT's +5 WAC solves are unlikely to
convert** — a 4-hour tournament avoided, IF the screen is trusted.

⚠️ **CAVEAT — the validation set is not apples-to-apples yet.** The differentiated case (qdelta OFF, 0.5% vs 0.8%)
was measured on the OLD corpus at n=1477; malus/QCUT are on the NEW graded corpus (baseline 1.0%). The pattern is
consistent but spans two sample sets. **Re-run qdelta-OFF on the graded corpus (~80 min) before leaning on this
screen for decisions.** Current status: **promising, one clean confirmation + one cross-corpus confirmation — not
proven.**

---

## Games results (neither shipped)
`ENABLE_HISTORY_MALUS` **+10.9 ±36.6** (478g) · `SEE_PRUNE_CAPTURES` **+9.5 ±36.7** (475g) — both n.s. despite
equal-time STS **+107 / +93**.

---

## NEXT (ordered)
1. **Finish the tail validation** — malus (running) then capture-prune. Flat tails on both ⇒ four consistent
   configs ⇒ the screen is usable.
2. **Root-cause `TT_MOVE` and `QCHECK_DEPTH0`** — one unreachable feature this week was worth +54 Elo.
3. **Re-test at fixed nodes what was rejected on the biased ruler**: `LMR_MIN_REM`, `LMR_REM_FLOOR_PCT` (both cut
   nodes, both rejected at fixed depth).
4. **ProbCut** — fires but costs 12 solves. My "margin too wide" theory was **wrong in direction**; sweep WIDER
   (3000/4000) or look at verification depth.
5. **Static-table ordering tiebreaker** (user's idea; see memory `ordering-and-reduction-idea-backlog`) —
   placement delta + king-zone pseudo-attacks for the history-0 quiets, second use as **reduction confidence**.
   Build accurate-then-cheap to avoid an uninterpretable null.
6. **Corrhist re-siting** — `ENABLE_CORR_HIST` EXISTS but is wired at the **RFP site only** (`:3660`); SF applies
   at the `staticEval` assignment so everything downstream inherits it. **GATE IT** on first measuring whether
   pawn structure explains our eval error (cp-space variance decomposition, win%-weighted for sizing). Unlocks
   the RFP return blend and `RFP_MAX_DEPTH>6`, both of which failed *because* corrhist is missing.
7. **PARKED:** passers (re-check trigger only, not scheduled — node-neutral, negative on both rulers at the
   guessed floor config, and the tested floors were never recorded); quiet-position FMC; `improving` in the
   futility margin (needs a flag split to avoid the LMR confound).

## OPERATIONAL (hard-won)
- **Auto-approve requires the command to START with** `bash '<abs runner path>'` — no `R='…'` wrapper, no `cd`.
  Read results with the **Read tool** on the Windows-path task output; a standalone `tail`/`grep` PROMPTS.
- **Do NOT pipe long runs through `grep`/`tail`/`sed`** — they buffer, so there is no progress until exit
  (`--line-buffered` on grep is not enough if `sed` follows).
- **NEVER edit a script while a job is executing it** — bash reads incrementally; shifting offsets produced a
  phantom syntax error mid-run. Same family as rebuilding the `.so` during a tournament.
- The `ps` sub greps only `tactical_test|sts_test|movematch|tournament.py|setupAI` — it **cannot see**
  `cploss_frozen.py`, so "none running" is not proof.
- **win% is how WE judge; millipawns are how the ENGINE computes.** The sigmoid is offline-only; SF's runtime
  corrhist path is integer-only.
