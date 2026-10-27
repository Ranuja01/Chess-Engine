# Session handoff — 2026-07-24

## ★ HEADLINE: DE-KING SHIPPED — the biggest eval win of the campaign
`KS_ZONE_ATTACK_PCT` default flipped **100 → 50** (`search_engine.h`). Halves the KING-DIRECTED boost inside
`setAttackingLayer` (`cpp_bitboard.cpp` ~8194, applied as `kinc`), which was **TRIPLE-COUNTED**: into `pieces`
(via `positional_bonus`), into **OvD** (`imbalance_white`, via the offensive/defensive scores), and again by the
dedicated king-safety term.

**3-seed × 200g SF@2400, matched vs base_s1/s2/s3, per-class `profile_collapses`:**
| | base | de-king | Δ |
|---|---|---|---|
| score | 40.1% | **47.5%** | **+7.4%, UP ALL 3 SEEDS** (42.8→45.2 / 38.0→47.5 / 39.5→49.8) ≈ +50 Elo |
| **positional collapses** (target) | 55.0 | **43.0** | **−22%, DOWN ALL 3 SEEDS** (52→44 / 58→46 / 55→39) |
| total collapses | 78.7 | 65.7 | −13.0 |
| ks_attack | 20.0 | 19.0 | −1.0 |
| STS / WAC | 1555 / 247 | **1647 / 248** | **+92 / +1** |

**Byte-id after ship:** new default = **WAC 248 / 44,038,704**. Identity path `KS_ZONE_ATTACK_PCT=100` still
reproduces **247 / 39,971,153** exactly (also proves the new `EVAL_PROFILE` drill terms are inert).

**Tuning notes:** 50 is the PEAK — PCT=0 (full de-king) REGRESSES (STS 1538), 25→1574, 75→1587. The long-term
"aimed piece" signal has real value; we only wanted the duplication gone. Speed is neutral (king loops already
cached). **Do NOT stack blindly:** de-king + `ENABLE_KS_CHECK_V2` = STS 1481 (even rebalancing
`KING_SAFETY_MAG`→2500 only reaches 1582 < 1647 alone); de-king + V3-without-floor = WAC −12.

## ★★ THE DURABLE THEORY (explains the whole session)
| change type | effect | outcome |
|---|---|---|
| **TARGETED double-count removal** (de-king) | removes redundancy, KEEPS discrimination | **HELPS** |
| **UNIFORM de-bias / global magnitude shrink** (corrhist@RFP, `SCALE_PLACE_*` cuts) | FLATTENS the landscape → sibling moves less distinguishable | HURTS |

This **contradicts** the old corrhist verdict that "load-bearing optimism is FUNDAMENTAL" — de-king removed a
large systematic over-read and STS went UP +92. Prefer structural de-duplication over blanket scaling.

## METHOD VALIDATED (reuse it)
`collapse_term_attribution.py` (our `ev_breakdown` terms vs SF11 **and** SF15 classical per-term tables, on
real collapse FENs **vs a quiet control set**) → find the structural double-count → **targeted** removal →
**bench-guard** (`fit_bench_guarded.py`) → 3-seed games with the per-CLASS verdict.
⚠️ The corpus win%-MSE fit **REJECTED de-king three times** (it is blind to move choice) and its own picks were
bench-NEGATIVE 4/4. **Never let the corpus fit be the arbiter** — see `corpus-fit-flattens-eval` memory.

## Still gated (NOT shipped) — status
- **`ENABLE_PASSER_V3` + new `PASSER_RFLOOR_R5/R6`** — the floor-first fix (SF/Ethereal grant the rank table
  UNCONDITIONALLY; realizability is additive upside, never a total discount). Bench-guard **PASSED**
  (`deking50_v3_floor`: STS 1600 / WAC 245) and it **rescued** V3, which fails without it (WAC −12 → −2).
  **Not yet game-tested** → next game candidate, and test it as an INCREMENT on the new de-king default.
- **`ENABLE_KS_CHECK_V2`** (saturated per-type safe-checks) — rejected in every bench combination; keep gated.
- **`ENABLE_KS_ZONE_CLAMP` / `KS_ZONE_NORM` / `KS_CLAMP_SHELTER`** — low value, gated.
- **`ENABLE_CAPG_REALIZ`** — built but **INERT** (calls `realizability_factor`, whose `REALIZ_*` knobs default
  to 0 ⇒ factor 256 ⇒ identity). Needs the `REALIZ_*` knobs opened, or the own-king-exposure signal, to do
  anything. Its material-edge signal also MISSES the target case (winning-capture-with-compensation).

## New tooling this session
- `diagnostics/collapse_term_attribution.py` — term-level attribution of a collapse class vs SF11+SF15.
- `diagnostics/fit_bench_guarded.py` — corpus PROPOSES, real STS/WAC DISPOSE (bench as a GUARD, not an objective).
- `diagnostics/build_diverse_wide.py` + `add_sf15_static.py` — wide multi-family corpus with the **two-reference
  (SF11+SF15) static-achievability gate** (drop SF18 targets no classical eval endorses = search-only tactics).
- `diagnostics/ks_fit_wholesystem.py` — whole-system grid (~35 knobs).
- `diagnostics/attack_overread_dossier.py`, `fen_term_dump.py` — per-FEN ours/SF11/SF15c/SF15nn/SF18 tables.
- `diagnostics/sf_bench_ceiling.py` — SF11/SF15(off/on)/SF18 on our STS suite. **WRITTEN, NEVER RUN.**
- `_ks_fit_eval.py` now takes `LOSS=mse|logloss|hybrid` (MSE stays primary — see `fit-metric-mse-vs-logloss`).
- `EVAL_PROFILE` drill terms `PROF_PAWN_PPINC` / `PROF_PAWN_ATKLOOP` (compile-gated, inert in production).
- Fixed `eval_vs_sf11.py` to honour `SF11_BIN` (was hardcoding a stale Windows .exe path → unusable under WSL).

## ★ SEARCH INVESTIGATION (later 2026-07-24) — see `speed-and-qsearch-findings-2026-07-24.md`
- **Speed lane MEASURED and CLOSED.** SF11 2.53M NPS / SF15-classical 1.33M / ours ~497k. Our static eval ≈
  5,900 cyc/call = **65-84% of per-node cost** ⇒ the NPS gap is ALL eval; search machinery is only 15-35%.
  **No eval hotspot exists.** Pawn-hash cache (~6.5% of eval) and lazy-capgains (whole term = +1.4% NPS) both
  **REJECTED on measurement before writing risky code.** 4× NPS ≈ +1.1 plies vs EBF 3.68→2.5 ≈ +5 plies ⇒
  **EBF/pruning is the lever, not throughput.**
- **Capgains is a PRUNING-SAFETY device**, not a qsearch substitute: off-everywhere = −15 WAC / **+45% nodes** /
  only +1.4% NPS. Its value is at INTERIOR nodes (RFP/futility/null-move) — [[eval-accuracy-payoff-is-pruning]].
- **Qsearch delta prune is calibrated to a capgains-inflated stand-pat** (fable's theory, CONFIRMED): the −13
  solve penalty from `QSTANDPAT_EVAL_MODE=2` appears **only when `ENABLE_QDELTA=1`**.
- **PRUNING INVENTORY** (full table in the findings note). Highlights: LMR **6.58×** nodes/−15 solves; LMP best
  ratio (34.3M nodes per solve); **null-move and futility are FREE** (fewer nodes AND better accuracy);
  razoring worst ratio; `ENABLE_CONT_HIST` = **unreachable dead code** under the shipped `ENABLE_STATSCORE_LMR`
  default (legacy tier path is vestigial → delete candidate).
- **TWO ERRORS MADE AND CORRECTED (read these):** (1) ran `QDELTA=0` when the registration is
  `env_flag("ENABLE_QDELTA")` — silently ignored ⇒ briefly concluded "delta is dead code". **A byte-identical
  A/B usually means the knob was IGNORED** ([[env-knob-name-verify]]). (2) my `ENABLE_QDELTA_PERMOVE`
  per-capture futility **never fires** (identical to `ENABLE_QDELTA=0`); left GATED OFF, needs a fire-counter
  before any claim.
- **NEXT SEARCH LEADS:** (1) **LMR safety** — its −15 solves is the WHOLE accuracy cost of the pruning stack, so
  safer reductions recover accuracy without giving back the 6.58× (= the roadmap's ordering × prune-push pair,
  now with a measured target); (2) **razoring retune-or-remove** (worst ratio; margins likely fit to the old
  eval); (3) **lazy-resort re-judge in games** (+8% nodes for zero d10 gain, but it shipped via a
  games-measured bundle).

## Next up
1. **`EVAL_PROFILE` run** → `PAWN_PPINC / PAWNS` ratio → **pawn-hash cache go/no-go** (fable's full dependency
   map + risk list is in this session's record; the documented "safe boundary" was proven **UNSAFE** — score,
   rank-bonuses and the min(225) saturation all mix pure-pawn with king/piece-dependent terms).
2. **Search-margin co-tune on the NEW de-king baseline** — margins are absolute constants fit to the OLD, more
   inflated eval; de-king shrinks attack-position evals, so the same margin now prunes differently. Likely we
   are still UNDER-measuring de-king. Knob-only (no build). Use the offline **prune-log/AUC ladder**: record a
   prune-decision dataset once, then sweep thousands of configs offline (`ENABLE_PRUNE_SHADOW` may already be a
   partial implementation); gate on wrong-rate ≤0.3% → bench → one game confirmation.
3. **Passer V3 + R-floor games** as an increment on de-king.
4. Re-test the previously-flat search items — **their closures predate Kaufman and CAPG_PIN shipping**, so they
   were measured on a materially worse eval (see `search-reopening-angles-2026-07-24.md`).
