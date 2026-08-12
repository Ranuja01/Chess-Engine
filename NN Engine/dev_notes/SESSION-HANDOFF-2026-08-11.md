# Session handoff — 2026-08-11: low-depth regret method + de-noising bundle + the KS detector plan

> ## 🚨 READ THIS BLOCK FIRST
>
> **Nothing shipped, nothing committed.** All new eval knobs default to byte-identical. Baseline register:
>
>     250 / 35,426,396 / EBF 3.800 / STS 1631   ·   MIRROR 245 / 35,727,805 / STS 1741
>     BALANCED  tactical 495   positional 3372
>
> ### ★★★ THE STATE IN ONE PARAGRAPH
> The eval-arc unlocked a real method this session: **tune on a LOW-DEPTH SEARCH's move choice (regret /
> SF18-best match), NOT static corpus fit** (which is anti-correlated with Elo). Using it we confirmed the
> **de-noised OvD + central** (bounded, gated) is a small real gain — **+15.6 ±33.2 Elo over 580 games**
> (a lean, not yet significant) AND it beats default on held-out regret (2.65 vs 2.80). We also proved the
> **eval CONSTANTS are tapped out** (a full held-out-gated broad tune found zero generalizing moves). So the
> lever is STRUCTURE (de-noise / detector-redesign), not constants. Current program: **accumulate de-noised/
> upgraded subsystems (OvD ✓, central ✓, KS next) as a growing bundle, validate each on the fast deterministic
> instruments, confirm the whole BUNDLE in ONE game tournament** (a bigger effect resolves faster than
> confirming three small ones).
>
> ### ▶️ NEXT ACTIONS (in order)
> 1. **KS detector rebalance** — the plan is `dev_notes/KS-DETECTOR-REBALANCE-PLAN-2026-08-11.md`. Build the
>    coherent unit (defender-aware attacker weighting + `attackedBy2` weak + ratio rebalance) behind ONE gate,
>    screen firing-decomp FIRST, then the full battery. ⚠️ Solve the KS detectors as ONE unit — one-at-a-time is
>    proven to mislead (channel law).
> 2. **Cheap standard KS wins** (`attackedBy2` in weak, attack-count multiplicity) can be banked into the
>    bundle first if a tonight-ready candidate is wanted — they're correct, byte-id-off, quick to screen.
> 3. **Confirm the bundle in games** — OvD+central (+ whatever KS screens clean) vs default, a long tournament.
>    Screened seed config: `OVD_BOUNDED_MODE=2 OVD_CAP=300 OVD_KNEE=40 CENTRAL_BOUNDED_MODE=1 CENTRAL_CAP=150
>    CENTRAL_KNEE=200` (STS −75/noise, WAC +4/−4.4% nodes, symmetry clean, held-regret beats default).

## THE LOW-DEPTH REGRET METHOD (the session's methodological unlock)
Static corpus fitting is anti-correlated with Elo (re-proven: even the asymmetric SF11 objective flattened).
**Fix: tune on a low-depth SEARCH's move choice** — the Elo-bearing quantity. Two variants:
- **SF18-best top-1 match** (`_lowdepth_tune.py`) — binary, coarse (no gradient; use for a quick read).
- **Win%-REGRET** (`_regret_tune.py` focused / `_regret_tune_broad.py` broad+parallel) — graded = the tuning
  objective. `regret = winpct(SF18_best) − winpct(SF18_of_our_move)`, **side-to-move POV** (a POV bug that 0'd
  black-to-move rows is FIXED), from a multi-PV cache. Robust to collinearity (optimises the TOTAL's move
  quality, not the decomposition → no flatten, no ridge).
**The two phases (coarse-to-fine):** (1) STATIC corpus is the cheap coarse region-finder BUT it's
anti-correlated with Elo and constants are tapped out ⇒ **do not fit constants on it**; use it only to seed a
sane region. (2) **D7 low-depth regret is the live instrument** — our engine searches each position at **fixed
DEPTH 7** (~61 ms/pos; heavy pruning ⇒ ~14k nodes), takes its chosen move, and scores `winpct(SF18_best) −
winpct(SF18_eval_of_that_move)` from the cached multi-PV. Used to VALIDATE structural changes (de-noise /
detector-redesign) on held-out, NOT to tune constants.
🧰 **DATA — where to find it (all in `NN Engine/diagnostics/`):**
- `ks_sets/game_regret_set.csv` — **~15k GAME-representative** FENs mined from the 108k stored selfplay games,
  each with **SF18 multi-PV top-8 @ depth 14** cached (`best_uci,best_cp,moves`), disjoint from STS/WAC/collapse.
  Builder: `_build_game_regret_set.py` (STOCKFISH_PATH = the Linux SF18). **THE set to use.**
- `ks_sets/regret_set.csv` — the older 1k CORPUS multi-PV set (from `diverse_corpus_wide`); `_build_regret_set.py`.
- `ks_sets/diverse_corpus_wide.csv` (23k, SF18-search `target_total`) + `ks_sets/sf11_static_labels.csv` — the
  STATIC-fit corpus + SF11/SF18-static labels (the closed lane).
- `_mp_target.csv` / `_mp_holdout.csv` — the standing move-match validation sets (SF-best, ⚠️ time-based/noisy).
⚠️ Split MUST be shuffled (seeded) — an index split over multi-config game dirs = distribution mismatch that
looks exactly like universal overfitting (caught + fixed this session). Held-out gate REJECTS overfits; it's
the regularizer, NOT ridge. CV + more data (HOURS of labeling) sharpen it.
🔭 **Future arc (parked, high-value):** a **fixed-NODE** regret variant re-opens SEARCH-knob tuning and
PRESEARCH reduction (search knobs need fixed-nodes, not fixed-depth — pruning trades nodes for depth).

## THE DE-NOISING PROGRAM (durable)
Diagnose each subsystem's STRUCTURAL problem, fix it as a coherent gated unit, validate on the fast
deterministic instruments (held-regret + move-match + STS + symmetry + byte-id), accumulate, confirm the
bundle in games. OvD/central were COLLINEARITY (bound the duplicate). KS is DETECTOR mis-calibration (proximity
over-weighted, discrimination under-weighted). `every-eval-term-error-is-bidirectional` — de-dup/re-shape, not
global scale. See `eval-degeneracy-and-the-dedup-then-retune-program` (memory).

## KS SUMMARY (full plan in the KS doc)
Over-read = Channel-1 PROXIMITY firing on defended pieces; safe-check (genuine threat) is structurally tiny
(~1× attacker weight vs the giants' 15×). Fix = a coherent DISCRIMINATION unit (defender-aware weighting +
attackedBy2 + ratio), solved TOGETHER. `ks-twelve-attempt-history-and-the-channel-law` — additive KS 0-for-9,
one-at-a-time misleads, only subtractive/redistributive works. Tools: `_ks_channel_decomp.py`, `_ks_c1_decomp.py`.

## STATE / BUILD / DISCIPLINES
HEAD `63a879d` on `NN-ENgine`, nothing committed (multi-day). Uncommitted: `cpp_bitboard.cpp` (`ovd_imbalance`,
`central_bounded`, gated), `search_engine.h/.cpp` (knobs+registration), many `diagnostics/` (`_regret_tune*`,
`_build_*regret*`, `_ks_*decomp`, `_asym_corpus`, `_move_match_arms`, …), new `dev_notes/`.
Build/bench: `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>' <sub> [KEY=VAL]"` (ONLY the runner form is
reliably auto-approved; a complex compound got permission-DENIED unattended). Byte-id every build + `wac_speed`
peak. **|balanced STS| < ~150 is UNRESOLVABLE. Validate on MOVES/regret, not cp.** Games decide, run ALONE,
above the ~20-40 Elo floor. Commit only when asked, no footer.

## ⚠️ WEIGHT MY EXPLANATIONS ACCORDINGLY
This session I: kept proposing to re-try dormant KS features until the owner insisted on understanding the past
failures (which then corrected me — `KS_MIN_ATTACKERS` is inert with queens; safe-check cranks are toxic
solo); shipped a train/test distribution-mismatch bug (index split over multi-config games) that faked
universal overfitting until caught; and framed the collinear KS leak as the target before the decomposition
showed it's ~9% and proximity is the real lever. ★ The owner's discipline was load-bearing throughout —
"understand why it failed before re-trying," "solve the system together not one-at-a-time," and "be inspired
by the giants but stay uniquely ours" are what produced the actual plan. The measurements held; the stories
on top needed correcting. The +15.6 lean is real but UNCONFIRMED; the KS plan is well-founded but UNBUILT.
</content>
