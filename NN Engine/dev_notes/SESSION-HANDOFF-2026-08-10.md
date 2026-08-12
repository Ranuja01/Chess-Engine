# Session handoff — 2026-08-10: the eval is DEGENERATE (root-caused), de-dup program started

> ## 🚨 READ THIS BLOCK FIRST
>
> **Nothing shipped, nothing committed.** Register still reproduces exactly:
>
>     250 / 35,426,396 / EBF 3.800 / STS 1631   ·   MIRROR 245 / 35,727,805 / STS 1741
>     BALANCED  tactical 495   positional 3372
>
> ### ★★★ THE DELIVERABLE — a DIAGNOSIS, not a strength gain (yet)
> The eval is **DEGENERATE**: ~30 terms over ~2 independent signals behind 4-5 **collinear channels**
> (`central_score` and the O/D-imbalance term re-read the same `attackingLayer`/placement cells already in
> `total`; king-zone pressure is credited four ways). Collinearity ⇒ the fit sees only the SUM of collinear
> weights ⇒ **non-identifiable ⇒ every retune FLATTENS.** The −85.6-Elo corpus fit was three stacked
> shrink pressures: (1) eval-DISTANCE objective rewards shrinking, (2) degenerate system, (3) **ridge
> "fixes" collinearity BY shrinking.** This is the first *unifying* explanation for the eval-arc's failures.
> 📐 Canonical: `dev_notes/collinearity-why-the-eval-cannot-be-tuned.md` (the math) and
> `dev_notes/eval-architecture-degeneracy-map.md` (the code map, fable-authored, key claims verified).
>
> ⚠️ **This is a HYPOTHESIS with a validated first step, NOT a proven win.** Zero Elo gained. "Make it
> identifiable, retune correctly, and it won't flatten" is untested — the retune's SPRT is the experiment.
>
> ### ▶️ WHERE WE STOPPED — first de-dup done, retune not started
> Screened all 5 redundancy candidates on move-match (target=collapse decisions / holdout=general): **only
> O/D imbalance is noise** (reshuffled ~87 moves for net-zero signal); **central / king-zone / capgain→
> material contamination are all LOAD-BEARING** (removing each hurts — so de-dup by REMOVAL sheds
> load-bearing terms; RE-SHAPE instead). Built the bounded O/D (`OVD_BOUNDED_MODE`, gated, byte-id off):
> **MODE 2 (dynamic-KNEE) = balanced STS −25 (neutral), reduces colour skew, colour-symmetry clean** — the
> identifiable form. MODE 1 (ratio) rejected (−217, mis-scales at seed). It is **not a standalone win**
> (O/D is small-signal); its value is that O/D is now tunable, not collinear.
> ▶️ NEXT: either continue de-dup (central is the next re-shape candidate) or go to **THE RETUNE** — the
> payoff — per the plan `~/.claude/plans/cryptic-moseying-penguin.md`.

---

## THE RETUNE OBJECTIVE (owner spec — the second half, not yet run)

On the identifiable system, ONE fit, **NO ridge** (de-dup replaces it). Corpus: `diverse_corpus_wide.csv`
(NOT the collapse set — biased). Labels → **win%** (k=0.00368208): ours, SF11-static, SF18.
- **Validity gate:** keep only positions where **SF11-static directionally agrees with SF18-search**,
  *including near-0* (SF11≈0 when truth≈0, and we're off, counts). Strips SF18's tactical component,
  leaving what a static eval can both *reach* and get *right*.
- **Asymmetric, strength-preserving** (the crux): where SF11 is closer to SF18 than us → fit **toward
  SF11** (learn; pull up our losses); where **we** are closer → **anchor to our current value** (preserve
  our wins, don't regress to SF11's level).
- Weight by **win%-impact** (a +3→0 error ≫ +10→+7). Held-out split; collapses = validation only.
🧰 ⚠️ CORRECTION (verified 2026-08-10): the spec pieces are NOT pre-packaged. `_ks_learnable.py` uses an
ABSOLUTE near/miss threshold, NOT the sign gate, and writes no file. What EXISTS: `diverse_corpus_wide.csv`
(23,113 rows, train/val), row-aligned `ks_sets/sf11_static_labels.csv` (sf11_static + sf18_static per row),
and a NO-RIDGE win%-MSE held-out harness `joint_fit.py`+`_ks_fit_eval.py` (`winpct` k=0.00368208, symmetric
toward `target_total`). What must be BUILT: (1) a labeling/join step applying the single-witness sign gate
(incl. near-0) → per-row win% targets; (2) the ASYMMETRIC strength-preserving branch in `_ks_fit_eval.py`
`loss_of` (target = SF11 vs our-current by |SF11−SF18| vs |ours−SF18|); (3) GRID edit in `joint_fit.py` to
target OVD/CENTRAL knobs with base `OVD_BOUNDED_MODE=2 CENTRAL_BOUNDED_MODE=1` (else CAP/KNEE are inert).

## Method order (durable — the program)
1. Screen each subsystem's de-dup on move-match (target+holdout, FIXED depth) → survivor map.
2. Re-shape load-bearing-but-collinear terms to **bounded** (build BOTH candidate forms, TEST — mode1 vs
   mode2 diverged −217 vs −25; you cannot pick on theory). Keep the accumulator collection, change only
   consumption. Gate, default off = byte-identical.
3. Batch survivors → identifiable structure. Verify byte-id + colour-symmetry + four-suite.
4. ONE win% SF11 retune, no ridge. 5. Games (SPRT vs current) — the truth gate.

## Historical context this session also produced (do NOT re-derive)
- **Capgain lane CLOSED** — fictional-compensation root-caused; selection knobs cost ~−85 STS per +0.85
  footprint (footprint purchased with accuracy). Symmetry knobs drag the mirror branch DOWN.
- **11th symmetry defect** — 2 typo'd queen-PST cells (`QUEEN_PST_FILE_SYM_MODE=3`, free, gated off, 13/14
  file violations). A constant magnitude can come from a TABLE, not a knob.
- **KS: 12-attempt history synthesised** — additive KS is **0-for-9 in games**; only SUBTRACTIVE wins are
  Elo-confirmed (de-king +7.4% score, `MOD_KS_REALIZ` +36.7). **The channel law: no KS lever has intrinsic
  value — its sign depends on which king-credit channels are live.** Triage: 17/23 ks_attack collapses are
  our OWN attack over-read ⇒ strengthening KS pushes the WRONG way. See
  `ks-twelve-attempt-history-and-the-channel-law` (memory).
- **Fresh 400g collapse corpus** — score 44.5% → 48.3% vs SF@2400 (the symmetry work helped), but the
  failure MIX and +2.01-pawn over-read are UNCHANGED. 67% statically fixable; `place` dominates the worst.
- **Qsearch finding** — capgains is a STATIC substitute for qsearch: its errors are FROZEN (search can't
  un-see them), SF resolves the same positions at d1. Lever idea: fade capgains with remaining depth.
- **3-arm search bundle SPRT'd −15 Elo / 1152g = FAILED** (additivity failed a 4th time). Search lane
  stays 2-for-15.

## BASELINE — RECORD ALL FOUR SUITES
    250 / 35,426,396 / EBF 3.800 / STS 1631     MIRROR 245 / 35,727,805 / STS 1741
    BALANCED tactical 495   positional 3372
⚠️ `sts300` 177w/123b, `wac` 190w/110b — both colour-skewed; the skew flipped a sign twice this session.
Always `sts_suite sts300_mirror.epd` / `wac_suite wac_mirror.epd`, score orig+mirror.

## Build/bench — LITERAL runner, single line, nothing before bash
`wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' <sub> [KEY=VAL]"`. ☠️ Leading `cd/tail/grep/git` prompts. Read
outputs with the Read tool. ⚠️ Knobs latch at init ⇒ one process per setting; registration ≠ WIRED ≠ READ
IN TIME (a knob a table-rebuild bakes in must be registered BEFORE the rebuild). ⚠️ `move_proxy` needs
`game_<digits>` tags; `SF11Eval.eval()` returns None on in-check FENs and has no readline timeout (a binfmt
hiccup hangs forever — print flushed progress, never pipe a long run).

## STATE
HEAD `63a879d` on `NN-ENgine`, **nothing committed this session** (multi-day). Large uncommitted set:
`cpp_bitboard.cpp` (`ovd_imbalance` + gated CAPG/queen-PST knobs), `search_engine.h/.cpp` (all new knobs +
registration + toggles), many `diagnostics/` scripts (`_move_match_arms.py`, `_ks_learnable.py`,
`_material_decompose.py`, `_qsearch_hypothesis.py`, `_ks_case_set.py`, `_gap_two_set.py`, `_preserve_vssf.py`,
`_ovd`/`_logic_pass` helpers), new `dev_notes/` (collinearity, degeneracy-map, KS-MORNING-PLAN,
collapse-leverage-map-refreshed, this handoff), `selfplay/games/_archive/` snapshots. Nothing running.

## DISCIPLINES
Byte-identity every build + same-session `wac_speed` peak. **|balanced STS| < ~150 is UNRESOLVABLE.**
**Validate on MOVES not cp** (a cp-distance screen failed its own control — any shrink games it). Footprint-
filter before spending. Games decide, only above the ~20-40 Elo floor, never peek early, run alone. Commit
only when asked, no footer.

## ⚠️ WEIGHT MY EXPLANATIONS ACCORDINGLY
The measurements held; the stories on top repeatedly did not. This session I: called a "double-count" that
was actually contamination (corrected on reading the restore is gated off); over-recommended de-dup by
removal until the central screen (net −4) refuted it; built a cp-based validator that violated our own
"validate on moves" rule and failed its own control; broke a probe's tag format so it discarded 100% of
rows silently; and seeded OvD mode 1 badly (−217) before mode 2 (−25) worked. ★ The owner's pushbacks were
load-bearing throughout — "if the material is real, shouldn't we care?" corrected the capgain-material
framing; "how many subsystems even are there?" right-sized the whole program; and the demand for chess-
semantic knobs (CAP/KNEE) over a blind `×3` is what makes the bounded term interpretable and tunable at all.
The degeneracy diagnosis is well-measured; that the retune will harvest it is the untested story — prove it
at the SPRT, don't assume it.
