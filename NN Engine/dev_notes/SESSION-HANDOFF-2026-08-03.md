# Session handoff — 2026-08-03 (compensation arc: source read, and a clean kill)

**Read this top block first.** Prior: `SESSION-HANDOFF-2026-07-31.md` (the ship), then `OPTIMIZATION_LOG.md`.

---

## STATE — nothing changed in the engine
Default build is still the shipped **`246 / 35,089,668 / EBF 3.846 / STS 1746`** (`c2d4259`, +36.7 Elo).
Everything this session was env-knob or diagnostics-side. **Nothing running. Nothing committed since
`505ad84`.** Uncommitted, all diagnostics-side except one gated knob:
- `search_engine.h` / `search_engine.cpp` / `cpp_bitboard.cpp` — `PIECEVAL_RECOMPUTE_LATE` (**default false,
  byte-identity verified**: WAC `246 / 35,089,668 / EBF 3.846`)
- `diagnostics/fit_bench_guarded.py` — argv→env fix + Phase-A candidate grid
- `diagnostics/collapse_term_attribution.py` — `OUR2SF` mapping fix + `FENS=` per-position mode
- `diagnostics/probe_fens.py` — SF15.1c/SF15.1n columns, `--table`, win% error
- `diagnostics/sf11_collapse_gap.py` — worst-N now ranked by **win%**, not centipawns
- new: `dev_notes/DIAGNOSTICS-TOOLKIT.md`, `_collapse_leverage.py`, `_sacrifice_loss_mine.py`,
  `_sts_theme_diff.py`, `sf_ceiling_win.py`, `_judge_moves.py`, `_pick_probe.py`, `_depth_nodes_probe.py`
- `dev_notes/passed-pawn-subsystem-map-2026-07-18.md` — **stamped with a 2026-08-01 delta header**

## ☠️ THE HEADLINE — the passer R-floor fix is DEAD, killed properly
`fit_bench_guarded` on the **relabelled** `passer_fit.csv` (288 rows):
| | baseline | 64/96 | r6only192 | r6only256 | 96/128 | 128/160 | 160/192 | 192/256 |
|---|---|---|---|---|---|---|---|---|
| **corpus VAL** | **539.3** ✅ | 546.2 | 546.5 | 548.4 | 549.5 | 552.9 | 556.5 | **561.1** ☠️ |
| STS | 1746 | 1597 | 1639 | 1604 | — | 1636 | 1636 | 1729 (−17) |
**Train flat (~297-299), validation degrades MONOTONICALLY with floor strength.** The STS-neutral arm
(192/256, −17 STS / +1 WAC — inside noise) is the **worst** on held-out passer accuracy.
★★ **Why: `R` collapsing is CORRECT whenever the passer really is stopped.** A rank-based floor fires on
EVERY advanced passer, so it buys the one case we looked at and loses the many where damping was right.
**Same blunt-clamp failure as `KS_REALIZ_FLOOR` on the sacrificial tail — second time this pattern decided
a question.** ⇒ The miss is real; the fix must DISCRIMINATE stopped from unstoppable, not floor everything.

## ✅ WHAT SURVIVED — the audit (durable, use this)
- Under V3 the inline passer rank bonus is **DEFERRED, not added** (`g_passer_mid_deferred` is written and
  **never read** — vestigial), so `evaluate_passers`' **`mag × R/256` is the SOLE payer** of advanced-passer
  value. If R collapses, the passer gets NOTHING — not even a reduced rank bonus.
- V3 also killed **8 rook-evaluator passer sites** (mid `ROOK_PASSER_OWN/ENEMY`; end `rank×75`), replaced by
  one narrow `PASSER_REAR_OWN=48` R-credit. **Three reductions in one change**, compensation left at 0.
- ✅ **No double-count**: rear support = 2 live channels (#16 + `PASSER_REAR_OWN`); advanced-passer value = 1.

## 📚 SOURCE READ — SF11 / SF15.1 / Ethereal (banked in `compensation-blindness-threats-and-passers`)
- **Threats are INVARIANT SF11→SF15.1** (constants within a few %, `ThreatByPawnPush` identical) ⇒ by the
  portability heuristic, a safe form-port. **Ours (`threats_by`) is missing six structural pieces** — we skip
  pawn-defended pieces entirely (SF still pays `ThreatByMinor` on defended), never target pawns, have no
  pawn-push threat, no `RestrictedPiece`, no queen-threat terms, and a weaker `stronglyProtected`.
  ☠️ Moot until re-tuned: **`ENABLE_THREATS=0`**, which is why `threats` reads exactly 0.00 everywhere.
- **Passers EVOLVED monotonically toward stricter gating** ⇒ port SF15.1's form, not SF11's.
- **All three engines gate passer safety on ATTACKS**; Ethereal makes it a 2×2 table
  (`PassedPawn[canAdvance][safeAdvance][rank]`). ✅ **We do too** — `passer_realizability_R` is a CONTINUOUS
  version of SF's binary ladder. My "we lack this" claim was WRONG; owner corrected it.

## 🐛 THREE TOOL BUGS FIXED (they invalidated real conclusions)
1. `fit_bench_guarded.py` **silently ignored `KEY=VAL` argv** (read env; runner passes argv) ⇒ `CORPUS`,
   `TOPK`, `STS_TOL`, `WAC_TOL` all fell back to defaults. ✅ fixed.
2. `passer_corpus.csv` (raw mined) ≠ `passer_fit.csv` (fit schema) ⇒ feeding the raw one gives **MSE 0.000
   for every candidate**, a silent degenerate tie.
3. `collapse_term_attribution.py`'s `OUR2SF` counted non-additive `material` and mapped `pieces` to SF's
   activity-only terms ⇒ our `pieces` looked **+3-5 pawns over-read** in any position where we were merely
   ahead on material. **Two headline "patterns" WITHDRAWN.** Owner caught it from chess reasoning first.

## ▶️ PROPOSED NEXT (nothing started — owner's call)
1. **Re-mine collapses** on the shipped build and let the new class profile pick the lane, rather than
   following the pre-committed threats→KS→endgame list.
2. **If threats:** it needs a JOINT retune (it was disabled for unbalancing others) via `fit_bench_guarded`
   across MULTIPLE corpora **with a held-out split** — this session proved train-only fits mislead.
3. **A discriminating passer fix**, if one exists: something that separates a genuinely-unstoppable advanced
   passer from a genuinely-stopped one. A rank floor provably cannot.
🚨 Any eval change touching material/passers feeds `priced_passer[]` → capgains → the material accumulator →
**`MOD_KS_REALIZ`, which is load-bearing and shipped**. Per-class guard + games are mandatory.

## Discipline reminders earned THIS session
- **Check `dev_notes/DIAGNOSTICS-TOOLKIT.md` before writing any probe** — I rebuilt four existing tools.
- **A subsystem map goes stale the moment a gate ships** — stamp it in the same session.
- **Rank eval errors by win% (k=0.00368208), not centipawns.**
- **Always hold out a split** — the floors looked flat on train and failed on validation.
- ☠️ Only `wsl.exe -e bash -lc "bash '<runner>' …"` is prompt-free. A leading `cd`/`pgrep`/`tail` prompts and,
  if unattended, **freezes the whole queued chain**.
