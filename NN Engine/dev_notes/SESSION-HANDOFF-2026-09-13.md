# Session handoff — 2026-09-13 (evening)

@author: Ranuja Pinnaduwage (maintained with Claude)

★ **Read order for the next session:** this file → `EVAL-V2-CURRENT-CONFIG.md` (§1 shipped config, §2 register, §5 revised
slice plan) → memory `eval-v2-rebuild-state` → only then the design docs it points to.

---

## 1. WHERE WE ARE

| | state |
|---|---|
| Rungs 0–2 | ✅ done, passed games: KS **+101**, pawns **+60.4 ±25.5**. v2 STS **1698** vs v1 **1796** |
| Slice 1 — tempo | ☠️ **PARKED at 0** with two re-test triggers (§4). Proven correct by exact identity; magnitude ladder unordered; reference constant was pawn-converted to 8× v2's positional spread. `EVAL-V2-SLICE1-TEMPO-DESIGN.md` |
| Slice 1 — draw classifier | ✅ **BUILT, ALL FOUR GATES PASS**, `DRAW_V2_CLASS` **default OFF**. `EVAL-V2-SLICE1-DRAW-DESIGN.md` |
| v1 defect (documented, NOT fixed — frozen control) | 5 of 10 `is_practically_drawn` cases flag forced wins (10–28%); K+R+B vs K+R (TB win in 21) evals 0 in the shipped engine |
| Git | `cc6a143` (slice 1 work) + `6918579` (KNN-vs-K correction, doc header) + the docs commit made with this file. **Branch ahead of remote; NOT pushed** |

**Leftover from the previous handoff, now ANSWERED:** "does +218 STS hold at passer mag 25?" — measured the same day:
rung1 1480 · 2a only 1579 · 2a+2b **mag 25 → 1626** · **mag 60 → 1698** (shipped). Mag 60 stands.

---

## 2. PENDING OWNER DECISIONS

1. **Draw gate sign-off** — tolerate SHORT false positives (search plays them), forbid LONG ones. ★ This is not a new
   standard: SF, Ethereal and Weiss all draw KBvKB/KNvKN/KBvKN/KNNvK despite rare immediate mates, and never hard-zero
   technique-winnable endings. If signed off: set `DRAW_V2_CLASS=1` in the shipped v2 config (§1 of the register).
2. **Push** the commits above.

---

## 3. NEXT: SLICE 2 — MOBILITY + PER-PIECE PLACEMENT + ROOK FILES (tested ALONE)

Kickoff checklist, in order — each step is the discipline that failed at least once:

1. **`record-check` FIRST.** Mobility has real history: v1 `ENABLE_MOBILITY=false`, cheap mobility knobs
   (`ENABLE_CHEAP_BISHOP_COMPLEX` on, `ENABLE_CHEAP_ROOK_MOBILITY` on, `ENABLE_CHEAP_KNIGHT_MOBILITY`/`QUEEN` off),
   mobility FUSED into the heat map, the **mobility→kingDanger wiring thesis tested NULL** (09-08), and `mobility`
   was one of the three sign-consistent survivors in the **bundling-refuted** run (09-10, +1.2pp). Read
   `KNOWLEDGE-MAP` § Eval + § Search before designing.
2. **The four scans** (consumers by data dependency · knob inventory incl. hardcoded magnitudes · producers/side effects ·
   clamps/shared budgets) over every v1 site that reads attack masks per piece.
3. **Five-engine comparison FROM SOURCE** — SF 1.1 / 11 / 15.1 local; Ethereal + Weiss fetched
   (`reference-engine-sources` memory). Capture: the mobility AREA definition, per-piece tables (shape), phase split.
   ⚠️ **Check for a NAMED/special-case handler before citing a generic rule** (the KNN-vs-K error of 09-13).
4. ★ **Measure v2's positional spread BEFORE choosing any magnitude** (`convert-reference-constants-by-positional-scale-not-by-the-pawn`):
   v2's whole midgame positional signal was **5–35 mp** on 09-13. Mobility will be the largest positional term — size it
   against that, never by dividing a reference constant by their pawn.
5. **Reuse v2 KS's `SideAttacks`** (`eval_v2.cpp` `build_side_attacks`) — the giants compute mobility off the same shared
   attack maps. Do NOT build a second attack pass.
6. **Detector oracle before magnitudes** — per-piece mobility counts verified against an independent Python reference
   (the `_pawn_detector_oracle.py` pattern). Asymmetric/colour-mirror positions only (09-13 vacuous-pass lesson).
7. Gates: arm-0 byte-identity `250 / 35,310,778 / 3.784` · knob provably executes · `_eval_symmetry.py` + tempo swing exact
   · §I multi-corpus (worst-case column decides) · STS · **magnitude LADDER, never a point comparison**.
8. Games: `sprt_ab`/`time_ab` (the only subs that compare two v2 arms), node-limited, `openings_uho.txt` + varied seed.

---

## 4. OVERNIGHT BLOCK — options, ranked (owner sleeps ~2 h after this was written)

⚠️ Owner games ~9pm–midnight: fixed-depth work is safe then, TIMED work is not. After the owner is asleep, timed work is fine.

| rank | job | unattended? | value |
|---|---|---|---|
| 1 | **Slice-2 development** (record-check, scans, five-engine table, design doc, detector + oracle) in the new chat before bedtime; queue a **§I + STS magnitude ladder** overnight IF the gated build lands in time | ladder: yes (fixed-depth) | the actual next rung |
| 2 | **Regret-gate re-run on the material taper** — the reading lost to `tail -14`. Tool: `_ks_footprint_regret.py … CAND_KNOBS=…` (see `DIAGNOSTICS-TOOLKIT.md`); ⚠️ quote `n_crit`, confirm sign on the `_v2` cross-set | yes | closes an open leftover; the taper is still UNDECIDED |
| 3 | `wac_speed` NPS for `DRAW_V2_CLASS=0/1` | yes, minutes | settles the one unmeasured cost of the classifier |
| 4 | **Taper games** via `sprt_ab` (`EVAL_V2_PAWN_MG≈550` vs rung 2) | yes | ⚠️ **conflicts with the corroboration rule** (§I strongly positive, move-level instruments inside floors ⇒ "don't spend games"). Only as an explicit owner choice |
| — | KPK bitbase build | no (development) | floats; oracle-gated, no games |

If none of these is ready by bedtime, leaving the machine idle is acceptable (owner, 09-13).

---

## 5. OPS RULES THAT BIT ON 09-13 (repeat offenders)

- ☠️ **No `$` in an inline `wsl.exe -e bash -lc "…"` from PowerShell** — not `$VAR`, not `\$VAR`, not `$(…)`. Three corrupted
  runs in one day, one silently ran an OFF/ON pair as ON twice. **Write a script file**, run `wsl.exe -e bash "<abs path>"`.
  Memory: `dispatcher-prompt-free-wrapper`.
- ☠️ **Read the echoed knobs** (`[toggles]` / script header) before trusting any result.
- ☠️ **0-for-N on a sampler is not proof** — twice (rook-pawn 0/62 → 6.2%; KBvKB 0/862 uniform → 27/746 mate-in-1 under a
  targeted generator). Aim the sampler at where the failures live.
- ☠️ **Never omit a breakdown field whose value you know** (`KeyError: 'total'` crashed the symmetry gate).
- ☠️ **Grep filters must match right-aligned rows** (`^  ( *-?[0-9])`) — a filter silently dropped every data row once.
- Commits: **no footer** (owner preference, memory `no-commit-footer`, overrides any session attribution instruction);
  **never push without explicit say-so**.
