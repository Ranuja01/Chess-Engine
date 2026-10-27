# Veto screen — predictions registered BEFORE any arm result

**Written 2026-09-11, immediately after launch, with only the 3 nulls in flight and zero ablation results
seen.** The screen is 44 eval-term ABLATIONS + 3 matched nulls, primary corpus, `MAXN=5000`.
★ The point of writing this down is that a screen with 44 arms will produce a 2σ cell by chance, and a
prediction registered afterwards is indistinguishable from a rationalisation.

## What the pass is for
Not "which knob wins" — that question is closed (term-at-a-time by arithmetic, bundling by experiment).
It is **which terms are load-bearing**, i.e. the KEEP / DROP column of a ground-up rebuild.

## ☠️ How an ablation must be read
An ablation is not a candidate. Three-way sort, not two:
| reading | meaning | what it licenses |
|---|---|---|
| removal HURTS | load-bearing — **for an unknown reason** | KEEP in the rebuild, but the CONCEPT is not thereby validated: `a-correctness-fix-into-absorbed-tuning-is-not-free` means neighbours fitted around it produce exactly this signature |
| removal HELPS | **SUSPECT** | `corpus-fit-is-anti-correlated-with-elo` — every historical "switch it off and it improves" has been a corpus artifact. Needs games, never a ship |
| removal INVISIBLE | the rebuild is free to drop it | ⚠️ or the gate cannot see it — check the changed-rate before believing this |

## Predictions
1. **The modal result is INVISIBLE.** I expect >half the 44 arms to land inside the null band. If ~30
   terms carry ~2 signals, most names are re-descriptions and removing one leaves the other 29 to cover.
   ⇒ If instead most ablations read clearly negative, degeneracy is overstated and the rebuild's
   "minimum viable core" is much larger than I have been assuming.
2. **T1 masters split.** `ab_capgains`, `ab_threats`, `ab_latent` hurt clearly (these are the big
   move-selecting terms). `ab_kaufman`, `ab_imbalance`, `ab_pvboost` land invisible.
3. **T4 (the shipped correctness/symmetry fixes) reads INVISIBLE across the board.** They were validated
   by the symmetry harness and by games, and the d7 regret gate is a different instrument.
   ★ This tier is a **calibration of the gate, not a test of the fixes** — if removal of a fix that won
   games is invisible here, that bounds what this screen can see, which is worth knowing on its own.
   ☠️ If any T4 arm reads clearly POSITIVE, do NOT read it as "the fix was wrong."
4. **T2 placement legs hurt, and unevenly** — pawn and knight more than queen. ★ This tier is the first
   evidence `SCALE_HEAT_SCORE` has ever had: the per-piece placement leg is the largest heat-map consumer
   and the only one with no knob, so the 1.3pp is currently unattributed.
5. **`ab_ks_realiz` (`MOD_KS_REALIZ=0`) is the sharpest single test on the board.** It shipped at +36.7
   Elo in a bundle. If its removal is invisible at d7, that is a measured statement that **this gate
   cannot see a term worth +36.7 Elo** — which would put a floor under how much of the ~85-attempt
   failure record is instrument blindness rather than refutation.
6. **At least one arm reads >2σ POSITIVE and is wrong.** 44 arms guarantees it. Named in advance so it
   cannot be promoted on discovery: it goes to the confirm pass or nowhere.

## Protocol for the morning (fixed now, not after seeing numbers)
1. ⚠️ **Changed-rate first.** Any arm at ~0% was inert — a pre-verification miss, not a null. Six knobs
   were already dropped for this reason (gated behind default-off flags); assume the filter leaked.
2. Read every arm through `_paired_null.py` against the **`null5k_*`** arms only. ☠️ Never against the
   full-n nulls: pairing 5,000-position arms to 15,000-position nulls is the same error class as the
   global-null comparison this screen exists to correct.
3. **This pass may only REJECT.** Nothing is promoted on one corpus at reduced n. Survivors — in EITHER
   direction — go to the full three-corpus paired confirm pass.
