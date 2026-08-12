# KS morning plan — 2026-08-10

Ready-to-run package. Everything here is **subtractive or conditioning**, because additive KS is
**0-for-9 in games** while the only two Elo-confirmed KS wins were removals (de-king `KS_ZONE_ATTACK_PCT=50`
= +7.4% score / −22% positional collapses; `MOD_KS_REALIZ=128` = +36.7 Elo in bundle).
Full history + citations: memory `ks-twelve-attempt-history-and-the-channel-law`.

## The two facts that should drive the morning

1. **The channel law.** No KS lever has an intrinsic value — its SIGN depends on which king-credit
   channels are live. Proven 3×, most sharply: `ENABLE_KS_CHECK_V2` was **bench-neutral (WAC −2 / STS −2)
   and +2.6% score on 3 seeds BEFORE de-king**, and **−14 WAC / −166 STS after**. Nothing about the knob
   changed. ⇒ Any knob sweep measures the channel structure, not the knob.
2. **FOUR live channels credit king-zone pressure** (`latent_threat` is DEAD — `ENABLE_KS_REPLACE_LT=true`):
   | # | channel | state |
   |---|---|---|
   | 1 | unit-KS (`KING_SAFETY_MAG=3000`, damped by `MOD_KS_REALIZ=128`) | live, the only phase-tapered one |
   | 3 | attackingLayer king-directed boost → `pieces` | live **at 50%** (de-king halved it; peak was 50, not 0) |
   | 4 | O/D accumulators → OvD imbalance ×`IMBALANCE_SCALE=3` | live, unconditioned, untapered, **never de-duplicated** |
   | 5 | flat mg-king shelter 185/75 + `baseIncrement` | live (`ENABLE_KS_V2=false`) — **duplicates `KS_SHIELD`/`KS_OPEN_FILE` inside unit-KS** |

⚠️ And the triage says the direction matters: of 23 ks_attack collapses, **17 are our engine OVER-READING
its own attack** (SF18 ≈0..+2.5 where our deep eval reads +50..+80), 5 eval-blind, 1 search-bound.
**Strengthening KS pushes the wrong way.** This agrees with the 2026-08-09 finding that the KS term error
is bidirectional (13 over / 21 under, signed +0.036, mean |gap| 0.995).

## ⚠️ READ FIRST — do NOT start from a single-cause theory

Three different "the answer is X" stories were proposed on 2026-08-10 alone (missing SF terms → channel
de-duplication → per-king asymmetry), each on the newest fact, each over-rotated. The owner's read is that
the situation is more nuanced than any of them. **The morning starts with DIAGNOSIS, not a candidate.**

Three hypotheses, and they demand different fixes, so separate them BEFORE touching a knob:

| hypothesis | signature | how to test |
|---|---|---|
| **H1 MISSING** — we lack the concept | SF11's KS fires substantially where ours reads ~0 **AND no setting of any existing knob can produce that signal** | force each candidate gate ON at its max and check whether the position can ever fire |
| **H2 MISTUNED** — right concept, wrong number | our term fires but at wrong magnitude/side; **some** knob setting reproduces SF-like output | sweep the responsible knob and see if SF's value is reachable |
| **H3 BAD DETECTOR** — right concept and weight, wrong INPUT | our score differs because our attacker set / weak squares / safe-check squares differ from SF's on the SAME position | compare **detector outputs, not scores**: `ks_explain.py`, `ks_component_dump.py`, `ks_firing_profile.py` vs SF11's per-term table |

★ H3 is the one never systematically tested, and it is invisible to every measurement made so far —
every past experiment compared SCORES. If our detectors disagree with SF's about *which squares are weak*
or *which checks are safe*, then tuning weights on top of bad inputs cannot converge, which would explain
eight iterations of "the knob has no intrinsic value".

## Case set to build first (span the failure modes, don't cherry-pick)

Pull 3-5 FENs in EACH bucket so static and search failures aren't conflated:
- **A. static wrong, search recovers** — our d1 eval is off but d13+ finds SF's move ⇒ eval-only defect, safe to fix in eval.
- **B. static right, search still fails** ⇒ NOT an eval problem; do not spend KS work on it.
- **C. both wrong** ⇒ the eval error is deep enough to survive search — the highest-value class.
- **D. we UNDER-count danger to our own king** (the owner's repeated observation; today's position 1: ours `king_safety ≈ 0` vs SF11 **+4.12**).
- **E. we OVER-count our own attack** (the 07-23 triage's claim: 17/23, our deep eval +50..+80 vs SF18 ≈0..+2.5).
🧰 `diagnostics/depth_probe.py` separates A/B/C by re-searching at d10/16/22. D vs E is the per-king split.
⚠️ The 17/23 triage measured our **deep SEARCH eval**, not the static KS term — it does **not** isolate KS as
the cause, and both D and E produce the same symptom ("we think we're winning"). Do not treat it as settled.

## EXPERIMENT 1 (only after the above) — sweep the shelter channel DOWN

`ENABLE_KS_V2=1` re-homes the flat 185/75 shelter constants into tunable `KS_SHELTER_*` (identity at
185/75/100) and lets `MOD_KS_REALIZ` damp the whole budget. **Built and byte-id-verified 2026-07-23, then
the session pivoted to REALIZ and never came back — it is the only live duplicate channel never measured
alone.** Same recipe as de-king, the biggest eval win we have.

    # 0. IDENTITY CHECK FIRST — must reproduce 250 / 35,426,396 / EBF 3.800 exactly
    wac ksv2id ENABLE_KS_V2=1 KS_SHELTER_FULL=185 KS_SHELTER_PARTIAL=75 KS_SHELTER_MAG=100
    # 1. sweep, expect an INTERIOR optimum (de-king peaked at 50, NOT 0)
    sts  ksv2m75  ENABLE_KS_V2=1 KS_SHELTER_MAG=75     (then 50, 25)
    sts_suite sts300_mirror.epd ksv2m75m ENABLE_KS_V2=1 KS_SHELTER_MAG=75    (then 50, 25)
    # 2. best value -> WAC + wac_mirror, then the move screen
    pyrun diagnostics/_move_match_arms.py ARM=ENABLE_KS_V2=1,KS_SHELTER_MAG=<best> SET=diagnostics/_mp_target.csv  N=79  DEPTH=10
    pyrun diagnostics/_move_match_arms.py ARM=ENABLE_KS_V2=1,KS_SHELTER_MAG=<best> SET=diagnostics/_mp_holdout.csv N=150 DEPTH=10
    pyrun diagnostics/_eval_symmetry.py N=800 ENABLE_KS_V2=1 KS_SHELTER_MAG=<best>

🚦 **Ship gate:** identity holds at 100 · balanced totals (never a single column) · **plugged > broke on
TARGET and broke not inflated on HOLDOUT** · colour-symmetric · then games.
☠️ If step 0 is not byte-identical, STOP — the re-homing is not neutral and nothing downstream is readable.

## EXPERIMENT 2 — partial de-king of the OvD re-spend (needs one small gated knob)

Channel 4 is the one leg of the triple-count never touched. The O/D accumulators re-spend the same
king-centric `attackingLayer` writes that de-king already halved in channel 3, then feed
`(off − max(def,0)) × IMBALANCE_SCALE` with no conditioning and no phase taper.
**Build:** a gated percent on the KING-ZONE SLICE of the O/D write (default 100 = byte-identical), swept
exactly like `KS_ZONE_ATTACK_PCT`. ⚠️ Do NOT purge OvD — the 07-23 scope correction showed OvD+placement
alone priced FEN-3 at +3.55 vs SF18 +3.04, i.e. those channels were paying the RIGHT amount there.
⚠️ The old "imbalance is only 4% of over-reads" debunk PREDATES the material fix that rewired every
consumer of the material edge — re-measure, don't inherit it.

## EXPERIMENT 3 — smooth-form `MOD_KS_REALIZ` (do the FREE step first)

The +36.7 bundle carried 44 unresolved **sacrificial-attack regressions**, and the floor sweep proved that
tail is NOT separable via `KS_REALIZ_FLOOR` (only the most aggressive value wins).
**Zero-compute first:** mine `selfplay/games/sprt_ksr128/` losses for the signature "materially ahead at
phase ~24, then mated". If the signature is real, the fix is one gate + a smooth polynomial blended on the
OUTPUT, to separate "under-backed because thin" from "under-backed because sacrificed".

## ☠️ DO NOT re-test
`KS_CHK_*` · `KS_FLOOR` · `KS_MIN_ATTACKERS` (queen exception drops the bar to 1 ⇒ inert) · `KS_DYN` ·
`KS_ATT_PRODUCT` · `KS_OVERLOAD` (blanket form; the GRADED per-square form using our
`num_attackers[]`/`num_supporters[]` is genuinely untried and is OURS, not SF's) · `KS_AIM` · `KS_INTERACT` ·
zone knobs · any port of SF's weak/check/no-queen terms into unit-KS (it would be a **5th** credit channel).

## 🧬 Uniquely ours, if we want new content rather than de-duplication
Graded per-square contest (`num_attackers[]`/`num_supporters[]`) — SF/Ethereal use binary weak/safe-check
flags. `KS_OVERLOAD` failed *because it ignored defenders per-square*; the graded, win%-calibrated form (the
passer-playbook shape) has never been built. · `g_capg_tension` is a free convertibility detector, proven
for capgain, **never pointed at KS** — and the 07-23 map says any future KS accuracy lever needs exactly a
non-material, non-magnitude convertibility signal.
