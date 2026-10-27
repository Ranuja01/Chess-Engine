# Session handoff — 2026-09-15 (early hours), mid-queue

@author: Ranuja Pinnaduwage (maintained with Claude)

★ Read order: this file → `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` §2.3.3 (the overnight queue, with pass criteria) → §3.5 (mobility games)
→ `EVAL-V2-CURRENT-CONFIG.md` §1 (shipped config) and §5 (collinearity gate, showdown fairness rule).

## 1. RUNNING RIGHT NOW (read these when they finish)
| job | output file (Read it; never shell-tail) |
|---|---|
| d7 regret, bundle E vs shipped v2, **primary** corpus | `C:\Users\Kumodth\AppData\Local\Temp\claude\c--Users-Kumodth-OneDrive-Desktop-Programming-Chess-Engine-Chess-Engine\2fc0e860-5b32-4734-bacf-71a2eae00807\tasks\b7lldmsx4.output` |
| d7 regret, bundle E vs shipped v2, **`_v2`** corpus | `…\tasks\bwiacdvm8.output` (same directory) |
Each prints its table only when both arms finish. Read the `placeE_*` row `win%`.

✅ **UPDATE 2026-09-15: both finished — E CLEAN** (primary 49.9 vs 49.8 = +0.1pp · `_v2` 49.2 vs 50.7 = −1.5pp).
▶️ **Regression SPRT on SHIP + E is RUNNING** (`s2_placeE_regress`, seed 13, elo0 −10 / elo1 0). Background task output:
`…\tasks\bmswv35ki.output` (same directory). Read it; when it decides, go to §5.

☠️ **SEGMENT 1 KILLED 03:29 by a Windows Update planned restart** (event 1074, TrustedInstaller / MoUsoCoreWorker — NOT memory).
Segment 1 (seed 13): **+103 −83 =36 / 222 games, LLR +0.804, elo ≈ +31**. `.so` last written 2026-09-14 23:53, before the launch ⇒
unchanged ⇒ poolable. **Segment 2 RUNNING** from 07:46: `s2_placeE_regress_seg2`, seed 14, max 978, same bounds, output
`…\tasks\b7s96bsau.output`. ★ The segment's own LLR is per-segment; decide on the POOLED W/L/D (add segment 1).
⚠️ To avoid a repeat: pause Windows Update or set active hours to cover overnight runs.

✅ **SPRT FINISHED — INCONCLUSIVE at max games, positive lean.** Segment 2: +396 −380 =202 / 978, elo +5.7 ±25.6, LLR +1.091
(`selfplay/games/s2_placeE_regress_seg2/sprt.json`). **Pooled +499 −463 =238 / 1,200 = 51.5% ≈ +10 Elo, LLR ≈ +1.9.** Never negative.
▶️ **SEGMENT 3 RUNNING (launched 2026-09-16, just after midnight, owner's go):** `s2_placeE_regress_seg3`, seed 15, max 1,500,
same arms and bounds as segments 1-2 (elo0 −10 / elo1 0), output `…\tasks\b0x3d04kd.output`. ★ Poolable: the `.so` fingerprint
is unchanged across today's rebuild (`250 / 61,352,373` before AND after), and the mobility form knobs are all default-off.
✅ **DONE 2026-09-16: segment 3 ACCEPTED H1 on its own — `+611 −548 =296 / 1,455, elo +15.1 ±21.0, LLR +3.039`. Pooled over 3
segments: `+1,110 −1,011 =534 / 2,655 ≈ +13 Elo`. Placement bundle E is GAMES-CONFIRMED.** (Original instruction below kept for
the record.) **Decide on the POOLED tally** (seg1 +103 −83 =36 · seg2 +396 −380 =202 · seg3 …), not on segment 3's own LLR.
Expected: at ≈ +10 Elo, ~1,000-1,500 more games should carry the pooled LLR past +2.94 ⇒ a formal H1, replacing today's
"inconclusive, positive lean". If it drifts NEGATIVE instead, propose UNSHIPPING E — that is the drift guard working.

✅ **OWNER SHIPPED E (2026-09-15 afternoon).** §1 + rebuild log updated. ☠️ **New shipped fingerprint WAC d10 `250 / 61,352,373 /
EBF 4.114`** — supersedes the §3 "shipped v2" row below. NEXT: mobility form bake-off on the mobility+E base (plan: table shape
SF11/SF15.1/Ethereal/Weiss at equal knight range · area knobs · eg share · pin line · x-ray only if no second attack pass · 600/800/1000
on regret · v1 form after a record-check); re-measure neutrals on this base; any change gets its own SPRT vs the shipped base.

▶️ **MOBILITY BAKE-OFF STATE (09-15 afternoon)** — detail in `EVAL-V2-SLICE2-MOBILITY-DESIGN.md` §2.1b:
- Built + gated: `MOB_V2_TABLE` 0-3 (SF11/SF15.1/Ethereal/Weiss, each rescaled by ITS OWN knight range + pawn pair) ·
  `MOB_V2_EG_PCT` · `MOB_V2_PIN` (SF blockers_for_king; mobility only) · `MOB_V2_SAFE` 1/2 (OURS-FIRST, from v1's
  lower-value-attacker test; per-piece masks stored in the ONE attack pass). All default = shipped form.
- ✅ Gates: byte-identity EXACT both arms (SHIP+E `250 / 61,352,373`, v1 `250 / 35,310,778`) · mobility oracle 0 mismatches on
  ship / TABLE 1,2,3 / PIN / SAFE 1,2 / a combined XRAY=0 arm · placement oracle 0 mismatches under PIN with all 10 terms
  firing · symmetry colour-swap 0/800, file mirror at the pre-existing 21 @ 5 mp.
- ★★ **VERDICT: NO CHANGE — shipped mobility form CONFIRMED, due-diligence gap CLOSED.** §I ladder (14 arms): SF11's table
  shape beat SF15.1 / Ethereal / Weiss; only `pin` (−0.51/−0.06) and `exlow` (−1.08/−0.34) cleared the both-better rule, and
  BOTH are regret nulls (pin −0.1/+0.6 · exlow +0.8/−0.4 · combined +0.6/−1.8, cross-set fail). Ours-first `MOB_V2_SAFE` lost.
  All knobs stay built and default-off; see design doc §2.1b (ladder + prediction scorecard) and §2.1c (parked + TRIGGERS).
- Neutrals for the regret step (measured on SHIP+E): **primary 49.9 · `_v2` 51.1**.
- ⚠️ Check at ladder time: a `pin` arm reading EXACTLY `ship` means the pin path never fires, not that it is neutral.

## 2. WHAT TO DO WHEN THEY FINISH (pre-registered rule — do not bend it to the numbers)
Neutrals on this base: **primary 49.8 · `_v2` 50.7**. E is CLEAN if neither corpus reads ≥ ~2pp below its neutral.
- **E clean on both** → launch the regression SPRT on E.
- **Otherwise** → launch it on bundle D (already fully gated: primary −1.2pp, `_v2` 0.0pp).

**SHIP** = `EVAL_ARM=1 KS_V2_ZONE_SF=1 KS_V2_XRAY=1 KS_V2_COORD=256 KS_V2_WEAK=57 KS_V2_ADJ=61 KS_V2_NO_QUEEN=321 KS_V2_CHK_Q=126 KS_V2_CHK_R=122 KS_V2_CHK_B=80 KS_V2_CHK_N=152 KS_V2_MAX=4000 KS_V2_HALF=600 KS_V2_ONSET=450 PS_V2_MAG=100 PASSER_V2_MAG=60 DRAW_V2_CLASS=1 MOB_V2_MAG=600 DRAW_V2_KPK_EXACT=1`
**D** = `OUTPOST_V2_PCT=100 BADB_V2_PCT=50 TRAPROOK_V2_PCT=10 WEAKQ_V2_PCT=25 BEHIND_V2_PCT=25 BEHIND_V2_FORM=1`
**E** = D with `BADB_V2_PCT=100 BADB_V2_FORM=1` (SF15.1 bad bishop)

SPRT (auto-approved runner form; arg order `'<p1>' '<p2>' tag max_games elo1 conc openings seed elo0`):
```
wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' sprt_ab '<SHIP> <E or D>' '<SHIP>' s2_placeE_regress 1200 0 4 openings_uho.txt 13 -10"
```
(elo1 = 0, elo0 = −10 ⇒ H1 means "costs no more than ~10 Elo" — the small-terms policy bar, NOT a gain claim.) Run it in the background.
☠️ Timed LIGHTNING games — stop before the owner games (kill by PID after `overnight_runner.sh ps`; `pkill -f` inside `bash -lc` kills itself —
use bracket patterns like `[e]ngine_server`). Completed games pool across segments if the .so fingerprint is unchanged.

## 3. STATE
- **Shipped v2 (§1)**: rungs 0-2 + draw classifier + **mobility 600 (≈ +162 Elo, 319 games, H1)** + **exact KPK bitbase**.
- **Fingerprints (reverify every build)**: v1 `250 / 35,310,778` · shipped v2 `250 / 60,036,572` · SHIP+D `248 / 57,474,821`. All exact after the last build.
- **Built + gated, default off**: placement sub-terms; per-term FORM knobs (`OUTPOST_V2_FORM`, `BADB_V2_FORM`, `TRAPROOK_V2_FORM`); `LATENT_V2_PCT`
  (ours); trapped-rook reuse of mobility's per-rook counts (efficiency); `sprt_ab` optional `elo0`.
- **Nothing committed, nothing pushed.**

## 4. RESULTS TODAY (one line each; detail in the design doc)
- Placement bundle D: §I −1.51% (6/6), regret ≈ neutral on both corpora (diffuse, no member carries it — leave-one-out needed a DILUTION control).
- Collinearity gate (new tool `_v2_term_collinearity.py`): placement terms VIF ≤ 1.30 vs mobility and each other — the trapped-rook double-pay
  concern was NOT supported by data.
- Form ladder vs D: **only SF15.1 bad bishop @100 wins** (−0.37 / −0.12, better 6/6). Ethereal/SF15.1 outposts, Weiss/Ethereal bad bishop: worse.
  SF1.1 trapped rook @25: near-miss (worst +0.02). **Our LATENT term: no gain, harmful as it grows** — stays off.
- Tempo re-test after mobility: sign flipped on STS (+107..+121) AND nodes (−5.1%) but IDENTICAL across 4× magnitude ⇒ a binary switch on
  search thresholds, not eval knowledge ⇒ parked for the margin re-sweep checkpoint.
- The old `_kpk_oracle.py` was broken (move-counter keying); June's KPK validation is UNVERIFIED; the new exact bitbase passed 165,676 states.

## 5. OPEN / NEXT (after the SPRT)
- If the regression SPRT passes: propose shipping the bundle (owner's call). If it fails: rethink implementation first, leave-one-out WITH a dilution control.
- Candidates for a later ladder: SF1.1 trapped rook @25; outpost/bad-bishop magnitudes around the winners.
- ★ **MOBILITY DUE-DILIGENCE GAP (owner question, 09-15):** mobility got record-check, four scans, the five-engine contrast (area =
  4/5 consensus), oracle, symmetry, byte-identity, §I/STS ladders, regret vs neutrals and games — but NOT the alternative-forms bake-off
  that placement got. Shipped on SF11's tables because games were decisive (+162), which is not proof it is the best version. Next, after
  the placement SPRT: a **mobility form ladder** — table shape (SF11 / SF15.1 / Ethereal / Weiss, normalised to the same knight range) ·
  area candidates (`MOB_V2_EXCL_QUEEN` measured once on §I only; `MOB_V2_EXCL_LOWRANK` never) · x-ray occupancy decoupled from
  `KS_V2_XRAY` (all five references differ) · SF's pinned-piece restriction (not built) · endgame share · magnitude 600 vs 800/1000 on regret
  (§I peaked ~1000) · an ours-first candidate from v1 · collinearity vs KS once a KS count probe exists · the KS-critical worst case growing
  with magnitude (+1.91% at 600). Any change from the shipped form gets its own SPRT.
- ▶️ **SLICE 3 STARTED: `EVAL-V2-SLICE3-DESIGN.md`** (central / space / threats / Kaufman + pairs). §0 record-check is written;
  §1 five-engine contrast running. ★ Headlines: **space was never refuted** (only v1's FLAT form, on the contaminated harness;
  SF's gated form was never built) · **central has no v2 feeder** (heat map retired) so any v2 build is a NEW mechanism — but it
  is a candidate SECOND OWNER of what PST + mobility already price, so the collinearity gate runs BEFORE the ladder ·
  **threats' ownership premise changed** (v1's "hanging is 87% capture gains" does not transfer — v2 has no capture gains) ·
  **Kaufman is the only weakly-positive games record** (port the census-product FORM, not v1's fitted tables) · **pairs were
  never measured alone** anywhere (v1's flat pairs are dead code behind `ENABLE_KAUFMAN_IMBALANCE`).
  ☠️ §0.3 lists the instruments that must NOT be cited as closures here (contaminated STS era, fixed-opening SPRTs, superseded
  global-null screen, corpus-fit rankings, v1-regime §I switch-off winners).
- ✅ **SLICE 3 RESULTS SO FAR (09-16):** **bishop pair BUILT, gated and measured as ALREADY OWNED** by v2's PST + mobility
  (§I nothing clears both-better · per-class §I flat in all 5 classes · regret null on both corpora) ⇒ knob stays default-off,
  trigger = a later PST/mobility re-scale. **`central` NOT built** (0/5 references; the owner's reframe replaced it with a
  per-class test of its OWNERS). 🧰 **New tool `_position_class.py`** — pawn-structure classes + a `pin_dense` tag, written WITH
  the source SF18 labels so the accuracy and regret tools consume them unchanged. ☠️ **`MOB_V2_PIN` is a NULL on its own class**
  (§I −4.75% on `pin_dense`, 9× its global value; regret 49.4% vs a class-measured neutral of 50.0% = −0.6pp) ⇒ a per-class §I
  win is NOT a second instrument ([[a-per-class-corpus-win-is-not-a-second-instrument]]). ⏳ running: `exlow` on
  `centre_tension` + that class's neutral.
- ✅ **SPACE: built, verified, PARKED (09-16).** Oracle 0 mismatches (both form families), symmetry 0/800, byte-identical off.
  §I: SF quadratic form harmful at every magnitude; Ethereal linear form at scale parity INERT. Only effect is class-local on
  `centre_locked` (−0.04..−0.13, mechanism: locked centres are where mobility collapses) and its regret there is +1.5pp on 467
  changed moves = **unresolvable** (±2.3pp candidate + ±2pp bar). Trigger: a purpose-built closed-centre corpus. NOT in the bundle.
  ☠️ Two transferable lessons: **a form comparison must be SCALE-NORMALISED first** (my linear arm was 14× under-scaled and
  would have been filed as "neutral" untested), and **the changed-move subset is the sample, not the corpus** (+13,500 positions
  bought 322 class rows and 37 changed moves).
- ✅✅ **SHIPPED 2026-09-17: `MOB_V2_PIN=1 MOB_V2_EXCL_LOWRANK=1`** (≈ +31 Elo, 1,178 games, two seeds — H1 at the ≥10 bound,
  H0 at ≥50 ⇒ bracketed 10 < true < 50). **New shipped fingerprint `250 / 59,549,832 / EBF 4.080`** (same 250 solves as the
  pre-area config in 1.8M FEWER nodes). ☠️ Both terms were PARKED as §I-only nulls when measured alone — they became
  measurable only when BUNDLED (owner's proposal). ⚠️ The mobility FORM bake-off still stands as NO CHANGE; these two are a
  separate later result. Bishop pair + space RETIRED from the candidate list (built, default-off, triggers intact).
- ✅ **THREATS: built and FULLY VERIFIED, but the §I ladder fails the both-better rule at every magnitude.** Oracle 0
  mismatches on both gate forms × all 7 legs × both occupancies (fire 38.6 → 82.3 → 94.8%) · symmetry 0/800 on the full stack ·
  byte-identity exact off · collinearity 21 terms NO FLAGS (★ `th_restricted` VIF **1.10** — it does NOT duplicate mobility's
  area despite sharing its attack maps: sharing an INPUT is not sharing a SIGNAL). Ladder: every arm improves the 4 general
  corpora AND the variant corpus strongly (`th100` −9.71% on variant, the largest single-corpus gain on this base) while taxing
  `lichess_ks_labelled` in near-exact proportion, monotone both ways (th10 −0.74/+0.35 … th100 −3.80/+3.90).
  ✅ **THREATS VERDICT: PARKED — NULL on moves, cross-set confirmed.** `th10` primary **50.1% vs a bar of 50.1% (0.0pp)** ·
  `_v2` **49.7% vs 50.5% (−0.8pp)**, on 35-36% of all moves, both bars measured ON THIS BASE (the old 49.9/51.1 belong to
  SHIP+E). Built, default-off. Triggers: the KS count probe · lazy eval · the unbuilt SF legs (`Knight/SliderOnQueen`).
  ☠️ **SLICE-3 PATTERN IS THE FINDING:** central (0/5, not built) · bishop pair (already owned) · space (inert) · threats
  (verified, move-null) — four concepts the giants carry, none adding move-level information on top of v2's existing terms,
  while the only thing that paid (+31) refined the AREA of a term we already own. ⇒ Hypothesis: **v2 is nearer positional
  saturation than its term count suggests; the gains left are in DEFINITIONS, not new concepts.** Needs its own test.
- ▶️ **NEXT AND NOW LOAD-BEARING: a KS COUNT PROBE.** Threats' KS-critical tax looks like a double-count with king safety, and
  KS is the ONE subsystem the collinearity gate still cannot see (flagged as a coverage hole since slice 2). Until it exists,
  "threats overlaps KS" is an untested story — and ☠️ elegance of explanation is not evidence.
 — the two terms §0 showed were never
  properly built here. Then one slice-3 regression bundle with `MOB_V2_EXCL_LOWRANK` riding along, then the slice-end
  cumulative SPRT vs the pre-slice-3 ship.
- Collinearity gate coverage still missing: KS and pawn structure (no count probes).
- v2-vs-v1 showdown rule (owner): NPS pair in a quiet window → margin re-sweep for v2 → play it twice (v1 margins / re-swept). Tempo folds into that re-sweep.
- Owner charter restated: v2 must stay clean — one owner per concept, no collinearity, efficient; pick the best definition per term from any engine or ours.

## 6. OPS RULES THAT BIT TODAY
- `pkill -f <pat>` inside `wsl.exe -e bash -lc "…"` matches its own command line and kills itself (exit 15).
- Never edit `overnight_runner.sh` while a runner job is executing; an ambiguous anchor there matched TWO subs (`sprt_ab` and `gate`) — anchor on unique text.
- `Select-String` patterns must be line-anchored, or they print the whole `[toggles]` line.
- The placement probe PACKS form-dependent fields (Ethereal outpost cells 8 bits, SF15.1 bad-bishop classes 12 bits) — unpack before any analysis.
- `EVAL-V2-CURRENT-CONFIG.md` has CRLF line endings — single-line Edit anchors only.
