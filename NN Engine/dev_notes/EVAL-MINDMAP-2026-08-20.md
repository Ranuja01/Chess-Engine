# EVAL MIND-MAP — why it keeps failing, why THIS looks different, and where to push next
*Synthesis of 3 research passes over the full dev_notes + eval code, 2026-08-20/21. A working map to steer from.*

---

## 0. THE THROUGH-LINE (read this first)
The eval is **degenerate** (≈30 terms projecting ≈2 independent signals, val/train ≈ 0.99) → it **cannot be tuned** by magnitude
(every retune flattens). Its per-term errors are **bidirectional** (mean|gap|≈1.0 pawn but signed mean ≈0.04) → **no scalar
works**. And its over-optimism is **load-bearing** (the thing generating its only active play) → **damping the error also damps
the strength**. Every proxy that ever said otherwise (corpus, STS, AUC) has been **directionally wrong** at least once.

**⇒ The ONLY things that have ever converted to Elo: STRUCTURAL de-duplications / subtractive-or-bounded re-shapes,
validated on MOVES or GAMES (never static corpus), confirmed as a BUNDLE in one tournament.**
Confirmed wins: OvD+central+defaware **+20.8**, capped threats **+45**, material-fix bundle **+36.7**, `MOD_KS_REALIZ` damp,
de-king, 7-fix symmetry (+70 STS pure correctness). *Not one winner was an added additive magnitude.*

---

## 1. WHY EVAL FAILS — the taxonomy (8 modes, 3 meta-roots)

### The 3 meta-root-causes (`strategy-reset-2026-07-15.md`)
- **M1 Local optimum** — single-lever hill-climbing nets ≈0 by definition; can't escape without a JOINT move.
- **M2 Measurement floor** — venue resolves ±15-30 Elo; real HCE gains are +3-15 Elo ⇒ *we can't SEE the good levers even when we build them.* (`|balanced STS|<~150` unresolvable; a +10 effect needs ~4,000+ games.)
- **M3 Coupled walls** — eval is over-optimistic AND load-bearing; no single lever moves one without breaking the other.

### The 8 failure modes (what actually went wrong, by frequency)
- **C1 Additive-magnitude tuning (~12, biggest bucket)** → bidirectional error, no scale works. *Additive KS 0-for-9 in games.*
- **C2 Corpus/static-fit anti-correlated with Elo (~6, all negative)** → **−85.6 Elo at the best-ever proxy fit**; fits magnitudes our errors flip; 52% of eval error is move-neutral. Winners of a corpus fit are all "switch it OFF" = the flattening signature.
- **C3 Collinearity/degeneracy (the MATH root under C2)** → non-identifiable params + shrinkage objective = guaranteed flatten. Fix is STRUCTURAL, never ridge. (`collinearity-*.md`, `eval-architecture-degeneracy-map.md`.)
- **C4 Coordinate-descent can't find GATED mechanisms (~4)** → a gate is one knob, its weights are others; flipping a gate at hand-guessed weights shows no gradient ("clean nulls" that were artifacts). *This is why the ring-gate needed the wave-tuner, not a sweep.*
- **C5 Load-bearing optimism / regression-to-the-mean (~6)** → + on low-baseline seeds, − on high, net ≈0. Killed a *mechanism-proven, correctly-signed* de-biaser (mobility). Damping the crutch → passive → lose more.
- **C6 Wrong instrument / bench-doesn't-transfer (~8)** → STS ranked colour fixes backwards 3×; STS +33→games −2.2%; AUC scale-blind (onset +0.150 AUC → SPRT ~0); passer move-match +6 masked STS −54.
- **C7 Measurement floor / unresolvable (~4)** → real gains read "neutral" (±30 Elo CIs on +12 effects).
- **C8 Structurally-wrong / mis-scoped diagnosis (~4)** → conclusions from reading a code fragment beat an available measurement (KS_INTERACT double-amplification; obstruction-blend ported a form SF doesn't have).

**KS specifically (12+ attempts, the CHANNEL LAW):** no KS lever has intrinsic value; its SIGN depends on which of 4-5 live
king-credit channels are active (proven 3×, e.g. `MOD_KS_REALIZ` −196 STS → +88 across the material fix). 17/23 `ks_attack`
collapses are our engine **over-reading its OWN attack** ⇒ naively strengthening KS pushes the WRONG way. Only SUBTRACTIVE KS won.

---

## 2. THE CURRENT PROMISING THREAD — the ring-gate (why it's a different SPECIES)

**Candidate (in games now, ~+10 Elo unresolved):** `ENABLE_KS_RING_GATE=1 KS_MIN_ATTACKERS=2 KS_FLOOR=0 CAPG_KS_DAMP=25 KING_SAFETY_MAG=4000`, `ENABLE_KS_UNIFIED=OFF`.
⚠️ **The tuner turned the endgame-unification OFF** — so the lean is from the **MIDGAME ring-gate + strong magnitude + capgains-damp**, NOT the endgame-blindness fix (that cliff is a separately-diagnosed, still-untested defect).

**The mechanism = DECOUPLE "when-to-fire" from "how-much".** Before: one knob (`KS_FLOOR`) controlled BOTH whether KS fires
and how big — a 0→1.26-pawn STEP on crossing, operating at 82% of signal (SF gates at ~7%). When-to-fire and how-much were the
SAME lever, so every past attempt traded detection vs quiet at a constant 36-47pp — the additive 0-for-9 direction.
The ring-gate adds an **orthogonal firing condition**: count enemy pieces attacking the SF-exact king ring (forward-inclusive
zone MINUS double-own-pawn-defended squares) = SF's `kingAttackersCount`; below `KS_MIN_ATTACKERS` → return 0 **regardless of
magnitude**. Now `KS_FLOOR→0` (recover all detection, no cliff) AND `MAG→4000` (strong) can coexist, because quiet positions
with 0-1 ring attackers stay 0. **Validated DECOUPLING:** at 3× magnitude, no-gate over-fires (safe-eg −0.069) but ring-gate ON
*improves* precision (+0.057) while keeping recall. Magnitude and over-fire, previously chained, now move in OPPOSITE directions.
`CAPG_KS_DAMP` then feeds the now-trustworthy danger into capgains realizability (the owner's real −6.75 game-loss term).

**Why it (maybe) validated where 12 others failed — two axes:**
1. It's a **firing-CONDITION change, not a magnitude re-fit** → the structural/subtractive class that is the ONLY class that ever shipped.
2. It was tuned/validated on **D7 searched-move regret + a first-class SAFE set** (silence-on-quiet, then crank magnitude while watching precision) — NOT static corpus (C2) and NOT a coordinate sweep (C4).

**☠️ THE HONEST CAVEAT (must travel with it):** additive/eval-KS is 0-for-9; bench-green has been NECESSARY-not-SUFFICIENT every
time. ~+10 is UNRESOLVED (CI still includes 0; needs days of diverse games). Collapses are rare tail ⇒ even if real, a modest
robustness gain, not a jump. **This is a reason to run games, not a result.**

---

## 3. THE VALIDATED TOOLKIT (what we now trust to measure eval)
- **D7 searched-move-regret** (`_regret_tune_broad.py`, `_ks_regret_score.py`, `_ks_wave_tune.py`): tune the TOTAL eval's move
  choice at fixed depth 7 vs SF18 multi-PV, win%-regret objective → robust to collinearity (no flatten, no ridge). ★ The **wave/curriculum** variant buckets by STAKES (win% best-vs-2nd) and tunes hardest-first with a general-stability guardrail — it PREDICTED this candidate. Fixed-depth PROXY; winner still needs games.
- **Specificity / SAFE-set gate** (`_ks_specificity.py`, `_ks_eg_specificity.py`): make the collateral (safe) set a first-class output; gate on "stays SILENT on safe positions." The rule that broke the KS over-fire cycle.
- **Calibration bands** (`_ks_calibration.py`): mean-ours/mean-SF ratio per SF-|KS| band → catches uniform magnitude deficits AUC is blind to. (AUC is scale-free ⇒ run `KS_FLOOR=0` + the `KS_ATTACK_COUNT=2` volume control, always.)
- **Ship gates (correctness):** `_eval_symmetry.py N=800` (mirror = −eval; MANDATORY for any eval change), STS+**mirror** scored orig+mirror on the balanced TOTAL (both suites colour-skewed), byte-id + `wac_speed` peak-NPS every build.
- **GAMES = ground truth**, run ALONE, JOBS≤4, only effects > ~20-40 Elo floor. ★ **Diversify openings** (`openings_uho.txt` 1000-book + VARIED seed per segment — the fixed-seed replay bug that invalidated seg1-6). No static statistic predicts Elo.

---

## 4. FORWARD ROADMAP — apply the "decouple when-from-how-much" pattern to the next terms
★ **The framework already exists, dormant:** `dev_notes/dynamic-conditional-eval.md` — a "detector-gated modulation layer" with a
`mod_gain()` primitive, `MOD_*` knobs (all default 0 = byte-id), and a **double-count matrix** naming which detector is legal per
term. The ring-gate is one instance. Check every candidate against that matrix before wiring.

**Prioritized targets (each = a FIRING-CONDITION decouple, validated on D7 regret + SAFE-set silence, mirror-tested, bundle-in-games):**
1. **`capture_gains` — HIGHEST.** The #1 static culprit + a real **−6.75 game blowup**. Over-read is REALIZABILITY = a WHEN
   problem (booked material that checks/deflections/pins make uncapturable). Scaffolding EXISTS but off: `CAPG_KS_DAMP`,
   `ENABLE_CAPG_REALIZ`, `ENABLE_CAPG_TEMPO`. Turn the damps into a **firing gate** (deflectable/defended-after/king-in-danger → ~0).
   ⚠️ search-absorbed at depth ⇒ judge at fixed **TIME**, not depth. (`cpp_bitboard.cpp:7643-7669` / `:7985-8011`.)
2. **passers — HIGH.** The −4.55 blowup; lane re-opened (§4b). Gate magnitude on realizability/blockade `R` as a firing condition
   (securely-blockaded passer → ~0; free passer → full). Detection half untested (misses 14% of SF passers). ⚠️ **phantom-knob
   hazard**: no master switch (`SCALE_PASSED_PAWN`=18%, `PASSER_MAG_SCALE`=32%) — past ablations measured a fraction; find ALL sources first.
3. **rook open-file — MEDIUM.** `ENABLE_ROOK_TENSION_COND` (off) already exists. Gate the file bonus on "file bears on a target,"
   not raw openness → magnitude can rise without over-crediting cosmetic open files.
4/5. **bishop-pair × openness** (`MOD_PAIR_OPEN` scaffolded), **central × center-contestability** (weak; matrix forbids the obvious phase detector). Low priority.

**CLOSED — do NOT re-propose:** imbalance/OvD (already structurally treated, in the +20.8 bundle), mobility (RULED OUT, load-bearing),
pawn placement/clamp magnitudes (lane closed; only passer DETECTION open), the refuted KS ideas (dbl-pawn prune, over-large zone,
KS x-ray, suppressor lane — `KING_SAFETY_MODEL.md §4`).

---

## 5. THE STANDING METHOD (the checklist for any eval attempt)
1. **Prove the knob is LIVE at defaults** (ablate on/off, dump evals — the pre-flight check; several "nulls" were inert knobs).
2. Build the change as a **firing CONDITION that stays SILENT on a SAFE set**; then raise magnitude while watching precision.
3. Validate move-change on **D7 regret** (game_regret_set), not static corpus. Rank by **win%**, not cp.
4. **Mirror-test** colour symmetry before ship (hard gate). Byte-id at default + `wac_speed` peak read.
5. Confirm the **BUNDLE in ONE tournament**, at fixed **TIME** for search-absorbed terms, on **diversified openings** (UHO + varied seed).
6. Accumulate structural fixes ONE unit at a time; periodically **re-baseline the whole stack vs the ORIGINAL default** so a lazily-accepted false-positive can't compound silently.

*The documented escape from the local optimum (M1): make the eval identifiable (bounded re-expression, de-dup) → then a JOINT
move (SPSA/SMP), not single knobs. The ring-gate + dynamic-conditional layer is the first real step on that path.*
