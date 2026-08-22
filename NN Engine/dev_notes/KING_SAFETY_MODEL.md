# KING SAFETY MODEL

**Canonical subsystem doc for king safety**, in the role `PAWN_MODEL.md` plays for pawns. It records the
measured system, the reference architectures with provenance, the instruments and *their resolution*, and a
**refutation record** of what has already been measured and killed.

**Read this before touching KS.** Update it when you change the system — a stale model doc is worse than
none. Superseded claims are struck through and kept, never deleted, so nobody re-runs a dead lane.

Symbols are the durable cite; line numbers drift and are omitted deliberately.

---

## 0. THE HEADLINE (measured 2026-08-18, n=5,481)

Our KS is an **under-detection** problem, not an over-read problem.

| | our KS reads exactly 0 |
|---|---|
| positions where SF sees real king danger (n=3,110) | **51%** |
| positions where SF sees none (n=2,371) | **98%** |

Baseline discrimination **AUC = 0.7365**. We miss half of all real king dangers outright, while quiet
positions are already essentially silent. ⚠️ This **refutes the 2026-08-17 "both-ends squeeze" premise**
that quiet is over-detected (that came from a 37-position hand-built counter set; on 2,371 corpus positions
the over-read is not there).

### ★★★ CALIBRATION ≠ DISCRIMINATION — measure BOTH (`_ks_calibration.py`, n=3,200)

AUC is scale-free BY CONSTRUCTION, so it is **structurally blind to a uniform magnitude deficit** — exactly
the failure a feeder defect produces (pinned defenders counted as real / no x-ray SHRINK danger while
preserving order). Ratio = mean|ours| / mean|SF| per SF-magnitude band, midgame:

| SF \|KS\| band | n | base ratio | base zeros | `ONSET=1 FLOOR=6` ratio | zeros |
|---|---|---|---|---|---|
| 0.05-0.25 | 36 | 2.24 | 56% | 2.08 | 8% |
| 0.25-0.50 | 54 | 1.39 | 33% | 1.03 | 9% |
| 0.50-1.00 | 247 | 0.64 | 49% | 0.58 | 8% |
| **1.00-2.00** | **1529** | **0.22** | **75%** | **0.29** | **24%** |
| 2.00-4.00 | 1161 | 0.54 | 26% | 0.49 | 8% |
| 4.00+ | 173 | 0.64 | 14% | 0.59 | 9% |
| **OVERALL** | 3200 | **0.45** | 51% | **0.44** | **16%** |
| Spearman | | 0.575 | | **0.608** | |

Two findings:
1. **The deficit is NOT uniform — there is a HOLE at SF |KS| = 1-2 pawns** (ratio 0.22, 75% silent), and
   that band is **48% of all positions where SF sees any king safety**. It is the floor's fingerprint: 1-2
   pawns of SF danger ≈ our units around 13, so the deadzone eats the middle of the distribution.
2. **The onset fix is a COVERAGE fix, not a CALIBRATION fix.** Zero-reads 51%→16%, the 1-2 hole 75%→24%,
   ordering +0.033 Spearman — but the overall magnitude ratio is UNCHANGED at ~0.44. **We remain at under
   half of SF's KS magnitude everywhere.** That residual is the real target for feeder work, and it is
   invisible to AUC. ⇒ **the feeder lane is NOT closed** — it was judged on the wrong dimension.
3. We simultaneously **over-read** the near-zero band (ratio 2.24) and **under-read** real danger (0.22-0.64)
   — the bidirectional-error pattern found in every eval term here.

### THE FEEDERS ARE NOT THE BOTTLENECK FOR *COVERAGE* — THE GATE IS

With `KS_FLOOR=0` our feeders give a NONZERO reading on **96%** of SF's danger positions (AUC 0.9113).
With the shipped floor they read zero on **51%**. ⇒ **of the danger we miss, ~47 points are discarded by the
OUTPUT GATE and only ~4 points are actually missed by the detectors.**

Full feeder sweep at `KS_FLOOR=0` (base 0.9113; volume buys ~0.0016 AUC per 10% magnitude):

| arm | AUC | Δ | vol Δ | Δ from volume | **excess** |
|---|---|---|---|---|---|
| `KS_ADJACENCY=1` | 0.9220 | +0.0107 | +31% | +0.005 | +0.005 |
| `KS_SQC_MODE=1` | 0.9180 | +0.0067 | +22% | +0.0035 | +0.003 |
| `ENABLE_KS_AIM=1` | 0.9130 | +0.0017 | +7% | +0.001 | +0.001 |
| `KS_PIN_MODE=1 KS_PIN_ATT=1` | 0.9120 | +0.0007 | +3% | +0.0005 | ~0 |
| `ENABLE_KS_WEAK_ATT2=1` | 0.9088 | −0.0025 | +3% | +0.0005 | −0.003 |
| `KS_WEAK_VAL_MODE=1` | 0.9039 | −0.0074 | +48% | +0.008 | **−0.015 (worse)** |
| *control* `KS_ATTACK_COUNT=2` | 0.9185 | +0.0072 | +44% | +0.007 | ~0 ✓ |

**No built feeder produces a resolvable detection gain** (every excess ≤0.005 vs SE 0.007). The control
lands at ~0 excess as it must, validating the arithmetic. ⇒ **The "KS is a detection-accuracy problem"
diagnosis is, on real data, mostly WRONG.** Spend effort on the gate/output stage first.

### The floor is the single biggest lever measured (n=5,481)

| `KS_FLOOR` | AUC | danger missed | quiet silent | step at crossing |
|---|---|---|---|---|
| **13 (default)** | 0.7365 | 51% | 98% | 1260 mp |
| 9 | 0.8476 (+0.111) | 26% | 94% | 630 mp |
| 6 | 0.8977 (+0.161) | 11% | 79% | 270 mp |
| 0 | 0.9113 (+0.175) | 4% | 53% | none |

`KS_FLOOR=13` costs **0.175 AUC**. The knee is at 9: **+0.111 AUC for 4 points of quiet silence**, and it
halves the discontinuity. Below 6 the trade turns expensive (9→6 costs 15 points of silence for +0.05).

⚠️ **Direction warning:** lowering the floor makes KS fire MORE, which is the additive direction that is
**0-for-9 in games**. The static case is strong and the game prior is bad — exactly the situation only an
SPRT can settle, and informative whichever way it lands. It is also NOT free in search: KS feeds RFP,
futility and qsearch stand-pat, so a lower floor changes pruning everywhere.

### ★ THE STRUCTURAL FIX — `KS_ONSET_MODE=1` (continuous onset)

The floor does TWO separable things: it silences small readings (its purpose) AND it charges
`ks_safety_table[KS_FLOOR]` as a STEP on crossing (an accident — the table already reads 42 at unit 13).
`KS_ONSET_MODE=1` subtracts the gate's own table value, so danger starts at 0 where firing starts. The
silencing is kept, the step is gone, and **everything above the gate is priced LOWER than today** — so the
gate can be lowered to recover detection without paying magnitude, which a hard floor cannot do.

| arm | AUC | danger mean | quiet mean | danger missed | balanced STS |
|---|---|---|---|---|---|
| base (hard 13) | 0.7365 | 0.929 | 0.028 | 51% | 3384 |
| hard 9 | 0.8476 | 1.058 ↑ | 0.052 | 26% | 3328 (−56) |
| hard 6 | 0.8977 | 1.077 ↑ | 0.089 | 11% | — |
| continuous 9 | 0.8216 | 0.731 ↓ | 0.026 | 32% | — |
| **continuous 6 ★** | **0.8865** | **0.916 ↓** | 0.048 | **16%** | **3376 (−8)** |

**★ D7 FULL-SAMPLE (15k, the Elo-bearing arbiter) — 2026-08-18. AUC was MONOTONE in the floor so it could
not choose this knob; D7 was the TUNER, and it found an INTERIOR OPTIMUM at 6:**

| config | TUNEreg | HELDreg | HELDcrit | dHcrit |
|---|---|---|---|---|
| bundle + defaware1 (base) | 2.5315 | 2.7042 | 3.2907 | — |
| **onset1 floor 6** | **2.5218** | **2.6239** | **2.5879** | **−0.7028** |
| onset1 floor 4 | 2.5360 | 2.6659 | 3.2447 | −0.0460 |
| onset1 floor 9 | 2.5170 | 2.6773 | 2.9173 | −0.3734 |

Improves BOTH tune and held regret (generalises, not fits) with a large critical-band gain. Compare the
PARKED V2 safe-check candidate: comparable −0.73 crit gain but bought with **STS −181**; this one is
**balanced STS −8**. Floor 6 also coincides with the independently-derived magnitude-neutrality criterion.
⇒ SPRT launched (`gate 'KS_ONSET_MODE=1 KS_FLOOR=6' onset6`, 1200 games, elo1=5).

**`KS_ONSET_MODE=1 KS_FLOOR=6`: +0.150 AUC, magnitude slightly BELOW base, balanced STS neutral, colour
symmetry identical to control, byte-identical at the gated default.** Discrimination up while magnitude goes
DOWN ⇒ it does not fight the 0-for-9 additive record the way a plain floor drop does.
⚠️ STS skew warning: orig went −110 while the mirror went +102. Reading `sts300` alone would have killed it.

**`KS_FLOOR=9` (the knob-only arm) gate status (2026-08-18):** AUC +0.111 · colour symmetry UNCHANGED vs the same-build control
(11/800 = 1.4%, file 14/651 = 2.2%, identical medians/worst — passes the ship gate) · **balanced STS −56**
(1713+1615=3328 vs base 1771+1613=3384), inside the ~150 unresolvable band · byte-identity holds at the
gated default. Remaining before games: **full-sample D7 criticality regret** (the Elo-bearing arbiter).
📌 Clean-harness **base STS mirror = 1613** (measured today). The `1741` in the fingerprint register is
CONTAMINATED-ERA — do not use it as the mirror baseline.

---

## 1. OUR ARCHITECTURE (code map)

All in `cpp_bitboard.cpp` unless noted. Entry: `placement_and_piece_eval` → `evaluate_king_safety` →
`king_safety_score` → `king_safety_danger` (per king).

**Phase 1 — feeders**, inside `king_safety_danger`:
- **Zone**: `white_king_ks_zone` / `black_king_ks_zone`, built in `initialize_attack_tables` as
  `ring1 | (ring1 shifted one rank toward the enemy)`. **Measured mean 9.2 squares** — the same SIZE as SF's
  9-square ring; the difference is SHAPE (a forward-staged 3-wide band vs SF's compact 3×3 around a
  centre clamped to files B..G / ranks 2..7). `ENABLE_KS_ZONE_CLAMP` selects a `_clamped` variant which
  **keeps the forward push and is therefore 12 squares — WIDER, not SF-shaped.**
- **Attack maps**: `attack_bitmasks[s]` = per-square OR-mask of attacker ORIGIN squares, populated by every
  per-piece eval loop (~24 sites) and shared with threats / capgains / OvD / placement. **No x-ray anywhere.**
- Per-square scan derives `attackers_sq`, `defenders_sq`, `attacked_zone_squares`, `weak_squares`,
  `contested_zone`, `overload_sum`.
- Weak predicate (`ENABLE_KS_SF_WEAK`, LIVE): ≤1 defender and no pawn/minor/rook defender — a faithful SF port.
- Safe checks (`ENABLE_KS_SF_SAFECHECK`, LIVE): geometry + safety predicate faithful to SF11.

**Phase 2 — transformation**: flat weighted attacker sum `{N2,B2,R3,Q5}` (⚠️ **queen HIGHEST — inverted vs
SF**, which uses `{N81,B52,R44,Q10}` with the queen LOWEST because queen danger belongs in the check
channel), plus `KS_ATTACK_COUNT` per attacked zone square, `KS_WEAK`, shield/open-file/storm, safe checks,
then `KS_DEFAWARE_MODE=1` (LIVE) which re-weights attackers by contested footprint fraction.

**Phase 3 — output**: `KS_FLOOR` deadzone → `ks_safety_table[units]` → netted `danger_white − danger_black`
→ phase taper → `MOD_KS_*` → `KING_SAFETY_MAG=3000`. Consumed by RFP, futility, qsearch stand-pat (all at
`EVAL_MODE=0`, i.e. full eval).

### The table and the floor (`rebuild_ks_tables`)
`table[u] = u²/KS_DIVISOR` up to `KS_KNEE`, then linear with slope `2·KNEE/DIVISOR`. At defaults
(DIVISOR 4, KNEE 12, CAP 80): `table[12]=36`, `table[13]=42`, slope 6/unit.

⚠️ **`KS_FLOOR=13` discards the table's ENTIRE quadratic region (u 0-12, danger 0-36)** and lands at 42 —
a **0 → 1260 mp (1.26 pawn) STEP**. Our live operating point (~15.8 units on real attacks) therefore sits
**entirely on the linear tail**; the gentle onset we built is never reached.

Reference contrast: SF fires at `kingDanger > 100` where real attacks run 1500-3000 — **a gate at ~7% of
signal, costing 2cp at crossing**. Ours is a gate at **82% of signal** plus a 1.26-pawn cliff.

---

## 2. REFERENCE ARCHITECTURES (verified from local source)

Sources are LOCAL — do not download: SF11 `stockfish_11\stockfish-11-win\src\`, SF15.1
`stockfish_15\stockfish_15.1_win_x64\src\` (`evaluate.cpp`, `pawns.cpp`, `position.cpp`, `bitboard.cpp`).
Ethereal is NOT local (github.com/AndyGrant/Ethereal). **SF15.1 is the LAST classical-KS Stockfish** — SF16+
have no `kingDanger` to read.

**SF's KS architecture did not change across five years** (SF11 → SF15.1): same >100 threshold, same
`d²/4096` transform, same −873 no-queen term, same 11-term signed sum. Only detector PRECISION was sharpened.

### SF11 `kingDanger` — the full signed sum
Positives: `attackersCount × attackersWeight` (a PRODUCT — super-linear coordination), `185 × weak-in-ring`,
`148 × unsafeChecks`, `98 × blockers_for_king` (pinned defenders are CHARGED), `69 × kingAttacksCount`
(adjacency, priced separately), `3 × flankAttack²/8`, typed safe checks (Q 780, R 1080, B 635, N 790;
SF15.1 adds a multiplicity table), `+37` base.
Negatives: **`−873 × !enemy-queen`** (≈24 Elo, its single largest annotated term), `−100 × knight-defender`,
**`−6 × mg(shelter)/8`** (graded and SIGN-FLIPPING), `−4 × flankDefense`, `+mg(mobility[Them] − mobility[Us])`.

Then: `if (kingDanger > 100) score -= S(d²/4096, d/16)`.

**Feeder-stage (structural) suppressors** — half of SF's suppression happens before units exist:
`kingRing &= ~dblAttackByPawn` · `weak` requires `~attackedBy2[Us]` · check `safe` requires the square not be
covered unless overwhelmed · pinned attackers clipped to `LineBB` at generation · x-ray in the attack maps
(bishops through ALL queens; rooks through all queens AND own rooks; knights and queens get none).

### Ethereal
Structural **entry gate**: `kingAttackersCount > 1 − popcount(enemyQueens)` (≥1 attacker with a queen, ≥2
without). Below it the whole block is skipped — exact zero, no tuned floor. Then a signed sum with
area-normalized attack counts (`×9/popcount(kingArea)`), `SafetyNoEnemyQueens S(−237,−259)`,
`SafetyAdjustment S(−74,−26)`, and a penalty-only output `−mg·MAX(0,mg)/720`.

### ⚠️ Scale note — our suppressors are NOT undersized
`KS_NO_QUEEN=6` is 38% of our mean attack sum (15.8); SF's −873 is 29-58% of its typical 1500-3000;
Ethereal's −237 ≈0.9 of its one-pawn point. **All three are the same RELATIVE size.** "Scale suppressors to
reference proportions" is chasing a gap that does not exist. SF's −873 ≈ 0.81 × its rook-safe-check (1080):
the design intent is that **a queenless attack survives iff it is CHECK-BACKED** — porting the constant
without the typed-check ladder ports the deletion but not the escape hatch.

---

## 3. INSTRUMENTS — and their resolution

**Use the right one. Most of the 2026-08-18 confusion was instrument error, not engine behaviour.**

| instrument | what it measures | resolution | trust |
|---|---|---|---|
| **`_ks_auc.py`** (NEW) | discrimination AUC over `ks_sets/diverse_corpus_wide.csv` (23,113 rows carrying SF's per-term `target_ks`; midgame subset n≈5,500) | SE ≈0.007 | ★ **PRIMARY — but RUN IT WITH `KS_FLOOR=0`** (see below) |

☠️ **AUC is scale-free ONLY with the floor off.** `KS_FLOOR` maps a whole range to exactly 0, manufacturing
mass TIES at the bottom (51% of danger positions AND 98% of quiet ones read 0 at the default floor), and AUC
scores ties 0.5 — so any VOLUME increase breaks those ties in danger's favour and inflates AUC. Measured:
the pure-volume control `KS_ATTACK_COUNT=2` gained **+0.107 AUC with the floor ON** but only **+0.007 with it
OFF**. That control is what caught a false "adjacency is a detection win" result. **Always run a
volume-control arm** (`KS_ATTACK_COUNT=2`) alongside any candidate, and read floor-free.
| `_ks_detect_dist.py` | per-king unit means, DANGER vs QUIET vs STS_REGRESS, `KS_FLOOR` forced 0 | moderate | good for phase-1 units; group means hide per-position effects |
| `_ks_bench_score.py` | 82-position archetype bench, netted KS in pawns | ☠️ POOR | see below |
| `_ks_bench_liveness.py` (NEW) | how much of the bench is non-zero | — | run before trusting any bench delta |
| `_ks_dblpawn_coverage.py` (NEW) | pure python-chess geometry probe | exact | no engine needed |
| D7 criticality regret (`_ks_regret_score.py MAXN=0`) | searched move quality | full 15k only | the Elo-bearing arbiter; sub-samples SIGN-FLIP |

### ☠️ The archetype bench is a weak instrument — read this before quoting it
At default `KS_FLOOR=13` only **29/82 positions are live (35%)**. `A4_other`, `B1_defended_crowd` and
`B3_shelter` have **ZERO** live positions (they read 0.000 in every config, forever). `A3_uncastled`,
`A5_weak` and `B2_queenless` are **ONE position each**. Its counter figure is ~0.09 pawns printed to 2dp, so
it **cannot resolve** the small feeder changes that matter. It is also **NETTED** between kings, so a floor
artifact on one king moves the other's number, and a purely SUBTRACTIVE change can appear to RAISE
over-production. **Always run bench arms with `KS_FLOOR=0`** (liveness 35% → 72%).

---

## 4. REFUTATION RECORD — measured and killed (do not re-propose)

- ☠️ **SF's `kingRing &= ~dblAttackByPawn` is INERT for us.** Only 0.09 (danger) / 0.32 (quiet)
  double-own-pawn-defended squares intersect our zone per position, and half are not enemy-attacked.
  Quiet units 5.08 → 5.03. Mechanism is GEOMETRY — such squares are simply rare. Billed as the "biggest
  single cut" by the 08-17 handoff; it is not.
- ☠️ **"Ring-only weak / the zone is over-large at ~15 squares"** — the zone measures **9.2** squares, the
  same size as SF's ring. The premise was wrong; the defect is shape.
- ☠️ **Quiet over-detection is not our problem** — 98% of SF-quiet corpus positions already read 0.
- ☠️ **Suppressors cannot buy discrimination.** Every suppressor tried (`KS_SQPRUNE_MODE=4` graded contest
  count, larger `KS_SHIELD`, disabling `KS_DEFAWARE_MODE`) trades danger for quiet at a near-constant
  **36-47 percentage points of danger per 1.0 of counter, with the floor ON or OFF.** The gate shape is not
  the mediator. Our own graded suppressor was MORE efficient than shipped defaware (37 vs 47) and still lost.
- ☠️ **Volume knobs degrade discrimination.** `ENABLE_KS_ZONE_CLAMP` posts the biggest raw detection (1.40)
  and the WORST bench ratio (5.4 vs base 12.1) — it makes everything fire more, quiet included.
- ☠️ **`KS_MIN_ATTACKERS=2` is Ethereal-exact** (our queen-decrement reproduces it) **and never binds** —
  every bench position already has ≥2 attackers. `=4` makes over-production WORSE via netting. The old `=3`
  null was stricter than Ethereal on both branches and never condemned the gate.
- ☠️ **KS-local x-ray is not worth building.** `KS_BATTERY` measured 0/46 collapse coverage and bench ratio
  9.9 (BELOW base). A faithful KS-local x-ray is a strict generalization of it and should measure the same;
  batteries are congestion, and congestion inflates counters as much as dangers. Faithful x-ray requires
  rebuilding `attack_bitmasks` at ~24 population sites, which also drives placement/OvD/threats/capgains —
  a multi-session whole-eval change.
- ☠️ **A5_weak's negative sign was one FEN.** Nine of ten positions read exactly 0.00; the set average was
  one position (subject king 5 units, opponent 18, floor 13). The clamp does not fix it — that position gets
  WORSE (18→23); the clamp wins the average by waking two OTHER silenced correct readings.
- ☠️ **Pinned-ATTACKER clipping (`KS_PIN_ATT`) is AUC-flat** (0.7365 → 0.7360; it does execute — danger mean
  moved 0.929 → 0.925). Pinned enemy attackers bearing on a king zone are rare, the same way
  double-pawn-defended squares are. Correct code, no measurable payoff. Kept gated off.
- 🔶 **Adjacency (`KS_ADJACENCY`) is NOT ESTABLISHED.** Floor-free it gains +0.0107 AUC for a 31% volume
  rise; the pure-volume control gains +0.007 for 44%. Per unit of volume it is ~2× more efficient, but the
  excess over volume alone (~0.005) is under the ~0.007 standard error. Promising mechanism, unproven.
  Deleting the flat zone count to "make room" for it is strictly bad (`KS_ATTACK_COUNT=0` → AUC 0.6308).
- ☠️ (older) Additive KS is **0-for-9 in games**; corpus-fit cost −85.6 Elo; the "endgame KS hurt" was a
  harness-contamination ghost. ⚠️ AUC is a STATIC corpus proxy, and corpus proxies in this engine are
  historically anti-correlated with Elo — treat a large AUC gain as a reason to run games, never as a result.

---

## 5. OPEN ITEMS

**Phase 1 (feeders), needing code:** pinned-ATTACKER clipping (`KS_PIN_ATT`, built 08-18, gated off) ·
pawn attacks seeding the attacker count (partial — pawns already populate `attack_bitmasks`, so they already
feed `attacked_zone_squares`/`weak`; only the weighted attacker sum excludes them, as SF's does too) ·
`unsafeChecks` channel (SF `148 ×`) · **shelter→danger feedback** — our shelter is a flat 185/75 subtraction
inside `evaluate_kings_midgame` that adjusts `total` directly and is **never fed back into
`king_safety_danger`**; SF's is graded, storm-aware and sign-flipping, and is its largest continuous
suppressor. We have no channel for it at all.

**Phase 2/3, mostly knob-testable:** the gate/onset shape (`KS_FLOOR`/`KS_DIVISOR`/`KS_KNEE`/`KS_CAP` —
never swept) · attacker-weight de-inversion (`KS_ATT_QUEEN` 5→2) · the count×weight PRODUCT
(`KS_COORD_GATE_MODE`) · typed saturating safe checks (`ENABLE_KS_CHECK_V2`, parked — real D7 critical gain
−0.73/n146 but STS −181, entangled) · the signed accumulator (`KS_ACCUM_MODE`) · the `isEndGame` 71%→0 phase
cliff (`KS_EXTEND_EG`), a second discontinuity nobody has touched.

**Adjacency re-price** (`KS_ADJACENCY`, built 08-18, gated off): SF prices attacks landing on squares the
king ITSELF defends at 69/instance, separately from its ring count; we charge a flat `KS_ATTACK_COUNT` per
zone square. Intended REDISTRIBUTIVELY (raise it while lowering `KS_ATTACK_COUNT` so volume holds and only
shape changes) — the only candidate shape that can mathematically raise danger and lower quiet at once.

---

## 5b. GIANT-GROUNDED PHASE ROADMAP (2026-08-19) — the standing KS plan

**CONVERGENCE (primary source, this session):** SF15.1 and Ethereal — two top engines built independently —
agree on the ENTIRE king-safety STRUCTURE. Two independent giants converging is the strongest evidence it is
the superior method, so we ADOPT the base (this is "converge to the field's answer," NOT "clone SF"):
- **NO phase/endgame gate.** KS is computed always and fades ONLY via the attacker set (Ethereal src: "no
  explicit isEndgame check"; SF likewise — phase interpolation happens AFTER, never zeroing it).
- **Entry gate = attackers + queen:** `kingAttackersCount > 1 − popcount(enemyQueens)` (near-identical wording
  in both). No attackers ⇒ 0 naturally — this is the giants' "when to fire", replacing a deadzone.
- **Non-linear (quadratic) danger→score:** SF `d²/4096`, Ethereal `−mg·mg/720`.
Ethereal's ONE distinctive trick = area-normalization (`×9/popcount(kingArea)`) — we tested it (`KS_ZONE_NORM`),
marginal for us. Where WE stay uniquely ours = **DEFAWARE** (contest-graded attacker weighting, finer than the
giants' binary `attackedBy2`) layered ON the correct base — innovate on the detection-quality layer, not the base.

**OUR TWO UN-GIANT DEVIATIONS (neither giant does these — they ARE the defects):**
1. **`isEndGame` cliff @ phase_score 65** (cpp_bitboard.cpp:7203) ⇒ ENDGAME BLINDNESS. On 92 real `ks_attack`
   collapses our KS reads 0.00 where SF-classical reads −1..−5.5 (a queen/rook hunting an exposed king). VERIFIED
   the isEndGame gate is the killer (collapses are phase<104, so NOT the phase taper — that `KS_PHASE_ZERO`
   detour was a no-op red herring). Lift knob `KS_EXTEND_EG` (adds to TOTAL, not the `king_safety` breakdown field).
2. **`KS_FLOOR` deadzone** ⇒ the 1-2 pawn HOLE. `_ks_calibration PHASE=midgame`: SF-|KS| 1-2 band ratio 0.22,
   **75% read exactly 0** — the floor maps real mid-danger to nothing.

**PHASE-MAPPED PLAN** (validate EACH step on calibration bands + collapse recall + the SAFE-set precision gate;
additive-KS is 0-for-9 in games ⇒ a corpus win is a REASON TO RUN GAMES, never a result):
- **P1 FEEDERS — KEEP** (giant-grade: 96% detection floor-free, AUC 0.91) + our defaware edge. Do not touch.
- **P2 TRANSFORM** — (a) adopt the **QUADRATIC** transform vs our linear `ks_safety_table`; (b) **de-invert the
  queen weight** (`{N2,B2,R3,Q5}` queen-HIGHEST → SF `{N81,B52,R44,Q10}` queen-LOWEST). ⚠️ COUPLED: measured
  `KS_ATT_QUEEN=2` pulls the mild-over-read down (0.05-0.25 ratio 2.24→1.69) but DEEPENS the hole (0.22→0.19),
  because that same weight is the only thing detecting real queen danger. The giants resolve it by routing queen
  danger through **CHECK terms** (low raw queen weight + strong safe-check). ⇒ P2 = de-invert queen AND
  strengthen safe-checks TOGETHER, never the queen knob alone.
- **P3 OUTPUT — DELETE the floor deadzone AND the phase cliff; REPLACE with the giants' attacker+queen entry
  gate.** One subtractive change kills BOTH the 1-2 pawn hole and the endgame blindness. Highest leverage, most
  giant-aligned, and subtractive (the only KS direction that's ever won). First build step.

**CALIBRATION SYMPTOM → PHASE:** mild over-read (0.05-0.5 band, ratio 1.4-2.2) = P2 queen weight · the 1-2 pawn
hole (0.22, 75% zero) = P3 floor · endgame 0.00 = P3 isEndGame cliff. 🧰 `_ks_calibration.py PHASE=` (bands),
`_ks_eg_specificity.py` (collapse recall + SAFE-set precision), `_ks_specificity.py`, collapse tail =
`ks_sets/collapse_dataset_classified.csv` ks_class=`ks_attack` (92). Full memory: [[ks-collapse-tail-is-endgame-gate-blindness]].

---

## 6. DISCIPLINES SPECIFIC TO KS

- **Judge arms on AUC, not on raw detection.** Raw detection is always purchasable with a volume knob.
- **Prove liveness before reading a null.** `KS_PIN_ATT` only ever REMOVES bits, so a zero diff on a bland
  FEN set is not proof it works — verify on a constructed position (e.g. a rook pinning a knight against its
  own king while that knight "attacks" the enemy king zone).
- Colour-symmetry (`_eval_symmetry.py N=800`) is a **ship gate** for any new or modified eval term.
- Byte-identity at the gated default, every build, plus a `wac_speed` peak-NPS read.
- Games decide. Only compare arms whose expected effect exceeds the ~20-40 Elo floor.
