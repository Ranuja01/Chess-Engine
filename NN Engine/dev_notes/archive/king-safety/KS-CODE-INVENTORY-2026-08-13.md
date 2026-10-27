# KS code inventory — the existing king-safety system, mapped for the signed-accumulator rebuild (2026-08-13)

**Purpose.** Code-grounded map of every KS_* lever so the `KS_ACCUM_MODE` rebuild (see
`KS-SIGNED-ACCUMULATOR-DESIGN-2026-08-13.md`) REUSES existing levers instead of duplicating them, and does not
re-enable levers that already failed. All citations are current as of this date; symbols are the anchor, line
numbers are a hint.

## 0. Architecture: where KS lives and how it flows

```
placement_and_piece_eval (cpp_bitboard.cpp ~7535-7568)
  ├─ heavy eval: evaluate_king_safety(...)        [gated: ENABLE_KS_REPLACE_LT || KING_SAFETY_MAG!=0]
  │    └─ king_safety_score(wk, bk, phase, turn)  [cpp_bitboard.cpp:5723]
  │         ├─ early-out: phase_score >= KS_PHASE_ZERO -> 0        (the cliff)
  │         ├─ king_safety_danger(sq, white, defensive) per king   [cpp_bitboard.cpp:5323]
  │         │    units = attackers + attack_count + weak + ... − shield − defenders
  │         │    → (old path) KS_MIN_ATTACKERS gate → KS_FLOOR deadzone → ks_safety_table[units]
  │         │    → (KS_ACCUM_MODE=1) net − suppressors → KS_ACCUM_THRESH → linear/square map
  │         │    → optional KS_DYN per-king scaling
  │         ├─ KS_DEF_MAG on the side-to-move king's danger
  │         └─ ks = danger_white − danger_black; ks * ks_phase_taper[phase] / 256
  │    then modulators: MOD_KS_BACKING → MOD_KS_CONTROL → MOD_KS_REALIZ (whole-budget damp)
  │    return KING_SAFETY_MAG * ks / 100                          [cpp_bitboard.cpp:5783]
  └─ light eval (g_eval_light): KS_LIGHT_MAG * king_safety_score / 100   [cpp_bitboard.cpp:7563-7568]

Separate KS-credit CHANNELS outside this object (the channel law):
  - attackingLayer king-directed boost, scaled by KS_ZONE_ATTACK_PCT (cpp_bitboard.cpp:9227,9244,9320)
  - flat mg pawn-shelter constants 185/75 in evaluate_kings_midgame (cpp_bitboard.cpp:3116,3119,3183,3186),
    re-homeable via ENABLE_KS_V2 / KS_CONSOLIDATE
  - OvD accumulators (offensive/defensive scores) — read by MOD_KS_CONTROL but scored elsewhere
```

Tables are rebuilt once at init: `rebuild_ks_tables()` (cpp_bitboard.cpp:412), called from the knob-load block
at search_engine.cpp:1553 — **after** all `env_int("KS_*")` reads (registration block search_engine.cpp:1415-1484),
so KS curve knobs latch correctly. Config-print block: search_engine.cpp:1992-2036 (all new detector + accum
knobs are printed).

## 1. Full knob table

Classes: **ACTIVE** (does work at committed defaults), **INERT** (default = no-op, byte-id off),
**FAILED** (tried and lost — do not re-enable as a standalone), **REUSABLE-FOR-ACCUM** (already implements a
piece the accumulator needs). A knob can be INERT *and* REUSABLE.

### 1.1 Master gates and channels

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `KING_SAFETY_MAG` | **3000** | Final percent scale: `return KING_SAFETY_MAG * ks / 100` (cpp_bitboard.cpp:5783). 3000 ⇒ 1 unit of `ks` = 30 mp (the rounding-bug amplifier, search_engine.h:779). | **ACTIVE** — the master magnitude. Accum's `KS_ACCUM_LIN` is derived for continuity WITH this held fixed. |
| `ENABLE_KS_REPLACE_LT` | **true** | Structural swap: skip `get_latent_threat_score` add, route king danger solely through `king_safety_score` (call site cpp_bitboard.cpp:7535, 7554-7562; header search_engine.h:1166-1170). | **ACTIVE** (shipped swap) |
| `KS_LIGHT_MAG` | 0 | Light-eval surrogate: `total += KS_LIGHT_MAG * king_safety_score / 100` on the `g_eval_light` path (cpp_bitboard.cpp:7563-7568). | INERT |
| `KS_ZONE_ATTACK_PCT` | **50** | Scales the KING-DIRECTED boost in `setAttackingLayer` (`kinc = increment*PCT/100`, cpp_bitboard.cpp:9227; skip loops at 9244/9320 when 0). 50 = the shipped de-triple-count (header search_engine.h:1323-1331: 0 = full de-king REGRESSES; 100 = pre-ship identity). | **ACTIVE** (shipped 50). A live parallel KS channel the accum must NOT double: proximity credit already flows here. |
| `KS_CONSOLIDATE` | false | Gates the flat 185/75 shelter constants in `evaluate_kings_midgame` to 0 (cpp_bitboard.cpp:3116-3186) so shelter is scored once via `KS_SHIELD`. Superseded by ENABLE_KS_V2. | INERT / superseded |
| `ENABLE_KS_V2` + `KS_SHELTER_FULL=185` / `KS_SHELTER_PARTIAL=75` / `KS_SHELTER_MAG=100` | false | Re-homes the flat shelter constants as tunable knobs (same sites, cpp_bitboard.cpp:3116-3186; identity at 185/75/100). | INERT / REUSABLE — if the accum wants shelter as a suppressor in ONE place, this is the existing mechanism to un-home the flat constants (channel law: don't add a shelter suppressor while 185/75 still pay flat). |
| `ENABLE_KS_ROUND_FIX` | **true** | `/256` instead of `>>8` on the signed ks in all three modulators (cpp_bitboard.cpp:5755,5763,5781) — the 30 mp colour-asymmetry fix. | **ACTIVE** (shipped 7-fix bundle) |
| `ENABLE_KS_DEBUG` | false | Diagnostic dump gate (registered search_engine.cpp:1510). | INERT (diagnostic) |

### 1.2 Positive-side inputs to `units` (king_safety_danger)

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `KS_ATT_KNIGHT/BISHOP/ROOK/QUEEN` | 2/2/3/5 | Flat attacker-presence sum over `attackers_sq` (pieces attacking any zone square), cpp_bitboard.cpp:5433-5436. | **ACTIVE**. REUSABLE — the accum's coordination product reuses these same weights (both `KS_COORD_GATE_MODE` at 5424-5427 and `KS_ATT_PRODUCT` at 5644-5647 recompute the identical `att_wsum`). |
| `KS_ATTACK_COUNT` | 1 | **Proximity**: `attack_count_units = KS_ATTACK_COUNT * attacked_zone_squares` (cpp_bitboard.cpp:5410), added into units at 5440. With a 9-12 square zone this is up to ~12 units of a typical real-attack 13-51 — and per the design doc (KS-SIGNED-ACCUMULATOR-DESIGN:40) it is "the ~85% over-read source". | **ACTIVE** — the lever the accum must DEMOTE (design step 0/1). |
| `KS_ZONE_NORM` | 0 | Ethereal density normalization of the proximity count to a reference ring size (cpp_bitboard.cpp:5411-5414). | INERT / REUSABLE if proximity is kept-but-demoted (normalizes the geometric zone-size artifact). |
| `KS_WEAK` | 2 | Per weak zone square (cpp_bitboard.cpp:5441). Weak definition at 5379-5386: with `ENABLE_KS_SF_WEAK` (default **true**) = ≤1 defender and that defender only K/Q; baseline = zero defenders. | **ACTIVE**, REUSABLE (the accum's "weak squares" positive). |
| `ENABLE_KS_SF_WEAK` | **true** | SF11 under-defended weak definition (above). | **ACTIVE** |
| `ENABLE_KS_WEAK_ATT2` | false | attackedBy2 extension: double-attacked ≤1-defender square is also weak (cpp_bitboard.cpp:5384-5386). | INERT / REUSABLE (standard SF/Ethereal form). |
| `KS_WEAK_VAL_MODE` | 0 | Value-coupled weak: each weak square weighted by heaviest attacker Q3/R2/minor1 (`weak_val_sum`, cpp_bitboard.cpp:5387-5391; consumed at 5441). | INERT (detector, gated) / **REUSABLE-FOR-ACCUM** — design names it explicitly (KS-SIGNED-ACCUMULATOR-DESIGN:36). ⚠ 2026-08-12: detector stack over-fires in general play (WHEN failure), so only under the accum's netting. |
| `KS_SAFE_CHECK` | 3 | Per safe-check square vs the ENEMY (offensive) king; flat count path cpp_bitboard.cpp:5595. Detection: check-from squares = king's own N/B/R rays (5550-5553), enemy piece of matching type attacks S, `check_safe(bm)` (see §3.4). | **ACTIVE**, REUSABLE. |
| `KS_SAFE_CHECK_DEF` | 5 | Same but for the side-to-move's OWN king (`defensive` arg, 5595); asymmetric 5 vs 3 games-validated (KS-attack collapse −23%, header search_engine.h:1300-1304). | **ACTIVE** |
| `ENABLE_KS_CHECK_V2` + `KS_CHK_QUEEN=14/ROOK=14/BISHOP=7/KNIGHT=9/MULTI=0` | false | Per-TYPE saturated safe-check weighting (once per type + graded second-square bump), cpp_bitboard.cpp:5589-5593; sized so a lone Q/R check clears KS_FLOOR=13. | INERT / **REUSABLE-FOR-ACCUM** — design step 1 names `KS_CHECK_V2` as the safe-check positive. ⚠ The magnitude route (`SC=8`/CHECK_V2 alone) is called "dead" in the design (:28) — reuse the FORM under the accum, don't re-A/B it standalone. |
| `ENABLE_KS_SF_SAFECHECK` | **true** | check_safe also accepts an overwhelmed square: weak(K/Q-only ≤1 defender) AND ≥2 enemy attackers (cpp_bitboard.cpp:5562-5564). | **ACTIVE** |
| `KS_STORM` | 1 | Enemy pawn-storm: per rank of advance (>1) on the king's three files (cpp_bitboard.cpp:5524-5539). | **ACTIVE**, REUSABLE (accum's "storm" positive already exists). |
| `KS_OPEN_FILE` | 2 | Per own-pawn-free file on/adjacent to the king file (cpp_bitboard.cpp:5511-5519). | **ACTIVE**, REUSABLE (accum's "open-files" positive already exists). |
| `KS_FLANK_MODE` / `KS_FLANK` | 0 / 1 | Flank-attack breadth (SF's strongest discriminator, AUC 0.72): enemy-attacked flank-camp squares, +1 if double-attacked (cpp_bitboard.cpp:5484-5505). MODE 2 = contest-weighted via `ks_sqc_breaks` (5500) — uniquely ours. | INERT / **REUSABLE-FOR-ACCUM** (design picks MODE 2). |
| `KS_PIN_MODE` | 0 | Excludes own-king-pinned pieces from defender masks via `slider_blockers` (cpp_bitboard.cpp:5342-5344; consumed at 5366 and 5500). | INERT / **REUSABLE-FOR-ACCUM** (design's "pins" positive — note it is a defender-mask FILTER, not an additive term; it raises weak/contested counts). |
| `KS_SQC_MODE` | 0 | square_control verdict `ks_sqc_breaks` replaces raw popcount contest for `contested_zone` (cpp_bitboard.cpp:5376). See §3.4 for the helper. | INERT / **REUSABLE-FOR-ACCUM** — the design's step-0 safe-check FEEDER upgrade should route `check_safe` through `ks_sqc_breaks` (currently it does NOT — see §3.4). |
| `ENABLE_KS_AIM` + `KS_AIM_BISHOP=1/ROOK=2/QUEEN=3` | false | Latent slider aim through exactly one blocker (empty-board rays + betweenPieces), cpp_bitboard.cpp:5605-5618. | INERT. Not in the accum design; leave off (latent_threat channel is DEAD per the channel law). |
| `KS_BATTERY` | 3 | Declared per-battery bonus. **UNWIRED** — registered (search_engine.cpp:1464) but never read in cpp_bitboard.cpp (only the comment at 5293: "the one declared knob still unwired"). | DEAD (unwired). Do not assume it does anything. |
| `KS_ZONE2` | 0 | Widen zone to full king_ring2 (cpp_bitboard.cpp:5347). | INERT |
| `ENABLE_KS_ZONE_CLAMP` + `KS_CLAMP_SHELTER=8` | false | Clamped ring center for edge kings, shelter-gated (cpp_bitboard.cpp:5328-5333). | INERT. ⚠ header search_engine.h:1332-1335: the shelter hand-gate "DID NOT help the wrongsign on the corpus". |

### 1.3 Negative-side / suppressors

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `KS_SHIELD` | 2 | Units subtracted per friendly pawn in `white/black_king_shield` (cpp_bitboard.cpp:5507-5509). | **ACTIVE**, **REUSABLE-FOR-ACCUM** — design step 2 explicitly reuses it ("reuse KS_SHIELD as a suppressor into net"). It ALREADY subtracts pre-net, so under KS_ACCUM_MODE it is automatically inside `net` — nothing to build. |
| `KS_DEFENDER` | 0 | Units subtracted per friendly N/B/R/Q in the zone (cpp_bitboard.cpp:5443). | INERT / REUSABLE (a defender-count suppressor exists; SF's −100 king-defended analog). |
| `KS_NO_QUEEN` | 6 | Small no-queen tax: `if (KS_NO_QUEEN && !(queens & enemy) && !KS_ACCUM_MODE) units -= KS_NO_QUEEN` (cpp_bitboard.cpp:5638). | **ACTIVE** (small). See §3.1 — the accum gates it OFF and replaces with `KS_NQ_SUP`. |
| `KS_NQ_SUP` | 35 | Accum no-queen suppressor in units: `if (!(queens & enemy)) net -= KS_NQ_SUP` (cpp_bitboard.cpp:5663). Derived from SF −873/1500 ≈ 58% → ~44% of our 80 (header search_engine.h:1230-1232). | INERT at MODE=0; **the accum's primary review knob** (range 30-46). |
| `KS_WIN_SUP` | 0 | Already-winning discount: `net -= KS_WIN_SUP * own_edge / 1000` when the defending king's side is ahead (cpp_bitboard.cpp:5664-5669). | INERT (isolate-first default 0); accum sub-knob. |
| `KS_FLOOR` | **13** | Blanket deadzone: `if (units < KS_FLOOR) return 0` (cpp_bitboard.cpp:5687), old path only. | **ACTIVE** — the fixed floor the accum's threshold replaces. |
| `KS_MIN_ATTACKERS` | 0 | Ethereal attacker-count gate (≥N pieces, −1 with queen) before the floor (cpp_bitboard.cpp:5680-5684). | INERT / **FAILED as tried** (design doc: every isolated WHEN lever regressed; kept as the gentler variant). The accum's threshold subsumes it. |

### 1.4 Coordination / super-linearity levers

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `KS_COORD_GATE_MODE` / `KS_COORD_DIVISOR` | 0 / 4 | Stage-1 coordination gate: REPLACE flat attacker sum with `(att_pieces * att_wsum) / KS_COORD_DIVISOR` (cpp_bitboard.cpp:5422-5430). | INERT; **Stage-1 tested 2026-08-13: necessary-but-insufficient in isolation** (opening null, midgame −0.135 @div=4 — design doc:3-5). REUSABLE — it IS the accum's "attacker coordination" positive (design:34). |
| `KS_ATT_PRODUCT` | 0 | ADDITIVE product variant: `units += (KS_ATT_PRODUCT * att_pieces * att_wsum) >> 4` (cpp_bitboard.cpp:5642-5649). | INERT / **FAILED family** (additive-KS 0-for-9; header search_engine.h:1213 "kept as a fit-testable variant"). Do not re-enable; KS_COORD_GATE_MODE is the replace-form successor. |
| `KS_OVERLOAD` | 0 | Per-square breakthrough sum `max(0, attackers−defenders)` (computed at cpp_bitboard.cpp:5371-5372, consumed at 5442). | INERT / REUSABLE (a discriminative coordination signal already computed free — `overload_sum` is always calculated). |
| `KS_DEFAWARE_MODE` / `KS_DEFAWARE_COUNT_SHR` | 0 / 0 | Defender-aware attacker weighting: swap presence weights for contested-footprint-scaled weights (fraction or count modes), cpp_bitboard.cpp:5453-5478. Part of the +15-lean bundle candidate (defaware1). | INERT at default but **the deployment comparator** — design validation step 4 compares vs "the defaware bundle". REUSABLE. ⚠ design:80: composition bug to fix — defaware subtracts the ASSUMED flat `legacy_att` (5454-5457), not the `attacker_units` actually computed (wrong when KS_COORD_GATE_MODE changed it). |
| `KS_INTERACT` | 0 | "Coffin" product `undefended * (open_files+1) * attackers >> 4` (cpp_bitboard.cpp:5626-5632). | INERT / **FAILED family** (additive super-linear lever, 0-for-9 additive law). |
| `KS_DYN` / `KS_DYN_PIVOT=4` / `KS_DYN_SHIFT=4` | 0 | Per-king danger rescale by realness `att_cnt*(open_files+weak)−pivot` via mod_gain, cpp_bitboard.cpp:5697-5704. | INERT / **FAILED variant recorded in-code**: folding safe_checks into realness "over-boosted and lost the ksattack gain — disconfirmed at the tuned coefficient" (cpp_bitboard.cpp:5700-5701). |

### 1.5 Curve / phase / accum map

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `KS_DIVISOR` | 4 | Table denominator (cpp_bitboard.cpp:413,418-423). | **ACTIVE** (couples the whole curve) |
| `KS_KNEE` | 12 | Quadratic→linear knee (cpp_bitboard.cpp:415-423). See §3.2: **quadratic band is dead at defaults.** | ACTIVE-but-degenerate |
| `KS_CAP` | 80 | Units clamp inside the table build (cpp_bitboard.cpp:414,421). | **ACTIVE** |
| `KS_FLOOR` | 13 | (see §1.3) | **ACTIVE** |
| `KS_PHASE_FULL` / `KS_PHASE_ZERO` | 48 / 104 | Taper: 256 at ps≤48, linear to 0 at ps≥104 (cpp_bitboard.cpp:425-437); plus the hard early-out cliff at `king_safety_score` (5724). | **ACTIVE** — the cliff the accum's two-function npm blend replaces. |
| `KS_ACCUM_MODE` | 0 | The rebuild master gate (cpp_bitboard.cpp:5655-5674). | INERT (the object under construction) |
| `KS_ACCUM_THRESH` | 13 | Per-position net threshold, seeded at the old floor (5670). | accum sub-knob |
| `KS_ACCUM_LIN` | 96 | Linear map `over * LIN / 16` (5674). 96/16 = 6 = exactly the live linear slope 2·KNEE/DIVISOR (§3.2) — continuity by construction. | accum sub-knob |
| `KS_ACCUM_SQUARE` / `KS_ACCUM_DIV` | 0 / 4 | Conditional square-after-gate `over²/DIV` (5673). | accum sub-knob (conditional on proven discrimination) |
| `KS_DEF_MAG` | 100 | Percent on the side-to-move king's FINAL danger (cpp_bitboard.cpp:5729-5732). 100 = identity. | INERT at 100 (identity), wired. |

### 1.6 Whole-budget modulators (evaluate_king_safety)

| Knob | Default | What it computes | Class |
|---|---|---|---|
| `MOD_KS_BACKING` | 0 | Damp-only material-backing modulator on netted ks (cpp_bitboard.cpp:5750-5757); saturates at MOD_FLOOR=128 (0.5x). | INERT. ⚠ Same function as MOD_KS_REALIZ when floors equal — "do not set both" (search_engine.h:1540). |
| `MOD_KS_CONTROL` | 0 | Scale by attacker's OvD control edge (cpp_bitboard.cpp:5758-5765). | INERT |
| `MOD_KS_REALIZ` | **128** | Whole-budget material-backing damp with its own floor, damp-only (cpp_bitboard.cpp:5769-5782). | **ACTIVE** — shipped in the +36.7 Elo bundle. The accum sits UNDER this: any suppressor the accum adds for "attacker lacks material backing" would DOUBLE with it. |
| `KS_REALIZ_FLOOR` | 128 | Min gain for MOD_KS_REALIZ (5773). | ACTIVE (bounds the above) |

## 2. What is live at committed defaults (the deployment baseline)

Positive units: `KS_ATT_* (2/2/3/5)` flat sum + `KS_ATTACK_COUNT=1` proximity + `KS_WEAK=2` (SF-weak definition)
+ `KS_SAFE_CHECK=3` / `KS_SAFE_CHECK_DEF=5` (SF-safecheck overwhelm clause on) + `KS_STORM=1` + `KS_OPEN_FILE=2`.
Negative: `KS_SHIELD=2`, `KS_NO_QUEEN=6`. Gate: `KS_FLOOR=13` → `ks_safety_table` (affine, §3.2) → taper
(full≤48, zero≥104 + cliff) → `KS_DEF_MAG=100` (identity) → `MOD_KS_REALIZ=128` damp → `KING_SAFETY_MAG=3000`.
Parallel channels: `KS_ZONE_ATTACK_PCT=50` attackingLayer boost; flat 185/75 shelter in evaluate_kings_midgame.
Everything else in §1 is byte-id off.

## 3. Specific questions answered

### 3.1 The no-queen tax: KS_NO_QUEEN vs KS_NQ_SUP
- `KS_NO_QUEEN=6` (search_engine.h:1365-1366) is **active**: `king_safety_danger` subtracts 6 units when the
  ENEMY of this king has no queen — cpp_bitboard.cpp:5638:
  `if (Config::KS_NO_QUEEN && !(queens & enemy) && !Config::KS_ACCUM_MODE) units -= Config::KS_NO_QUEEN;`
  It is applied PRE-floor, on the units scale, so 6 units against a typical 13-51 real-attack range is a
  ~12-46% haircut at the low end but nowhere near SF's 58%-of-max silencing — which is why queenless
  false-attacks survive (design doc:46-47).
- `KS_NQ_SUP=35` is the accum-scale replacement (cpp_bitboard.cpp:5663, inside `if (Config::KS_ACCUM_MODE)`),
  sized as SF's FRACTION ported to our 80-cap scale. **Mutual exclusion is already coded**: the `&&
  !Config::KS_ACCUM_MODE` on line 5638 gates the small tax off when the accum owns the signal (comment
  5636-5637: "to avoid double-counting"). Both key on the same predicate `!(queens & enemy)` (enemy of THIS
  king). No duplication risk remains here — it is handled.

### 3.2 The curve: rebuild_ks_tables and the DEAD QUADRATIC — CONFIRMED
`rebuild_ks_tables` (cpp_bitboard.cpp:412-438) builds
`ks_safety_table[u] = (c<=KNEE) ? c²/DIVISOR : KNEE²/DIVISOR + (2·KNEE/DIVISOR)·(c−KNEE)`, c = min(u, KS_CAP).
At defaults KNEE=12, DIVISOR=4, CAP=80: knee_val=36, slope=6.
**The quadratic band covers u ≤ 12, but `king_safety_danger` returns 0 for `units < KS_FLOOR=13`
(cpp_bitboard.cpp:5687) before the table is read.** So every table read has units ≥ 13 > KNEE and lands in the
linear branch: effective live curve = `danger = 36 + 6·(units − 12)` for units 13..80, i.e. **pure affine —
the "non-linear safety table" is a straight line in deployment. Dead-quadratic claim CONFIRMED** (holds
whenever FLOOR ≥ KNEE+1; here 13 ≥ 13). Note `KS_ACCUM_LIN=96` gives slope 96/16 = 6 — the accum's linear map
default deliberately reproduces this live slope.
The same function builds `ks_phase_taper` (425-437): 256 at ps ≤ FULL=48, 0 at ps ≥ ZERO=104, linear ramp
between, with a degenerate hard-step branch if ZERO ≤ FULL.

### 3.3 Proximity: attack_count_units
`attack_count_units = Config::KS_ATTACK_COUNT * attacked_zone_squares` (cpp_bitboard.cpp:5410), where
`attacked_zone_squares` increments once per zone square with any enemy attacker (5368-5370); optional
`KS_ZONE_NORM` rescale (5411-5414, off). Added directly into `units` (5440). The zone is ring1+one-rank
(`white/black_king_ks_zone`), ~9-12 squares, so proximity alone contributes up to ~12 of a typical 13-51 —
enough to CLEAR the floor with zero genuine-danger signals. The design doc (:22-24, :40) identifies it as
"the ~85% over-read source"; the 2026-08-09 leverage map's +2.01-pawn over-read is dominated by this channel.
This is the term the accum DEMOTES (demote vs remove = open review question, design:96-97).

### 3.4 Safe-checks: check_safe and square_control
`check_safe` lambda, cpp_bitboard.cpp:5560-5565:
```cpp
auto check_safe = [&](uint64_t bm) -> bool {
    uint64_t dmS = bm & own;
    if (!Config::ENABLE_KS_SF_SAFECHECK) return dmS == 0;
    bool weakS = (popcount(dmS) <= 1) && ((dmS & (knights|bishops|rooks|pawns)) == 0);
    return dmS == 0 || (weakS && popcount(bm & enemy) >= 2);
};
```
It is the crude **presence test**: any own piece covering the square (even a lone pawn vs a queen-check)
defeats "safe", except the SF overwhelm clause (K/Q-only ≤1 defender AND attacked twice). **It does NOT read
`ks_sqc_breaks`/square_control anywhere** — `ks_sqc_breaks` (5316-5322: undefended, OR cheapest-attacker <
cheapest-defender LVA, OR ≥2 attackers vs ≤1 defender) is currently consumed only by `contested_zone` under
`KS_SQC_MODE` (5376) and `KS_FLANK_MODE=2` (5500). Routing `check_safe` through `ks_sqc_breaks` is exactly the
design's un-tried "safe-check FEEDER upgrade" (design:25-28) — the helper already exists two hundred lines
above the lambda; the change is a one-line predicate swap behind a mode knob. Note the defender mask in
`check_safe` uses `bm & own` WITHOUT the `~own_pinned` filter the zone scan uses (5366) — a pinned "defender"
still defeats a safe check; folding `own_pinned` in is a second free honesty upgrade.

### 3.5 Phase taper + the KS_PHASE_ZERO cliff
`king_safety_score` (cpp_bitboard.cpp:5723-5735): line 5724 `if (phase_score >= Config::KS_PHASE_ZERO) return 0;`
is a hard early-out CLIFF at ps=104 (also a speed win); below it, `ks * ks_phase_taper[phase_score] / 256`
applies the single linear taper. Consequence: R/Q endgames past ps 104 get ZERO king safety even with a mating
attack on the board — the under-gating the design's two-function npm blend (design step 5) removes. Both
king dangers are computed with the `defensive` flag = side-to-move (5726-5727), feeding KS_SAFE_CHECK_DEF and
KS_DEF_MAG.

### 3.6 Magnitude-COUPLED knobs (change one ⇒ re-derive the others)
1. **`KS_ATT_*` / `KS_ATTACK_COUNT` / `KS_WEAK` / `KS_SAFE_CHECK*` / `KS_STORM` / `KS_OPEN_FILE` / `KS_SHIELD`
   / `KS_NO_QUEEN` ↔ `KS_FLOOR` (and `KS_ACCUM_THRESH`)**: the floor/threshold is a number ON the units scale;
   any reweighting of the positive side moves what clears it. The `KS_CHK_*` defaults are explicitly "sized so
   a LONE queen/rook safe-check clears KS_FLOOR (13)" (search_engine.h:1294-1296) — change the floor, those
   are stale.
2. **`KS_KNEE` ↔ `KS_DIVISOR` ↔ `KS_CAP` ↔ `KS_FLOOR`**: the live slope is 2·KNEE/DIVISOR and the floor
   decides whether the quadratic band exists at all (§3.2).
3. **`KING_SAFETY_MAG` ↔ everything**: 3000 makes 1 unit of final ks = 30 mp; every unit-scale choice is
   really a 30 mp choice. `KS_ACCUM_LIN` is derived "for rough magnitude-continuity ... KING_SAFETY_MAG holds
   final scale" (search_engine.h:1239-1241).
4. **`KS_COORD_DIVISOR` ↔ `KS_ATT_*`**: the product divisor is chosen to keep the product's scale near the
   flat sum for typical attacker counts (comment 5428-5429); reweight the attackers and 4 is stale.
5. **`KS_NQ_SUP` / `KS_WIN_SUP` / `KS_ACCUM_THRESH` ↔ the demoted-proximity positive side**: the design's own
   coupling law (design:28-31) — "a −35 no-queen only means something relative to the demoted-proximity
   positive size"; size them JOINTLY from the unit-trace.
6. **`MOD_KS_REALIZ`/`KS_REALIZ_FLOOR` ↔ any material-keyed suppressor** (`KS_WIN_SUP`): both damp by material
   edges; REALIZ keys on the ATTACKER's deficit, WIN_SUP on the DEFENDER's surplus — related but not identical
   signals, and both active at once needs a joint read (§4.3).
7. **`KS_DEF_MAG` ↔ `KS_SAFE_CHECK_DEF`**: two levers that both boost the defensive king's danger (one on
   final danger, one on a unit input); tuned together historically.

## 4. DUPLICATION RISKS — where the accum design must reuse, not re-implement

1. **Coordination product**: three implementations of `att_pieces × att_wsum` already exist —
   `KS_COORD_GATE_MODE` (replace-form, cpp_bitboard.cpp:5422-5430), `KS_ATT_PRODUCT` (additive FAILED form,
   5642-5649), and `KS_MIN_ATTACKERS` (count-only gate, 5680-5684). The accum's "attacker coordination"
   positive must BE `KS_COORD_GATE_MODE` (the design says so, :34); do not add a fourth. Consider retiring
   KS_ATT_PRODUCT / KS_MIN_ATTACKERS from consideration explicitly.
2. **No-queen**: already dual-implemented WITH the mutual-exclusion gate (§3.1). Nothing to build; only the
   KS_NQ_SUP magnitude is open.
3. **Shelter**: THREE shelter channels exist — `KS_SHIELD` (inside units, already a net suppressor), the flat
   185/75 constants in `evaluate_kings_midgame` (cpp_bitboard.cpp:3116-3186), and the un-homed `ENABLE_KS_V2`
   knobs. The design's "reuse KS_SHIELD as a suppressor into net" is ALREADY TRUE mechanically (it subtracts
   before the accum reads `units`, 5509 → 5662). Risk: adding a NEW shelter suppressor while 185/75 still pay
   flat would triple-count; if shelter needs to be bigger in the accum, the lever is ENABLE_KS_V2 re-homing,
   not a new term.
4. **Already-winning vs MOD_KS_REALIZ**: `KS_WIN_SUP` (defender's surplus, pre-map units, 5664-5669) and the
   ACTIVE `MOD_KS_REALIZ=128` (attacker's deficit, post-map budget damp, 5769-5782) plus the inert
   `MOD_KS_BACKING` all condition danger on material. When KS_WIN_SUP is turned on, validate against the LIVE
   REALIZ damp (the channel law: sign depends on live channels) — and never enable MOD_KS_BACKING alongside
   REALIZ (search_engine.h:1540).
5. **Overload / defaware / SQC contested**: `overload_sum`, `contested_zone`, and `breakthrough_sq` are
   computed unconditionally in the zone scan (5354-5376). Any accum "weak-square/breakthrough" input should
   read these existing accumulators, not rescan the zone. The defaware composition fix (subtract the actual
   `attacker_units`, design:80) belongs in the existing block at 5453-5478.
6. **Safe-check honesty upgrade**: reuse `ks_sqc_breaks` (5316) inside `check_safe` (5560) — see §3.4. Do not
   write a new LVA helper; `ks_lva` (5301) is the canonical, colour-blind one.
7. **Threshold**: `KS_ACCUM_THRESH` replaces BOTH `KS_FLOOR` and the inert `KS_MIN_ATTACKERS` gate. The old
   path keeps them verbatim behind MODE=0; nothing else should read them in MODE=1.
8. **Phase**: the accum's npm blend replaces the taper + cliff, but the cliff early-out at 5724 lives in
   `king_safety_score` ABOVE `king_safety_danger` — the MODE=1 path must handle the ps≥104 early-out
   explicitly or endgame KS stays dead regardless of the eg branch.
9. **Proximity double-pay**: proximity credit also flows through the ACTIVE `KS_ZONE_ATTACK_PCT=50`
   attackingLayer channel (9227). Demoting proximity inside units does not touch that channel — remember the
   channel law before concluding "proximity removed".
10. **Diagnostics**: unit traces already exist — `g_ks_units_white/black` (535-536, set at 5652), `KS_TRACE`
    (5393), `KS_SAFECHK_TRACE` (5566), `KS_DEBUG_DUMP` (5707) — all behind `g_capture_eval_breakdown` first.
    The design's unit-trace pass should extend `KS_DEBUG_DUMP`, not add a new probe.

## 5. FAILED ledger (do not re-enable standalone)
- **Additive KS levers, 0-for-9 in games** (memory: ks-twelve-attempt-history): `KS_ATT_PRODUCT`,
  `KS_INTERACT`, and the additive magnitude route generally (`SC=8`/`CHECK_V2` standalone, design:28).
- **`KS_DYN` with safe_checks folded into realness** — disconfirmed at the tuned coefficient (in-code,
  cpp_bitboard.cpp:5700-5701).
- **`KS_MIN_ATTACKERS`** — the isolated WHEN gate; superseded by the accum threshold.
- **`KS_CLAMP_SHELTER` hand-gate** — did not help the wrongsign corpus (search_engine.h:1332-1335).
- **`KS_ZONE_ATTACK_PCT=0` (full de-king)** — REGRESSES (search_engine.h:1327-1328); 50 is the optimum found.
- **Detector stack standalone** (SQC/pins/weak-val/flank, 2026-08-12): discrimination AUC 0.748→0.810 but
  over-fires general play (OPENING +0.37) — a WHEN failure; only valid as accum inputs.
- **Stage-1 coordination gate standalone** (2026-08-13): opening null, midgame −0.135 @div=4 —
  necessary-but-insufficient in isolation (design doc header).
- **`MOD_KS_BACKING` + `MOD_KS_REALIZ` together** — same function at equal floors; never both
  (search_engine.h:1533-1540).
