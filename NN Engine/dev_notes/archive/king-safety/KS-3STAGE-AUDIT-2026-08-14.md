# KS 3-Stage Subsystem Audit — 2026-08-14

**Purpose.** (1) A complete Feeders → Transformation → Output/Downstream audit of the king-safety (KS)
subsystem, verified against the working tree and benchmarked against SF11 / SF15.1 / Ethereal (adopting
the PRINCIPLE/shape of the superior system, never a clone). (2) A METHOD trial: the 3-stage scaffold in
§1 is written to be reusable verbatim for any subsystem (threats, passers, OvD, mobility, space…).

Builds on `KING-PHASE-TRANSITION-ANALYSIS-2026-08-14.md` (base + CLEAN-DATA REVISION + ATTACK-SIGNAL
UTILIZATION); the SF/Ethereal constants and the Stage-2 diagnosis live there and are cross-referenced,
not repeated. The NEW content here is the per-component matrix (§2) and the two previously unexamined
stages: Stage 1 feeder accuracy (§3) and Stage 3 downstream/pruning (§4). No engine code was modified.
Line numbers are as-of-today; symbols are the durable cite.

---

## 1. THE REUSABLE 3-STAGE TEMPLATE

Apply to any eval subsystem T. Every finding must land in exactly one stage — if it doesn't fit, the
stage boundaries are being drawn wrong for T, which is itself a finding.

### STAGE 1 — FEEDERS (collection)
*The raw data collected as input to T during the eval run.*
Diagnostic questions:
- **Right squares/pieces?** Is the region (zone/ring/file set) the right shape and size, for edge/corner
  cases too? Does region size vary and is the term normalized for that?
- **Correct predicate definitions?** Are "safe", "weak", "defended", "open", "attacking" defined the way
  the reference defines them — and does OUR definition execute at defaults (check the gate)?
- **Occlusion honesty?** Batteries, x-rays, pins, blocked pawns: does the collection see through /
  exclude what it should? What does the reference see that our masks structurally cannot?
- **Strength inputs present?** Does collection carry piece TYPE / material / side identity forward, or
  flatten it to a count before Stage 2 can use it?
- **Freshness?** Are the shared structures (attack masks, accumulators) populated BEFORE T reads them,
  on every path that calls T?
Failure signature: a feeder error poisons every downstream stage and cannot be fixed by reweighting.

### STAGE 2 — TRANSFORMATION (feeders → number)
*The weights, formulas, curves, and conditioners that turn feeders into T's raw score.*
Diagnostic questions:
- **Weight ordering sane vs reference?** (flat? typeless? inverted?)
- **Accumulation shape?** Sum vs product vs saturating; does coordination compound where it should?
- **Curve shape?** Quadratic/linear/knee/cap/floor — where does the LIVE operating point sit on the
  curve, and what is the marginal derivative there (the number search actually feels)?
- **Conditioners?** Gates on the enabling resources (queen present, attacker count, material backing) —
  scaled to actually gate, or nudges?
- **Phase handling?** Per-signal (mg,eg) shape vs one global taper vs a hard boolean.
Failure signature: right inputs, wrong derivative — magnitude can look fine at the root while the
per-move delta is mispriced.

### STAGE 3 — OUTPUT / DOWNSTREAM (number → score → consumers)
*How T's number reaches `total`, and everything that then CONSUMES a total containing it.*
Diagnostic questions:
- **Netting/scaling/phase-application correct?** Sign convention, side-to-move conditioning, magnitude
  knob, taper/gate at the call site; discontinuities (floors, cliffs) at the output.
- **Which prune gates read it?** Enumerate every search decision that compares the static eval (or a
  search score embedding it) against a margin: RFP, futility, razoring, null-move gates/R-scaling,
  probcut, qsearch stand-pat, improving, corrhist. For each: LIVE at defaults? Which eval mode (full /
  cheap / light — does that mode even CONTAIN T)?
- **Discontinuity vs margin arithmetic:** compare T's largest single-move step (floor crossing, phase
  cliff, cap) against each live margin. A step ≥ a margin means one quiet move can flip that prune.
- **Cross-term reads?** Does any other eval term read T's internals or share T's signal source
  (channel-law coupling)? Does move ordering read it (directly, or via cutoff-seeded history)?
Failure signature: a correct-looking term that distorts search through margins and discontinuities —
invisible to root-eval comparisons, visible only in searched-move instruments.

**Cell verdicts** used in the matrix: ✅ verified-flaw · 🟢 verified-OK · 🔶 hypothesis (mechanism read,
impact unmeasured) · ❓ unexamined.

---

## 2. COMPONENT × STAGE MATRIX — KS

Live-defaults context (all verified in `search_engine.h` today): KS is LIVE — `KING_SAFETY_MAG=3000`
(:1165), `ENABLE_KS_REPLACE_LT=true` (:1170). Live sub-config: `KS_DEFAWARE_MODE=1` (:1286),
`ENABLE_KS_SF_WEAK=true` (:1389), `ENABLE_KS_SF_SAFECHECK=true` (:1391), `KS_FLOOR=13` (:1380),
`KS_NO_QUEEN=6` (:1387), `KS_SAFE_CHECK=3`/`KS_SAFE_CHECK_DEF=5` (:1313/:1322), `MOD_KS_REALIZ=128`
with `KS_REALIZ_FLOOR=128` (:1563-1564), `ENABLE_KS_ROUND_FIX=true` (:782). OFF at defaults: zone
clamp/zone2/norm, pins, SQC, weak-val, flank, check-V2, aim, interact, coord/product, accum-mode,
min-attackers, KS_DYN, KS_EG_MAT_GATE, KS_EXTEND_EG, MOD_KS_BACKING/CONTROL, KS_DEFENDER, KS_WIN_SUP.

Code anchors: `king_safety_danger` cpp_bitboard.cpp:5323-5715 · `king_safety_score` :5723-5735 ·
`evaluate_king_safety` :5746-5798 · tables `rebuild_ks_tables` :409-438 · zone build :625-639 ·
call sites :7571-7584 (midgame), :7882-7885 (`KS_EXTEND_EG` twin, off) · phase :7178-7196.

| Component | S1 Feeders | S2 Transformation | S3 Output/Downstream |
|---|---|---|---|
| King-zone definition (:625-639, :5328-5347) | ✅ corner shrink: raw ring, clamp OFF (§3.1); 🔶 size-varying zone, no norm (§3.2) | 🔶 zone squares priced flat regardless of distance-to-king (fwd staging = ring-1 price); SF prices adjacency separately (69×kingAttacksCount) | 🟢 zone is internal only |
| Attacker detection + weights {N2,B2,R3,Q5} (:5421-5437, :5453-5478) | ✅ no x-ray/battery sight (§3.3); 🔶 pawns not attackers (§3.4); ✅ pinned defenders damp defaware (§3.5) | ✅ queen-HIGHEST inversion vs SF {N81..Q10}/Eth (prior doc §A2); flat sum, no count×weight product (built OFF :5422); defaware1 LIVE reweights by contested fraction — raw popcount contest, not value-aware (KS_SQC_MODE=0) | 🟢 flows only into `units` |
| Safe checks (:5541-5597) | 🟢 geometry + safe-def faithful to SF11 (§3.6); ❓ no unsafe-check channel (SF 148×, §3.6c) | ✅ flat typeless 3/square, unbounded stacking — queenless R+B out-scores a queen attack (prior doc §B); typed saturating V2 built OFF (:5589-5593) | 🔶 `KS_SAFE_CHECK_DEF=5` side-to-move asymmetry ripples into stand-pat parity (§4.5) |
| Weak squares (:5377-5391) | 🟢 SF under-defended predicate LIVE; 🔶 counted over the whole zone incl. forward staging vs SF kingRing-only (§3.7) | ✅ flat `KS_WEAK=2` typeless (SF 185/183 on its ~1500 scale weighs weak ~4× our relative weight; value-coupling built OFF :5391) | 🟢 |
| Zone-attack count (:5410-5414) | 🔶 scales with zone geometry, `KS_ZONE_NORM=0` (§3.2) | 🔶 flat 1/square, type-blind — the quiet-inflation source named in the prior doc §B | 🟢 |
| Pawn shelter / storm (:5507-5509, :5524-5539) | ✅ storm ignores blocked-vs-unblocked (§3.8); 🔶 shield mask = 3 files × 2 ranks, no rank-of-pawn grading | 🔶 flat −2/shield-pawn, +1/advance-rank vs SF's per-file ShelterStrength/UnblockedStorm tables and shelter→danger feedback (−6·mg/8) | 🟢 |
| Open/semi-open files (:5511-5519) | ✅ tests OWN pawns only — an enemy-rammed file scores as exposed (§3.9) | 🔶 flat +2, no open-vs-semi-open split | 🟢 |
| No-queen handling (:5638) | 🟢 correctly per-enemy-side keyed | ✅ flat −6 on a 20-32-unit firing sum = haircut not gate; SF −873 vs threshold 100 (~24 Elo), Eth entry gate — THE confirmed clean-regret defect (prior doc CLEAN-DATA) | 🟢 |
| ks_safety_table curve + floor + cap (:412-424, :5687-5689) | — | ✅ live operating point (units ~14-30) sits entirely on the LINEAR segment (knee 12 < floor 13): marginal = flat 0.18 pawn/unit, the search-integrated derivative | ✅ floor crossing is a ~0.9-1.26-pawn STEP into the static eval (§4.3) |
| MOD_KS_* conditioners (:5750-5782) | 🟢 reads material accumulators (post MATERIAL_COUNT_FIX) | 🟢 `MOD_KS_REALIZ=128` live, damp-only, floor 128 (0.5×) — games-validated (+36.7 bundle); BACKING/CONTROL off; ROUND_FIX shipped | 🔶 realizability applied POST-scaling to the netted ks — damps both kings' netted value, not the fantasy side alone |
| KS_DYN (:5697-5704) | — | ❓ off; realness = att×(open+weak) untested at defaults | — |
| KS_DEFAWARE (:5453-5478) | ✅ contested test inherits pinned-defender blindness (§3.5) | 🟢 mode 1 LIVE (games-leaned bundle 91485c4); graded, redistributive | 🟢 |
| Netting + KS_DEF_MAG (:5723-5735) | — | 🟢 danger_white − danger_black, Black-positive, matches `total` convention; `KS_DEF_MAG=100` inert | ✅ two floors net: each king independently 0-or-≥42-danger ⇒ net can jump ±1.26 pawn on one unit (§4.3) |
| Phase taper + isEndGame cliff (:425-437, :5724, :7185-7203) | — | ✅ single global taper, no per-signal (mg,eg) shape (prior doc §A4/B) | ✅ 71%→0 cliff at phase_score 64→65 (boolean branch truncates the designed fade; `KS_EXTEND_EG=0`) — a whole-term step INSIDE the search tree (§4.4) |
| KING_SAFETY_MAG → total (:5797, :7571-7576) | — | 🟢 ×30 linear; magnitude at root NOT inflated vs SF11 (triangulation, prior doc) | ✅ full static eval (mode 0) feeds RFP/futility/stand-pat — §4 in full |
| Downstream consumers (search_engine.cpp) | — | — | ✅/🔶 first audit: §4. RFP+futility+qsearch live on full eval; null-move does NOT read static eval at defaults; probcut off |

---

## 3. STAGE 1 FINDINGS — feeder accuracy

Overall verdict: the two headline predicates ("weak", "safe check") are FAITHFUL ports of SF11's
definitions and are LIVE — Stage 1 is healthier than expected. The real feeder errors are in the
**geometry** (zone edge cases, size normalization) and in **occlusion honesty** (x-ray/battery, pins,
blocked storms, own-pawn-only file test). None of these is the confirmed queenless over-read (that is
Stage 2), but §3.3/§3.9 systematically mis-collect in exactly the heavy-piece attack shapes.

### 3.1 ✅ Corner/edge kings get a shrunken zone (clamp built, OFF)
Zone = ring-1 + forward push (:626-630): 15 squares for a central king, but a castled g1 king gets 12
and an h1 king 9→~6. SF clamps the ring center to file B..G / rank 2..7 so a corner king keeps a full
ring; our clamped tables exist (:632-639) behind `ENABLE_KS_ZONE_CLAMP=false` (+ `KS_CLAMP_SHELTER`
gate, search_engine.h:1339, :1354). Consequence: attacks on the castled-corner king — the most common
real attack in games — are systematically under-collected relative to attacks on a centralized king.
Direction: makes us UNDER-read the most matey attacks, the opposite tail from the queenless over-read.

### 3.2 🔶 Zone size varies 2.5× and the count feeders don't normalize
`attacked_zone_squares` (+`KS_WEAK`) are raw counts over a 6-15-square region; `KS_ZONE_NORM=0`
(:5410-5414). Ethereal normalizes attack counts to area (`×9/popcount(kingArea)`). Interacts with 3.1:
the corner king's small zone lowers its counts twice.

### 3.3 ✅ No x-ray / battery sight — a whole attack class collected as zero
`attack_bitmasks` are occupancy-limited OR-masks (comment :5599-5604: "attack_bitmasks never x-rays").
A Q-behind-R file battery contributes ONE attacker and no attack units for the rear piece; SF11
computes king-zone slider attacks **x-raying through the side's own queen**, and SF15.1 adds
`RookOnKingRing`/`BishopOnKingRing` (latent aim through pawns). Our two built answers are both OFF:
`ENABLE_KS_AIM` (:5605-5618, single-blocker alignment) and `KS_BATTERY=3` which is **declared but
UNWIRED** — the header comment at :5293 says so, and no code reads `Config::KS_BATTERY`. Consequence:
the canonical heavy-piece file attack (doubled rooks / Q+R) is under-collected — again the
under-read tail, and consistent with the clean finding that Q-ON attacks are where KS *helps* (we may
still be under-reading them at the feeder level).

### 3.4 🔶 Pawns are not attackers
`attacker_units` keys only N/B/R/Q (:5433-5437), and `pieces_nk` (:5404) excludes pawns from every
attacker count (defaware, min-attackers, product, interact). SF seeds `kingAttackers` from pawn
attacks on the ring (SF11 evaluate.cpp:243). Pawn attacks DO enter `attacked_zone_squares` and the
weak test (attack_bitmasks includes pawns), so this is partial: a pawn-storm's attack pressure is
collected as generic square pressure, never as attacker presence/coordination.

### 3.5 ✅ Pinned defenders count as defenders (fix built, OFF)
`dm = bm & own & ~own_pinned` with `own_pinned=0` at `KS_PIN_MODE=0` (:5342-5344, :5366). An
absolutely-pinned "defender" suppresses weak/contested verdicts it cannot actually deliver. SF prices
the same fact positively (`+98 × blockers_for_king`). Feeder-level because it corrupts the
weak/defaware/check-safe predicates, not just a weight.

### 3.6 🟢 Safe-check collection is faithful — with two bounded gaps
(a) Geometry 🟢: check-from squares = king's own N/B/R rays under current occupancy (:5550-5553),
existence-gated per enemy piece type, enemy-occupied squares excluded (`& ~enemy`), queen counted once
per square via the diag/line split (comment :5570-5572) — matches SF11's construction. (b) Safety
predicate 🟢: `check_safe` = zero own cover OR SF's overwhelmed clause (weak + attacked twice), LIVE
via `ENABLE_KS_SF_SAFECHECK=true` (:5560-5565). (c) Gaps: no **unsafe-check** channel at all (SF:
`148 × popcount(unsafeChecks)` — latent checks that exist but aren't safe yet) ❓; and a pinned enemy
piece still generates "checks" (same as SF — attack masks include pinned attacks; not a divergence).

### 3.7 🔶 Weak squares counted over the whole zone, not the ring
The predicate is right (SF's ≤1-defender-and-only-K/Q, LIVE :5380-5381) but it runs over zone =
ring+forward staging; SF's `185 × popcount(kingRing & weak)` restricts weak to the 9-square ring. We
charge "weak king squares" up to two ranks in front of the king — mild systematic over-collection in
open-centre positions, type-blind by Stage 2 anyway.

### 3.8 ✅ Storm collection ignores blockage
`KS_STORM` weights every enemy pawn on the king's 3 files by advance (:5524-5539); SF's shelter model
splits `UnblockedStorm` (dangerous, big table) from `BlockedStorm` (rammed pawn, ~nothing, even
bonus-ish). A locked pawn chain in front of a castled king reads as an oncoming storm. Since shield
(+own pawn) and storm (+enemy pawn) are independent flat terms, the locked-chain case double-fires:
storm units accrue while the shield discount stays flat.

### 3.9 ✅ "Open file" = missing OWN pawn only
:5511-5519 tests `!(own_pawns & BB_FILES[ff])`. A file where OUR pawn is gone but the ENEMY's rammed
pawn still blocks the file scores identical +2 to a genuinely open file; and no distinction from a
half-open-for-attacker file. SF's shelter keys on our pawn's rank AND their pawn's rank per file —
the file's danger is the PAIR. This is a definitional feeder error (the predicate, not the weight) in
one of the highest-frequency components (fires on every castled king with any pawn exchange).

### 3.10 🟢 Freshness
`attack_bitmasks` reset at :7122 and populated by the per-piece eval loops before the KS call in both
branches (midgame :7574; the EG-extension comment :7877-7881 re-verifies for the off path). Sound.

---

## 4. STAGE 3 FINDINGS — output & downstream (first audit of this channel)

### 4.1 The consumer inventory (all verified in search_engine.cpp)

Eval-mode note: `eval_by_mode` (:816-833) — mode 0 = full eval (CONTAINS KS), 1 = cheap
material+PST (NO KS), 2 = light (`g_eval_light` SKIPS KS at cpp_bitboard.cpp:7571 unless
`KS_LIGHT_MAG`>0, default 0). **All three pruning eval modes default 0 = full**, so KS is inside every
static-eval prune at defaults.

| Consumer | Live at defaults? | Reads KS? | Site |
|---|---|---|---|
| RFP (reverse futility) | ✅ `ENABLE_RFP=true`, SHIPPED +73 SPRT (h:1716) | ✅ full eval, `RFP_EVAL_MODE=0` (h:1720); margin 1500 mp/ply (h:1717) | :4537-4551 (min: eval+M≤α), :5194 (max: eval−M≥β) |
| Futility (in-LMR) | ✅ `ENABLE_FUTILITY=true` (h:230) | ✅ full eval, `FUTILITY_EVAL_MODE=0` (h:1702); margins {200,450,650,950} mp for rd 1-4 (h:34) | :3526-3539 (max), :3907 (min) |
| qsearch stand-pat | ✅ | ✅ full eval, `QSTANDPAT_EVAL_MODE=0` (h:1703) | qsearch entry |
| Root razoring | ✅ `ENABLE_RAZORING/ROOT_RAZOR=true` (h:231, :528) | indirect — prunes on prior-iteration SEARCH scores, which embed KS | :2819-2939, :3020-3048 |
| Null-move | ✅ `ENABLE_NULLMOVE=true` | ❌ at defaults: `ENABLE_NULL_EVAL_GATE=false` (h:1730), `ENABLE_NULLMOVE_EVAL_R=false` (h:1866) — gated on material/`isUnsafeForNullMovePruning` only | :4554, :7176 |
| Probcut | ❌ `ENABLE_PROBCUT=false` (h:1737) | (would: qsearch verification embeds KS via stand-pat) | :4662-4668, :5316-5322 |
| Improving / corrhist | ❌ both false (h:412, :1787) | (improving would use `IMPROVING_CHEAP=true` → no KS anyway, h:418) | :785-811, :4532 |
| Move ordering | ✅ | ❌ direct (history/killer/SEE/PST only); ✅ indirect — KS-flavoured cutoffs seed history | move_gen.h |

**Net: the un-audited downstream channel is real and live in exactly three places — RFP, futility,
and qsearch stand-pat — plus razoring one step removed.** Null-move is clean at defaults (a common
assumption worth correcting: our null-move never sees the static eval).

### 4.2 The margin arithmetic — KS steps vs prune margins

KS internal scale at defaults (all derived from :412-424, :5687, :5797): live segment is LINEAR
(knee 12 < floor 13), marginal = 6 danger/unit × 30 = **180 mp = 0.18 pawn per unit** (× taper,
≥71% anywhere the term runs). Table entry at the floor: `ks_safety_table[13]=42` → **crossing
KS_FLOOR is a 0→1260 mp step** (0.9-1.26 pawn after taper). Cap ceiling 444 → 13.3 pawns is
unreachable in practice (live units ~14-30 → 0.9-3.9 pawns pre-net).

| Live margin | Value | KS event that meets/beats it |
|---|---|---|
| Futility rd 1 | 200 mp | **1.2 units** — a single extra zone square + anything |
| Futility rd 2 | 450 mp | 2.5 units — one attacker stepping in (R=3) |
| Futility rd 3/4 | 650/950 mp | 4-6 units — one safe-check square + an attacker |
| RFP, per ply | 1500 mp | floor crossing (~1260×taper ≈ 0.9-1.26 p) ≈ 0.6-0.84 ply of margin |
| Any margin | — | **isEndGame cliff: the WHOLE term (~0.9-1.4 p live) appears/vanishes on one phase point** |

So the confirmed flat ~0.18-pawn/unit derivative does not just reorder root moves — **every 1-3
type-blind units a quiet move adds or removes can flip a futility prune outright at rd 1-2**, and the
two KS discontinuities (floor, phase cliff) each exceed the rd-1..3 futility margins on their own.
This is the concrete mechanism by which a root-magnitude-correct KS still distorts the search: the
prune gates consume the DERIVATIVE and the STEPS, not the root value. (Same reasoning class as
[[never-prune-on-a-score-that-was-never-searched]]: the margins implicitly assume eval continuity
across one move; KS at defaults violates continuity twice.)

### 4.3 ✅ The floor is a netting-amplified step
`KS_FLOOR=13` zeroes each king independently (:5687). Position with both kings at ~12-14 units: one
quiet move nudges either king across, so the NET term jumps by ±1260 mp×taper in one ply. RFP compares
each node's own static eval against bounds propagated from siblings/parents that may sit on the other
side of either king's floor. The deadzone was added to protect quiet positions from KS noise
(the def1 passer bleed) — at the prune gates it does the opposite: it converts smooth 0.18-p/unit
noise into a >1-pawn binary flicker centred exactly on the "is this becoming an attack?" boundary,
the region search most needs priced smoothly.

### 4.4 ✅ The phase cliff runs INSIDE the tree
:7185-7203: one minor trade flips `phase_score` 64→65 → the child's static eval loses the entire KS
term (taper would still pass 71%) while the parent kept it. Every RFP/futility/stand-pat comparison
across that edge carries a systematic ~1-pawn shear whose SIGN equals whichever king was more
endangered — i.e. trades that RELIEVE the defender get mispriced in the defender's favour or against
it depending on netting, at exactly the moment (piece trades under attack) where prune correctness
matters. The prior doc established this cliff as an eval-smoothness defect; the new Stage-3 point is
that it fires at every depth of every search, not once at the root.

### 4.5 🔶 Side-to-move conditioning leaks a tempo-parity ripple into stand-pat
`king_safety_danger` takes `defensive` = side-to-move's own king (:5723-5727), and defensive kings
price safe checks at `KS_SAFE_CHECK_DEF=5` vs 3 (:5595; h:1322 — games-validated categorical gate).
`KS_DEF_MAG` (h:1327) is 100 = inert. Because qsearch stand-pat evaluates at alternating
sides-to-move, the SAME king's danger oscillates ±2 units/check-square (±60 mp each) with ply parity
inside qsearch. Small vs the §4.2 numbers but it is a systematic parity term inside the tightest
margin loop we have. Unmeasured — flagged.

### 4.6 🟢 No cross-term reads of KS internals; the parallel channel is attackingLayer
`g_ks_units_*` is diagnostic-only (:535-536, :5652, :8414-8415) — no eval term consumes it. Nothing
else calls `king_safety_danger`/`king_safety_score` except `evaluate_king_safety` and the
`KS_LIGHT_MAG` surrogate (:7577-7583, inert). The king-credit coupling that DOES exist is upstream and
already known (channel law): `setAttackingLayer`'s king-directed heat feeds `positional_bonus`, the
OvD accumulators and central (cpp_bitboard.cpp:189, :923-929, :1359-1365 …), at 50% since the de-king
ship (`KS_ZONE_ATTACK_PCT`, h:1345). KS reads none of it (MOD_KS_CONTROL would read OvD; off). So
Stage 3 cross-term contamination is CLEAN at defaults; the two king channels meet only in `total`.

### 4.7 🔶 MOD_KS_REALIZ acts post-netting
The one live conditioner (:5769-5782) damps the NETTED ks by the attacking side's material backing.
Because it is applied after `danger_white − danger_black`, a position where BOTH kings read danger has
its net (a difference of two fantasy readings) damped as if it were one attack — the conditioner
cannot distinguish "white's attack is fantasy" from "the net is small". Shape divergence from SF,
where every conditioner lives inside each king's own danger sum. Games said +36.7 as shipped, so this
is a shape note for future redistribution, not a defect claim.

---

## 5. RANKED ISSUES + the clean-regret validation for each

Format: [Stage / verdict / leverage]. Validation instrument = `_regret_tune.py` on
`game_regret_set.csv` (searched-move regret, depth 7, per-position `clearSearchTables()`), band
readouts Qless/Q-on × material, + the ours/SF11/SF18 triangulation for eval-level checks. Per the
channel law all candidates are subtractive/redistributive; every arm byte-id at defaults; mirror test
before any ship.

1. **No-queen gate under-scaled** [S2 / ✅ confirmed in clean data / HIGH]. Already the active queue's
   #1 (prior doc: sweep `KS_NO_QUEEN ∈ {12,20,28,35}`). *Validation:* regret Qless 13-27 & 28+ bands
   must fall; Q-on bands and enriched-critical must not degrade.
2. **Prune-gate exposure to KS discontinuities** [S3 / ✅ mechanism verified, impact unmeasured /
   HIGH — this is the never-audited channel]. Two cheap, orthogonal arms: (a) `FUTILITY_EVAL_MODE=2`
   +`RFP_EVAL_MODE=2` (light eval = KS-free margins; existing knobs, zero new code) — if the queenless
   over-read is substantially prune-transmitted, this arm alone moves the Qless regret bands; (b) keep
   modes 0 but compare `KS_FLOOR=13` vs a smoothed variant (floor as subtraction: `units−13` into the
   table rather than a zero-cliff — needs a small gated knob). *Validation:* regret bands + prune-fire
   counters (`log_prune_fire` exists, :4542) split by phase/queens; then fixed-time games only if the
   regret signal clears the ~20-40 Elo floor.
3. **Typed saturating safe checks — `ENABLE_KS_CHECK_V2=1`** [S2 / ✅ inversion verified / HIGH].
   Queued (prior doc candidate 1); §3.6 confirms the FEEDER under it is sound, so V2's reweight is not
   building on sand. *Validation:* as queued — `KS_CHK_*` sweep, Qless bands down, Q-on guarded.
4. **isEndGame cliff at the prune gates** [S3 / ✅ mechanism / MED-HIGH but BLOCKED]. The fix
   (`KS_EXTEND_EG=1`) is parked as additive-KS-in-the-endgame with no clean harm signal (CLEAN-DATA
   #2). The new Stage-3 evidence (cliff fires at every depth) justifies ONE cheap re-look, not a ship:
   *Validation:* regret on the subset of positions whose depth-7 PV crosses phase 64→65 (label with
   the existing phase instrumentation), arm = `KS_EXTEND_EG=1` alone. If that subset shows no harm
   reduction, the cliff stays parked for good.
5. **Open-file predicate wrong** [S1 / ✅ definitional / MED]. Smallest honest fix: require the file
   also not blocked by an enemy pawn ahead of the shield ranks, or split open/semi-open weights
   (SF pair-of-ranks principle). Needs one gated knob. *Validation:* static first (triangulation on
   locked-chain vs open-file FENs: our ks vs SF11's shelter term), then regret; expect the effect
   concentrated in closed-position over-reads.
6. **Corner-zone shrink — `ENABLE_KS_ZONE_CLAMP=1`** [S1 / ✅ geometry verified, impact 🔶 / MED].
   Under-read of castled-corner attacks; note this is an eval-INCREASING arm on attacked kings, so
   guard the general-play over-fire (the 2026-08-12 detector lesson: discrimination up, deployment
   over-fire). *Validation:* regret on Q-on high-material band (where KS helps — does clamp help
   more?) + the collapse pipeline's ks_attack class rate.
7. **X-ray/battery blindness** [S1 / ✅ gap verified / MED, additive-risk]. Wire nothing new yet:
   `ENABLE_KS_AIM=1` is the built arm; `KS_BATTERY` should be either wired or deleted (a declared,
   unwired knob is exactly the [[env-knob-name-verify]] trap). *Validation:* regret Q-on bands (this
   is under-read of real attacks, so the win would show as Q-on improvement); 0-for-9 law says expect
   failure — run only as a labelled additive-arm experiment.
8. **Pinned defenders / storm blockage / pawn-attackers / weak-over-zone** [S1 / 🔶 / LOW each].
   `KS_PIN_MODE=1` is built; the others need small knobs. Batch as feeder-hygiene arms behind the
   bigger levers; none matches a confirmed harm band today. *Validation:* pooled feeder-hygiene arm
   on regret; discard components that don't move a band.
9. **Stand-pat parity from KS_SAFE_CHECK_DEF** [S3 / 🔶 / LOW]. *Validation:* arm
   `KS_SAFE_CHECK_DEF=3` (restore symmetric) on regret + WAC (the asymmetry was WAC-motivated —
   check it still pays under the clean harness).

**Template debrief (method trial):** the 3-stage split held cleanly for KS — every finding placed in
exactly one stage, and the stages ranked themselves: S2 held the confirmed defect, S1 held faithful
predicates but dishonest geometry/occlusion, S3 held a live, never-measured transmission channel
(margins × discontinuities) that root-level comparisons structurally cannot see. Reusable lesson for
the next subsystem: **audit Stage 3's margin arithmetic FIRST** — it is cheap (a table like §4.2), it
tells you whether the subsystem's errors are amplified or absorbed by search, and it reframes what
Stage 1/2 precision is worth.

*Verified against: cpp_bitboard.cpp (:409-438, :535-536, :599-640, :5283-5798, :7122-7203,
:7571-7584, :7877-7885, :8414-8415), search_engine.h (:34, :230-231, :258, :412-418, :528, :782,
:1150-1404, :1548-1564, :1699-1742, :1787, :1864-1868), search_engine.cpp (:335-451, :780-833,
:1128-1210, :1586-1602, :2819-2939, :3020-3048, :3505-3539, :3907, :4515-4584, :4662-4668,
:5194-5223, :5316-5322, :6146, :7176), and the SF11/SF15.1/Ethereal constants as cited in
KING-PHASE-TRANSITION-ANALYSIS-2026-08-14.md. No engine code modified.*
