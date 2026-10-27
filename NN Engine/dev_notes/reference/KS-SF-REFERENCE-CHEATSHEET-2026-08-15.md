# KS SF/Ethereal reference cheat-sheet — per pattern (2026-08-15)

**Purpose.** A quick-reference to hold against Phase-0A per-pattern eval-loss numbers: for each KS failure
pattern, what SF11/SF15-classical (and Ethereal) does + the constant, how OURS differs, and the fix-handle.

⚠️ **PROVENANCE / verify before shipping any constant.** These constants are consolidated from our OWN
prior analysis docs (KING-PHASE-TRANSITION-ANALYSIS, sf-evolution fact-sheets, the 3-stage audit, the
capability index, the code inventory) — several of which are fable-produced and "re-rank pending." They are
**as-cited, NOT freshly re-verified against SF source.** Treat the specific numbers as strong leads, not
gospel; re-check against SF source before tuning a ported constant. Items the audit could not locate in-repo
are flagged inline. Our-side line numbers are as-of-today hints; the symbol is the durable cite.

Scale note (OURS): danger units 0..`KS_CAP=80`; live point ~14-30 on the LINEAR segment (knee 12 < floor 13)
⇒ 1 unit ≈ 6 danger ≈ **0.18 pawn** after `KING_SAFETY_MAG=3000`, pre-taper/net. SF danger fires only if
`kingDanger>100`, then `score −= S(d²/4096, d/16)` (quadratic mg, small linear eg leak); real attack d≈1500-3000.

---

## 1. Queenless attack / no-queen danger  — THE confirmed clean-regret defect (ranked #1)
- **REFERENCE:** SF11/15.1 subtract **`873 × !count<QUEEN>(Them)`** inside the danger sum (SF15.1 annotated
  **"~24 Elo"** — its single largest annotated danger term); vs the >100 fire threshold this zeroes most
  queenless sums. Ethereal: double gate — `kingAttackersCount > 1 − popcount(enemyQueens)` (needs ≥2 attackers
  when queenless) + `SafetyNoEnemyQueens S(−237,−259)`.
- **OURS:** flat `KS_NO_QUEEN=6` units, pre-floor, per-enemy-side keyed (:5638) — a ~1.1-pawn haircut, NOT a
  gate. Built-OFF: `KS_NQ_SUP=35` (accum), `KS_MIN_ATTACKERS=2` (Ethereal entry gate). Clean-regret: Qless
  13-27 +0.25/+0.37, 28+ +0.19/+0.26.
- **FIX-HANDLE:** sweep `KS_NO_QUEEN ∈ {12,20,28,35}` (subtractive, no rebuild); or `KS_NQ_SUP` under accum.
  Guard the confirmed Q-on-help bands.

## 2. Battery / x-ray king attacks  — UNDER-read tail
- **REFERENCE:** SF11 king-zone slider attacks **x-ray through the side's own queen**; SF15.1 adds latent
  `RookOnKingRing`/`BishopOnKingRing` (aim through pawns).
- **OURS:** ABSENT — `attack_bitmasks` never x-ray (:5599-5604); Q-behind-R = one attacker. `ENABLE_KS_AIM`
  OFF; **`KS_BATTERY=3` DECLARED but UNWIRED** (:5293) — phantom-knob hazard.
- **FIX-HANDLE:** `ENABLE_KS_AIM=1` (built); resolve `KS_BATTERY` (wire or delete). Additive ⇒ 0-for-9 risk,
  run only as a labelled additive experiment.

## 3. Open & semi-open file near king  — definitional feeder error (ranked #5)
- **REFERENCE:** SF keys each file on **BOTH** our pawn's rank AND their pawn's rank (danger = the PAIR);
  `BlockedStorm S(82,82)` distinct from `UnblockedStorm`; open/semi graded.
- **OURS:** `KS_OPEN_FILE=2` where **OWN pawn missing only** (:5511-5519) — an enemy-rammed file scores like a
  truly open one; no open/semi split. Fires on every castled king after any pawn trade.
- **FIX-HANDLE:** require file also not blocked by an enemy pawn ahead of the shield ranks, or split
  open/semi weights. Needs one gated knob. Correction, not addition.

## 4. Safe checks (+ the unsafe-check channel)  — flat/typeless inversion (ranked #3)
- **REFERENCE — safe:** SF11 per type, **once per type**: **Q 780, R 1080, B 635, N 790**. SF15.1 multiplicity
  table N{730,1128} B{650,984} R{1071,1886} Q{805,1292} (+~50-75% for ≥2, saturated) — ordering
  **rook ≥ queen > knight ≈ bishop**. Ethereal per-square typed: SafeKnight S(112,117) SafeRook S(90,98)
  SafeQueen S(93,83) SafeBishop S(59,59). **REFERENCE — unsafe:** SF `148 × popcount(unsafeChecks)`.
- **OURS:** flat **`KS_SAFE_CHECK=3` per SQUARE, TYPELESS**, unbounded stacking (`_DEF=5` own king, :5595);
  geometry + predicate faithful & LIVE. **No unsafe-check channel.** Typed V2 built-OFF (`ENABLE_KS_CHECK_V2`,
  `KS_CHK_Q14/R14/B7/N9 + KS_CHK_MULTI`, :5589-5593). A queenless R+B makes more check squares than a lone
  queen ⇒ flat 3/square can charge the queenless attack MORE.
- **FIX-HANDLE:** `ENABLE_KS_CHECK_V2=1`, sweep `KS_CHK_{Q,R,B,N}` ~{14,14,7,9}, `KS_CHK_MULTI ∈ {0,4,7}`.
  Unsafe-check build: skip (pure add).

## 5. Corner / edge king zone  — UNDER-read (ranked #6)
- **REFERENCE:** SF clamps ring center to file B..G / rank 2..7 (corner king keeps a full 9-sq ring);
  Ethereal area-normalizes (`×9 / popcount(kingArea)`).
- **OURS:** raw ring+forward zone (:626-630): central 15 sq, castled g1→12, h1→~6; `ENABLE_KS_ZONE_CLAMP` +
  `KS_ZONE_NORM` OFF ⇒ under-reads the most common (castled) attacks, hit twice by small zone + no norm.
- **FIX-HANDLE:** `ENABLE_KS_ZONE_CLAMP=1` — eval-INCREASING on attacked kings ⇒ guard general-play over-fire.

## 6. Weak squares
- **REFERENCE:** SF11 **`185 ×`** / SF15.1 **`183 × popcount(kingRing & weak)`**, weak restricted to the 9-sq
  kingRing; Ethereal `SafetyWeakSquares S(42,41)`.
- **OURS:** flat `KS_WEAK=2` typeless over the whole zone (ring+forward), predicate LIVE. SF weighs weak
  ~4× our relative weight. `KS_WEAK_VAL_MODE` (Q3/R2/minor1), `ENABLE_KS_WEAK_ATT2` built-OFF.
- **FIX-HANDLE:** restrict count to the ring and/or route `KS_WEAK_VAL_MODE` through the accum (input only).

## 7. Attacker detection + weighting + coordination  — INVERTED weights (ranked with S2)
- **REFERENCE:** SF11 `KingAttackWeights {N 81, B 52, R 44, Q 10}` / SF15.1 `{N 76, B 46, R 45, Q 14}` —
  **queen LOWEST for proximity**; consumed as the **product `attackersCount × attackersWeight`** (super-linear),
  seeded by pawn ring attacks; adjacency priced separately `69 × kingAttacksCount`. Ethereal dual-valued,
  R/Q collapse ~4-5× mg→eg; count-conditioned entry gate.
- **OURS:** flat **SUM `{N2,B2,R3,Q5}` — queen HIGHEST (INVERTED)** (:5433-5437); adjacency flat
  `KS_ATTACK_COUNT=1`/zone square (:5410, "~85% over-read source"); pawns NOT attackers. No live product
  (`KS_COORD_GATE_MODE`/`KS_ATT_PRODUCT`/`KS_MIN_ATTACKERS` OFF). `KS_DEFAWARE_MODE=1` LIVE (popcount, not
  value-aware).
- **FIX-HANDLE:** de-invert — sweep `KS_ATT_QUEEN 5→{3,2}` (queen danger belongs in typed checks + exists-gate);
  pair `KS_COORD_GATE_MODE`/`KS_MIN_ATTACKERS=2` inside the accum only (standalone product midgame −0.135).

## 8. Pawn shelter / storm
- **REFERENCE:** shelter mg-only, per-file per-rank tables `ShelterStrength[4][rank]` + `UnblockedStorm[4][rank]`,
  split from `BlockedStorm S(82,82)`; `do_king_safety` = MAX over current + both castling destinations, then
  `− S(0, 16·minPawnDist)`; shelter feeds danger as `−6·mg/8`. Ethereal `SafetyShelter/Storm[2][8]`.
  ⚠️ **exact ShelterStrength/UnblockedStorm cell values: constant NOT located in-repo** (only shape + BlockedStorm).
- **OURS:** three flat channels — `KS_SHIELD=2`/pawn (:5507-5509), `KS_STORM=1`/advance-rank (:5524-5539),
  PLUS flat `185/75` shelter in evaluate_kings_midgame (:3116-3186). Storm ignores blocked-vs-unblocked
  (locked chain reads as storm; double-fires). No feedback, no MAX-over-destinations, no per-file grading.
  Re-home lever: `ENABLE_KS_V2` (185/75 identity).
- **FIX-HANDLE:** split `KS_STORM` blocked/unblocked (feeder correction, hygiene batch). Full table = BUILD,
  deferred (shelter is a THIRD live channel ⇒ triple-count unless re-homed via `ENABLE_KS_V2` first).

---

## Cross-cutting facts to hold against the loss numbers
- SF fire threshold **>100**; transform **`S(d²/4096, d/16)`** — the `d/16` **eg leak** ⇒ KS never structurally
  zeroes in the endgame; SF/Ethereal have **ZERO phase gates** on KS.
- OURS: single scalar, no mg/eg split (no eg leak); hard `if(!isEndGame)` + taper ⇒ **71%→0 cliff** at
  phase 64→65; `KS_FLOOR=13` = **0→~1260 mp (≈0.9-1.26 pawn) step**. Both exceed rd-1..3 futility margins
  {200,450,650,950} and feed RFP/futility/qsearch-standpat (all `EVAL_MODE=0` full). `KS_EXTEND_EG` OFF.
- SF15.1 winnability queen-vs-no-queen scale `sf = 37 + 3·minors` (conversion damp); OURS has no
  scale-factor/winnability layer (`ENABLE_ENDGAME_SCALE` OFF).

Sources (as cited by the consolidation): KING-PHASE-TRANSITION-ANALYSIS-2026-08-14.md, KS-3STAGE-AUDIT-2026-08-14.md,
KS-CAPABILITY-GAP-INDEX-2026-08-14.md, KS-CODE-INVENTORY-2026-08-13.md, sf-evolution-fact-sheets-2026-07-05.md.
