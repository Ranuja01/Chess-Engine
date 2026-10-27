# Stockfish 11 classical eval — term-by-term reference

Reference for the last pre-NNUE Stockfish (`sf_11`, `evaluate.cpp` / `material.cpp` / `pawns.cpp` / `endgame.cpp`).
Our yardstick: apples-to-apples HCE. Captured so we never re-research. Constants are SF11-era `S(mg,eg)` (packed
midgame/endgame, tapered by phase at the end). **We do NOT distill SF's NNUE** — SF11 classical is the in-spirit
reference. Algorithms are stable; exact constants drift across versions (verify against the checkout if diffing numbers).

Pipeline: `material + imbalance + pawns(cached) + pieces() + mobility() + king() + threats() + passed() + space()`,
then `initiative()` nudges the raw score, then `scale_factor()` scales ONLY the endgame component, then interpolate by phase.

## Material + Imbalance — the key "what does Imbalance add over a material sum"
- **Material** = trivial sum of piece values.
- **Imbalance** (`material.cpp`, Tord Romstad's 2nd-degree polynomial, Kaufman-style): two 6×6 tables `QuadraticOurs`
  / `QuadraticTheirs` indexed by piece type, where **index 0 is a synthetic "bishop pair" pseudo-piece**, 1–5 =
  pawn/knight/bishop/rook/queen. `bonus += ourCount[pt1] * Σ_pt2 (QuadraticOurs[pt1][pt2]*ourCount[pt2] +
  QuadraticTheirs[pt1][pt2]*theirCount[pt2])`; `imbalance = (bonus(W)-bonus(B))/16`.
- **Captures what a per-piece constant CANNOT** (quadratic/pairwise, not linear): **bishop-pair** super-additive
  bonus (`QuadraticOurs[0][0]=1438`, fires only with 2 bishops); **knight↔pawn synergy** (knights gain value with more
  pawns / closed positions); **rook/queen redundancy** (2nd rook worth less — diminishing returns on duplicates);
  **opponent-material-relative exchange values** ("trade when ahead", rook worth more vs fewer defenders).
- Cost: trivial — two table walks, computed once per material config, cached in the material hash.

## Mobility
- `mobilityArea[Us]` = all squares EXCEPT: own blocked/low-rank pawns, own **king and queen**, pieces pinned to own king
  (`blockers_for_king`), and squares attacked by **enemy pawns**. `~(blockedPawns | pieces(Us,KING,QUEEN) | king_blockers | pawn_attacks(Them))`.
- Per N/B/R/Q: `mob = popcount(attacks & mobilityArea[Us]); mobility += MobilityBonus[Pt-2][mob]` — a `[4][32]`
  **non-linear** `S(mg,eg)` table (diminishing returns). Sliders x-ray through own queen/rook.
- Cost: cheap — the attack bitboard is already computed in `pieces<>()` for threats/king-ring; mobility is 1 AND +
  popcount + table lookup per piece.

## Threats — SF11 has NO named-motif detectors
**Critical:** no fork/skewer/discovered-attack/pin *finders*. `threats()` scores only STATIC threat relationships and
leaves real tactics to SEARCH. It scores: **weak enemies** (non-pawn, attacked, not strongly defended); **Hanging**
`S(69,36)` (weak + undefended-or-attacked-more-than-defended); **ThreatByMinor[pt]** (knight/bishop attacks a weak piece,
scaled by target: Pawn`S(6,32)`..Rook`S(90,119)`); **ThreatByRook[pt]**; **ThreatByKing** `S(24,89)`; **ThreatBySafePawn**
`S(173,94)` (a safe pawn attacks a piece — the nearest thing to a "pawn fork", scored per-target not as a motif);
**ThreatByPawnPush** `S(48,39)` (a pawn that can safely advance to attack); **RestrictedPiece** `S(7,7)`; **KnightOnQueen**
`S(16,12)`; **SliderOnQueen** `S(59,18)`. Pin-adjacent = `WeakQueen` + `blockers_for_king` (structural, not motifs).
**Lesson for us:** do NOT build motif detectors — SF doesn't; search + qsearch do tactics.

## Initiative (endgame winnability nudge)
Second-order correction to the whole-board score (not per-piece). `complexity = 9*passedCount + 11*pawnCount +
9*outflanking(|file dist of kings|) + 12*infiltration(king on rank≥4) + 21*pawnsOnBothFlanks + 51*(no non-pawn material)
- 43*almostUnwinnable - 100`. Then capped so it can shift but NOT flip a decisive score: mg add = `sign(mg)*max(min(cx+50,0),-|mg|)`,
eg add = `sign(eg)*max(cx,-|eg|)`. Estimates HOW WINNABLE a position is for the side ahead (pawns on both flanks / passers
/ king activity → more winnable; one-flank / near-symmetric → toward draw). Matters most in the endgame.

## King safety (`king<>()`)
During `pieces<>()` accumulate, per enemy attacker of the king ring: `kingAttackersCount`, `kingAttackersWeight`
(`KingAttackWeights`: N 81, B 52, R 44, Q 10, P 0), `kingAttacksCount` (attacks on squares adjacent to king). Then a
non-linear `kingDanger` "attack units" scalar: `attackersCount*attackersWeight + 185*popcount(kingRing&weakSquares) +
148*unsafeChecks + 98*knightBlockers + 69*kingAttacksCount + 3*kingFlankAttack²/8 + mg(mobility diff) - 873*(no enemy queen)
- 100*(defend all knight checks) - 6*mg(score)/8 - 4*kingFlankDefense + 37`. **Safe checks** weighted (Rook 1080, Queen 780,
Knight 790, Bishop 635). Weak squares = enemy-attacked, defended ≤ once (only by K/Q). Then the **quadratic conversion**: if
`kingDanger>100`, `score -= make_score(kingDanger²/4096, kingDanger/16)` — attacks compound super-linearly. Plus flat
`PawnlessFlank S(17,95)`.

## Passed pawns (`passed()`)
Per passed pawn: **rank bonus** `PassedRank[relRank]` (dominant); **king proximity** (eg, `w=5*rr-13` weighting own king near
/ enemy king far from the advance square); **path clearance** (`squaresToQueen` vs `unsafeSquares` = defended-by-us vs
attacked/occupied-by-enemy — boost if whole path clear+safe, reduce if front square attacked/occupied); **support** (block
square defended); **rook/queen behind the passer**; **hindered/candidate scaling** (roughly halve if not cleanly passed).
(doubled/isolated/backward/connected/phalanx structure is separate in `pawns.cpp`, cached in the pawn hash.)

## Space (`space()`)
0 unless non-pawn material ≥ `SpaceThreshold 12222` (early/middlegame only). SpaceMask = files C–F, ranks 2–4 (W) / 5–7 (B).
`safe` = SpaceMask squares not occupied by own pawns and not attacked by enemy pawns; `behind` = 1–3 ranks behind own pawns.
Bonus = `popcount(safe) + popcount(behind & safe & ~enemyAttacks)`, weighted ≈ `(pieceCount-1)²/16` (more pieces → more space
value). Middlegame-only Score.

## Per-piece placement (`pieces<>()`)
**Rooks:** `RookOnFile[semi/open]`; `RookOnQueenFile`; `TrappedRook` (rook mobility ≤ 3, **no castling rights that side**, own
king blocks the flank — doubled if truly stuck). Rook-behind-passer handled in `passed()`.
**Knights/Bishops:** `Outpost` (knight double-weight / bishop on a square defended by own pawn, not attackable by an enemy
pawn, in enemy half); `ReachableOutpost`; `MinorBehindPawn`; `BishopPawns` (**bad-bishop** penalty ∝ own pawns on the bishop's
colour, worse if blocked/central); `LongDiagonalBishop`; `CorneredBishop` (Chess960).
**Queen:** `WeakQueen` (pinned / discovered-attack exposed — SF's closest thing to a "pin", structural).

## Endgame scale factors (`endgame.cpp`)
Tapered eval's **endgame component ×= scaleFactor/64** (0 = draw .. 64 normal .. up to `SCALE_FACTOR_MAX`). Specialized
functions via the material hash + a generic fallback: **opposite-coloured bishops** (KBPsK / general OCB → heavily toward
draw; `KBPsK` = draw when all pawns on a rook file + defender holds queening square with wrong bishop); **KRPKR** (Philidor /
back-rank; draw for defended-king-on-queening-square + rook-checks-from-behind; graduated `MAX - 8*dist(pawn,queenSq)`
otherwise); **single-pawn / rook-pawn** wrong-corner draws; **KQKRPs** fortress; **generic** down-scaling of few-pawn /
one-pawn / opposite-bishop positions so the search doesn't overvalue technically-drawn material edges.

Sources: `official-stockfish/Stockfish@sf_11` (evaluate.cpp, material.cpp, pawns.cpp, endgame.cpp); chessprogramming.org.
See [[our_eval_reference.md]] for the ours↔SF correspondence and KEEP/BUILD/RETIRE status.
