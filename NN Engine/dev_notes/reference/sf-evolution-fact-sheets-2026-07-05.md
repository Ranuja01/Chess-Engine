# SF1 → SF11 → SF18 source fact sheets (mined 2026-07-05, for source-verification)

Provenance: SF11 sheet mined from `stockfish_11/stockfish-11-win/src/` on disk; SF16/17/18 sheet mined from
`stockfish_16`, `stockfish_17`, `stockfish_18_linux/src/` on disk (18 contains full source and is the modern
reference). **Note: the on-disk `stockfish_1/src` is NOT SF 1.x — it is a modern NNUE-era dev tree (©2004-2026);
SF1 facts below were read from the official repo's `sf_1.0` tag online (evaluate.cpp, search.cpp, bitboard.h).**
File anchors below are relative to each tree's src/. Everything here is raw extraction; the strategic synthesis
is in the chat answer of 2026-07-05.

---

## SF 1.0 (2008) — from sf_1.0 tag online

**Eval (evaluate.cpp):** attack-unit king safety ALREADY present:
`attackUnits = min((kingAttackersCount*kingAttackersWeight)/2, 25) + (kingAdjacentZoneAttacksCount + count(undefended))*3 + InitKingDanger[sq] - shelter/32` → `SafetyTable[100]`; attack weights Q=5 R=3 B=2 N=2; safe-check bonuses (Q contact 4, R contact 2, …). Mobility tables per piece (N9/B16/R16/Q32 mg+eg) with NO mobility-area restriction. NO threats section, NO space, NO tempo, imbalance delegated to MaterialInfoTable (no polynomial). Passers with king distance + unstoppable detection. Rook open/half-open 40/20, trapped rook, outposts (N/B 64-sq tables), 7th-rank bonuses, trapped-bishop a7/h7.

**Movegen (bitboard.h):** plain magic bitboards already (`RMult/RShift/RMask/RAttacks[0x19000]`, `(b*RMult[s])>>RShift[s]`), SWAR popcount, optional 32-bit folded variants. NOT rotated.

**Search (search.cpp):** null move R=4 FIXED + zugzwang verification (depth≥6) + `NullMoveMargin 0x300`; LMR (reduce 1 ply, killer/capture/promo/passed-push exempt, LMRPVMoves/LMRNonPVMoves thresholds); futility margins 0x80/0x100/0x300; razoring depth≤4 margin 0x300; IID (PV d≥5, nonPV d≥8); killers[2]; butterfly-style History; SEE-prune in qsearch; check ext 1 ply, pawn-to-7th ½, single-reply ½, pawn-endgame 1, mate-threat 0. ABSENT: aspiration, singular, ProbCut, countermoves, LMP/move-count pruning, static-null/RFP, eval-coupled null R.

---

## SF 11 (Jan 2020) — last classical peak (mined from disk)

### Evaluation
- **King safety** (evaluate.cpp:370-474): `KingAttackWeights={0,0,81,52,44,10}` (N,B,R,Q at idx 2-5). kingRing = king clamped to B-G/2-7 ± ring, minus squares defended by TWO pawns. weak = attacked & !attackedBy2(us) & (!attacked(us) | only K/Q defend). Safe-check constants: Q=780 R=1080 B=635 N=790 (rook counted first; queen excluded from squares that give rook check).
  Final:
  `kingDanger = count*weight + 185*pop(kingRing&weak) + 148*pop(unsafeChecks) + 98*pop(blockersForKing) + 69*kingAttacksCount + 3*flankAttack²/8 + mg(mobility[Them]-mobility[Us]) - 873*!enemyQueen - 100*(knight defends king ring) - 6*mg(score)/8 - 4*flankDefense + 37;`
  applied only `if (kingDanger>100)`: `score -= S(kingDanger²/4096, kingDanger/16)`. Plus `PawnlessFlank S(17,95)`, `FlankAttacks S(8,0)*flankAttack`.
- **Shelter/storm** (pawns.cpp:185-215): ShelterStrength[4][rank] + UnblockedStorm[4][rank] (tables in sheet source), BlockedStorm S(82,82) only at their-rank-3; do_king_safety takes MAX over current square and both castling destinations, then −S(0,16·minPawnDist).
- **Mobility area** (evaluate.cpp:230): `~( blocked-or-rank2/3 own pawns | own K,Q | blockers_for_king(us) | enemy pawn attacks )`; MobilityBonus[N≤9/B≤14/R≤15/Q≤28].
- **Threats** (479-568): ThreatByMinor[victimPT]={0,S(6,32),S(59,41),S(79,56),S(90,119),S(79,161)}; ThreatByRook={0,S(3,44),S(38,71),S(38,61),S(0,38),S(51,38)}; ThreatByKing S(24,89); Hanging S(69,36); RestrictedPiece S(7,7); ThreatBySafePawn S(173,94); ThreatByPawnPush S(48,39); SliderOnQueen S(59,18) (needs attackedBy2), KnightOnQueen S(16,12); WeakQueen S(49,15).
- **Passers** (573-651): PassedRank={0,S(10,28),S(17,33),S(15,41),S(62,72),S(168,177),S(276,260)}; w=5r−13; eg king-proximity ±; path-clear k∈{35,20,9,0} (+5 if defended/rook-behind); candidate halved; PassedFile S(11,8)·file-from-edge.
- **Pieces**: Outpost S(30,21) (×2 N), ReachableOutpost S(32,10), MinorBehindPawn S(18,3), KingProtector S(7,8)·dist, BishopPawns S(3,7)·sameColorPawns·(1+blockedCenter), LongDiagonalBishop S(45,0), RookOnFile {S(21,4),S(47,25)}, TrappedRook S(52,10)·(1+!castling), RookOnQueenFile S(7,6).
- **Imbalance** (material.cpp): Tord polynomial, QuadraticOurs/Theirs 6×6 (bishop-pair as piece[0]), /16.
- **Space** (661-691): only if npm ≥ 12222; center files ranks 2-4; bonus = safe + (behind∩safe∩unattacked); `score = bonus·weight²/16 mg` where weight = pieceCount−1.
- **Initiative** (698-737): `complexity = 9·passedCount + 11·pawnCount + 9·outflanking + 12·infiltration + 21·pawnsOnBothFlanks + 51·!npm − 43·almostUnwinnable − 100`; `u = ±max(min(complexity+50,0),−|mg|)`, `v = ±max(complexity,−|eg|)` — sign-preserving damp/boost toward/away from 0.
- Tempo=28; LazyThreshold=1400 (+npm/64); OCB scale sf=22, else min(sf, 36+(opp?2:7)·strongSidePawns); rule50 fade.

### MovePicker (movepick.cpp)
Main: MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE → REFUTATION (killers+countermove) → QUIET_INIT → QUIET → BAD_CAPTURE. QSearch: TT → QCAPTURE → QCHECK. Captures: `PieceValue[MG][victim]·6 + captureHistory[pc][to][victimType]`. Quiets: `mainHist + 2·contHist[0] + 2·contHist[1] + 2·contHist[3] + contHist[5]`. Good-capture split: `see_ge(m, −55·score/1024)`. Quiet partial insertion sort limit `−3000·depth`.

### TT (tt.h/tt.cpp)
10-byte entry {key16, move16, value16, eval16, genBound8(5 gen|1 pv|2 bound), depth8}; cluster=3+2pad=32B; index `(uint32(key)·clusterCount)>>32`; save keeps old move if none; overwrite if new-key OR depth>old−4 OR EXACT; probe-replacement victim = min(depth8 − 8·relativeAge); generation += 8/search; prefetch on key_after(move).

### Search (search.cpp)
Aspiration d≥4: `delta = 21+|prev|/256`, fail → delta += delta/4+5, fail-low pulls beta to (α+β)/2. Razoring: depth<2 & eval ≤ α−531 → qsearch. Futility: `217·(depth−improving)`, depth<6. Null: gate `eval≥beta && staticEval ≥ beta−32d+292−30·improving && (ss−1)statScore<23397`; `R=(854+68d)/258 + min((eval−beta)/192, 3)`; verification d≥13. ProbCut: d≥5, `raisedBeta=beta+189−45·improving`, ≤2+2·cutNode moves, qsearch-verify then search(d−4). IID: d≥7 & !ttMove → search(d−7). Singular: d≥6, ttMove, ttBound≥LOWER, ttDepth≥d−3; `singularBeta=ttValue−2·depth`, half-depth exclusion search; extend if fail; **multicut: if singularBeta≥beta return singularBeta**. Extensions: check (discovery or see_ge), shuffle, passed-pawn, castling. LMP: `(5+d²)·(1+improving)/2 − 1`. CounterMove-prune: lmrDepth<4+… & contHist[0],[1] < 0. Parent futility: lmrDepth<6 & sEval+235+172·lmrDepth ≤ α. SEE: quiets `−(32−min(lmrD,18))·lmrD²`, captures `−194·depth`. LMR: `Reductions[i]=(24.8+log(th)/2)·log(i)`; `r=(R[d]·R[mn]+511)/1024`; mods: ttPv−2, singularLMR−2, cutNode+2, ttCapture+1, (ss−1)mc>14 −1, escape-capture −2, `r −= statScore/16384` where statScore = mainHist+cont[0]+[1]+[3]−4926. Histories: butterfly(10692), capturePieceTo(10692), contHist at plies {1,2,4,6} (PieceTo, 29952), counterMoves[pc][sq]; killers[2]; `stat_bonus = 19d²+155d−132 (cap d>15 → −8?)`. improving = sEval ≥ (ss−2)sEval (fallback ss−4). QSearch: ttDepth ∈ {0,−1}, futilityBase = best+154, SEE≥0 filter.

---

## SF 16 → 17 → 18 (post-NNUE) — mined from disk (18 = reference)

### Eval remnants (evaluate.cpp)
Only `simple_eval = PawnValue·Δpawns + Δnpm` as NET ROUTER (smallnet if |simple|>962; SF16: 1050). Blend: `nnue=(125·psqt+131·positional)/128`; smallnet re-eval if |nnue|<277; `nnueComplexity=|psqt−positional|`; optimism += optimism·complexity/476; nnue −= nnue·complexity/18236; `material=534·pawns+npm`; `v=(nnue·(77871+material)+optimism·(7191+material))/77871`; rule50 damp v·rule50/199; TB clamp. ALL positional HCE deleted (wholesale at SF15.1; SF12-15 hybrid used classical mainly in material-imbalanced positions — from history, not on-disk). WDL/cp normalization lives in uci.cpp.

### Search additions vs SF11 [post-NNUE but mostly eval-agnostic]
- **Correction history** ★: corrects staticEval by running per-key error: pawn-key(×10347), minor-key(×8821), non-pawn W/B(×11665), continuation-corr ss−2/ss−4(×7841); `corrected = clamp(raw + Σ/131072)`; limit 1024. Feeds futility margin (|corr|/174665) and LMR (−|corr|/30370).
- IIR replaces IID: `!allNode & d≥6 & !ttMove & priorReduction≤3 → depth−−`.
- Hindsight: priorReduction≥3 & !opponentWorsening → depth++; opponentWorsening = sEval > −(ss−1)sEval.
- Singular now: `sBeta = ttValue − (53+75·(ttPv&!PV))·d/60`, double/triple ext margins, negative ext −2/−3, multicut retained.
- ProbCut: `beta+235−63·improving` + probCutDepth clamp; "small probcut" TT fast-path beta+418.
- Razor: `eval < α − 485 − 281·d²`. Futility: mult 76−23·!ttHit, −(2474·improving+331·oppWorsening)·mult/1024, d<14. Null: cutNode-only gate, `R = 7+d/3`, sEval ≥ beta−18d+350.
- LMR: base 2747/128·log(i); mods incl. cutoffCnt, ttMoveHistory; doDeeper (val>best+50) / doShallower (val<best+9) re-search bands.
- New tables: PawnHistory(8192), TTMoveHistory, LowPlyHistory; **killers/countermove REMOVED** (pure history ordering).
- LMP unchanged `(3+d²)/(2−improving)`. SEE: captures max(166d+capHist/29,0), quiets −25·lmrD².
- QSearch: stand-pat blend (best+beta)/2, futilityBase = sEval+351, moveCount>2 cut, SEE −80.

### TT / Movegen
TT identical 10-byte/3-cluster/32B shape as SF11 (index now mul_hi64). MovePicker same staging minus REFUTATION stage; skip_quiet_moves() added.
