# TRIANGULATION round 1 (2026-10-02) — owner + Claude, the v1 method on the DEPTH residual

Tool: `diagnostics/_triangulate_cases.py GAP=12 N=8` (quiet, equal-material std mg rows; SF best move quiet; our static ≈ our d10). 40 candidates (27 SF prefers White more than us, 13 less).

```
TRIANGULATION  candidates 40 (SF better for White than us: 27 · worse: 13) — showing 8

#1  1r4k1/2q5/2Ppbpp1/1P2p2r/3pPn1p/Q2B1N1P/5PP1/2R2R1K w - - 14 36
    SF18 d14  +222 cp (White) · ours: d10    +6, static   -14  ⇒ SF sees White BETTER by 18.8 pp
    SF top moves (White-POV cp): f3d2:222  f1d1:221  f1e1:214  c1a1:211
    our static terms (White-POV cp): pieces -40  king_safety +0  mobility +30  pawn_struct +4  v2_passers +27  v2_placement -36  v2_winnab +0

#2  2r3kr/4qpp1/pp2p2p/2n5/8/P2QP3/P1N2PPP/1K1R2R1 w - - 1 23
    SF18 d14  -252 cp (White) · ours: d10   -41, static   -18  ⇒ SF sees White WORSE by 17.9 pp
    SF top moves (White-POV cp): d3c4:-252  d3d6:-258  d3c3:-268  d3d4:-285
    our static terms (White-POV cp): pieces -0  king_safety +0  mobility +36  pawn_struct -21  v2_passers +0  v2_placement -32  v2_winnab +0

#3  r4r2/2pq1pk1/1p1p2pp/n2N4/p1PPP2P/Pn1Q1NP1/1P4K1/4RR2 b - - 4 28
    SF18 d14  +159 cp (White) · ours: d10   -41, static    -8  ⇒ SF sees White BETTER by 18.0 pp
    SF top moves (White-POV cp): d7c6:159  a8d8:210  b6b5:217  a8e8:242
    our static terms (White-POV cp): pieces +46  king_safety +0  mobility +34  pawn_struct -2  v2_passers +0  v2_placement -86  v2_winnab +0

#4  1r2k2r/4qp2/1pn3p1/p1p1p3/P1PpPbQ1/1RP5/1P2BP1P/3R1NK1 w k - 3 28
    SF18 d14  -232 cp (White) · ours: d10   -55, static   -85  ⇒ SF sees White WORSE by 15.1 pp
    SF top moves (White-POV cp): g4g2:-232  h2h3:-238  g1h1:-243  d1d3:-246
    our static terms (White-POV cp): pieces +54  king_safety -58  mobility -30  pawn_struct -14  v2_passers -4  v2_placement -32  v2_winnab +0

#5  r2q1rk1/1p3pp1/3p1b2/pPp1pP2/PnR1P1Pp/1Q3N1P/1P1BP3/5RK1 w - - 5 31
    SF18 d14  +296 cp (White) · ours: d10   +84, static   +51  ⇒ SF sees White BETTER by 17.2 pp
    SF top moves (White-POV cp): g4g5:296  f1f2:262  g1h1:251  f1d1:242
    our static terms (White-POV cp): pieces +43  king_safety +0  mobility +26  pawn_struct +0  v2_passers +0  v2_placement -17  v2_winnab +0

#6  2r2b2/1p3kp1/p2p1n2/2p2Pp1/PPP4r/B1NpP3/3P2KP/1R3R2 b - - 0 26
    SF18 d14  -122 cp (White) · ours: d10   +33, static   +21  ⇒ SF sees White WORSE by 14.0 pp
    SF top moves (White-POV cp): c8b8:-122  b7b6:-108  c8c7:-106  f7g8:-63
    our static terms (White-POV cp): pieces -6  king_safety +0  mobility +22  pawn_struct +21  v2_passers +0  v2_placement -16  v2_winnab +0

#7  2kr3r/4n1b1/pq1ppp2/4p2p/1pPP4/1P2PN2/1B3PPP/R2Q1RK1 b - - 0 18
    SF18 d14  +182 cp (White) · ours: d10    -3, static   +22  ⇒ SF sees White BETTER by 16.5 pp
    SF top moves (White-POV cp): e5e4:182  c8b7:215  a6a5:220  h5h4:221
    our static terms (White-POV cp): pieces +2  king_safety +0  mobility +6  pawn_struct +12  v2_passers +2  v2_placement +0  v2_winnab +0

#8  r3r1k1/1p2b1pp/2p1b3/p3Pp2/1nN5/BP2P2P/5PP1/2R1KB1R b K - 1 17
    SF18 d14  -283 cp (White) · ours: d10  -109, static  -114  ⇒ SF sees White WORSE by 14.0 pp
    SF top moves (White-POV cp): b7b5:-283  a5a4:-262  f5f4:-205  b4a2:-199
    our static terms (White-POV cp): pieces +3  king_safety +0  mobility -66  pawn_struct -10  v2_passers +15  v2_placement -55  v2_winnab +0
```

## First reading (Claude, pending the owner's)
1. KS reads 0 where kings are structurally exposed (#2 b1 king without a b-pawn + half-open c-file; #5 g4-g5 lever vs a castled king; #7 long-castled king behind advanced a6/b4 pawns) — danger from STRUCTURE, not attackers present.
2. Levers SF actually plays (#5 g5, #8 ...b5) — the POT-shaped signal, concretely.
3. Piece quality: stranded/offside pieces scored as active (#3 knights a5/b3; #4).
4. A protected far-advanced passer under-valued (#1 c6) — the opposite of the Carlsen 47.c7 over-valuation ⇒ passer value is context-dependent.

## Side by side with SF11's static eval (`diagnostics/_triangulate_sf11.py`; owner: "same subsystems — does SF11 get it, and where?")
| case | SF18 d14 | SF11 static | ours static | ours d10 |
|---|---|---|---|---|
| #1 | +222 | +74 | −14 | +6 |
| #2 | −252 | −108 | −18 | −41 |
| #3 | +159 | +58 | −8 | −41 |
| #4 | −232 | −188 | −85 | −55 |
| #5 | +296 | +156 | +51 | +84 |
| #6 | −122 | −111 | +21 | +33 |
| #7 | +182 | +93 | +22 | −3 |
| #8 | −283 | −138 | −114 | −109 |
SF11 static is closer to SF18 than our static in 7/8 (≈ equal in #8, which neither gets) ⇒ most of the gap IS statically
expressible. Term differences (SF11 blended with its own phase, White cp):
- **THREATS** — SF11 #2 −56 · #4 −46 · #5 +46 · #6 −46 · #7 −24 · #1 −18; **v2 ships no threats term** (slice-3 threats was
  parked "move-null", judged before the depth target existed) ⇒ strongest case to REOPEN, on the depth residual.
- **STRUCTURAL KING SAFETY** — SF11 #7 +93 · #2 −62 · #6 −50 · #1 +32; ours 0 in all four ⇒ KS-B shelter/storm (built at 0)
  in the FULL KS TUNE.
- **PASSED** #1 SF11 +86 vs ours +27; **PAWNS** #4 SF11 −87 vs ours −14 ⇒ passer / pawn-structure re-tunes.
- SPACE #3 +28 · #5 +17 (not shipped). #8: beyond static for both (SF18 finds …b5 by search) — the POT-shaped case.

## OWNER DECISION (2026-10-02)
Tune the known-powerful items first, keep POT's middlegame PARKED, then re-measure: material classes (gating overnight)
→ THREATS reopened (confirm the pattern over all 40 cases, Texel-fit on the depth target, gate) → full KS tune incl.
KS-B shelter/storm → passers + pawn structure → re-measure the remaining gap to SF18 (depth target + fresh
triangulation) → POT only for a non-tactical gap no tuned term explains (so it never steps on a term that was merely
under-tuned). Then the v2 search retune (owner games: relaxed selectivity matched SF 5/12 vs 2/12) and the giant-corpus
final retune.

## ☠️ CORRECTION — the aggregate over ALL candidates (2026-10-02, `_triangulate_sf11.py MODE=aggregate GAP=8`)
111 quiet equal-material cases (gap ≥ 8pp): SF11 static closer to SF18 than ours in 80/111 (72%). Per SF11 term vs ours,
"helps" = the difference (≥ 30 cp) points the same way as SF18's disagreement with our d10 search:
| term | helps | hurts | mean push toward SF18 |
|---|---|---|---|
| **King safety** | **48** | 17 | **+24.1 cp** |
| Pawns | 15 | 6 | +5.6 |
| Passed | 6 | 3 | +3.1 |
| pieces (N+B+R+Q) | 10 | 10 | +3.7 |
| **Threats** | 20 | 18 | **+0.4** |
| Mobility / Space | 0 / 0 | 1 / 0 | −2.2 / +1.0 |
⇒ The 8 hand-picked cases OVERSTATED threats (6/8 there; a wash over 111). **Structural KING SAFETY is the consistent,
largest statically-expressible miss** (≈3:1, +24 cp mean) ⇒ the FULL KS TUNE with KS-B shelter/storm moves to the front
of the post-material order; threats drops to "fit it only if the KS tune leaves a threats-shaped residual".
Lesson (again): a hand-picked set reads a pattern the population does not have — aggregate before reordering lanes.
