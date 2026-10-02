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
