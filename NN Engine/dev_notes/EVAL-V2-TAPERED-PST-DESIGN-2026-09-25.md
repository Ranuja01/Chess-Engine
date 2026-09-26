# Tapered PST for v2 — cross-engine design (for discussion, nothing built)

@author: Ranuja Pinnaduwage (maintained with Claude)

Sources: a read-only contrast of SF1.1, SF11, SF15.1, Ethereal and Weiss (2026-09-25). I verified our side in code:
- `eval_v2.cpp` `rung0_material_and_placement`;
- `cpp_bitboard.cpp` `whitePlacementLayerBase` and `rebuild_scaled_placement`.

This file is a pointer from `EVAL-V2-INVENTORY-2026-09-25.md` §5 item 4.

## 1. WHAT v2 HAS TODAY (verified)
- One hand-written table per piece, `[type][file][rank]`, **no mg/eg split**, **never fitted**. It is v1's table.
- v2 reads all six tables **phase-flat** (the census loop has no phase gate). That includes:
  - **the rook table**, which v1 never reads ("Rook PST is dead code", `rebuild_scaled_placement`);
  - **the king table**, which is labelled "Kings - Endgame" (a centralising table, 0-35 mp) and is paid in the opening
    and middlegame too.
- **There is no middlegame king table**: nothing prices the castled corner. Shelter/storm is also absent.
- Magnitudes (mp):

  | piece | range | what it rewards |
  |---|---|---|
  | pawn | 5-60 | centre occupation, not advancement |
  | knight | 0-40 | — |
  | bishop | 10-35 | — |
  | rook | 0-25 | d4/d5 centralisation |
  | **queen** | 15-65 | **middlegame centralisation** |
  | king | 0-35 | centralisation |

- The tables' means are positive (knight ~20, queen ~38), so they re-price material at census time. That is harmless at
  equal counts, but it leaks into imbalances.

## 2. WHAT THE REFERENCES DO
Conversion to our units by POSITIONAL scale: `k = 30 mp / engine's knight a3→e5 mg delta`. ⚠️ Per memory
`convert-reference-constants-by-positional-scale-not-by-the-pawn`, k is the LOW end of a ladder whose high end is the
pawn conversion. For SF11 the two ends are ~29× apart.

| engine | storage | mirror-safe for our gate? | tuning |
|---|---|---|---|
| SF1.1 | full 64, mg + eg, file-symmetric | ✅ | hand (Glaurung) |
| SF11 / SF15.1 | half-board 32 cells mirrored; separate pawn table | ✅ pieces · ☠️ pawn table asymmetric | fishtest SPSA |
| Ethereal | 64 `S(mg,eg)`, absolute file | ☠️ | gradient tuner |
| Weiss | 64 `S(mg,eg)`, absolute file | ☠️ | Texel-style tuner |

Per-piece spread (max−min), mg / eg, in our mp:

| piece | SF1.1 | SF11 | Ethereal* | Weiss* | ours (flat) |
|---|---|---|---|---|---|
| pawn | 30 / 0 | 17 / 13 | 207 / 188 | 91 / 64 | 55 |
| knight | 69 / 37 | 69 / 37 | 328 / 171 | 161 / 100 | 40 |
| bishop | 19 / 18 | 25 / 20 | 235 / 137 | — | 25 |
| rook | 6 / 0 | 13 / 9 | 180 / 142 | — | 25 |
| queen | **0** / 24 | **5** / 27 | 152 / 278 | — | **50** |
| king | 90 / 72 | 88 / 53 | 287 / 313 | 117 / 131 | 35 (eg shape only) |

\* The Ethereal and Weiss conversions are unreliable: their mg knight centralisation is tiny, because their mobility
carries it. That makes k large.

## 3. WHERE ALL REFERENCES AGREE, AND WHERE WE DIFFER
- **King (5/5): mg = castled corner on rank 1, eg = centre.** The mg spread is ≈3× the knight's.
  ⇒ **We have only the eg half, and pay it in the middlegame.**
- **Queen mg ≈ flat** (SF1.1 0, SF11 5 mp). The eg value comes from centralisation.
  ⇒ **Ours is the reverse: 50 mp of middlegame queen centralisation.**
- **Knight rim penalty dominant** everywhere; ours agrees in shape.
- **Rook:** no reference centralises rooks on d4/d5. The shapes they use are rank 7 and central files. ⇒ **Ours is
  v2-only and has never been measured.**
- **Pawn eg advancement: no consensus.** The passer terms own it; Ethereal is even negative on rank 7.

## 4. OVERLAP (one-owner rule)
- **Centralisation of N/B is already owned by mobility.** v2 at `MOB_V2_MAG=600` gives ~4.5× the PST's 30 mp for the
  same knight move, so v2 is already shaped like Ethereal (mobility ≫ PST).
  ⇒ **Do not enlarge the mg N/B tables.** That half of any port is redundant.
- **Outposts** overlap the knight's rank 4-6 cells. Keep them small.
- **Passers** own pawn advancement. Keep the pawn eg table small.
- **The king's mg corner table is a static proxy for SHELTER.** It sits in the same ownership question as KS-B
  shelter/storm and OvD (memory `ks-b-shelter-was-deferred-not-rejected`). ⚠️ Decide ownership before building it.
- The (mg,eg) pair is **not** a prerequisite. An eg table can blend at its own site, as every v2 term does.

## 5. CANDIDATE ARMS
Constraints common to all arms:
- NEW tables (`v2PstMg`/`v2PstEg`), half-board 32 cells mirrored at init, so the file-mirror gate passes by construction.
- Consumed only by v2's rung 0.
- **Never write `whitePlacementLayer`**: v1 and move ordering share it.
- Percent knobs, default 0 = byte-identical.
- Register the knobs before any rebuild call (init-order hazard, memory `eleventh-symmetry-defect`).
- Mirror-safe sources are SF1.1 and SF11/15 pieces only.

| arm | change | fire / size | notes |
|---|---|---|---|
| **(a) tapered KING** | mg = SF11-shaped corner table (b1/g1 high, centre ≈0); eg = today's table | ~100% of positions, up to ~90 mp mg | 30× `PST_V2_KING_EG_ONLY`'s 3 mp, so resolvable. Ladder the mg magnitude {30, 90, 300}. **Needs the shelter-ownership call** |
| **(b) de-flattened pieces** | queen: mg → flat, keep eg centralisation · rook: mg → 0 (or the SF rank-7 shape) · N/B: eg = ours + k·(SF11 eg − SF11 mg) | ~100%, ~10-30 mp | each cell is below the floor alone ⇒ bundle or retune |
| **(c) Texel-fitted tables** | fit the 6×32×2 cells on our own game results | — | needs file-loadable tables + the Texel pipeline (`_texel_extract.py`, stage 1 running 09-25) |

**Suggested sequence:**
1. **(b-lite) as a subtractive screen now**, because it needs almost no build. Arms:
   - rook table off in v2 (one knob);
   - queen mg flattened.

   These are v2-only departures from every reference, i.e. candidates for "wrong-signed" terms like the king was.
2. **(a)**, after the shelter-ownership decision.
3. **(c)**, if the Texel pipeline validates on a smaller vector first.

## 6. RECORD
- Tapered PST: **never tried.**
- `PST_V2_KING_EG_ONLY`: 3 mp, a magnitude null, not a concept test.
- v1 `SCALE_PLACE_*` levers: "rejected" on pre-08-14 STS (the contamination era, ±150 floor) ⇒ unreadable, and a
  different form (it scaled a flat table).
- Texel with PST cells: never tried.

## 7. UNRESOLVED
- SLICE2's "SF11 knight PST spread 84" and "Ethereal 9-13×" ratios cannot be reproduced; their anchors were not
  recorded.
- The orientation of SF1.1's table (row 0 = rank 1) is inferred, not confirmed.
- Weiss's B/R/Q tables were not fetched.
- The rook table read by v2 has never been ablated. There is no `SCALE_PLACE_ROOK` knob.
