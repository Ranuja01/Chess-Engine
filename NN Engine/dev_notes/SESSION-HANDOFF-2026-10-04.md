# SESSION HANDOFF — 2026-10-04 (session 09-30 → 10-04)

Four ships, two instrument corrections, three lanes closed. The anchor for the next chat; details live in the docs it
points to (C3 doc = `dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md`).

## 1. Shipped (V2_PRESET=shipped) — fingerprint **WAC d10 255 / 47,218,480 / 4.007** (v1 unchanged 250 / 35,310,778)
| ship | form | evidence |
|---|---|---|
| POT winnability (`POT_V2_WIN*`, 10-01) | multiplicative eg SCALE FACTOR (the 4/4 reference form; the additive `sign(T)·C` lost −24 twice) | vs SF18 +14.9 [+3.4, +26.7] (2,000) · calibrated @800 +12.9 · self-play ≈ 0 (50k and 250k) · C3 §16-16e |
| Kaufman fitted cells (`KAUF_V2_FORM=3`, 10-03) | SF11 census cells Texel-fitted on SF18 labels (compiled `ship_tables_v2.h`) | calibrated +13.1 (2,000) · self-play +11.8 · C3 §18c/§18m/§18p |
| Joint KS + KS-B (10-03) | KS attack knobs + 56 shelter/storm cells fitted JOINTLY on the DEPTH target (compiled) | calibrated +6.0 (2,000) · self-play +17.7 · KS-B alone −13/+16 · C3 §18k-§18q |
| **10-03 ship confirmed** (new vs 10-01) | — | calibrated +19.6 · self-play +13.6 ⇒ **≈ +15.7 ± 7.3** · C3 §18s |
v2 vs v1 at EQUAL nodes (10-01, before the last ships): **+65.4 ± 17.9** (2,000 games).

## 2. Instrument corrections (INSTRUMENT-MAP "2026-10-01/04")
- **Depth target:** search for missing eval knowledge on SF18 d14 − our d10 SEARCH (`_depth_residual_pass.py`), not the
  static residual (mostly search-fixable).
- **Calibrated judge:** SF18 @400 drifted to ~75% and FLIPPED material terms; the anchor is now **SF18 @800 nodes**
  (54% at recalibration, ~56% after the 10-03 ship — re-check it after each ship). Gate on calibrated + self-play.
- Memory: `the-sf18-gauntlet-anchor-drifted-too-weak`, `unattended-jobs-must-have-bounded-memory`.

## 3. Lanes closed / parked this session
- **POT middlegame — PARKED** (owner decision 10-02): T1/T3/T4/T5 + lever reach null on the depth target (8,128 rows);
  the castling-race feeder fades at depth. Revisit only after the known terms are tuned AND the gap is re-measured.
  `dev_notes/POT-TYPE-DEFINITIONS-2026-09-30.md`, `POT-TRANSFORMATION-KNOWLEDGE-2026-09-30.md`.
- **Narrow material classes (MCL_V2)** — neutral on the calibrated judge (−2.4); built at 0.
- **Passers — CLOSED for now:** PX passer system (51 cells, `PX_V2`, built at 0) fits only −0.66% on the depth target
  and adds nothing at depth on the v1-era passer corpora; v2 already plugs v1's passer weaknesses at depth (passer
  design §10a). The passer rank re-price with free mg legs HURTS (−22 calibrated). Pawn structure re-price +5.7 (n.s.).
  `dev_notes/PASSER-SYSTEM-DESIGN-2026-10-03.md`, C3 §19-19c.
- Threats: the triangulation AGGREGATE says a wash (20 helps / 18 hurts) — not reopened.

## 4. Running at handoff
- Queue #16's last step: self-play JOINT pawn arm (`pawnjoint_selfplay`, 2,000 @50k, seed 72). Informational only (the
  calibrated reads already say no ship). Check with `overnight_runner.sh ps`; result in `selfplay/games/pawnjoint_selfplay/`.

## 5. Owner decisions in force (this session)
- Universal concepts kept; single-lineage ideas added at 0 and priced by the fit; "find what works for OUR engine".
- Passers before pawn structure, never convoluted (both done/closed for now).
- Piece values: `values[]` is SHARED with v1 and search — never tune it; re-pricing goes through `v2_piece_value`
  jointly with the Kaufman cells, in the final retune.
- Final retune: one GIANT diverse corpus (incl. variant/odds/imbalance positions), everything fitted jointly, each part
  still gated (memory `final-retune-needs-a-giant-diverse-corpus`).
- POT stays parked until the tuned eval's remaining gap to SF18 is re-measured.

## 6. Next — the agreed order
0. **Where does SF11's remaining STATIC edge come from — weighting or missing knowledge?** (owner, 10-04) On the passer
   corpora SF11 static is still ≈ 2pp better (9.94 vs 12.01; under-fire 14.8 vs 18.6) although our d10 search beats it
   easily. (a) PREVIEW of the final retune: fit ALL linear terms JOINTLY on the depth target (PST, mobility, pawns,
   passers, PX, KS-B, Kaufman) and re-run `_passer_corpus_check.py MODE=static` — most of the gap closing ⇒ WEIGHTING
   (the final retune's job); little closing ⇒ MISSING KNOWLEDGE ⇒ (b) triangulate the under-fire class against SF11's
   term table (`_triangulate_sf11.py` style) to name it. Diagnostic only — static closeness is not the goal, Elo is.
1. **Re-measure the remaining gap** to SF18 on the depth target + fresh TRIANGULATION (`_triangulate_cases.py`,
   `_triangulate_sf11.py MODE=aggregate`) on the CURRENT ship — the 10-02 aggregate pointed at structural KS (now shipped).
2. **The v2 SEARCH retune** (owner's games: a less selective d12 matched SF18 5/12 vs 2/12 for normal d12 AND d18 —
   `dev_notes/OWNER-GAMES-ANALYSIS-2026-10-01.md`; every search setting was tuned on v1). Includes the RFP/margin
   re-sweep (a truer eval buys pruning headroom) and quality, not just depth.
3. Optional mobility / placement depth-target re-fits if step 1 shows room.
4. POT only if a non-tactical gap remains.
5. The giant-corpus final retune → the NPS rewrite decision → the owner's NN.

## 7. Engineering notes
- Shipped tables are COMPILED (`ship_tables_v2.h`); `KAUF_V2_FILE` / `KSB_V2_FILE` override them for experiments.
- `v2_features` is now 235/side (184 + 51 PX at 184-234); ChessAI.pyx buffers match.
- Never rebuild under a running queue (they run from the working tree); bound memory in long single-process loops.
- 36+ commits unpushed at handoff (push only when the owner says).
