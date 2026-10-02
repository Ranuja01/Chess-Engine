# -*- coding: utf-8 -*-
"""TRIANGULATION cases (owner, 10-02; the v1 method): representative positions where our SEARCH still disagrees with
SF18 a lot, but the gap is neither TACTICAL nor a known MATERIAL class — so the owner can read the FEN by eye, with our
eval broken down by term, and name the pattern we are missing.

Selection (std middlegame rows with a d10 score, `_depth_residual_pass.py`):
  · material equal piece-for-piece (no imbalance — those are the MCL classes)
  · SF18's best move is QUIET (not a capture, not a check, not a promotion)
  · our static ≈ our d10 search (|diff| ≤ 40 cp) — the position is quiet for US too
  · |win%(SF18 d14) − win%(our d10)| ≥ GAP pp
Prints the N largest, alternating the sign (we over-rate White / we over-rate Black), with the ev_breakdown terms.
Needs the engine in-process (ev_breakdown only). Run with V2_PRESET=shipped.

  pyrun diagnostics/_triangulate_cases.py [GAP=12] [N=8]
"""
import os, sys, csv, glob
import numpy as np
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import ChessAI

K = 0.00368208
wp = lambda cp: 100.0 / (1.0 + np.exp(-K * np.clip(cp, -1500, 1500)))
GAP, N = float(KV.get("GAP", 12)), int(KV.get("N", 8))
TERMS = ["pieces", "material", "king_safety", "mobility", "pawn_struct", "v2_passers", "v2_placement", "v2_winnab"]

ours_d = {}
for p in glob.glob(os.path.join(THIS, "ks_sets", "fitC_mg_ours_d10_s*of4.csv")):
    for r in csv.DictReader(open(p, newline="")):
        ours_d[r["fen"]] = float(r["ours_cp_white"])
std = {r["fen"] for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sample.csv"))) if r["src"] == "std"}
ai = ChessAI.ChessAI(None, None, chess.Board(), True)
cands = []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets/fitC_mg_sf18.csv"), newline="")):
    f = r["fen"]
    if not r.get("best_cp") or f not in std or f not in ours_d:
        continue
    b = chess.Board(f)
    if any(len(b.pieces(p, chess.WHITE)) != len(b.pieces(p, chess.BLACK))
           for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
        continue
    mv = chess.Move.from_uci(r["best_uci"])
    if b.is_capture(mv) or b.gives_check(mv) or mv.promotion:
        continue
    sf, d10 = float(r["best_cp"]), ours_d[f]
    bd = ai.ev_breakdown(b)
    stat = -float(bd["total"]) / 10.0
    if abs(stat - d10) > 40:
        continue
    gap = wp(sf) - wp(d10)
    if abs(gap) >= GAP:
        cands.append((abs(gap), gap, f, sf, d10, stat, r["moves"], {t: bd.get(t) for t in TERMS}))
cands.sort(key=lambda c: -c[0])
pos = [c for c in cands if c[1] > 0]
neg = [c for c in cands if c[1] < 0]
pick = [x for pair in zip(pos, neg) for x in pair][:N]
print("TRIANGULATION  candidates %d (SF better for White than us: %d · worse: %d) — showing %d" % (len(cands), len(pos), len(neg), len(pick)))
for i, (_, gap, f, sf, d10, stat, moves, terms) in enumerate(pick, 1):
    print("\n#%d  %s" % (i, f))
    print("    SF18 d14 %+5.0f cp (White) · ours: d10 %+5.0f, static %+5.0f  ⇒ SF sees White %s by %.1f pp"
          % (sf, d10, stat, "BETTER" if gap > 0 else "WORSE", abs(gap)))
    print("    SF top moves (White-POV cp): %s" % moves.replace(";", "  "))
    print("    our static terms (White-POV cp): " + "  ".join(
        "%s %+.0f" % (t, -v / 10.0) for t, v in terms.items() if v is not None and t != "material"))
