# -*- coding: utf-8 -*-
"""Does ENABLE_PASSER_V3 actually close our eval gap vs SF18, at corpus scale?

A 2-position static probe showed V3 moving passed_pawn_support from ~0 to SF-ballpark values, but two
hand-picked collapse FENs cannot establish that it closes the gap. This recomputes our eval over every
banked position carrying an SF18 label and scores win%-space error, so V3-on and V3-off are compared on
the same 1600+ positions.

Knobs latch at extension init => ONE PROCESS PER SETTING. Run it twice:
    pyrun diagnostics/_v3_gap.py
    pyrun diagnostics/_v3_gap.py ENABLE_PASSER_V3=1

Also breaks the error out by whether the position HAS a passed pawn, since that is where V3 can act at all
-- a corpus-wide null with a passer-subset win would still be informative.
⚠️ Static-fit error is not Elo (corpus-fit-flattens-eval). This can kill an idea, not promote one.
"""
import os, sys, csv, math
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
os.chdir(os.path.dirname(THIS))

import chess
from ChessAI import ChessAI

WIN_K = 0.00368208
def winpct(cp): return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)

def has_passer(b):
    """True if either side has a passed pawn (no enemy pawn ahead on its file or adjacent files)."""
    for color in (chess.WHITE, chess.BLACK):
        them = not color
        for sq in b.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            blocked = False
            for df in (-1, 0, 1):
                nf = f + df
                if not 0 <= nf <= 7:
                    continue
                for esq in b.pieces(chess.PAWN, them):
                    if chess.square_file(esq) != nf:
                        continue
                    er = chess.square_rank(esq)
                    if (er > r) if color == chess.WHITE else (er < r):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return True
    return False

rows = []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "position_bank.csv"))):
    if not r.get("sf18"):
        continue
    try:
        sf18 = float(r["sf18"])
    except Exception:
        continue
    if abs(sf18) > 20.0:
        continue
    rows.append((r["fen"], sf18))

seed = chess.Board(rows[0][0])
ai = ChessAI(None, None, seed, seed.turn)

all_e, pas, nopas = [], [], []
for fen, sf18 in rows:
    b = chess.Board(fen)
    bd = ai.ev_breakdown(b)
    if bd.get("checkmate"):
        continue
    ours_white = -bd["total"] / 1000.0            # raw is Black-positive -> White-POV pawns
    e = winpct(ours_white * 100.0) - winpct(sf18 * 100.0)
    all_e.append(e)
    (pas if has_passer(b) else nopas).append(e)

def report(name, errs):
    if not errs:
        print(f"  {name:<22} (none)")
        return
    mse = sum(x * x for x in errs) / len(errs)
    mae = sum(abs(x) for x in errs) / len(errs)
    med = sorted(abs(x) for x in errs)[len(errs) // 2]
    print(f"  {name:<22} n={len(errs):>5}  meanAbsErr={mae:>7.3f}  MSE={mse:>8.1f}  median={med:>7.3f}")

print(f"ENABLE_PASSER_V3={os.environ.get('ENABLE_PASSER_V3','0')}  "
      f"ENABLE_PASSER_V2={os.environ.get('ENABLE_PASSER_V2','0')}")
print("win%-space error vs SF18 (WHITE-POV, |sf18|<=20):")
report("ALL", all_e)
report("has passed pawn", pas)
report("no passed pawn", nopas)
