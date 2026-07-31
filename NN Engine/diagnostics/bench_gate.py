#!/usr/bin/env python3
"""Collapse-bench eval gate: our STATIC eval vs SF on the over-read bench (diagnostics/overread_bench.csv).
Counts how many of the 2,172 decision-flip positions our static eval STILL gets sign-wrong -> the single
number an eval fix (KS rework, conditioned term, ...) must drive DOWN without wrecking mean|gap|.

Single process, static (ev_breakdown), NO games. Set eval env knobs (KING_SAFETY_MAG / ENABLE_KS_REPLACE_LT
/ KS_* / SCALE_*) BEFORE the first ChessAI(...) -- initialize_engine parses env ONCE per process
(static bool toggles_loaded latch), so a ChessAI built earlier freezes Config at defaults.
Usage: [env knobs] python bench_gate.py <out_csv> [bench_csv]
bench_csv (relative to NN Engine/) defaults to diagnostics/overread_bench.csv; pass the
_train/_holdout split files for the anti-overfit protocol.
"""
import os, sys, csv, chess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # NN Engine/
os.chdir(BASE)
sys.path.insert(0, BASE)
from ChessAI import ChessAI

out_csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_gate.csv"
bench_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join("diagnostics", "overread_bench.csv")
rows = list(csv.DictReader(open(os.path.join(BASE, bench_csv))))

seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)   # ev_breakdown reads only the board arg


def our_white_pawns(fen):
    return -ai.ev_breakdown(chess.Board(fen))["total"] / 1000.0


def is_flip(ow, sf):
    return (ow >= 0.5 and sf <= -0.75) or (ow <= -0.5 and sf >= 0.75)


n = flips = 0
gap_sum = 0.0
recs = []
for r in rows:
    fen, sf = r["fen"], float(r["sf_pawns"])
    try:
        ow = our_white_pawns(fen)
    except Exception:
        continue
    n += 1
    f = is_flip(ow, sf)
    flips += 1 if f else 0
    gap_sum += abs(ow - sf)
    recs.append((fen, ow, sf, 1 if f else 0))

with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["fen", "our_static", "sf", "flip"])
    for fen, ow, sf, f in recs:
        w.writerow([fen, "%.3f" % ow, "%.3f" % sf, f])

print("KS: KING_SAFETY_MAG=%s ENABLE_KS_REPLACE_LT=%s KS_ZONE2=%s KS_DYN=%s" % (
    os.environ.get("KING_SAFETY_MAG", "0"), os.environ.get("ENABLE_KS_REPLACE_LT", "0"),
    os.environ.get("KS_ZONE2", "0"), os.environ.get("KS_DYN", "0")))
print("bench n=%d  STATIC sign-flips vs SF = %d (%.1f%%)  mean|gap| = %.2f pawns  -> %s" % (
    n, flips, 100.0 * flips / max(1, n), gap_sum / max(1, n), out_csv))
