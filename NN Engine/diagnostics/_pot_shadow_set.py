# -*- coding: utf-8 -*-
"""The quiet middlegame set for POT test D (_pot_shadow.py): wdlcal's quiet own-play rows (cal_d10_in.csv) with non-pawn
material 10-22 (N, B 1, R 2, Q 4; 24 = the start) and move number >= 10, the first LIMIT (1,500) → /mnt/e/chess_data/pot_d/mg_quiet.csv."""
import csv, sys, chess
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SRC = KV.get("SRC", "/mnt/e/chess_data/wdlcal/cal_d10_in.csv")
OUT = KV.get("OUT", "/mnt/e/chess_data/pot_d/mg_quiet.csv")
LIMIT = int(KV.get("LIMIT", 1500))
NPM = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
rows = []
for r in csv.DictReader(open(SRC, newline="")):
    b = chess.Board(r["fen"])
    npm = sum(NPM.get(p.piece_type, 0) for p in b.piece_map().values())
    if 10 <= npm <= 22 and b.fullmove_number >= 10:
        rows.append(r["fen"])
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["fen"])
    for fen in rows[:LIMIT]:
        w.writerow([fen])
print("quiet middlegame set: %d of %d eligible rows → %s" % (min(LIMIT, len(rows)), len(rows), OUT))
