# -*- coding: utf-8 -*-
"""BALANCED endgame start positions for game tests (2026-10-16).

  python diagnostics/_make_balanced_eg_starts.py [IN=/mnt/e/chess_data/pe/potfit_dump.csv] [OUT=selfplay/eg_balanced_fens.txt]
        [CP=80] [N_PAWN=300] [N_MINOR=300] [PER_GAME=2]

WHY: the random pawn-dominated starts (kp_fens / kp_dense) are mostly DECIDED before move 1 — in #74's targeted match 513 of
600 opening pairs split 1-1 by colour (170 resigned on the first ply) and only 87 could discriminate. Here: pure-pawn and
one-minor endings from EGX (own games) that SF18 d14 rates CLOSE (|cp| ≤ CP), at most PER_GAME per source game, so endgame
knowledge — not the start — decides the result. Lines are tournament.py FEN openings with a class tag.
"""
import os, sys, csv, random
KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
IN = KV.get("IN", "/mnt/e/chess_data/pe/potfit_dump.csv")
OUT = KV.get("OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "selfplay", "eg_balanced_fens.txt"))
CP, PER = float(KV.get("CP", "80")), int(KV.get("PER_GAME", "2"))
NEED = {"pawn": int(KV.get("N_PAWN", "300")), "minor": int(KV.get("N_MINOR", "300"))}
rows = [r for r in csv.DictReader(open(IN, newline="")) if abs(float(r["sf_cp"])) <= CP]
random.Random(1016).shuffle(rows)
got, per_game, seen = {"pawn": [], "minor": []}, {}, set()
for r in rows:
    c, g = r["cls"], r["game"]
    if c not in got or len(got[c]) >= NEED[c] or per_game.get(g, 0) >= PER or r["fen"] in seen:
        continue
    got[c].append(r["fen"]); per_game[g] = per_game.get(g, 0) + 1; seen.add(r["fen"])
with open(OUT, "w") as fh:
    fh.write("# BALANCED endgame starts (|SF18 d14| <= %.0f cp), pure pawn + one-minor, <= %d per source game — %s\n" % (CP, PER, IN))
    for c in ("pawn", "minor"):
        for f in got[c]:
            fh.write("%s ; %s\n" % (f, c))
print("wrote %s: pawn %d, minor %d" % (OUT, len(got["pawn"]), len(got["minor"])))
