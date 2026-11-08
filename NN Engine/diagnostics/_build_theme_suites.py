"""Themed critical-position suites from the Lichess puzzle database (CC0, database.lichess.org/lichess_db_puzzle.csv.zst), in the
UCI `bench` format (FEN | best move), for the lazy eval critical-situation test (dev_notes/active/LAZY-EVAL-RESEARCH-2026-10-30.md
§4 item 5) and any later search or eval change that could cost tactics.

Usage: python diagnostics/_build_theme_suites.py SRC.csv[.zst] OUTDIR [N=200] [SEED=1107] [GROUP=critical]
  GROUP: `critical` (the 18 sets below) or `passer` (passed-pawn sets, built 11-08 to test the lazy eval passer guard).

A Lichess puzzle's FEN is the position BEFORE the opponent's move; the first UCI move is that move and the second is the solution's
first move, so each line here is the position after the first move with the second as the answer. Filters: rating 1800-2800,
rating deviation < 90, popularity > 80, more than 500 plays. A set is a seeded sample of N puzzles carrying the theme(s);
`sacrifice_long` = sacrifice with long or veryLong (a sequence forced into a sacrifice). Mates count as solved only on the listed move.
"""
import csv
import io
import os
import random
import sys

import chess

SETS = {
    "sacrifice": {"sacrifice"},
    "sacrifice_long": {"sacrifice", "long|veryLong"},
    "mateIn2": {"mateIn2"}, "mateIn3": {"mateIn3"}, "mateIn4": {"mateIn4"}, "mateIn5": {"mateIn5"},
    "attraction": {"attraction"}, "deflection": {"deflection"}, "clearance": {"clearance"},
    "interference": {"interference"}, "quietMove": {"quietMove"}, "defensiveMove": {"defensiveMove"},
    "exposedKing": {"exposedKing"}, "kingsideAttack": {"kingsideAttack"}, "xRayAttack": {"xRayAttack"},
    "zugzwang": {"zugzwang"}, "advancedPawn": {"advancedPawn"}, "endgame_crushing": {"endgame", "crushing"},
}
# Positions where a passed pawn decides: the lazy eval guard for an advanced passer is meant to protect these.
PASSER_SETS = {
    "promotion": {"promotion"},
    "advancedPawn_mg": {"advancedPawn", "middlegame"},
    "pawnEndgame": {"pawnEndgame"},
    "rookEndgame_passer": {"rookEndgame", "advancedPawn|promotion"},
    "minorQueenEndgame_passer": {"bishopEndgame|knightEndgame|queenEndgame", "advancedPawn|promotion"},
}
GROUPS = {"critical": SETS, "passer": PASSER_SETS}


def matches(themes, need):
    return all(any(alt in themes for alt in t.split("|")) for t in need)


def rows(src):
    if src.endswith(".zst"):
        import zstandard
        fh = zstandard.ZstdDecompressor().stream_reader(open(src, "rb"))
        yield from csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
    else:
        yield from csv.DictReader(open(src, encoding="utf-8"))


def main(src, outdir, n=200, seed=1107, group="critical"):
    sets = GROUPS[group]
    os.makedirs(outdir, exist_ok=True)
    pools = {k: [] for k in sets}
    for r in rows(src):
        try:
            if not (1800 <= int(r["Rating"]) <= 2800 and int(r["RatingDeviation"]) < 90
                    and int(r["Popularity"]) > 80 and int(r["NbPlays"]) > 500):
                continue
        except (KeyError, ValueError):
            continue
        themes = set(r["Themes"].split())
        for k, need in sets.items():
            if matches(themes, need):
                pools[k].append((r["FEN"], r["Moves"]))
    rng = random.Random(seed)
    for k, pool in pools.items():
        rng.shuffle(pool)
        with open(os.path.join(outdir, f"{k}.txt"), "w") as o:
            o.write(f"# {k}: {min(n, len(pool))} of {len(pool)} Lichess puzzles (rating 1800-2800), seed {seed}\n")
            for fen, moves in pool[:n]:
                mv = moves.split()
                b = chess.Board(fen)
                b.push_uci(mv[0])
                o.write(f"{b.fen()} | {mv[1]}\n")
        print(f"{k:18s} {min(n, len(pool)):4d} of {len(pool):7d}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 200, int(a[3]) if len(a) > 3 else 1107, a[4] if len(a) > 4 else "critical")
