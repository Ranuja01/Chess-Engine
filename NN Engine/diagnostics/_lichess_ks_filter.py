# -*- coding: utf-8 -*-
"""STREAM the ~1GB Lichess puzzle CSV (never load it whole) and extract a small, KS-themed, phase-stratified subset
of CRITICAL positions we lack — especially ENDGAME king attacks. For each matched puzzle, apply the opponent's
setup move (Moves[0]) to get the position the solver faces, and emit that FEN + metadata. SF18 multi-PV re-labeling
is a SEPARATE step (this only mines positions). Caps per phase stratum so the output stays small and balanced.

  pyrun diagnostics/_lichess_ks_filter.py [SRC=<abs csv>] [OUT=ks_sets/lichess_ks.csv] [CAP=2500] [MINRATING=1400]
"""
import os, sys, csv
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
import chess

SRC = os.environ.get("SRC", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Coach/chess-coach/data/lichess/lichess_db_puzzle.csv")
if not os.path.exists(SRC):
    SRC = SRC.replace("/mnt/c/", "/c/")
OUT = os.environ.get("OUT", "ks_sets/lichess_ks.csv")
if not os.path.isabs(OUT): OUT = os.path.join(THIS, OUT)
CAP = int(os.environ.get("CAP", "2500"))          # per phase stratum
MINRATING = int(os.environ.get("MINRATING", "1400"))
MAXRATING = int(os.environ.get("MAXRATING", "2600"))  # avoid the freak-hard tail (off our strength)

# KS-relevant themes: king attacks / exposed king / mates / sacrifices. NOT generic tactics (fork/pin/hangingPiece).
KS_THEMES = {"kingsideAttack", "queensideAttack", "exposedKing", "attackingF2F7", "sacrifice",
             "mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "smotheredMate",
             "backRankMate", "defensiveMove", "kingAttack"}
PHASES = ("opening", "middlegame", "endgame")

counts = {p: 0 for p in PHASES}
rows_out = []
scanned = 0
with open(SRC, newline="", encoding="utf-8", errors="replace") as f:
    rd = csv.reader(f)
    header = next(rd, None)
    for row in rd:
        scanned += 1
        if len(row) < 8:
            continue
        fen, moves, rating, themes = row[1], row[2], row[3], row[7]
        try:
            rt = int(rating)
        except Exception:
            continue
        if rt < MINRATING or rt > MAXRATING:
            continue
        tset = set(themes.split())
        if not (tset & KS_THEMES):
            continue
        phase = "endgame" if "endgame" in tset else ("opening" if "opening" in tset else "middlegame")
        if counts[phase] >= CAP:
            if all(counts[p] >= CAP for p in PHASES):
                break
            continue
        # apply the opponent setup move (Moves[0]) -> the position the solver actually faces
        mv = moves.split()
        if not mv:
            continue
        try:
            b = chess.Board(fen)
            b.push_uci(mv[0])
            pfen = b.fen()
        except Exception:
            continue
        counts[phase] += 1
        rows_out.append((pfen, phase, rt, mv[1] if len(mv) > 1 else "", " ".join(sorted(tset & KS_THEMES))))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fen", "phase_bucket", "rating", "puzzle_best", "ks_themes"])
    for r in rows_out:
        w.writerow(r)
print("scanned %d puzzles; extracted %d KS-themed positions -> %s" % (scanned, len(rows_out), os.path.basename(OUT)))
print("  by phase: " + "  ".join("%s=%d" % (p, counts[p]) for p in PHASES), flush=True)
