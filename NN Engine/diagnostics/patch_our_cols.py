# -*- coding: utf-8 -*-
"""Fast corpus iteration for eval-feature builds: recompute only OUR columns (our_total + a named term) with the
current engine build, keeping SF11 labels + result + game intact. Avoids the slow SF11 relabel (~30 min) when only
our reimplementation changed. Set ENABLE_<term>=1 in the env before running. Usage:
    ENABLE_THREATS=1 python diagnostics/patch_our_cols.py <in_corpus.csv> <out_corpus.csv> <term>
"""
import csv, os, sys
src, dst, term = sys.argv[1], sys.argv[2], sys.argv[3]
extra = [a for a in sys.argv[4:] if "=" in a]   # explicit KEY=VAL env knobs (e.g. KING_SAFETY_MAG=4000)
if extra:
    for kv in extra:
        k, v = kv.split("=", 1); os.environ[k] = v
else:
    os.environ["ENABLE_" + term.upper()] = "1"   # default: enable the term by name (env latches at first ChessAI())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
from ChessAI import ChessAI

b0 = chess.Board(); ai = ChessAI(None, None, b0, b0.turn)
rows = list(csv.DictReader(open(src)))
for i, r in enumerate(rows):
    bd = ai.ev_breakdown(chess.Board(r["fen"]))
    r["our_total"] = bd["total"]
    r[term] = bd[term]
    if (i + 1) % 5000 == 0:
        print("  %d/%d" % (i + 1, len(rows)))
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("patched %s (%s + our_total) -> %s" % (term, len(rows), dst))
