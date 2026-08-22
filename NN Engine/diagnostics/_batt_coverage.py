# -*- coding: utf-8 -*-
"""Battery-feeder coverage over the collapse set. Engine-only (no SF): dumps the White-POV king_safety
term for every collapse FEN under whatever KS_BATTERY the caller sets. Knobs latch once per process, so
this sets os.environ BEFORE constructing ChessAI and is run once per setting.

Run (via the allowlisted runner):
  bash <runner> pyrun diagnostics/_batt_coverage.py <KS_BATTERY_value> <out.tsv> [fens.csv]
Then Read the two out files and diff. Default FEN source = the collapse decision FENs (dp_fens.csv).
"""
import os, sys
os.environ['KS_BATTERY'] = sys.argv[1] if len(sys.argv) > 1 else '0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

OUT = sys.argv[2] if len(sys.argv) > 2 else 'diagnostics/_batt_cov.tsv'
FENS = sys.argv[3] if len(sys.argv) > 3 else 'selfplay/games/vssf_2400/dp_fens.csv'

import csv
import chess
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
from ChessAI import ChessAI

seed = chess.Board()
ai = ChessAI(None, None, seed, seed.turn)

rows = []
with open(os.path.join(ENGINE_DIR, FENS)) as f:
    rd = csv.DictReader(f)
    col = 'fen_start' if 'fen_start' in (rd.fieldnames or []) else (rd.fieldnames or ['fen'])[0]
    for r in rd:
        fen = (r.get(col) or '').strip()
        if fen:
            rows.append(fen)

with open(os.path.join(ENGINE_DIR, OUT), 'w') as out:
    for fen in rows:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        bd = ai.ev_breakdown(b)
        ks = -bd.get('king_safety', 0) / 1000.0      # White-POV pawns
        tot = -bd.get('total', 0) / 1000.0
        out.write("%.3f\t%.3f\t%s\n" % (ks, tot, fen))
print("wrote %s (%d fens, KS_BATTERY=%s)" % (OUT, len(rows), os.environ['KS_BATTERY']))
