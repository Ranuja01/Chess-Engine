# -*- coding: utf-8 -*-
"""Does the calm-control false-fire band (units in 1..13, currently deadzoned) have safe_checks==0 while real
attacks (P2b) have safe>=1? If so, boosting KS_SAFE_CHECK (or a safe-check-conditional deadzone) separates them
cleanly. Dumps per-king KSD for control_calm with the deadzone off, tallies safe-check vs units."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_DEBUG_DUMP'] = '1'
os.environ['KS_FLOOR'] = '0'
import sys, re
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
import io, contextlib

ai = ChessAI(None, None, chess.Board(), True)
path = os.path.join(THIS, "ks_sets", "control_calm.txt")
fens = [ln.rstrip("\n").split("\t", 1)[-1].strip() for ln in open(path) if ln.strip()]

# We can't easily capture C stderr from Python; instead re-derive by running ev_breakdown and reading the
# dumped lines from a redirected fd. Simpler: just count over all fens how many king-evals land in each band.
# Since KSD goes to real stderr, run this whole script under the runner and grep KSD externally. Here we only
# print the fens + a marker so the external grep can associate.
print("CALM_N=%d" % len(fens))
for i, fen in enumerate(fens):
    print("CALMFEN %d %s" % (i, fen)); sys.stdout.flush(); sys.stderr.flush()
    ai.ev_breakdown(chess.Board(fen)); sys.stderr.flush()
