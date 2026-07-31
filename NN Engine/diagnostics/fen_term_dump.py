# -*- coding: utf-8 -*-
"""Full term-by-term dump for specific FENs: OUR ev_breakdown (every term, WHITE-POV pawns) beside SF11-static and
SF15-static classical per-term tables -- to see exactly which terms build our eval and where SF sees material
compensation / a balanced king-attack. Edit FENS below (or pass FEN='...' args).
  pyrun diagnostics/fen_term_dump.py
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
argfens = []
for a in sys.argv[1:]:
    if a.startswith('FEN='): argfens.append(a.split('=', 1)[1])
    elif '=' in a: k, v = a.split('=', 1); os.environ.setdefault(k, v)
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from eval_vs_sf11 import SF11Eval, SF11
from ChessAI import ChessAI
SF15BIN = os.environ.get('SF15_BIN', "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")

FENS = argfens or [
    "r7/5pk1/1pR3p1/2p5/4R2P/2PP2p1/pP2B1q1/Q1K5 w - - 0 33",              # #1 material + compensation
    "r6r/3b1p2/2p5/1pp1kPpp/p1PnP3/P1NP3P/1P6/1R1B1R1K w - - 0 38",        # #2 +1 pawn, doubled/pair, OvD hot
    "1Q6/p4kr1/2p3q1/3p2B1/6bP/3P4/PPP5/3N2K1 w - - 2 26",                 # #3 KS overpowering
    "r7/ppN2k2/3p1np1/3Pnb2/P1PQ2pq/4P1R1/1P3P1r/R1B1KB2 w Q - 3 22",      # #4 equal material, ours +8.7
]


class SF15C:  # SF15.1 classical (NNUE off) per-term table
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._s('uci'); self._d('uciok'); self._s('setoption name Use NNUE value false'); self._s('isready'); self._d('readyok')
    def _s(self, x): self.p.stdin.write(x + '\n'); self.p.stdin.flush()
    def _d(self, t):
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip().startswith(t): break
    def eval(self, fen):
        self._s('position fen %s' % fen); self._s('eval'); self._s('isready')
        total, terms = None, {}
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.startswith('readyok'): break
            m = re.search(r'Classical evaluation\s+([-+]?\d+\.\d+)', ln)
            if m: total = float(m.group(1))
            mm = re.match(r'\|\s*([A-Za-z ]+?)\s*\|[^|]*\|[^|]*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)', ln)
            if mm:
                try: terms[mm.group(1).strip()] = float(mm.group(2))
                except ValueError: pass
        return total, terms
    def close(self):
        try: self._s('quit'); self.p.wait(timeout=2)
        except Exception: self.p.kill()


ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11); sf15 = SF15C(SF15BIN)
# OUR terms to show (WHITE-POV pawns), grouped
SHOW = ["material", "kaufman_imbalance", "pair_bonus", "pieces", "pt_pawns", "pt_knights", "pt_bishops",
        "pt_rooks", "pt_queens", "pt_kings", "king_safety", "imbalance_white", "imbalance_black",
        "capture_gains", "passed_pawn_support", "latent_threat", "threats", "central", "space", "mobility",
        "outpost", "pawn_struct", "pawn_majority", "rook_cond", "piece_value_boost"]

for fen in FENS:
    b = chess.Board(fen)
    bd = ai.ev_breakdown(b)
    our = -bd.get("total", 0.0) / 1000.0
    s11t, s11 = sf11.eval(fen); s15t, s15 = sf15.eval(fen)
    print("\n" + "=" * 100)
    print("FEN: %s   (stm=%s)" % (fen, "white" if b.turn else "black"))
    print("  OURS total = %+.2f    SF11-static = %+.2f    SF15-static = %+.2f" % (our, s11t or 0, s15t or 0))
    print("  --- OUR terms (white-POV pawns) ---")
    line = ""
    for k in SHOW:
        if k in bd and isinstance(bd[k], (int, float)):
            v = -float(bd[k]) / 1000.0
            if abs(v) >= 0.01:
                line += "%s=%+.2f  " % (k, v)
    print("   " + (line or "(none)"))
    print("  --- SF11 terms ---")
    print("   " + "  ".join("%s=%+.2f" % (k, v) for k, v in s11.items() if abs(v) >= 0.01))
    print("  --- SF15 terms ---")
    print("   " + "  ".join("%s=%+.2f" % (k, v) for k, v in s15.items() if abs(v) >= 0.01))

sf11.close(); sf15.close()
