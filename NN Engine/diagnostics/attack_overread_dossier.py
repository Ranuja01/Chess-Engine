# -*- coding: utf-8 -*-
"""Representative-FEN dossier for the attackingLayer/OvD attack-over-read (the residual 'positional' collapse
class). Ranks positional-class collapse decision_fens by how much WE over-read vs SF11-static (that's where the
attackingLayer bite shows), and prints each with the reference evals side by side for manual analysis:
  ours | SF11-static | SF15-static(classical) | SF15-NNUE | SF18-search(depth)   (all WHITE-POV pawns)
plus our king_safety and our attack-family split (pieces + imbalance_white) so the hot term is visible.
  pyrun diagnostics/attack_overread_dossier.py [N=20] [DEPTH=18] [CLASS=positional]
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
for a in sys.argv[1:]:
    if '=' in a: k, v = a.split('=', 1); os.environ.setdefault(k, v)
N = int(os.environ.get('N', '20')); DEPTH = int(os.environ.get('DEPTH', '18'))
CLASS = os.environ.get('CLASS', 'positional'); POOL = int(os.environ.get('POOL', '250'))
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish
from ChessAI import ChessAI
SF15BIN = os.environ.get('SF15_BIN', "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")


class SFEval:
    """SF15 UCI eval, NNUE on or off. Returns white-POV pawns (parses Classical/NNUE/Final evaluation line)."""
    def __init__(self, path, nnue):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._s('uci'); self._d('uciok')
        self._s('setoption name Use NNUE value %s' % ('true' if nnue else 'false')); self._s('isready'); self._d('readyok')
    def _s(self, x): self.p.stdin.write(x + '\n'); self.p.stdin.flush()
    def _d(self, tok):
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip().startswith(tok): break
    def eval(self, fen):
        self._s('position fen %s' % fen); self._s('eval'); self._s('isready')
        val = None
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.startswith('readyok'): break
            m = re.search(r'(?:Classical|NNUE|Final) evaluation\s+([-+]?\d+\.\d+)', ln)
            if m: val = float(m.group(1))
        return val
    def close(self):
        try: self._s('quit'); self.p.wait(timeout=2)
        except Exception:
            try: self.p.kill()
            except Exception: pass


ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf15s = SFEval(SF15BIN, False); sf15n = SFEval(SF15BIN, True)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
try: sf18.configure({"Threads": 4})
except Exception: pass


def s18(fen):
    i = sf18.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH)); s = i["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)


CLS = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
fens = [r["decision_fen"] for r in csv.DictReader(open(CLS))
        if r.get("ks_class") == CLASS and r.get("decision_fen")][:POOL]

# rank by |our - SF11-static| over-read
scored = []
for fen in fens:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        if bd.get("checkmate"): continue
        our = -bd.get("total", 0.0) / 1000.0
        s11, _ = sf11.eval(fen)
        scored.append((abs(our - (s11 or 0.0)), fen, bd, our, s11))
    except Exception:
        continue
scored.sort(reverse=True)

print("\n%-4s %7s %7s %7s %7s %7s | %6s %6s %6s  %s" %
      ("stm", "ours", "sf11s", "sf15s", "sf15nn", "sf18", "ks", "pieces", "imbW", "fen"))
print("-" * 120)
for _, fen, bd, our, s11 in scored[:N]:
    stm = "w" if chess.Board(fen).turn else "b"
    ks = -bd.get("king_safety", 0.0) / 1000.0
    pieces = -bd.get("pieces", 0.0) / 1000.0
    imbw = -bd.get("imbalance_white", 0.0) / 1000.0
    print("%-4s %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f | %+6.2f %+6.2f %+6.2f  %s" %
          (stm, our, (s11 or 0.0), (sf15s.eval(fen) or 0.0), (sf15n.eval(fen) or 0.0), s18(fen),
           ks, pieces, imbw, fen))

sf11.close(); sf15s.close(); sf15n.close(); sf18.quit()
