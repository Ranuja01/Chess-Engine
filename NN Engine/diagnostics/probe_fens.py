# -*- coding: utf-8 -*-
"""Static-vs-static (+search reference) probe on an EXPLICIT list of FENs. For each: our ev_breakdown, SF11's
classical static breakdown, SF18's static (NNUE `eval`) and SF18 depth search. ALL in WHITE-POV pawns
(positive = good for White) so cross-engine reasoning is apples-to-apples. Reads 'label<TAB>fen' lines.

Run: pyrun diagnostics/probe_fens.py <file> [--sf-depth 22]
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import re
import subprocess
import argparse

import chess
import chess.engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)
from eval_vs_sf11 import SF11Eval, SF11

SF18 = os.environ["STOCKFISH_PATH"]
OUR_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "latent_threat", "threats",
             "king_safety", "central", "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost",
             "pawn_majority", "pawn_struct", "outpost", "mobility",
             "pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


class RawStatic:
    """UCI `eval` command reader -> Final evaluation (White-POV pawns)."""
    def __init__(self, path):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\n"); self.p.stdin.flush()
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "uciok":
                break

    def final_eval(self, fen):
        self.p.stdin.write("position fen %s\neval\nisready\n" % fen); self.p.stdin.flush()
        val = None
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break
            m = re.search(r"Final evaluation\s*:?\s*([-+]?\d+\.\d+)", ln)
            if m:
                val = float(m.group(1))
        return val

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush(); self.p.wait(timeout=2)
        except Exception:
            try: self.p.kill()
            except Exception: pass


SF15 = os.environ.get("SF15_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/"
                                  "stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")
K_LICHESS = 0.00368208   # same logistic the fit scripts use


def winpct(pawns):
    """Lichess win% from an eval in pawns. Ranking by this rather than raw cp matters: 2 pawns of error at
    +8 barely changes the expected result, while 2 pawns at 0.0 flips the game."""
    import math
    return 100.0 / (1.0 + math.exp(-K_LICHESS * pawns * 100.0))


class NNUEStatic(RawStatic):
    """RawStatic with an explicit Use NNUE setting, so one binary gives both a classical and an NNUE column.
    SF15.1 is the LAST classical-king-safety Stockfish: NNUE-off is the ceiling a handcrafted eval can reach,
    and the on-vs-off delta on the SAME binary separates 'un-encodable by hand' from 'we are simply missing it'."""
    def __init__(self, path, nnue):
        super().__init__(path)
        self.p.stdin.write("setoption name Use NNUE value %s\nisready\n" % ("true" if nnue else "false"))
        self.p.stdin.flush()
        while True:
            ln = self.p.stdout.readline()
            if not ln or ln.strip() == "readyok":
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--sf-depth", type=int, default=22)
    ap.add_argument("--table", action="store_true",
                    help="compact one-row-per-FEN table (FEN, engine columns, win%% error last)")
    args = ap.parse_args()
    items = []
    for ln in open(args.file):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        label, fen = (ln.split("\t", 1) if "\t" in ln else ("", ln))
        items.append((label.strip(), fen.strip()))

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)
    sf11 = SF11Eval(SF11)
    sf18s = RawStatic(SF18)
    sf15c = NNUEStatic(SF15, nnue=False)
    sf15n = NNUEStatic(SF15, nnue=True)
    sf18 = chess.engine.SimpleEngine.popen_uci(SF18)
    if args.table:
        print("%-72s %7s %7s %7s %7s %7s %9s %9s"
              % ("FEN", "OURS", "SF11", "SF15c", "SF15n", "SF18s", "SF18@d%d" % args.sf_depth, "win%err"))
    try:
        for label, fen in items:
            b = chess.Board(fen)
            bd = ai.ev_breakdown(b)
            our_white = -bd["total"] / 1000.0
            our_terms = {t: -bd.get(t, 0.0) / 1000.0 for t in OUR_TERMS}   # White-POV
            sf11_tot, sf11_terms = sf11.eval(fen)
            sf18_static = sf18s.final_eval(fen)
            s15c = sf15c.final_eval(fen)
            s15n = sf15n.final_eval(fen)
            info = sf18.analyse(b, chess.engine.Limit(depth=args.sf_depth))
            sf18_search = info["score"].white().score(mate_score=100000) / 100.0
            if args.table:
                fmt = lambda v: "n/a" if v is None else "%+.2f" % v
                # Win% error is vs SF18 search (the truth column), signed in the side-to-move's favour.
                sgn = 1.0 if b.turn == chess.WHITE else -1.0
                werr = (winpct(our_white) - winpct(sf18_search)) * sgn
                print("%-72s %7s %7s %7s %7s %7s %9s %+8.1f"
                      % (fen, fmt(our_white), fmt(sf11_tot), fmt(s15c), fmt(s15n),
                         fmt(sf18_static), fmt(sf18_search), werr))
                continue
            print("=" * 100)
            print("%s   [%s]" % (label, fen))
            print("  OURS static  %+6.2f   |  SF11 static %+6.2f   |  SF18 static %s   |  SF18 d%d %+.2f"
                  % (our_white, sf11_tot if sf11_tot is not None else float('nan'),
                     ("%+.2f" % sf18_static) if sf18_static is not None else "n/a", args.sf_depth, sf18_search))
            ot = sorted(our_terms.items(), key=lambda kv: -abs(kv[1]))
            print("  OUR terms : " + "  ".join("%s=%+.2f" % (t, v) for t, v in ot if abs(v) > 0.08))
            st = sorted(((k, v) for k, v in sf11_terms.items() if k != "Total"), key=lambda kv: -abs(kv[1]))
            print("  SF11 terms: " + "  ".join("%s=%+.2f" % (t, v) for t, v in st if abs(v) > 0.08))
    finally:
        sf11.close(); sf18s.close(); sf15c.close(); sf15n.close(); sf18.quit()


if __name__ == "__main__":
    main()
