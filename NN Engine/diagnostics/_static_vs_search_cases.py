# -*- coding: utf-8 -*-
"""STATIC vs SEARCH on hand-picked cases (owner, 2026-10-07): for each FEN, can a STATIC eval (ours, SF11 classical,
SF15.1 classical) see the verdict, and at what DEPTH does each engine's SEARCH find it? Separates "the eval lacks the
knowledge" (static wrong, every engine needs search) from "OUR search fails where theirs succeeds at the same depth"
(a search-arc problem: pruning, null move, extensions). Also prints SF11's static trace rows and ours, to see WHICH term
(if any) carries the verdict statically.
All White-POV cp. Writes IN for our own depth pass: ks_sets/<TAG>_cases.csv (fen,best_uci,best_cp = SF18 deepest).
  pyrun diagnostics/_static_vs_search_cases.py FENS=<file, one FEN per line, '# note' allowed> [TAG=pe_cases]
                                              [DEPTHS=10,14,20]      (env: V2_PRESET=shipped)
then: MAX_DEPTH=10 / 14 … pyrun diagnostics/_depth_residual_pass.py IN=ks_sets/<TAG>_cases.csv OUT=ks_sets/<TAG>_ours_d10.csv
"""
import os, sys, csv, re, subprocess
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import chess, chess.engine
import ChessAI
from _triangulate_sf11 import SF11, sf11_eval, sf11_phase, SF11_TEMPO_PAWNS

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ROOT = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/"
SF15 = ROOT + "stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64"
SF18 = ROOT + "stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"


def uci_static(path, fen, opts=()):
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    cmds = "uci\n" + "".join("setoption name %s value %s\n" % o for o in opts) + "position fen %s\neval\nisready\n" % fen
    out, _ = p.communicate(cmds + "quit\n", timeout=60)
    m = re.findall(r"(Classical|Final|Total) evaluation:?\s*([-+]?\d+\.\d+)", out)
    return 100 * float(m[-1][1]) if m else None


def cp_of(info, turn):
    s = info["score"].white()
    return s.score(mate_score=100000)


def main():
    lines = [l.rstrip("\n") for l in open(KV["FENS"]) if l.strip() and not l.startswith("#")]
    cases = [(l.split("#")[0].strip(), (l.split("#", 1)[1].strip() if "#" in l else "")) for l in lines]
    depths = [int(d) for d in KV.get("DEPTHS", "10,14,20").split(",")]
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    p11 = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p11.stdin.write("uci\n"); p11.stdin.flush()
    while p11.stdout.readline().strip() != "uciok":
        pass
    eng = {n: chess.engine.SimpleEngine.popen_uci(pth, timeout=120) for n, pth in (("SF11", SF11), ("SF15c", SF15), ("SF18", SF18))}
    eng["SF15c"].configure({"Use NNUE": False})
    rows = []
    for i, (fen, note) in enumerate(cases, 1):
        b = chess.Board(fen)
        bd = ai.ev_breakdown(b)
        ours = -float(bd["total"]) / 10.0
        tot, terms = sf11_eval(p11, fen)
        sf11s = 100 * tot - (SF11_TEMPO_PAWNS if b.turn == chess.WHITE else -SF11_TEMPO_PAWNS) * 100 if tot is not None else None
        sf15s = uci_static(SF15, fen, (("Use NNUE", "false"),))
        print("\n#%d %s   %s" % (i, fen, note))
        print("  STATIC  ours %+6.0f · SF11 %s · SF15c %s" % (ours, "%+6.0f" % sf11s if sf11s is not None else "  n/a",
                                                         "%+6.0f" % sf15s if sf15s is not None else "  n/a"))
        sph = sf11_phase(b)
        if terms and terms.get("Total", (0, 0)) != (0.0, 0.0):
            bl = lambda t: (t[0] * sph + t[1] * (128 - sph)) / 128 * 100
            print("  SF11 rows: " + "  ".join("%s %+.0f" % (k, bl(v)) for k, v in terms.items()
                                              if k != "Total" and abs(bl(v)) >= 5))
        else:
            print("  SF11 rows: (specialised endgame eval — no term table)")
        print("  ours rows: " + "  ".join("%s %+.0f" % (k, -float(bd[k]) / 10.0) for k in
              ("material", "pieces", "pawn_struct", "v2_passers", "kaufman_imbalance", "v2_winnab", "king_safety")
              if bd.get(k) is not None and abs(float(bd[k])) >= 50))
        best = None
        for n in ("SF11", "SF15c", "SF18"):
            out = []
            for d in depths:
                info = eng[n].analyse(b, chess.engine.Limit(depth=d))
                cp = cp_of(info, b.turn)
                out.append("d%d %+6s" % (d, cp))
                if n == "SF18":
                    best = (info["pv"][0].uci(), cp)
            print("  SEARCH  %-5s %s" % (n, "  ".join(out)))
        rows.append((fen, best[0], best[1]))
    for e in eng.values():
        e.quit()
    p11.stdin.write("quit\n"); p11.stdin.flush()
    out = os.path.join(THIS, "ks_sets", KV.get("TAG", "pe_cases") + "_cases.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "best_uci", "best_cp"])
        w.writerows(rows)
    print("\nIN for our depth pass → %s" % out)


if __name__ == "__main__":
    main()
