# -*- coding: utf-8 -*-
"""TRIANGULATION, side by side with SF11's static eval (owner, 10-02): for each case in a triangulation note, SF11's
classical term table (MG + EG, blended with SF11's own phase) next to OUR term breakdown — "same subsystems, does SF11
get it and where?". SF11 is the achievability control (memory `static-eval-vs-search-move-has-a-42-percent-ceiling`).

Units: everything White-POV centipawns. SF11 prints terms in pawns of PawnValueEg (Trace::to_cp) ⇒ ×100.
SF11 phase (evaluate.cpp / material.cpp): npm (mg values N 781, B 825, R 1276, Q 2538, both sides), clamped to
[EndgameLimit 3915, MidgameLimit 15258] → ph ∈ [0, 128]; blended = (mg·ph + eg·(128 − ph))/128 (the scale factor and
initiative are already inside SF11's Total; per-term blends ignore the scale factor).

  pyrun diagnostics/_triangulate_sf11.py [NOTE=dev_notes/TRIANGULATION-2026-10-02.md]   (run with V2_PRESET=shipped)
"""
import os, sys, re, subprocess
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
import ChessAI
SF11 = os.environ.get("SF11_BIN", "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_11_linux/"
                                  "stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
NOTE = os.path.join(ENGINE, KV.get("NOTE", "dev_notes/TRIANGULATION-2026-10-02.md"))
OURS = ["pieces", "king_safety", "mobility", "pawn_struct", "v2_passers", "v2_placement", "v2_winnab"]


def sf11_eval(p, fen):
    p.stdin.write("position fen %s\neval\nisready\n" % fen); p.stdin.flush()
    total, terms = None, {}
    while True:
        ln = p.stdout.readline()
        if not ln or ln.strip() == "readyok":
            break
        m = re.search(r"Total evaluation:\s*([-+]?\d+\.\d+)", ln)
        if m:
            total = float(m.group(1))
        mm = re.match(r"\s*([A-Za-z ]+?)\s*\|.*\|.*\|\s*([-+]?\d+\.\d+|----)\s+([-+]?\d+\.\d+|----)\s*$", ln)
        if mm and mm.group(2) != "----":
            terms[mm.group(1).strip()] = (float(mm.group(2)), float(mm.group(3)))
    return total, terms


def sf11_phase(b):
    npm = sum(len(b.pieces(pt, c)) * v for pt, v in ((chess.KNIGHT, 781), (chess.BISHOP, 825), (chess.ROOK, 1276),
                                                     (chess.QUEEN, 2538)) for c in (chess.WHITE, chess.BLACK))
    npm = max(3915, min(15258, npm))
    return (npm - 3915) * 128 // (15258 - 3915)


def main():
    cases = re.findall(r"#(\d+)\s+(\S+ [wb] \S+ \S+ \d+ \d+)\n\s+SF18 d14\s+([-+]?\d+) cp.*?d10\s+([-+]?\d+)", open(NOTE).read())
    p = subprocess.Popen([SF11], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    for num, fen, sf18, d10 in cases:
        b = chess.Board(fen)
        total, terms = sf11_eval(p, fen)
        ph = sf11_phase(b)
        bd = ai.ev_breakdown(b)
        ours_static = -float(bd["total"]) / 10.0
        print("\n#%s  %s" % (num, fen))
        print("  TOTALS (White cp):  SF18 d14 %+5d · SF11 static %+5.0f · ours static %+5.0f · ours d10 %+5d"
              % (int(sf18), 100 * total if total is not None else float("nan"), ours_static, int(d10)))
        sf = {k: (mg * ph + eg * (128 - ph)) / 128 * 100 for k, (mg, eg) in terms.items() if k != "Total"}
        print("  SF11 terms (blended, ph %d/128): " % ph + "  ".join("%s %+.0f" % (k, v) for k, v in sf.items() if abs(v) >= 5))
        print("  ours terms:                      " + "  ".join(
            "%s %+.0f" % (t, -float(bd[t]) / 10.0) for t in OURS if bd.get(t) is not None and abs(float(bd[t])) >= 50))
    p.stdin.write("quit\n"); p.stdin.flush()


if __name__ == "__main__":
    main()
