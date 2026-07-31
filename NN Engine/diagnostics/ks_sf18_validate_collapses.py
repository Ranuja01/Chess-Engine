# -*- coding: utf-8 -*-
"""SF18-validate GAME-COLLAPSE decision fens (post-Kaufman KS-attack class). Each row = (our_color, peak_eval,
decision_fen): a position where OUR engine thought we were clearly winning (high peak) and the game then
collapsed via a king attack (classifier tag). Question per position: did SF18-SEARCH also think we were winning
at the DECISION point?
  - SF18 says NOT winning (our-POV small/negative) while we thought winning  -> GENUINE eval OVER-READ at the
    decision (our eval, incl. king safety, missed the danger) = a real must-fix example.
  - SF18 also says winning                                                    -> ARTIFACT: the loss came LATER
    (tactical/search), not a decision-point eval error. Not a KS-eval target.
Keeps genuine, ranks by over-read magnitude, prints a middle-layer sample + writes ks_underread_sf18_postkauf.txt.
Run: pyrun diagnostics/ks_sf18_validate_collapses.py [DEPTH=20]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
DEPTH = int(os.environ.get("DEPTH", "20"))
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

rows = []
for ln in open(os.path.join(THIS, "ks_sets", "post_kauf_ks_decisions.tsv")):
    parts = ln.rstrip("\n").split("\t")
    if len(parts) == 3:
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

ai = ChessAI(None, None, chess.Board(), True)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def sf18_wpov(fen, depth=DEPTH):
    b = chess.Board(fen); info = sf18.analyse(b, chess.engine.Limit(depth=depth)); s = info["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

def to_our_pov(white_pov, our_color):
    return white_pov if our_color.startswith("w") else -white_pov

genuine = []
print("%-6s %8s %8s %8s %8s  %-9s fen" % ("side", "ourEval", "SF18", "ourKS", "overread", "verdict"))
for our_color, peak_eval, fen in rows:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        our_wpov = -bd.get("total", 0.0) / 1000.0          # abs black-positive -> white POV
        ks_wpov = -bd.get("king_safety", 0.0) / 1000.0
        s18_w = sf18_wpov(fen)
    except Exception:
        continue
    ours = to_our_pov(our_wpov, our_color)
    s18 = to_our_pov(s18_w, our_color)
    ks = to_our_pov(ks_wpov, our_color)
    overread = ours - s18
    # GENUINE decision-point over-read: we thought clearly winning, SF18 says we were not.
    confirmed = ours >= 1.5 and s18 <= 0.75 and overread >= 1.5
    v = "GENUINE" if confirmed else ("artifact(SF18-agrees-win)" if s18 > 0.75 else "weak")
    if confirmed:
        genuine.append((overread, our_color, ours, s18, ks, fen))
    print("%-6s %+8.2f %+8.2f %+8.2f %+8.2f  %-9s %s" % (our_color[:5], ours, s18, ks, overread, v[:9], fen))

sf18.quit()
genuine.sort(reverse=True)
out = os.path.join(THIS, "ks_sets", "ks_underread_sf18_postkauf.txt")
with open(out, "w") as f:
    for _, _, _, _, _, fen in genuine:
        f.write(fen + "\n")
print("\nGENUINE decision-point over-reads (SF18-confirmed): %d / %d  -> %s" % (len(genuine), len(rows), out))
print("\n== middle-layer sample (worst over-reads) for dissection ==")
for overread, col, ours, s18, ks, fen in genuine[:5]:
    print("  we(%s) thought %+.2f, SF18 %+.2f (overread %+.2f), our KS term %+.2f\n    %s" % (
        col, ours, s18, overread, ks, fen))
