# -*- coding: utf-8 -*-
"""Re-validate the SF11-STATIC KS dossier against SF18-SEARCH (truth). The dossier (ks_underread_vs_sf11.txt)
flagged 36 positions where SF11-static saw big king-safety and we read <half. But SF11-static is unreliable on
sharp positions (#3 proved it: SF11-static -1.53, SF18-search 0.0 = ours correct). So keep only positions where
SF18-SEARCH confirms a genuine edge in the KS direction that WE under-read -> the real must-fix set. Writes
ks_underread_sf18.txt."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

fens = [ln.strip() for ln in open(os.path.join(THIS, "ks_sets", "ks_underread_vs_sf11.txt")) if ln.strip()]
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def sf18_eval(fen, depth=22):
    b = chess.Board(fen); info = sf18.analyse(b, chess.engine.Limit(depth=depth)); s = info["score"].white()
    return 99.0 if s.is_mate() and s.mate() > 0 else (-99.0 if s.is_mate() else s.score() / 100.0)

genuine, artifact = [], []
print("%-8s %-8s %-8s %-8s %-9s  fen" % ("SF18", "our_tot", "our_KS", "SF11_KS", "verdict"))
for fen in fens:
    b = chess.Board(fen)
    try:
        bd = ai.ev_breakdown(b)
        our = -bd.get("total", 0.0) / 1000.0
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        _, terms = sf11.eval(fen)
        sf11_ks = terms.get("King safety", 0.0)
        s18 = sf18_eval(fen)
    except Exception:
        continue
    # GENUINE KS gap: SF18 confirms an edge in the SF11-KS direction (same sign, |SF18|>=1.0) that we under-read
    # (our static total at least 0.6 short of SF18 in that direction). ARTIFACT: SF18 ~0 or opposite sign.
    same_dir = (s18 > 0) == (sf11_ks > 0)
    confirmed = same_dir and abs(s18) >= 1.0 and (abs(s18) - abs(our)) >= 0.6
    v = "GENUINE" if confirmed else ("artifact" if abs(s18) < 0.75 or not same_dir else "weak")
    (genuine if confirmed else artifact).append(fen)
    print("%+8.2f %+8.2f %+8.2f %+8.2f  %-9s  %s" % (s18, our, our_ks, sf11_ks, v, fen))
sf11.close(); sf18.quit()

out = os.path.join(THIS, "ks_sets", "ks_underread_sf18.txt")
with open(out, "w") as f:
    for fen in genuine: f.write(fen + "\n")
print("\nGENUINE (SF18-confirmed KS gaps): %d / %d   -> %s" % (len(genuine), len(fens), out))
print("dropped as SF11-static artifacts / weak: %d" % len(artifact))
