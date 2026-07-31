# -*- coding: utf-8 -*-
"""THE eval-vs-search triage question: when we over-read a position (see material, miss the attack), does
SF11's STATIC eval SEE THROUGH the material to the attack (agree with SF18-search), or does SF11-static ALSO
just count material (agree with us)? If SF11-static overturns the material -> it's a real STATIC-eval capability
we lack (worth building). If SF11-static ALSO over-reads while SF18-SEARCH refutes -> it's a SEARCH/tactical
property no static eval captures (chasing a KS-magnitude eval fix would repeat past failures).

All values WHITE-POV pawns. Run: pyrun diagnostics/static_vs_search_triage.py "<fen>" ["<fen2>" ...]"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

fens = [a for a in sys.argv[1:] if "/" in a]
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

def sf18_wpov(fen, depth=22):
    b = chess.Board(fen); info = sf18.analyse(b, chess.engine.Limit(depth=depth)); s = info["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

print("%-9s %-9s %-9s %-9s  %-24s fen" % ("ourStat", "ourKS", "SF11stat", "SF18srch", "verdict"))
for fen in fens:
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        our = -bd.get("total", 0.0) / 1000.0            # white-pov
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        sf11_tot, terms = sf11.eval(fen)                # white-pov total
        s18 = sf18_wpov(fen)
    except Exception as e:
        print("ERR", fen, e); continue
    # Does SF11-static side with SEARCH (sees attack) or with US (sees material)?
    sf11_like_search = abs(sf11_tot - s18) < abs(sf11_tot - our) and (s18 > 0) == (sf11_tot > 0)
    if abs(our - s18) < 1.0:
        verdict = "we're-fine"
    elif sf11_like_search:
        verdict = "SF11-STATIC-sees-attack"   # real static-eval lever
    else:
        verdict = "SF11-static-ALSO-misses"   # search property, not static eval
    print("%+9.2f %+9.2f %+9.2f %+9.2f  %-24s %s" % (our, our_ks, sf11_tot, s18, verdict, fen))
sf11.close(); sf18.quit()
