# -*- coding: utf-8 -*-
"""pt_pawns class dossier: among the OTHER-class collapses (units<13), rank by our pt_pawns over-read, pick the
worst 3 + a median, and line up our pt_pawns / material / total (our-POV) vs SF11-static Total and SF18-search.
Goal: see WHERE/HOW we over-value pawns (advanced? passed? too many rank bonuses?) to name the mechanism."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_SAFE_CHECK_DEF'] = '5'
os.environ['KS_FLOOR'] = '0'
import sys, csv
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish
ai = ChessAI(None, None, chess.Board(), True)
GAMES = os.path.join(THIS, "..", "selfplay", "games")

rows = []
for tag in ["night_def5", "night_def5_s1", "night_def5_s2"]:
    cp = os.path.join(GAMES, tag, "collapses.csv")
    if not os.path.exists(cp): continue
    for r in csv.DictReader(open(cp)):
        oc = (r.get("our_color") or "").strip(); drop = (r.get("drop_fen") or "").strip(); dec = (r.get("decision_fen") or "").strip()
        if oc not in ("white", "black") or not drop or not dec: continue
        bdd = ai.ev_breakdown(chess.Board(drop))
        u = bdd.get("det_ks_units_w" if oc == "white" else "det_ks_units_b")
        if u is None or u >= 13: continue    # keep OTHER class only
        b = chess.Board(dec); pov = 1.0 if b.turn else -1.0
        bd = ai.ev_breakdown(b)
        ptp = (-bd.get("pt_pawns", 0.0) / 1000.0) * pov
        rows.append(dict(fen=dec, stm=("W" if b.turn else "B"), ptp=ptp,
                         mat=(-bd.get("material",0)/1000.0)*pov, tot=(-bd.get("total",0)/1000.0)*pov))
rows.sort(key=lambda r: -r["ptp"])
picks = rows[:3] + [rows[len(rows)//2]]
labels = ["WORST ptp", "2nd", "3rd", "MEDIAN"]

sf = SF11Eval(SF11); sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def sf18s(fen):
    bb = chess.Board(fen); info = sf18.analyse(bb, chess.engine.Limit(depth=24)); s = info["score"].white()
    pov = 1.0 if bb.turn else -1.0
    v = (s.mate() and ("M%d"%s.mate())) or ("%+.2f" % (s.score()/100.0 * pov))
    return v
try:
    print("%-11s %3s %8s %8s %8s | %8s %8s" % ("pick","stm","pt_pawns","material","our_tot","SF11tot","SF18(ourPOV)"))
    for lab, r in zip(labels, picks):
        pov = 1.0 if r["stm"]=="W" else -1.0
        sf_tot = sf.eval(r["fen"])[1].get("Total",0.0)*pov
        print("%-11s %3s %+8.2f %+8.2f %+8.2f | %+8.2f %8s" % (lab, r["stm"], r["ptp"], r["mat"], r["tot"], sf_tot, sf18s(r["fen"])))
    print("\nFENs:")
    for lab, r in zip(labels, picks): print("  [%s] %s" % (lab, r["fen"]))
finally:
    sf.close(); sf18.quit()
