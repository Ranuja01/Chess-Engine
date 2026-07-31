# -*- coding: utf-8 -*-
"""Re-attribute the nighttime collapses by whether OUR king was genuinely under attack at the crash (drop_fen),
using OUR validated KS detector (raw attack-units, DEF=5-aligned) -- a FIXED, non-contaminated (not SF11-static),
non-circular classifier applied identically to baseline and DEF=5 arms. Answers: did the KS-ATTACK collapse
CLASS shrink under DEF=5, even though the TOTAL was flat? (the 'flat total != failure' categorical test)."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['KS_SAFE_CHECK_DEF'] = '5'   # fixed SF-aligned detector for classification
os.environ['KS_FLOOR'] = '0'            # raw units (see below-deadzone attacks too)
import sys, csv
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
GAMES = os.path.join(THIS, "..", "selfplay", "games")
KS_ATTACK_UNITS = 13   # our-king danger units >= deadzone = a real attack on our king

# key check
bd0 = ai.ev_breakdown(chess.Board("1rBq4/2p2pk1/3p2p1/3Pp2n/2p1P2r/1RN1N3/P1P2P1P/4R1K1 w - - 0 31"))
has_units = ("det_ks_units_w" in bd0) and ("det_ks_units_b" in bd0)
print("det_ks_units keys present: %s  (P2b W units=%s B units=%s)" % (
    has_units, bd0.get("det_ks_units_w"), bd0.get("det_ks_units_b")))

def our_units(fen, our_color):   # our_color: 'white'/'black'
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
    except Exception:
        return None
    return bd.get("det_ks_units_w" if our_color == "white" else "det_ks_units_b")

def classify_arm(tags):
    ks_att, other, n = 0, 0, 0
    for t in tags:
        cp = os.path.join(GAMES, t, "collapses.csv")
        if not os.path.exists(cp):
            continue
        for r in csv.DictReader(open(cp)):
            fen = (r.get("drop_fen") or r.get("decision_fen") or "").strip()
            oc = (r.get("our_color") or "").strip()
            if not fen or oc not in ("white", "black"):
                continue
            u = our_units(fen, oc)
            if u is None:
                continue
            n += 1
            if u >= KS_ATTACK_UNITS:
                ks_att += 1
            else:
                other += 1
    return n, ks_att, other

base = classify_arm(["night_base", "night_base_s1", "night_base_s2"])
def5 = classify_arm(["night_def5", "night_def5_s1", "night_def5_s2"])
print("\n%-10s %8s %10s %8s" % ("arm(3seed)", "collapse", "KS-attack", "other"))
print("baseline   %8d %10d %8d" % base)
print("DEF=5      %8d %10d %8d" % def5)
print("\nCATEGORICAL (DEF5 - baseline):  KS-attack %+d   other %+d   total %+d" % (
    def5[1]-base[1], def5[2]-base[2], def5[0]-base[0]))
