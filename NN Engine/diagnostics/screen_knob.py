# -*- coding: utf-8 -*-
"""FAST offline class-residual screen for an eval knob (no games, no SF re-run).

Reads a verify_triage_static --dump corpus (needs fen + sf18_static_cp cached), recomputes OUR static eval
under the given KEY=VAL knobs, and reports the COLLAPSE-vs-CONTROL residual shift — Fable's screen:
a good fix moves the collapse residual toward control's ~+8 while holding control within ±20cp.

  overnight_runner.sh pyrun diagnostics/screen_knob.py diagnostics/corpus_ks.csv MOD_PVBOOST_COMP=200 ...
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

args = sys.argv[1:]
corpus = next((a for a in args if a.endswith(".csv")), os.path.join(THIS, "corpus_ks.csv"))
knobs = [a for a in args if "=" in a and not a.endswith(".csv")]
for kv in knobs:
    k, v = kv.split("=", 1); os.environ[k] = v      # MUST precede ChessAI construction

import chess
from ChessAI import ChessAI
def wp(ev): return -ev / 1000.0

def mean(v): return sum(v) / len(v) if v else 0.0

def main():
    rows = [r for r in csv.DictReader(open(corpus)) if r.get("sf18_static_cp") not in ("", None)]
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    print(f"[screen] knobs: {' '.join(knobs) if knobs else '(none=default)'}   corpus={os.path.basename(corpus)}  n={len(rows)}")
    buck = {}
    for r in rows:
        try:
            b = chess.Board(r["fen"]); bd = ai.ev_breakdown(b)
            if bd.get("checkmate"): continue
        except Exception:
            continue
        mover = 1 if b.turn == chess.WHITE else -1
        new_static = wp(bd["total"]) * 100.0 * mover
        new_res = new_static - float(r["sf18_static_cp"])
        old_res = float(r["over_read_cp"])
        eg = "EG" if r["is_endgame"] == "1" else "MG"
        key = ("COLLAPSE-" + eg) if r["is_collapse"] == "1" else "CONTROL"
        buck.setdefault(key, {"old": [], "new": []})
        buck[key]["old"].append(old_res); buck[key]["new"].append(new_res)
    print(f"  {'bucket':>14} {'n':>4} {'old_resid':>10} {'new_resid':>10} {'Δ':>8}")
    for k in ("COLLAPSE-MG", "COLLAPSE-EG", "CONTROL"):
        if k in buck:
            o, n = mean(buck[k]["old"]), mean(buck[k]["new"])
            print(f"  {k:>14} {len(buck[k]['new']):>4} {o:>+10.0f} {n:>+10.0f} {n-o:>+8.0f}")
    print("  PASS = collapse residual drops toward control's ~+8 AND CONTROL Δ within ±20cp (else damp-the-tail).")

if __name__ == "__main__":
    main()
