# -*- coding: utf-8 -*-
"""Slice screen for the npedge damp: the l1a guard. Recompute our static under a knob and report the
residual shift PER SLICE that matters for the fantasy-vs-real fix:

  FANTASY   (collapse, our_orig>=150, sf<=50)   -> want DROP toward control.
  REAL-WIN  (control,  our_orig>=150, sf>=150)  -> want HELD within +-20cp (l1a killed these).
  REAL-LOWNP(real-win subset with npedge<250)   -> the 15 the damp could wrongly hit; the tightest guard.
  CONTROL   (all control)                        -> broad hold +-20cp.
  COLLAPSE/CONTROL split MG/EG for the endgame-gate correctness check (EG deltas must be ~0).

  overnight_runner.sh pyrun diagnostics/screen_slices.py diagnostics/corpus_ks.csv ENABLE_NPEDGE_DAMP=true [NPEDGE_DAMP_MAX=150 ...]
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)

args = sys.argv[1:]
corpus = next((a for a in args if a.endswith(".csv")), os.path.join(THIS, "corpus_ks.csv"))
knobs = [a for a in args if "=" in a and not a.endswith(".csv")]
for kv in knobs:
    k, v = kv.split("=", 1); os.environ[k] = v

import chess
from ChessAI import ChessAI
def wp(ev): return -ev / 1000.0
def mean(v): return sum(v) / len(v) if v else 0.0


def main():
    rows = [r for r in csv.DictReader(open(corpus)) if r.get("sf18_static_cp") not in ("", None)]
    seed = chess.Board(); ai = ChessAI(None, None, seed, seed.turn)
    print(f"[slices] knobs: {' '.join(knobs) if knobs else '(default)'}  n={len(rows)}")
    buck = {}
    def add(key, old, new):
        buck.setdefault(key, {"old": [], "new": []}); buck[key]["old"].append(old); buck[key]["new"].append(new)
    for r in rows:
        try:
            b = chess.Board(r["fen"]); bd = ai.ev_breakdown(b)
            if bd.get("checkmate"): continue
        except Exception:
            continue
        mover = 1 if b.turn == chess.WHITE else -1
        new_res = wp(bd["total"]) * 100.0 * mover - float(r["sf18_static_cp"])
        old_res = float(r["over_read_cp"])
        sf = float(r["sf18_static_cp"]); our = old_res + sf
        eg = r["is_endgame"] == "1"; coll = r["is_collapse"] == "1"
        npe = float(r.get("npedge", 0) or 0)
        add("CONTROL-EG" if eg else "CONTROL-MG", old_res, new_res) if not coll else \
            add("COLLAPSE-EG" if eg else "COLLAPSE-MG", old_res, new_res)
        if coll and our >= 150 and sf <= 50:
            add("FANTASY", old_res, new_res)
            if not eg: add("FANTASY-MG", old_res, new_res)
        if not coll and our >= 150 and sf >= 150:
            add("REAL-WIN", old_res, new_res)
            if npe < 250: add("REAL-WIN-LOWNP", old_res, new_res)
            if not eg: add("REAL-WIN-MG", old_res, new_res)
    order = ["FANTASY", "FANTASY-MG", "REAL-WIN", "REAL-WIN-MG", "REAL-WIN-LOWNP",
             "COLLAPSE-MG", "COLLAPSE-EG", "CONTROL-MG", "CONTROL-EG"]
    print(f"  {'slice':>16} {'n':>4} {'old':>8} {'new':>8} {'Δ':>7}   note")
    notes = {"FANTASY": "want DROP", "FANTASY-MG": "want DROP (target)", "REAL-WIN": "HOLD +-20",
             "REAL-WIN-MG": "HOLD +-20", "REAL-WIN-LOWNP": "HOLD (tightest)",
             "COLLAPSE-EG": "gate: ~0", "CONTROL-EG": "gate: ~0", "CONTROL-MG": "HOLD +-20"}
    for k in order:
        if k in buck:
            o, n = mean(buck[k]["old"]), mean(buck[k]["new"])
            print(f"  {k:>16} {len(buck[k]['new']):>4} {o:>+8.0f} {n:>+8.0f} {n-o:>+7.0f}   {notes.get(k,'')}")
    print("  PASS = FANTASY-MG drops, REAL-WIN* held +-20, EG slices ~0 (midgame gate). Trade at the frontier.")


if __name__ == "__main__":
    main()
