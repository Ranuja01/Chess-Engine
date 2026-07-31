# -*- coding: utf-8 -*-
"""Split the 'collapses' by ACTUAL OUTCOME (loss vs draw vs win) — are we plugging real LOSSES or just draws?

A "collapse" = our eval peaked winning then we didn't win. But that bundles draws-we'd-draw-anyway (cosmetic)
with real LOSSES (the games worth saving). Reads triage CSVs (have game result + our_color) and reports the
loss/draw/win split, overall and by class (EVAL/HORIZON/PRUNING) and phase (EG/MG via is-endgame-ish peak).

  python diagnostics/wdl_collapse.py games/g_base_s0/triage.csv games/g_base_s1/triage.csv
"""
import sys, csv


def outcome(result, color):
    if result == "1/2-1/2":
        return "draw"
    if (result == "1-0" and color == "white") or (result == "0-1" and color == "black"):
        return "win"
    return "loss"


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".csv")]
    rows = []
    for p in paths:
        try:
            rows += list(csv.DictReader(open(p)))
        except Exception:
            pass
    if not rows:
        print("no rows"); return

    def tally(sub, label):
        n = {"loss": 0, "draw": 0, "win": 0}
        for r in sub:
            n[outcome(r.get("result", ""), r.get("our_color", ""))] += 1
        t = len(sub)
        print(f"  {label:>16}: n={t:>3}  loss={n['loss']:>3} ({100*n['loss']/t:>3.0f}%)  "
              f"draw={n['draw']:>3} ({100*n['draw']/t:>3.0f}%)  win={n['win']:>2}")

    print(f"[wdl] {len(rows)} collapses across {len(paths)} corpora")
    tally(rows, "ALL")
    for cls in ("EVAL", "HORIZON", "PRUNING"):
        s = [r for r in rows if r.get("class") == cls]
        if s: tally(s, cls)
    print("\nREAD: high LOSS% => collapses are real dropped points (worth a sharp fix). high DRAW% => cosmetic")
    print("      (we'd draw anyway; damping to remove the flag only risks turning our WINS into draws = l1a's -11%).")


if __name__ == "__main__":
    main()
