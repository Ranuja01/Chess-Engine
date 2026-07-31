# -*- coding: utf-8 -*-
"""Does OUR engine avoid the over-push at a FIXED depth (matched to SF11's d6/d10/d14 test)? The SF11 test
compared SF11 to our GAME move (250k nodes); this compares OUR ENGINE at a fixed depth to the move we
played, so we can put our per-depth soundness next to SF11's (d6 66% / d10 83% / d14 94%).

  # run once per depth (run_one reads MAX_DEPTH from env, latched at engine init):
  overnight_runner.sh pyrun diagnostics/our_depth_avoid.py diagnostics/collapse_classified.csv --cat overpush MAX_DEPTH=10 PRESET=LONG_FORMAT

Reads the played move from the classified dump; avoid = our fixed-depth move != the move we played.
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
for _kv in [a for a in sys.argv[1:] if "=" in a and not a.startswith("-") and not a.endswith(".csv")]:
    _k, _v = _kv.split("=", 1); os.environ[_k] = _v
argv = [a for a in sys.argv[1:] if not ("=" in a and not a.startswith("-") and not a.endswith(".csv"))]
CAT = "overpush"
if "--cat" in argv: i = argv.index("--cat"); CAT = argv[i+1]; del argv[i:i+2]
csv_path = next((a for a in argv if a.endswith(".csv")), os.path.join(THIS, "collapse_classified.csv"))

import chess
from tactical_test import run_one


def main():
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("category") == CAT]
    depth = os.environ.get("MAX_DEPTH", "?")
    n = 0; avoid = 0; match_sf = 0
    for r in rows:
        fen = r["fen"]; played = r["our_move"]; sfb = r.get("sf_best", "")
        try:
            our = run_one(fen, set())["uci"]
        except Exception:
            continue
        n += 1
        if our != played: avoid += 1        # our fixed-depth engine does NOT repeat the over-push
        if our == sfb: match_sf += 1         # our fixed-depth move equals SF18's best
    if not n:
        print("no rows"); return
    print(f"[our_depth_avoid] cat={CAT}  OUR engine MAX_DEPTH={depth}  n={n}")
    print(f"  avoids the over-push (move != played): {100*avoid/n:.0f}%  ({avoid}/{n})")
    print(f"  plays SF18's exact best move:          {100*match_sf/n:.0f}%  ({match_sf}/{n})")
    print("  compare to SF11 classical search: d6 66% / d10 83% / d14 94%.")
    print("  READ: ours << SF11 at equal depth => our search+eval weaker per-depth (the real gap).")
    print("        ours ~= SF11 => we CAN avoid it shallow; the game blunder is not reaching depth at 250k nodes.")


if __name__ == "__main__":
    main()
