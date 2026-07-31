# -*- coding: utf-8 -*-
"""Triage a category slice of collapse_classified.csv as EVAL / HORIZON / PRUNING — is the losing move
search-fixable or a genuine eval hole? For each FEN in the category, compare our DEEP move (env MAX_DEPTH)
and our deep LOW-PRUNE move against the move we actually played (the blunder):
  HORIZON : deep search already picks a DIFFERENT move than the blunder -> more depth fixes it.
  PRUNING : deep still plays the blunder, but low-prune (LMP/null/LMR relaxed) fixes it -> pruning hole.
  EVAL    : deep AND low-prune still play the blunder -> genuine eval hole (the counterplay-term's turf).

  overnight_runner.sh pyrun diagnostics/triage_slice.py diagnostics/collapse_classified.csv --category overpush MAX_DEPTH=16 PRESET=LONG_FORMAT
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
# pyrun env forwarding
for _kv in [a for a in sys.argv[1:] if "=" in a and not a.startswith("-") and not a.endswith(".csv")]:
    _k, _v = _kv.split("=", 1); os.environ[_k] = _v
argv = [a for a in sys.argv[1:] if not ("=" in a and not a.startswith("-") and not a.endswith(".csv"))]

CAT = "overpush"
if "--category" in argv: i = argv.index("--category"); CAT = argv[i+1]; del argv[i:i+2]
csv_path = next((a for a in argv if a.endswith(".csv")), os.path.join(THIS, "collapse_classified.csv"))

from tactical_test import run_one
from triage_collapses import lowprune_move


def main():
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("category") == CAT]
    print(f"[triage_slice] category={CAT}  n={len(rows)}  MAX_DEPTH={os.environ.get('MAX_DEPTH','?')}")
    c = {"HORIZON": 0, "PRUNING": 0, "EVAL": 0, "err": 0}; eval_sf = 0
    for r in rows:
        fen = r["fen"]; blunder = r["our_move"]; sf_best = r.get("sf_best", "")
        try:
            deep = run_one(fen, set())["uci"]
        except Exception:
            c["err"] += 1; continue
        if deep != blunder:
            c["HORIZON"] += 1
        else:
            lp = lowprune_move(fen) or ""
            if lp and lp != blunder:
                c["PRUNING"] += 1
            else:
                c["EVAL"] += 1
                if deep == sf_best: eval_sf += 1     # deep plays SF's move yet labelled blunder (shouldn't happen in overpush)
    tot = max(1, c["HORIZON"] + c["PRUNING"] + c["EVAL"])
    print(f"    HORIZON (depth fixes): {c['HORIZON']:>3} ({100*c['HORIZON']/tot:3.0f}%)")
    print(f"    PRUNING (low-prune fixes): {c['PRUNING']:>3} ({100*c['PRUNING']/tot:3.0f}%)")
    print(f"    EVAL    (deep+lowprune still blunders): {c['EVAL']:>3} ({100*c['EVAL']/tot:3.0f}%)  <- the eval-term constituency")
    print(f"    errors: {c['err']}")
    print("  READ: high EVAL => search won't fix it, the counterplay eval term is the lane. High HORIZON/")
    print("        PRUNING => a search fix (depth/pruning/qsearch) is cheaper than the eval term.")


if __name__ == "__main__":
    main()
