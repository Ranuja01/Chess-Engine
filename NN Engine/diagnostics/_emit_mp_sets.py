# -*- coding: utf-8 -*-
"""Emit TARGET and HOLDOUT FEN csvs in the `fen_start` schema `move_proxy`/`fen_vs_sf --csv` expects.

Why move-based and not cp-based: a two-set mean-|gap|-vs-SF11 screen was built first and FAILED ITS OWN
CONTROL -- `PV_BOOST_MAG=0`, a uniform shrink we have independent reason to distrust, scored BETTER on it
than any real candidate (target −12.5%, holdout −20.0%). Our eval is systematically larger than SF11's,
so ANY shrink improves a cp-distance metric whether or not it improves play. Centipawn distance cannot
validate an eval change; SF-best MOVE MATCH can, because shrinking an eval does not move its argmax
toward SF's choice.

  TARGET  = positional collapse decision-FENs (where the defect lives)
  HOLDOUT = general wide-corpus positions, target FENs excluded (where regressions hide)

  pyrun diagnostics/_emit_mp_sets.py [NT=120] [NH=300] [FAMILY=vssf_2400] [CLASS=positional]
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
NT = int(os.environ.get("NT", "120"))
NH = int(os.environ.get("NH", "300"))
FAMILY = os.environ.get("FAMILY", "vssf_2400")
CLASS = os.environ.get("CLASS", "positional")

seen, target = set(), []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv"))):
    if r.get("family") != FAMILY or r.get("ks_class") != CLASS:
        continue
    f = (r.get("decision_fen") or "").strip()
    if f and f not in seen:
        seen.add(f); target.append(f)

hold = []
for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))):
    f = (r.get("fen") or "").strip()
    # Excluding target FENs is the whole point: overlap would mask the very trade we are testing for.
    if f and f not in seen:
        hold.append(f)

for name, fens, n in (("_mp_target.csv", target, NT), ("_mp_holdout.csv", hold, NH)):
    p = os.path.join(THIS, name)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen_start", "game", "drop"])
        for i, f in enumerate(fens[:n]):
            # MUST be `game_<digits>`: move_proxy's row parser keys on the regex group (game_\d+) and
            # silently DISCARDS every line whose tag does not match -- it then reports "no parsed rows"
            # only at the very end, after running both engine passes in full.
            w.writerow([f, "game_%03d" % i, 0])
    print("  %-18s %d FENs -> %s" % (name, min(n, len(fens)), p))
