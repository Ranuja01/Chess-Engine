# -*- coding: utf-8 -*-
"""Eval-vs-SEARCH triage for the KS-attack collapse class: for each base ks_attack collapse decision-FEN,
does OUR DEEP search still fail to see the danger (EVAL-blind -> KS/eval work COULD help), or does deep search
agree with SF18 that we are worse (SEARCH/depth-bound -> the blitz game just didn't reach depth; static-KS
tuning CANNOT fix it)? This answers why strengthening KS didn't move the ks_attack collapse count in games.

Per FEN (from the collapsing side's POV): our_static, our_deep (run_one), SF18. Classify:
  EVAL-blind  : our_deep is >= EVAL_GAP cp MORE optimistic than SF18 (we don't see the danger even deep)
  SEARCH-bound: our_deep roughly agrees with SF18 that we're worse (we CAN see it deep; blitz didn't)
  not-danger  : SF18 doesn't actually show us worse (classifier false-positive)

  MAX_DEPTH=13 PRESET=LONG_FORMAT pyrun diagnostics/ks_collapse_triage.py [EVAL_GAP=200] [N=30]
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
os.environ.setdefault('MAX_DEPTH', '13'); os.environ.setdefault('PRESET', 'LONG_FORMAT')
os.environ.setdefault('USE_OPENING_BOOK', '0')
EVAL_GAP = int(os.environ.get('EVAL_GAP', '200'))
NMAX = int(os.environ.get('N', '30'))
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
import chess, chess.engine
from tactical_test import run_one
from arbiter import find_stockfish

CSVP = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
rows = [r for r in csv.DictReader(open(CSVP))
        if r.get("family") == "base" and r.get("ks_class") == "ks_attack"
        and r.get("seed") in ("0", "1") and r.get("decision_fen", "")]
# dedup by fen
seen = set(); fens = []
for r in rows:
    f = r["decision_fen"]
    if f not in seen:
        seen.add(f); fens.append((f, r.get("our_color", "?")))
fens = fens[:NMAX]

sf = chess.engine.SimpleEngine.popen_uci(find_stockfish())
def sf18_wpov(fen):
    s = sf.analyse(chess.Board(fen), chess.engine.Limit(depth=18))["score"].white()
    return 9999 if s.is_mate() and s.mate() > 0 else (-9999 if s.is_mate() else s.score())

def our_pov(wpov, color):   # convert white-POV cp to the collapsing side's POV
    return wpov if color == "white" else -wpov

print("KS-attack collapse triage (base seeds 0,1) — EVAL_GAP=%d, depth=%s\n" % (EVAL_GAP, os.environ["MAX_DEPTH"]))
print("%-4s %-6s %8s %8s %8s  %-11s" % ("#", "side", "ourStat", "ourDeep", "SF18", "verdict"))
tally = {"EVAL-blind": 0, "SEARCH-bound": 0, "not-danger": 0}
for i, (fen, color) in enumerate(fens):
    b = chess.Board(fen)
    ai_static = None
    r = run_one(fen, set())
    # our_deep white-POV: run_one eval is side-to-move relative
    ev = r.get("eval")
    if ev is None:
        continue
    deep_wpov = ev if b.turn == chess.WHITE else -ev
    sfw = sf18_wpov(fen)
    # static via a fresh breakdown is optional; skip to keep one engine. Use run_one depth1? just show deep.
    us_deep = our_pov(deep_wpov, color); us_sf = our_pov(sfw, color)
    if us_sf > -150:
        v = "not-danger"           # SF doesn't show the collapsing side clearly worse
    elif us_deep - us_sf >= EVAL_GAP:
        v = "EVAL-blind"           # we're much more optimistic than SF even deep
    else:
        v = "SEARCH-bound"         # our deep agrees we're worse; blitz just didn't reach it
    tally[v] += 1
    print("%-4d %-6s %8s %8d %8d  %-11s  %s" % (i, color, "", us_deep, us_sf, v, fen))
sf.quit()
print("\nTALLY:", tally)
print("(EVAL-blind => KS/eval work CAN move this collapse; SEARCH-bound => depth-limited, static-KS can't.)")
