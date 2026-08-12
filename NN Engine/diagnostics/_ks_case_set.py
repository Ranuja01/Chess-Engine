# -*- coding: utf-8 -*-
"""Build the KS diagnostic CASE SET: bucket each collapse position by WHY we fail, before touching a knob.

Twelve KS iterations all compared SCORES and all measured a global magnitude. This separates the failure
modes first, so the next candidate is aimed at a diagnosed cause rather than at the term's name:

  SEARCH BUCKETS (our move at shallow vs deep, against SF18's best)
    A static-wrong / search-RECOVERS  -> an eval-only defect; safe to fix in eval
    B static-right / search-BREAKS    -> NOT an eval problem; do not spend KS work here
    C wrong at BOTH depths            -> eval error deep enough to survive search = highest value
    D right at both                   -> fine

  KS LEAN (who do we charge, vs who SF11 charges)  -- the per-king ledger question
    our attack units per king come from the breakdown's det_ks_units_w / det_ks_units_b.
    UNDER  = we charge the collapsing side's OWN king far less than SF11's King-safety term implies
             (the owner's repeated observation: we miss the counterbalance to material)
    OVER   = we charge the ENEMY king a lot while SF11 does not (phantom attack by us)
    The 07-23 note found both in one position: "we charge the swarmed king 33 units, our own airy king 4;
    SF11 charges -5.30/-5.06 BOTH". One lopsided ledger produces BOTH symptoms, which is why every global
    magnitude sweep netted to nothing.

⚠️ MAX_DEPTH latches at engine init, so each depth runs in its OWN process.
⚠️ SF11 cannot evaluate a position whose side to move is in check (returns None) -> marked n/a, not dropped.

  pyrun diagnostics/_ks_case_set.py [SET=diagnostics/_mp_target.csv] [N=30] [D1=10] [D2=18]
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

SET = os.environ.get("SET", os.path.join(THIS, "_mp_target.csv"))
if not os.path.isabs(SET):
    SET = os.path.join(ENGINE, SET)
N = int(os.environ.get("N", "30"))
D1 = int(os.environ.get("D1", "10"))
D2 = int(os.environ.get("D2", "18"))


def fens():
    return [r["fen_start"].strip() for r in csv.DictReader(open(SET)) if r.get("fen_start", "").strip()][:N]


if os.environ.get("WORKER") == "1":
    import time
    from tactical_test import run_one
    rows = fens()
    with open(os.environ["OUT"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "move"])
        for i, fen in enumerate(rows, 1):
            t0 = time.time()
            try:
                r = run_one(fen, set())
                w.writerow([fen, r["uci"]])
                print("    [%2d/%2d] d%s %5.1fs %s" % (i, len(rows), os.environ.get("MAX_DEPTH"),
                                                       time.time() - t0, r["uci"]), flush=True)
            except Exception as e:
                print("    [%2d/%2d] SKIP %s" % (i, len(rows), type(e).__name__), flush=True)
    sys.exit(0)

outs = {}
for d in (D1, D2):
    out = os.path.join(THIS, "_cs_d%d.csv" % d)
    outs[d] = out
    print("\n  === our move at depth %d ===" % d, flush=True)
    env = dict(os.environ, PRESET="LONG_FORMAT", MAX_DEPTH=str(d), USE_OPENING_BOOK="0", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "WORKER=1", "OUT=" + out,
                    "SET=" + SET, "N=%d" % N], cwd=ENGINE, check=True, env=env)

import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
from arbiter import find_stockfish

ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())


def load(p):
    return {r["fen"]: r["move"] for r in csv.DictReader(open(p, newline=""))}


m1, m2 = load(outs[D1]), load(outs[D2])
print("\n" + "=" * 116)
print("  %-4s %-6s %-6s %-6s | %7s %7s %7s | %6s %6s %7s | %s"
      % ("buck", "d%d" % D1, "d%d" % D2, "sfbest", "ours", "sf11", "sf18", "ksW", "ksB", "sf11KS", "lean"))
print("=" * 116)
buckets = {"A": [], "B": [], "C": [], "D": [], "?": []}
for fen in fens():
    if fen not in m1 or fen not in m2:
        continue
    b = chess.Board(fen)
    try:
        info = sf18.analyse(b, chess.engine.Limit(depth=18))
        sc = info["score"].white()
        s18 = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
        sfbest = info["pv"][0].uci()
        bd = ai.ev_breakdown(b)
        sf_total, sf_terms = sf11.eval(fen)
    except Exception:
        continue
    ok1, ok2 = m1[fen] == sfbest, m2[fen] == sfbest
    bucket = "D" if (ok1 and ok2) else "A" if (not ok1 and ok2) else "B" if (ok1 and not ok2) else "C"
    ours = -bd.get("total", 0) / 1000.0
    ksw, ksb = bd.get("det_ks_units_w", 0), bd.get("det_ks_units_b", 0)
    our_ks = -bd.get("king_safety", 0) / 1000.0
    sf_ks = sf_terms.get("King safety", 0.0) if sf_total is not None else None
    # Lean: SF11 sees king danger that we do not (UNDER) vs we assign danger SF11 does not (OVER).
    if sf_ks is None:
        lean = "n/a(check)"
    elif abs(sf_ks) - abs(our_ks) > 0.75:
        lean = "UNDER  we miss it"
    elif abs(our_ks) - abs(sf_ks) > 0.75:
        lean = "OVER   we invent it"
    else:
        lean = "match"
    buckets[bucket].append(fen)
    print("  %-4s %-6s %-6s %-6s | %+7.2f %7s %+7.2f | %6d %6d %7s | %s"
          % (bucket, m1[fen], m2[fen], sfbest, ours,
             ("%+.2f" % sf_total) if sf_total is not None else "n/a", s18,
             ksw, ksb, ("%+.2f" % sf_ks) if sf_ks is not None else "n/a", lean))
    print("       %s" % fen)

sf18.quit(); sf11.close()
print("\n  A static-wrong/search-RECOVERS %d   B static-right/search-BREAKS %d   C wrong at BOTH %d   D fine %d"
      % (len(buckets["A"]), len(buckets["B"]), len(buckets["C"]), len(buckets["D"])))
print("  ⇒ C is the eval lane. B is search's problem. A is fixable in eval but search already saves it.\n")
