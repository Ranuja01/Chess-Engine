# -*- coding: utf-8 -*-
"""Is capgains a STATIC SUBSTITUTE for a qsearch we don't fully trust? (owner's hypothesis)

Claim: SF needs no capgain term because its qsearch resolves captures dynamically and lands in a quiet
position where material is just material. We approximate that resolution STATICALLY (approximate_capture_gains)
and inherit its errors. If true, then on capgain-heavy positions:
  - OUR shallow search WITH capgains OFF should be WRONG (misses the tactic) and CONVERGE to truth as depth
    grows (qsearch/search eventually resolves it).
  - OUR capgains ON is a static patch: it may help at shallow depth but if it OVER-reads (books a capture
    that doesn't actually win) it will keep the eval too high even as depth grows -- a static lie the search
    cannot un-see.
  - SF's shallow eval should already be near its own deep truth (fast, accurate qsearch).

So per position we print, in White-POV pawns, the eval at rising depths:
  ours d1/d4/d8  x  capgains {ON, OFF}     vs    SF d1/d4/d12
and the gap to SF-d16 TRUTH. The tell:
  * capgOFF gap SHRINKS with depth while capgON stays high  => capgains is an OVER-READING static crutch here.
  * capgOFF stays large at all depths                       => our search genuinely can't resolve it; capgains
                                                                is a real (if imperfect) substitute.
  * SF-d1 already near truth while ours needs depth          => SF's qsearch is the advantage (owner's claim).

⚠️ Depth latches at engine init -> one worker process per (depth, capgains) combo.
⚠️ Convention: our eval/search score is absolute Black-positive millipawns; White-POV pawns = -v/1000.
   The script prints a d1-search-vs-static sanity line; if they disagree in SIGN the convention is wrong.

  pyrun diagnostics/_qsearch_hypothesis.py [N=120] [CAPG_MIN=1.0] [KMAX=8]
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

N = int(os.environ.get("N", "120"))
CAPG_MIN = float(os.environ.get("CAPG_MIN", "1.0"))
KMAX = int(os.environ.get("KMAX", "8"))
NEAR, MISS = 1.25, 2.0
OUR_DEPTHS = [1, 4, KMAX]


def select_positions():
    """Learnable positions (SF11 tracks SF18, we miss) that are CAPGAIN-DRIVEN (|capture_gains| >= CAPG_MIN)."""
    import chess, chess.engine
    from ChessAI import ChessAI
    from eval_vs_sf11 import SF11Eval, SF11
    from arbiter import find_stockfish
    ai = ChessAI(None, None, chess.Board(), True)
    sf11 = SF11Eval(SF11); sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    seen, out = set(), []
    for r in csv.DictReader(open(os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv"))):
        if r.get("family") != "vssf_2400" or r.get("ks_class") != "positional":
            continue
        f = (r.get("decision_fen") or "").strip()
        if not f or f in seen:
            continue
        seen.add(f)
        if len(out) >= 10:
            continue
        try:
            b = chess.Board(f)
            bd = ai.ev_breakdown(b)
            sf_total, _ = sf11.eval(f)
            sc = sf18.analyse(b, chess.engine.Limit(depth=16))["score"].white()
            s18 = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
        except Exception:
            continue
        if sf_total is None or abs(sf_total - s18) > NEAR or abs(-bd.get("total", 0) / 1000.0 - s18) < MISS:
            continue
        if abs(-bd.get("capture_gains", 0) / 1000.0) >= CAPG_MIN:
            out.append((f, s18, -bd.get("total", 0) / 1000.0, -bd.get("capture_gains", 0) / 1000.0))
    sf18.quit(); sf11.close()
    return out


if os.environ.get("WORKER") == "1":
    import chess
    from tactical_test import run_one
    fens = os.environ["FENS"].split("|")
    with open(os.environ["OUT"], "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "raw"])
        for f in fens:
            try:
                r = run_one(f, set())
                w.writerow([f, float(r["eval"])])   # raw; parent converts by side to move
            except Exception:
                w.writerow([f, ""])
    sys.exit(0)


def to_wpov(raw, fen):
    """run_one's score is side-to-move relative (+ = good for mover). Convert to White-POV pawns.
    Values near the mate sentinel (|raw| > 900000) are unresolved leaves, not real evals -> None."""
    if raw is None or raw == "":
        return None
    v = float(raw)
    if abs(v) > 900000:
        return None
    v = v / 1000.0
    return v if chess.Board(fen).turn == chess.WHITE else -v

import chess
sel = select_positions()
if not sel:
    sys.exit("no capgain-driven learnable positions found")
fens = [s[0] for s in sel]
truth = {s[0]: s[1] for s in sel}
static_on = {s[0]: s[2] for s in sel}
capg = {s[0]: s[3] for s in sel}
print("\n  %d capgain-driven learnable positions\n" % len(sel), flush=True)

# One worker per (depth, capgains) combo.
results = {}   # (depth, capg_on) -> {fen: score}
for d in OUR_DEPTHS:
    for on in (True, False):
        out = os.path.join(THIS, "_qs_d%d_%s.csv" % (d, "on" if on else "off"))
        env = dict(os.environ, WORKER="1", OUT=out, FENS="|".join(fens),
                   PRESET="LONG_FORMAT", MAX_DEPTH=str(d), USE_OPENING_BOOK="0", OMP_NUM_THREADS="1",
                   SCALE_CAPTURE_GAINS=("100" if on else "0"))
        print("    our engine  d%d  capgains %s ..." % (d, "ON" if on else "OFF"), flush=True)
        subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], cwd=ENGINE, check=True, env=env)
        results[(d, on)] = {r["fen"]: to_wpov(r["raw"], r["fen"])
                            for r in csv.DictReader(open(out, newline=""))}

# SF at rising depth.
import chess.engine
from arbiter import find_stockfish
sf = chess.engine.SimpleEngine.popen_uci(find_stockfish())
sf_at = {}
for d in (1, 4, 12):
    sf_at[d] = {}
    for f in fens:
        sc = sf.analyse(chess.Board(f), chess.engine.Limit(depth=d))["score"].white()
        sf_at[d][f] = 99.0 if (sc.is_mate() and sc.mate() > 0) else (-99.0 if sc.is_mate() else sc.score() / 100.0)
sf.quit()

print("\n" + "=" * 108)
print("  QSEARCH HYPOTHESIS — eval convergence to SF-d16 truth (White-POV pawns).  capg=our capture_gains term")
print("=" * 108)
d1, d2, d3 = OUR_DEPTHS
for f in fens:
    t = truth[f]
    print("\n  truth %+.2f   static(capgON) %+.2f   capg-term %+.2f" % (t, static_on[f], capg[f]))
    print("    %s" % f)

    def g(d, on):
        v = results[(d, on)].get(f)
        return "  n/a" if v is None else "%+.2f" % v
    print("    ours capgON   d%d %s  d%d %s  d%d %s" % (d1, g(d1, True), d2, g(d2, True), d3, g(d3, True)))
    print("    ours capgOFF  d%d %s  d%d %s  d%d %s" % (d1, g(d1, False), d2, g(d2, False), d3, g(d3, False)))
    print("    SF            d1 %+.2f  d4 %+.2f  d12 %+.2f" % (sf_at[1][f], sf_at[4][f], sf_at[12][f]))

# Sanity: d1 search (capgON) should sit near the static(capgON) value if the POV convention is right.
print("\n  --- POV sanity (d1-search capgON vs static capgON; same sign expected) ---")
for f in fens[:3]:
    v = results[(d1, True)].get(f)
    print("    static %+.2f   d1search %s   %s"
          % (static_on[f], ("n/a" if v is None else "%+.2f" % v),
             "OK" if (v is not None and (v > 0) == (static_on[f] > 0)) else "SIGN MISMATCH?"))

# Aggregate: mean |gap to truth| for capgON vs capgOFF at each depth.
print("\n  --- mean |gap to truth| ---")
for d in OUR_DEPTHS:
    for on in (True, False):
        vals = [abs(results[(d, on)][f] - truth[f]) for f in fens if results[(d, on)].get(f) is not None]
        print("    ours d%-2d capg%-3s  %.2f" % (d, "ON" if on else "OFF", sum(vals) / max(1, len(vals))))
for d in (1, 4, 12):
    vals = [abs(sf_at[d][f] - truth[f]) for f in fens]
    print("    SF   d%-2d           %.2f" % (d, sum(vals) / max(1, len(vals))))
print()
