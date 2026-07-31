# -*- coding: utf-8 -*-
"""Concrete reach on the 64 other_collapses: how many positions do PIN and TEMPO each change (>=0.5 pawns),
and of those how many move our-POV eval TOWARD SF11-static (reduce over-confidence). Fresh child process per
config (C++ Config read once at init). NOTE: SF11-static is contaminated on attack positions, so 'toward SF11'
is only a rough directional proxy -- the raw 'reached' count is the reliable number."""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
CONF = {"base": ("0", "0"), "pin": ("1", "0"), "pintempo": ("1", "1")}

if len(sys.argv) > 1:   # CHILD: one config, print "idx,ourpov_total[,sf11]" per fen
    name = sys.argv[1]
    os.environ["ENABLE_CAPG_PIN"], os.environ["ENABLE_CAPG_TEMPO"] = CONF[name]
    sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    sf = None
    if name == "base":
        from eval_vs_sf11 import SF11Eval, SF11
        sf = SF11Eval(SF11)
    fens = [ln.rstrip("\n").split("\t",1)[-1].strip() for ln in open(os.path.join(THIS,"ks_sets","other_collapses.txt")) if ln.strip()]
    for i, fen in enumerate(fens):
        b = chess.Board(fen); pov = 1.0 if b.turn else -1.0
        tot = (-ai.ev_breakdown(b).get("total", 0.0) / 1000.0) * pov
        if sf is not None:
            s = sf.eval(fen)[1].get("Total", 0.0) * pov
            print("%d,%.3f,%.3f" % (i, tot, s))
        else:
            print("%d,%.3f" % (i, tot))
    if sf: sf.close()
    sys.exit(0)

# PARENT: run 3 children, aggregate
import subprocess
def run(name):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), name], capture_output=True, text=True)
    out = {}
    for ln in r.stdout.splitlines():
        p = ln.split(",")
        if len(p) >= 2 and p[0].isdigit():
            out[int(p[0])] = tuple(float(x) for x in p[1:])
    return out
base = run("base"); pin = run("pin"); pt = run("pintempo")
n = len(base)
def reached(a, b, thr=0.5): return [i for i in a if abs(a[i][0] - b[i][0]) >= thr]
pin_touch = reached(base, pin); tempo_touch = reached(pin, pt)
def toward_sf(i, before, after):   # does 'after' reduce |our - sf11| vs 'before'?
    sf = base[i][1]
    return abs(after[i][0] - sf) < abs(before[i][0] - sf) - 0.1
print("n=%d" % n)
print("PIN reaches %d/%d positions (|Δtotal|>=0.5); of those %d move toward SF11-static" % (
    len(pin_touch), n, sum(1 for i in pin_touch if toward_sf(i, base, pin))))
print("TEMPO reaches %d/%d additional (on top of pin); of those %d move toward SF11-static" % (
    len(tempo_touch), n, sum(1 for i in tempo_touch if toward_sf(i, pin, pt))))
print("mean our-POV total: base=%.2f  pin=%.2f  pin+tempo=%.2f  (SF11 mean=%.2f)" % (
    sum(base[i][0] for i in base)/n, sum(pin[i][0] for i in pin)/n, sum(pt[i][0] for i in pt)/n,
    sum(base[i][1] for i in base)/n))
