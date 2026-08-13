# -*- coding: utf-8 -*-
"""WHY does the coupled config drift? For a BASE vs CAND config, find the positions where the candidate CHANGES
our D7 move and makes it WORSE per SF18 (the drift), and dump — for the worst ones — the FEN, both moves, SF18's
best, the regret jump, AND the raw KS attack-units per king (det_ks_units_w/b) under BOTH configs. So we can see
WHETHER the drift is the compounding curve inflating units on proximity-only kings, the safe-check weight
overshooting, etc. Also summarizes: on drifted-WORSE vs drifted-BETTER positions, how much did units inflate?

  pyrun diagnostics/_ks_drift_analysis.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [JOBS=4] [TOPN=20]

CAND defaults to coupled config A. Deterministic. Read against the footprint sign.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        k, v = _a.split('=', 1); os.environ[k] = v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7'); os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from ChessAI import ChessAI
    from tactical_test import run_one
    ai = ChessAI(None, None, chess.Board(), True)
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    rows = list(csv.DictReader(open(SET, newline="")))
    import random as _r
    _r.Random(1234).shuffle(rows)
    rows = rows[si::sn]
    out = open(os.environ["OUT"], "w", newline=""); w = csv.writer(out)
    for r in rows:
        fen = r.get("fen")
        try:
            best_cp = float(r["best_cp"]); best_uci = r.get("best_uci")
            mm = {}
            for pair in (r.get("moves") or "").split(";"):
                if ":" in pair:
                    u, c = pair.rsplit(":", 1); mm[u] = float(c)
        except Exception:
            continue
        if not mm:
            continue
        try:
            our = run_one(fen, set())["uci"]
            bd = ai.ev_breakdown(chess.Board(fen))
            uw = bd.get("det_ks_units_w", 0); ub = bd.get("det_ks_units_b", 0)
        except Exception:
            continue
        stm_white = (fen.split()[1] == 'w')
        our_cp = mm.get(our)
        if our_cp is None:
            worst = min(mm.values()) if stm_white else max(mm.values())
            our_cp = (worst - MISS) if stm_white else (worst + MISS)
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(our_cp) if stm_white else (100.0 - _winpct(our_cp))
        reg = max(0.0, bw - ow)
        w.writerow([fen, our, "%.4f" % reg, uw, ub, best_uci])
    out.close(); sys.exit(0)

# ---------------- DRIVER ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7"); JOBS = int(os.environ.get("JOBS", "4")); TOPN = int(os.environ.get("TOPN", "20"))

BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
          "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


BASE = cfg(KS_DEFAWARE_MODE=1)
CAND = cfg(KS_DEFAWARE_MODE=1, KS_SQC_MODE=1, KS_PIN_MODE=1, KS_WEAK_VAL_MODE=1,
           KS_FLANK_MODE=2, KS_FLOOR=15, KS_KNEE=40, KS_DIVISOR=8)   # compound OURS (the current candidate)


def collect(config, tag):
    env = dict(os.environ, SET=SET, DEPTH=DEPTH)
    ka = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        outp = "/tmp/_drift_%s_%d.csv" % (tag, i)
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS), OUT=outp)
        procs.append((subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + ka,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=ENGINE, env=e), outp))
    d = {}
    for p, outp in procs:
        p.communicate()
        try:
            for row in csv.reader(open(outp, newline="")):
                if len(row) == 6:
                    d[row[0]] = {"mv": row[1], "reg": float(row[2]), "uw": int(row[3]), "ub": int(row[4]), "best": row[5]}
        except Exception:
            pass
    return d


base = collect(BASE, "base"); cand = collect(CAND, "cand")
fens = [f for f in base if f in cand]
drift = [f for f in fens if base[f]["mv"] != cand[f]["mv"]]
worse = [f for f in drift if cand[f]["reg"] > base[f]["reg"] + 0.01]
better = [f for f in drift if cand[f]["reg"] < base[f]["reg"] - 0.01]


def uinfl(fs):  # mean per-position max-king unit change base->cand
    if not fs: return 0.0
    return sum(max(cand[f]["uw"]-base[f]["uw"], cand[f]["ub"]-base[f]["ub"]) for f in fs) / len(fs)


print("DRIFT ANALYSIS  cand=A(redistribute+compound)  set=%s  positions=%d" % (os.path.basename(SET), len(fens)))
print("  move changed %d | worse %d | better %d   (net worse-better = %+d)" % (len(drift), len(worse), len(better), len(worse)-len(better)))
print("  mean max-king unit change base->cand:  on WORSE %+.1f   on BETTER %+.1f" % (uinfl(worse), uinfl(better)))
print("\n  worst %d drift positions (cand made our move worse):" % TOPN)
print("  %-46s %-6s %-6s %-6s  %-9s %-9s" % ("fen", "base", "cand", "sf18", "reg b->c", "unitsWB b->c"))
for f in sorted(worse, key=lambda x: cand[x]["reg"] - base[x]["reg"], reverse=True)[:TOPN]:
    b, c = base[f], cand[f]
    print("  %-46s %-6s %-6s %-6s  %4.1f->%4.1f  %d/%d->%d/%d"
          % (f[:46], b["mv"], c["mv"], b["best"], b["reg"], c["reg"], b["uw"], b["ub"], c["uw"], c["ub"]))
print("\n  read: if 'unit change on WORSE' >> 'on BETTER', the drift is the curve INFLATING units where it\n"
      "  shouldn't (proximity-only kings) -> the compounding is amplifying noise, not signal. Eyeball the FENs.", flush=True)
