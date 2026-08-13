# -*- coding: utf-8 -*-
"""Is the KS over-read concentrated in ENDGAMES (a phase-gate fix) or spread across phases (deeper over-attack)?
Runs base (ship) vs a candidate KS config, and buckets the footprint move-regret drift by the regret set's
phase_bucket (opening/midgame/endgame). A big positive delta ONLY in endgame => our phase taper under-gates
(flank/breadth fires on endgame activity) => SF/Ethereal-style phase gate likely fixes it. Positive across all
phases => the over-attack is not just a phase artifact.

  pyrun diagnostics/_ks_phase_split.py [SET=ks_sets/game_regret_set_v2.csv] [DEPTH=7] [JOBS=4]
"""
import os, sys, csv, math, subprocess
from collections import defaultdict

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
    from tactical_test import run_one
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    rows = list(csv.DictReader(open(SET, newline="")))[si::sn]
    out = open(os.environ["OUT"], "w", newline=""); w = csv.writer(out)
    for r in rows:
        fen = r.get("fen")
        try:
            best_cp = float(r["best_cp"]); mm = {}
            for pair in (r.get("moves") or "").split(";"):
                if ":" in pair:
                    u, c = pair.rsplit(":", 1); mm[u] = float(c)
        except Exception:
            continue
        if not mm:
            continue
        try:
            our = run_one(fen, set())["uci"]
        except Exception:
            continue
        stm_white = (fen.split()[1] == 'w')
        oc = mm.get(our)
        if oc is None:
            worst = min(mm.values()) if stm_white else max(mm.values()); oc = (worst - MISS) if stm_white else (worst + MISS)
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(oc) if stm_white else (100.0 - _winpct(oc))
        w.writerow([fen, our, "%.4f" % max(0.0, bw - ow), r.get("phase_bucket", "?")])
    out.close(); sys.exit(0)

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS); PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set_v2.csv")
if not os.path.isabs(SET): SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7"); JOBS = int(os.environ.get("JOBS", "4"))
BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40, "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


BASE = cfg(KS_DEFAWARE_MODE=1)
CAND = cfg(KS_DEFAWARE_MODE=1, KS_SQC_MODE=1, KS_PIN_MODE=1, KS_WEAK_VAL_MODE=1, KS_FLANK_MODE=2, KS_FLOOR=15, KS_KNEE=40, KS_DIVISOR=8)


def collect(config, tag):
    env = dict(os.environ, SET=SET, DEPTH=DEPTH)
    ka = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        outp = "/tmp/_ps_%s_%d.csv" % (tag, i)
        procs.append((subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + ka,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=ENGINE,
                     env=dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS), OUT=outp)), outp))
    d = {}
    for p, outp in procs:
        p.communicate()
        for row in csv.reader(open(outp, newline="")):
            if len(row) == 4:
                d[row[0]] = (row[1], float(row[2]), row[3])
    return d


base = collect(BASE, "base"); cand = collect(CAND, "cand")
shared = [f for f in base if f in cand]
changed = [f for f in shared if base[f][0] != cand[f][0]]
byph = defaultdict(lambda: [0, 0.0, 0.0])   # phase -> [n, sum_base, sum_cand]
allph = defaultdict(int)
for f in shared:
    allph[base[f][2]] += 1
for f in changed:
    ph = base[f][2]; byph[ph][0] += 1; byph[ph][1] += base[f][1]; byph[ph][2] += cand[f][1]

print("PHASE SPLIT of the KS over-read  (base=ship vs compound OURS)  set=%s\n" % os.path.basename(SET))
print("  %-10s %8s %8s   %9s %9s %9s" % ("phase", "positions", "changed", "reg_base", "reg_cand", "delta"))
for ph in ("opening", "midgame", "endgame"):
    n, sb, sc = byph[ph]
    if n:
        print("  %-10s %8d %8d   %9.4f %9.4f %+9.4f" % (ph, allph[ph], n, sb/n, sc/n, (sc-sb)/n))
tot_n = sum(byph[p][0] for p in byph); tot_b = sum(byph[p][1] for p in byph); tot_c = sum(byph[p][2] for p in byph)
print("  %-10s %8d %8d   %9.4f %9.4f %+9.4f" % ("ALL", len(shared), tot_n, tot_b/max(1,tot_n), tot_c/max(1,tot_n), (tot_c-tot_b)/max(1,tot_n)))
print("\n  read: if the positive delta is CONCENTRATED in endgame, it's a phase-gate problem (our taper under-gates\n"
      "  R/Q endgames that SF/Ethereal phase out). If it's positive across ALL phases, the over-attack is deeper.", flush=True)
