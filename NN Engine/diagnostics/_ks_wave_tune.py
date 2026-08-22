# -*- coding: utf-8 -*-
"""WAVE (curriculum) regret tuner. Instead of tuning mean regret on the whole set (which lets already-correct
positions drown the collapses), tune HARDEST-first: bucket positions by STAKES = win%(SF-best) - win%(SF-2nd)
from the side-to-move POV (config-independent = how blunderous it is to miss the one best move), then run a
curriculum of nested waves (stakes >= T, T decreasing). Each wave coordinate-descends the KS knobs to minimise
that wave's TUNE-split regret, carries the winning config into the next (wider) wave, and gates every accepted
move on BOTH the wave's HELD split AND the FULL-set HELD regret (the general-stability guardrail). Final wave =
all positions, so the config ends anchored to the collapses but validated on everything.

Reuses the same win%-regret math + seeded(1234) shuffle/split as _regret_tune_broad / _ks_regret_score, so the
numbers compare. FIXED depth 7 (deterministic PROXY) -- winner still needs STS/symmetry + games.

  pyrun diagnostics/_ks_wave_tune.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [JOBS=4] [SPLIT_FRAC=0.7]
        [WAVES=50,40,30,20,10,0] [PASSES=1] [HELD_TOL=0.02] [GEN_TOL=0.01] [MAXN=0]
        [GRID_ONLY=k1,k2] [SMOKE=1 -> just print wave sizes and exit]
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def _stakes(fen, mm, best_uci):
    """win% gap best-vs-2nd, side-to-move POV. Config-independent severity."""
    stm_white = (fen.split()[1] == 'w')
    best_cp = mm.get(best_uci)
    if best_cp is None:
        best_cp = max(mm.values()) if stm_white else min(mm.values())
    others = [c for u, c in mm.items() if u != best_uci]
    if not others:
        return 0.0
    second = max(others) if stm_white else min(others)
    bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
    sw = _winpct(second) if stm_white else (100.0 - _winpct(second))
    return max(0.0, bw - sw)


def _load(setpath):
    rows = []
    for r in csv.DictReader(open(setpath, newline="")):
        mm = {}
        for pair in (r.get("moves") or "").split(";"):
            if ":" in pair:
                u, c = pair.rsplit(":", 1)
                try:
                    mm[u] = float(c)
                except ValueError:
                    pass
        if not mm:
            continue
        try:
            r["_stakes"] = _stakes(r["fen"], mm, r.get("best_uci"))
        except Exception:
            continue
        rows.append(r)
    import random as _r
    _r.Random(1234).shuffle(rows)
    return rows


# ---------------- WORKER: config + slice + split + STAKES_MIN -> summed regret over that sub-slice ----------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PRESET'] = 'LONG_FORMAT'
    os.environ['MAX_DEPTH'] = os.environ.get('DEPTH', '7')
    os.environ['USE_OPENING_BOOK'] = '0'
    THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS); sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
    import chess
    from tactical_test import run_one
    SET = os.environ["SET"]; MISS = float(os.environ.get("MISS", "30"))
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))
    frac = float(os.environ.get("SPLIT_FRAC", "0.7"))
    which = os.environ.get("SPLIT", "tune")
    smin = float(os.environ.get("STAKES_MIN", "0"))
    MAXN = int(os.environ.get("MAXN", "0"))
    rows = _load(SET)
    if MAXN:
        rows = rows[:MAXN]
    cut = int(frac * len(rows))
    rows = rows[:cut] if which == "tune" else (rows[cut:] if which == "held" else rows)
    rows = [r for r in rows if r["_stakes"] >= smin][si::sn]
    tot_reg = 0.0; n = 0; match = 0; miss = 0
    for r in rows:
        fen = r["fen"]; best_uci = r.get("best_uci")
        best_cp = float(r["best_cp"])
        mm = {}
        for pair in (r.get("moves") or "").split(";"):
            if ":" in pair:
                u, c = pair.rsplit(":", 1); mm[u] = float(c)
        try:
            our = run_one(fen, set())["uci"]
        except Exception:
            continue
        stm_white = (fen.split()[1] == 'w')
        our_cp = mm.get(our)
        if our_cp is None:
            worst = min(mm.values()) if stm_white else max(mm.values())
            our_cp = (worst - MISS) if stm_white else (worst + MISS)
            miss += 1
        bw = _winpct(best_cp) if stm_white else (100.0 - _winpct(best_cp))
        ow = _winpct(our_cp) if stm_white else (100.0 - _winpct(our_cp))
        reg = max(0.0, bw - ow)
        tot_reg += reg; n += 1
        if our == best_uci:
            match += 1
    print("SLICEOUT sum=%.6f n=%d match=%d miss=%d" % (tot_reg, n, match, miss))
    sys.exit(0)

# ---------------- DRIVER ----------------
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable
SET = os.environ.get("SET", "ks_sets/game_regret_set.csv")
if not os.path.isabs(SET):
    SET = os.path.join(THIS, SET)
DEPTH = os.environ.get("DEPTH", "7")
JOBS = int(os.environ.get("JOBS", "4"))
SPLIT_FRAC = os.environ.get("SPLIT_FRAC", "0.7")
PASSES = int(os.environ.get("PASSES", "1"))
WAVES = [float(x) for x in os.environ.get("WAVES", "50,40,30,20,10,0").split(",")]
HELD_TOL = float(os.environ.get("HELD_TOL", "0.02"))   # wave-held slack for accepting a move
GEN_TOL = float(os.environ.get("GEN_TOL", "0.01"))     # full-set (general) held slack -- the stability guardrail
MAXN = int(os.environ.get("MAXN", "0"))

# Seed = shipped de-dup bundle + endgame KS ON (the on-state we are validating). The descent may turn pieces
# back off (grid includes the off values) if that generalises better.
SEED = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
        "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200,
        # union of the pre-split bundle (ring-gate + floor-removed + damp) and the wave-winner endgame split
        # (unified + trimmed magnitude + material-gate off). The descent may drop any piece that doesn't earn it.
        "ENABLE_KS_UNIFIED": 1, "KS_EG_MAT_GATE": 0, "KING_SAFETY_MAG": 2500,
        "ENABLE_KS_RING_GATE": 1, "KS_MIN_ATTACKERS": 2, "KS_FLOOR": 0, "CAPG_KS_DAMP": 25}

GRID = {
    "ENABLE_KS_UNIFIED": [0, 1],
    "KS_EG_MAT_GATE": [0, 1],
    "KS_EG_MAT_FLOOR": [10, 25, 40],
    "KS_PHASE_FLOOR": [0, 16, 32],
    "KS_NO_QUEEN": [0, 6, 12],
    "KING_SAFETY_MAG": [2500, 3000, 4000],
    "ENABLE_KS_RING_GATE": [0, 1],
    "KS_MIN_ATTACKERS": [0, 2, 3],
    "KS_FLOOR": [0, 6, 13],
    "CAPG_KS_DAMP": [0, 25, 50],
}
_only = [x.strip() for x in os.environ.get("GRID_ONLY", "").split(",") if x.strip()]
if _only:
    GRID = {k: GRID[k] for k in _only if k in GRID}


def evaluate(cfg, split, stakes_min):
    env = dict(os.environ, SET=SET, DEPTH=DEPTH, SPLIT=split, SPLIT_FRAC=SPLIT_FRAC,
               STAKES_MIN=str(stakes_min), MAXN=str(MAXN))
    knob_args = ["%s=%s" % (k, v) for k, v in cfg.items()]
    procs = []
    for i in range(JOBS):
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS))
        procs.append(subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + knob_args,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, cwd=ENGINE, env=e))
    tot = 0.0; n = 0; match = 0
    for p in procs:
        out, _ = p.communicate()
        for line in out.splitlines():
            if line.startswith("SLICEOUT"):
                d = dict(tok.split("=") for tok in line.split()[1:])
                tot += float(d["sum"]); n += int(d["n"]); match += int(d["match"])
    return (tot / max(1, n)), match, n


def fmt(cfg):
    return " ".join("%s=%s" % (k, v) for k, v in sorted(cfg.items()))


# Smoke: just show wave sizes on the real data so thresholds can be sanity-checked before the long run.
if os.environ.get("SMOKE") == "1":
    rows = _load(SET)
    if MAXN:
        rows = rows[:MAXN]
    cut = int(float(SPLIT_FRAC) * len(rows))
    tune, held = rows[:cut], rows[cut:]
    print("SET=%s  total=%d  tune=%d held=%d" % (os.path.basename(SET), len(rows), len(tune), len(held)), flush=True)
    print("  %-8s %8s %8s   (nested: stakes >= T)" % ("stakesT", "tune_n", "held_n"), flush=True)
    for T in WAVES:
        tn = sum(1 for r in tune if r["_stakes"] >= T)
        hn = sum(1 for r in held if r["_stakes"] >= T)
        print("  %-8.0f %8d %8d" % (T, tn, hn), flush=True)
    sys.exit(0)

# Curriculum.
cur = dict(SEED)
off = {"OVD_BOUNDED_MODE": 0, "CENTRAL_BOUNDED_MODE": 0}   # endgame-KS-off reference (default eval)
gen_off = evaluate(off, "held", 0)[0]
gen_cur = evaluate(cur, "held", 0)[0]
print("SET=%s DEPTH=%s JOBS=%d  general(all) held: default-off=%.4f  seed(on)=%.4f"
      % (os.path.basename(SET), DEPTH, JOBS, gen_off, gen_cur), flush=True)
print("waves(stakes>=T) high->low: %s   (lower regret=better; general held must stay <= off+%.3f)"
      % (WAVES, GEN_TOL), flush=True)

for T in WAVES:
    base_t, _, tn = evaluate(cur, "tune", T)
    base_h = evaluate(cur, "held", T)[0]
    best_t = base_t
    print("\n== WAVE stakes>=%g  (tune_n~%d)  start tune=%.4f held=%.4f ==" % (T, tn, base_t, base_h), flush=True)
    for p in range(PASSES):
        moved = 0
        for knob, vals in GRID.items():
            for v in vals:
                if cur.get(knob) == v:
                    continue
                trial = dict(cur); trial[knob] = v
                t, _, _ = evaluate(trial, "tune", T)
                if t < best_t - 1e-6:
                    h = evaluate(trial, "held", T)[0]
                    g = evaluate(trial, "held", 0)[0]            # general stability guardrail
                    if h <= base_h + HELD_TOL and g <= gen_off + GEN_TOL:
                        best_t = t; base_h = min(base_h, h); cur[knob] = v; moved += 1
                        print("  %-18s -> %-6s tune=%.4f waveHeld=%.4f genHeld=%.4f" % (knob, v, t, h, g), flush=True)
                    else:
                        print("  %-18s -> %-6s REJECT (tune=%.4f waveHeld=%.4f genHeld=%.4f)" % (knob, v, t, h, g), flush=True)
        if not moved:
            print("  (converged this wave)", flush=True)
            break

gen_final = evaluate(cur, "held", 0)[0]
print("\nFINAL CONFIG: %s" % fmt(cur), flush=True)
print("general(all) held: default-off=%.4f  final=%.4f  (final should be <= off+%.3f)" % (gen_off, gen_final, GEN_TOL), flush=True)
for T in WAVES:
    fo = evaluate(off, "held", T)[0]
    fc = evaluate(cur, "held", T)[0]
    print("  wave stakes>=%-4g held: off=%.4f final=%.4f  dvsoff=%+.4f" % (T, fo, fc, fc - fo), flush=True)
