# -*- coding: utf-8 -*-
"""BROAD regret tune: coordinate-descend the WHOLE eval knob set to MINIMISE mean side-to-move win%-regret
of our fixed-depth move, on the game-representative multi-PV set. The move-based objective is robust to the
eval's collinearity (it optimises the TOTAL's move quality, not the scalar decomposition -> no flattening,
no ridge needed), so it tunes the messy eval as-is -- de-noising is NOT a prerequisite here.

Parallel: each candidate's positions are split across JOBS workers (JOBS=1 = pure single-core). Coordinate
descent stays sequential (correct); only the per-candidate eval is parallelised.

Split: first SPLIT_FRAC of the set = TUNE (fitted), remainder = HELD-OUT (winner checked at the end only).

  pyrun diagnostics/_regret_tune_broad.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [PASSES=2] [JOBS=4]
        [SPLIT_FRAC=0.7] [GRID_ONLY=k1,k2] [MISS=30]

⚠️ FIXED depth (deterministic). Winner must be validated on STS/WAC/symmetry + games. Low depth is a PROXY.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---------------- WORKER: config + position SLICE + SPLIT -> summed regret over that slice ----------------
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
    si, sn = (int(x) for x in os.environ["SLICE"].split("/"))       # this worker's slice i of n
    frac = float(os.environ.get("SPLIT_FRAC", "0.7"))
    which = os.environ.get("SPLIT", "tune")                          # tune | held | all
    rows = list(csv.DictReader(open(SET, newline="")))
    # FIXED-SEED shuffle before the split so TUNE and HELD are the SAME distribution. The set is collected
    # file-by-file from DIFFERENT A/B game configs (ab_base/ab_k50/ab_damp/...), so an index split would put
    # different game POPULATIONS in tune vs held -> a distribution mismatch that looks exactly like universal
    # overfitting. Seeded so every worker/eval builds the identical partition (reproducible).
    import random as _r
    _r.Random(1234).shuffle(rows)
    cut = int(frac * len(rows))
    rows = rows[:cut] if which == "tune" else (rows[cut:] if which == "held" else rows)
    rows = rows[si::sn]                                              # this worker's share
    tot_reg = 0.0; n = 0; match = 0; miss = 0
    for r in rows:
        fen = r.get("fen"); best_uci = r.get("best_uci")
        try:
            best_cp = float(r["best_cp"])
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
        reg = bw - ow
        if reg < 0:
            reg = 0.0
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
PASSES = int(os.environ.get("PASSES", "2"))
JOBS = int(os.environ.get("JOBS", "4"))
SPLIT_FRAC = os.environ.get("SPLIT_FRAC", "0.7")

# Seed: shipped defaults + the de-dup bounded terms ON (part of the current eval). Everything is free to move.
SEED = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
        "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}

# Broad eval grid (magnitude/context knobs that move move-quality). Trim via GRID_ONLY.
GRID = {
    "OVD_CAP": [200, 300, 450], "OVD_KNEE": [20, 40, 80],
    "CENTRAL_CAP": [125, 150, 200], "CENTRAL_KNEE": [100, 200, 300],
    "SCALE_CENTRAL": [70, 100, 130], "IMBALANCE_SCALE": [0, 2, 3, 4],
    "KAUFMAN_SCALE": [70, 100, 130], "SCALE_ATTACK_LAYER": [70, 100, 130],
    "SCALE_THREATS": [55, 75, 100], "THREAT_PER_TARGET_CAP": [600, 800, 1100],
    "SCALE_CAPTURE_GAINS": [80, 100, 120], "MOD_KS_REALIZ": [0, 128, 256],
    "KS_REALIZ_FLOOR": [64, 128, 200], "SCALE_PASSED_PAWN": [70, 100, 130],
    "PASSER_MAG_SCALE": [80, 100, 130], "PASSER_R_CAP": [256, 320, 400],
    "PP_HORIZ_SUPPORT": [110, 160, 225], "PP_FILE_CLEAR": [60, 100, 150],
    "SCALE_PAWN_CHAIN": [70, 100, 130], "SCALE_PAWN_WALL": [70, 100, 130],
    "EG_SUPPORT": [95, 135, 180], "EG_DEFEND": [55, 80, 115],
    "MG_CLAMP_KNIGHT": [3000, 3750, 4500], "MG_CLAMP_BISHOP_A": [3100, 3850, 4600],
    "EG_EXIST_ROOK": [250, 350, 470], "EG_EXIST_QUEEN": [500, 700, 900],
    "PV_BOOST_TRIGGER": [1100, 1500, 2000], "PV_BOOST_MAG": [5000, 7500, 10000],
}
_only = [x.strip() for x in os.environ.get("GRID_ONLY", "").split(",") if x.strip()]
if _only:
    GRID = {k: GRID[k] for k in _only if k in GRID}


def evaluate(cfg, split="tune"):
    """Spawn JOBS position-slice workers concurrently; aggregate to mean regret."""
    env = dict(os.environ, SET=SET, DEPTH=DEPTH, SPLIT=split, SPLIT_FRAC=SPLIT_FRAC)
    knob_args = ["%s=%s" % (k, v) for k, v in cfg.items()]
    procs = []
    for i in range(JOBS):
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS))
        procs.append(subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + knob_args,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       text=True, cwd=ENGINE, env=e))
    tot = 0.0; n = 0; match = 0; miss = 0
    for p in procs:
        out, _ = p.communicate()
        for line in out.splitlines():
            if line.startswith("SLICEOUT"):
                d = dict(tok.split("=") for tok in line.split()[1:])
                tot += float(d["sum"]); n += int(d["n"]); match += int(d["match"]); miss += int(d["miss"])
    return (tot / max(1, n)), match, miss, n


def fmt(cfg):
    return " ".join("%s=%s" % (k, v) for k, v in sorted(cfg.items()))


HELD_TOL = float(os.environ.get("HELD_TOL", "0.005"))   # a move must not worsen held beyond this
default_reg, default_m, _, tot = evaluate({"OVD_BOUNDED_MODE": 0, "CENTRAL_BOUNDED_MODE": 0})
cur = dict(SEED)
seed_reg, seed_m, seed_miss, tot = evaluate(cur)
best_reg = seed_reg
best_held = evaluate(cur, split="held")[0]           # gate moves on held-out generalisation, not just tune
print("SET=%s DEPTH=%s JOBS=%d  tune=%d rows  (lower regret = better)" % (os.path.basename(SET), DEPTH, JOBS, tot), flush=True)
print("default(off) regret=%.4f match=%d   seed(de-dup) tune=%.4f held=%.4f match=%d miss=%d"
      % (default_reg, default_m, seed_reg, best_held, seed_m, seed_miss), flush=True)

for p in range(PASSES):
    moved = 0
    for knob, vals in GRID.items():
        for v in vals:
            if cur.get(knob) == v:
                continue
            trial = dict(cur); trial[knob] = v
            reg, m, _, _ = evaluate(trial, "tune")
            if reg < best_reg - 1e-6:                 # improves tune -> check it GENERALISES before accepting
                hreg = evaluate(trial, "held")[0]
                if hreg <= best_held + HELD_TOL:
                    best_reg = reg; best_held = min(best_held, hreg); cur[knob] = v; moved += 1
                    print("  p%d %-22s -> %-6s tune=%.4f held=%.4f match=%d" % (p, knob, v, best_reg, hreg, m), flush=True)
                else:
                    print("  p%d %-22s -> %-6s REJECT (tune=%.4f but held=%.4f > %.4f, overfit)"
                          % (p, knob, v, reg, hreg, best_held), flush=True)
    print("[pass %d] %d moves, tune=%.4f held=%.4f" % (p, moved, best_reg, best_held), flush=True)
    if not moved:
        print("converged", flush=True)
        break

# Held-out check on the winner + the two references (overfit guard).
h_win = evaluate(cur, split="held")
h_seed = evaluate(SEED, split="held")
h_def = evaluate({"OVD_BOUNDED_MODE": 0, "CENTRAL_BOUNDED_MODE": 0}, split="held")
print("\nBEST CONFIG: %s" % fmt(cur), flush=True)
print("TUNE  regret=%.4f (seed %.4f, default %.4f)" % (best_reg, seed_reg, default_reg), flush=True)
print("HELD  regret win=%.4f  seed=%.4f  default=%.4f  (win should beat both to be real)"
      % (h_win[0], h_seed[0], h_def[0]), flush=True)
