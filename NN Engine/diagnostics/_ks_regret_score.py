# -*- coding: utf-8 -*-
"""Score a FIXED LIST of KS-unit configs (layered on the OvD+central bundle) on side-to-move win%-regret
of our fixed-depth move, on the game-representative multi-PV set. This is the Elo-bearing arbiter for the
KS discrimination unit: it has a BUILT-IN positive signal (dropping genuine threats -> our D7 search picks
worse moves -> regret RISES; removing only move-flipping over-reads -> regret FALLS), which the firing
decomp lacks (that set is all collapses, no genuine-attack positives).

Self-contained (own worker) so MAXN subsampling actually takes effect -- must NOT be delegated to a worker
that ignores it. Same shuffle+split(1234) and win%-regret math as _regret_tune_broad, so numbers compare.

  pyrun diagnostics/_ks_regret_score.py [SET=ks_sets/game_regret_set.csv] [DEPTH=7] [JOBS=4]
        [SPLIT_FRAC=0.7] [MAXN=0]   (MAXN>0 = subsample the first N shuffled rows for a fast read)

⚠️ FIXED depth (deterministic), a PROXY for game depth. Winner still needs STS/symmetry + games.
"""
import os, sys, csv, math, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v


def _winpct(cp):
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


# ---------------- WORKER: one config, one position slice, one split -> summed regret ----------------
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
    MAXN = int(os.environ.get("MAXN", "0"))
    rows = list(csv.DictReader(open(SET, newline="")))
    import random as _r
    _r.Random(1234).shuffle(rows)
    if MAXN:
        rows = rows[:MAXN]                                   # subsample the SAME shuffled prefix for all configs
    cut = int(frac * len(rows))
    rows = rows[:cut] if which == "tune" else (rows[cut:] if which == "held" else rows)
    rows = rows[si::sn]
    tot_reg = 0.0; n = 0; match = 0; miss = 0; crit_reg = 0.0; crit_n = 0
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
        others = [c for u, c in mm.items() if u != best_uci]
        if others:
            second = max(others) if stm_white else min(others)
            sw = _winpct(second) if stm_white else (100.0 - _winpct(second))
            if (bw - sw) >= 20.0:                    # CRITICAL: one clearly-best move (>20% win gap)
                crit_reg += reg; crit_n += 1
        if our == best_uci:
            match += 1
    print("SLICEOUT sum=%.6f n=%d match=%d miss=%d cs=%.6f cn=%d" % (tot_reg, n, match, miss, crit_reg, crit_n))
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
MAXN = int(os.environ.get("MAXN", "0"))

BUNDLE = {"OVD_BOUNDED_MODE": 2, "OVD_CAP": 300, "OVD_KNEE": 40,
          "CENTRAL_BOUNDED_MODE": 1, "CENTRAL_CAP": 150, "CENTRAL_KNEE": 200}


def cfg(**kw):
    d = dict(BUNDLE); d.update(kw); return d


CONFIGS = [
    # 2026-08-18 FULL-SAMPLE sweep of the continuous-onset gate. AUC turned out MONOTONE in KS_FLOOR (it
    # just wants 0), so AUC cannot choose this knob -- D7 is the TUNER here, not a confirmation gate.
    # Mode 1 subtracts ks_safety_table[KS_FLOOR] so the gate costs no step and everything above it is
    # priced lower (net-subtractive). Floor 6 is the lowest gate that does not raise danger magnitude
    # over today; 4 and 9 bracket it.
    ("bundle + defaware1 (base)", cfg(KS_DEFAWARE_MODE=1)),
    ("onset1 floor6 (candidate)", cfg(KS_DEFAWARE_MODE=1, KS_ONSET_MODE=1, KS_FLOOR=6)),
    ("onset1 floor4", cfg(KS_DEFAWARE_MODE=1, KS_ONSET_MODE=1, KS_FLOOR=4)),
    ("onset1 floor9", cfg(KS_DEFAWARE_MODE=1, KS_ONSET_MODE=1, KS_FLOOR=9)),
]


def evaluate(config, split):
    env = dict(os.environ, SET=SET, DEPTH=DEPTH, SPLIT=split, SPLIT_FRAC=SPLIT_FRAC, MAXN=str(MAXN))
    knob_args = ["%s=%s" % (k, v) for k, v in config.items()]
    procs = []
    for i in range(JOBS):
        e = dict(env, WORKER="1", SLICE="%d/%d" % (i, JOBS))
        procs.append(subprocess.Popen([PY, "-u", os.path.abspath(__file__)] + knob_args,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, cwd=ENGINE, env=e))
    tot = 0.0; n = 0; match = 0; miss = 0; cs = 0.0; cn = 0
    for p in procs:
        out, _ = p.communicate()
        for line in out.splitlines():
            if line.startswith("SLICEOUT"):
                d = dict(tok.split("=") for tok in line.split()[1:])
                tot += float(d["sum"]); n += int(d["n"]); match += int(d["match"]); miss += int(d["miss"])
                cs += float(d.get("cs", 0)); cn += int(d.get("cn", 0))
    return (tot / max(1, n)), match, n, (cs / max(1, cn)), cn


print("SET=%s DEPTH=%s JOBS=%d MAXN=%s  (lower regret = better; match = top-1 vs SF18)" %
      (os.path.basename(SET), DEPTH, JOBS, MAXN or "full"), flush=True)
print("  %-40s %8s %8s %8s %7s" % ("config", "TUNEreg", "HELDreg", "HELDcrit", "n_crit"), flush=True)
base_t = base_h = base_hc = None
for label, c in CONFIGS:
    t, tm, tn, tc, tcn = evaluate(c, "tune")
    h, hm, hn, hc, hcn = evaluate(c, "held")
    if base_t is None:
        base_t, base_h, base_hc = t, h, hc
    print("  %-40s %8.4f %8.4f %8.4f %6d   (dH=%+.4f dHcrit=%+.4f)"
          % (label, t, h, hc, hcn, h - base_h, hc - base_hc), flush=True)
print("\n  read: a KS row with LOWER tune AND held regret than 'bundle' is a real move-quality gain layered\n"
      "  on the bundle. Held is the generalisation gate. dT/dH are vs 'default (all off)'.", flush=True)
