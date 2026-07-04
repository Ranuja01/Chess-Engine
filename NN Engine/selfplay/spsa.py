#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SPSA tuner for the C++ engine's env-knob search/eval parameters.

Simultaneous Perturbation Stochastic Approximation over a small knob vector, using paired self-play A/B
games as the (noisy) objective. Each iteration perturbs every knob at once by +/- c_k, plays theta+ vs
theta- as ONE tournament (the paired game directly measures the strength DIFFERENCE), and steps every knob
along the shared gradient sign. This is the cheap volume path: run it on the d4/6 fixed-depth tournament
(EVAL knobs) or equal-time lightning (SEARCH knobs) for ~20k games/night as a candidate GENERATOR; the
winner is ratified by a separate lightning SPRT (`overnight_runner.sh gate ...`), never shipped on SPSA alone.

Reuses `tournament.py` verbatim: two configs are space-separated `KEY=VAL` env-knob strings passed as
--p1-config / --p2-config; the per-run standings land in `games/<tag>_kNN/tournament.json` (`score_pct` =
p1's score fraction). No engine changes, no new game harness.

KNOB LANE (critical): search/prune knobs must be tuned at EQUAL TIME (--preset LIGHTNING), because a
fixed-depth game penalises a pruning win (it just searches fewer nodes to the same depth). EVAL knobs may
use fixed-depth (--preset LONG_FORMAT --games-config 'MAX_DEPTH=<d>') where the A/B is cleaner. Pick with
--lane.

Usage (from NN Engine/, WSL, thread-pinned like the other subs):
    python selfplay/spsa.py --spec spsa_spec.json --iters 40 --games 60 --lane search --tag spsa_seecap
where spsa_spec.json = [{"name":"SEE_PRUNE_CAPTURE_MARGIN","init":1000,"min":0,"max":4000,"scale":600}, ...]
(`scale` = the knob's expected sensitivity span; the SPSA c/a gains are expressed as fractions of it so all
knobs move on a comparable footing regardless of their raw units).

Resumable: writes theta + iter to `<tag>_state.json` each step; --resume continues from it.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import json
import csv
import math
import argparse
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
GAMES_DIR = os.path.join(THIS_DIR, "games")
PY = sys.executable


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _cfg_string(theta, spec, base):
    """Render a theta vector (+ any fixed base knobs) as a space-separated KEY=VAL env string."""
    parts = list(base)
    for v, k in zip(theta, spec):
        parts.append("%s=%d" % (k["name"], int(round(v))))
    return " ".join(parts)


def _rademacher(n, it, idx_salt):
    """Deterministic +/-1 perturbation vector (no RNG — Date/random are avoided for reproducibility).
    Derived from a cheap integer hash of (iteration, knob index) so each iteration gets a fresh pattern."""
    out = []
    for j in range(n):
        h = (it * 2654435761 + (j + idx_salt) * 40503) & 0xFFFFFFFF
        out.append(1 if (h & 0x10000) else -1)
    return out


def _run_ab(cfg_plus, cfg_minus, games, preset, base_depth, tag, seed, adj_sf):
    """Play cfg_plus (P1) vs cfg_minus (P2) as one tournament; return P1 score fraction in [0,1] or None."""
    ttag = tag
    cmd = [PY, os.path.join("selfplay", "tournament.py"),
           "--p1-label", "plus", "--p1-config", cfg_plus,
           "--p2-label", "minus", "--p2-config", cfg_minus,
           "--games", str(games), "--concurrency", str(args.concurrency),
           "--preset", preset, "--max-plies", "400",
           "--openings", os.path.join("selfplay", "openings_uho.txt"),
           "--seed", str(seed), "--tag", ttag, "--quiet"]
    if preset == "LONG_FORMAT" and base_depth:
        # fixed-depth lane: bound both sides by depth, not clock
        cmd[cmd.index("--p1-config") + 1] = ("MAX_DEPTH=%d " % base_depth) + cfg_plus
        cmd[cmd.index("--p2-config") + 1] = ("MAX_DEPTH=%d " % base_depth) + cfg_minus
    if adj_sf:
        cmd += ["--adjudicate-draw", "--sf-path", adj_sf]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[k] = "1"
    subprocess.run(cmd, cwd=ENGINE_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    jf = os.path.join(GAMES_DIR, ttag, "tournament.json")
    try:
        with open(jf) as f:
            d = json.load(f)
        return float(d.get("score_pct", 50.0)) / 100.0
    except Exception:
        return None


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of {name,init,min,max,scale}")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=60, help="paired games per iteration")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--lane", choices=["search", "eval"], default="search",
                    help="search=equal-time LIGHTNING; eval=fixed-depth LONG_FORMAT")
    ap.add_argument("--depth", type=int, default=6, help="fixed depth for the eval lane")
    ap.add_argument("--base", default="", help="fixed KEY=VAL knobs applied to BOTH sides")
    ap.add_argument("--adj-sf", default="", help="SF path for draw adjudication (optional)")
    ap.add_argument("--tag", default="spsa")
    ap.add_argument("--a", type=float, default=0.15, help="SPSA step gain (fraction of knob scale)")
    ap.add_argument("--c", type=float, default=0.30, help="SPSA perturb gain (fraction of knob scale)")
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    n = len(spec)
    preset = "LIGHTNING" if args.lane == "search" else "LONG_FORMAT"
    base = args.base.split() if args.base else []

    theta = [float(k["init"]) for k in spec]
    start_it = 1
    state_path = os.path.join(GAMES_DIR, "%s_state.json" % args.tag)
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        theta = st["theta"]
        start_it = st["iter"] + 1
        print("[spsa] resumed at iter %d theta=%s" % (start_it, theta))

    csv_path = os.path.join(GAMES_DIR, "%s_log.csv" % args.tag)
    new_csv = not os.path.exists(csv_path)
    log = open(csv_path, "a", newline="")
    w = csv.writer(log)
    if new_csv:
        w.writerow(["iter", "y_plus_score"] + [k["name"] for k in spec])

    A = max(1, args.iters // 10)
    for it in range(start_it, args.iters + 1):
        ck = args.c / (it ** args.gamma)
        ak = args.a / ((it + A) ** args.alpha)
        delta = _rademacher(n, it, 7)
        theta_p = [_clamp(theta[j] + ck * spec[j]["scale"] * delta[j], spec[j]["min"], spec[j]["max"]) for j in range(n)]
        theta_m = [_clamp(theta[j] - ck * spec[j]["scale"] * delta[j], spec[j]["min"], spec[j]["max"]) for j in range(n)]
        y = _run_ab(_cfg_string(theta_p, spec, base), _cfg_string(theta_m, spec, base),
                    args.games, preset, args.depth, "%s_k%02d" % (args.tag, it), 1000 + it, args.adj_sf)
        if y is None:
            print("[spsa] iter %d: A/B failed, skipping step" % it)
            continue
        # Paired measurement: y-0.5 IS L(theta+)-L(theta-) in score units. Step every knob along the shared sign.
        g_scalar = (y - 0.5)
        for j in range(n):
            theta[j] = _clamp(theta[j] + ak * spec[j]["scale"] * g_scalar / (ck * delta[j]),
                              spec[j]["min"], spec[j]["max"])
        print("[spsa] iter %d  y+=%.3f  theta=%s" % (it, y, [int(round(v)) for v in theta]))
        w.writerow([it, "%.4f" % y] + [int(round(v)) for v in theta])
        log.flush()
        with open(state_path, "w") as f:
            json.dump({"iter": it, "theta": theta}, f)

    log.close()
    final = _cfg_string(theta, spec, [])
    print("\n[spsa] DONE. best theta config: %s" % final)
    print("[spsa] ratify with:  overnight_runner.sh gate '%s' %s sprt_%s 1000 5" % (final, args.tag, args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
