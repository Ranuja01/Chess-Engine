# -*- coding: utf-8 -*-
"""Pattern-search co-tune of an eval CLUSTER against the deterministic WDL-cploss compass (holistic campaign,
Stage 1: eval cluster, search margins FROZEN). The compass is noise-free, so we use coordinate/pattern search
(Hooke-Jeeves) with an adaptive step + a ridge-to-status-quo penalty — NOT SPSA (which wastes evals averaging
absent noise). Objective = BALANCED mean of the four per-stratum cploss means (so the 24-position collapse
stratum can't be sacrificed for the bulk) + ridge. Optimises on the TRAIN shard; validates the winner on HOLDOUT.

    python diagnostics/surrogate_tune.py <spec.json> <tag> [--evals 80] [--our-depth 10] [--judge-depth 12] [--ridge 0.5]

Deterministic, resumable (<tag>_state.json), logged (<tag>_log.csv). Prints the winning KEY=VAL config for node_ab.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
PY = sys.executable
RESULTS = os.path.join(THIS, "results")
PINS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                          "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS")}
STRATA = ["collapse", "sts", "neutral", "game"]
_LINE = re.compile(r"^\s*(collapse|sts|neutral|game)\s+([-\d.]+)\s+n=(\d+)", re.M)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def run_compass(theta, params, base, our_depth, judge_depth, shard):
    """One deterministic compass eval: returns (balanced_mean, per_stratum dict) or (None, {})."""
    env = dict(os.environ); env.update(PINS)
    env["STOCKFISH_PATH"] = env.get("SF", env.get("STOCKFISH_PATH", ""))
    env["PRESET"] = "LONG_FORMAT"; env["MAX_DEPTH"] = str(our_depth); env["USE_OPENING_BOOK"] = "0"
    for kv in base.split():
        k, v = kv.split("=", 1); env[k] = v
    for val, p in zip(theta, params):
        env[p["name"]] = str(int(round(val)))
    cmd = [PY, os.path.join("diagnostics", "cploss_frozen.py"), "--depth", str(judge_depth), "--shard", shard]
    out = subprocess.run(cmd, cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode("utf-8", "ignore")
    per = {m.group(1): float(m.group(2)) for m in _LINE.finditer(out)}
    if not per:
        return None, {}
    bal = sum(per.get(s, 0.0) for s in STRATA) / len([s for s in STRATA if s in per])
    return bal, per


def objective(theta, params, base, cfg, ridge):
    bal, per = run_compass(theta, params, base, cfg["od"], cfg["jd"], "train")
    if bal is None:
        return 1e9, {}
    pen = ridge * sum(((v - p["init"]) / p["scale"]) ** 2 for v, p in zip(theta, params))
    return bal + pen, per


def cfg_string(theta, params, base):
    return base + " " + " ".join("%s=%d" % (p["name"], int(round(v))) for v, p in zip(theta, params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("tag")
    ap.add_argument("--evals", type=int, default=80)
    ap.add_argument("--our-depth", type=int, default=10)
    ap.add_argument("--judge-depth", type=int, default=12)
    ap.add_argument("--ridge", type=float, default=0.5)
    a = ap.parse_args()
    spec = json.load(open(a.spec if os.path.isabs(a.spec) else os.path.join(ENGINE, a.spec)))
    base, params = spec["base"], spec["params"]
    cfg = {"od": a.our_depth, "jd": a.judge_depth}
    os.makedirs(RESULTS, exist_ok=True)
    logp = os.path.join(RESULTS, "%s_log.csv" % a.tag)
    statep = os.path.join(RESULTS, "%s_state.json" % a.tag)

    theta = [float(p["init"]) for p in params]
    step = [0.5 * (p["max"] - p["min"]) for p in params]      # trust region: 50% of range, halves on stall
    n_eval = [0]
    log_rows = []

    def logged_obj(th):
        val, per = objective(th, params, base, cfg, a.ridge)
        n_eval[0] += 1
        row = {"eval": n_eval[0], "obj": round(val, 3), **{p["name"]: int(round(v)) for v, p in zip(th, params)},
               **{("s_" + s): per.get(s) for s in STRATA}}
        log_rows.append(row)
        print("  eval %2d  obj=%.3f  %s" % (n_eval[0], val,
              " ".join("%s=%d" % (p["name"], round(v)) for v, p in zip(th, params))), flush=True)
        return val

    print("[surrogate] baseline (init theta) on TRAIN:", flush=True)
    best = logged_obj(theta)
    min_step = [max(1.0, 0.02 * (p["max"] - p["min"])) for p in params]
    while n_eval[0] < a.evals:
        improved = False
        for j in range(len(params)):
            for sgn in (+1, -1):
                if n_eval[0] >= a.evals:
                    break
                cand = list(theta)
                cand[j] = clamp(theta[j] + sgn * step[j], params[j]["min"], params[j]["max"])
                if abs(cand[j] - theta[j]) < 1e-9:
                    continue
                v = logged_obj(cand)
                if v < best - 1e-6:
                    best, theta, improved = v, cand, True
                    break
        if not improved:
            if all(step[j] <= min_step[j] for j in range(len(params))):
                print("[surrogate] converged (trust region minimal).", flush=True)
                break
            step = [max(s * 0.5, min_step[j]) for j, s in enumerate(step)]
            print("[surrogate] no improvement -> shrink step: %s" % [round(s) for s in step], flush=True)

    # winner -> holdout validation + node_ab-ready config
    win_cfg = cfg_string(theta, params, base)
    bh, ph = run_compass(theta, params, base, a.our_depth, a.judge_depth, "holdout")
    b0h, p0h = run_compass([float(p["init"]) for p in params], params, base, a.our_depth, a.judge_depth, "holdout")
    with open(logp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys())); w.writeheader(); w.writerows(log_rows)
    json.dump({"theta": theta, "best_train": best, "holdout": bh, "holdout_init": b0h}, open(statep, "w"))
    print("\n[surrogate] WINNER  train_obj=%.3f" % best)
    print("[surrogate] HOLDOUT balanced: init=%.2f -> winner=%.2f  (%s)" %
          (b0h if b0h else -1, bh if bh else -1, "IMPROVED" if (bh and b0h and bh < b0h) else "no-improve"))
    print("[surrogate] holdout per-stratum winner:", {s: ph.get(s) for s in STRATA})
    print("[surrogate] NODE_AB CONFIG: %s" % win_cfg)


if __name__ == "__main__":
    main()
