# -*- coding: utf-8 -*-
"""KS discrimination fit against SF11-static, on the STS-anchored corpus (build_ks_sts_corpus.py).
Asks the decisive question: can our KS knobs FIRE on the genuine-attack tier WITHOUT waking the
quiet_neg tier (where SF11-static stays silent and our recalibration over-fired)?

Coordinate-descends the KS knobs to minimise the 'attack' tier win%-error SUBJECT TO 'quiet_neg'
(and 'mid') not rising beyond tolerance above baseline. Baseline = engine defaults (KS effectively
off on this corpus -> quiet_neg already ~perfect, attack under-fires). If the descent can cut attack
error while holding quiet_neg, that config is the shippable discriminating zone; if it cannot, the
knob space can't express SF's discrimination -> detection rebuild.

  pyrun diagnostics/ks_fit_sts.py [ROUNDS=2] [GUARD_TOL=1.0]
"""
import os, sys, subprocess, re
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
ROUNDS = int(os.environ.get('ROUNDS', '2'))
TOL = float(os.environ.get('GUARD_TOL', '1.0'))
THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
PY = sys.executable

GRID = {
    "KS_MIN_ATTACKERS": [0, 1, 2, 3],
    "KS_OVERLOAD":      [0, 1, 2, 3, 5],
    "KS_ATT_PRODUCT":   [0, 1, 2, 4],
    "KS_FLOOR":         [0, 3, 6, 9, 13],
    "KS_CAP":           [80, 150, 300, 600],
    "KS_KNEE":          [12, 24, 40, 80],
    "KS_DIVISOR":       [2, 3, 4, 6],
    "KS_SAFE_CHECK":    [3, 8, 15, 25],
    "KS_WEAK":          [2, 6, 12],
    "KS_ATTACK_COUNT":  [0, 1, 2, 4],
    "KS_ATT_KNIGHT":    [2, 5, 9, 16],
    "KS_ATT_BISHOP":    [2, 5, 9, 16],
    "KS_ATT_ROOK":      [3, 6, 10, 18],
    "KS_ATT_QUEEN":     [5, 10, 16, 24],
    "KING_SAFETY_MAG":  [3000, 4500, 6000],
}
cur = {"KS_MIN_ATTACKERS": 0, "KS_OVERLOAD": 0, "KS_ATT_PRODUCT": 0, "KS_FLOOR": 13, "KS_CAP": 80,
       "KS_KNEE": 12, "KS_DIVISOR": 4, "KS_SAFE_CHECK": 3, "KS_WEAK": 2, "KS_ATTACK_COUNT": 1,
       "KS_ATT_KNIGHT": 2, "KS_ATT_BISHOP": 2, "KS_ATT_ROOK": 3, "KS_ATT_QUEEN": 5, "KING_SAFETY_MAG": 3000}

TARGET = "attack"
GUARDS = ["quiet_neg", "mid"]


def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {}
    for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out):
        d[k] = float(v)
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    if m:
        d["ALL.train"], d["ALL.val"] = float(m.group(1)), float(m.group(2))
    d["_dir"] = out
    return d


def g(d, key):
    return d.get(key, 9e9)


base_d = evaluate(cur)
base_guard = {t: g(base_d, t + ".train") for t in GUARDS}
best_tgt = g(base_d, TARGET + ".train")
print("baseline: %s.train=%.3f  guards=%s  ALL.val=%.3f"
      % (TARGET, best_tgt, {t: round(base_guard[t], 3) for t in GUARDS}, g(base_d, "ALL.val")))
# show baseline direction lines
for ln in base_d["_dir"].splitlines():
    if ln.startswith("DIR"):
        print("  " + ln)


def guards_ok(d):
    return all(g(d, t + ".train") <= base_guard[t] + TOL for t in GUARDS)


for rnd in range(ROUNDS):
    print("\n=== round %d ===" % (rnd + 1))
    for knob, cands in GRID.items():
        base = cur[knob]
        for v in cands:
            if v == cur[knob]:
                continue
            trial = dict(cur); trial[knob] = v
            d = evaluate(trial)
            if g(d, TARGET + ".train") < best_tgt - 1e-4 and guards_ok(d):
                best_tgt, cur[knob] = g(d, TARGET + ".train"), v
        if cur[knob] != base:
            print("  %-16s %s -> %s   (%s.train=%.3f)" % (knob, base, cur[knob], TARGET, best_tgt))

final = evaluate(cur)
print("\nBEST CONFIG:", " ".join("%s=%d" % (k, v) for k, v in cur.items()))
print("%s.train=%.3f (baseline %.3f)  guards=%s  ALL.val=%.3f" % (
    TARGET, g(final, TARGET + ".train"), g(base_d, TARGET + ".train"),
    {t: round(g(final, t + ".train"), 3) for t in GUARDS}, g(final, "ALL.val")))
print("\nfinal direction metrics:")
for ln in final["_dir"].splitlines():
    if ln.startswith("DIR"):
        print("  " + ln)
