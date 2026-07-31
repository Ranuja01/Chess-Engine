# -*- coding: utf-8 -*-
"""Joint KS suppressor-portfolio fit on the DIVERSE, SF18-anchored corpus (build_diverse_corpus.py).
Constrained win%-space coordinate descent (reuses the _ks_fit_eval.py worker): minimise the KS-under-read
`target` tier error SUBJECT TO every control family (calm / working / crowded_safe / sts_guard) not rising
above baseline by more than GUARD_TOL. The `sts_guard` tier is the diverse "prone-to-break" positional set
whose omission hid this session's STS regression.

DOUBLE-COUNT DISCIPLINE: the primary fit tunes only the CLEAN KS levers (gate/floor/no-queen/defender/
safe-check/weak/attack-count/per-piece/knee/cap/divisor). The king-zone attacker-minus-defender TRIPLE
(KS_OVERLOAD / IMBALANCE_SCALE / SCALE_ATTACK_LAYER / SCALE_LATENT_THREAT / MOD_KS_CONTROL) and shelter
(KS_SHIELD/KS_CONSOLIDATE) are HELD FIXED here to avoid gradient-splitting; a separate co-fit pass can open
them deliberately.

  pyrun diagnostics/ks_fit_diverse.py [ROUNDS=2] [GUARD_TOL=1.0]
"""
import os, sys, subprocess, re
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
ROUNDS = int(os.environ.get('ROUNDS', '2'))
TOL = float(os.environ.get('GUARD_TOL', '1.0'))
THIS = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(THIS, "ks_sets", "diverse_corpus_ksplus.csv")
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
PY = sys.executable

# CLEAN + modest suppressor levers only (double-count-safe; OvD triple + shelter held fixed).
# ENABLE_KS_CHECK_V2 is FIXED ON (not in GRID) -> per-type SATURATED safe-checks replace the flat KS_SAFE_CHECK
# path; KS_CHK_* are the tunable per-type weights. The convertibility lever is KS_WEAK (weak ring-squares behind
# a check separate real from phantom) + the stricter ENABLE_KS_SF_WEAK under-defended definition.
GRID = {
    "KS_CHK_QUEEN":     [10, 14, 20, 28],       # per-type SATURATED safe-check weights (V2). Lone Q/R must clear
    "KS_CHK_ROOK":      [8, 14, 20],            # KS_FLOOR=13 on its own merit.
    "KS_CHK_BISHOP":    [5, 7, 12],
    "KS_CHK_KNIGHT":    [6, 9, 14],
    "KS_CHK_MULTI":     [0, 4, 8],             # graded 2nd-same-type-square bump (SF15 more_than_one)
    "ENABLE_KS_SF_WEAK":[0, 1],                # stricter weak = under-defended (<=1 def, K/Q only)
    "KS_WEAK":          [2, 3, 4, 6],           # CONVERTIBILITY: weak ring-squares (SF's highest-value KS term)
    "KS_MIN_ATTACKERS": [0, 1, 2, 3],          # coordination gate (structural, adds no magnitude)
    "KS_FLOOR":         [0, 3, 6, 9, 13],       # magnitude deadzone
    "KS_NO_QUEEN":      [6, 12, 18, 24],        # suppressor: no enemy queen (CLEAN channel)
    "KS_DEFENDER":      [0, 2, 3, 5],           # suppressor: friendly king-zone defenders (modest; mild imbal overlap)
    "KS_ATTACK_COUNT":  [1, 2],                 # positive: proximity (kept modest)
    "KS_ATT_KNIGHT":    [2, 5, 9],
    "KS_ATT_BISHOP":    [2, 5, 9],
    "KS_ATT_ROOK":      [3, 6, 10],
    "KS_ATT_QUEEN":     [5, 10, 16],
    "KS_KNEE":          [12, 24, 40],
    "KS_CAP":           [80, 150, 300],
    "KS_DIVISOR":       [3, 4, 6],
}
cur = {"ENABLE_KS_CHECK_V2": 1,
       "KS_CHK_QUEEN": 14, "KS_CHK_ROOK": 14, "KS_CHK_BISHOP": 7, "KS_CHK_KNIGHT": 9, "KS_CHK_MULTI": 0,
       "ENABLE_KS_SF_WEAK": 0,
       "KS_MIN_ATTACKERS": 0, "KS_FLOOR": 13, "KS_NO_QUEEN": 6, "KS_DEFENDER": 0,
       "KS_WEAK": 2, "KS_ATTACK_COUNT": 1, "KS_ATT_KNIGHT": 2, "KS_ATT_BISHOP": 2, "KS_ATT_ROOK": 3,
       "KS_ATT_QUEEN": 5, "KS_KNEE": 12, "KS_CAP": 80, "KS_DIVISOR": 4}

TARGET = "target"
GUARDS = ["calm", "working", "crowded_safe", "sts_guard", "ks_fixable", "ks_phantom", "ks_edge"]


def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {"_dir": out}
    for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out):
        d[k] = float(v)
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    if m:
        d["ALL.train"], d["ALL.val"] = float(m.group(1)), float(m.group(2))
    return d


def g(d, key):
    return d.get(key, 9e9)


base_d = evaluate(cur)
base_guard = {t: g(base_d, t + ".train") for t in GUARDS}
best_tgt = g(base_d, TARGET + ".train")
print("baseline: %s.train=%.3f  guards=%s  ALL.val=%.3f"
      % (TARGET, best_tgt, {t: round(base_guard[t], 3) for t in GUARDS}, g(base_d, "ALL.val")))
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
