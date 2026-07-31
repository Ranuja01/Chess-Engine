# -*- coding: utf-8 -*-
"""KS FIT driver. Builds an SF18-gated target corpus from the position bank, then coordinate-descends the KS
knobs (via subprocess workers _ks_fit_eval.py, since env->Config is per-process) to minimise
(our_ks - target_ks)^2 across all tiers/phases, with a held-out validation split so we don't overfit the targets.

Target (WHITE-POV pawns), per the design:
  - anchor = SF11's King-safety term (sf11_ks).
  - VALIDITY (SF11 vs SF18 totals): drop rows where SF11-static and SF18-search disagree in direction AND both
    are substantial (SF11 completely off -> lost cause / search-only).
  - ASPIRATION: where SF18 confirms the SAME direction but STRONGER, scale the KS target up (bounded) -> aim
    beyond SF11 where it's safe.
Tiers (from bank labels): target (gap), working (agree), crowded_safe (dense but SF18-safe), calm (quiet).

Run: pyrun diagnostics/ks_fit.py [ROUNDS=2] [ASPIRE_CAP=1.5]   (writes ks_sets/fit_corpus.csv + prints best cfg)
"""
import os, sys, csv, subprocess, re
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
ROUNDS     = int(os.environ.get('ROUNDS', '2'))
ASPIRE_CAP = float(os.environ.get('ASPIRE_CAP', '1.5'))
THIS = os.path.dirname(os.path.abspath(__file__))
BANK   = os.path.join(THIS, "ks_sets", "position_bank.csv")
CORPUS = os.path.join(THIS, "ks_sets", "fit_corpus.csv")
WORKER = os.path.join(THIS, "_ks_fit_eval.py")
PY = sys.executable

def sgn(x): return (x > 0) - (x < 0)
def phase_bucket(ps):
    ps = float(ps)
    return "opening" if ps < 24 else "midgame" if ps < 64 else "endgame" if ps < 104 else "adveg"

# ---- build target corpus from SF18-labeled bank rows ----
rows = [r for r in csv.DictReader(open(BANK)) if r.get("sf18", "") not in ("", None)]
corpus = []
for r in rows:
    try:
        our_ks = float(r["our_ks"]); sf11_ks = float(r["sf11_ks"])
        our_total = float(r["our_total"])
        sf11_tot = float(r["sf11_total"]); sf18 = float(r["sf18"])
        kz = max(int(r["kzone_w"]), int(r["kzone_b"]))
    except Exception:
        continue
    # validity: SF11-static completely off from SF18-search -> exclude (lost cause / search-only)
    if sgn(sf11_tot) != sgn(sf18) and abs(sf11_tot) >= 1.0 and abs(sf18) >= 1.0:
        continue
    # target KS: apply SF11's KS attribution ONLY where SF18 CONFIRMS real danger in that direction; elsewhere
    # keep our current KS (SF18 says nothing to add -> don't move it, so controls stay put by construction and the
    # loss isolates the KS-attributable gap rather than the whole static-vs-search difference).
    if abs(sf18) >= 0.75 and sgn(sf11_ks) == sgn(sf18):
        target = sf11_ks
        if sf11_ks != 0 and abs(sf18) > abs(sf11_tot):    # SF18 stronger -> aspire beyond SF11 (bounded)
            target = sf11_ks * min(ASPIRE_CAP, abs(sf18) / max(abs(sf11_tot), 0.5))
    else:
        target = our_ks    # SF18 does not confirm KS danger here -> leave KS unchanged
    # tier
    if abs(sf11_ks) >= 1.0 and abs(our_ks) < 0.3:            tier = "target"
    elif abs(sf11_ks) >= 0.5 and abs(our_ks - sf11_ks) < 0.5: tier = "working"
    elif kz >= 3 and abs(sf18) < 0.75:                        tier = "crowded_safe"
    elif abs(sf11_ks) < 0.3:                                  tier = "calm"
    else:                                                     tier = "other"
    if tier == "other":                                       # non-golden, non-control -> noisy (drop from joint fit)
        continue
    split = "val" if (hash(r["fen"]) % 5 == 0) else "train"   # ~20% held out
    # JOINT fit toward TRUTH: target_total = SF18 (win% space). On GOLDEN targets SF18 ~ SF11-static (statically
    # achievable); on CONTROLS our_total ~ SF18 so the target says "don't change". KS + OvD knobs fit TOGETHER
    # (anti double-count) via the worker's FULL-total win%. target_ks kept for the side-direction metric only.
    target_total = sf18
    corpus.append({"fen": r["fen"], "target_ks": round(target, 3), "target_total": round(target_total, 3),
                   "our_total_base": round(our_total, 3), "our_ks_base": round(our_ks, 3), "tier": tier,
                   "phase_bucket": phase_bucket(r["phase_score"]), "split": split})

with open(CORPUS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fen", "target_ks", "target_total", "our_total_base", "our_ks_base",
                                      "tier", "phase_bucket", "split"])
    w.writeheader(); w.writerows(corpus)
from collections import Counter
print("fit corpus: %d rows -> %s" % (len(corpus), CORPUS))
print("tiers:", dict(Counter(c["tier"] for c in corpus)))
print("phases:", dict(Counter(c["phase_bucket"] for c in corpus)))
print("splits:", dict(Counter(c["split"] for c in corpus)))

# ---- coordinate descent over knobs ----
GRID = {
    "KS_OVERLOAD":     [0, 1, 2, 3, 5],  # per-square attackers-defenders (the discriminative breakthrough signal)
    "KS_MIN_ATTACKERS": [0, 1, 2, 3],   # Ethereal-style discrimination gate
    "KS_ATT_PRODUCT":  [0, 1, 2, 4],    # SF-style coordination product (constrained fit finds any SAFE amount)
    "IMBALANCE_SCALE": [1, 2, 3, 4, 5], # OvD offense-vs-defense magnitude (JOINT: anti double-count with KS)
    "SCALE_ATTACK_LAYER": [50, 75, 100, 125],  # king-zone attack-map scale (OvD input)
    "KS_FLOOR":        [0, 3, 6, 9, 13],
    "KS_CAP":          [80, 150, 300, 600],
    "KS_KNEE":         [12, 24, 40, 80],
    "KS_DIVISOR":      [2, 3, 4, 6],
    "KS_SAFE_CHECK":   [3, 8, 15, 25, 40],
    "KS_WEAK":         [2, 6, 12, 20],
    "KS_ATTACK_COUNT": [0, 1, 2, 4],
    "KS_ATT_KNIGHT":   [2, 5, 9, 16],
    "KS_ATT_BISHOP":   [2, 5, 9, 16],
    "KS_ATT_ROOK":     [3, 6, 10, 18],
    "KS_ATT_QUEEN":    [5, 10, 16, 24],
    "KING_SAFETY_MAG": [3000, 4500, 6000, 9000],   # MORE RANGE (user #3): does extra magnitude help or overshoot?
}
cur = {"KS_OVERLOAD": 0, "KS_MIN_ATTACKERS": 0, "KS_ATT_PRODUCT": 0, "IMBALANCE_SCALE": 3, "SCALE_ATTACK_LAYER": 100,
       "KS_FLOOR": 13, "KS_CAP": 80, "KS_KNEE": 12, "KS_DIVISOR": 4, "KS_SAFE_CHECK": 3, "KS_WEAK": 2,
       "KS_ATTACK_COUNT": 1, "KS_ATT_KNIGHT": 2, "KS_ATT_BISHOP": 2, "KS_ATT_ROOK": 3, "KS_ATT_QUEEN": 5,
       "KING_SAFETY_MAG": 3000}

# CONSTRAINED objective: minimise TARGET error SUBJECT TO the control tiers not regressing beyond tolerance.
# An unweighted total lets the many targets buy improvement by paying with calm/working -> the past failure.
GUARDS = ["calm", "working", "crowded_safe"]
TOL = float(os.environ.get('GUARD_TOL', '2.0'))   # per-guard-tier win%²-MSE allowed to rise above baseline

def evaluate(cfg):
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%d" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    d = {}
    for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out):
        d[k] = float(v)
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    if m: d["ALL.train"], d["ALL.val"] = float(m.group(1)), float(m.group(2))
    return d

base_d = evaluate(cur)
def g(d, key): return d.get(key, 9e9)
base_guard = {t: g(base_d, t + ".train") for t in GUARDS}
best_tgt = g(base_d, "target.train")
print("\nbaseline: target.train=%.3f  guards(train)=%s  ALL.val=%.3f" %
      (best_tgt, {t: round(base_guard[t], 3) for t in GUARDS}, g(base_d, "ALL.val")))

def guards_ok(d):
    return all(g(d, t + ".train") <= base_guard[t] + TOL for t in GUARDS)

for rnd in range(ROUNDS):
    print("\n=== round %d ===" % (rnd + 1))
    for knob, cands in GRID.items():
        base = cur[knob]
        for v in cands:
            if v == cur[knob]: continue
            trial = dict(cur); trial[knob] = v
            d = evaluate(trial)
            # accept only if TARGET improves AND every guard tier stays within tolerance of baseline
            if g(d, "target.train") < best_tgt - 1e-4 and guards_ok(d):
                best_tgt, cur[knob] = g(d, "target.train"), v
        if cur[knob] != base:
            print("  %-16s %s -> %s   (target.train=%.3f)" % (knob, base, cur[knob], best_tgt))

final = evaluate(cur)
print("\nBEST CONFIG:", " ".join("%s=%d" % (k, v) for k, v in cur.items()))
print("target.train=%.3f (baseline %.3f)  guards(train)=%s  ALL.val=%.3f" % (
    g(final, "target.train"), g(base_d, "target.train"),
    {t: round(g(final, t + ".train"), 3) for t in GUARDS}, g(final, "ALL.val")))
