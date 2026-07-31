# -*- coding: utf-8 -*-
"""Two-stage BENCH-GUARDED tuner: the corpus win%-MSE proposes, the REAL benches dispose.

Why: the corpus fit is BLIND to move choice, so it rejected de-king (real STS +92) and selected KS_CHK cuts that
measurably HURT STS -- i.e. every fit silently discarded what we learned on the benches. The `sts_guard` corpus
tier is only SF18-labeled STS positions scored by win%-MSE; it is NOT the STS move-choice score.

Design (deliberate): the bench is a GUARD, not the objective -- STS is deterministic but JAGGED, so maximizing it
would just overfit the bench. Stage 1 ranks candidates by corpus win%-MSE (smooth gradient). Stage 2 runs the
REAL STS (+ optional WAC) on the top-K and REJECTS any candidate that regresses a bench below baseline - TOL.
The surviving best-corpus candidate wins, so bench knowledge can never be thrown away by a later fit.

  pyrun diagnostics/fit_bench_guarded.py [TOPK=6] [STS_TOL=0] [WAC_TOL=2] [CORPUS=...]
Candidate list: edit CANDIDATES below (name -> knob dict). Baseline = all-identity.
"""
import os, sys, subprocess, re
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
RUNNER = os.path.join(ENGINE, "selfplay", "overnight_runner.sh")
PY = sys.executable
TOPK = int(os.environ.get("TOPK", "6"))
STS_TOL = float(os.environ.get("STS_TOL", "0"))     # allowed STS drop vs baseline (0 = none)
WAC_TOL = float(os.environ.get("WAC_TOL", "2"))     # WAC is jagged/tactical -> small slack
CORPUS = os.environ.get("CORPUS", os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"))
WORKER = os.path.join(THIS, "_ks_fit_eval.py")

# Candidate configs to screen (knob dicts). Baseline is implicit (empty = all identity).
CANDIDATES = {
    "baseline":            {},
    "deking50":            {"KS_ZONE_ATTACK_PCT": 50},
    "deking75":            {"KS_ZONE_ATTACK_PCT": 75},
    "deking50_v3":         {"KS_ZONE_ATTACK_PCT": 50, "ENABLE_PASSER_V3": 1},
    "deking50_v3_floor":   {"KS_ZONE_ATTACK_PCT": 50, "ENABLE_PASSER_V3": 1,
                            "PASSER_RFLOOR_R5": 100, "PASSER_RFLOOR_R6": 130},
    "deking50_ksv2_mag25": {"KS_ZONE_ATTACK_PCT": 50, "ENABLE_KS_CHECK_V2": 1, "KING_SAFETY_MAG": 2500},
    "ksv2_only":           {"ENABLE_KS_CHECK_V2": 1},
}


def knobstr(cfg):
    return " ".join("%s=%s" % (k, v) for k, v in sorted(cfg.items()))


def corpus_loss(cfg):
    """Stage 1: corpus win%-MSE (train/val) + per-tier, via the shared worker."""
    args = [PY, WORKER, "CORPUS=" + CORPUS] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    m = re.search(r"train_mse=([\d.]+) val_mse=([\d.]+)", out)
    tiers = {k: float(v) for k, v in re.findall(r"(\w+\.\w+)=([\d.]+)", out)}
    return (float(m.group(1)), float(m.group(2)), tiers) if m else (9e9, 9e9, tiers)


def _runner(sub, tag, cfg):
    # runs INSIDE wsl (pyrun) -> call the runner with bash directly; knobs as separate KEY=VAL argv entries
    cmd = ["bash", RUNNER, sub, tag] + ["%s=%s" % (k, v) for k, v in sorted(cfg.items())]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def bench_sts(cfg, tag):
    m = re.search(r"STS score:\s*(\d+)", _runner("sts", tag, cfg))
    return int(m.group(1)) if m else -1


def bench_wac(cfg, tag):
    m = re.search(r"Solved (\d+)/", _runner("wac", tag, cfg))
    return int(m.group(1)) if m else -1


print("STAGE 1 — corpus win%%-MSE (proposes)   corpus=%s" % os.path.basename(CORPUS))
stage1 = []
for name, cfg in CANDIDATES.items():
    tr, va, tiers = corpus_loss(cfg)
    stage1.append((va, tr, name, cfg))
    print("  %-22s train=%9.3f  val=%9.3f   %s" % (name, tr, va, knobstr(cfg) or "(identity)"))
stage1.sort()

top = stage1[:TOPK]
print("\nSTAGE 2 — REAL benches on top-%d (disposes; guard = no regression vs baseline)" % len(top))
base_sts = bench_sts({}, "GUARD_BASE"); base_wac = bench_wac({}, "GUARD_BASE")
print("  baseline: STS=%d  WAC=%d" % (base_sts, base_wac))

survivors = []
for va, tr, name, cfg in top:
    if name == "baseline":
        continue
    s = bench_sts(cfg, "G_" + name.upper()); w = bench_wac(cfg, "G_" + name.upper())
    ok = (s >= base_sts - STS_TOL) and (w >= base_wac - WAC_TOL)
    print("  %-22s STS=%4d (%+d)  WAC=%3d (%+d)  corpus_val=%8.3f  -> %s"
          % (name, s, s - base_sts, w, w - base_wac, va, "PASS" if ok else "REJECT"))
    if ok:
        survivors.append((va, name, cfg, s, w))

print("\nRESULT")
if not survivors:
    print("  no candidate passed the bench guard -> keep baseline")
else:
    survivors.sort()
    va, name, cfg, s, w = survivors[0]
    print("  WINNER (best corpus among bench-clean): %s   STS=%d WAC=%d corpus_val=%.3f" % (name, s, w, va))
    print("  knobs: %s" % (knobstr(cfg) or "(identity)"))
    print("  all bench-clean candidates:")
    for va2, n2, c2, s2, w2 in survivors:
        print("    %-22s corpus_val=%8.3f  STS=%4d  WAC=%3d" % (n2, va2, s2, w2))
