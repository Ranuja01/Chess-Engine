# -*- coding: utf-8 -*-
"""KS discrimination-unit firing screen (2026-08-11 rebalance). Reuses the _ks_c1_decomp worker
(mean |king_safety| over the 92 ks_attack OVER-read collapses vs the 171 positional CONTROLS) and
sweeps the new unit's configs. The unit is GOOD iff it DROPS the OVER firing (false alarms) while
SPARING the CTRL firing (genuine) -> selectivity = dropO - dropC should be POSITIVE and larger than
any uniform shrink. A config that drops OVER and CTRL equally is just a global magnitude move (fails
in both directions per every-eval-term-error-is-bidirectional); we want SELECTIVE.

Deterministic, no SF. Run: pyrun diagnostics/_ks_unit_screen.py
"""
import os, sys, subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
WORKER_SCRIPT = os.path.join(THIS, "_ks_c1_decomp.py")
PY = sys.executable

# label -> knob config. Item 1 (defender-aware weighting, both forms), item 2 (attackedBy2), plus a
# couple of item-3 ratio-rebalance probes (safe-check up / proximity down) layered on the winner form.
CONFIGS = [
    ("baseline (default)", {}),
    ("-- item 1: defender-aware weighting --", None),
    ("defaware1 (contested-fraction)", {"KS_DEFAWARE_MODE": 1}),
    ("defaware2 (breakthrough-count)", {"KS_DEFAWARE_MODE": 2}),
    ("defaware2 shr1", {"KS_DEFAWARE_MODE": 2, "KS_DEFAWARE_COUNT_SHR": 1}),
    ("-- item 2: attackedBy2 weak --", None),
    ("weakAtt2", {"ENABLE_KS_WEAK_ATT2": 1}),
    ("-- combined unit --", None),
    ("defaware1 + weakAtt2", {"KS_DEFAWARE_MODE": 1, "ENABLE_KS_WEAK_ATT2": 1}),
    ("-- item 3: ratio rebalance on defaware1 --", None),
    ("defaware1 + SC=8", {"KS_DEFAWARE_MODE": 1, "KS_SAFE_CHECK": 8}),
    ("defaware1 + SC=15", {"KS_DEFAWARE_MODE": 1, "KS_SAFE_CHECK": 15}),
    ("-- control: uniform proximity shrink (NOT selective) --", None),
    ("KS_ATT halved", {"KS_ATT_KNIGHT": 1, "KS_ATT_BISHOP": 1, "KS_ATT_ROOK": 2, "KS_ATT_QUEEN": 3}),
]


def run(cfg):
    env = dict(os.environ, WORKER="1")
    args = [PY, "-u", WORKER_SCRIPT] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE, env=env).stdout
    for line in out.splitlines():
        if line.startswith("RES"):
            return dict(tok.split("=") for tok in line.split()[1:])
    return {}


base = None
print("  mean |king_safety| (pawns) on OVER-reads vs CONTROLS; want dropO large, dropC ~0\n", flush=True)
print("  %-38s %7s %7s   %7s %7s   %8s" % ("config", "OVER", "dropO", "CTRL", "dropC", "select"), flush=True)
for label, cfg in CONFIGS:
    if cfg is None:
        print("  %s" % label); continue
    d = run(cfg)
    if not d:
        print("  %-38s (no result)" % label); continue
    o, c = float(d["over"]), float(d["ctrl"])
    if base is None:
        base = (o, c)
    dO, dC = base[0] - o, base[1] - c
    print("  %-38s %7.3f %7.3f   %7.3f %7.3f   %8.3f" % (label, o, dO, c, dC, dO - dC), flush=True)
print("\n  n_over=%s n_ctrl=%s" % (d.get("n_over", "?"), d.get("n_ctrl", "?")), flush=True)
print("  read: a config that drops OVER much more than CTRL (high 'select') is DISCRIMINATING; one that\n"
      "  drops both equally is a uniform shrink. Compare defaware forms against the halved-KS_ATT control.", flush=True)
