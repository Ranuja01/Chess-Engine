# -*- coding: utf-8 -*-
"""KS channel DECOMPOSITION: how much does each KS-adjacent term FIRE on the over-read collapses vs a
control, and how much does each channel-ablation reduce it? Makes the king-zone triple-count visible (the
same king pressure showing up in king_safety AND the O/D imbalance AND central), so we can see which channel
is the noise -- the OvD-style "understand it before you bound it" pass. Deterministic, no Stockfish.

Metric per set = mean firing magnitude of three breakdown components:
  KS  = |king_safety|                       (Channel 1, the sharp short-term term)
  OvD = |imbalance_white| + |imbalance_black|(Channel 3, O/D king-slice leaks in here)
  CEN = |central|                           (central slice; king-zone attackingLayer leaks here too)

Sets (collapse_dataset_classified.csv): OVER = ks_class 'ks_attack' (our KS over-reads), CTRL = 'positional'.
Configs ablate one channel each; compare firing vs baseline to see each channel's contribution, and whether
it DISCRIMINATES (fires on OVER but not CTRL) or fires indiscriminately.

  pyrun diagnostics/_ks_channel_decomp.py [FAMILY=all] [N=0]
"""
import os, sys, csv, subprocess

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)


def load_sets():
    p = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
    fam = os.environ.get("FAMILY", "all")
    over, ctrl = [], []
    seen = set()
    for r in csv.DictReader(open(p, newline="")):
        if fam != "all" and r.get("family") != fam:
            continue
        f = (r.get("decision_fen") or "").strip()
        if not f or f in seen:
            continue
        seen.add(f)
        if r.get("ks_class") == "ks_attack":
            over.append(f)
        elif r.get("ks_class") == "positional":
            ctrl.append(f)
    return over, ctrl


# ---------------- WORKER ----------------
if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    over, ctrl = load_sets()
    N = int(os.environ.get("N", "0"))
    if N:
        over, ctrl = over[:N], ctrl[:N]

    def firing(fens):
        ks = ovd = cen = 0.0; n = 0
        for f in fens:
            try:
                bd = ai.ev_breakdown(chess.Board(f))
            except Exception:
                continue
            ks += abs(bd.get("king_safety", 0.0)) / 1000.0
            ovd += (abs(bd.get("imbalance_white", 0.0)) + abs(bd.get("imbalance_black", 0.0))) / 1000.0
            cen += abs(bd.get("central", 0.0)) / 1000.0
            n += 1
        n = max(1, n)
        return ks / n, ovd / n, cen / n, n

    ok, oo, oc, no = firing(over)
    ck, cv, cc, nc = firing(ctrl)
    print("RES over_ks=%.3f over_ovd=%.3f over_cen=%.3f n_over=%d ctrl_ks=%.3f ctrl_ovd=%.3f ctrl_cen=%.3f n_ctrl=%d"
          % (ok, oo, oc, no, ck, cv, cc, nc))
    sys.exit(0)

# ---------------- DRIVER ----------------
PY = sys.executable
CONFIGS = [
    ("baseline", {}),
    ("deking (KS_ZONE_ATTACK_PCT=0)", {"KS_ZONE_ATTACK_PCT": 0}),
    ("ch1 off (KING_SAFETY_MAG=0)", {"KING_SAFETY_MAG": 0}),
    ("shelter off (KS_SHELTER_MAG=0)", {"KS_SHELTER_MAG": 0}),
    ("ovd off (IMBALANCE_SCALE=0)", {"IMBALANCE_SCALE": 0}),
    ("central off (SCALE_CENTRAL=0)", {"SCALE_CENTRAL": 0}),
]


def run(cfg):
    env = dict(os.environ, WORKER="1")
    args = [PY, "-u", os.path.abspath(__file__)] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE, env=env).stdout
    for line in out.splitlines():
        if line.startswith("RES"):
            return dict(tok.split("=") for tok in line.split()[1:])
    return {}


rows = []
base = None
print("  firing magnitude (pawns) — mean |component| per position\n", flush=True)
print("  %-32s %7s %7s %7s   %7s %7s %7s" % ("config", "O:KS", "O:OvD", "O:CEN", "C:KS", "C:OvD", "C:CEN"), flush=True)
for label, cfg in CONFIGS:
    d = run(cfg)
    if not d:
        print("  %-32s (no result)" % label); continue
    if base is None:
        base = d
    print("  %-32s %7s %7s %7s   %7s %7s %7s" % (
        label, d["over_ks"], d["over_ovd"], d["over_cen"], d["ctrl_ks"], d["ctrl_ovd"], d["ctrl_cen"]), flush=True)
print("\n  n_over=%s  n_ctrl=%s" % (base.get("n_over", "?"), base.get("n_ctrl", "?")), flush=True)
print("  read: baseline row = how much each term fires. An ablation's DROP vs baseline = that channel's"
      "\n  contribution. A channel that fires high on O (over-read) but the same on C fires indiscriminately;"
      "\n  one whose de-king drops O:OvD / O:CEN reveals the king-zone triple-count leaking across terms.", flush=True)
