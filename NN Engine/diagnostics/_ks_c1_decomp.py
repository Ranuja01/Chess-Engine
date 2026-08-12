# -*- coding: utf-8 -*-
"""Channel-1 (king_safety) SUB-decomposition: split the danger term into its PROXIMITY signals (per-attacker
weights, attacked-square count, weak squares) vs its GENUINE-THREAT signals (safe-checks, storm, open files),
and measure how much each fires on the over-read collapses vs the control. The hypothesis (from the code's own
comment: safe-checks "fire on real attacks, not mere proximity"): the over-read is PROXIMITY firing on false
alarms while the genuine-threat detectors stay quiet -- i.e. a discrimination failure, not a magnitude one.

Ablate each sub-component to 0; the DROP in mean |king_safety| = that component's contribution. Read the
OVER column (false alarms): whatever contributes most there is what's over-firing. Deterministic, no SF.

  pyrun diagnostics/_ks_c1_decomp.py [FAMILY=all] [N=0]
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
    over, ctrl, seen = [], [], set()
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

    def mean_ks(fens):
        s = 0.0; n = 0
        for f in fens:
            try:
                bd = ai.ev_breakdown(chess.Board(f))
            except Exception:
                continue
            s += abs(bd.get("king_safety", 0.0)) / 1000.0; n += 1
        return s / max(1, n), n

    o, no = mean_ks(over); c, nc = mean_ks(ctrl)
    print("RES over=%.3f n_over=%d ctrl=%.3f n_ctrl=%d" % (o, no, c, nc))
    sys.exit(0)

PY = sys.executable
CONFIGS = [
    ("baseline", {}),
    ("-- PROXIMITY --", None),
    ("attacker weights (KS_ATT_*=0)", {"KS_ATT_KNIGHT": 0, "KS_ATT_BISHOP": 0, "KS_ATT_ROOK": 0, "KS_ATT_QUEEN": 0}),
    ("attacked-sq count (KS_ATTACK_COUNT=0)", {"KS_ATTACK_COUNT": 0}),
    ("weak squares (KS_WEAK=0)", {"KS_WEAK": 0}),
    ("-- GENUINE THREAT --", None),
    ("safe checks (KS_SAFE_CHECK=0)", {"KS_SAFE_CHECK": 0}),
    ("storm (KS_STORM=0)", {"KS_STORM": 0}),
    ("open files (KS_OPEN_FILE=0)", {"KS_OPEN_FILE": 0}),
]


def run(cfg):
    env = dict(os.environ, WORKER="1")
    args = [PY, "-u", os.path.abspath(__file__)] + ["%s=%s" % (k, v) for k, v in cfg.items()]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ENGINE, env=env).stdout
    for line in out.splitlines():
        if line.startswith("RES"):
            return dict(tok.split("=") for tok in line.split()[1:])
    return {}


base = None
print("  mean |king_safety| (pawns);  DROP vs baseline = that component's contribution\n", flush=True)
print("  %-40s %8s %8s   %8s %8s" % ("config", "OVER", "dropO", "CTRL", "dropC"), flush=True)
for label, cfg in CONFIGS:
    if cfg is None:
        print("  %s" % label); continue
    d = run(cfg)
    if not d:
        print("  %-40s (no result)" % label); continue
    o, c = float(d["over"]), float(d["ctrl"])
    if base is None:
        base = (o, c)
    print("  %-40s %8.3f %8.3f   %8.3f %8.3f" % (label, o, base[0] - o, c, base[1] - c), flush=True)
print("\n  n_over=%s n_ctrl=%s" % (base and d.get("n_over", "?"), d.get("n_ctrl", "?")), flush=True)
print("  read: the biggest dropO is the largest contributor to the over-read firing. If PROXIMITY dominates"
      "\n  dropO while safe-checks barely move it, the over-read is proximity-driven -> the discrimination fix"
      "\n  is to lean on genuine-threat (safe-checks) and gate/shrink proximity.", flush=True)
