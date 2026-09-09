# -*- coding: utf-8 -*-
"""GO/NO-GO for the 'consolidate-then-compound' KS thesis: is the king-attack credit counted through MULTIPLE
COLLINEAR channels at linear order? Ablate each king-credit channel one at a time on attacked-king positions,
take each channel's per-position CONTRIBUTION (base_total - ablated_total), and correlate them across positions.

  - HIGH pairwise correlation (they rise/fall together) => the channels DUPLICATE one underlying proximity signal
    => squaring `units` would amplify quadruple-counted noise => the thesis holds => GREEN LIGHT for consolidation.
  - LOW correlation (independent) => the channels carry DISTINCT information => de-dup would shed signal => STOP,
    bank the +15 bundle.

Channels (ablation knob): ch1 unit-KS (KING_SAFETY_MAG=0) · ch3 attackingLayer king-slice (KS_ZONE_ATTACK_PCT=0)
· ch4 OvD imbalance (IMBALANCE_SCALE=0). Deterministic, no SF, no games.

  pyrun diagnostics/_ks_channel_collinearity.py [N=400]
"""
import os, sys, csv, subprocess, math

for _a in sys.argv[1:]:
    if '=' in _a:
        k, v = _a.split('=', 1); os.environ[k] = v

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
PY = sys.executable


def load_fens(N):
    fens, seen = [], set()
    cp = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
    if os.path.exists(cp):
        for r in csv.DictReader(open(cp, newline="")):
            if r.get("ks_class") == "ks_attack":
                f = (r.get("decision_fen") or "").strip()
                if f and f not in seen: seen.add(f); fens.append(f)
    gp = os.path.join(THIS, "ks_sets", "game_regret_set.csv")
    if os.path.exists(gp):
        for i, r in enumerate(csv.DictReader(open(gp, newline=""))):
            if i % 23 == 0:
                f = (r.get("fen") or "").strip()
                if f and f not in seen: seen.add(f); fens.append(f)
    return fens[:N]


if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    out = open(os.environ["OUT"], "w", newline=""); w = csv.writer(out)
    for f in load_fens(int(os.environ.get("N", "400"))):
        try:
            w.writerow([f, ai.ev_breakdown(chess.Board(f)).get("total", 0)])
        except Exception:
            pass
    out.close(); sys.exit(0)

# MODE=heat (default) targets the attackingLayer CONSUMERS -- the question "does the heat table soak up
# variance that PSTs/placement should explain, i.e. does it hurt the TUNABILITY of everything it touches".
# MODE=ks reproduces the original king-channel set.
# 🐛 The original ch4 ablated with IMBALANCE_SCALE=0, which only affects OvD MODE 0 and has been INERT since
# OVD_BOUNDED_MODE=2 shipped -- that channel silently contributed nothing to every earlier run of this tool.
# OVD_CAP=0 is the correct ablation for the shipped mode.
if os.environ.get("MODE", "heat") == "ks":
    CONFIGS = {"base": {}, "ch1_off": {"KING_SAFETY_MAG": 0},
               "ch2_off": {"KS_ZONE_ATTACK_PCT": 0}, "ch3_off": {"OVD_CAP": 0}}
    LABELS = [("ch1", "unit-KS (KING_SAFETY_MAG)"),
              ("ch2", "attackingLayer king slice (KS_ZONE_ATTACK_PCT)"),
              ("ch3", "OvD imbalance (OVD_CAP)")]
else:
    CONFIGS = {"base": {}, "ch1_off": {"SCALE_ATTACK_LAYER": 0},
               "ch2_off": {"SCALE_CENTRAL": 0}, "ch3_off": {"OVD_CAP": 0}}
    LABELS = [("ch1", "ALL attackingLayer heat (SCALE_ATTACK_LAYER)"),
              ("ch2", "central re-sum (SCALE_CENTRAL)"),
              ("ch3", "OvD imbalance (OVD_CAP)")]
tot = {}
for tag, knobs in CONFIGS.items():
    outp = "/tmp/_col_%d_%s.csv" % (os.getpid(), tag)  # PID-unique: concurrent runs must not share /tmp
    env = dict(os.environ, WORKER="1", OUT=outp, **{k: str(v) for k, v in knobs.items()})
    subprocess.run([PY, "-u", os.path.abspath(__file__)], cwd=ENGINE, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    tot[tag] = {r[0]: float(r[1]) for r in csv.reader(open(outp, newline=""))}

fens = [f for f in tot["base"] if all(f in tot[t] for t in CONFIGS)]
# Per-channel contribution = base - ablated. Restrict to positions where unit-KS actually fires (|d1|>0),
# i.e. a king is genuinely under attack — the regime the thesis is about.
D = {"ch1": [], "ch2": [], "ch3": []}
for f in fens:
    d1 = tot["base"][f] - tot["ch1_off"][f]
    d2 = tot["base"][f] - tot["ch2_off"][f]
    d3 = tot["base"][f] - tot["ch3_off"][f]
    if abs(d1) < 1:   # gate channel inert here -> not a position this question is about
        continue
    D["ch1"].append(d1); D["ch2"].append(d2); D["ch3"].append(d3)


def pear(a, b):
    n = len(a)
    if n < 3: return float("nan")
    ma, mb = sum(a)/n, sum(b)/n
    va = sum((x-ma)**2 for x in a); vb = sum((x-mb)**2 for x in b)
    if va <= 0 or vb <= 0: return float("nan")
    cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    return cov / math.sqrt(va*vb)


n = len(D["ch1"])
print("channel collinearity (MODE=%s) over %d positions  (contribution = base_total - channel_off)\n"
      % (os.environ.get("MODE", "heat"), n))
print("  channel                                       mean|contribution| (mp)")
for c, lbl in LABELS:
    v = D[c]
    print("  %-45s %8.1f" % (lbl, sum(abs(x) for x in v)/max(1, len(v))))
print("\n  pairwise correlation of per-position contributions (collinearity):")
print("    %s x %s: %+.3f" % (LABELS[0][1][:28], LABELS[1][1][:24], pear(D["ch1"], D["ch2"])))
print("    %s x %s: %+.3f" % (LABELS[0][1][:28], LABELS[2][1][:24], pear(D["ch1"], D["ch3"])))
print("    %s x %s: %+.3f" % (LABELS[1][1][:28], LABELS[2][1][:24], pear(D["ch2"], D["ch3"])))
print("\n  read: HIGH positive correlations (~>0.5) => the channels DUPLICATE one proximity signal => consolidate-\n"
      "  then-compound thesis HOLDS (green light). LOW/mixed => channels carry distinct info => de-dup sheds\n"
      "  signal => STOP + bank the bundle. Also weigh the magnitudes: a tiny-magnitude channel can't be the noise.", flush=True)
