# -*- coding: utf-8 -*-
"""The BOTTOM-UP verification metric: does our KS `units` sum DISCRIMINATE genuine danger from proximity?
For each config, evaluate raw attack-units (KS_FLOOR=0 so nothing is zeroed) on the SF-labeled ks_sts_corpus
and report, over the attack (SF genuine danger) vs quiet_neg (SF-quiet, pieces-near-king) tiers:
  - mean units per tier,
  - the separation AUC (does attack rank above quiet_neg),
  - the fraction of GENUINE-danger positions still stuck at LOW units (<13, the old floor) = the detection gap.

A detector UPGRADE (pins, weak-square value-coupling, ...) must RAISE the AUC and/or lift the low-unit genuine
fraction. This is the leading indicator that survives the dead linear curve (move-regret does NOT — honest inputs
are move-neutral until compounding). Guardrails (move-regret, symmetry, byte-id) are checked separately.

  pyrun diagnostics/_ks_discrimination.py
"""
import os, sys, csv, subprocess

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS); PY = sys.executable
CORPUS = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")

if os.environ.get("WORKER") == "1":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KS_FLOOR'] = '0'   # raw units visible (see the true signal, not the floored one)
    sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
    import chess
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)
    out = open(os.environ["OUT"], "w", newline=""); w = csv.writer(out)
    for r in csv.DictReader(open(CORPUS, newline="")):
        t = r.get("tier"); fen = r.get("fen")
        if t not in ("attack", "quiet_neg") or not fen:
            continue
        try:
            bd = ai.ev_breakdown(chess.Board(fen))
            u = max(abs(bd.get("det_ks_units_w", 0)), abs(bd.get("det_ks_units_b", 0)))
            w.writerow([t, u])
        except Exception:
            pass
    out.close(); sys.exit(0)


def auc(pos, neg):
    if not pos or not neg: return 0.5
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b: wins += 1
            elif a == b: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# Config list — extend as detector upgrades are built (e.g. KS_PIN_MODE=1). Each must raise AUC / lift low-genuine.
CONFIGS = [
    ("base (defaware1)", {"KS_DEFAWARE_MODE": 1}),
    ("+SQC", {"KS_DEFAWARE_MODE": 1, "KS_SQC_MODE": 1}),
    ("+pins", {"KS_DEFAWARE_MODE": 1, "KS_PIN_MODE": 1}),
    ("+SQC+pins", {"KS_DEFAWARE_MODE": 1, "KS_SQC_MODE": 1, "KS_PIN_MODE": 1}),
    ("stack (no flank)", {"KS_DEFAWARE_MODE": 1, "KS_SQC_MODE": 1, "KS_PIN_MODE": 1, "KS_WEAK_VAL_MODE": 1}),
    ("+flank1 (SF raw breadth)", {"KS_DEFAWARE_MODE": 1, "KS_SQC_MODE": 1, "KS_PIN_MODE": 1,
                                   "KS_WEAK_VAL_MODE": 1, "KS_FLANK_MODE": 1}),
    ("+flank2 (OURS contest-weighted)", {"KS_DEFAWARE_MODE": 1, "KS_SQC_MODE": 1, "KS_PIN_MODE": 1,
                                          "KS_WEAK_VAL_MODE": 1, "KS_FLANK_MODE": 2}),
]

print("KS unit DISCRIMINATION (attack vs quiet_neg, raw units KS_FLOOR=0)\n")
print("  %-26s %10s %10s   %6s   %-s" % ("config", "attack_mu", "quiet_mu", "AUC", "genuine<13 (detection gap)"))
for label, knobs in CONFIGS:
    outp = "/tmp/_disc_%s.csv" % abs(hash(label))
    env = dict(os.environ, WORKER="1", OUT=outp, **{k: str(v) for k, v in knobs.items()})
    subprocess.run([PY, "-u", os.path.abspath(__file__)], cwd=ENGINE, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    A, Q = [], []
    for row in csv.reader(open(outp, newline="")):
        (A if row[0] == "attack" else Q).append(float(row[1]))
    lowg = sum(1 for u in A if u < 13) / max(1, len(A))
    print("  %-26s %10.2f %10.2f   %6.3f   %.0f%% of %d" %
          (label, (sum(A)/max(1,len(A))), (sum(Q)/max(1,len(Q))), auc(A, Q), 100*lowg, len(A)))
print("\n  read: AUC = can our units RANK genuine danger above proximity. 'genuine<13' = SF-real attacks our\n"
      "  detector leaves in the old deadzone = the detection gap a better DETECTOR (not a weight) must close.", flush=True)
