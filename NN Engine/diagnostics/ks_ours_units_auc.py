# -*- coding: utf-8 -*-
"""Does OUR raw KS detector (units) rank the SF-fires 'attack' tier above the SF-quiet 'quiet_neg' tier?
Compares our units' separation AUC to SF's combined-feature AUC (0.80). If ours is much lower -> we lack
discriminating FEATURES (flank breadth); if ours is ~0.80 too -> we CAN rank them and the regression is
pure SCALING (floor too low), fixable by threshold not features.

Pure join of existing CSVs (no engine): ks_band_on (our unitsW/B per STS idx, recalibrated config),
sts_results_ksoff (idx->fen), ks_sts_corpus (fen->tier).
  pyrun diagnostics/ks_ours_units_auc.py
"""
import os, csv
THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")
CORPUS = os.path.join(THIS, "ks_sets", "ks_sts_corpus.csv")

band = {}
for r in csv.DictReader(open(os.path.join(RES, "ks_band_on.csv"))):
    band[int(r["idx"])] = max(abs(float(r["unitsW"])), abs(float(r["unitsB"])))
band_off = {}
for r in csv.DictReader(open(os.path.join(RES, "ks_band_off.csv"))):
    band_off[int(r["idx"])] = max(abs(float(r["unitsW"])), abs(float(r["unitsB"])))
idx2fen = {int(r["idx"]): r["fen"] for r in csv.DictReader(open(os.path.join(RES, "sts_results_ksoff.csv")))}
fen2tier = {r["fen"]: r["tier"] for r in csv.DictReader(open(CORPUS))}


def auc(pos, neg):
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


for label, src in (("ON-config units", band), ("OFF-config units", band_off)):
    a, q = [], []
    for idx, fen in idx2fen.items():
        t = fen2tier.get(fen)
        u = src.get(idx)
        if u is None or t not in ("attack", "quiet_neg"):
            continue
        (a if t == "attack" else q).append(u)
    am = sum(a) / len(a) if a else 0
    qm = sum(q) / len(q) if q else 0
    print("OUR %-16s  attack_mean=%6.2f (n=%d)  quiet_mean=%6.2f (n=%d)  AUC=%.3f"
          % (label, am, len(a), qm, len(q), auc(a, q)))
print("\n(SF combined-feature AUC was 0.80; best single SF signal flank_attack 0.72.)")
