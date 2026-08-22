# -*- coding: utf-8 -*-
"""Build the combined collapse+safe labelled FENS file for the KS precision/recall gate
(_ks_specificity.py / _ks_eg_specificity.py). Collapse group = ks_class==ks_attack decision_fens
(label 'ksatk'); safe group = the SF18-safe midgame control set (label 'safe').

  pyrun diagnostics/_build_p2_specset.py
Writes ks_sets/_p2_specset.txt (label<TAB>fen per line).
"""
import os, csv
THIS = os.path.dirname(os.path.abspath(__file__))
KS = os.path.join(THIS, "ks_sets")
out = os.path.join(KS, "_p2_specset.txt")

rows = []
with open(os.path.join(KS, "collapse_dataset_classified.csv")) as f:
    for r in csv.DictReader(f):
        if r.get("ks_class") == "ks_attack":
            fen = (r.get("decision_fen") or "").strip()
            if fen:
                rows.append(("ksatk", fen))
n_atk = len(rows)

with open(os.path.join(KS, "control_sf18safe.txt")) as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            rows.append(("safe", ln))
n_safe = len(rows) - n_atk

with open(out, "w") as f:
    for lbl, fen in rows:
        f.write("%s\t%s\n" % (lbl, fen))

print("wrote %s : %d ksatk (collapse) + %d safe = %d lines" % (out, n_atk, n_safe, len(rows)))
