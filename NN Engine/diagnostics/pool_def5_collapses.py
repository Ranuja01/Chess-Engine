# -*- coding: utf-8 -*-
"""Pool the CURRENT shipped-DEF-5 baseline collapses (from the ab_base_s0/s1 A/B games) into one deduped
corpus + a severity-ranked summary, so the next collapse class can be diagnosed from real data. No SF needed
here (pure pooling); SF18-search triage is the user-steered next step. Writes ks_sets/def5_baseline_collapses.txt."""
import os, sys, csv
THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "..", "selfplay", "games")
OUT = os.path.join(THIS, "ks_sets", "def5_baseline_collapses.txt")

rows = []
seen = set()
for tag in ["ab_base_s0", "ab_base_s1"]:
    cp = os.path.join(GAMES, tag, "collapses.csv")
    if not os.path.exists(cp):
        print("MISSING:", cp); continue
    for r in csv.DictReader(open(cp)):
        drop = (r.get("drop_fen") or "").strip()
        if not drop or drop in seen:
            continue
        seen.add(drop)
        try:
            peak = int(r.get("peak_eval") or 0); dropv = int(r.get("drop_eval") or 0)
        except ValueError:
            peak = dropv = 0
        rows.append(dict(tag=tag, game=r.get("game"), color=(r.get("our_color") or "").strip(),
                         result=(r.get("result") or "").strip(), peak=peak, drop=dropv,
                         swing=peak - dropv, decision=(r.get("decision_fen") or "").strip(), drop_fen=drop))

rows.sort(key=lambda x: -x["swing"])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(r["drop_fen"] + "\n")

print("pooled %d unique DEF-5 baseline collapses -> %s" % (len(rows), OUT))
print("\nworst 18 by peak->drop swing (peak_eval is our-POV millipawns at the peak, before the losing drift):")
print("%-4s %-6s %-8s %8s %8s %8s  decision_fen" % ("#", "color", "result", "peak", "drop", "swing"))
for i, r in enumerate(rows[:18]):
    print("%-4d %-6s %-8s %8d %8d %8d  %s" % (i, r["color"], r["result"], r["peak"], r["drop"], r["swing"], r["decision"]))
