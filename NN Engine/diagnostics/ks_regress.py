# -*- coding: utf-8 -*-
"""Characterize WHERE the king-safety bundle regresses STS — so the conditioner is designed from the
actual failures, not an averaged metric.

Diffs two STS result CSVs (sts_results_<base>.csv vs sts_results_<cand>.csv) into REGRESSIONS (baseline
scored higher -> KS-on picked a worse move) and IMPROVEMENTS (KS-on scored higher). For each, it reads
the live KS detectors with the CANDIDATE env (anchor + KS_ZONE2): the king_safety term (White-POV pawns),
raw det-units per king, and the `control_edge` the MOD_KS_CONTROL conditioner sees (attacker offense -
defender defense). The point: if regressions cluster at HIGH |king_safety| + LOW control_edge (term fired
with no real attack backing) while improvements have HIGH term + HIGH control_edge, then control_edge is
the separating detector and MOD_KS_CONTROL is the right damp — designed from the failures.

Run in WSL from NN Engine/ (needs the result CSVs from `sts <base>` and `sts <cand>`):
    KS_ZONE2=1 /home/ranuja/anaconda3/bin/python diagnostics/ks_regress.py base0 candA4k
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault("ENABLE_KS_REPLACE_LT", "1")
os.environ.setdefault("KING_SAFETY_MAG", "4000")
os.environ.setdefault("MOD_KS_CONTROL", "256")
os.environ.setdefault("KS_ZONE2", "1")
import sys
import csv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
RES = os.path.join(THIS_DIR, "results")

import chess  # noqa: E402


def load(tag):
    path = os.path.join(RES, "sts_results_%s.csv" % tag)
    return {r["fen"]: r for r in csv.DictReader(open(path))}


def main():
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "base0"
    cand_tag = sys.argv[2] if len(sys.argv) > 2 else "candA4k"
    base, cand = load(base_tag), load(cand_tag)
    from ChessAI import ChessAI
    ai = ChessAI(None, None, chess.Board(), True)

    def detect(fen):
        bd = ai.ev_breakdown(chess.Board(fen))
        raw = bd["king_safety"]                      # Black-positive: >=0 => White king in danger (Black attacks)
        ks = -raw / 1000.0                           # White-POV pawns
        if raw >= 0:
            ce = bd["det_b_offense"] - max(bd["det_w_defense"], 0)
        else:
            ce = bd["det_w_offense"] - max(bd["det_b_defense"], 0)
        units = bd["det_ks_units_w"] if raw >= 0 else bd["det_ks_units_b"]
        return ks, ce, units, bd["phase_score"]

    def sc(r):
        try:
            return int(r["score"])
        except Exception:
            return None

    regress, improve = [], []
    for fen, b in base.items():
        c = cand.get(fen)
        if not c:
            continue
        bs, cs = sc(b), sc(c)
        if bs is None or cs is None:
            continue
        if cs < bs:
            regress.append((fen, b, c, bs - cs))
        elif cs > bs:
            improve.append((fen, b, c, cs - bs))

    def stats(rows):
        if not rows:
            return (0, 0.0, 0.0, 0.0)
        ks_abs = ce = units = 0.0
        for fen, _, _, _ in rows:
            k, e, u, _p = detect(fen)
            ks_abs += abs(k); ce += e; units += u
        n = len(rows)
        return (n, ks_abs / n, ce / n, units / n)

    rn, rks, rce, ru = stats(regress)
    inn, iks, ice, iu = stats(improve)
    print("REGRESS n=%d  IMPROVE n=%d  net pts=%+d (regress-pts %d, improve-pts %d)"
          % (rn, inn, sum(d for *_, d in improve) - sum(d for *_, d in regress),
             sum(d for *_, d in regress), sum(d for *_, d in improve)))
    print("  signature  [mean |ks term|]  [mean control_edge]  [mean det_units on dangerous king]")
    print("  REGRESS :   %6.3f             %8.1f            %6.1f" % (rks, rce, ru))
    print("  IMPROVE :   %6.3f             %8.1f            %6.1f" % (iks, ice, iu))
    print()

    # Theme breakdown of regressions (which positional themes the over-fire breaks).
    th = {}
    for fen, b, c, d in regress:
        th[b["theme"]] = th.get(b["theme"], 0) + d
    print("regressions by theme (points lost):")
    for t, pts in sorted(th.items(), key=lambda kv: -kv[1]):
        print("  %-22s -%d" % (t, pts))
    print()

    # Worst individual regressions with their KS signature (the semantic dossier).
    print("worst regressions (baseline move -> candidate move | ks term, control_edge, units, phase):")
    for fen, b, c, d in sorted(regress, key=lambda r: -r[3])[:12]:
        k, e, u, p = detect(fen)
        print("  -%d  %-18s base=%s cand=%s | ks=%+.2f ce=%+d u=%d ph=%d | %s"
              % (d, b["theme"], b["engine"], c["engine"], k, e, u, p, fen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
