# -*- coding: utf-8 -*-
"""Characterize the pt_pawns +172 over-read (the dominant `pieces` hole): missing-vs-tuning vs SF11, phase, driver.

Reads verify_triage_static --dump corpus (needs our_pawn_cp, sf11_pawns_cp, sf11_passed_cp, pt_pawns, pawn_adv,
pawn_lead, is_endgame). All mover-POV cp.
  1) our pawn eval vs SF11 (Pawns+Passed): corr + slope => missing detector vs tuning (same logic as ks_gap).
  2) pt_pawns over-read (collapse vs control) split endgame vs midgame => is it endgame-only or broad?
  3) within collapse, corr(pt_pawns, pawn_adv) and corr(pt_pawns, pawn_lead) => advancement- or material-driven?

  python diagnostics/pawn_gap.py diagnostics/corpus_ks.csv
"""
import sys, csv
import numpy as np


def col(rows, k):
    return np.array([float(r[k]) for r in rows if r.get(k) not in ("", None)])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    rows = list(csv.DictReader(open(path)))
    coll = [r for r in rows if r["is_collapse"] == "1"]
    ctrl = [r for r in rows if r["is_collapse"] == "0"]

    # 1) missing vs tuning: our pawn eval vs SF11 (Pawns+Passed).
    print("=== 1) our pawn eval vs SF11 (Pawns+Passed), COLLAPSE ===")
    pair = [(float(r["our_pawn_cp"]), float(r["sf11_pawns_cp"]) + float(r["sf11_passed_cp"]))
            for r in coll if r.get("sf11_pawns_cp") not in ("", None) and r.get("sf11_passed_cp") not in ("", None)]
    if len(pair) >= 5:
        o = np.array([p[0] for p in pair]); s = np.array([p[1] for p in pair])
        print(f"  n={len(o)}  mean our_pawn={o.mean():+.0f}cp  mean sf11_pawn={s.mean():+.0f}cp")
        print(f"  corr={np.corrcoef(o,s)[0,1]:+.2f}  slope our~sf11={np.polyfit(s,o,1)[0]:+.2f}"
              f"   (high corr+slope>1 => we OVER-weight what SF also sees = TUNING; low corr => different object)")

    # 2) phase split.
    print("\n=== 2) pt_pawns over-read (collapse - control), by phase ===")
    for lbl, cf, kf in [("ALL", coll, ctrl),
                        ("ENDGAME", [r for r in coll if r["is_endgame"] == "1"], [r for r in ctrl if r["is_endgame"] == "1"]),
                        ("MIDGAME", [r for r in coll if r["is_endgame"] == "0"], [r for r in ctrl if r["is_endgame"] == "0"])]:
        cm = col(cf, "pt_pawns").mean() if cf else 0; km = col(kf, "pt_pawns").mean() if kf else 0
        print(f"  {lbl:>8}: collapse={cm:+.0f}  control={km:+.0f}  Δ={cm-km:+.0f}  (n_coll={len(cf)})")

    # 3) driver: advancement vs material, within collapse.
    print("\n=== 3) within COLLAPSE, what drives pt_pawns? ===")
    p = col(coll, "pt_pawns"); adv = col(coll, "pawn_adv"); lead = col(coll, "pawn_lead")
    if len(p) >= 5:
        print(f"  corr(pt_pawns, pawn_adv) ={np.corrcoef(p,adv)[0,1]:+.2f}   (advancement-driven PST over-credit?)")
        print(f"  corr(pt_pawns, pawn_lead)={np.corrcoef(p,lead)[0,1]:+.2f}   (material-lead-driven?)")
    print("\nREAD: low corr vs SF pawn eval => our pawn term computes a DIFFERENT thing (missing/wrong);")
    print("      Δ concentrated in ENDGAME => drawn-passer over-credit; high corr w/ advancement => PST over-rewards pushed pawns.")


if __name__ == "__main__":
    main()
