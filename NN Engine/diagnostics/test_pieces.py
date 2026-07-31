# -*- coding: utf-8 -*-
"""Fable Q3 verification: is the `pieces` +177 over-read the relocated KS hole, or independent? And WHICH
piece-type over-reads?

Test A: split COLLAPSE FENs by SF11 king-danger (sf11_ks_cp <= -100 vs > -100); compare mean pieces_cp.
        concentrates in the danger subset => same hole priced in the wrong term (KS bundle drains it).
        roughly uniform => independent hole (needs its own fix). [Fable predicts ~uniform/independent.]
Test B: mean per-piece-type placement (pt_*) collapse vs control => which piece placement over-credits.
        [Fable predicts advanced/active pieces (rooks/queens / attacking-layer activity).]

  python diagnostics/test_pieces.py diagnostics/corpus_ks.csv
"""
import sys, csv
import numpy as np

PT = ["pt_pawns", "pt_knights", "pt_bishops", "pt_rooks", "pt_queens", "pt_kings"]


def m(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    return (sum(v) / len(v)) if v else 0.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "diagnostics/corpus_ks.csv"
    rows = list(csv.DictReader(open(path)))
    coll = [r for r in rows if r["is_collapse"] == "1"]
    ctrl = [r for r in rows if r["is_collapse"] == "0"]

    print("=== Test A: pieces_cp over-read, COLLAPSE split by SF11 king-danger ===")
    for lbl, sub in [("SF-danger (sf11_ks<=-100)", [r for r in coll if r.get("sf11_ks_cp") not in ("", None) and float(r["sf11_ks_cp"]) <= -100]),
                     ("no-danger (sf11_ks>-100)",  [r for r in coll if r.get("sf11_ks_cp") not in ("", None) and float(r["sf11_ks_cp"]) > -100])]:
        print(f"  {lbl:>28}: n={len(sub):>3}  mean pieces_cp={m(sub,'pieces_cp'):+.0f}  capg={m(sub,'capg_cp'):+.0f}")
    print("  READ: similar pieces_cp across the split => INDEPENDENT of king-danger (Fable's prediction).")

    print("\n=== Test B: per-piece-type placement (mover-POV cp), collapse vs control ===")
    print(f"  {'type':>10} {'collapse':>9} {'control':>9} {'Δ(over-read)':>13}")
    for k in PT + ["pieces_cp"]:
        cm, km = m(coll, k), m(ctrl, k)
        print(f"  {k:>10} {cm:>+9.0f} {km:>+9.0f} {cm-km:>+13.0f}")
    print("  READ: the piece-type with the largest Δ is where our placement over-credits in the collapse class.")


if __name__ == "__main__":
    main()
