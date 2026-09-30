# -*- coding: utf-8 -*-
"""POT middlegame screen against SF18 SEARCH labels, with the owner's NO-OVERLAP condition built in (C3 doc §14).

For each candidate transformation feature, on standard middlegame rows (96 <= phase <= 224) labelled by SF18 d14:
  SIGNAL   corr(feature, residual), residual = win%(SF18 search) − win%(our shipped eval), White-POV.
           Signed features (White − Black) use the raw residual; leader-relative ones ("projected winnability" inputs,
           colour-symmetric) use sign(our eval)·residual — the same convention as the eg winnability term.
  OVERLAP  max |corr| of the feature with each of the existing C1 feature columns (fitC_features.npz diff: mobility,
           pawn structure, passers/candidates, placement) — a feature that mostly re-describes an owned concept is out.
Candidates: lever_now, tension_centre, mobile_majority (from _ovd_lever_proto) and the winnability inputs read in the
middlegame (pawns, both_flanks, pawn count on one flank only, passed, outflanking).

  pyrun diagnostics/_pot_mg_screen.py [LABELS=ks_sets/fitC_mg_sf18.csv] [SAMPLE=ks_sets/fitC_mg_sample.csv]
"""
import os, sys, csv, math
import numpy as np

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
LABELS = os.path.join(THIS, KV.get("LABELS", "ks_sets/fitC_mg_sf18.csv"))
SAMPLE = os.path.join(THIS, KV.get("SAMPLE", "ks_sets/fitC_mg_sample.csv"))
import _ovd_lever_proto as P
K_WIN = 0.00368208


def winpct(cp):
    return 100.0 / (1.0 + np.exp(-K_WIN * np.clip(cp, -1500, 1500)))


def main():
    sample = {r["fen"]: int(r["row"]) for r in csv.DictReader(open(SAMPLE, newline="")) if r["src"] == "std"}
    lab = [(r["fen"], float(r["best_cp"])) for r in csv.DictReader(open(LABELS, newline=""))
           if r.get("best_cp") and r["fen"] in sample]
    zw = np.load(os.path.join(DATA, "fitC_win.npz"))
    zf = np.load(os.path.join(DATA, "fitC_features.npz"))
    fens = [f for f, _ in lab]
    rows = np.array([sample[f] for f in fens])
    sf = np.array([c for _, c in lab])
    T = zw["total"][rows].astype(np.float64)                  # shipped total, Black-positive mp
    ph = zw["phase"][rows].astype(np.float64)
    ok = (ph >= 0) & (np.abs(T) < 30000) & (np.abs(sf) < 50000)
    rows, sf, T, ph = rows[ok], sf[ok], T[ok], ph[ok]
    fens = [f for f, k in zip(fens, ok) if k]
    ours_cp = -T / 10.0
    res = winpct(sf) - winpct(ours_cp)                      # White-POV residual
    lead = np.sign(ours_cp)
    print("rows %d  mean|residual| %.2f pp" % (len(res), np.abs(res).mean()))

    feats, kinds = {}, {}
    ln, lc, mm = np.zeros(len(fens)), np.zeros(len(fens)), np.zeros(len(fens))
    win_in = np.array(zw["inputs"][rows], dtype=np.float64)   # passed, pawns, outflanking, infiltration, both, pawn_end, almost
    for i, fen in enumerate(fens):
        wp, bp, occ = P.board_bits(fen)
        wmg = ph[i] / 256.0
        base = P.rel(wp, bp, wmg)
        ln[i] = P.side_options(wp, bp, occ, True, wmg, base)[0] - P.side_options(wp, bp, occ, False, wmg, base)[0]
        lc[i] = (P.side_options(wp, bp, occ, True, wmg, base, True)[0]
                 - P.side_options(wp, bp, occ, False, wmg, base, True)[0])
        mm[i] = P.mobile_majority(wp, bp, occ)
    for name, v in (("lever_now", ln), ("tension_centre", lc), ("mobile_majority", mm)):
        feats[name], kinds[name] = v, "signed"
    for j, name in enumerate(["passed", "pawns", "outflanking", "infiltration", "both_flanks"]):
        feats["proj_" + name], kinds["proj_" + name] = win_in[:, j], "leader"

    C1 = zf["diff"][rows, :106].astype(np.float64)            # existing features (Black − White counts)
    C1 = C1[:, C1.std(axis=0) > 0]
    print("\n  %-22s %-7s %7s %8s %6s   %s" % ("feature", "kind", "fires", "r", "σ", "overlap: max|corr| with existing"))
    for name, v in feats.items():
        m = v != 0 if kinds[name] == "signed" else np.ones(len(v), bool)
        y = res[m] if kinds[name] == "signed" else (lead * res)[m]
        x = v[m]
        if m.sum() < 50 or x.std() == 0:
            print("  %-22s %-7s %6.1f%%   (too few / constant)" % (name, kinds[name], 100 * m.mean()))
            continue
        r = np.corrcoef(x, y)[0, 1]
        se = 1 / math.sqrt(m.sum() - 3)
        ov = np.nanmax(np.abs([np.corrcoef(v, C1[:, k])[0, 1] for k in range(C1.shape[1])]))
        print("  %-22s %-7s %6.1f%% %+8.4f %6.1f   %.2f%s" % (name, kinds[name], 100 * m.mean(), r, r / se, ov,
                                                          "  ⚠️ OVERLAP" if ov > 0.6 else ""))
    print("\n  read: |σ| >= 3 AND overlap < 0.6 => a candidate worth building (then incremental held-out + games).")


if __name__ == "__main__":
    main()
