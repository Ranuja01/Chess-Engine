# One-off: how does Fit K's KS change WHEN kings fire, and does firing separate outcomes on near-equal positions?
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/diagnostics")
import _texel_k_fit as F

D = "/mnt/e/chess_data/texel/"
st = pd.read_csv(D + "fitC_stage1.csv.gz", usecols=["result_white"])
fz = np.load(D + "fitC_features.npz")
kz = np.load(D + "fitC_ks.npz")
names = [str(x) for x in kz["names"]]
ph = kz["phase"].astype(np.float64)
full = fz["total"].astype(np.float64)
ok = ph >= 0
y = st["result_white"].values.astype(np.float64)

fit = dict(F.KS_START)
for kv in open(D + "ks_%s.txt" % (sys.argv[1] if len(sys.argv) > 1 else "fitK1")).read().split("\n")[1].split():
    k, v = kv.split("=")
    fit[k.replace("KS_V2_", "")] = float(v)
th_s = np.array([F.KS_START[k] for k in F.KS_NAMES])
th_f = np.array([fit[k] for k in F.KS_NAMES])
lin_s, lin_f = th_s[:F.NLIN], th_f[:F.NLIN]


def danger(th, Fm):
    u = np.maximum(Fm @ th[:F.NLIN], 0.0)
    return th[F.NLIN] * u * u / (u * u + th[F.NLIN + 1] ** 2)


near = ok & (np.abs(full) <= 1000)
buckets = [("opening/mg (ph>192)", ph > 192), ("middle (64-192)", (ph >= 64) & (ph <= 192)), ("endgame (<64)", (ph < 64) & ok)]
print("kings: %d positions scored, %d near-equal (|eval| <= 1 pawn)" % (ok.sum(), near.sum()))
for side, s in (("White king", 0), ("Black king", 1)):
    Fm = F.ks_design(kz["ch"][:, s], names)
    ds, df = danger(th_s, Fm), danger(th_f, Fm)
    score = y if s == 0 else 1.0 - y           # result for the king's OWN side
    print("\n== %s" % side)
    print("  %-22s %10s %10s   %-32s %-32s" % ("phase", "fire ship", "fire fit", "near-eq score fire/quiet (ship)", "near-eq score fire/quiet (fit)"))
    for nm, m in buckets:
        m = m & ok
        fs, ff = (ds > 0) & m, (df > 0) & m
        mn = m & near
        def sep(fire):
            a, b = fire & mn, ~fire & mn
            return "%.3f (n=%d) / %.3f" % (score[a].mean() if a.any() else float("nan"), a.sum(), score[b].mean() if b.any() else float("nan"))
        print("  %-22s %9.1f%% %9.1f%%   %-32s %-32s" % (nm, 100 * fs.sum() / m.sum(), 100 * ff.sum() / m.sum(), sep(ds > 0), sep(df > 0)))
    # danger magnitude vs outcome on near-equal: correlation of danger (mp) with own score
    for lab, d in (("ship", ds), ("fit", df)):
        m = near & (d > 0)
        r = np.corrcoef(d[m], score[m])[0, 1] if m.sum() > 10 else float("nan")
        print("  corr(danger, own score | near-eq & firing) %-4s r = %+.3f  (n=%d)" % (lab, r, m.sum()))
