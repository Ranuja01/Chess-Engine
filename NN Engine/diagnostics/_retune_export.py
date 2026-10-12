# -*- coding: utf-8 -*-
"""Export one _retune_fit.py arm as ENGINE TABLE FILES, then prove the engine reproduces it (closure).

  MODE=tables  NPZ=fit_hybrid_l1.npz ARM='hybrid/λ1' OUT=/mnt/e/chess_data/retune/arm_h1      (needs V2_PRESET=shipped)
      Writes, each = SHIPPED value + δ (absolute values; every cell written, so loaders whose missing cells mean 0 —
      the C3 blocks — still reproduce the ship where δ is 0):
        c1.txt    C1_V2_FILE   k 0-105 (mobility · pawn struct · passers · placement)      [C1_V2_FIT=1]
        ksb.txt   KSB_V2_FILE  k 106-161                                                     [KSB_V2=1 (shipped)]
        kfl.txt   KFL_V2_FILE  k 162-171                                                     [KFL_V2=1]
        kprot.txt KPROT_V2_FILE k 172-183                                                    [KPROT_V2=1]
        px.txt    PX_V2_FILE   k 184-234                                                     [PX_V2=1]
        conn.txt  CONN_V2_FILE k 235-336                                                     [CONN_V2=1]
        kauf.txt  KAUF_V2_FILE O/T a b mp = ship_tables_v2.h cells + δ                       [KAUF_V2_FORM=3 (shipped)]
        pst.txt   PST_V2_FILE  768 ints = the shipped tapered tables (PST_V2_DUMP) + δ (file-tied)
        env.txt   the one-line knob set for this arm
      θ (the shipped start values) comes from v2_features_theta of the shipped engine; δ from the fit (mp, Black − White
      columns, adds to the Black-positive total ⇒ every White-POV table value moves by +δ).
  MODE=verify ARMDIR=… N=3000   (run with the arm's env.txt knobs AND V2_PRESET=shipped)
      Engine static total under the arm's tables vs the linear prediction ship_total + X·δ on N held-out corpus rows.
      Expected residual: table rounding (cells are integers) + the winnability scale on the eg part (the fit is linear,
      the engine scales the eg total on ~8% of rows). Reports median / p95 / p99 |residual| and the share over 10 mp.
"""
import os, sys, csv, glob, re
import numpy as np

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)
D = KV.get("D", "/mnt/e/chess_data/retune")
os.environ.setdefault("PRESET", "LONG_FORMAT"); os.environ.setdefault("USE_OPENING_BOOK", "0")
BLOCKS = [("c1", 0, 106, "C1_V2_FIT=1 C1_V2_FILE"), ("ksb", 106, 162, "KSB_V2_FILE"), ("kfl", 162, 172, "KFL_V2=1 KFL_V2_FILE"),
          ("kprot", 172, 184, "KPROT_V2=1 KPROT_V2_FILE"), ("px", 184, 235, "PX_V2=1 PX_V2_FILE"),
          ("conn", 235, 337, "CONN_V2=1 CONN_V2_FILE")]


def load_arm():
    z = np.load(os.path.join(D, KV["NPZ"]))
    labels = list(z["labels"])
    i = int(KV["ARM_I"]) if "ARM_I" in KV else labels.index(KV["ARM"])   # ARM_I=<index>: keeps 'λ' off the command line
    KV["ARM"] = labels[i]
    p = z["deltas"][i]
    names = list(z["names"])
    d = {n: float(v) for n, v in zip(names, p[:len(names)])}
    if "ONLY" in KV:
        # ONLY=c1,px,… (or none): keep δ only for those blocks, the rest at the ship — the per-block closure ablation
        want = set(KV["ONLY"].split(","))
        def blk(n):
            if n.startswith("pst"):
                return "pst"
            if n.startswith("kauf"):
                return "kauf"
            k = int(n.split(":")[1])
            if k < 106:                       # C1 sub-blocks, so one can be isolated (ONLY=c1 keeps all four)
                return "c1mob" if k < 66 else ("c1ps" if k < 77 else ("c1pass" if k < 97 else "c1place"))
            return next(b for b, lo, hi, _ in BLOCKS if lo <= k < hi)
        d = {n: v for n, v in d.items() if blk(n) in want or (blk(n).startswith("c1") and "c1" in want)}
    return d


def ship_gen():
    """The ship generation the export's θ comes from: V2_PRESET=shipped seeds gen 2 unless V2_SHIP_GEN says otherwise."""
    if os.environ.get("V2_SHIP_GEN"):
        return int(os.environ["V2_SHIP_GEN"])
    return 3 if os.environ.get("V2_PRESET") == "shipped" else 0     # the preset's default generation (gen 3 since 10-12)


# the piece values each ship generation carries (V2_VAL_* mp deltas; P = EVAL_V2_PAWN_MG) — an arm's knobs are BASE + δ
SHIP_VALUES = {3: {"N:mg": 17, "B:mg": 10, "R:mg": -136, "Q:mg": -3, "N:eg": -70, "B:eg": -64, "R:eg": 127, "Q:eg": -68, "P:mg": -1}}


def ship_kauf():
    # the Kaufman cells of the ship the arm is a δ on: gen 3 = ship_tables_v4.h, gen 2 = ship_tables_v3.h, else ship_tables_v2.h
    g = ship_gen()
    hdr, sfx = ("ship_tables_v4.h", "3") if g >= 3 else (("ship_tables_v3.h", "2") if g >= 2 else ("ship_tables_v2.h", ""))
    src = open(os.path.join(ENGINE, hdr)).read()
    out = {}
    for tag, key in (("OURS", "O"), ("THEIRS", "T")):
        body = re.search(r"KAUF_SHIP%s_%s\[6\]\[6\] = \{(.*?)\};" % (sfx, tag), src, re.S).group(1)
        rows = [[int(x) for x in re.findall(r"-?\d+", r)] for r in re.findall(r"\{([^{}]*)\}", body)]
        for a in range(6):
            for b in range(6):
                out[(key, a, b)] = rows[a][b]
    return out


def export():
    out = KV["OUT"]; os.makedirs(out, exist_ok=True)
    dump = os.path.join(out, "_pst_ship_dump.txt")
    os.environ["PST_V2_DUMP"] = dump
    import chess, ChessAI
    ChessAI.ChessAI(None, None, chess.Board(), True)            # init writes the shipped PST dump
    tmg, teg = (np.array(t) for t in ChessAI.v2_feature_theta())
    d = load_arm()
    env = []
    for fn, lo, hi, knob in BLOCKS:
        nchg, mx = 0, 0.0
        with open(os.path.join(out, fn + ".txt"), "w") as f:
            f.write("# %s = shipped θ + δ (%s, arm %s) — k leg start fitted\n" % (fn, KV["NPZ"], KV["ARM"]))
            for k in range(lo, hi):
                for leg, th in ((0, tmg[k]), (1, teg[k])):
                    dd = d.get("v2:%d:%s" % (k, "mg" if leg == 0 else "eg"), 0.0)
                    if fn == "c1" and 93 <= k <= 96 and leg == 0 and abs(dd) > 0.5:
                        print("  ⚠️ dropped KD mg δ k=%d %.1f mp (the engine has no mg leg there)" % (k, dd)); dd = 0.0
                    nchg += abs(dd) >= 0.5; mx = max(mx, abs(dd))
                    # ☠️ full precision: the king-distance cells (93-96) are FRACTIONAL per unit and kept as doubles by
                    # the C1 loader — "%.0f" rounded them and broke the endgame closure by up to ~50 mp (10-10)
                    f.write("%d %d %.4f %.4f\n" % (k, leg, th, th + dd))
        env.append("%s=%s" % (knob, os.path.join(out, fn + ".txt")))
        print("  %-6s %3d cells changed (≥ 0.5 mp) · max |δ| %4.0f mp" % (fn, nchg, mx))
    import _texel_kauf_fit as KF
    ship = ship_kauf()
    with open(os.path.join(out, "kauf.txt"), "w") as f:
        f.write("# Kaufman cells = ship_tables_v2.h + δ (%s, arm %s)\n" % (KV["NPZ"], KV["ARM"]))
        for i, (k_, a, b) in enumerate(KF.CELLS[:36]):
            v = ship[(k_, a, b)] + d.get("kauf:%d" % i, 0.0)
            f.write("%s %d %d %.0f\n" % (k_, a, b, v))
    for i in range(36, len(KF.CELLS)):
        if abs(d.get("kauf:%d" % i, 0.0)) > 0.5:
            print("  ⚠️ piece-value cell kauf:%d has δ %.1f mp — NOT exported (no engine home)" % (i, d["kauf:%d" % i]))
    env.append("KAUF_V2_FILE=%s" % os.path.join(out, "kauf.txt"))
    vals = [int(t) for line in open(dump) if not line.startswith("#") for t in line.split()]
    assert len(vals) == 768, "PST dump has %d values" % len(vals)
    newv = list(vals)
    for t in range(6):
        for leg in (0, 1):
            for sq in range(64):
                r, fl = sq >> 3, sq & 7
                dd = d.get("pst:%d:%s" % (t * 32 + r * 4 + min(fl, 7 - fl), "mg" if leg == 0 else "eg"), 0.0)
                newv[t * 128 + leg * 64 + sq] = int(round(vals[t * 128 + leg * 64 + sq] + dd))
    with open(os.path.join(out, "pst.txt"), "w") as f:
        f.write("# v2 PST = shipped (PST_V2_DUMP) + δ (file-tied; %s, arm %s)\n" % (KV["NPZ"], KV["ARM"]))
        for t in range(6):
            for leg in (0, 1):
                for r in range(8):
                    f.write(" ".join(str(newv[t * 128 + leg * 64 + r * 8 + c]) for c in range(8)) + "\n")
    env.append("PST_V2_FILE=%s" % os.path.join(out, "pst.txt"))
    # PIECE VALUES (V2_VAL_* knobs, 10-11): deltas on top of the ship's (gen 2 ships them at 0; the pawn mg leg is
    # EVAL_V2_PAWN_MG, 1000 in every ship so far)
    base = SHIP_VALUES.get(ship_gen(), {})
    for p in "NBRQ":
        for leg in ("mg", "eg"):
            key = "%s:%s" % (p, leg)
            v = base.get(key, 0) + round(d.get("val:" + key, 0.0))
            if v != 0 or key in base:          # always restate the ship's own values (explicit beats implicit)
                env.append("V2_VAL_%s_%s=%d" % (p, leg.upper(), v))
    pmg = base.get("P:mg", 0) + round(d.get("val:P:mg", 0.0))
    if pmg != 0 or "P:mg" in base:
        env.append("EVAL_V2_PAWN_MG=%d" % (1000 + pmg))
    print("  values: %s" % " ".join(e for e in env if e.startswith(("V2_VAL", "EVAL_V2_PAWN"))))
    line = " ".join(env)
    open(os.path.join(out, "env.txt"), "w").write(line + "\n")
    print("EXPORT → %s\n  env: %s" % (out, line))


def closure():
    import chess, ChessAI
    import _retune_fit as R
    ai = ChessAI.ChessAI(None, None, chess.Board(), True)
    d = load_arm()
    n = int(KV.get("N", 3000))
    rows = []
    with open(os.path.join(D, "corpus_v1.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val" and len(rows) < n * 7:
                rows.append(r["fen"])
    rows = rows[::7][:n]
    # the shipped totals from the export chunks (same pass that gave the features)
    want = set(rows); ship = {}
    for p in sorted(glob.glob(os.path.join(D, KV.get("FEAT", "feat_ship") + "_c*.npz"))):
        z = np.load(p)
        for f, t, df, ph in zip(z["fen"], z["total"], z["diff"], z["phase"]):
            if f in want:
                ship[f] = (int(t), df.astype(float), float(ph))
    res, eg_res, win, P, G = [], [], [], [], []
    for f in rows:
        t0, df, ph = ship[f]
        a, e = ph / 256.0, (256.0 - ph) / 256.0
        occ, kf = R.pst_kauf_row(f)
        pred = t0
        for k in range(len(df)):
            if df[k]:
                # the export drops the king-distance cells' mg δ (no engine leg) — predict what was WRITTEN
                dmg = 0.0 if 93 <= k <= 96 else d.get("v2:%d:mg" % k, 0.0)
                pred += df[k] * (a * dmg + e * d.get("v2:%d:eg" % k, 0.0))
        for k in range(192):
            if occ[k]:
                pred += occ[k] * (a * d.get("pst:%d:mg" % k, 0.0) + e * d.get("pst:%d:eg" % k, 0.0))
        for k in range(36):
            if kf[k]:
                pred += kf[k] * d.get("kauf:%d" % k, 0.0)
        pl = f.split()[0]
        for p in "pnbrq":                     # piece values: Black − White count × leg (the pawn has an mg leg only)
            cd = pl.count(p) - pl.count(p.upper())
            if cd:
                P_ = p.upper()
                pred += cd * (a * round(d.get("val:%s:mg" % P_, 0.0)) + e * round(d.get("val:%s:eg" % P_, 0.0)))
        bd = ai.ev_breakdown(chess.Board(f))
        got = int(bd["total"])
        res.append(got - pred); P.append(pred); G.append(got); eg_res.append(ph < 128); win.append(int(bd.get("v2_winnab", 0) or 0) != 0)
    # does the ENGINE realise the fitted gain? win% MSE vs SF18 on these rows (quiet only), each series with its own
    # fitted (STM, SCALE) nuisance, as in the fitter
    from scipy.optimize import minimize
    sfl = {r_["fen"]: (float(r_["sf_cp"]), r_["quiet"] == "1") for r_ in csv.DictReader(open(os.path.join(D, "corpus_v1.csv"), newline=""))
           if r_["fen"] in want}
    q = np.array([sfl[f][1] for f in rows]); t = R.wp(np.array([sfl[f][0] for f in rows]))
    stm = np.array([1.0 if f.split()[1] == "w" else -1.0 for f in rows])
    t0s = np.array([ship[f][0] for f in rows], float); gots = np.array(G, float); preds = np.array(P, float)
    def mse(tot):
        cpw = -tot / 10.0
        f_ = lambda n: float(np.mean((R.wp(cpw[q] * (1 + n[1]) + n[0] * stm[q]) - t[q]) ** 2))
        return f_(minimize(f_, np.zeros(2), method="Nelder-Mead").x)
    ms, mp_, me = mse(t0s), mse(preds), mse(gots)
    print("  REALISED (quiet rows %d): win%% MSE ship %.2f · fit's prediction %.2f (%+.2f%%) · ENGINE arm %.2f (%+.2f%%)"
          % (q.sum(), ms, mp_, 100 * (mp_ / ms - 1), me, 100 * (me / ms - 1)))
    r = np.abs(np.array(res)); eg = np.array(eg_res); wn = np.array(win)
    for nm, m in (("winnability scale INACTIVE", ~wn), ("winnability scale active", wn)):
        if m.any():
            print("  %-27s %4d rows · median %.1f · p95 %.1f · p99 %.1f · max %.0f · >10 mp %.1f%%"
                  % (nm, m.sum(), np.median(r[m]), np.percentile(r[m], 95), np.percentile(r[m], 99), r[m].max(), 100 * (r[m] > 10).mean()))
    print("CLOSURE %s · %d rows · |engine − ship−linear prediction| median %.1f · p95 %.1f · p99 %.1f · max %.0f mp · >10 mp %.1f%% (eg-leaning %.1f%%, mg-leaning %.1f%%)"
          % (KV.get("ARMDIR", ""), len(r), np.median(r), np.percentile(r, 95), np.percentile(r, 99), r.max(),
             100 * (r > 10).mean(), 100 * (r[eg] > 10).mean(), 100 * (r[~eg] > 10).mean()))
    print("  signed mean residual %+.2f mp (a systematic offset means a sign/convention error in one block)" % np.mean(res))


if __name__ == "__main__":
    {"tables": export, "verify": closure}[KV.get("MODE", "tables")]()  # ☠️ never "export"/"closure": _texel_kauf_fit (imported) runs its OWN export / closure on those at import
