"""Read the lazy eval probe's CSV rows (a -DLAZY_PROBE build; LazyProbe in eval_v2.h, lazy_probe_log in search_engine.cpp).

Usage: python diagnostics/_lazy_probe_read.py FILE [FILE ...]   (one CSV per position set; the set name is the file stem)

Each row is one freshly computed eval at a search site, with that site's window. The question is when a cheap value can stand in
for the full eval (dev_notes/active/LAZY-EVAL-RESEARCH-2026-10-30.md §4):
  1. the gap |full - cheap|, overall and by guard feature: a window rule with margin M can only change a decision when gap > M;
  2. position-only exits (|cheap| > T): how often they fire and how often the site's decision changes when they do;
  3. window-aware exits (cheap clears the site's bound by M): the same two numbers;
  4. which guard features the decision changes ("misses") carry, so a guard can be chosen from them.
Cheap values: A = material + PST + bishop pair + Kaufman + narrow classes (everything before the attack build);
A+P = A + pawn structure (pawn-hashed). Millipawns throughout (pawn = 1000).
"""
import sys
import os
import numpy as np
import pandas as pd

SITES = {0: "other", 1: "qs standpat", 2: "qs horizon", 3: "rfp/null entry", 4: "futility child", 5: "improving"}
STAGES = ["cheap", "mob", "space", "threats", "ks", "pawns", "passers", "rookfile", "placement", "c3", "px", "tempo", "win"]


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["set"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    n0 = len(d)
    d = d[d.special == 0].copy()
    print(f"rows {n0:,}; special (draw class / tier-2b) dropped {n0 - len(d):,}")
    # Probe integrity: the stages partition the total, and the search's value is the total in the search frame.
    part = d[STAGES].sum(axis=1)
    bad_part = int((part != d.total).sum())
    sgn = np.where(d.stp == 1, -1, 1)
    bad_ret = int((d.ret != sgn * d.total).sum())
    print(f"integrity: stages != total on {bad_part} rows; ret != frame(total) on {bad_ret} rows (both must be 0)")
    d = d[(part == d.total) & (d.ret == sgn * d.total)].copy()
    d["full"] = sgn * d.total
    d["cheapA"] = sgn * d.cheap
    d["cheapAP"] = sgn * (d.cheap + d.pawns)
    d["gapA"] = (d.full - d.cheapA).abs()
    d["gapAP"] = (d.full - d.cheapAP).abs()
    d["imb"] = (d.mat_w - d.mat_b).abs()
    d["npm"] = d.npm_w + d.npm_b
    d["queens"] = d.q_w + d.q_b
    d["near_max"] = d[["near_wk", "near_bk"]].max(axis=1)
    d["advp"] = d.advp_w.clip(lower=0) + d.advp_b.clip(lower=0)
    d["ksu_max"] = d[["ksu_w", "ksu_b"]].max(axis=1)
    d["pv"] = (d.beta - d.alpha) > 1
    d["attack_part"] = d.mob + d.space + d.threats + d.ks + d.placement + d.c3 + d.px + d.passers + d.rookfile
    return d


def cat(v, a, b):
    return np.where(v <= a, 0, np.where(v >= b, 2, 1))


def decision(d, v):
    """The site's decision for value v (search frame). Rows of sites without a window decision get -1."""
    out = np.full(len(d), -1)
    a, b, m, mx = d.alpha.values, d.beta.values, d.margin.values, d["max"].values == 1
    qs = d.site.isin([1, 2]).values
    out[qs] = cat(v[qs], a[qs], b[qs])
    rfp = (d.site == 3).values
    # node entry: RFP prune (max: v - m >= beta; min: v + m <= alpha) and the null gate (max: v >= beta; min: v <= alpha)
    r_prune = np.where(mx, v - m >= b, v + m <= a)
    r_gate = np.where(mx, v >= b, v <= a)
    out[rfp] = (r_prune * 2 + r_gate)[rfp]
    fut = (d.site == 4).values
    f_prune = np.where(mx, v + m < a, v - m > b)
    out[fut] = f_prune[fut].astype(int)
    return out


def pct(x):
    return f"{100 * x:5.1f}%"


def main(paths):
    d = load(paths)
    print(f"rows used {len(d):,}\n")
    print("rows by set and site:")
    print(d.groupby(["set", "site"]).size().unstack(fill_value=0).rename(columns=SITES).to_string(), "\n")

    # 1. the gap
    qs = [50, 90, 99, 99.9]
    print("gap |full - cheap| (millipawns), percentiles", qs)
    for name, g in d.groupby("set"):
        print(f"  {name:10s} A {np.percentile(g.gapA, qs).astype(int)}   A+P {np.percentile(g.gapAP, qs).astype(int)}")
    print(f"  {'ALL':10s} A {np.percentile(d.gapA, qs).astype(int)}   A+P {np.percentile(d.gapAP, qs).astype(int)}\n")

    print("P(gap A > M) by guard feature (rows; then share with gap above 1000 / 1500 / 2000 / 3000):")
    feats = [
        ("all", d.index == d.index),
        ("no queens", d.queens == 0),
        ("queens on", d.queens > 0),
        ("imbalance < 1 pawn", d.imb < 1000),
        ("imbalance 1-3 pawns", (d.imb >= 1000) & (d.imb < 3000)),
        ("imbalance >= 3 pawns", d.imb >= 3000),
        ("enemy pieces near a king >= 2", d.near_max >= 2),
        ("enemy pieces near a king < 2", d.near_max < 2),
        ("advanced passer", d.advp > 0),
        ("no advanced passer", d.advp == 0),
        ("phase >= 128 (mg)", d.phase >= 128),
        ("phase < 128 (eg)", d.phase < 128),
        ("PV window", d.pv),
        ("null window", ~d.pv),
    ]
    for label, mask in feats:
        g = d[mask]
        if len(g) == 0:
            continue
        print(f"  {label:32s} {len(g):9,}  " + "  ".join(pct((g.gapA > M).mean()) for M in (1000, 1500, 2000, 3000)))
    print()

    full_dec = decision(d, d.full.values)
    has_dec = full_dec >= 0

    # 2. position-only exits on the cheap part A
    print("position-only exit |cheap A| > T + k_npm*npm/1000 + k_imb*imbalance/1000 (decision sites only):")
    print("  T      k_npm k_imb   fires   miss/fire   miss per 1000 evals   mean gap on fires")
    dd = d[has_dec]
    fd = full_dec[has_dec]
    for T in (2000, 3000, 4000, 6000, 8000):
        for k_npm, k_imb in ((0, 0), (40, 0), (0, 500), (40, 500)):
            thr = T + k_npm * dd.npm / 1000.0 + k_imb * dd.imb / 1000.0
            fire = (dd.cheapA.abs() > thr).values
            ld = decision(dd, dd.cheapA.values)
            miss = fire & (ld != fd)
            nf = fire.sum()
            print(f"  {T:5d}  {k_npm:5d} {k_imb:5d}  {pct(fire.mean())}   {pct(miss.sum() / max(nf, 1))}"
                  f"      {1000 * miss.mean():6.2f}            {int(dd.gapA[fire].mean()) if nf else 0}")
    print()

    # 3. window-aware exits at each site: the cheap value clears the site's bound by M
    print("window-aware exit, cheap A clears the site's bound by M (per site):")
    print("  site              M     fires   miss/fire   miss per 1000 evals")
    for s in (1, 2, 3, 4):
        g = d[d.site == s]
        if len(g) == 0:
            continue
        gf = decision(g, g.full.values)
        mx = g["max"].values == 1
        a, b, m = g.alpha.values, g.beta.values, g.margin.values
        for M in (1000, 1500, 2000, 3000):
            v = g.cheapA.values
            if s in (1, 2):      # stand-pat: cut decision (max: v >= beta, min: v <= alpha), either side of the window
                fire = (v - M >= b) | (v + M <= a)
            elif s == 3:         # node entry: the RFP bound
                fire = np.where(mx, v - m - M >= b, v + m + M <= a) | np.where(mx, v + M <= a, v - M >= b)
            else:                # futility: the futility bound
                fire = np.where(mx, v + m + M < a, v - m - M > b) | np.where(mx, v - M >= a, v + M <= b)
            miss = fire & (decision(g, v) != gf)
            print(f"  {SITES[s]:16s} {M:5d}  {pct(fire.mean())}   {pct(miss.sum() / max(fire.sum(), 1))}      {1000 * miss.mean():6.2f}")
    print()

    # 4. what the misses carry: a position-only exit at T=3000 as the probe
    thr = 3000
    fire = (dd.cheapA.abs() > thr).values
    miss = fire & (decision(dd, dd.cheapA.values) != fd)
    fires, misses = dd[fire], dd[miss]
    print(f"misses of |cheap A| > {thr}: {len(misses):,} of {len(fires):,} fires; feature rates, fires vs misses:")
    for label, col, test in (("queens on", "queens", lambda x: x > 0), ("imbalance >= 3 pawns", "imb", lambda x: x >= 3000),
                             ("near a king >= 2", "near_max", lambda x: x >= 2), ("advanced passer", "advp", lambda x: x > 0),
                             ("PV window", "pv", lambda x: x), ("mg (phase >= 128)", "phase", lambda x: x >= 128),
                             ("KS units >= 10", "ksu_max", lambda x: x >= 10)):
        rf = test(fires[col]).mean() if len(fires) else 0
        rm = test(misses[col]).mean() if len(misses) else 0
        print(f"  {label:24s} fires {pct(rf)}   misses {pct(rm)}")
    if len(misses):
        print("  stage means on misses (search frame sign not applied; magnitudes):")
        print("   ", {k: int(misses[k].abs().mean()) for k in STAGES})
    print()
    guards(d)
    print()
    inherit(d)


def window_fire(g, v, M):
    """Window-aware exit per site (see section 3): the cheap value clears the site's bound by M (M may be per row)."""
    mx = g["max"].values == 1
    a, b, m, s = g.alpha.values, g.beta.values, g.margin.values, g.site.values
    qs = (v - M >= b) | (v + M <= a)
    rfp = np.where(mx, v - m - M >= b, v + m + M <= a) | np.where(mx, v + M <= a, v - M >= b)
    fut = np.where(mx, v + m + M < a, v - m - M > b) | np.where(mx, v - M >= a, v + M <= b)
    return np.where(np.isin(s, [1, 2]), qs, np.where(s == 3, rfp, np.where(s == 4, fut, False)))


def guards(d):
    """5. Window-aware exits with guards: which cheap-to-know conditions remove the misses, and what they cost in fires."""
    g = d[d.site.isin([1, 2, 3, 4])]
    full = decision(g, g.full.values)
    lazy = decision(g, g.cheapA.values)
    changed = lazy != full
    G = {
        "none": np.ones(len(g), bool),
        "no king crowd (near<2)": (g.near_max < 2).values,
        "+ not PV": ((g.near_max < 2) & ~g.pv).values,
        "+ no adv passer": ((g.near_max < 2) & ~g.pv & (g.advp == 0)).values,
        "+ phase >= 64": ((g.near_max < 2) & ~g.pv & (g.advp == 0) & (g.phase >= 64)).values,
    }
    print("window-aware exit with guards, all decision sites (stand-pat, horizon, node entry, futility):")
    print("  guard                       M rule              fires   miss/fire   miss per 1000 evals")
    imb = g.imb.values
    for gname, ok in G.items():
        for mname, M in (("1500", 1500), ("2000", 2000), ("3000", 3000),
                         ("1500 + imb/4", 1500 + imb // 4), ("2000 + imb/4", 2000 + imb // 4)):
            fire = window_fire(g, g.cheapA.values, M) & ok
            miss = fire & changed
            print(f"  {gname:26s} {mname:16s}  {pct(fire.mean())}   {pct(miss.sum() / max(fire.sum(), 1))}      {1000 * miss.mean():6.2f}")
    print("  per site at guard '+ no adv passer', M 2000 + imb/4:")
    ok = G["+ no adv passer"]
    fire = window_fire(g, g.cheapA.values, 2000 + imb // 4) & ok
    for s in (1, 2, 3, 4):
        k = g.site.values == s
        print(f"    {SITES[s]:16s} fires {pct(fire[k].mean())}   miss/fire {pct((fire & changed)[k].sum() / max(fire[k].sum(), 1))}")
    print("  by set at the same rule:")
    for name in sorted(g.set.unique()):
        k = g.set.values == name
        print(f"    {name:12s} fires {pct(fire[k].mean())}   miss/fire {pct((fire & changed)[k].sum() / max(fire[k].sum(), 1))}"
              f"   miss per 1000 {1000 * (fire & changed)[k].mean():6.2f}")


def inherit(d):
    """6. Step 0b, IvanHoe's estimate: this position's cheap part + its parent's positional part (full - cheap, read when the parent
    position was evaluated one ply up the same line). Rows without a parent record fall back to no lazy exit."""
    if "parent_pos" not in d.columns:
        print("no parent_pos column (a step 0 file); section 6 needs the 0b probe")
        return
    g = d[d.site.isin([1, 2, 3, 4])].copy()
    known = (g.parent_pos != -999999999).values
    sgn = np.where(g.stp == 1, -1, 1)
    g["est"] = sgn * (g.cheap + np.where(known, g.parent_pos, 0))
    g["gapI"] = (g.full - g.est).abs()
    print("inherited estimate (cheap + parent's positional part): parent known on "
          f"{pct(known.mean())} of decision rows; by site:", {SITES[s]: pct(known[g.site.values == s].mean()) for s in (1, 2, 3, 4)})
    qs = [50, 90, 99, 99.9]
    k = g[known]
    print(f"  gap on rows with a parent, percentiles {qs}: cheap A {np.percentile(k.gapA, qs).astype(int)}"
          f"   inherited {np.percentile(k.gapI, qs).astype(int)}")
    for label, mask in (("no queens", k.queens == 0), ("queens on", k.queens > 0), ("near a king >= 2", k.near_max >= 2),
                        ("advanced passer", k.advp > 0), ("imbalance >= 3 pawns", k.imb >= 3000), ("eg (phase < 128)", k.phase < 128)):
        kk = k[mask]
        if len(kk):
            print(f"    {label:22s} {len(kk):9,}  inherited p90 {int(np.percentile(kk.gapI, 90))}  p99 {int(np.percentile(kk.gapI, 99))}"
                  f"   P(> 1000) {pct((kk.gapI > 1000).mean())}")
    full = decision(g, g.full.values)
    lazy = decision(g, g.est.values)
    changed = lazy != full
    guard = ((g.near_max < 2) & ~g.pv & (g.advp == 0)).values
    print("  window-aware exit on the inherited estimate (no exit where the parent is unknown):")
    print("  guard        M      fires   miss/fire   miss per 1000 evals")
    for gname, ok in (("none", np.ones(len(g), bool)), ("crowd/PV/passer", guard)):
        for M in (500, 750, 1000, 1500, 2000):
            fire = window_fire(g, g.est.values, M) & ok & known
            miss = fire & changed
            print(f"  {gname:14s} {M:5d}  {pct(fire.mean())}   {pct(miss.sum() / max(fire.sum(), 1))}      {1000 * miss.mean():6.2f}")
    print("  per site, guard crowd/PV/passer, M 1000:")
    fire = window_fire(g, g.est.values, 1000) & guard & known
    for s in (1, 2, 3, 4):
        kk = g.site.values == s
        print(f"    {SITES[s]:16s} fires {pct(fire[kk].mean())}   miss/fire {pct((fire & changed)[kk].sum() / max(fire[kk].sum(), 1))}")
    print("  by set, same rule:")
    for name in sorted(g.set.unique()):
        kk = g.set.values == name
        print(f"    {name:12s} fires {pct(fire[kk].mean())}   miss/fire {pct((fire & changed)[kk].sum() / max(fire[kk].sum(), 1))}"
              f"   miss per 1000 {1000 * (fire & changed)[kk].mean():6.2f}")


MOVES = {-1: "root", 0: "quiet piece", 1: "pawn move", 2: "push to 6th/7th", 3: "king move", 4: "pawn captured",
         5: "piece captured", 6: "promotion", 7: "null move"}


def build_rule(g, m_par, m_cheap, guards=7):
    """The build's exit (LAZY-EVAL.md §3), cut side only: the estimate clears the site's cut bound by M (M_PAR with the parent's
    positional part, M_CHEAP without), unless a guard applies. Returns (fire, miss): miss = the full eval would not have cut."""
    known = (g.parent_pos != -999999999).values
    sgn = np.where(g.stp == 1, -1, 1)
    est = sgn * (g.cheap.values + np.where(known, g.parent_pos.values, 0))
    full = g.full.values
    M = np.where(known, m_par, m_cheap)
    mx = g["max"].values == 1
    a, b, m, s = g.alpha.values, g.beta.values, g.margin.values, g.site.values
    sp = s == 1
    # cut bounds in the search frame: stand-pat max >= beta / min <= alpha; RFP max >= beta + m / min <= alpha - m;
    # futility of a child (prune) max <= alpha - m - 1 / min >= beta + m + 1
    up = np.where(sp, mx, np.where(s == 3, mx, ~mx))
    bound = np.where(sp, np.where(mx, b, a), np.where(s == 3, np.where(mx, b + m, a - m), np.where(mx, a - m - 1, b + m + 1)))
    fire = np.where(up, est >= bound + M, est <= bound - M) & (M > 0)
    cut_full = np.where(up, full >= bound, full <= bound)
    ok = np.ones(len(g), bool)
    if guards & 1:
        ok &= (g.near_max < 2).values
    if guards & 2:
        ok &= ~(g.pv.values & (s != 3))
    if guards & 4:
        ok &= (g.advp == 0).values
    fire &= ok
    return fire, fire & ~cut_full


def context(d):
    """7. The context split (11-08, owner's gating idea): where the build's misses fall, by the move that led to the position and by
    depth, at the candidate's margins (keep_m1000: 1000 with a parent value, 2000 without) and at build 1's (1500 / 3000)."""
    if "mv" not in d.columns:
        print("no mv column (a probe before 11-08); section 7 needs the context probe")
        return
    g = d[d.site.isin([1, 3, 4])].copy()   # the build's three sites (2, the q-depth cap's return value, is not one)
    g["sitek"] = np.where(g.site == 1, "stand-pat", np.where(g.site == 3, "rfp", "futility"))
    known = (g.parent_pos != -999999999).values
    print(f"context probe: {len(g):,} decision rows; parent known on {pct(known.mean())}; by site:",
          {k: pct(known[(g.sitek == k).values].mean()) for k in ("stand-pat", "rfp", "futility")})
    for label, mp, mc in (("candidate M 1000 / 2000", 1000, 2000), ("build 1 M 1500 / 3000", 1500, 3000)):
        fire, miss = build_rule(g, mp, mc)
        print(f"  {label}: fires {pct(fire.mean())} of asks, miss/fire {pct(miss.sum() / max(fire.sum(), 1))},"
              f" misses per 1000 asks {1000 * miss.mean():.2f}")
    fire, miss = build_rule(g, 1000, 2000)
    nf, nm = max(fire.sum(), 1), max(miss.sum(), 1)
    print("\n  by site and last move (candidate rule): share of asks / of fires / of misses, miss per fire")
    print(f"  {'site':10s} {'last move':16s} {'asks':>7s} {'fires':>7s} {'misses':>7s} {'miss/fire':>9s}")
    for sk in ("stand-pat", "rfp", "futility"):
        for code, name in MOVES.items():
            k = (g.sitek == sk).values & (g.mv == code).values
            if k.sum() == 0:
                continue
            print(f"  {sk:10s} {name:16s} {pct(k.mean())} {pct(fire[k].sum() / nf)} {pct(miss[k].sum() / nm)}"
                  f"   {pct(miss[k].sum() / max(fire[k].sum(), 1))}")
    cap = g.mv.isin([4, 5]).values
    for label, k in (("capture near a king", cap & (g.kz == 1).values), ("capture elsewhere", cap & (g.kz == 0).values)):
        print(f"  {'all':10s} {label:16s} {pct(k.mean())} {pct(fire[k].sum() / nf)} {pct(miss[k].sum() / nm)}"
              f"   {pct(miss[k].sum() / max(fire[k].sum(), 1))}")
    print("\n  by remaining depth (rfp, futility) and qsearch depth (stand-pat): share of fires / of misses, miss per fire")
    for sk in ("rfp", "futility", "stand-pat"):
        ks = (g.sitek == sk).values
        for dl in sorted(g.depth_left[ks].unique(), reverse=True)[:8]:
            k = ks & (g.depth_left == dl).values
            print(f"  {sk:10s} depth {dl:3d}   {pct(fire[k].sum() / nf)} {pct(miss[k].sum() / nm)}   {pct(miss[k].sum() / max(fire[k].sum(), 1))}")
    print("\n  candidate gates (no lazy exit when the condition holds): fires kept, misses removed")
    rfp, fut, sp = (g.sitek == "rfp").values, (g.sitek == "futility").values, (g.sitek == "stand-pat").values
    gates = {
        "after any capture": cap,
        "after a capture near a king": cap & (g.kz == 1).values,
        "after a piece capture": (g.mv == 5).values,
        "after a promotion or 6th/7th push": g.mv.isin([2, 6]).values,
        "after a king move": (g.mv == 3).values,
        "after a null move": (g.mv == 7).values,
        "stand-pat after any capture": sp & cap,
        "rfp at depth >= 4": rfp & (g.depth_left >= 4).values,
        "rfp at depth >= 5": rfp & (g.depth_left >= 5).values,
        "no parent value (cheap path)": ~known,
    }
    for label, k in gates.items():
        kept = (fire & ~k).sum() / nf
        removed = (miss & k).sum() / nm
        print(f"  {label:36s} fires kept {pct(kept)}   misses removed {pct(removed)}")


if __name__ == "__main__":
    if os.environ.get("ONLY_CONTEXT") == "1":
        context(load(sys.argv[1:]))
    elif os.environ.get("ONLY_GUARDS") == "1":
        guards(load(sys.argv[1:]))
    elif os.environ.get("ONLY_INHERIT") == "1":
        inherit(load(sys.argv[1:]))
    else:
        main(sys.argv[1:])
