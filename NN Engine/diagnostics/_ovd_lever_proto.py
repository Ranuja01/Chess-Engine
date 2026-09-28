# -*- coding: utf-8 -*-
"""OvD PROTOTYPE, feature 1 -- LEVER OUTCOME: does the pawn TRANSFORMATION a side can force predict the game BEYOND the eval?

Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §8 (the owner's concept: long-term pressure = which side can
force a favourable positional transformation, e.g. a break that leaves the opponent two isolanis and us a majority).
Pure Python, no engine: a FEASIBILITY pilot for the outcome channel, not a verdict (the outcome screen has a resolution
floor -- C3 §6b), run before any C++ exists.

FEATURE. For each side S, its structure OPTIONS (pawn-for-pawn only; a line that just wins or loses a pawn is a tactic,
not a transformation, and is skipped):
  now   S takes an enemy pawn with a pawn and the opponent recaptures with a pawn (the opponent picks the recapture
        that is best for it);
  push  S pushes a pawn one square into contact (the square empty, the pushed pawn defended by an S pawn); the opponent
        then either takes (S recaptures) or declines (S takes next, the opponent recaptures if it can) -- the opponent
        picks whichever is better for it.
Each resolved structure is scored with v2's own pawn-structure values (the shipped C1 values stored in the feature pass:
doubled, isolated by file, backward, weak-unopposed, passed and candidate by relative rank), phase-blended, as
(S's structure − the opponent's). Δ = after − before. Per side: best_now, best_push = max(0, best Δ) over its options.
The row's feature (White-positive): lever_now = best_now(W) − best_now(B), lever_push likewise.

TEST (near-equal rows, |eval| <= NEAR mp): the logistic residual of the result on the SHIPPED eval (K fitted on the
sample), correlated with the feature; plus binned mean residual. A real signal shows a positive correlation that the eval
does not already carry. The option-not-execution caveat (C3 §8a: premature transformations are 3x commoner than
critical ones) is why the feature is the BEST option held, not whether one was played.

  pyrun diagnostics/_ovd_lever_proto.py [N=200000] [NEAR=1000] [SEED=1]
"""
import os, sys, time, math
import numpy as np
import pandas as pd

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
N = int(KV.get("N", 200000))
NEAR = float(KV.get("NEAR", 1000))
SEED = int(KV.get("SEED", 1))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from _pawn_term_overlap import terms, watt, batt, FULL
T0 = time.time()

fz = np.load(os.path.join(DATA, "fitC_features.npz"))
TMG, TEG = fz["theta_mg"], fz["theta_eg"]


def struct_value(own, enemy, white, wmg):
    """v2's pawn-structure value of OWN's pawns (to own's side), phase-blended (wmg = phase/256)."""
    t = terms(own, enemy, white)
    th = lambda k: TMG[k] * wmg + TEG[k] * (1.0 - wmg)
    v = bin(t["doubled"]).count("1") * th(66) + bin(t["backward"]).count("1") * th(75) \
        + bin(t["weak_unopp"]).count("1") * th(76)
    b = t["isolated"]
    while b:
        sq = (b & -b).bit_length() - 1
        b &= b - 1
        v += th(67 + (sq & 7))
    for key, base in (("passed", 77), ("candidate", 85)):
        b = t[key] & ~(t["candidate"] if key == "passed" else 0)
        while b:
            sq = (b & -b).bit_length() - 1
            b &= b - 1
            r = (sq >> 3) if white else 7 - (sq >> 3)
            v += th(base + r)
    return v


def rel(wp, bp, wmg):
    """White's structure minus Black's (White-positive)."""
    return struct_value(wp, bp, True, wmg) - struct_value(bp, wp, False, wmg)


def bits(b):
    while b:
        sq = (b & -b).bit_length() - 1
        b &= b - 1
        yield sq


def pawn_att_from(sq, white):
    """Squares a pawn of this colour on sq attacks."""
    m = 1 << sq
    return watt(m) if white else batt(m)


def attackers_of(sq, pawns, white):
    """Pawns (of colour `white`) that attack sq."""
    m = 1 << sq
    return pawns & (batt(m) if white else watt(m))


def exchange_now(wp, bp, white, frm, to):
    """S (colour `white`) plays pawn frm x pawn on `to`; the opponent must recapture pawn-for-pawn. Returns the list of
    resulting (wp, bp) -- one per recapture -- or [] when no pawn recapture exists (a pawn win: skipped)."""
    own, opp = (wp, bp) if white else (bp, wp)
    own2 = (own & ~(1 << frm)) | (1 << to)
    opp2 = opp & ~(1 << to)
    outs = []
    for rc in bits(attackers_of(to, opp2, not white)):
        o3 = (opp2 & ~(1 << rc)) | (1 << to)
        s3 = own2 & ~(1 << to)
        outs.append((s3, o3) if white else (o3, s3))
    return outs


def side_options(wp, bp, occ, white, wmg, base):
    """(best_now, best_push) Δ for side `white`, from S's view (S's structure minus the opponent's)."""
    sgn = 1.0 if white else -1.0
    own, opp = (wp, bp) if white else (bp, wp)
    best_now = best_push = 0.0
    # now: pawn x pawn with a pawn recapture; the opponent picks the recapture that is best for IT
    for frm in bits(own):
        for to in bits(pawn_att_from(frm, white) & opp):
            outs = exchange_now(wp, bp, white, frm, to)
            if outs:
                d = min(sgn * (rel(a, b, wmg) - base) for a, b in outs)
                best_now = max(best_now, d)
    # push into contact, defended; the opponent takes (we recapture) or declines (we take, it recaptures)
    fwd = 8 if white else -8
    for frm in bits(own):
        to = frm + fwd
        if not (0 <= to < 64) or (occ >> to) & 1:
            continue
        if not (pawn_att_from(to, white) & opp):
            continue                                   # not a lever push
        own_p = (own & ~(1 << frm)) | (1 << to)
        if not attackers_of(to, own_p, white):
            continue                                   # undefended: the opponent just wins it
        wp1, bp1 = (own_p, opp) if white else (opp, own_p)
        replies = []
        for cap in bits(attackers_of(to, opp, not white)):    # the opponent takes; we recapture
            opp2 = (opp & ~(1 << cap)) | (1 << to)
            own2 = own_p & ~(1 << to)
            for rc in bits(attackers_of(to, own2, white)):
                own3 = (own2 & ~(1 << rc)) | (1 << to)
                opp3 = opp2 & ~(1 << to)
                a, b = (own3, opp3) if white else (opp3, own3)
                replies.append(sgn * (rel(a, b, wmg) - base))
        decl = []                                      # the opponent declines; we take one of the pawns we hit
        for tgt in bits(pawn_att_from(to, white) & opp):
            outs = exchange_now(wp1, bp1, white, to, tgt)
            if outs:
                decl.append(min(sgn * (rel(a, b, wmg) - base) for a, b in outs))
        if decl:
            replies.append(max(decl))
        if replies:
            best_push = max(best_push, min(replies))
    return best_now, best_push


def board_bits(fen):
    wp = bp = occ = 0
    r, f = 7, 0
    for ch in fen.split(" ", 1)[0]:
        if ch == "/":
            r -= 1
            f = 0
        elif ch <= "8":
            f += ord(ch) - 48
        else:
            sq = r * 8 + f
            occ |= 1 << sq
            if ch == "P":
                wp |= 1 << sq
            elif ch == "p":
                bp |= 1 << sq
            f += 1
    return wp, bp, occ


def main():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["fen", "result_white"])
    full = fz["total"].astype(np.float64)
    ph = fz["phase"].astype(np.float64)
    ok = np.where((ph >= 0) & (np.abs(full) < 30000))[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(ok, size=min(N, len(ok)), replace=False))
    fens = st["fen"].values[idx]
    y = st["result_white"].values[idx].astype(np.float64)
    E = -full[idx]                                      # White-positive eval (the engine is Black-positive)
    wmg = ph[idx] / 256.0
    ln = np.zeros(len(idx))
    lp = np.zeros(len(idx))
    anyopt = np.zeros(len(idx), dtype=bool)
    for i, fen in enumerate(fens):
        wp, bp, occ = board_bits(fen)
        base = rel(wp, bp, wmg[i])
        wn, wpu = side_options(wp, bp, occ, True, wmg[i], base)
        bn, bpu = side_options(wp, bp, occ, False, wmg[i], base)
        ln[i], lp[i] = wn - bn, wpu - bpu
        anyopt[i] = (wn or bn or wpu or bpu) != 0
        if (i + 1) % 50000 == 0:
            print("  %d rows  %.0fs" % (i + 1, time.time() - T0), flush=True)

    # K on the sample, then residual of the result on the eval
    def mse(k):
        p = 1.0 / (1.0 + np.exp(-np.clip(k * E, -60, 60)))
        return float(np.mean((p - y) ** 2))
    lo, hi = 1e-4, 1e-2
    for _ in range(60):
        a, b = math.exp(math.log(lo) + 0.382 * (math.log(hi) - math.log(lo))), math.exp(math.log(lo) + 0.618 * (math.log(hi) - math.log(lo)))
        lo, hi = (lo, b) if mse(a) < mse(b) else (a, hi)
    K = (lo + hi) / 2
    res = y - 1.0 / (1.0 + np.exp(-np.clip(K * E, -60, 60)))
    near = np.abs(E) <= NEAR
    print("\nLEVER-OUTCOME PROTOTYPE  rows %d  near-equal %d  K %.6f  %.0fs" % (len(idx), near.sum(), K, time.time() - T0))
    print("  rows with any option: %.1f%%  (near-equal: %.1f%%)" % (100 * anyopt.mean(), 100 * anyopt[near].mean()))
    for name, f in (("lever_now", ln), ("lever_push", lp)):
        m = near & (f != 0)
        r = np.corrcoef(f[m], res[m])[0, 1] if m.sum() > 50 else float("nan")
        se = 1.0 / math.sqrt(max(m.sum() - 3, 1))
        print("\n  %s: non-zero on %.1f%% of near-equal rows; corr(feature, result residual | non-zero) r = %+.4f (±%.4f, %.1fσ)"
              % (name, 100 * m.sum() / max(near.sum(), 1), r, se, r / se if se else 0))
        qs = np.quantile(f[m], [0.1, 0.3, 0.5, 0.7, 0.9]) if m.sum() > 50 else []
        edges = [-np.inf] + list(qs) + [np.inf]
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            mm = m & (f > lo_) & (f <= hi_)
            if mm.sum():
                print("    feature in (%8.1f, %8.1f]  n %7d  mean residual %+.4f  (mean feature %+.1f mp)"
                      % (lo_, hi_, mm.sum(), res[mm].mean(), f[mm].mean()))
    print("\n⚠️ Feasibility pilot: a residual correlation is necessary, not sufficient -- the fit + games decide.")


if __name__ == "__main__":
    main()
