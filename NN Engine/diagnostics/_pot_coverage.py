# -*- coding: utf-8 -*-
"""POT COVERAGE — how often is each transformation type's gate ON, and how often does its result arrive?

POT = "OvD reworked" (the owner's v1 invention); type definitions: dev_notes/POT-TYPE-DEFINITIONS-2026-09-30.md.
Answers the owner's "is the concept narrow?" (09-30) before any fitting. Pure Python on stage-1 game sequences.
Structure only; every gate = PRECURSORS present ∧ RESULT absent, per attacker side A (the side holding the option).

  T1 central opening vs uncastled king: D king on d/e; every file kf±1 ∩ c-f holds a D pawn; A has a central (c-f)
     lever now or lever REACH ≤ 2 pushes.                        EVENT: D king still on d-f, a file near it lost D pawns.
  T3 chain-base attack: a ram on c-f whose D pawn is supported diagonally by a D pawn (the BASE); A can reach a square
     attacking the base in ≤ 2 pushes; the base has a D pawn on an adjacent file (not isolated).
                                                                  EVENT: the base square no longer holds that D pawn,
                                                                  or the base pawn is isolated.
  T4 majority → passer: a flank (a-c or f-h) where A has more pawns than D, A's flank pawns not doubled, and no A
     passer or candidate on it yet.                               EVENT: an A passed pawn on that flank.
  T5 minority attack: a flank where A has fewer (≥1) pawns than D (≥2), A lever reach ≤ 2 against a D flank pawn, no
     D flank pawn isolated yet.                                   EVENT: a D pawn on that flank is isolated.
LEVER REACH k: an A pawn that by ≤ k single pushes along an EMPTY path reaches a square attacking a D pawn.
Middlegame = ≥ 20 men on the board. EVENT horizon: the first stored position with ply ≥ t + N.

  pyrun diagnostics/_pot_coverage.py [GAMES=3000] [N=20] [MODE=lift]
"""
import os, sys, math
import numpy as np
import pandas as pd
import chess

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
GAMES = int(KV.get("GAMES", 3000))
N = int(KV.get("N", 20))
DATA = KV.get("DATA", "/mnt/e/chess_data/texel")
CENTRE = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F
FLANKS = {"q": chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C, "k": chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H}
TYPES = ["T1", "T3", "T4", "T5"]


def pawns(b, c):
    return int(b.pieces(chess.PAWN, c))


def lever_reach(b, A, k, targets):
    """A pawns that reach, by ≤ k pushes on an empty path, a square attacking a pawn in `targets` (bitboard)."""
    occ, fwd, n = int(b.occupied), (8 if A == chess.WHITE else -8), 0
    for sq in chess.SquareSet(pawns(b, A)):
        s = sq
        for step in range(k + 1):
            if chess.BB_PAWN_ATTACKS[A][s] & targets:
                n += 1
                break
            t = s + fwd
            if not (0 <= t < 64) or occ >> t & 1:
                break
            s = t
    return n


def isolated(bb_own, sq):
    f = chess.square_file(sq)
    adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    return not (bb_own & adj)


def passed(b, c, sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for e in chess.SquareSet(pawns(b, not c)):
        if abs(chess.square_file(e) - f) <= 1 and ((chess.square_rank(e) > r) if c == chess.WHITE else (chess.square_rank(e) < r)):
            return False
    return True


def candidate(b, c, sq):
    """No enemy pawn ahead on the same file, and own supporters (adjacent files, level or behind) ≥ enemy stoppers."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = lambda rr: (rr > r) if c == chess.WHITE else (rr < r)
    stop = sup = 0
    for e in chess.SquareSet(pawns(b, not c)):
        ef, er = chess.square_file(e), chess.square_rank(e)
        if ef == f and ahead(er):
            return False
        if abs(ef - f) == 1 and ahead(er):
            stop += 1
    for o in chess.SquareSet(pawns(b, c)):
        of, orr = chess.square_file(o), chess.square_rank(o)
        if abs(of - f) == 1 and not ahead(orr):
            sup += 1
    return sup >= stop


def half_open_near(b, D, kf):
    dp = pawns(b, D)
    return sum(1 for f in range(max(2, kf - 1), min(5, kf + 1) + 1) if not dp & chess.BB_FILES[f])


def gates(b, A):
    """{type: context} for the gates that are ON for attacker A (context feeds the event check)."""
    D = not A
    ap, dp = pawns(b, A), pawns(b, D)
    out = {}
    # T1
    kf = chess.square_file(b.king(D))
    if kf in (3, 4) and half_open_near(b, D, kf) == 0 and lever_reach(b, A, 2, dp & CENTRE) > 0:
        out["T1"] = kf
    # T3: rams on c-f whose D pawn is supported by a D pawn (the base)
    fwd = 8 if A == chess.WHITE else -8
    bases = []
    for sq in chess.SquareSet(ap & CENTRE):
        t = sq + fwd
        if 0 <= t < 64 and dp >> t & 1:
            for base in chess.SquareSet(chess.BB_PAWN_ATTACKS[A][t] & dp):     # D pawns defending t
                if not isolated(dp, base) and lever_reach(b, A, 2, chess.BB_SQUARES[base]) > 0:
                    bases.append(base)
    if bases:
        out["T3"] = bases
    # T4 / T5 per flank
    t4, t5 = [], []
    for name, fm in FLANKS.items():
        na, nd = chess.popcount(ap & fm), chess.popcount(dp & fm)
        if na > nd:
            files = [chess.square_file(s) for s in chess.SquareSet(ap & fm)]
            if len(files) == len(set(files)) and not any(passed(b, A, s) or candidate(b, A, s)
                                                          for s in chess.SquareSet(ap & fm)):
                t4.append(name)
        if 1 <= na < nd and nd >= 2 and lever_reach(b, A, 2, dp & fm) > 0 \
                and not any(isolated(dp, s) for s in chess.SquareSet(dp & fm)):
            t5.append(name)
    if t4:
        out["T4"] = t4
    if t5:
        out["T5"] = t5
    return out


def controls(b, A):
    """MODE=lift: the SAME gates with the KEY precursor removed (a near-miss population), so lift = event rate with
    the precursor / without it. T1: central king + closed files, NO central lever reach. T3: supported ram bases with
    NO lever reach against the base. T4: flanks with NO majority for A (na ≤ nd, nd ≥ 1), no A passer/candidate there.
    T5: A minority flanks (1 ≤ na < nd, nd ≥ 2), no D isolani, NO lever reach on the flank."""
    D = not A
    ap, dp = pawns(b, A), pawns(b, D)
    out = {}
    kf = chess.square_file(b.king(D))
    if kf in (3, 4) and half_open_near(b, D, kf) == 0 and lever_reach(b, A, 2, dp & CENTRE) == 0:
        out["T1"] = kf
    fwd = 8 if A == chess.WHITE else -8
    bases = []
    for sq in chess.SquareSet(ap & CENTRE):
        t = sq + fwd
        if 0 <= t < 64 and dp >> t & 1:
            for base in chess.SquareSet(chess.BB_PAWN_ATTACKS[A][t] & dp):
                if not isolated(dp, base) and lever_reach(b, A, 2, chess.BB_SQUARES[base]) == 0:
                    bases.append(base)
    if bases:
        out["T3"] = bases
    t4, t5 = [], []
    for name, fm in FLANKS.items():
        na, nd = chess.popcount(ap & fm), chess.popcount(dp & fm)
        if 1 <= nd and na <= nd and na >= 1 and not any(passed(b, A, s) or candidate(b, A, s)
                                                         for s in chess.SquareSet(ap & fm)):
            t4.append(name)
        if 1 <= na < nd and nd >= 2 and lever_reach(b, A, 2, dp & fm) == 0 \
                and not any(isolated(dp, s) for s in chess.SquareSet(dp & fm)):
            t5.append(name)
    if t4:
        out["T4"] = t4
    if t5:
        out["T5"] = t5
    return out


def main_lift():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["game_id", "ply", "fen"])
    games = st["game_id"].unique()[:GAMES]
    st = st[st["game_id"].isin(set(games))]
    ev = {(t, k): [0, 0] for t in TYPES for k in ("gate", "ctrl")}
    for gid, g in st.groupby("game_id", sort=False):
        plies, fens = g["ply"].values, g["fen"].values
        for i in range(len(g)):
            b = chess.Board(fens[i])
            if chess.popcount(int(b.occupied)) < 20:
                continue
            j = np.searchsorted(plies, plies[i] + N)
            if j >= len(g):
                continue
            fut = chess.Board(fens[j])
            for A in (chess.WHITE, chess.BLACK):
                for kind, fn in (("gate", gates), ("ctrl", controls)):
                    for typ, ctx in fn(b, A).items():
                        ev[(typ, kind)][1] += 1
                        ev[(typ, kind)][0] += bool(event(fut, A, typ, ctx))
    print("POT GATE PRECISION  games %d · horizon %d plies (event rate WITH the key precursor vs the near-miss WITHOUT)"
          % (len(games), N))
    for t in TYPES:
        (eg, ng), (ec, nc) = ev[(t, "gate")], ev[(t, "ctrl")]
        pg, pc = eg / max(ng, 1), ec / max(nc, 1)
        se = math.sqrt(pg * (1 - pg) / max(ng, 1) + pc * (1 - pc) / max(nc, 1))
        print("  %s  with %5.1f%% (n %6d) · without %5.1f%% (n %6d) · lift %.2f× · diff %+.1fpp (%.1fσ)"
              % (t, 100 * pg, ng, 100 * pc, nc, pg / max(pc, 1e-9), 100 * (pg - pc), (pg - pc) / max(se, 1e-9)))


def event(fut, A, typ, ctx):
    D = not A
    ap, dp = pawns(fut, A), pawns(fut, D)
    if typ == "T1":
        fk = chess.square_file(fut.king(D))
        return fk in (3, 4, 5) and half_open_near(fut, D, fk) > 0
    if typ == "T3":
        return any(not (dp >> s & 1) or isolated(dp, s) for s in ctx)
    if typ == "T4":
        return any(any(passed(fut, A, s) for s in chess.SquareSet(ap & FLANKS[f])) for f in ctx)
    if typ == "T5":
        return any(any(isolated(dp, s) for s in chess.SquareSet(dp & FLANKS[f])) for f in ctx)


def main():
    st = pd.read_csv(os.path.join(DATA, "fitC_stage1.csv.gz"), usecols=["game_id", "ply", "fen"])
    games = st["game_id"].unique()[:GAMES]
    st = st[st["game_id"].isin(set(games))]
    npos = 0
    on = {t: 0 for t in TYPES}
    ev = {t: [0, 0] for t in TYPES}          # [events, gated rows with a future]
    any_on = 0
    combos = {}
    for gid, g in st.groupby("game_id", sort=False):
        plies, fens = g["ply"].values, g["fen"].values
        for i in range(len(g)):
            b = chess.Board(fens[i])
            if chess.popcount(int(b.occupied)) < 20:
                continue
            npos += 1
            j = np.searchsorted(plies, plies[i] + N)
            fut = chess.Board(fens[j]) if j < len(g) else None
            types_here = set()
            for A in (chess.WHITE, chess.BLACK):
                for typ, ctx in gates(b, A).items():
                    types_here.add(typ)
                    if fut is not None:
                        ev[typ][1] += 1
                        ev[typ][0] += bool(event(fut, A, typ, ctx))
            for t in types_here:
                on[t] += 1
            if types_here:
                any_on += 1
            key = "+".join(sorted(types_here)) or "none"
            combos[key] = combos.get(key, 0) + 1
    print("POT COVERAGE  games %d · middlegame positions %d · horizon %d plies" % (len(games), npos, N))
    print("  ANY gate on (either side): %.1f%% of positions" % (100 * any_on / npos))
    for t in TYPES:
        e, n = ev[t]
        print("  %s  gate on %5.1f%% of positions · result arrives within %d plies: %5.1f%% (of %d gated side-rows)"
              % (t, 100 * on[t] / npos, N, 100 * e / max(n, 1), n))
    print("  combinations (share of positions):")
    for k, v in sorted(combos.items(), key=lambda kv: -kv[1])[:10]:
        print("    %-14s %5.1f%%" % (k, 100 * v / npos))


if __name__ == "__main__":
    main_lift() if KV.get("MODE") == "lift" else main()
