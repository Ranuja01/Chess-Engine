# -*- coding: utf-8 -*-
"""Misevaluation-pattern miner — find RECURRING static-eval errors by material configuration and across
ply transitions, the two axes term_diag.py never aggregates over (it only buckets by phase).

term_diag localized that our static eval's worst clean offender is "R+N vs R" (a theoretical draw we
read ~-700) -- but only by eyeballing a top-20 list. This finds such clusters systematically, separates
fixable directional bias from irreducible scatter, and flags transformations (trades/simplifications)
into positions we misprice -- the practical strength cost.

All three analyses are PURE OFFLINE reads of the already-annotated games (no engine, no Stockfish).
Every position carries our static eval (eval_breakdown), SF's NNUE static (sf_static_cp) and SF's SEARCH
eval (sf_cp, the gold standard). Where NNUE-static disagrees with search it is untrustworthy, so the
near-equal analyses apply the same trust gate as term_diag (is_trusted).

  [1] clusters   -- positions BOTH NNUE and search call level but we read big, grouped by
                    (material config x co-dominant terms), ranked by frequency x over-read.
  [2] bias       -- per material configuration, the SIGNED static error (our - NNUE), canonicalized so
                    colour-mirror configs aggregate. High |mean| / bias-ratio = fixable directional bias;
                    mean ~0 with large |.| = irreducible scatter.
  [3] transforms -- consecutive plies where a trade/simplification lands in a position we misprice.
                    The valuable case is the SILENT misprice (the trade is objectively fine per search,
                    but our eval mis-scores the result -- positions our search then steers toward).

Run from NN Engine/:
    python selfplay/pattern_diag.py --tag hlmr_more1_lightning
    python selfplay/pattern_diag.py --tag hlmr_more1_lightning --mode clusters --coarse
    python selfplay/pattern_diag.py --tag hlmr_more1_lightning --no-trust        # A/B the trust gate
"""

import os
import sys
import glob
import argparse
import statistics as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from deep_diag import (load_game, piece_count, our_static_cp,  # noqa: E402
                       CAL_CAP_CP, CONV_T)
from term_diag import (material_sig, is_trusted, cp, _stat_block,  # noqa: E402
                       NEAR_EQ_CP, TRUST_WINPCT, DISP_TERMS, ALLTERMS)

GAMES_DIR = os.path.join(THIS_DIR, "games")
PIECES = "QRBNP"           # ordering for the signed material-delta vector


def clamp(x, cap=CAL_CAP_CP):
    return max(-cap, min(cap, x))


def pos_key(fen):
    """Dedup key for an identical position: board + side-to-move + castling + ep (drop the clocks)."""
    return " ".join(fen.split()[:4])


def _counts(fen):
    b = fen.split()[0]
    return ({p: b.count(p) for p in PIECES}, {p: b.count(p.lower()) for p in PIECES})


def material_key(fen, coarse=False):
    """material_sig (W:.. B:..) but, when coarse, dropping pawns so e.g. R+N-vs-R merges across pawn
    counts. The non-coarse case is exactly term_diag.material_sig (reused)."""
    if not coarse:
        return material_sig(fen)
    w, bl = _counts(fen)
    ws = "".join(f"{w[p]}{p}" for p in "QRBN" if w[p])
    bs = "".join(f"{bl[p]}{p}" for p in "QRBN" if bl[p])
    return f"W:{ws or '-'} B:{bs or '-'}"


def material_delta(fen):
    """Signed per-type material vector (Q,R,B,N,P) = white - black."""
    w, bl = _counts(fen)
    return tuple(w[p] - bl[p] for p in PIECES)


def canon_delta(fen, err, coarse=False):
    """Canonicalize the material-delta vector by lexicographic orientation so colour-mirror configs fold
    together: if d is lex-negative, mirror it (negate) AND flip the error sign. Returns (oriented d,
    oriented err). Makes 'stronger side up a knight' aggregate regardless of which colour holds it,
    with no notion of 'stronger side' needed. d==0 is its own fixed point."""
    d = material_delta(fen)
    if coarse:
        d = d[:4] + (0,)
    if d < tuple(0 for _ in d):
        d = tuple(-x for x in d)
        err = -err
    return d, err


def delta_label(d):
    parts = [f"{v:+d}{p}" for p, v in zip(PIECES, d) if v]
    return "".join(parts) or "even"


def codominant(eb, frac=0.5):
    """The set of terms whose |White-POV cp| is >= frac of the row's largest term magnitude. Captures
    co-firing drivers (the endgame eval rides on top of a piece-type term) instead of a brittle single
    argmax that would split one config across two term rows."""
    vals = [(t, abs(cp(eb.get(t, 0)))) for t in DISP_TERMS]
    mx = max((v for _, v in vals), default=0.0)
    if mx < 1.0:
        return frozenset()
    return frozenset(t.replace("pt_", "") for t, v in vals if v >= frac * mx)


def iter_games(tags):
    """Yield each game's move list (ply order) across one or more tags (pooled)."""
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        for gdir in sorted(glob.glob(os.path.join(GAMES_DIR, tag, "game_*"))):
            moves = load_game(gdir)
            if moves:
                yield moves


def iter_moves(tags):
    """Yield every move dict across one or more tags (pooled), in ply order per game."""
    for moves in iter_games(tags):
        for m in moves:
            yield m


def _playable(m):
    return not (m.get("opening") or m.get("booked")) and m.get("eval_breakdown") is not None


# --------------------------------------------------------------------------------------------------
# [1] Offender clusters
# --------------------------------------------------------------------------------------------------
def clusters(tags, label, args):
    cl = {}  # (material_key, codominant frozenset) -> {"distinct": {pkey: (our, pawns, fen, eb)}, "total": n}
    for m in iter_moves(tags):
        if not _playable(m):
            continue
        eb = m["eval_breakdown"]
        sfs, sfc = m.get("sf_static_cp"), m.get("sf_cp")
        if sfs is None:
            continue
        if abs(sfs) >= NEAR_EQ_CP:
            continue
        if args.trust is not None and not is_trusted(sfs, sfc, args.trust):
            continue
        our = our_static_cp(eb)
        if abs(our) < args.big_cp:
            continue
        fen = m["fen"]
        key = (material_key(fen, args.coarse), codominant(eb))
        bucket = cl.setdefault(key, {"distinct": {}, "total": 0})
        bucket["total"] += 1
        pk = pos_key(fen)
        if pk not in bucket["distinct"]:
            bucket["distinct"][pk] = (our, piece_count(fen) - 2, fen, eb)  # -2 = drop the two kings

    gate = _gate_desc(args)
    print("\n" + "=" * 100)
    print(f"[1] OFFENDER CLUSTERS  ({label})  --  both NNUE & search call it level (|sf_static|<{NEAR_EQ_CP}, "
          f"{gate}), we read |our|>={args.big_cp:.0f}cp")
    print("    grouped by (material x co-dominant terms); ranked by n_distinct x mean|our_cp|")
    if not cl:
        print("    (no clusters)")
        return
    rows = []
    for (mat, terms), b in cl.items():
        vals = list(b["distinct"].values())
        ours = [v[0] for v in vals]
        pawns = [v[1] for v in vals]
        nd = len(vals)
        rows.append((nd * st.mean([abs(x) for x in ours]), nd, b["total"], st.mean(ours),
                     st.mean([abs(x) for x in ours]), st.mean(pawns), mat, terms, vals))
    rows.sort(key=lambda r: r[0], reverse=True)

    print(f"\n     {'nDist':>5} {'nTot':>5} {'mean_our':>9} {'mean|our|':>10} {'pcs':>4}  "
          f"{'material':<22} co-dominant terms")
    for _, nd, ntot, mo, mao, pawns, mat, terms, _vals in rows[:args.top]:
        ts = "+".join(sorted(terms)) if terms else "-"
        print(f"     {nd:>5} {ntot:>5} {mo:>9.0f} {mao:>10.0f} {pawns:>4.1f}  {mat:<22} {ts}")

    # drill into the top few clusters with the reused per-term stat table
    print("\n  -- per-term breakdown of the top clusters (White-POV cp; ranked by mean|.|) --")
    for _, nd, _ntot, _mo, _mao, _pawns, mat, terms, vals in rows[:args.drill]:
        ts = "+".join(sorted(terms)) if terms else "-"
        ebs = [v[3] for v in vals]
        label = f"{mat}  [{ts}]  (n_distinct={nd})  e.g. {vals[0][2]}"
        _stat_block(label, {t: [cp(eb.get(t, 0)) for eb in ebs]
                            for t in ALLTERMS + ["advanced_endgame_total"]})
        print()


# --------------------------------------------------------------------------------------------------
# [2] Signed bias by config
# --------------------------------------------------------------------------------------------------
def _ms(errs):
    """(n, mean) for a list; (0, 0.0) if empty."""
    return (len(errs), st.mean(errs) if errs else 0.0)


def bias(tags, label, args):
    seen = set()
    buckets = {}            # oriented d -> {"mid": [err], "end": [err]}
    zero = {"mid": [], "end": []}   # d==0 (material-balanced): scatter only
    for m in iter_moves(tags):
        if not _playable(m):
            continue
        eb = m["eval_breakdown"]
        sfs, sfc = m.get("sf_static_cp"), m.get("sf_cp")
        if sfs is None:
            continue
        if args.trust is not None and not is_trusted(sfs, sfc, args.trust):
            continue
        pk = pos_key(m["fen"])
        if pk in seen:
            continue
        seen.add(pk)
        err = clamp(our_static_cp(eb)) - clamp(sfs)
        d, oerr = canon_delta(m["fen"], err, args.coarse)
        ph = "end" if piece_count(m["fen"]) < 17 else "mid"   # endgame = early/late-end (pc<17)
        if all(x == 0 for x in d):
            zero[ph].append(err)                              # balanced: keep raw (unoriented) error
        else:
            buckets.setdefault(d, {"mid": [], "end": []})[ph].append(oerr)

    print("\n" + "=" * 100)
    print(f"[2] SIGNED BIAS BY CONFIG  ({label})  --  static error our-NNUE per material delta, split "
          f"MID vs END ({_gate_desc(args)}; n>={args.min_n})")
    print("    canonicalized to the stronger side; high |mean| & bias-ratio = FIXABLE directional bias. "
          "MID-vs-END tells whether the over-read is an")
    print("    endgame-CONVERSION gap (safe to phase-gate a scale factor) or a midgame material-vs-"
          "COMPENSATION gap (gambit-relevant).")
    rows = []
    for d, b in buckets.items():
        allerrs = b["mid"] + b["end"]
        n = len(allerrs)
        if n < args.min_n:
            continue
        mean = st.mean(allerrs)
        mabs = st.mean([abs(e) for e in allerrs])
        sd = st.pstdev(allerrs) if n > 1 else 0.0
        ratio = abs(mean) / mabs if mabs else 0.0
        tlike = mean / (sd / (n ** 0.5)) if sd > 0 else 0.0
        rows.append((abs(mean), n, mean, ratio, tlike, _ms(b["mid"]), _ms(b["end"]), d))
    rows.sort(key=lambda r: r[0], reverse=True)

    print(f"\n     {'delta':<14} {'n':>5} {'mean':>7} {'bias_r':>7} {'t':>6}   "
          f"{'mid_n':>6} {'mid_mean':>9}   {'end_n':>6} {'end_mean':>9}")
    for _, n, mean, ratio, tlike, (mn, mm), (en, em), d in rows[:args.top]:
        print(f"     {delta_label(d):<14} {n:>5} {mean:>7.0f} {ratio:>7.2f} {tlike:>6.1f}   "
              f"{mn:>6} {mm:>9.0f}   {en:>6} {em:>9.0f}")
    zall = zero["mid"] + zero["end"]
    if zall:
        (mn, mm), (en, em) = _ms(zero["mid"]), _ms(zero["end"])
        print(f"\n     {'even (d=0)':<14} {len(zall):>5} {st.mean(zall):>7.0f} {'--':>7} {'--':>6}   "
              f"{mn:>6} {mm:>9.0f}   {en:>6} {em:>9.0f}   (balanced: scatter only)")


# --------------------------------------------------------------------------------------------------
# [3] Transformation leaks
# --------------------------------------------------------------------------------------------------
def transforms(tags, label, args):
    blunder = {}   # landing material_key -> leak list   (objective loss AND we misprice)
    silent = {}    # landing material_key -> leak list   (loss~0 but we misprice -- the dangerous one)
    seen_b, seen_s = set(), set()

    for moves in iter_games(tags):
        for i in range(len(moves) - 1):
            a, b = moves[i], moves[i + 1]
            if b.get("ply") != a.get("ply", -99) + 1:
                continue
            if not (a.get("uci") and b.get("uci")):
                continue
            if a.get("opening") or b.get("opening") or a.get("booked") or b.get("booked"):
                continue
            sfa, sfb = a.get("sf_cp"), b.get("sf_cp")
            ebb = b.get("eval_breakdown")
            if sfa is None or sfb is None or ebb is None:
                continue
            if abs(sfa) >= CAL_CAP_CP or abs(sfb) >= CAL_CAP_CP:      # drop decided/mate: loss & misprice meaningless
                continue
            if material_sig(a["fen"]) == material_sig(b["fen"]):      # no trade/simplification
                continue
            mover = b.get("color")                                    # ply i+1 made the trade
            sign = 1 if mover == "white" else -1
            loss = sign * sfa - sign * sfb                            # mover-POV cp lost across the trade (sf within ±cap)
            misprice = abs(clamp(our_static_cp(ebb)) - sfb)          # how wrong our static is at the landing
            if misprice < args.big_cp:
                continue
            rec = (loss, misprice, our_static_cp(ebb), sfb, b.get("uci") == b.get("sf_best"), b["fen"])
            mkey = material_key(b["fen"], args.coarse)
            pk = pos_key(b["fen"])
            if loss >= args.loss_cp:
                if pk not in seen_b:
                    seen_b.add(pk)
                    blunder.setdefault(mkey, []).append(rec)
            elif abs(loss) < NEAR_EQ_CP and abs(sfb) < CONV_T:        # silent: trade is FINE and landing is contested
                if pk not in seen_s:
                    seen_s.add(pk)
                    silent.setdefault(mkey, []).append(rec)

    print("\n" + "=" * 100)
    print(f"[3] TRANSFORMATION LEAKS  ({tag})  --  trades/simplifications landing where our static "
          f"misprices by >={args.big_cp:.0f}cp (vs SF search)")
    _print_leaks("(b) SILENT misprice  (|loss|<%dcp -- trade is FINE per search, we mis-score the result; "
                 "positions our search steers TOWARD)" % NEAR_EQ_CP, silent, args)
    _print_leaks("(a) BLUNDER into misprice  (loss>=%dcp AND we misprice)" % args.loss_cp, blunder, args)


def _print_leaks(title, groups, args):
    print(f"\n  {title}")
    if not groups:
        print("    (none)")
        return
    rows = []
    for mkey, recs in groups.items():
        n = len(recs)
        rows.append((n * st.mean([r[1] for r in recs]), n, st.mean([r[0] for r in recs]),
                     st.mean([r[1] for r in recs]), st.mean([r[2] for r in recs]),
                     st.mean([r[3] for r in recs]), st.mean([1.0 if r[4] else 0.0 for r in recs]),
                     mkey, recs))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"     {'n':>4} {'mean_loss':>9} {'misprice':>9} {'mean_our':>9} {'mean_sf':>8} "
          f"{'=sfbest':>8}  {'landing material':<22} example FEN")
    for _, n, ml, mp, mo, msf, sb, mkey, recs in rows[:args.top]:
        print(f"     {n:>4} {ml:>9.0f} {mp:>9.0f} {mo:>9.0f} {msf:>8.0f} {sb*100:>7.0f}%  "
              f"{mkey:<22} {recs[0][5]}")


def _gate_desc(args):
    return "trust gate ON" if args.trust is not None else "trust gate OFF"


def analyze(tags, label, args):
    print("\n" + "#" * 100)
    print(f"# {label}")
    print("#" * 100)
    if args.mode in ("clusters", "all"):
        clusters(tags, label, args)
    if args.mode in ("bias", "all"):
        bias(tags, label, args)
    if args.mode in ("transforms", "all"):
        transforms(tags, label, args)


def main():
    ap = argparse.ArgumentParser(description="Misevaluation-pattern miner (offline, by material config "
                                             "& across ply transitions).")
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--mode", choices=["clusters", "bias", "transforms", "all"], default="all")
    ap.add_argument("--trust", type=float, default=TRUST_WINPCT,
                    help="win%% agreement gate between NNUE-static and SF-search (default %(default)s)")
    ap.add_argument("--no-trust", action="store_true", help="disable the trust gate (raw NNUE-static)")
    ap.add_argument("--big-cp", type=float, default=150.0,
                    help="|our_static| threshold for an offender / misprice (default %(default)s)")
    ap.add_argument("--loss-cp", type=float, default=100.0,
                    help="[3] mover-POV cp-loss threshold for a blunder-into-misprice (default %(default)s)")
    ap.add_argument("--min-n", type=int, default=25, help="[2] min bucket size to rank (default %(default)s)")
    ap.add_argument("--coarse", action="store_true", help="drop pawns from the material key (merge by pieces)")
    ap.add_argument("--pool", action="store_true", help="treat all --tag args as ONE pooled corpus (e.g. all standard games)")
    ap.add_argument("--top", type=int, default=15, help="rows per section (default %(default)s)")
    ap.add_argument("--drill", type=int, default=4, help="[1] clusters to per-term drill (default %(default)s)")
    args = ap.parse_args()
    args.trust = None if args.no_trust else args.trust
    if args.pool and len(args.tag) > 1:
        analyze(args.tag, "POOL[" + "+".join(args.tag) + "]", args)
    else:
        for t in args.tag:
            analyze([t], t, args)


if __name__ == "__main__":
    main()
