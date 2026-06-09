# -*- coding: utf-8 -*-
"""Imbalance-aware paired analysis of a UHO-book tournament (P1 vs P2, head-to-head).

The raw win count washes out the structure the UHO book is designed to expose: every opening is played
from BOTH colors on consecutive games, so in each pair P1 gets the favored side once and the disfavored
side once (P2 gets the opposite). This decomposes P1's score by whether it was handed the better or the
worse side of the (mildly imbalanced) start, so we can read CONVERSION (score when favored) vs
RESILIENCE (score when disfavored) instead of a single Elo that buries both.

How: pair games by opening_idx (from summary.csv), reconstruct each opening's post-book starting
position from a game's meta (start_fen + opening UCI), Stockfish-evaluate it ONCE for the favored side +
cp margin, bucket by margin, and tally P1's result split favored/disfavored + the 3x3 pair-outcome matrix.

Run (from NN Engine/, after the tournament finishes):
    python selfplay/paired_analysis.py --tag newstack_uho
    python selfplay/paired_analysis.py --tag newstack_uho --sf-movetime 0.2
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from arbiter import Arbiter, find_stockfish

# Imbalance buckets by |SF cp| at the start of play (the book targeted 15-90cp; a deeper re-eval spreads it).
BUCKETS = [(0, 15, "~equal"), (15, 40, "slight"), (40, 80, "edge"), (80, 1e9, "clear")]


def _bucket(cp):
    a = abs(cp)
    for lo, hi, name in BUCKETS:
        if lo <= a < hi:
            return name
    return "clear"


def _meta_of(logdir, game_id):
    """First (meta) record of a game's jsonl, or None."""
    path = os.path.join(logdir, f"game_{int(game_id):03d}", "game.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _start_position(meta):
    """Board after the recorded opening moves — the position from which the engines actually played."""
    board = chess.Board(meta.get("start_fen", chess.STARTING_FEN))
    for uci in (meta.get("opening", "") or "").split():
        try:
            board.push(chess.Move.from_uci(uci))
        except Exception:
            break
    return board


def run(args):
    logdir = os.path.join(THIS_DIR, "games", args.tag)
    summ = os.path.join(logdir, "summary.csv")
    if not os.path.exists(summ):
        print(f"[paired] no summary.csv under {logdir}", flush=True)
        return
    rows = list(csv.DictReader(open(summ)))
    # Group games by opening; each opening should have a P1-white game and a P1-black game.
    by_open = defaultdict(list)
    for r in rows:
        if r["p1_score"] in ("0.0", "0.5", "1.0"):
            by_open[r["opening_idx"]].append(r)

    sf = args.sf_path or find_stockfish()
    if not sf:
        print("[paired] no Stockfish found (set STOCKFISH_PATH or --sf-path)", flush=True)
        return
    arb = Arbiter(sf, movetime=args.sf_movetime, depth=args.sf_depth)

    # One SF eval per distinct opening (favored side + cp margin), cached by opening_idx.
    p1lbl = rows[0]["white"] if rows[0]["p1_color"] == "white" else rows[0]["black"]
    print(f"[paired] {args.tag}: {len(rows)} games, {len(by_open)} distinct openings; "
          f"evaluating openings at SF {args.sf_movetime}s ...", flush=True)
    opening_cp = {}
    done = 0
    for oidx, games in by_open.items():
        meta = _meta_of(logdir, games[0]["game"])
        if meta is None:
            continue
        try:
            cp, _, _ = arb.evaluate(_start_position(meta))   # White-POV cp
        except Exception:
            cp = None
        opening_cp[oidx] = cp
        done += 1
        if done % 100 == 0:
            print(f"[paired]   {done}/{len(by_open)} openings evaluated", flush=True)
    arb.close()

    # Tally P1's result split by favored/disfavored x bucket, and the 3x3 pair-outcome matrix.
    # conv[bucket] = [score_sum, n] when P1 had the favored side; res[bucket] likewise when disfavored.
    conv = defaultdict(lambda: [0.0, 0])
    res = defaultdict(lambda: [0.0, 0])
    pair_cell = defaultdict(int)   # (favored_result, disfavored_result) -> count, results in {W,D,L}
    RES = {"1.0": "W", "0.5": "D", "0.0": "L"}
    n_pairs = 0
    skipped = 0

    for oidx, games in by_open.items():
        cp = opening_cp.get(oidx)
        if cp is None:
            skipped += len(games)
            continue
        bk = _bucket(cp)
        favored_color = "white" if cp >= 0 else "black"
        near_equal = abs(cp) < BUCKETS[0][1]
        per = {}   # 'favored' / 'disfavored' -> P1 result letter (last game of that kind)
        for g in games:
            p1_fav = (g["p1_color"] == favored_color)
            sc = float(g["p1_score"])
            (conv if p1_fav else res)[bk][0] += sc
            (conv if p1_fav else res)[bk][1] += 1
            if not near_equal:
                per["favored" if p1_fav else "disfavored"] = RES[g["p1_score"]]
        if not near_equal and "favored" in per and "disfavored" in per:
            pair_cell[(per["favored"], per["disfavored"])] += 1
            n_pairs += 1

    def pct(d, bk):
        s, n = d[bk]
        return (100 * s / n, n) if n else (0.0, 0)

    print(f"\n==== P1 = {p1lbl}  ({args.tag}) ====")
    print("CONVERSION (P1 score when handed the FAVORED side) vs RESILIENCE (when handed the DISFAVORED side)")
    print("a fair pair of equal engines -> conv% + res% == 100; conv%+res% > 100 means P1 is the better engine in that bucket\n")
    print(f"{'bucket':>8} | {'conv% (n)':>16} | {'res% (n)':>16} | {'conv+res':>9}")
    print("-" * 60)
    for _, _, bk in BUCKETS:
        cp_, cn = pct(conv, bk)
        rp_, rn = pct(res, bk)
        tot = (cp_ + rp_) if (cn and rn) else 0.0
        print(f"{bk:>8} | {cp_:8.1f} ({cn:4d}) | {rp_:8.1f} ({rn:4d}) | {tot:8.1f}")
    # overall
    cs = sum(conv[b][0] for _, _, b in BUCKETS); cnn = sum(conv[b][1] for _, _, b in BUCKETS)
    rs = sum(res[b][0] for _, _, b in BUCKETS); rnn = sum(res[b][1] for _, _, b in BUCKETS)
    co = 100 * cs / cnn if cnn else 0; ro = 100 * rs / rnn if rnn else 0
    print("-" * 60)
    print(f"{'ALL':>8} | {co:8.1f} ({cnn:4d}) | {ro:8.1f} ({rnn:4d}) | {co+ro:8.1f}")

    print(f"\nPAIR-OUTCOME MATRIX  (non-equal openings only; {n_pairs} pairs, {skipped} games skipped/unpaired)")
    print("rows = P1 result when FAVORED, cols = P1 result when DISFAVORED")
    print(f"{'':>10}" + "".join(f"{'dis '+c:>8}" for c in "WDL"))
    for rf in "WDL":
        line = f"{'fav '+rf:>10}"
        for rd in "WDL":
            line += f"{pair_cell.get((rf, rd), 0):8d}"
        print(line)
    # the cells the user cares about, named
    print("\nkey cells:")
    print(f"  won-both          (fav W, dis W) = {pair_cell.get(('W','W'),0)}  (dominant: won ahead AND from behind)")
    print(f"  convert+hold      (fav W, dis D) = {pair_cell.get(('W','D'),0)}  (converted the edge, held the worse side)")
    print(f"  par               (fav W, dis L) = {pair_cell.get(('W','L'),0)}  (imbalance played out both ways)")
    print(f"  conversion FAIL   (fav D/L)      = {sum(pair_cell.get((x,y),0) for x in 'DL' for y in 'WDL')}  (didn't win the favored side)")
    print(f"  defensive WIN     (dis W)        = {sum(pair_cell.get((x,'W'),0) for x in 'WDL')}  (won from the disfavored side)")


def main():
    ap = argparse.ArgumentParser(description="Imbalance-aware paired convert/defend analysis of a UHO tournament.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.2)
    ap.add_argument("--sf-depth", type=int, default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
