# -*- coding: utf-8 -*-
"""Post-hoc Stockfish analysis for recorded self-play games.

Stockfish is a pure function of the recorded per-ply FEN, so we score games AFTER they finish — the
tournament runs at full engine speed and this pass can run SF *deeper* than any live budget. For each
game it writes an annotated JSONL + PGN (engine eval beside SF cp) and computes per-game
"interestingness" metrics, then ranks the whole tournament in analysis.csv so the worth-watching
games surface (biggest engine-vs-SF disagreement in PLAYABLE positions, blunders, cp-loss).

Run (from NN Engine/):
  python selfplay/annotate.py --tag asp_500_vs_0                 # whole tournament (runs SF)
  python selfplay/annotate.py --tag asp_500_vs_0 --recompute     # rebuild metrics from saved SF (no SF)
  python selfplay/annotate.py --game selfplay/games/<tag>/game_007/game.jsonl
  python selfplay/annotate.py --tag <tag> --sf-depth 18          # fixed depth instead of time

Eval scales: the engine reports milli-pawns (pawn=1000, mate ~9.99e6); Stockfish reports centipawns
(pawn=100, mate=100000 via mate_score). Mates would dominate raw divergence/cp-loss, so the metrics
CAP both: divergence is measured only where BOTH sides see a non-decided position (|eval| <= DIV_CAP
pawns), and cp-loss clamps each eval to +-CPL_CAP centipawns before differencing.
"""

import os
import sys
import csv
import json
import glob
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from arbiter import Arbiter, find_stockfish
from selfplay import write_pgn

BLUNDER_CP = 150     # a move that swings SF's (clamped) assessment against the mover by > this = blunder
DIV_CAP_P = 15.0     # only measure engine-vs-SF divergence where BOTH see |eval| <= this (pawns)
CPL_CAP_CP = 1500    # clamp each SF eval to +-this (centipawns) before cp-loss differencing

ANALYSIS_FIELDS = ["game", "white", "black", "result", "plies", "max_div_pawns", "max_div_ply",
                   "white_avg_cploss", "black_avg_cploss", "white_blunders", "black_blunders"]


def _read_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _clamp_cp(c):
    return max(-CPL_CAP_CP, min(CPL_CAP_CP, c))


def _metrics(meta, result_rec, moves):
    """Compute the per-game metrics row from move records that already carry `sf_cp` (White-POV cp).
    Divergence is ignored in decided/mate positions; cp-loss clamps to a decided threshold so a
    mate-vs-non-mate transition can't masquerade as a giant blunder."""
    max_div, max_div_ply = 0.0, None
    cploss = {"white": [0.0, 0], "black": [0.0, 0]}
    blunders = {"white": 0, "black": 0}
    prev_cp = None
    for r in moves:
        cp = r.get("sf_cp")
        ev = r.get("eval_white_pov")
        # Engine-vs-SF divergence, but only where BOTH consider the position non-decided (no mate noise).
        if isinstance(ev, int) and isinstance(cp, int):
            ep, sp = ev / 1000.0, cp / 100.0
            if abs(ep) <= DIV_CAP_P and abs(sp) <= DIV_CAP_P:
                d = abs(ep - sp)
                if d > max_div:
                    max_div, max_div_ply = d, r.get("ply")
        # Centipawn loss: how far SF's (clamped) White-POV assessment swung against the mover.
        if not r.get("opening") and isinstance(cp, int) and isinstance(prev_cp, int):
            delta = _clamp_cp(cp) - _clamp_cp(prev_cp)
            side = r["color"]
            loss = max(0, -delta if side == "white" else delta)
            cploss[side][0] += loss
            cploss[side][1] += 1
            if loss > BLUNDER_CP:
                blunders[side] += 1
        if isinstance(cp, int):
            prev_cp = cp

    def avg(side):
        s, c = cploss[side]
        return round(s / c, 1) if c else 0.0

    return {
        "game": os.path.basename(os.path.dirname(meta.get("_path", ""))) or meta.get("_game", ""),
        "white": meta.get("white", ""), "black": meta.get("black", ""),
        "result": result_rec.get("result", "*"), "plies": len(moves),
        "max_div_pawns": round(max_div, 2), "max_div_ply": max_div_ply,
        "white_avg_cploss": avg("white"), "black_avg_cploss": avg("black"),
        "white_blunders": blunders["white"], "black_blunders": blunders["black"],
    }


def _split(recs):
    meta = next((r for r in recs if r.get("type") == "meta"), {})
    result_rec = next((r for r in recs if r.get("type") == "result"), {})
    moves = [r for r in recs if r.get("type") == "move" and r.get("uci")]
    return meta, result_rec, moves


def annotate_game(jsonl_path, arbiter):
    """Run SF over each recorded FEN, write annotated JSONL + PGN, and return the metrics row."""
    recs = _read_jsonl(jsonl_path)
    meta, result_rec, moves = _split(recs)
    start_fen = meta.get("start_fen", chess.STARTING_FEN)
    for r in moves:
        try:
            cp, best, depth = arbiter.evaluate(chess.Board(r["fen"]))
        except Exception:
            cp, best, depth = None, None, None
        r["sf_cp"], r["sf_best"], r["sf_depth"] = cp, best, depth
    gdir = os.path.dirname(jsonl_path)
    with open(os.path.join(gdir, "game.annotated.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    write_pgn(gdir, meta, moves, result_rec.get("result", "*"), result_rec.get("reason", ""),
              start_fen, filename="game.annotated.pgn")
    meta["_path"] = jsonl_path
    return _metrics(meta, result_rec, moves)


def recompute_game(annotated_path):
    """Rebuild the metrics row from a saved game.annotated.jsonl (SF cp already present) — no SF."""
    recs = _read_jsonl(annotated_path)
    meta, result_rec, moves = _split(recs)
    meta["_path"] = annotated_path
    return _metrics(meta, result_rec, moves)


def _write_analysis(logdir, rows):
    rows.sort(key=lambda r: (r["max_div_pawns"], r["white_blunders"] + r["black_blunders"]),
              reverse=True)
    csv_path = os.path.join(logdir, "analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANALYSIS_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[annotate] ranked by engine-vs-SF divergence (playable positions) -> {csv_path}", flush=True)
    if rows:
        t = rows[0]
        print(f"[annotate] most interesting: {t['game']} — engine vs SF diverge {t['max_div_pawns']} "
              f"pawns at ply {t['max_div_ply']}", flush=True)


def annotate_tag(tag, depth=None, movetime=None, sf_path=None):
    logdir = os.path.join(THIS_DIR, "games", tag)
    jsonls = sorted(glob.glob(os.path.join(logdir, "game_*", "game.jsonl")))
    if not jsonls:
        print(f"[annotate] no games found under {logdir}", flush=True)
        return
    sf = sf_path or find_stockfish()
    if not sf:
        print("[annotate] no Stockfish binary found (set STOCKFISH_PATH or --sf-path)", flush=True)
        return
    mt = movetime if (movetime or depth) else 0.5
    arbiter = Arbiter(sf, movetime=mt, depth=depth)
    print(f"[annotate] {len(jsonls)} games, Stockfish {f'depth {depth}' if depth else f'{mt}s/pos'}", flush=True)
    rows = []
    try:
        for jp in jsonls:
            rows.append(_log_row(annotate_game(jp, arbiter)))
    finally:
        arbiter.close()
    _write_analysis(logdir, rows)


def recompute_tag(tag):
    """Rebuild analysis.csv from already-annotated games (no Stockfish run)."""
    logdir = os.path.join(THIS_DIR, "games", tag)
    anns = sorted(glob.glob(os.path.join(logdir, "game_*", "game.annotated.jsonl")))
    if not anns:
        print(f"[annotate] no annotated games under {logdir} (run without --recompute first)", flush=True)
        return
    print(f"[annotate] recomputing metrics from {len(anns)} annotated games (no SF)", flush=True)
    rows = [_log_row(recompute_game(a)) for a in anns]
    _write_analysis(logdir, rows)


def _log_row(row):
    print(f"[annotate] {row['game']}: {row['result']}  maxDiv {row['max_div_pawns']}p@ply"
          f"{row['max_div_ply']}  cploss W{row['white_avg_cploss']}/B{row['black_avg_cploss']}  "
          f"blunders W{row['white_blunders']}/B{row['black_blunders']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description="Post-hoc Stockfish annotation for recorded self-play games.")
    ap.add_argument("--tag", help="annotate every game under games/<tag>/ + write analysis.csv")
    ap.add_argument("--game", help="annotate a single game.jsonl")
    ap.add_argument("--recompute", action="store_true",
                    help="with --tag: rebuild analysis.csv from saved SF cp (no Stockfish run)")
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-depth", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=None, help="default 0.5s when neither set")
    args = ap.parse_args()

    if args.recompute and args.tag:
        recompute_tag(args.tag)
    elif args.game:
        sf = args.sf_path or find_stockfish()
        if not sf:
            print("[annotate] no Stockfish binary found", flush=True)
            return
        mt = args.sf_movetime if (args.sf_movetime or args.sf_depth) else 0.5
        arb = Arbiter(sf, movetime=mt, depth=args.sf_depth)
        try:
            print(json.dumps(annotate_game(args.game, arb), indent=2), flush=True)
        finally:
            arb.close()
    elif args.tag:
        annotate_tag(args.tag, depth=args.sf_depth, movetime=args.sf_movetime, sf_path=args.sf_path)
    else:
        ap.error("need --tag or --game")


if __name__ == "__main__":
    main()
