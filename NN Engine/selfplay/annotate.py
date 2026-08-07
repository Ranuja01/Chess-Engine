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
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from arbiter import Arbiter, find_stockfish
from selfplay import write_pgn

BLUNDER_CP = 150     # a move that swings SF's (clamped) assessment against the mover by > this = blunder
DIV_CAP_P = 15.0     # only measure engine-vs-SF divergence where BOTH see |eval| <= this (pawns)
CPL_CAP_CP = 1500    # clamp each SF eval to +-this (centipawns) before cp-loss differencing

ANALYSIS_FIELDS = ["game", "white", "black", "result", "plies", "max_div_pawns", "max_div_ply",
                   "white_avg_cploss", "black_avg_cploss", "white_blunders", "black_blunders",
                   "mean_calib_gap", "mean_pvboost"]


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
    # our static eval vs SF's static (NNUE) eval, both White-POV pawns; + the piece_value_boost contribution.
    calib_sum, calib_n, pvboost_sum, pvboost_n = 0.0, 0, 0.0, 0
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
        # Calibration: our static term breakdown (absolute Black-positive milli-pawns) vs SF's static eval.
        bd = r.get("eval_breakdown")
        if isinstance(bd, dict) and not bd.get("checkmate"):
            our_sw = -bd.get("total", 0) / 1000.0
            pvboost_sum += -bd.get("piece_value_boost", 0) / 1000.0
            pvboost_n += 1
            sfs = r.get("sf_static_cp")
            if isinstance(sfs, (int, float)):
                sf_sw = sfs / 100.0
                if abs(our_sw) <= DIV_CAP_P and abs(sf_sw) <= DIV_CAP_P:
                    calib_sum += our_sw - sf_sw
                    calib_n += 1
        if isinstance(cp, int):
            prev_cp = cp

    def avg(side):
        s, c = cploss[side]
        return round(s / c, 1) if c else 0.0

    return {
        # Nested layout identifies a game by its directory; flat layout by the filename stem. Using the
        # directory for both would label all 600 flat games with the tag name and make the ranking useless.
        "game": (os.path.basename(meta.get("_path", ""))[:-len(".jsonl")]
                 if os.path.basename(meta.get("_path", "")) not in ("game.jsonl", "")
                 else os.path.basename(os.path.dirname(meta.get("_path", "")))) or meta.get("_game", ""),
        "white": meta.get("white", ""), "black": meta.get("black", ""),
        "result": result_rec.get("result", "*"), "plies": len(moves),
        "max_div_pawns": round(max_div, 2), "max_div_ply": max_div_ply,
        "white_avg_cploss": avg("white"), "black_avg_cploss": avg("black"),
        "white_blunders": blunders["white"], "black_blunders": blunders["black"],
        "mean_calib_gap": round(calib_sum / calib_n, 2) if calib_n else "",
        "mean_pvboost": round(pvboost_sum / pvboost_n, 2) if pvboost_n else "",
    }


def _split(recs):
    meta = next((r for r in recs if r.get("type") == "meta"), {})
    result_rec = next((r for r in recs if r.get("type") == "result"), {})
    moves = [r for r in recs if r.get("type") == "move" and r.get("uci")]
    return meta, result_rec, moves


def _load_ai():
    """Load the C++ ChessAI engine for our static-eval term breakdown. Guarded: returns None (annotation
    degrades to SF-only) if the .so can't be imported (e.g. headless build mismatch)."""
    try:
        engine_dir = os.path.dirname(THIS_DIR)  # NN Engine/  (has ChessAI*.so)
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        from ChessAI import ChessAI
        return ChessAI(None, None, chess.Board(), True)
    except Exception as e:
        print(f"[annotate] ChessAI unavailable ({e}) -- recording SF only, no eval_breakdown", flush=True)
        return None


class _LockedAI:
    """Serialize ev_breakdown across annotation threads. Our ChessAI static eval shares C++ globals
    (the eval cache) that aren't reentrant, so concurrent breakdown calls would race. The breakdown is
    microseconds vs Stockfish's per-position movetime, so the lock costs ~nothing while the expensive
    SF analysis still runs fully in parallel (each worker has its own SF process)."""

    def __init__(self, ai, lock):
        self._ai = ai
        self._lock = lock

    def ev_breakdown(self, board):
        with self._lock:
            return self._ai.ev_breakdown(board)


def annotate_game(jsonl_path, arbiter, ai=None):
    """Run SF over each recorded FEN, write annotated JSONL + PGN, and return the metrics row.
    Also records SF's static (NNUE) eval and our static term breakdown per move for eval calibration."""
    recs = _read_jsonl(jsonl_path)
    meta, result_rec, moves = _split(recs)
    start_fen = meta.get("start_fen", chess.STARTING_FEN)
    for r in moves:
        try:
            cp, best, depth = arbiter.evaluate(chess.Board(r["fen"]))
        except Exception:
            cp, best, depth = None, None, None
        r["sf_cp"], r["sf_best"], r["sf_depth"] = cp, best, depth
        try:
            r["sf_static_cp"] = arbiter.evaluate_static(chess.Board(r["fen"]))
        except Exception:
            r["sf_static_cp"] = None
        if ai is not None:
            try:
                r["eval_breakdown"] = ai.ev_breakdown(chess.Board(r["fen"]))
            except Exception:
                r["eval_breakdown"] = None
    # Nested layout (…/game_007/game.jsonl) gets a fixed filename inside its own directory; the flat
    # layout (…/game_007.jsonl) must keep the per-game stem or all 600 games would overwrite one file.
    gdir = os.path.dirname(jsonl_path)
    stem = os.path.basename(jsonl_path)
    flat = stem != "game.jsonl"
    out_jsonl = (stem[:-len(".jsonl")] + ".annotated.jsonl") if flat else "game.annotated.jsonl"
    out_pgn = (stem[:-len(".jsonl")] + ".annotated.pgn") if flat else "game.annotated.pgn"
    with open(os.path.join(gdir, out_jsonl), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    write_pgn(gdir, meta, moves, result_rec.get("result", "*"), result_rec.get("reason", ""),
              start_fen, filename=out_pgn)
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


def annotate_tag(tag, depth=None, movetime=None, sf_path=None, concurrency=1):
    logdir = os.path.join(THIS_DIR, "games", tag)
    # Two archive layouts exist. The older harness wrote games/<tag>/game_NNN/game.jsonl; the current
    # gauntlet writes them FLAT as games/<tag>/game_NNN.jsonl. Only the nested form was matched here, so
    # this tool silently reported "no games found" for every tournament the current harness has produced
    # -- tens of thousands of games that were never annotatable.
    jsonls = sorted(glob.glob(os.path.join(logdir, "game_*", "game.jsonl")))
    if not jsonls:
        flat = [p for p in sorted(glob.glob(os.path.join(logdir, "game_*.jsonl")))
                if not p.endswith(".annotated.jsonl")]
        if flat:
            # The flat files the current gauntlet writes contain ONLY per-ply move records -- no meta
            # record (white/black) and no result record. Every metric here is computed from those two,
            # so the pass completes and reports zeros for everything rather than failing. Refuse instead:
            # a silent zero is indistinguishable from "the engines agreed everywhere".
            print("[annotate] %d flat game_*.jsonl found under %s, but they carry no meta/result "
                  "records -- this tool's metrics cannot be computed from them. Use the position-bank "
                  "pipeline (build_position_bank.py -> add_sf18_labels.py) for a disagreement corpus."
                  % (len(flat), logdir), flush=True)
            return
    if not jsonls:
        print(f"[annotate] no games found under {logdir}", flush=True)
        return
    sf = sf_path or find_stockfish()
    if not sf:
        print("[annotate] no Stockfish binary found (set STOCKFISH_PATH or --sf-path)", flush=True)
        return
    mt = movetime if (movetime or depth) else 0.5
    ai = _load_ai()
    conc = max(1, concurrency)
    print(f"[annotate] {len(jsonls)} games, Stockfish {f'depth {depth}' if depth else f'{mt}s/pos'}, "
          f"concurrency={conc}{' + ChessAI eval breakdown' if ai is not None else ''}", flush=True)
    rows = []
    t0 = time.time()
    n_pos = 0
    if conc <= 1:
        arbiter = Arbiter(sf, movetime=mt, depth=depth)
        try:
            for jp in jsonls:
                row = _log_row(annotate_game(jp, arbiter, ai))
                n_pos += row.get("plies", 0)
                rows.append(row)
        finally:
            arbiter.close()
    else:
        # Game-level parallelism: each worker steps one game through its OWN Stockfish process
        # (SimpleEngine isn't reentrant); the shared ChessAI breakdown is lock-guarded.
        _tls = threading.local()
        arbiters = []
        arb_lock = threading.Lock()

        def thread_arbiter():
            a = getattr(_tls, "arb", None)
            if a is None:
                a = Arbiter(sf, movetime=mt, depth=depth)
                _tls.arb = a
                with arb_lock:
                    arbiters.append(a)
            return a

        locked_ai = _LockedAI(ai, threading.Lock()) if ai is not None else None

        def work(jp):
            return annotate_game(jp, thread_arbiter(), locked_ai)

        try:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                futs = [pool.submit(work, jp) for jp in jsonls]
                for fut in as_completed(futs):
                    row = _log_row(fut.result())
                    n_pos += row.get("plies", 0)
                    rows.append(row)
        finally:
            for a in arbiters:
                a.close()
    dt = time.time() - t0
    # Measured throughput, so future overnight sizing uses real ms/pos instead of an estimate.
    ms_pos = (dt / n_pos * 1000.0) if n_pos else 0.0
    print(f"[annotate] timing: {len(jsonls)} games / {n_pos} positions in {dt:.0f}s "
          f"= {ms_pos:.0f} ms/pos ({n_pos / dt:.1f} pos/s) at SF {mt}s/pos" if dt > 0 else
          "[annotate] timing: no positions", flush=True)
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


def reeval_tag(tag):
    """Refresh ONLY our eval_breakdown on already-annotated games — re-run ev_breakdown on each saved
    FEN with the current ChessAI build, preserving the SF fields (sf_cp / sf_static_cp / sf_best / ...).
    Use after an eval rebuild to pick up new breakdown accumulators without a full (SF) re-annotation."""
    logdir = os.path.join(THIS_DIR, "games", tag)
    anns = sorted(glob.glob(os.path.join(logdir, "game_*", "game.annotated.jsonl")))
    if not anns:
        print(f"[annotate] no annotated games under {logdir} (run a normal annotation first)", flush=True)
        return
    ai = _load_ai()
    if ai is None:
        print("[annotate] ChessAI unavailable — cannot re-eval", flush=True)
        return
    t0 = time.time()
    n = 0
    for a in anns:
        recs = _read_jsonl(a)
        for r in recs:
            if r.get("type") == "move" and r.get("fen"):
                try:
                    r["eval_breakdown"] = ai.ev_breakdown(chess.Board(r["fen"]))
                    n += 1
                except Exception:
                    r["eval_breakdown"] = None
        with open(a, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    print(f"[annotate] re-eval: refreshed eval_breakdown on {len(anns)} games / {n} positions "
          f"in {dt:.0f}s ({n / dt:.0f} pos/s) — SF fields preserved" if dt > 0 else
          "[annotate] re-eval: no positions", flush=True)


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
    ap.add_argument("--reeval", action="store_true",
                    help="with --tag: refresh ONLY our eval_breakdown on annotated games (re-run "
                         "ev_breakdown with the current build; SF fields preserved) — use after an eval rebuild")
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-depth", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=None, help="default 0.5s when neither set")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="annotate N games in parallel, each with its own Stockfish process (default 1). "
                         "~6 on 8c/16t turns a multi-hour pass into roughly 1/6th the wall time.")
    args = ap.parse_args()

    if args.reeval and args.tag:
        reeval_tag(args.tag)
    elif args.recompute and args.tag:
        recompute_tag(args.tag)
    elif args.game:
        sf = args.sf_path or find_stockfish()
        if not sf:
            print("[annotate] no Stockfish binary found", flush=True)
            return
        mt = args.sf_movetime if (args.sf_movetime or args.sf_depth) else 0.5
        arb = Arbiter(sf, movetime=mt, depth=args.sf_depth)
        ai = _load_ai()
        try:
            print(json.dumps(annotate_game(args.game, arb, ai), indent=2), flush=True)
        finally:
            arb.close()
    elif args.tag:
        annotate_tag(args.tag, depth=args.sf_depth, movetime=args.sf_movetime,
                     sf_path=args.sf_path, concurrency=args.concurrency)
    else:
        ap.error("need --tag or --game")


if __name__ == "__main__":
    main()
