# -*- coding: utf-8 -*-
"""Our engine vs a strength-targeted Stockfish — the scalable collapse-mining source.

Self-play can't surface the worst-case eval holes that lose real games: both sides share the eval, so a
blind spot is never exploited. A DIFFERENT, strong opponent exploits our holes differently — which is the
whole point of playing Stockfish (limited to ~2200-2700 so the games stay competitive and the losses are
about a specific misjudgement, not raw strength). This driver plays N games (alternating colors), and for
each one records OUR engine's eval trajectory; a game where our eval PEAKED clearly winning and we then
failed to win is a COLLAPSE — its run-up FENs are dumped for the diagnose->fix->ship loop.

Reuses `EngineProc` (our engine_server line protocol) for our side and python-chess for Stockfish, so it
does NOT touch the committed self-play game loop. Runs under WSL (our .so is Linux); needs WSL->SF interop
up (the same prerequisite as the arbiter / fen_vs_sf).

Run (WSL, from NN Engine/):
  python selfplay/vs_sf.py --sf-elo 2400 --games 40 --preset LIGHTNING \
      --openings selfplay/openings_uho.txt --tag vssf_2400
Output under selfplay/games/<tag>/:  per-game game_NNN.jsonl + collapses.csv (the corpus seed).
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
import chess.engine
from selfplay import EngineProc, Adjudicator
from arbiter import find_stockfish, Arbiter
from tournament import load_openings, schedule, _config_with_preset


def _build_play_sf(sf_path, sf_elo):
    """Open + strength-cap a fresh playing-Stockfish process. One per game (handles aren't reentrant)."""
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    opts = {"Threads": 1}
    if sf_elo:                                   # strength-cap SF to keep games competitive
        opts.update({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    try:
        sf.configure(opts)
    except Exception as e:
        print(f"[vs_sf] WARNING: SF configure failed ({e})", flush=True)
    return sf


def our_pov(eval_white_pov, our_is_white):
    """Our-POV milli-pawns (positive = good for us) from a White-POV eval."""
    if eval_white_pov is None:
        return None
    return eval_white_pov if our_is_white else -eval_white_pov


def play_one(our_config, our_label, sf, sf_movetime, sf_depth, our_is_white, start_fen,
             opening_moves, max_plies, gpath, verbose, adjudicator=None):
    """One game: our engine vs Stockfish. Returns a per-game dict with our eval trajectory + result."""
    our_color = "white" if our_is_white else "black"
    cfg = (our_config + " USE_OPENING_BOOK=0").strip() if opening_moves else our_config
    eng = EngineProc(our_color, cfg, start_fen, our_label, gpath + ".stderr")
    board = chess.Board(start_fen)
    moves, traj = [], []          # traj: (ply, our_pov_eval, fen) for OUR moves only
    result, reason = "*", "in-progress"
    fh = open(gpath + ".jsonl", "w")
    try:
        eng.wait_ready()
        for uci in (opening_moves or []):
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                raise ValueError(f"illegal opening move {uci} at {board.fen()}")
            board.push(mv); eng.push(uci); moves.append(uci)
        sf_limit = (chess.engine.Limit(depth=sf_depth) if sf_depth
                    else chess.engine.Limit(time=sf_movetime))
        while True:
            if board.is_game_over(claim_draw=True):
                result, reason = board.result(claim_draw=True), "natural"
                break
            if board.ply() >= max_plies:
                result, reason = "1/2-1/2", f"max-plies ({max_plies})"
                break
            if board.turn == our_is_white:                       # our move
                kind, uci, calc = eng.go()
                if kind == "RESIGN":
                    result = "0-1" if board.turn else "1-0"
                    reason = f"{our_label} resigned"
                    break
                ev = (calc or {}).get("eval")
                ewp = (ev if board.turn else -ev) if isinstance(ev, int) else None
                decision_fen = board.fen()                        # position we CHOSE this move from
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    result, reason = "*", f"ILLEGAL {uci} at {board.fen()}"
                    break
                board.push(mv); moves.append(uci)
                opov = our_pov(ewp, our_is_white)
                traj.append((len(moves), opov, board.fen(), decision_fen, uci))
                _wr(fh, {"ply": len(moves), "color": our_color, "uci": uci,
                         "our_pov_eval": opov, "depth": (calc or {}).get("depth"),
                         "nodes": (calc or {}).get("nodes"), "fen": board.fen()})
                if verbose and opov is not None:
                    print(f"  {len(moves):>3}. us  {uci:<6} {opov/1000:+.2f}", flush=True)
                if adjudicator is not None:
                    a = adjudicator.after_move(board, ewp, len(moves), fh)
                    if a is not None:
                        result, reason = a; break
            else:                                                # Stockfish move
                res = sf.play(board, sf_limit, info=chess.engine.INFO_ALL)
                if res.move is None:
                    result, reason = board.result(claim_draw=True), "sf-no-move"
                    break
                uci = res.move.uci()
                board.push(res.move); moves.append(uci)
                eng.push(uci)
                # Log SF's search depth + nodes per move (the search-speed TARGET: how deep / how many nodes
                # SF reaches at this time control vs our engine reaching at the same).
                _wr(fh, {"ply": len(moves), "color": ("black" if our_is_white else "white"),
                         "uci": uci, "sf": True, "fen": board.fen(),
                         "depth": res.info.get("depth"), "nodes": res.info.get("nodes")})
                if verbose:
                    print(f"  {len(moves):>3}. SF  {uci:<6}", flush=True)
                if adjudicator is not None:
                    a = adjudicator.after_move(board, None, len(moves), fh)  # SF move: no our-eval
                    if a is not None:
                        result, reason = a; break
    finally:
        try: eng.quit()
        except Exception: pass
        fh.close()

    our_score = _our_score(result, our_is_white)
    peak = max((t[1] for t in traj if t[1] is not None), default=None)
    return {"our_color": our_color, "result": result, "reason": reason, "our_score": our_score,
            "plies": len(moves), "peak_our_eval": peak, "traj": traj}


def _our_score(result, our_is_white):
    if result == "1/2-1/2":
        return 0.5
    if result not in ("1-0", "0-1"):
        return None
    return 1.0 if ((result == "1-0") == our_is_white) else 0.0


def _wr(fh, rec):
    fh.write(json.dumps(rec) + "\n"); fh.flush()


def extract_collapse(g, win_threshold, drop_to):
    """A collapse = our eval peaked >= win_threshold (we thought clearly winning) but our_score < 1.
    Returns the peak ply/eval, the DECISION fen we chose the peak move from (the position to re-examine
    eval-vs-horizon) + that move, and the drop fen where our eval first fell below drop_to. None if no
    collapse. The decision fen is the diagnosis target: re-search it deep + compare SF."""
    if g["peak_our_eval"] is None or g["peak_our_eval"] < win_threshold or g["our_score"] == 1.0:
        return None
    traj = [t for t in g["traj"] if t[1] is not None]
    if not traj:
        return None
    peak_i = max(range(len(traj)), key=lambda i: traj[i][1])
    pk = traj[peak_i]
    drop = pk
    for t in traj[peak_i + 1:]:                  # walk forward to where we crossed back below drop_to
        drop = t
        if t[1] < drop_to:
            break
    return {"peak_ply": pk[0], "peak_eval": pk[1], "decision_fen": pk[3], "peak_move": pk[4],
            "drop_ply": drop[0], "drop_eval": drop[1], "drop_fen": drop[2]}


def run(args):
    logdir = os.path.join(THIS_DIR, "games", args.tag)
    os.makedirs(logdir, exist_ok=True)
    openings = load_openings(args.openings)
    if not openings:
        print(f"[vs_sf] no openings parsed from {args.openings}", flush=True); return
    sf_path = args.sf_path or find_stockfish()
    if not sf_path:
        print("[vs_sf] no Stockfish found (set STOCKFISH_PATH)", flush=True); return
    # The OPPONENT SF (sf_path) and the ARBITER SF can differ: e.g. play vs classical SF11 (HCE yardstick)
    # while a strong neutral SF18 adjudicates. Defaults to the same binary (back-compat).
    arb_path = args.sf_arb_path or sf_path

    sched = schedule(args.games, len(openings), args.seed)
    our_cfg = _config_with_preset(args.our_config, args.preset)

    do_adj = args.adjudicate_draw or args.adjudicate_win
    if do_adj:
        # Draw adjudication ends dead-drawn tails early (the main overnight-throughput win); win adjudication
        # is weak here (gated on a full window of OUR evals, but SF moves contribute none) so it rarely fires.
        print(f"[vs_sf] adjudication ON (per-game separate SF arbiter @ {args.sf_arb_movetime}s; "
              f"draw={args.adjudicate_draw} win={args.adjudicate_win})", flush=True)

    coll_path = os.path.join(logdir, "collapses.csv")
    cf = open(coll_path, "w", newline="")
    cw = csv.DictWriter(cf, fieldnames=["game", "our_color", "result", "peak_ply", "peak_eval",
                                        "peak_move", "drop_ply", "drop_eval", "decision_fen", "drop_fen"])
    cw.writeheader()
    conc = max(1, args.concurrency)
    print(f"[vs_sf] our={args.our_label} vs SF(elo={args.sf_elo or 'full'}) — {args.games} games, "
          f"preset={args.preset}, win_thresh={args.win_threshold/1000:+.1f}, concurrency={conc}", flush=True)

    def play_game_g(g, oi, our_white):
        """Run one game. Thread-safe: own engine subprocess, own playing-SF, own arbiter SF.
        Stockfish handles aren't reentrant, so every game opens (and closes) its own. Returns the
        per-game result dict (with 'game' index attached); collapse detection / CSV / prints stay on
        the main thread. Returns None on driver error."""
        gpath = os.path.join(logdir, f"game_{g:03d}")
        sf = arb = None
        try:
            sf = _build_play_sf(sf_path, args.sf_elo)
            adj = None
            if do_adj:
                try:
                    arb = Arbiter(arb_path, movetime=args.sf_arb_movetime)
                    adj = Adjudicator(arb, do_draw=args.adjudicate_draw, do_win=args.adjudicate_win)
                except Exception as e:
                    print(f"[vs_sf] game {g}: arbiter init failed ({e}); no adjudication", flush=True)
            res = play_one(our_cfg, args.our_label, sf, args.sf_movetime, args.sf_depth, our_white,
                           chess.STARTING_FEN, openings[oi], args.max_plies, gpath, not args.quiet,
                           adjudicator=adj)
            res["game"] = g
            return res
        except Exception as e:
            print(f"[vs_sf] game {g}: error {e}", flush=True)
            return None
        finally:
            if sf is not None:
                try: sf.quit()
                except Exception: pass
            if arb is not None:
                try: arb.close()
                except Exception: pass

    def record(res):
        # Print the live per-game line as each game COMPLETES (order may interleave under concurrency).
        # The collapses.csv itself is written in strict game order AFTER the run (see below) so its
        # content is byte-for-byte identical to the single-threaded run for the same games.
        g = res["game"]
        coll = extract_collapse(res, args.win_threshold, args.drop_to)
        flag = (f"  *** COLLAPSE peak {coll['peak_eval']/1000:+.1f} -> {res['result']}"
                if coll else "")
        peak = res["peak_our_eval"]
        print(f"[vs_sf] game {g}: {res['result']} ({res['reason']}) our={res['our_score']} "
              f"as {res['our_color']} peak={'-' if peak is None else f'{peak/1000:+.1f}'}{flag}", flush=True)

    summ = []
    t0 = time.monotonic()
    pending = [(g, oi, pw) for g, (oi, pw) in enumerate(sched)]
    try:
        if conc <= 1:
            for g, oi, pw in pending:
                res = play_game_g(g, oi, pw)
                if res is not None:
                    summ.append(res); record(res)
        else:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                futs = {pool.submit(play_game_g, g, oi, pw): g for g, oi, pw in pending}
                inflight = set(futs)
                while inflight:
                    done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    # record() (prints only) is main-thread; append is main-thread too -> no lock needed.
                    for fut in done_set:
                        res = fut.result()
                        if res is not None:
                            summ.append(res); record(res)
        # Write collapses.csv in strict game order -> identical content regardless of completion order.
        for res in sorted(summ, key=lambda r: r["game"]):
            coll = extract_collapse(res, args.win_threshold, args.drop_to)
            if coll:
                cw.writerow({"game": res["game"], "our_color": res["our_color"], "result": res["result"],
                             "peak_ply": coll["peak_ply"], "peak_eval": coll["peak_eval"],
                             "peak_move": coll["peak_move"], "drop_ply": coll["drop_ply"],
                             "drop_eval": coll["drop_eval"], "decision_fen": coll["decision_fen"],
                             "drop_fen": coll["drop_fen"]})
        cf.flush()
    finally:
        cf.close()

    dec = [s for s in summ if s["our_score"] is not None]
    sc = sum(s["our_score"] for s in dec) / len(dec) if dec else 0.0
    ncol = sum(1 for s in summ if extract_collapse(s, args.win_threshold, args.drop_to))
    print(f"\n[vs_sf] {len(dec)} decided, our score {100*sc:.1f}% — {ncol} COLLAPSES -> {coll_path}",
          flush=True)
    print(f"[vs_sf] {len(summ)} games in {(time.monotonic()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Our engine vs strength-targeted Stockfish; mine collapses.")
    ap.add_argument("--our-config", default="", help="env knobs for our engine")
    ap.add_argument("--our-label", default="ours")
    ap.add_argument("--sf-elo", type=int, default=2400, help="UCI_Elo cap for SF (0 = full strength)")
    ap.add_argument("--sf-movetime", type=float, default=0.3, help="SF seconds/move (if --sf-depth unset)")
    ap.add_argument("--sf-depth", type=int, default=None, help="SF fixed depth (overrides movetime)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--openings", default=os.path.join(THIS_DIR, "openings_uho.txt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="LIGHTNING")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="play N games in parallel (default 1 = sequential). Each game opens its own "
                         "engine + playing-SF (+ arbiter SF if adjudicating); match to physical cores "
                         "(engines are OMP_NUM_THREADS=1). ~6 on 8c/16t.")
    ap.add_argument("--win-threshold", type=int, default=2000,
                    help="our-POV milli-pawn peak above which a non-win counts as a collapse (2000 = +2.0)")
    ap.add_argument("--drop-to", type=int, default=500,
                    help="dump the run-up window from the peak until our eval falls below this (milli-pawns)")
    ap.add_argument("--tag", default="vssf")
    ap.add_argument("--sf-path", default=None, help="opponent SF binary (the engine we PLAY against)")
    ap.add_argument("--sf-arb-path", default=None, help="arbiter SF binary (adjudicator); defaults to --sf-path")
    ap.add_argument("--quiet", action="store_true")
    # Game adjudication (separate SF arbiter): draw ends dead-drawn tails early (overnight throughput),
    # win ends clearly-decided games. Recommended overnight: --adjudicate-draw.
    ap.add_argument("--adjudicate-draw", action="store_true", help="end dead-drawn positions early (SF-confirmed)")
    ap.add_argument("--adjudicate-win", action="store_true", help="also end clearly-won positions early")
    ap.add_argument("--sf-arb-movetime", type=float, default=0.2, help="arbiter SF seconds/confirm")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
