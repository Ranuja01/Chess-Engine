# -*- coding: utf-8 -*-
"""Player-vs-player self-play tournament.

Two PLAYERS (each = a label + an env-knob config) play N games from seeded openings, swapping colors
each game to cancel first-move bias. Each game is a fresh pair of `engine_server` processes (cold
caches) driven by `selfplay.play_game`. The Stockfish arbiter is OFF during play — analysis is done
post-hoc by `annotate.py` over the recorded FENs (SF is a pure function of position, so deferring
loses nothing and keeps games at full engine speed).

Output under games/<tag>/:
  game_NNN/            per game: game.jsonl, game.pgn, white.stderr, black.stderr, [crash.json]
  summary.csv          one row per game, appended live (tailable)
  tournament.json      final standings: P1 W/L/D (overall + split by color), score%, Elo±
  tournament.pgn       all games concatenated (load in any GUI)

Run (from NN Engine/):
  python selfplay/tournament.py --p1-config "ASPIRATION_DELTA=500" --p2-config "ASPIRATION_DELTA=0" \
      --p1-label asp500 --p2-label asp0 --games 8 --preset LIGHTNING --quiet --tag asp_500_vs_0
"""

import os
import sys
import csv
import json
import math
import time
import random
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from selfplay import play_game, Adjudicator
from arbiter import Arbiter, find_stockfish

# Per-WORKER-THREAD Stockfish arbiter: chess.engine.SimpleEngine.analyse() is not reentrant, so a
# pool-parallel run gives each thread its own SF process (created lazily on first adjudication).
_tls = threading.local()
_arbiters = []
_arb_lock = threading.Lock()


def _thread_arbiter(sf, movetime, depth):
    a = getattr(_tls, "arb", None)
    if a is None:
        a = Arbiter(sf, movetime=movetime, depth=depth)
        _tls.arb = a
        with _arb_lock:
            _arbiters.append(a)
    return a


def _close_arbiters():
    with _arb_lock:
        for a in _arbiters:
            try:
                a.close()
            except Exception:
                pass
        _arbiters.clear()


def load_openings(path):
    """Parse openings.txt → list of UCI-move lists (blank / '#' lines ignored)."""
    openings = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            openings.append(line.split())
    return openings


def schedule(n_games, n_openings, seed):
    """(opening_idx, p1_is_white) per game. Openings are shuffled by seed, then each is played from
    both colors on consecutive games so every opening contributes a balanced pair."""
    order = list(range(n_openings))
    random.Random(seed).shuffle(order)
    return [(order[(g // 2) % n_openings], (g % 2 == 0)) for g in range(n_games)]


def p1_score(result, p1_is_white):
    """Player-1's score for a White-POV result string; None for a void ('*') game."""
    if result == "1/2-1/2":
        return 0.5
    if result not in ("1-0", "0-1"):
        return None
    p1_won = (result == "1-0") == p1_is_white
    return 1.0 if p1_won else 0.0


def elo_from_score(score):
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def _config_with_preset(config, preset):
    return (f"PRESET={preset} {config}").strip() if preset else config


# Rough per-game wall-clock by preset (seconds) — seeds the timed-mode estimate before real game
# durations are measured; overridable via --avg-seconds.
PRESET_DUR = {"LIGHTNING": 130.0, "BLITZ": 322.0, "STANDARD": 1320.0, "LONG_FORMAT": 1320.0}


def run(args):
    logdir = os.path.join(THIS_DIR, "games", args.tag)
    os.makedirs(logdir, exist_ok=True)
    summary_path = os.path.join(logdir, "summary.csv")
    openings = load_openings(args.openings)
    if not openings:
        print(f"[tournament] no openings parsed from {args.openings}", flush=True)
        return
    # In timed mode the game count isn't known up front; build a large schedule and stop on the clock.
    n_sched = args.games if not args.max_minutes else max(args.games, 200000)
    games = schedule(n_sched, len(openings), args.seed)

    p1c = _config_with_preset(args.p1_config, args.preset)
    p2c = _config_with_preset(args.p2_config, args.preset)

    # Optional in-play adjudication. Verify Stockfish ONCE up front; the Arbiter itself is created
    # lazily PER WORKER THREAD (not reentrant) so concurrent games never share one SF process.
    # --adjudicate = both; --adjudicate-draw / --adjudicate-win enable them independently.
    do_draw = args.adjudicate or args.adjudicate_draw
    do_win = args.adjudicate or args.adjudicate_win
    sf = None
    if do_draw or do_win:
        sf = args.sf_path or find_stockfish()
        if sf:
            modes = (["win>%gp" % args.win_p] if do_win else []) + \
                    (["draw<%dcp (no-progress %dplies, low-pieces %d)" % (args.draw_cp, args.noprog_plies, args.low_pieces)] if do_draw else [])
            print(f"[tournament] adjudication ON (per-thread SF confirm @ {args.sf_movetime}s: {', '.join(modes)})", flush=True)
        else:
            print("[tournament] WARNING: adjudication requested but no Stockfish found "
                  "(set STOCKFISH_PATH or --sf-path); OFF", flush=True)
            do_draw = do_win = False

    fields = ["game", "opening_idx", "p1_color", "white", "black", "result", "p1_score", "plies", "reason"]
    rows = []
    done = {}
    if args.resume and os.path.exists(summary_path):
        for r in csv.DictReader(open(summary_path)):
            done[int(r["game"])] = r
        rows = list(done.values())

    # (Re)open summary.csv: keep existing rows on resume, else write a fresh header.
    sfh = open(summary_path, "a" if done else "w", newline="")
    writer = csv.DictWriter(sfh, fieldnames=fields)
    if not done:
        writer.writeheader()
        sfh.flush()

    conc = max(1, args.concurrency)
    print(f"[tournament] {args.p1_label} vs {args.p2_label} — {args.games} games, "
          f"{len(openings)} openings, preset={args.preset or 'STANDARD'}, concurrency={conc}, "
          f"arbiter {'ON' if (do_draw or do_win) else 'OFF (post-hoc)'}", flush=True)

    def play_one(g, opening_idx, p1_white):
        """Run one game. Thread-safe: own gdir, own engine subprocesses, own SF arbiter.
        Returns (row, wall_seconds)."""
        t_game = time.monotonic()
        gdir = os.path.join(logdir, f"game_{g:03d}")
        white_cfg, white_lbl = (p1c, args.p1_label) if p1_white else (p2c, args.p2_label)
        black_cfg, black_lbl = (p2c, args.p2_label) if p1_white else (p1c, args.p1_label)
        adj = None
        if do_draw or do_win:
            try:
                adj = Adjudicator(_thread_arbiter(sf, args.sf_movetime, args.sf_depth),
                                  draw_cp=args.draw_cp, win_p=args.win_p, noprog_plies=args.noprog_plies,
                                  low_pieces=args.low_pieces, window=args.adj_window,
                                  do_win=do_win, do_draw=do_draw)
            except Exception as e:
                print(f"[tournament] game {g}: arbiter init failed ({e}); no adjudication", flush=True)
        try:
            res = play_game(white_cfg, black_cfg, white_lbl, black_lbl, chess.STARTING_FEN,
                            args.max_plies, gdir, jsonl_path=os.path.join(gdir, "game.jsonl"),
                            verbose=not args.quiet, arbiter=None, opening_moves=openings[opening_idx],
                            adjudicator=adj)
            result, reason, plies = res["result"], res["reason"], res["plies"]
        except Exception as e:
            result, reason, plies = "*", f"driver error: {e}", 0
        return ({"game": g, "opening_idx": opening_idx, "p1_color": "white" if p1_white else "black",
                 "white": white_lbl, "black": black_lbl, "result": result,
                 "p1_score": p1_score(result, p1_white), "plies": plies, "reason": reason},
                time.monotonic() - t_game)

    def record(row):
        # Called ONLY on the main thread (sequentially, as futures complete) -> no lock needed.
        writer.writerow(row)
        sfh.flush()
        rows.append(row)
        sc = row["p1_score"]
        print(f"[tournament] game {row['game']}: {row['result']}  ({row['reason']})  "
              f"P1={'-' if sc is None else sc}  [{args.p1_label} as {row['p1_color']}]", flush=True)

    pending = [(g, oi, pw) for g, (oi, pw) in enumerate(games) if g not in done]
    for g in sorted(done):
        print(f"[tournament] game {g}: skipped (resume)", flush=True)

    # Timed mode (--max-minutes): play as many games as fit the wall-clock budget. `avg_dur` is the
    # measured per-game wall time under the current concurrency load (seeded by a preset default); we
    # start another game only while elapsed + avg_dur <= budget, then let in-flight games drain.
    budget = args.max_minutes * 60.0 if args.max_minutes else None
    avg_dur = args.avg_seconds or PRESET_DUR.get((args.preset or "STANDARD").upper(), 322.0)
    durations = []
    t0 = time.monotonic()

    def have_budget():
        return budget is None or (time.monotonic() - t0) + avg_dur <= budget

    try:
        if conc <= 1:
            for i, (g, oi, pw) in enumerate(pending):
                if i > 0 and not have_budget():        # always run the first game (measure a real dur)
                    break
                row, dur = play_one(g, oi, pw)
                record(row)
                durations.append(dur)
                avg_dur = sum(durations) / len(durations)
        else:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                it = iter(pending)
                inflight = set()

                def submit_next():
                    nxt = next(it, None)
                    if nxt is None:
                        return False
                    gg, oi, pw = nxt
                    inflight.add(pool.submit(play_one, gg, oi, pw))
                    return True

                for _ in range(conc):                  # always prime a batch (measure before gating)
                    if not submit_next():
                        break
                while inflight:
                    done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        row, dur = fut.result()
                        record(row)
                        durations.append(dur)
                    avg_dur = sum(durations) / len(durations)
                    if have_budget():                       # else stop submitting; let in-flight drain
                        for _ in range(len(done_set)):
                            if not submit_next():
                                break
    finally:
        sfh.close()
        _close_arbiters()

    if budget is not None:
        elapsed = time.monotonic() - t0
        gph = (len(durations) / elapsed * 3600.0) if elapsed > 0 else 0.0
        print(f"[tournament] timed: {len(durations)} games in {elapsed/60:.1f} min "
              f"({gph:.0f} games/hr, avg {avg_dur:.0f}s/game)", flush=True)

    _finalize(args, logdir, rows)


def _finalize(args, logdir, rows):
    # Aggregate from P1's perspective, overall and split by the color P1 had.
    agg = {"overall": [0, 0, 0], "white": [0, 0, 0], "black": [0, 0, 0]}  # [W, L, D]
    void = 0
    for r in rows:
        sc = r["p1_score"]
        sc = None if sc in (None, "", "None") else float(sc)
        if sc is None:
            void += 1
            continue
        idx = 0 if sc == 1.0 else (1 if sc == 0.0 else 2)
        agg["overall"][idx] += 1
        agg[r["p1_color"]][idx] += 1
    W, L, D = agg["overall"]
    n = W + L + D
    score = (W + 0.5 * D) / n if n else 0.0
    elo = elo_from_score(score) if n else 0.0
    margin = 800.0 / math.sqrt(n) if n else 0.0
    standings = {
        "p1_label": args.p1_label, "p2_label": args.p2_label,
        "p1_config": args.p1_config, "p2_config": args.p2_config,
        "preset": args.preset or "STANDARD", "games": args.games, "decided": n, "void": void,
        "p1_W": W, "p1_L": L, "p1_D": D, "score_pct": round(100.0 * score, 1),
        "elo": round(elo, 1), "elo_margin": round(margin, 1),
        "by_color": {"as_white": agg["white"], "as_black": agg["black"]},
    }
    with open(os.path.join(logdir, "tournament.json"), "w") as f:
        json.dump(standings, f, indent=2)

    # Rebuild the combined PGN from each game's game.pgn (correct under --resume too).
    with open(os.path.join(logdir, "tournament.pgn"), "w") as out:
        for r in sorted(rows, key=lambda x: int(x["game"])):
            pgn = os.path.join(logdir, f"game_{int(r['game']):03d}", "game.pgn")
            if os.path.exists(pgn):
                out.write(open(pgn).read().rstrip() + "\n\n")

    print(f"\n[tournament] {args.p1_label} vs {args.p2_label}: "
          f"+{W} -{L} ={D} of {n}  ({standings['score_pct']}%)  "
          f"Elo {standings['elo']:+} ±{standings['elo_margin']}"
          + (f"  [{void} void]" if void else ""), flush=True)
    print(f"[tournament]   as White: +{agg['white'][0]} -{agg['white'][1]} ={agg['white'][2]}   "
          f"as Black: +{agg['black'][0]} -{agg['black'][1]} ={agg['black'][2]}", flush=True)
    print(f"[tournament] standings -> {os.path.join(logdir, 'tournament.json')}", flush=True)
    print(f"[tournament] combined  -> {os.path.join(logdir, 'tournament.pgn')}", flush=True)

    if args.annotate:
        print("[tournament] running post-hoc Stockfish annotation ...", flush=True)
        try:
            import annotate
            annotate.annotate_tag(args.tag, depth=args.sf_depth, movetime=args.sf_movetime,
                                  sf_path=args.sf_path)
        except Exception as e:
            print(f"[tournament] annotation failed: {e}  "
                  f"(run `python selfplay/annotate.py --tag {args.tag}` manually)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Player-vs-player self-play tournament.")
    ap.add_argument("--p1-config", default="", help="env knobs for player 1")
    ap.add_argument("--p2-config", default="", help="env knobs for player 2")
    ap.add_argument("--p1-label", default="P1")
    ap.add_argument("--p2-label", default="P2")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--openings", default=os.path.join(THIS_DIR, "openings.txt"))
    ap.add_argument("--seed", type=int, default=0, help="seeds the opening order + color schedule")
    ap.add_argument("--preset", default="LIGHTNING", help="time control applied to BOTH players")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="play N games in parallel (default 1 = sequential). ~6 on 8c/16t.")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="timed mode: play as many games as fit this wall-clock budget, then stop "
                         "(0 = off, use --games). Single --preset. e.g. 540 = 9h.")
    ap.add_argument("--avg-seconds", type=float, default=0.0,
                    help="seed the per-game wall estimate for timed mode (0 = preset default)")
    ap.add_argument("--tag", default="tourney")
    ap.add_argument("--quiet", action="store_true", help="suppress per-move prints")
    ap.add_argument("--resume", action="store_true", help="skip games already in summary.csv")
    ap.add_argument("--annotate", action="store_true", help="run the SF analysis pass at the end")
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-depth", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.5)
    # In-play adjudication (heuristic-gated, single-SF-confirm). Off unless a flag is passed.
    ap.add_argument("--adjudicate", action="store_true",
                    help="enable BOTH win and draw adjudication (shorthand for --adjudicate-draw --adjudicate-win)")
    ap.add_argument("--adjudicate-draw", action="store_true",
                    help="end dead-drawn positions early (recommended; wins still grind to mate/resign)")
    ap.add_argument("--adjudicate-win", action="store_true",
                    help="also end clearly-won positions early (off by default so conversion stays observable)")
    ap.add_argument("--draw-cp", type=int, default=40, help="SF |cp| below this confirms a draw")
    ap.add_argument("--win-p", type=float, default=5.0, help="engine win-margin gate in pawns (White-POV)")
    ap.add_argument("--noprog-plies", type=int, default=40, help="halfmove-clock plies of no progress that gate a draw")
    ap.add_argument("--low-pieces", type=int, default=12, help="piece count at/under which a flat eval gates a draw")
    ap.add_argument("--adj-window", type=int, default=6, help="consecutive plies for the eval window / SF cooldown")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
