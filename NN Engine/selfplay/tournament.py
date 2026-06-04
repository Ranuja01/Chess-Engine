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
import random
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from selfplay import play_game


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


def run(args):
    logdir = os.path.join(THIS_DIR, "games", args.tag)
    os.makedirs(logdir, exist_ok=True)
    summary_path = os.path.join(logdir, "summary.csv")
    openings = load_openings(args.openings)
    if not openings:
        print(f"[tournament] no openings parsed from {args.openings}", flush=True)
        return
    games = schedule(args.games, len(openings), args.seed)

    p1c = _config_with_preset(args.p1_config, args.preset)
    p2c = _config_with_preset(args.p2_config, args.preset)

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

    print(f"[tournament] {args.p1_label} vs {args.p2_label} — {args.games} games, "
          f"{len(openings)} openings, preset={args.preset or 'STANDARD'}, arbiter OFF (post-hoc)",
          flush=True)

    for g, (opening_idx, p1_white) in enumerate(games):
        if g in done:
            print(f"[tournament] game {g}: skipped (resume)", flush=True)
            continue
        gdir = os.path.join(logdir, f"game_{g:03d}")
        white_cfg, white_lbl = (p1c, args.p1_label) if p1_white else (p2c, args.p2_label)
        black_cfg, black_lbl = (p2c, args.p2_label) if p1_white else (p1c, args.p1_label)
        try:
            res = play_game(white_cfg, black_cfg, white_lbl, black_lbl, chess.STARTING_FEN,
                            args.max_plies, gdir, jsonl_path=os.path.join(gdir, "game.jsonl"),
                            verbose=not args.quiet, arbiter=None, opening_moves=openings[opening_idx])
            result, reason, plies = res["result"], res["reason"], res["plies"]
        except Exception as e:
            result, reason, plies = "*", f"driver error: {e}", 0
        row = {"game": g, "opening_idx": opening_idx, "p1_color": "white" if p1_white else "black",
               "white": white_lbl, "black": black_lbl, "result": result,
               "p1_score": p1_score(result, p1_white), "plies": plies, "reason": reason}
        writer.writerow(row)
        sfh.flush()
        rows.append(row)
        sc = row["p1_score"]
        print(f"[tournament] game {g}: {result}  ({reason})  P1={'-' if sc is None else sc}  "
              f"[{args.p1_label} as {row['p1_color']}]", flush=True)
    sfh.close()

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
    ap.add_argument("--tag", default="tourney")
    ap.add_argument("--quiet", action="store_true", help="suppress per-move prints")
    ap.add_argument("--resume", action="store_true", help="skip games already in summary.csv")
    ap.add_argument("--annotate", action="store_true", help="run the SF analysis pass at the end")
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-depth", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.5)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
