# -*- coding: utf-8 -*-
"""Sequential Probability Ratio Test (SPRT) for an A/B engine config — the fast statistical judge.

This is `tournament.py` with one thing added: an early-stopping decision rule. Instead of playing a
fixed N games and eyeballing the Elo±margin afterwards, it updates a log-likelihood ratio (LLR) after
every game and STOPS the moment the evidence is conclusive — typically far fewer games for a clear
result, more for a marginal one, with controlled error rates. This is how Fishtest gates every
Stockfish patch.

Hypotheses (P1's strength over P2, in Elo):
  H0: <= elo0   (default 0  — "no improvement")
  H1: >= elo1   (default 5  — "a real improvement")
Decision: accept H1 when LLR >= log((1-beta)/alpha); accept H0 when LLR <= log(beta/(1-alpha)).
With the default alpha = beta = 0.05 the bounds are +/-2.944.

LLR uses the GSPRT (generalized SPRT) normal approximation with the per-game score variance estimated
from the observed W/L/D, which is the standard practical estimator:
  mu_hat = (W + 0.5 D) / N         observed score
  var    = (W + 0.25 D) / N - mu_hat^2   per-game score variance
  LLR    = N * (mu1 - mu0) * (mu_hat - (mu0 + mu1)/2) / var
where mu0, mu1 are elo0, elo1 mapped to score fractions.

Everything else (engine configs, opening schedule with color-swap, optional Stockfish draw/win
adjudication) is reused verbatim from tournament.py / selfplay.py, so an SPRT run is directly
comparable to a fixed tournament — it just stops early.

Run (from NN Engine/), e.g. reduce-more LMR vs baseline at lightning:
  python selfplay/sprt.py \
    --p1-config "ENABLE_HISTORY_LMR=1 HISTORY_LMR_CAP=0 HISTORY_LMR_MORE_CAP=1" --p1-label hlmr \
    --p2-config "" --p2-label base \
    --preset LIGHTNING --elo0 0 --elo1 5 --max-games 400 --adjudicate-draw --quiet --tag sprt_hlmr

Output under games/<tag>/: summary.csv (one row per game, tailable) + sprt.json (the verdict).
"""

import os
import sys
import csv
import json
import math
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from selfplay import play_game, Adjudicator
from arbiter import Arbiter, find_stockfish
# Reuse the tested tournament helpers so an SPRT run matches a fixed tournament exactly.
from tournament import (load_openings, schedule, p1_score, elo_from_score, _config_with_preset,
                        _thread_arbiter, _close_arbiters, PRESET_DUR)


def elo_to_score(elo):
    """Logistic Elo model: expected score for a +elo advantage."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(W, L, D, elo0, elo1):
    """GSPRT log-likelihood ratio from P1-perspective W/L/D counts (variance estimated from data)."""
    N = W + L + D
    if N == 0:
        return 0.0
    w, d, l = W / N, D / N, L / N
    mu_hat = w + 0.5 * d
    var = (w + 0.25 * d) - mu_hat * mu_hat
    if var <= 1e-9:                      # all-same-result early on: no usable variance yet
        return 0.0
    mu0, mu1 = elo_to_score(elo0), elo_to_score(elo1)
    return N * (mu1 - mu0) * (mu_hat - 0.5 * (mu0 + mu1)) / var


def run(args):
    logdir = os.path.join(THIS_DIR, "games", args.tag)
    os.makedirs(logdir, exist_ok=True)
    summary_path = os.path.join(logdir, "summary.csv")

    openings = load_openings(args.openings)
    if not openings:
        print(f"[sprt] no openings parsed from {args.openings}", flush=True)
        return
    n_sched = args.max_games if not args.max_minutes else max(args.max_games, 200000)
    games = schedule(n_sched, len(openings), args.seed)

    p1c = _config_with_preset(args.p1_config, args.preset)
    p2c = _config_with_preset(args.p2_config, args.preset)

    # Optional in-play adjudication. Verify Stockfish once; the Arbiter is created lazily PER WORKER
    # THREAD (not reentrant) so concurrent games never share one SF process.
    do_draw = args.adjudicate or args.adjudicate_draw
    do_win = args.adjudicate or args.adjudicate_win
    sf = None
    if do_draw or do_win:
        sf = args.sf_path or find_stockfish()
        if sf:
            print(f"[sprt] adjudication ON (per-thread SF confirm @ {args.sf_movetime}s)", flush=True)
        else:
            print("[sprt] WARNING: adjudication requested but no Stockfish found "
                  "(set STOCKFISH_PATH or --sf-path); OFF", flush=True)
            do_draw = do_win = False

    A = math.log((1.0 - args.beta) / args.alpha)   # accept H1 at/above this
    B = math.log(args.beta / (1.0 - args.alpha))   # accept H0 at/below this

    fields = ["game", "opening_idx", "p1_color", "white", "black", "result",
              "p1_score", "plies", "reason", "W", "L", "D", "llr"]
    sfh = open(summary_path, "w", newline="")
    writer = csv.DictWriter(sfh, fieldnames=fields)
    writer.writeheader()
    sfh.flush()

    conc = max(1, args.concurrency)
    print(f"[sprt] {args.p1_label} vs {args.p2_label}  H0:<={args.elo0} H1:>={args.elo1} Elo  "
          f"alpha={args.alpha} beta={args.beta}  bounds=[{B:+.3f}, {A:+.3f}]  "
          f"preset={args.preset or 'STANDARD'}  concurrency={conc}  max_games={args.max_games}", flush=True)

    def play_one(g, opening_idx, p1_white):
        """Run one game (thread-safe: own gdir, subprocesses, SF arbiter). Returns (dict, wall_seconds);
        the MAIN thread scores the dict into W/L/D/LLR (workers never touch shared counters)."""
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
                print(f"[sprt] game {g}: arbiter init failed ({e}); no adjudication", flush=True)
        try:
            res = play_game(white_cfg, black_cfg, white_lbl, black_lbl, chess.STARTING_FEN,
                            args.max_plies, gdir, jsonl_path=os.path.join(gdir, "game.jsonl"),
                            verbose=not args.quiet, arbiter=None, opening_moves=openings[opening_idx],
                            adjudicator=adj)
            result, reason, plies = res["result"], res["reason"], res["plies"]
        except Exception as e:
            result, reason, plies = "*", f"driver error: {e}", 0
        return ({"g": g, "opening_idx": opening_idx, "p1_white": p1_white,
                 "white": white_lbl, "black": black_lbl, "result": result,
                 "reason": reason, "plies": plies, "sc": p1_score(result, p1_white)},
                time.monotonic() - t_game)

    W = L = D = void = 0
    llr = 0.0
    decision = None

    def process(r):
        """Score one finished game into W/L/D/LLR + CSV (MAIN THREAD ONLY -> no lock). Returns a
        decision string if a bound is crossed, else None."""
        nonlocal W, L, D, void, llr
        sc = r["sc"]
        if sc is None:
            void += 1
        elif sc == 1.0:
            W += 1
        elif sc == 0.0:
            L += 1
        else:
            D += 1
        n = W + L + D
        llr = sprt_llr(W, L, D, args.elo0, args.elo1)
        score = (W + 0.5 * D) / n if n else 0.0
        elo = elo_from_score(score) if n else 0.0
        writer.writerow({"game": r["g"], "opening_idx": r["opening_idx"],
                         "p1_color": "white" if r["p1_white"] else "black",
                         "white": r["white"], "black": r["black"], "result": r["result"],
                         "p1_score": sc, "plies": r["plies"], "reason": r["reason"],
                         "W": W, "L": L, "D": D, "llr": round(llr, 4)})
        sfh.flush()
        print(f"[sprt] game {r['g']}: {r['result']} ({r['reason']})  +{W} -{L} ={D}"
              f"{f' [{void} void]' if void else ''}  "
              f"LLR={llr:+.3f} (need {B:+.2f}/{A:+.2f})  elo~{elo:+.0f}", flush=True)
        if n >= args.min_games:
            if llr >= A:
                return "H1 accepted: P1 is stronger (>= elo1)"
            if llr <= B:
                return "H0 accepted: P1 is NOT a >= elo1 improvement"
        return None

    # Timed mode (--max-minutes): stop submitting once the clock can't fit another game (in addition to
    # the SPRT bound + max_games). avg_dur = measured per-game wall time under load (preset-seeded).
    budget = args.max_minutes * 60.0 if args.max_minutes else None
    avg_dur = args.avg_seconds or PRESET_DUR.get((args.preset or "STANDARD").upper(), 322.0)
    durations = []
    t0 = time.monotonic()

    def have_budget():
        return budget is None or (time.monotonic() - t0) + avg_dur <= budget

    try:
        if conc <= 1:
            for g, (opening_idx, p1_white) in enumerate(games):
                if durations and not have_budget():    # always run the first game (measure a real dur)
                    break
                r, dur = play_one(g, opening_idx, p1_white)
                durations.append(dur)
                avg_dur = sum(durations) / len(durations)
                d = process(r)
                if d:
                    decision = d
                    break
        else:
            with ThreadPoolExecutor(max_workers=conc) as pool:
                it = iter(list(enumerate(games)))
                inflight = set()

                def submit_next():
                    nxt = next(it, None)
                    if nxt is None:
                        return False
                    gg, (oi, pw) = nxt
                    inflight.add(pool.submit(play_one, gg, oi, pw))
                    return True

                for _ in range(conc):                  # always prime a batch (measure before gating)
                    if not submit_next():
                        break
                # On each completion: score it (main thread), then refill UNLESS a bound crossed or the
                # time budget is spent (first crossing wins; let in-flight games drain).
                while inflight:
                    done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        r, dur = fut.result()
                        durations.append(dur)
                        d = process(r)
                        if d and decision is None:
                            decision = d
                    avg_dur = sum(durations) / len(durations)
                    if decision is None and have_budget():
                        for _ in range(len(done_set)):
                            if not submit_next():
                                break
        if decision is None:
            decision = ("inconclusive: time budget reached" if budget is not None
                        else "inconclusive: hit max_games without crossing a bound")
    finally:
        sfh.close()
        _close_arbiters()

    if budget is not None:
        elapsed = time.monotonic() - t0
        gph = (len(durations) / elapsed * 3600.0) if elapsed > 0 else 0.0
        print(f"[sprt] timed: {len(durations)} games in {elapsed/60:.1f} min "
              f"({gph:.0f} games/hr, avg {avg_dur:.0f}s/game)", flush=True)

    n = W + L + D
    score = (W + 0.5 * D) / n if n else 0.0
    standings = {
        "p1_label": args.p1_label, "p2_label": args.p2_label,
        "p1_config": args.p1_config, "p2_config": args.p2_config,
        "preset": args.preset or "STANDARD",
        "elo0": args.elo0, "elo1": args.elo1, "alpha": args.alpha, "beta": args.beta,
        "bound_lower": round(B, 4), "bound_upper": round(A, 4),
        "W": W, "L": L, "D": D, "void": void, "decided": n,
        "score_pct": round(100.0 * score, 1), "elo": round(elo_from_score(score) if n else 0.0, 1),
        "elo_margin": round(800.0 / math.sqrt(n), 1) if n else 0.0,
        "llr": round(llr, 4), "decision": decision,
    }
    with open(os.path.join(logdir, "sprt.json"), "w") as f:
        json.dump(standings, f, indent=2)

    print(f"\n[sprt] DECISION: {decision}", flush=True)
    print(f"[sprt]   +{W} -{L} ={D} of {n}  ({standings['score_pct']}%)  "
          f"elo~{standings['elo']:+}+/-{standings['elo_margin']}  LLR={llr:+.3f}"
          + (f"  [{void} void]" if void else ""), flush=True)
    print(f"[sprt]   verdict -> {os.path.join(logdir, 'sprt.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="SPRT A/B test for two engine configs.")
    ap.add_argument("--p1-config", default="", help="env knobs for player 1 (the candidate)")
    ap.add_argument("--p2-config", default="", help="env knobs for player 2 (the baseline)")
    ap.add_argument("--p1-label", default="P1")
    ap.add_argument("--p2-label", default="P2")
    ap.add_argument("--openings", default=os.path.join(THIS_DIR, "openings.txt"))
    ap.add_argument("--seed", type=int, default=0, help="seeds the opening order + color schedule")
    ap.add_argument("--preset", default="LIGHTNING", help="time control applied to BOTH players")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="play N games in parallel (default 1 = sequential). ~6 on 8c/16t. "
                         "Early-stop overshoots by <= N games as in-flight games drain.")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="extra stop: end once this wall-clock budget can't fit another game "
                         "(alongside the SPRT bound + --max-games). 0 = off.")
    ap.add_argument("--avg-seconds", type=float, default=0.0,
                    help="seed the per-game wall estimate for --max-minutes (0 = preset default)")
    ap.add_argument("--tag", default="sprt")
    ap.add_argument("--quiet", action="store_true", help="suppress per-move prints")
    # SPRT parameters
    ap.add_argument("--elo0", type=float, default=0.0, help="H0 Elo bound (null: no improvement)")
    ap.add_argument("--elo1", type=float, default=5.0, help="H1 Elo bound (alt: real improvement)")
    ap.add_argument("--alpha", type=float, default=0.05, help="false-accept rate")
    ap.add_argument("--beta", type=float, default=0.05, help="false-reject rate")
    ap.add_argument("--max-games", type=int, default=1000, help="hard cap (inconclusive if reached)")
    ap.add_argument("--min-games", type=int, default=10, help="games before bounds are tested")
    # Adjudication (same flags/semantics as tournament.py)
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--sf-depth", type=int, default=None)
    ap.add_argument("--sf-movetime", type=float, default=0.5)
    ap.add_argument("--adjudicate", action="store_true", help="enable BOTH win and draw adjudication")
    ap.add_argument("--adjudicate-draw", action="store_true")
    ap.add_argument("--adjudicate-win", action="store_true")
    ap.add_argument("--draw-cp", type=int, default=40)
    ap.add_argument("--win-p", type=float, default=5.0)
    ap.add_argument("--noprog-plies", type=int, default=40)
    ap.add_argument("--low-pieces", type=int, default=12)
    ap.add_argument("--adj-window", type=int, default=6)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
