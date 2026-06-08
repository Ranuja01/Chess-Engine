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
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import chess
from selfplay import play_game, Adjudicator
from arbiter import Arbiter, find_stockfish
# Reuse the tested tournament helpers so an SPRT run matches a fixed tournament exactly.
from tournament import load_openings, schedule, p1_score, elo_from_score, _config_with_preset


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
    games = schedule(args.max_games, len(openings), args.seed)

    p1c = _config_with_preset(args.p1_config, args.preset)
    p2c = _config_with_preset(args.p2_config, args.preset)

    # Optional in-play adjudication (one shared Stockfish, fresh per-game Adjudicator) — same wiring
    # as tournament.py so SPRT and fixed runs adjudicate identically.
    do_draw = args.adjudicate or args.adjudicate_draw
    do_win = args.adjudicate or args.adjudicate_win
    adj_arbiter = None
    if do_draw or do_win:
        sf = args.sf_path or find_stockfish()
        if sf:
            try:
                adj_arbiter = Arbiter(sf, movetime=args.sf_movetime, depth=args.sf_depth)
                print(f"[sprt] adjudication ON (SF confirm @ {args.sf_movetime}s)", flush=True)
            except Exception as e:
                print(f"[sprt] WARNING: adjudication requested but Stockfish failed ({e}); OFF", flush=True)
                do_draw = do_win = False
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

    print(f"[sprt] {args.p1_label} vs {args.p2_label}  H0:<={args.elo0} H1:>={args.elo1} Elo  "
          f"alpha={args.alpha} beta={args.beta}  bounds=[{B:+.3f}, {A:+.3f}]  "
          f"preset={args.preset or 'STANDARD'}  max_games={args.max_games}", flush=True)

    W = L = D = void = 0
    llr = 0.0
    decision = "inconclusive"
    for g, (opening_idx, p1_white) in enumerate(games):
        gdir = os.path.join(logdir, f"game_{g:03d}")
        white_cfg, white_lbl = (p1c, args.p1_label) if p1_white else (p2c, args.p2_label)
        black_cfg, black_lbl = (p2c, args.p2_label) if p1_white else (p1c, args.p1_label)
        adj = (Adjudicator(adj_arbiter, draw_cp=args.draw_cp, win_p=args.win_p,
                           noprog_plies=args.noprog_plies, low_pieces=args.low_pieces,
                           window=args.adj_window, do_win=do_win, do_draw=do_draw)
               if adj_arbiter is not None else None)
        try:
            res = play_game(white_cfg, black_cfg, white_lbl, black_lbl, chess.STARTING_FEN,
                            args.max_plies, gdir, jsonl_path=os.path.join(gdir, "game.jsonl"),
                            verbose=not args.quiet, arbiter=None, opening_moves=openings[opening_idx],
                            adjudicator=adj)
            result, reason, plies = res["result"], res["reason"], res["plies"]
        except Exception as e:
            result, reason, plies = "*", f"driver error: {e}", 0

        sc = p1_score(result, p1_white)
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

        writer.writerow({"game": g, "opening_idx": opening_idx,
                         "p1_color": "white" if p1_white else "black",
                         "white": white_lbl, "black": black_lbl, "result": result,
                         "p1_score": sc, "plies": plies, "reason": reason,
                         "W": W, "L": L, "D": D, "llr": round(llr, 4)})
        sfh.flush()

        print(f"[sprt] game {g}: {result} ({reason})  +{W} -{L} ={D}"
              f"{f' [{void} void]' if void else ''}  "
              f"LLR={llr:+.3f} (need {B:+.2f}/{A:+.2f})  elo~{elo:+.0f}", flush=True)

        # Only test the bounds once there is enough data to have a meaningful variance estimate.
        if n >= args.min_games:
            if llr >= A:
                decision = "H1 accepted: P1 is stronger (>= elo1)"
                break
            if llr <= B:
                decision = "H0 accepted: P1 is NOT a >= elo1 improvement"
                break
    else:
        decision = "inconclusive: hit max_games without crossing a bound"

    sfh.close()
    if adj_arbiter is not None:
        adj_arbiter.close()

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
