# -*- coding: utf-8 -*-
"""WDL-cploss compass over a FROZEN stratified corpus — the deterministic inner-loop objective for the holistic
co-tune. For each corpus position, SF18 (the judge) scores the position (best play) and the position after OUR move;
loss = the drop in WIN-PROBABILITY (Lichess WDL model), which caps decided-position influence and matches outcome
units. Lower = our eval picks better moves. Deterministic: fixed SF depth + fixed engine depth + env-latched knobs.

SF18 = judge of CONSEQUENCES (we minimise regret as it measures it; we never fit our eval toward its numbers), so
the SF-magnitude-target trap does not apply (argmax). SF's per-position best-eval is CACHED (position-only, config-
independent) so a candidate run scores only the after-our-move branch + our move.

    [KNOBS...] python diagnostics/cploss_frozen.py [corpus.csv] [--depth D] [--limit N]
Prints per-stratum + overall mean WDL-loss (×1000). Knobs (SCALE_*/KING_SAFETY_MAG/... ) via process env.
"""
import argparse
import csv
import json
import math
import os
import random
import sys

import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))
from arbiter import Arbiter, find_stockfish   # noqa: E402
from tactical_test import run_one              # noqa: E402

WIN_K = 0.00368208                              # Lichess cp->win% sigmoid constant (per cp); reused from deep_diag.py


def winpct(cp):
    cp = max(-1500, min(1500, cp))
    return 100.0 / (1.0 + math.exp(-WIN_K * cp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default=os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))
    ap.add_argument("--depth", type=int, default=12)      # SF18 judge depth (fixed => deterministic labels)
    ap.add_argument("--limit", type=int, default=0)        # 0 = all
    ap.add_argument("--shard", choices=["all", "train", "holdout"], default="all")  # even/odd split for overfit guard
    # Random subset of the corpus. --limit takes a PREFIX (same rows every run, so repeated measurement can be
    # overfit to them); --sample draws fresh rows instead. The SF-best cache below is keyed by fen@depth and is
    # config-independent, so sampling within ONE large judged corpus costs nothing extra -- the cache just fills
    # in as new positions are drawn. seed 0 = fresh draw per run; nonzero = reproducible.
    # Per-position dump. The mean alone cannot answer whether a config changes the TAIL (rare large losses), and
    # Elo appears to be tail-driven while cploss/STS are mean-like. Dumping lets a finished run be re-analysed for
    # blunder rate / percentiles without paying for the search again.
    ap.add_argument("--dump", default="")
    ap.add_argument("--sample", type=int, default=0)
    # Fixed by default so configs compared in one campaign see the SAME positions -- a per-run random draw would
    # make them incomparable. Change the seed deliberately BETWEEN campaigns to rotate onto fresh positions.
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.corpus)))
    if args.shard == "train":
        rows = rows[0::2]
    elif args.shard == "holdout":
        rows = rows[1::2]
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed or None).sample(rows, args.sample)
    if args.limit:
        rows = rows[: args.limit]

    # position-only SF-best cache (config-independent) keyed by fen@depth
    cache_path = args.corpus.replace(".csv", ".sfcache_d%d.json" % args.depth)
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}

    arb = Arbiter(find_stockfish(), depth=args.depth)
    per = {}       # stratum -> [losses]
    dump_rows = []  # (fen, stratum, our_move, loss) when --dump is set
    stats = {"dirty": False}

    def sf_eval(board):
        """SF18 White-POV cp at fixed depth, cached by position (clocks ignored) — config-independent, so it
        warms across candidates: cp_best always hits after round 1, cp_after hits whenever the move recurs."""
        key = " ".join(board.fen().split()[:4])
        if key in cache:
            return cache[key]
        cp, _, _ = arb.evaluate(board)
        if cp is not None:
            cache[key] = cp; stats["dirty"] = True
        return cp

    for r in rows:
        fen = r["fen"]; st = r.get("stratum", "?")
        try:
            b = chess.Board(fen)
            cp_best = sf_eval(b)
            if cp_best is None:
                continue
            our = run_one(fen, set())["uci"]                # our engine's move (env-latched knobs)
            b.push_uci(our)
            cp_after = sf_eval(b)
            if cp_after is None:
                continue
        except Exception:
            continue
        white_to_move = fen.split()[1] == "w"
        if white_to_move:
            loss = winpct(cp_best) - winpct(cp_after)
        else:
            loss = winpct(cp_after) - winpct(cp_best)
        loss = max(0.0, loss)
        per.setdefault(st, []).append(loss)
        dump_rows.append((fen, st, our, loss))

    if hasattr(arb, "close"):
        arb.close()
    if stats["dirty"]:
        json.dump(cache, open(cache_path, "w"))

    allv = [x for v in per.values() for x in v]
    # Known strata first (older corpora), then ANY others present. Previously this was a hardcoded list, so a
    # corpus using different stratum names printed no breakdown at all -- the numbers were computed and silently
    # dropped. Never gate display on a fixed name list.
    known = ["collapse", "sts", "neutral", "game"]
    order = [s for s in known if s in per] + sorted(s for s in per if s not in known)
    def mean(v):
        return sum(v) / len(v) if v else 0.0
    # ×10 so the win% (0..100) prints as milli-units comparable across runs; report overall first (the score).
    def tailstr(v):
        """Catastrophe rates + upper percentiles. The MEAN is blind to the thing that tracks Elo: default vs
        qdelta-OFF (a +53.8 Elo gap) scored an IDENTICAL mean while differing 60% in rate>20%."""
        if not v:
            return ""
        s = sorted(v)
        q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
        return "  >5%%=%.1f%% >10%%=%.1f%% >20%%=%.1f%%  p95=%.1f p99=%.1f" % (
            100.0 * sum(1 for x in s if x > 5) / len(s),
            100.0 * sum(1 for x in s if x > 10) / len(s),
            100.0 * sum(1 for x in s if x > 20) / len(s),
            q(0.95), q(0.99))

    print("cploss_wdl OVERALL=%.2f  n=%d  depth=%d%s" % (mean(allv) * 10, len(allv), args.depth, tailstr(allv)))
    # Per-stratum TAIL as well as mean: if catastrophes concentrate in one tier we can sample it heavily (and
    # reweight) for cheap resolution; if they are spread evenly, only raw volume helps.
    for st in order:
        print("  %-11s %.2f  n=%-5d%s" % (st, mean(per[st]) * 10, len(per[st]), tailstr(per[st])))

    # TAIL statistics. The mean is dominated by many small losses; a game's RESULT is decided by the rare large
    # ones. cploss/STS means have repeatedly failed to predict Elo (a change worth +53.8 Elo barely moved any
    # mean), so report blunder RATE and percentiles alongside -- these are the candidates for a metric that
    # actually tracks outcomes. Thresholds are win% (loss is 0..100).
    sv = sorted(allv)
    def pct(p):
        return sv[min(len(sv) - 1, int(p * len(sv)))] if sv else 0.0
    if sv:
        print("  tail: rate>5%%=%.1f%%  rate>10%%=%.1f%%  rate>20%%=%.1f%%  p90=%.1f  p95=%.1f  p99=%.1f" % (
            100.0 * sum(1 for x in sv if x > 5) / len(sv),
            100.0 * sum(1 for x in sv if x > 10) / len(sv),
            100.0 * sum(1 for x in sv if x > 20) / len(sv),
            pct(0.90), pct(0.95), pct(0.99)))

    if args.dump:
        with open(args.dump, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fen", "stratum", "our_move", "loss_winpct"])
            for row in dump_rows:
                w.writerow(["%s" % row[0], row[1], row[2], "%.4f" % row[3]])
        print("  dumped %d rows -> %s" % (len(dump_rows), args.dump))


if __name__ == "__main__":
    main()
