"""Attribute self-play results to the inherited opening, using a NEUTRAL Stockfish eval.

The UHO book is deliberately unbalanced and every line is played from both colours, so a raw W/D/L
tally mixes "lost because the opening was already lost" with "lost a playable game". This scores each
opening-EXIT position (the FEN after the last booked move — identical for both games of a pair) with
Stockfish once, then buckets player-1's results by how good/bad the position it INHERITED was, from
p1's point of view. The question it answers: were p1's losses expected (handed a worse/lost position)
or real (lost from equal/winning)?

Usage (run via WSL anaconda python):
    python diagnostics/opening_attribution.py --tag current_vs_old_blitz [--depth 16]
"""
import argparse
import csv
import json
import os
import sys

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SELFPLAY_DIR = os.path.join(os.path.dirname(THIS_DIR), "selfplay")
sys.path.insert(0, SELFPLAY_DIR)
from arbiter import Arbiter, find_stockfish  # noqa: E402


def opening_exit_fen(jsonl_path):
    """Return the FEN after the last booked (opening) move — the position the engines inherit."""
    last = None
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") == "move" and rec.get("opening") and rec.get("fen"):
                last = rec["fen"]
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth", type=int, default=16, help="fixed Stockfish depth per opening-exit position")
    args = ap.parse_args()

    games_dir = os.path.join(SELFPLAY_DIR, "games", args.tag)
    summary = os.path.join(games_dir, "summary.csv")
    rows = list(csv.DictReader(open(summary)))

    # opening-exit FEN per opening_idx (both games of a pair share it -> eval once)
    fen_by_op = {}
    for r in rows:
        op = r["opening_idx"]
        if op in fen_by_op:
            continue
        jl = os.path.join(games_dir, "game_%03d" % int(r["game"]), "game.jsonl")
        fen = opening_exit_fen(jl)
        if fen:
            fen_by_op[op] = fen

    sf = find_stockfish()
    arb = Arbiter(sf, depth=args.depth)
    print("[attrib] Stockfish=%s depth=%d, scoring %d unique opening exits..."
          % (os.path.basename(str(sf)), args.depth, len(fen_by_op)), flush=True)
    wpov_by_op = {}
    for i, (op, fen) in enumerate(fen_by_op.items()):
        cp, _, _ = arb.evaluate(chess.Board(fen))
        wpov_by_op[op] = cp
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(fen_by_op)), flush=True)
    arb.engine.quit()

    # buckets by inherited eval from p1's POV (SF centipawns)
    edges = [(-1e9, -150, "handed LOST   (<=-150)"),
             (-150, -50, "handed worse  (-150..-50)"),
             (-50, 50, "~ EQUAL       (-50..+50)"),
             (50, 150, "handed better (+50..+150)"),
             (150, 1e9, "handed WON    (>=+150)")]
    buckets = {lbl: {"n": 0, "w": 0, "d": 0, "l": 0} for _, _, lbl in edges}

    skipped = 0
    for r in rows:
        op = r["opening_idx"]
        if op not in wpov_by_op or wpov_by_op[op] is None:
            skipped += 1
            continue
        wpov = wpov_by_op[op]
        inh = wpov if r["p1_color"] == "white" else -wpov  # from p1 (current) POV
        s = float(r["p1_score"])
        for lo, hi, lbl in edges:
            if lo <= inh < hi:
                b = buckets[lbl]
                b["n"] += 1
                b["w"] += 1 if s == 1.0 else 0
                b["d"] += 1 if s == 0.5 else 0
                b["l"] += 1 if s == 0.0 else 0
                break

    print("\n=== %s — current's results by INHERITED opening eval (neutral SF, current's POV) ===" % args.tag)
    print("  %-26s %5s  %4s %4s %4s   %6s" % ("inherited", "games", "W", "D", "L", "score"))
    tot_l = 0
    for _, _, lbl in edges:
        b = buckets[lbl]
        tot_l += b["l"]
        sc = (b["w"] + 0.5 * b["d"]) / b["n"] * 100 if b["n"] else 0.0
        print("  %-26s %5d  %4d %4d %4d   %5.1f%%" % (lbl, b["n"], b["w"], b["d"], b["l"], sc))

    # headline: where did the losses come from?
    print("\n=== WHERE current's %d LOSSES came from ===" % tot_l)
    lost_from_ok = 0
    for _, _, lbl in edges:
        b = buckets[lbl]
        if not tot_l:
            break
        pct = b["l"] / tot_l * 100
        tag = ""
        if lbl.startswith("~ EQUAL") or "better" in lbl or "WON" in lbl:
            lost_from_ok += b["l"]
            tag = "  <-- REAL (lost an equal/winning position)"
        elif "worse" in lbl or "LOST" in lbl:
            tag = "  (expected: handed a worse/lost position)"
        print("  %-26s %4d losses  (%4.1f%% of losses)%s" % (lbl, b["l"], pct, tag))
    if tot_l:
        print("\n  => %d / %d losses (%.1f%%) were from EQUAL-OR-BETTER positions (real failures);"
              % (lost_from_ok, tot_l, lost_from_ok / tot_l * 100))
        print("     the rest were from positions Stockfish already judged worse/lost for current.")
    if skipped:
        print("  (%d games skipped: no opening-exit FEN / eval)" % skipped)


if __name__ == "__main__":
    main()
