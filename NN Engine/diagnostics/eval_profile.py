# -*- coding: utf-8 -*-
"""
Phase-resolved per-term EVAL profiler for the C++ chess engine.

The static eval (placement_and_piece_eval) is the engine's double bottleneck:
imprecise (the strength ceiling) AND slow (its non-incremental cost is what killed
the improving heuristic). Before making any term lazy / only-when-necessary we need
to know WHERE the eval spends its cycles, by game phase. `perf` is dead in this WSL2
and -Ofast -flto inlines every eval sub-evaluator, so this uses manual __rdtsc
instrumentation compiled into a gated build.

Build the instrumented extension first (from NN Engine/, in WSL):
    PROFILE_EVAL=1 python setupAI.py build_ext --inplace

Then run (from NN Engine/):
    python diagnostics/eval_profile.py                       # defaults: tag=newstack_uho
    python diagnostics/eval_profile.py --tag away_standard --reps 2000 --cap 200

A normal `python setupAI.py build_ext --inplace` (no PROFILE_EVAL) leaves the C++
profiler hooks as no-ops; this script then prints a notice telling you to rebuild.

Output: a per-term x 8-phase %-share table (relative cycles, not absolute ns) plus
calls/eval, so you can spot the hot terms and how they move across phases — the
lazy-gate candidates ("term X = N% of eval but only matters in phase Y").

This is strictly diagnostic — it never changes engine behaviour.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TensorFlow startup chatter

import sys
import glob
import json
import argparse

# --- diagnostics layout: tools live in NN Engine/diagnostics/; the built ChessAI
#     .so lives one level up in NN Engine/. Mirrors tactical_test.py. ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)            # NN Engine/  (has ChessAI*.so)
GAMES_DIR = os.path.join(ENGINE_DIR, "selfplay", "games")
sys.path.insert(0, ENGINE_DIR)

import chess
from ChessAI import ChessAI

MAX_PHASE = 24  # must match cpp_bitboard.h

# The 8 phase bins, in display order. Keys are returned by classify_bin().
PHASE_BINS = [
    "1_early_opening",
    "2_late_opening",
    "3_early_middle",
    "4_mid_middle",
    "5_late_middle",
    "6_early_endgame",
    "7_mid_endgame",
    "8_late_endgame",
]


def material_phase_score(board):
    """Replicate the engine's material phase: 0 (full material) .. 128 (bare kings).
    phase = 4*Q + 2*R + 1*(B|N) popcounts; phase_score = 128*(MAX_PHASE-phase)/MAX_PHASE."""
    q = chess.popcount(board.queens)
    r = chess.popcount(board.rooks)
    bn = chess.popcount(board.bishops | board.knights)
    phase = 4 * q + 2 * r + bn
    ps = 128 * (MAX_PHASE - phase) // MAX_PHASE
    return max(0, min(128, ps))


def classify_bin(board, move_no):
    """Bin a position into one of the 8 phases by move number + material phase_score.
    The two opening bins are peeled off by move number (while material is still high);
    the rest are keyed on phase_score, whose thresholds straddle the eval's own
    midgame/endgame/near-end branch points (64 / 96)."""
    ps = material_phase_score(board)
    if move_no <= 6:
        return "1_early_opening"
    if move_no <= 12 and ps <= 35:
        return "2_late_opening"
    if ps <= 32:
        return "3_early_middle"
    if ps <= 50:
        return "4_mid_middle"
    if ps <= 64:
        return "5_late_middle"
    if ps <= 80:
        return "6_early_endgame"
    if ps <= 100:
        return "7_mid_endgame"
    return "8_late_endgame"


def collect_positions(tag, per_bin_cap):
    """Walk selfplay/games/<tag>/game_*/game.jsonl and bucket every non-terminal
    move position into its phase bin. Returns {bin: [chess.Board, ...]} capped per bin
    (sampled with a stride so the cap spreads across the whole corpus, not the first
    few games)."""
    pattern = os.path.join(GAMES_DIR, tag, "game_*", "game.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, files

    raw = {b: [] for b in PHASE_BINS}
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("type") != "move":
                        continue
                    fen = rec.get("fen")
                    if not fen:
                        continue
                    board = chess.Board(fen)
                    # placement_and_piece_eval assumes a non-terminal position.
                    if board.is_checkmate() or board.is_stalemate():
                        continue
                    move_no = rec.get("move_no") or board.fullmove_number
                    raw[classify_bin(board, move_no)].append(board)
        except OSError:
            continue

    # Stride-sample each bin down to the cap so we cover the breadth of the corpus.
    binned = {}
    for b in PHASE_BINS:
        positions = raw[b]
        if len(positions) > per_bin_cap:
            stride = len(positions) / float(per_bin_cap)
            positions = [positions[int(i * stride)] for i in range(per_bin_cap)]
        binned[b] = positions
    return binned, files


def profile_bin(ai, positions, reps):
    """Reset accumulators, drive `reps` evals over every position in the bin, return the
    per-term accumulator snapshot (a list of dicts: term/cycles/calls/exclusive)."""
    ai.reset_profile()
    for board in positions:
        ai.profile_eval(board, reps)
    return ai.get_profile()


def main():
    ap = argparse.ArgumentParser(description="Phase-resolved per-term eval profiler.")
    ap.add_argument("--tag", default="newstack_uho",
                    help="self-play game tag under selfplay/games/ to source FENs from")
    ap.add_argument("--reps", type=int, default=2000,
                    help="evals per position (more = steadier per-term cycle signal)")
    ap.add_argument("--cap", type=int, default=200,
                    help="max positions per phase bin (stride-sampled across the corpus)")
    args = ap.parse_args()

    # One warm engine, reused for every position: the attack tables + Config toggles are
    # global and set once in the constructor; placement_and_piece_eval sets its own
    # per-position masks. (No need for a fresh ChessAI per FEN here.)
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    if ai.get_profile() == []:
        print("=" * 72)
        print("This ChessAI was built WITHOUT the eval profiler (production build).")
        print("Rebuild the instrumented extension first, from NN Engine/ in WSL:")
        print("    PROFILE_EVAL=1 python setupAI.py build_ext --inplace")
        print("=" * 72)
        return

    binned, files = collect_positions(args.tag, args.cap)
    if binned is None:
        print("No games found for tag '%s' under %s" % (args.tag, GAMES_DIR))
        print("Pick an existing tag, e.g. --tag away_standard")
        return

    print("Corpus: tag=%s (%d game files) | reps=%d/pos | cap=%d/bin"
          % (args.tag, len(files), args.reps, args.cap))
    print("Positions per bin: " + ", ".join("%s=%d" % (b, len(binned[b])) for b in PHASE_BINS))
    print()

    # Drive each bin and collect its per-term snapshot.
    results = {}     # bin -> {term: {cycles, calls, exclusive}}
    term_order = None
    for b in PHASE_BINS:
        if not binned[b]:
            results[b] = None
            continue
        snap = profile_bin(ai, binned[b], args.reps)
        results[b] = {row["term"]: row for row in snap}
        if term_order is None:
            term_order = [row["term"] for row in snap]
        # Echo the C++ stderr table too (quick eyeball / cross-check).
        ai.dump_profile(b)

    # Per-bin exclusive base = sum of cycles over the exclusive (top-level) terms.
    def bin_base(b):
        if not results[b]:
            return 0
        return sum(r["cycles"] for r in results[b].values() if r["exclusive"]) or 1

    # ---- per-term x phase %-share table ----
    active_bins = [b for b in PHASE_BINS if results[b]]
    print()
    print("PER-TERM %-SHARE BY PHASE (share of each bin's exclusive-term cycle base)")
    header = "%-16s" % "term" + "".join("%9s" % b.split("_", 1)[1][:8] for b in active_bins) + "%9s" % "calls/ev"
    print(header)
    print("-" * len(header))

    # Order terms by total cycles across all bins (hottest first).
    totals = {t: 0 for t in term_order}
    total_evals = {}
    for b in active_bins:
        total_evals[b] = len(binned[b]) * args.reps
        for t in term_order:
            totals[t] += results[b][t]["cycles"]
    ordered = sorted(term_order, key=lambda t: totals[t], reverse=True)

    for t in ordered:
        row = "%-16s" % t
        calls_sum = 0
        evals_sum = 0
        for b in active_bins:
            r = results[b][t]
            share = 100.0 * r["cycles"] / bin_base(b)
            row += "%8.1f%%" % share
            calls_sum += r["calls"]
            evals_sum += total_evals[b]
        calls_per_eval = (calls_sum / evals_sum) if evals_sum else 0.0
        excl = "" if results[active_bins[0]][t]["exclusive"] else " (nested)"
        row += "%9.2f" % calls_per_eval + excl
        print(row)

    print("-" * len(header))
    print("Note: 'nested' terms (ROOK_ACTIVITY, SEE, BISHOP_*) are subsets of a parent")
    print("term and are excluded from the %-share base; SEE/BISHOP_* read 0 until the")
    print("line-level drill scopes are enabled. calls/ev should scale with piece count.")

    # Hottest exclusive term per phase — the headline lazy-gate signal.
    print()
    print("HOTTEST EXCLUSIVE TERM PER PHASE:")
    for b in active_bins:
        base = bin_base(b)
        excl_terms = [(t, results[b][t]["cycles"]) for t in term_order
                      if results[b][t]["exclusive"]]
        t, cyc = max(excl_terms, key=lambda x: x[1])
        print("  %-16s %-14s %.1f%%" % (b, t, 100.0 * cyc / base))


if __name__ == "__main__":
    main()
