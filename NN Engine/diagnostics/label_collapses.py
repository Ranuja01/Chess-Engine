# -*- coding: utf-8 -*-
"""Label recorded collapse positions with Stockfish's best move -> STS-schema EPD (one-time, needs SF).

The self-play harness records every game where our eval peaked winning then collapsed
(selfplay/games/<corpus>/collapses.csv, decision_fen = the run-up position where we chose the losing
move). Those positions are exactly the broad failure set we want the eval tuned against, but move-match
scores positions from an STS-style EPD (c8 scores / c9 UCI). This converts each decision_fen into one
such line by asking Stockfish for the move we SHOULD have played:

    <fen> bm <SAN>; id "collapse.NNN"; c8 "10"; c9 "<best_uci>";

Single oracle move scored 10, so move-match credits us only when our search now picks SF's move. The
emitted lines parse natively through sts_test.load_sts_epd (set_epd needs the bm SAN; c8/c9 drive the
{uci: score} map). Two outputs: a standalone collapses_<corpus>.epd per corpus, and the merged
failure_corpus.epd (STS suite + all collapse lines) that the staged funnel samples.

This is the ONE Stockfish/interop step before the inner loop. Run via the dispatcher (STOCKFISH_PATH is
exported by `pyrun`); after an interop restore, run it FIRST so the busy SF process holds interop:
    bash overnight_runner.sh pyrun diagnostics/label_collapses.py

Optional positional args (no '=' tokens, so the dispatcher stays permission-clean):
    diagnostics/label_collapses.py [corpus ...]   # default: vssf_2400 vssf_2700 (those that exist)
"""

import os
import csv
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
SUITES_DIR = os.path.join(THIS_DIR, 'suites')
GAMES_DIR = os.path.join(ENGINE_DIR, 'selfplay', 'games')
DEFAULT_SUITE = os.path.join(SUITES_DIR, 'STS1-STS15_LAN_v3.epd')
MERGED_OUT = os.path.join(SUITES_DIR, 'failure_corpus.epd')
DEFAULT_CORPORA = ('vssf_2400', 'vssf_2700')

# arbiter lives beside the self-play harness; reuse it (and its find_stockfish) wholesale.
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, os.path.join(ENGINE_DIR, 'selfplay'))

import chess  # noqa: E402
from arbiter import Arbiter, find_stockfish  # noqa: E402

# Time-gate the oracle generously: this runs once and the move quality must be trustworthy.
SF_DEPTH = int(os.environ.get('LABEL_SF_DEPTH', '20'))


def label_corpus(corpus, arb, seen_fens):
    """Read games/<corpus>/collapses.csv, label each decision_fen, return a list of EPD lines.

    seen_fens de-dups across corpora (the same run-up can recur) so a position isn't double-weighted in
    the merged suite. Rows whose FEN is terminal / unparseable, or where SF returns no move, are skipped."""
    path = os.path.join(GAMES_DIR, corpus, 'collapses.csv')
    if not os.path.exists(path):
        print("  (skip) %s: no collapses.csv" % corpus)
        return []
    lines = []
    with open(path) as f:
        for row in csv.DictReader(f):
            fen = (row.get('decision_fen') or '').strip()
            if not fen or fen in seen_fens:
                continue
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            if board.is_game_over():
                continue
            cp, best, _ = arb.evaluate(board)
            if not best:
                continue
            try:
                san = board.san(chess.Move.from_uci(best))
            except Exception:
                continue
            seen_fens.add(fen)
            # Emit an STS-shaped id "collapse(v1) <Theme>.NNN" so sts_test._theme_of groups every collapse
            # under one per-corpus theme line (Collapse2400 / Collapse2700) instead of 63 singletons — that
            # grouped readout is how the funnel classifies helped-vs-hurt classes.
            theme = corpus.replace('vssf_', 'Collapse')
            # STS lines (and load_sts_epd's set_epd) expect a 4-field EPD prefix, not a full 6-field FEN;
            # the trailing half/full-move counters would break set_epd. Drop them like the STS suite does.
            epd4 = " ".join(fen.split()[:4])
            lines.append('%s bm %s; id "collapse(v1) %s.%03d"; c8 "10"; c9 "%s";'
                         % (epd4, san, theme, len(lines), best))
    print("  %s: %d labeled (depth=%d)" % (corpus, len(lines), SF_DEPTH))
    return lines


def main():
    corpora = sys.argv[1:] or list(DEFAULT_CORPORA)
    sf = find_stockfish()
    if not sf:
        print("ERROR: Stockfish not found (set STOCKFISH_PATH).")
        return 1
    arb = Arbiter(sf, depth=SF_DEPTH)
    seen = set()
    all_lines = []
    try:
        for corpus in corpora:
            lines = label_corpus(corpus, arb, seen)
            if lines:
                out = os.path.join(SUITES_DIR, 'collapses_%s.epd' % corpus)
                with open(out, 'w') as f:
                    f.write("\n".join(lines) + "\n")
                print("  -> %s" % out)
            all_lines.extend(lines)
    finally:
        arb.close()

    if not all_lines:
        print("no collapse positions labeled; merged corpus not written.")
        return 1

    sts_lines = []
    if os.path.exists(DEFAULT_SUITE):
        with open(DEFAULT_SUITE) as f:
            sts_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    with open(MERGED_OUT, 'w') as f:
        f.write("\n".join(sts_lines + all_lines) + "\n")
    print("\nmerged: %d STS + %d collapse = %d -> %s"
          % (len(sts_lines), len(all_lines), len(sts_lines) + len(all_lines), MERGED_OUT))

    # Self-verify: the collapse lines must re-parse natively through the move-match loader (set_epd +
    # c8/c9 zip), or the funnel would silently score only the STS positions. Fail loud if any drop out.
    from sts_test import load_sts_epd
    parsed = load_sts_epd(MERGED_OUT)
    n_col = sum(1 for p in parsed if p[4].startswith('collapse'))
    print("verify: load_sts_epd parsed %d total, %d collapse (expected %d)"
          % (len(parsed), n_col, len(all_lines)))
    if n_col != len(all_lines):
        print("ERROR: %d collapse lines did not round-trip through load_sts_epd." % (len(all_lines) - n_col))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
