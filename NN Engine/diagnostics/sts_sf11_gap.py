# -*- coding: utf-8 -*-
"""Per-theme STS-failure eval-breakdown gap vs Stockfish 11 — WHICH of our terms misjudges, by theme.

Reads a completed STS run (diagnostics/results/sts_results_<tag>.csv), takes the FAILURE positions
(score < max, non-book), groups by STS theme, and compares OUR per-term eval breakdown
(ChessAI.ev_breakdown) side-by-side with SF11's classical `eval` term table. The point is NOT to copy
SF11 but to SEE where our per-term judgment diverges on the exact positions we play wrong — so we can
correct/condition the culprit term (the capg70 template). Pawn-advancement play is a SYSTEM (passer +
rook-behind + king-march + piece activity), so a per-theme run ALSO dumps the worst individual
positions with our-move vs best-move and the full cluster of terms, to read the interaction.

Reuses SF11Eval + OUR_TERMS from eval_vs_sf11.py. Uses the native-ELF SF11 (subprocess-safe under WSL).
Our eval config is set by env knobs read at ChessAI init (e.g. SCALE_CAPTURE_GAINS=70) — pass them
before invoking. Units: our terms are Black-positive milli-pawns -> White-POV pawns = -v/1000; SF11's
`eval` table is already White-POV pawns.

Run (WSL, from NN Engine/):
    SCALE_CAPTURE_GAINS=70 python diagnostics/sts_sf11_gap.py <tag> [theme-substr] [dump_n]
or via the dispatcher `sts_gap <tag> [theme] [dump_n] [KNOBS]`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import csv
import contextlib
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, THIS_DIR)

import chess  # noqa: E402
from eval_vs_sf11 import SF11Eval  # reuse the SF11 `eval` parser
from sts_test import load_sts_epd  # for the best-move (c9) join

SF11_ELF = ("/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/"
            "stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2")
STS_EPD = os.path.join(THIS_DIR, "suites", "STS1-STS15_LAN_v3.epd")
RESULTS_DIR = os.path.join(THIS_DIR, "results")

# Terms to display (base + the gated advancement-cluster terms; 0 if a gate is off).
DISPLAY_TERMS = ["material", "pieces", "capture_gains", "passed_pawn_support", "pawn_majority",
                 "central", "mobility", "pawn_struct", "outpost", "latent_threat", "king_safety",
                 "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]
# Rough ours->SF11-label correspondence (directional suspects, not strict 1:1).
CORR = {
    "material": "Material", "pieces": "Material", "piece_value_boost": "Material",
    "capture_gains": "Threats", "latent_threat": "Threats", "passed_pawn_support": "Passed",
    "pawn_majority": "Passed", "king_safety": "King safety", "central": "Space",
    "mobility": "Mobility", "imbalance_white": "Imbalance", "imbalance_black": "Imbalance",
    "pair_bonus": "Bishops",
}


def wp(bd, term):
    return -bd.get(term, 0) / 1000.0  # White-POV pawns


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "ship"
    theme_filter = sys.argv[2] if len(sys.argv) > 2 else None
    dump_n = int(sys.argv[3]) if len(sys.argv) > 3 else (8 if theme_filter else 0)
    csv_path = os.path.join(RESULTS_DIR, "sts_results_%s.csv" % tag)
    if not os.path.exists(csv_path):
        print("no STS results at %s (run `sts_full %s ...` first)" % (csv_path, tag))
        return 1

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    fails = []
    for r in rows:
        if r.get("score", "") == "":
            continue
        try:
            if int(r["score"]) < int(r["max"]):
                fails.append(r)
        except ValueError:
            continue
    if theme_filter:
        fails = [r for r in fails if theme_filter.lower() in r["theme"].lower()]
    if not fails:
        print("no failure rows for tag=%s%s" % (tag, (" theme~%s" % theme_filter) if theme_filter else ""))
        return 1

    # best-move (highest-c8) per FEN, for the per-position dump
    best_of = {}
    for fen, score_map, mx, theme, epd_id in load_sts_epd(STS_EPD):
        if score_map:
            best_of[fen] = max(score_map, key=score_map.get)

    by_theme = defaultdict(list)
    for r in fails:
        by_theme[r["theme"]].append(r)

    from ChessAI import ChessAI  # noqa
    seed = chess.Board()
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        ai = ChessAI(None, None, seed, seed.turn)   # suppress the init knob-dump
    sf = SF11Eval(SF11_ELF)
    try:
        for theme in sorted(by_theme, key=lambda t: -len(by_theme[t])):
            frows = by_theme[theme]
            our_sum = defaultdict(float)
            sf_sum = defaultdict(float)
            gap_sum = 0.0
            nn = 0
            detail = []
            for r in frows:
                try:
                    board = chess.Board(r["fen"])
                except Exception:
                    continue
                bd = ai.ev_breakdown(board)
                if bd.get("checkmate"):
                    continue
                sf_total, sf_terms = sf.eval(r["fen"])
                if sf_total is None:
                    continue
                nn += 1
                our = -bd["total"] / 1000.0
                gap_sum += our - sf_total
                for t in DISPLAY_TERMS:
                    our_sum[t] += wp(bd, t)
                for lbl, v in sf_terms.items():
                    if lbl != "Total":
                        sf_sum[lbl] += v
                detail.append((int(r["score"]), r["fen"], r["engine"], bd, our, sf_total, sf_terms))
            if nn == 0:
                continue
            print("=" * 96)
            print("THEME: %-24s failures=%-4d mean total gap (ours-SF11, WP pawns)=%+.2f"
                  % (theme, nn, gap_sum / nn))
            ot = sorted(((t, our_sum[t] / nn) for t in DISPLAY_TERMS), key=lambda kv: -abs(kv[1]))
            print("  OURS mean : " + "  ".join("%s=%+.2f" % (t, v) for t, v in ot if abs(v) > 0.02))
            st = sorted(sf_sum.items(), key=lambda kv: -abs(kv[1]))
            print("  SF11 mean : " + "  ".join("%s=%+.2f" % (l, v / nn) for l, v in st if abs(v / nn) > 0.02))

            if dump_n and theme_filter:
                print("\n  --- worst %d positions (our move vs best; cluster terms ours|SF11) ---" % dump_n)
                for sc, fen, mv, bd, our, sft, sft_terms in sorted(detail, key=lambda x: x[0])[:dump_n]:
                    best = best_of.get(fen, "?")
                    print("  " + "-" * 92)
                    print("  %s" % fen)
                    print("    our=%-6s best=%-6s  ourEval=%+.2f  SF11=%+.2f  gap=%+.2f"
                          % (mv, best, our, sft, our - sft))
                    oc = [(t, wp(bd, t)) for t in DISPLAY_TERMS if abs(wp(bd, t)) > 0.03]
                    oc.sort(key=lambda kv: -abs(kv[1]))
                    print("    OURS: " + "  ".join("%s=%+.2f" % (t, v) for t, v in oc))
                    sc2 = sorted(((l, v) for l, v in sft_terms.items() if l != "Total" and abs(v) > 0.03),
                                 key=lambda kv: -abs(kv[1]))
                    print("    SF11: " + "  ".join("%s=%+.2f" % (l, v) for l, v in sc2))
    finally:
        sf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
