# -*- coding: utf-8 -*-
"""Failure-attribution dossiers — turn each wrong move into (failure type ?, knobs involved).

The funnel finds WHERE the engine is wrong (move-match misses); this answers WHY, in a form an agent can
classify chess-semantically and then map to eval knobs. For every miss in a movematch scorecard it builds
a dossier:

  - the position, the move the engine PLAYED (wrong), and the suite's CORRECT move;
  - the per-term STATIC-EVAL DELTA between the two resulting positions, oriented to the mover. The eval is
    a sum of named terms; whichever term credits the played position more than the correct one is the knob
    that over-valued the blunder. Ranked, this is "exactly the knobs involved in this failure";
  - the live cheap detectors at the decision position (offense/defense/control/material edges, openness,
    king danger, phase) — the raw board state a conditioner would key on.

Aggregate mode prints, across all misses, the mean signed term-delta (which knob SYSTEMATICALLY over-credits
wrong moves) and a per-theme view. Dossier mode emits JSONL (one object per miss) for an agent to read,
classify the failure TYPE (overextension / passivity / bad trade / king exposure / missed prophylaxis / ...),
and bucket — the chess classification is NOT hardcoded here; this tool only assembles the evidence.

Run in WSL from NN Engine/:
    python diagnostics/classify_failures.py agg    mmcollapse --epd diagnostics/suites/failure_corpus.epd
    python diagnostics/classify_failures.py dossier mmcollapse --epd diagnostics/suites/failure_corpus.epd --limit 30 > misses.jsonl
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import csv
import json
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

from sts_test import load_sts_epd  # noqa: E402
from diagnose_misses import detectors, DET_KEYS  # reuse the detector derivation  # noqa: E402

# Terms that sum to `total` (material is informational). imbalance_white/black are folded into one net term.
ATTRIB_TERMS = ["pieces", "capture_gains", "passed_pawn_support", "latent_threat", "king_safety",
                "central", "imbalance", "pair_bonus", "piece_value_boost"]


def mover_pov_terms(bd, mover_white):
    """Per-term contributions in MOVER-POV pawns (+ = good for the side that just moved). Engine eval is
    Black-positive milli-pawns; White-POV = -ev/1000; mover-POV flips that for Black."""
    sign = -1.0 if mover_white else 1.0   # White-POV = -ev; mover=white wants White-POV, so -ev -> sign -1
    out = {}
    for t in ATTRIB_TERMS:
        if t == "imbalance":
            v = bd["imbalance_white"] + bd["imbalance_black"]
        else:
            v = bd[t]
        out[t] = sign * v / 1000.0
    return out


def attribute(ai, fen, uci_wrong, uci_right):
    """Term-delta (played - correct) in mover-POV pawns at the two resulting positions. Positive term =>
    that knob credits the WRONG move more than the right one => suspect. Returns (deltas, detectors)."""
    board = chess.Board(fen)
    mover_white = board.turn == chess.WHITE
    dets = detectors(ai.ev_breakdown(board), mover_white)

    def terms_after(uci):
        b = chess.Board(fen)
        try:
            b.push(chess.Move.from_uci(uci))
        except Exception:
            return None
        bd = ai.ev_breakdown(b)
        if bd.get("checkmate"):
            return {t: 0.0 for t in ATTRIB_TERMS}
        return mover_pov_terms(bd, mover_white)

    tw, tr = terms_after(uci_wrong), terms_after(uci_right)
    if tw is None or tr is None:
        return None, dets
    return {t: tw[t] - tr[t] for t in ATTRIB_TERMS}, dets


def _load(tag, epd, theme):
    path = tag if os.path.exists(tag) else os.path.join(RESULTS_DIR, "movematch_%s.csv" % tag)
    rows = {r["id"]: r for r in csv.DictReader(open(path)) if r["score"] != ""}
    score_maps = {i: sm for (_, sm, _, _, i) in load_sts_epd(epd)}
    misses = []
    for i, r in rows.items():
        if i not in score_maps:
            continue
        if theme and not r["theme"].lower().startswith(theme.lower()):
            continue
        if int(r["score"]) >= int(r["max"]):
            continue
        right = max(score_maps[i].items(), key=lambda kv: kv[1])[0]   # top-scored uci
        misses.append((i, r, right))
    return misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["agg", "dossier"])
    ap.add_argument("tag")
    ap.add_argument("--epd", required=True)
    ap.add_argument("--theme", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    misses = _load(args.tag, args.epd, args.theme)
    if args.limit:
        misses = misses[:args.limit]

    if args.mode == "agg":
        sums = {t: 0.0 for t in ATTRIB_TERMS}
        absums = {t: 0.0 for t in ATTRIB_TERMS}
        n = 0
        for i, r, right in misses:
            deltas, _ = attribute(ai, r["fen"], r["engine"], right)
            if deltas is None:
                continue
            n += 1
            for t in ATTRIB_TERMS:
                sums[t] += deltas[t]
                absums[t] += abs(deltas[t])
        print("failure attribution  tag=%s theme=%s  n=%d misses" % (args.tag, args.theme or "ALL", n))
        print("  term            mean_signed   mean_|delta|   (signed>0 => term over-credits the WRONG move)")
        for t in sorted(ATTRIB_TERMS, key=lambda k: absums[k], reverse=True):
            print("    %-20s %+8.3f     %8.3f" % (t, sums[t] / n if n else 0, absums[t] / n if n else 0))
        return

    for i, r, right in misses:
        deltas, dets = attribute(ai, r["fen"], r["engine"], right)
        if deltas is None:
            continue
        top = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        print(json.dumps({
            "id": i, "theme": r["theme"], "fen": r["fen"],
            "played": r["engine"], "correct": right,
            "suspect_terms": [{"term": t, "delta_pawns": round(v, 3)} for t, v in top],
            "detectors": {k: dets[k] for k in DET_KEYS},
        }))


if __name__ == "__main__":
    main()
