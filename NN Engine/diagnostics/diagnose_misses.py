# -*- coding: utf-8 -*-
"""Detector-dump loop — name the board feature that separates the engine's HITS from its MISSES.

This is the structure-discovery half of the funnel: chess picks a conditioner's structure/direction, but
WHICH detector to condition on is found, not guessed. movematch.py writes a per-position scorecard
(theme, engine move, score, max, fen) for each tag. This reads that scorecard and, for every position the
engine got wrong, dumps the cheap LIVE detectors (EvalBreakdown det_* via ChessAI.ev_breakdown) oriented
to the side to move. The detector whose HIT-vs-MISS means diverge most is the candidate conditioner (or,
if nothing separates them, a MISSING detector to add).

Two modes:
  single (one tag)      : split that tag's positions into HIT (score==max) / MISS (score<max) and print,
                          per detector, the mean over each class and the gap. Big gap => that detector
                          predicts failure => condition the over/under-reading term on it.
  diff (base + cand)    : classify shared positions as HELPED / HURT / SAME by score delta, and print the
                          detector means per class. The helped-vs-hurt gap is the EFFICACY signal (search-
                          propagated), the read that decides whether a built conditioner aimed true.

Run in WSL from NN Engine/ (same interpreter that built ChessAI):
    python diagnostics/diagnose_misses.py single mm100base [--theme "Knight Outposts"] [--dump 12]
    python diagnostics/diagnose_misses.py diff   mm100base mm100cand [--theme ...] [--dump 12]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import csv
import argparse

import chess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
SELFPLAY_DIR = os.path.join(ENGINE_DIR, "selfplay")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SELFPLAY_DIR)

# Detectors derived from the raw EvalBreakdown det_* fields, oriented to the SIDE TO MOVE (the side the
# move-match scores), since that is the perspective the engine is choosing from. "edge" = stm - opponent.
def detectors(bd, stm_white):
    def edge(w, b):
        return (bd[w] - bd[b]) if stm_white else (bd[b] - bd[w])
    own_ks = bd["det_ks_units_w"] if stm_white else bd["det_ks_units_b"]
    opp_ks = bd["det_ks_units_b"] if stm_white else bd["det_ks_units_w"]
    offense = edge("det_w_offense", "det_b_offense")
    # Openness surrogate, same form the eval already uses for MOD_PAIR_OPEN (12 - popcount(pawns)): higher =
    # fewer pawns = more open. "overextension" is the multiplicative coupling offense_edge x openness — an
    # offensive edge is only over-credited (you can't really extend that far) when the position is OPEN.
    openness = 12 - bd["det_pawn_count"]
    return {
        "offense_edge":  offense,
        "defense_edge":  edge("det_w_defense", "det_b_defense"),
        "control_edge":  edge("det_w_mobility", "det_b_mobility"),
        "pieceval_edge": edge("det_w_pieceval", "det_b_pieceval"),
        "own_king_danger": own_ks,
        "opp_king_danger": opp_ks,
        "openness":    openness,
        "overextend":  offense * openness,   # multiplicative interaction: aggressive AND open
        "central":     bd["det_central"] if stm_white else -bd["det_central"],
        "pawn_count":  bd["det_pawn_count"],
        "phase_score": bd["phase_score"],
    }


DET_KEYS = ["offense_edge", "defense_edge", "control_edge", "pieceval_edge",
            "own_king_danger", "opp_king_danger", "openness", "overextend",
            "central", "pawn_count", "phase_score"]


def _load_csv(tag):
    path = tag if os.path.exists(tag) else os.path.join(RESULTS_DIR, "movematch_%s.csv" % tag)
    with open(path) as f:
        return list(csv.DictReader(f))


def _playable(rows, theme):
    out = []
    for r in rows:
        if r["score"] == "":              # book hit — no scored decision
            continue
        if theme and not r["theme"].lower().startswith(theme.lower()):
            continue
        out.append(r)
    return out


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _dump_classes(classes, dets_by_id, dump):
    """classes: dict name -> list of row ids. dets_by_id: id -> det dict. Print per-detector class means."""
    print("\n%-16s %s" % ("detector", "  ".join("%12s(n=%d)" % (n, len(ids)) for n, ids in classes.items())))
    for k in DET_KEYS:
        cells = []
        for _, ids in classes.items():
            cells.append("%16.1f" % _mean([dets_by_id[i][k] for i in ids if i in dets_by_id]))
        print("  %-14s %s" % (k, "".join(cells)))
    names = list(classes)
    if len(names) >= 2:
        a, b = names[0], names[1]
        print("\n  gap (%s - %s), sorted by |gap|:" % (a, b))
        gaps = []
        for k in DET_KEYS:
            ma = _mean([dets_by_id[i][k] for i in classes[a] if i in dets_by_id])
            mb = _mean([dets_by_id[i][k] for i in classes[b] if i in dets_by_id])
            gaps.append((k, ma - mb))
        for k, g in sorted(gaps, key=lambda kv: abs(kv[1]), reverse=True):
            print("      %-16s %+10.1f" % (k, g))


def main():
    ap = argparse.ArgumentParser(description="Detector-dump loop over a movematch scorecard.")
    ap.add_argument("mode", choices=["single", "diff"])
    ap.add_argument("tags", nargs="+", help="single: <tag> ; diff: <base> <cand>")
    ap.add_argument("--theme", default=None, help="restrict to theme prefix")
    ap.add_argument("--dump", type=int, default=0, help="print this many example MISS/HURT fens")
    args = ap.parse_args()

    from ChessAI import ChessAI
    seed = chess.Board()
    ai = ChessAI(None, None, seed, seed.turn)

    def dets_for(rows):
        out = {}
        for r in rows:
            board = chess.Board(r["fen"])
            bd = ai.ev_breakdown(board)
            if bd.get("checkmate"):
                continue
            out[r["id"]] = detectors(bd, board.turn == chess.WHITE)
        return out

    if args.mode == "single":
        rows = _playable(_load_csv(args.tags[0]), args.theme)
        hit = [r for r in rows if int(r["score"]) == int(r["max"])]
        miss = [r for r in rows if int(r["score"]) < int(r["max"])]
        dets = dets_for(rows)
        print("single tag=%s theme=%s  HIT=%d MISS=%d" % (args.tags[0], args.theme or "ALL", len(hit), len(miss)))
        _dump_classes({"HIT": [r["id"] for r in hit], "MISS": [r["id"] for r in miss]}, dets, args.dump)
        for r in miss[:args.dump]:
            print("  MISS %-22s played %-6s (%s/%s)  %s" % (r["theme"], r["engine"], r["score"], r["max"], r["fen"]))
        return

    base = {r["id"]: r for r in _playable(_load_csv(args.tags[0]), args.theme)}
    cand = {r["id"]: r for r in _playable(_load_csv(args.tags[1]), args.theme)}
    ids = [i for i in base if i in cand]
    helped = [i for i in ids if int(cand[i]["score"]) > int(base[i]["score"])]
    hurt = [i for i in ids if int(cand[i]["score"]) < int(base[i]["score"])]
    same = [i for i in ids if int(cand[i]["score"]) == int(base[i]["score"])]
    dets = dets_for([base[i] for i in ids])
    print("diff base=%s cand=%s theme=%s  HELPED=%d HURT=%d SAME=%d"
          % (args.tags[0], args.tags[1], args.theme or "ALL", len(helped), len(hurt), len(same)))
    _dump_classes({"HELPED": helped, "HURT": hurt, "SAME": same}, dets, args.dump)
    for i in hurt[:args.dump]:
        b = base[i]
        print("  HURT %-22s %s->%s  %s" % (b["theme"], b["engine"], cand[i]["engine"], b["fen"]))


if __name__ == "__main__":
    main()
