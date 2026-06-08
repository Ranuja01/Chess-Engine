# -*- coding: utf-8 -*-
"""Aggregate self-play tournament telemetry into a per-arm summary + (if annotated) an eval-calibration report.

Two independent readouts, both from the recorded files (no engine / no Stockfish needed):

  1) NULL-MOVE / per-arm telemetry  -- from game.jsonl (always available):
     groups non-opening move records by `label` (the arm, e.g. nmp_off / nmp_on) and reports mean depth,
     nodes, engine_time, nps. At a fixed preset (equal clock) the A/B read is "does nmp_on reach higher mean
     depth / fewer nodes than nmp_off?".

  2) EVAL CALIBRATION  -- from game.annotated.jsonl (only after annotate.py ran):
     our static eval vs Stockfish's static (NNUE) eval, White-POV pawns, with the per-term breakdown. Reports
     the our_static - SF_static gap (overall / by phase / by who's winning) + mean per-term contributions.

Run (from NN Engine/):
  python selfplay/summary.py --tag nmp_standard
  python selfplay/summary.py --tag nmp_standard --tag nmp_blitz --tag nmp_lightning
"""

import os
import sys
import glob
import json
import argparse
import statistics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(THIS_DIR, "games")

DIV_CAP_P = 15.0  # only measure calibration where both evals are non-decided (pawns)
TERMS = ["pieces", "capture_gains", "passed_pawn_support", "latent_threat", "central",
         "imbalance_white", "imbalance_black", "pair_bonus", "piece_value_boost"]


def _read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _moves(tag, annotated=False):
    """Yield move records across all games of a tag. annotated=True reads game.annotated.jsonl."""
    name = "game.annotated.jsonl" if annotated else "game.jsonl"
    for p in sorted(glob.glob(os.path.join(GAMES_DIR, tag, "game_*", name))):
        for r in _read_jsonl(p):
            if r.get("type") == "move":
                yield r


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def per_arm_telemetry(tag):
    arms = {}
    for r in _moves(tag):
        if r.get("opening"):
            continue
        lbl = r.get("label", "?")
        a = arms.setdefault(lbl, {"depth": [], "nodes": [], "time": [], "nps": []})
        if isinstance(r.get("depth"), (int, float)):
            a["depth"].append(r["depth"])
        if isinstance(r.get("nodes"), (int, float)):
            a["nodes"].append(r["nodes"])
        if isinstance(r.get("engine_time"), (int, float)):
            a["time"].append(r["engine_time"])
        if isinstance(r.get("nps"), (int, float)):
            a["nps"].append(r["nps"])
    if not arms:
        return
    print(f"\n=== {tag} — per-arm telemetry (non-opening moves) ===")
    print(f"  {'arm':10} {'moves':>6} {'mean_depth':>10} {'med_depth':>9} {'mean_nodes':>12} "
          f"{'mean_time_s':>11} {'mean_knps':>9}")
    for lbl in sorted(arms):
        a = arms[lbl]
        n = len(a["depth"])
        md = statistics.median(a["depth"]) if a["depth"] else 0
        print(f"  {lbl:10} {n:>6} {_mean(a['depth']):>10.2f} {md:>9.1f} {_mean(a['nodes']):>12,.0f} "
              f"{_mean(a['time']):>11.2f} {_mean(a['nps']) / 1000:>9.1f}")


def calibration(tag):
    rows = list(_moves(tag, annotated=True))
    if not rows:
        return False
    # gap = our_static - SF_static (White-POV pawns); split by phase and by who's ahead (per SF static).
    buckets = {"all": [], "midgame": [], "endgame": [], "white_winning": [], "black_winning": [], "near_equal": []}
    term_sums = {k: 0.0 for k in TERMS}
    term_n = 0
    for r in rows:
        bd = r.get("eval_breakdown")
        sfs = r.get("sf_static_cp")
        if not (isinstance(bd, dict) and not bd.get("checkmate")):
            continue
        our_sw = -bd.get("total", 0) / 1000.0
        for k in TERMS:
            term_sums[k] += -bd.get(k, 0) / 1000.0
        term_n += 1
        if not isinstance(sfs, (int, float)):
            continue
        sf_sw = sfs / 100.0
        if abs(our_sw) > DIV_CAP_P or abs(sf_sw) > DIV_CAP_P:
            continue
        gap = our_sw - sf_sw
        buckets["all"].append(gap)
        buckets["endgame" if bd.get("is_endgame") else "midgame"].append(gap)
        if sf_sw >= 1.0:
            buckets["white_winning"].append(gap)
        elif sf_sw <= -1.0:
            buckets["black_winning"].append(gap)
        else:
            buckets["near_equal"].append(gap)

    print(f"\n=== {tag} — eval calibration: our_static - SF_static (White-POV pawns; + = we read hotter) ===")
    print(f"  {'bucket':14} {'n':>6} {'mean':>7} {'median':>7}")
    for b in ["all", "midgame", "endgame", "white_winning", "near_equal", "black_winning"]:
        xs = buckets[b]
        if xs:
            print(f"  {b:14} {len(xs):>6} {_mean(xs):>7.2f} {statistics.median(xs):>7.2f}")
    if term_n:
        print(f"  mean term contributions (White-POV pawns, + favours White), over {term_n} positions:")
        for k, v in sorted(term_sums.items(), key=lambda kv: abs(kv[1]), reverse=True):
            print(f"      {k:22} {v / term_n:+7.3f}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Per-arm telemetry + eval-calibration summary for self-play tags.")
    ap.add_argument("--tag", action="append", required=True, help="tournament tag (repeatable)")
    args = ap.parse_args()
    for tag in args.tag:
        if not os.path.isdir(os.path.join(GAMES_DIR, tag)):
            print(f"[summary] no such tag: {tag}")
            continue
        per_arm_telemetry(tag)
        if not calibration(tag):
            print(f"  (no annotated data for {tag} yet — run: python selfplay/annotate.py --tag {tag})")


if __name__ == "__main__":
    main()
