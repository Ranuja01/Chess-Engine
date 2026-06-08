# -*- coding: utf-8 -*-
"""Deep A/B diagnostics over a recorded tournament — the cuts tournament_diag.py / summary.py don't do.

Joins each game's raw telemetry (game.jsonl: depth, nodes, eval_white_pov, fen, ordering, cache,
aspiration) with its SF annotation (game.annotated.jsonl: sf_cp search eval, sf_static_cp NNUE static,
sf_depth, eval_breakdown) by ply, and reports, per arm (engine config label):

  1) WIN%-SCATTER vs NNUE  — both our static and our search eval converted to lichess win% and compared
     to SF's (static→NNUE, search→SF search). MAE + sample SD of the win% difference, overall / near-
     equal / by phase. This is the "~11% scatter" metric (near-equal is the one that matters).
  2) DEPTH & NODES by 5 game-phase buckets (by piece count) per arm + vs SF depth.
  3) CONVERSION / RESILIENCE  — per game, the max and min SF eval (that arm's POV) reached after the
     opening vs the actual result: "blew a winning position" (peaked >= +Tcp, didn't win) and
     "saved a lost one" (troughed <= -Tcp, didn't lose). hlmr vs base.
  4) MOVE-ORDERING & SEARCH tells — first-move-ordering-hit rate (played move == top-ordered),
     TT hit-rate, aspiration fail rate, per arm.

Pure data analysis (no engine / no Stockfish). Run from NN Engine/:
    python selfplay/deep_diag.py --tag hlmr_more1_lightning
    python selfplay/deep_diag.py --tag hlmr_more1_lightning hlmr_more1_blitz
"""

import os
import sys
import csv
import glob
import json
import math
import argparse
import statistics as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(THIS_DIR, "games")

CAL_CAP_CP = 1500          # skip calibration where |either eval| exceeds this (decided/mate)
CONV_T = 300               # |cp| threshold for "winning"/"losing" in conversion analysis (3 pawns)
WIN_K = 0.00368208         # lichess win% sigmoid constant (per cp)


def winpct(cp):
    cp = max(-CAL_CAP_CP, min(CAL_CAP_CP, cp))
    return 100.0 / (1.0 + math.exp(-WIN_K * cp))


def piece_count(fen):
    return sum(1 for c in fen.split()[0] if c.isalpha())


def phase_of(pc):
    if pc >= 28: return "1.opening   "
    if pc >= 22: return "2.early-mid "
    if pc >= 17: return "3.late-mid  "
    if pc >= 12: return "4.early-end "
    return "5.late-end  "


def our_static_cp(eb):
    # eval_breakdown total is ABSOLUTE Black-positive milli-pawns; White-POV cp = -total/10.
    return -eb["total"] / 10.0


def load_game(gdir):
    """Return (meta, [per-ply dict]) joining raw + annotated by ply; None if unreadable."""
    raw_p = os.path.join(gdir, "game.jsonl")
    ann_p = os.path.join(gdir, "game.annotated.jsonl")
    if not os.path.exists(raw_p):
        return None
    raw = {}
    for line in open(raw_p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("type") == "move":
            raw[r["ply"]] = r
    ann = {}
    if os.path.exists(ann_p):
        for line in open(ann_p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "move":
                ann[r["ply"]] = r
    moves = []
    for ply in sorted(raw):
        m = dict(raw[ply])
        if ply in ann:
            m.update({k: ann[ply][k] for k in ("sf_cp", "sf_static_cp", "sf_depth",
                                               "sf_best", "eval_breakdown") if k in ann[ply]})
        moves.append(m)
    return moves


def analyze(tag):
    tdir = os.path.join(GAMES_DIR, tag)
    # per-game color/result for the conversion pass (summary.csv: p1 == first --p1-label arm)
    g_meta = {}
    sp = os.path.join(tdir, "summary.csv")
    p1_label = None
    if os.path.exists(sp):
        for row in csv.DictReader(open(sp)):
            g_meta[int(row["game"])] = row
            p1_label = row["white"] if row["p1_color"] == "white" else row["black"]

    # accumulators
    depth = {}          # arm -> phase -> [depths]
    nodes = {}          # arm -> phase -> [nodes]
    sd_search = {}      # arm -> bucket -> [win% diff]  (our search vs SF search)
    sd_static = {}      # arm -> bucket -> [win% diff]  (our static vs SF NNUE static)
    order_hit = {}      # arm -> [0/1 first-move-ordering hit]
    tt = {}             # arm -> [hit, probe]
    asp = {}            # arm -> [fails, windows]
    sf_depths = []
    conv = {}           # arm -> {"blew":0,"won_winning":0,"saved":0,"lost_losing":0,"games":0}

    def buck(d, *ks):
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d.setdefault(ks[-1], [])
        return d[ks[-1]]

    for gdir in sorted(glob.glob(os.path.join(tdir, "game_*"))):
        moves = load_game(gdir)
        if not moves:
            continue
        gidx = int(os.path.basename(gdir).split("_")[1])
        meta = g_meta.get(gidx)
        for m in moves:
            arm = m.get("label")
            if not arm or arm == "opening" or m.get("booked"):
                continue
            ph = phase_of(piece_count(m["fen"]))
            if m.get("depth") is not None:
                buck(depth, arm, ph).append(m["depth"])
            if m.get("nodes") is not None:
                buck(nodes, arm, ph).append(m["nodes"])
            # ordering hit: was the played move the top-ordered one?
            od = m.get("ordering")
            if od:
                top = min(od, key=lambda e: e["i"])
                order_hit.setdefault(arm, []).append(1 if top.get("uci") == m.get("uci") else 0)
            c = m.get("cache") or {}
            if c.get("tt_probes"):
                tt.setdefault(arm, [0, 0]); tt[arm][0] += c.get("tt_hits", 0); tt[arm][1] += c["tt_probes"]
            a = m.get("aspiration") or {}
            if a.get("windows"):
                asp.setdefault(arm, [0, 0]); asp[arm][0] += a.get("fails", 0); asp[arm][1] += a["windows"]
            # calibration (skip decided / missing SF)
            if m.get("sf_cp") is not None and m.get("eval_white_pov") is not None:
                our = m["eval_white_pov"] / 10.0; sf = m["sf_cp"]
                if abs(our) <= CAL_CAP_CP and abs(sf) <= CAL_CAP_CP:
                    diff = winpct(our) - winpct(sf)
                    buck(sd_search, arm, "all").append(diff)
                    if abs(sf) < 100: buck(sd_search, arm, "near-equal").append(diff)
                    buck(sd_search, arm, phase_of(piece_count(m["fen"]))).append(diff)
            if m.get("sf_static_cp") is not None and m.get("eval_breakdown") is not None:
                our = our_static_cp(m["eval_breakdown"]); sf = m["sf_static_cp"]
                if abs(our) <= CAL_CAP_CP and abs(sf) <= CAL_CAP_CP:
                    diff = winpct(our) - winpct(sf)
                    buck(sd_static, arm, "all").append(diff)
                    if abs(sf) < 100: buck(sd_static, arm, "near-equal").append(diff)
                    buck(sd_static, arm, phase_of(piece_count(m["fen"]))).append(diff)
            if m.get("sf_depth") is not None:
                sf_depths.append(m["sf_depth"])

        # conversion / resilience (needs SF evals + this game's arm colors + result)
        if meta:
            for arm, color in ((meta["white"], "white"), (meta["black"], "black")):
                series = [(m["sf_cp"] if color == "white" else -m["sf_cp"])
                          for m in moves if m.get("sf_cp") is not None and not m.get("opening") and not m.get("booked")
                          and abs(m["sf_cp"]) <= 30000]
                if not series:
                    continue
                cd = conv.setdefault(arm, {"blew": 0, "won_winning": 0, "saved": 0, "lost_losing": 0, "games": 0})
                cd["games"] += 1
                res = meta["result"]
                won = (res == "1-0" and color == "white") or (res == "0-1" and color == "black")
                lost = (res == "1-0" and color == "black") or (res == "0-1" and color == "white")
                if max(series) >= CONV_T:
                    cd["won_winning"] += 1 if won else 0
                    cd["blew"] += 0 if won else 1
                if min(series) <= -CONV_T:
                    cd["lost_losing"] += 1 if lost else 0
                    cd["saved"] += 0 if lost else 1

    arms = sorted(depth)
    print("=" * 78); print(f"{tag}  — deep A/B diagnostics  (arms: {', '.join(arms)})"); print("=" * 78)

    print("\n[1] WIN%-SCATTER vs SF  (MAE = mean |our_win% - SF_win%|, SD = sample stdev)")
    for which, store in (("our STATIC vs SF NNUE static", sd_static), ("our SEARCH vs SF search", sd_search)):
        print(f"  -- {which} --")
        for arm in sorted(store):
            row = []
            for b in ("all", "near-equal", "1.opening   ", "2.early-mid ", "3.late-mid  ", "4.early-end ", "5.late-end  "):
                xs = store[arm].get(b)
                row.append(f"{b.strip()}: MAE {st.mean([abs(x) for x in xs]):4.1f} SD {st.pstdev(xs):4.1f} (n{len(xs)})" if xs else f"{b.strip()}: -")
            print(f"     {arm:5} " + " | ".join(row[:2]))
            print(f"           " + " | ".join(row[2:]))

    print("\n[2] DEPTH by phase (mean ply) | NODES by phase (mean)   [SF mean depth %.1f]" % (st.mean(sf_depths) if sf_depths else 0))
    phases = ["1.opening   ", "2.early-mid ", "3.late-mid  ", "4.early-end ", "5.late-end  "]
    hdr = "  arm   " + "".join(f"{p.strip():>12}" for p in phases)
    print(hdr)
    for arm in arms:
        ds = "  %-5s " % arm + "".join(f"{(st.mean(depth[arm][p]) if depth.get(arm,{}).get(p) else 0):>12.1f}" for p in phases)
        print(ds + "   (depth)")
        ns = "        " + "".join(f"{(st.mean(nodes[arm][p])/1000 if nodes.get(arm,{}).get(p) else 0):>12.0f}" for p in phases)
        print(ns + "   (knodes)")

    print("\n[3] CONVERSION / RESILIENCE  (T=%dcp)" % CONV_T)
    print("  arm    games  reached-winning: won / blew   |  reached-losing: lost / saved")
    for arm in sorted(conv):
        c = conv[arm]
        rw = c["won_winning"] + c["blew"]; rl = c["lost_losing"] + c["saved"]
        print(f"  {arm:5}  {c['games']:5}    {rw:3}: {c['won_winning']:3} / {c['blew']:3} "
              f"({100*c['blew']/rw:.0f}% blown)   |   {rl:3}: {c['lost_losing']:3} / {c['saved']:3} ({100*c['saved']/rl:.0f}% saved)"
              if rw and rl else f"  {arm:5}  {c['games']:5}    rw={rw} rl={rl}")

    print("\n[4] ORDERING & SEARCH tells")
    for arm in arms:
        oh = order_hit.get(arm, [])
        tth = tt.get(arm, [0, 1]); aspf = asp.get(arm, [0, 1])
        print(f"  {arm:5}  first-move-ordering-hit {100*st.mean(oh):.1f}% (n{len(oh)})  | "
              f"TT hit {100*tth[0]/max(tth[1],1):.1f}%  | aspiration fail {100*aspf[0]/max(aspf[1],1):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    args = ap.parse_args()
    for t in args.tag:
        analyze(t)


if __name__ == "__main__":
    main()
