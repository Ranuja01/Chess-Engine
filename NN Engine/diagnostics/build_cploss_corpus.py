# -*- coding: utf-8 -*-
"""Freeze a STRATIFIED position corpus for the WDL-cploss compass (holistic co-tune campaign).

Assembles one static FEN list, each row tagged by stratum, from four sources (Fable breadth discipline:
themed + neutral-book + our-vs-SF11 + collapse-tail, NOT only our own games). Deterministic (fixed seed +
sorted iteration) so the freeze is reproducible. Held-out shard is carved by the compass at scoring time
(even/odd index), not here.

    python diagnostics/build_cploss_corpus.py [out.csv] [--sts N] [--neutral N] [--games N] [--seed S]

Strata: sts | neutral | game | collapse. Output columns: fen,stratum.
"""
import argparse
import csv
import glob
import json
import os
import random

import chess

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
SUITES = os.path.join(THIS, "suites")


def pcount(fen):
    return sum(1 for ch in fen.split()[0] if ch.isalpha())


def sts_fens():
    """Every STS position as a legal FEN (4-field EPD + ' 0 1'), tagged nothing (theme in id ignored for v1)."""
    path = os.path.join(SUITES, "STS1-STS15_LAN_v3.epd")
    out = []
    for line in open(path):
        toks = line.split()
        if len(toks) < 4:
            continue
        fen = " ".join(toks[:4]) + " 0 1"
        try:
            chess.Board(fen)
        except Exception:
            continue
        out.append(fen)
    return out


def neutral_fens():
    """Play each UHO opening line out to its final (≈12-ply) FEN — SF-verified balanced openings = de-biased."""
    path = os.path.join(ENGINE, "selfplay", "openings_uho.txt")
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        b = chess.Board()
        ok = True
        for mv in line.split():
            try:
                b.push_uci(mv)
            except Exception:
                ok = False
                break
        if ok and not b.is_game_over():
            out.append(b.fen())
    return out


def game_fens(tag="sf11_evalgap_d8"):
    """General midgame positions our engine actually reached vs SF11 (biased-but-realistic stratum)."""
    gdir = os.path.join(ENGINE, "selfplay", "games", tag)
    out = []
    for jf in sorted(glob.glob(os.path.join(gdir, "game_*.jsonl"))):
        for ln in open(jf):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            f = o.get("fen")
            if f and 12 <= pcount(f) <= 30:
                out.append(f)
    return out


def collapse_fens(tag="sf11_evalgap_d8", eval_only=True):
    """The decision FENs where our eval over-read (peaked winning then lost). EVAL-classified = the gold targets."""
    tpath = os.path.join(ENGINE, "selfplay", "games", tag, "triage.csv")
    out = []
    if os.path.exists(tpath):
        for r in csv.DictReader(open(tpath)):
            if eval_only and r.get("class", "").strip() != "EVAL":
                continue
            f = r.get("decision_fen", "").strip()
            if f:
                out.append(f)
        return out
    # fallback: raw collapses.csv (all 35, unclassified)
    cpath = os.path.join(ENGINE, "selfplay", "games", tag, "collapses.csv")
    for r in csv.DictReader(open(cpath)):
        f = r.get("decision_fen", "").strip()
        if f:
            out.append(f)
    return out


def sample(pool, n, rng):
    """Deterministic de-dup + downsample to n."""
    seen, uniq = set(), []
    for f in pool:
        key = " ".join(f.split()[:4])          # ignore move clocks for de-dup
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    if len(uniq) <= n:
        return uniq
    idx = sorted(rng.sample(range(len(uniq)), n))
    return [uniq[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(ENGINE, "selfplay", "tune_data", "cploss_corpus.csv"))
    ap.add_argument("--sts", type=int, default=350)
    ap.add_argument("--neutral", type=int, default=200)
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--game-tag", default="sf11_evalgap_d8")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    strata = [
        ("sts", sample(sts_fens(), args.sts, rng)),
        ("neutral", sample(neutral_fens(), args.neutral, rng)),
        ("game", sample(game_fens(args.game_tag), args.games, rng)),
        ("collapse", collapse_fens(args.game_tag, eval_only=True)),   # all EVAL collapses, no downsample
    ]
    # cross-stratum de-dup (collapse/game can overlap): first stratum wins, collapse kept last so it always survives
    seen, rows = set(), []
    order = ["collapse", "sts", "neutral", "game"]                    # collapse highest priority
    by = dict(strata)
    for st in order:
        for f in by[st]:
            key = " ".join(f.split()[:4])
            if key in seen:
                continue
            seen.add(key)
            rows.append((f, st))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "stratum"])
        w.writerows(rows)
    counts = {}
    for _, st in rows:
        counts[st] = counts.get(st, 0) + 1
    print("froze %d positions -> %s" % (len(rows), args.out))
    print("  strata: " + "  ".join("%s=%d" % (k, counts.get(k, 0)) for k in order))


if __name__ == "__main__":
    main()
