# -*- coding: utf-8 -*-
"""Build REWOUND start FENs for the collapse-replay: for each collapse, take the position N plies BEFORE the
peak (where the game was still ~equal, before we walked into the exposed-king line) — replaying from the peak
itself is useless because the peak is already lost (our +2 was the over-optimism; SF sees -3..-8). Optionally
restrict to the KS-caused collapses (decision_fen in the danger set). Feed the output to vs_sf --start-fens.

Run: pyrun diagnostics/collapse_rewind_fens.py --tags sfelo2400_base200,mediocre_mine [--rewind 12 --ks-only]
"""
import os
import sys
import csv
import json
import glob
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "..", "selfplay", "games")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--rewind", type=int, default=12, help="plies before the peak to start from")
    ap.add_argument("--ks-only", action="store_true", help="restrict to KS-caused (decision_fen in danger.txt)")
    ap.add_argument("--out", default=os.path.join(THIS, "ks_sets", "rewound.txt"))
    args = ap.parse_args()

    ks_set = None
    if args.ks_only:
        ks_set = set()
        for ln in open(os.path.join(THIS, "ks_sets", "danger.txt")):
            if ln.strip():
                ks_set.add(ln.rstrip("\n").split("\t", 1)[-1].strip())

    out = []
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        d = os.path.join(GAMES, tag)
        cp = os.path.join(d, "collapses.csv")
        if not os.path.exists(cp):
            continue
        for r in csv.DictReader(open(cp)):
            if ks_set is not None and (r.get("decision_fen") or "").strip() not in ks_set:
                continue
            try:
                peak_ply = int(r["peak_ply"])
            except (KeyError, ValueError):
                continue
            gid = str(r.get("game"))
            jf = os.path.join(d, "game_%03d.jsonl" % int(gid)) if gid.isdigit() else None
            if not jf or not os.path.exists(jf):
                cands = glob.glob(os.path.join(d, "game_*%s.jsonl" % gid))
                jf = cands[0] if cands else None
            if not jf or not os.path.exists(jf):
                continue
            recs = [json.loads(l) for l in open(jf) if l.strip()]
            by_ply = {rc.get("ply"): rc.get("fen") for rc in recs if rc.get("fen")}
            target = max(1, peak_ply - args.rewind)
            fen = by_ply.get(target)
            while fen is None and target < peak_ply:      # nearest available ply at/after target
                target += 1; fen = by_ply.get(target)
            if fen:
                out.append((fen, "%s/%s peakply=%d start=%d" % (tag, gid, peak_ply, target)))

    seen = set(); uniq = []
    for fen, tagn in out:
        if fen not in seen:
            seen.add(fen); uniq.append((fen, tagn))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for fen, tagn in uniq:
            fh.write("%s\t%s\n" % (tagn, fen))
    print("wrote %d rewound start FENs (rewind=%d, ks_only=%s) -> %s"
          % (len(uniq), args.rewind, args.ks_only, args.out))


if __name__ == "__main__":
    main()
