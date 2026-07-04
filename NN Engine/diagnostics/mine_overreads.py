#!/usr/bin/env python3
"""Mine the SF-annotated self-play games for the COLLAPSE signature and build the over-read bench.
Core-free: pure JSONL parsing, no engine, no games (each annotated move stores our eval_white_pov AND sf_cp).
Signature = opposite-sign disagreement (we call one side winning, SF the other) in a MIDDLEGAME (>=14 pieces)
= the decision-flipping holes that lose games (distinct from benign endgame magnitude inflation).
Writes diagnostics/overread_bench.csv. Units: our eval_white_pov = milli-pawns (/1000); sf_cp = centipawns (/100).
"""
import json, glob, csv, os, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(BASE + "/selfplay/games/*/game_*/game.annotated.jsonl"))

recs = []
n_files = n_both = 0
for f in files:
    n_files += 1
    tag = f.split("/games/")[1].split("/")[0]
    try:
        lines = open(f).readlines()
    except Exception:
        continue
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") != "move" or r.get("opening") or r.get("booked"):
            continue
        if r.get("sf_cp") is None or r.get("eval_white_pov") is None:
            continue
        op, sp = r["eval_white_pov"] / 1000.0, r["sf_cp"] / 100.0
        if abs(op) > 25 or abs(sp) > 25:   # drop mate-ish / decided
            continue
        n_both += 1
        recs.append((tag, r.get("ply"), r.get("fen"), op, sp, op - sp, r.get("depth")))


def piece_count(fen):
    return sum(1 for c in fen.split()[0] if c.isalpha())


def is_signflip(op, sp):
    return (op >= 0.5 and sp <= -0.75) or (op <= -0.5 and sp >= 0.75)


flips = [r for r in recs if is_signflip(r[3], r[4])]
mid = sorted([r for r in flips if piece_count(r[2]) >= 14], key=lambda r: -abs(r[5]))
print("annotated files=%d  moves w/ both evals=%d" % (n_files, n_both))
print("opposite-sign disagreements: %d total | middlegame(>=14 pcs): %d" % (len(flips), len(mid)))
print("  middlegame -- over-value White: %d  over-value Black: %d"
      % (sum(1 for r in mid if r[3] > 0), sum(1 for r in mid if r[3] < 0)))

out = BASE + "/diagnostics/overread_bench.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tag", "ply", "pieces", "our_pawns", "sf_pawns", "gap_pawns", "our_depth", "fen"])
    for t, ply, fen, op, sp, gap, d in mid:
        w.writerow([t, ply, piece_count(fen), "%.3f" % op, "%.3f" % sp, "%.3f" % gap, d, fen])
print("wrote %d middlegame collapse-bench positions -> %s" % (len(mid), out))
