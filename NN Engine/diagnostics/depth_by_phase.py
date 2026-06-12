# -*- coding: utf-8 -*-
"""Per-phase mean search depth by tournament arm, from self-play game JSONLs.

For a tournament A/B this attributes each (non-opening) move's reported search depth to the
arm that made it -- via the meta record's per-side config -- buckets by move number, and
reports mean depth per arm per phase plus the delta. This is the honest "speed -> depth at
equal clock" measurement: a matched, per-phase comparison, instead of eyeballing log tails
that happen to be at different game phases (endgames reach ~2x midgame depth regardless of
eval cost, so cross-phase peeks are meaningless).

Run (from NN Engine/):
    python diagnostics/depth_by_phase.py <tag> <on-config-substring>
e.g. python diagnostics/depth_by_phase.py bishop_uho ENABLE_CHEAP_BISHOP_COMPLEX=1
"""
import json
import glob
import os
import sys
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
GAMES_DIR = os.path.join(ENGINE_DIR, "selfplay", "games")


def bucket(move_no):
    if move_no <= 8:
        return None  # opening / book moves (no real search)
    if move_no <= 15:
        return "1_opening-out(9-15)"
    if move_no <= 25:
        return "2_midgame(16-25)"
    if move_no <= 35:
        return "3_late-mid(26-35)"
    if move_no <= 50:
        return "4_early-end(36-50)"
    return "5_late-end(51+)"


def main():
    tag = sys.argv[1]
    on_sub = sys.argv[2]  # substring identifying the ON arm's per-side config
    games = glob.glob(os.path.join(GAMES_DIR, tag, "game_*", "game.jsonl"))
    acc = defaultdict(lambda: [0, 0])  # (arm, bucket) -> [depth_sum, count]
    n_games = 0
    for path in games:
        white_arm = black_arm = None
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "meta":
                    white_arm = "ON" if on_sub in rec.get("config_white", "") else "OFF"
                    black_arm = "ON" if on_sub in rec.get("config_black", "") else "OFF"
                    n_games += 1
                elif rec.get("type") == "move" and white_arm is not None:
                    d = rec.get("depth")
                    if d is None or rec.get("opening"):
                        continue
                    b = bucket(rec.get("move_no") or 0)
                    if b is None:
                        continue
                    arm = white_arm if rec.get("color") == "white" else black_arm
                    acc[(arm, b)][0] += d
                    acc[(arm, b)][1] += 1

    print("tag=%s  on-config='%s'  games=%d" % (tag, on_sub, n_games))
    print("%-22s %8s %8s %8s %9s" % ("phase", "ON", "OFF", "delta", "n(ON)"))
    print("-" * 60)
    all_on = [0, 0]
    all_off = [0, 0]
    for b in sorted(set(bk for (_, bk) in acc)):
        on, off = acc[("ON", b)], acc[("OFF", b)]
        mon = on[0] / on[1] if on[1] else 0.0
        moff = off[0] / off[1] if off[1] else 0.0
        all_on[0] += on[0]; all_on[1] += on[1]
        all_off[0] += off[0]; all_off[1] += off[1]
        print("%-22s %8.3f %8.3f %+8.3f %9d" % (b, mon, moff, mon - moff, on[1]))
    print("-" * 60)
    mon = all_on[0] / all_on[1] if all_on[1] else 0.0
    moff = all_off[0] / all_off[1] if all_off[1] else 0.0
    print("%-22s %8.3f %8.3f %+8.3f %9d" % ("ALL (9+)", mon, moff, mon - moff, all_on[1]))


if __name__ == "__main__":
    main()
