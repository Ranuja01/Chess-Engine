#!/bin/bash
# 2026-10-07 (owner: "a quick hour of equal time" vs Mediocre — for fun / a rough absolute anchor; memory
# later-test-vs-mediocre-for-an-absolute-rating). v2 shipped at PRESET=LIGHTNING (TIME_LIMIT 1.0 s, no new iteration after
# 0.75 s) vs Mediocre v0.5 (Java) at movetime 1.0 s — ≈ equal time, slightly favouring Mediocre. Raw-pipe UCI driver
# (Mediocre re-emits id/uciok). SF18 is the draw ARBITER only. UHO openings, 50 games, 2 at a time (4 engine processes).
# Reference: CCRL 40/40 lists Mediocre 0.4 = 2274; v0.5 is "noticeably stronger" (author) — unverified, ~2300-2375.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[med] $(date) start"
bash "$R" pyrun selfplay/vs_sf.py --our-label v2 --our-config "V2_PRESET=shipped MAX_DEPTH=64 USE_OPENING_BOOK=0" \
  --preset LIGHTNING --sf-path "$ND/selfplay/mediocre_uci.sh" --opponent-raw --sf-elo 0 --sf-movetime 1.0 \
  --sf-arb-path "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2" \
  --games 50 --concurrency 2 --openings selfplay/openings_uho.txt --seed 91 --tag mediocre_v05_1s --adjudicate-draw --quiet
echo "[med] $(date) DONE"
