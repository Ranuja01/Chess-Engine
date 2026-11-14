#!/bin/bash
# 2026-11-14 ~13:00, queue #239: the owner's lost-nodes question for H2 (11-13: "whether its node reductions are losing
# important nodes"), step 2 of the plan in memory: a fixed-depth move-difference study. Light work for the owner's gaming
# (~14:00-17:00): fixed depth, JOBS=2, no timing. Starts when queue #238 reaches its wait for q238.go (its screens and builds
# done, the production module restored); stops before #238's part B (never runs while q238.go exists).
#   _ks_footprint_regret.py, base = today's root (V2_PRESET=shipped), arms riding one base pass:
#     H2      A4 + PLY1_NODE_PRUNE=3 (the games' arm)
#     A4      pre-search and razor off, tiers + root table, NO ply-1 cuts (splits "the cuts lose nodes" from "no pre-search")
#     neutral ASPIRATION_DELTA=300 (the measured-null arm the tool asks for)
#   at depth 7 (the ruler's validated depth; the null bands below are at d7) on game_regret_set (15k) then game_regret_set_v2
#   (11,940, the cross-set). Read win% of changed moves against the measured null bands 49.8-50.4 (set 1) / 50.7 (v2) and the
#   neutral arm of this run; per-position dumps in /mnt/e/chess_data/r8a/ for step 3 (classify H2's changed moves).
#   REGISTERED: if H2's changed-move win% sits below the neutral arm by >= 2pp on BOTH sets, H2 picks worse moves where it
#   differs; if A4 reads the same, the cause is the missing pre-search, not the ply-1 cuts (then guards on the cuts cannot fix
#   it). Within 2pp of neutral on either set: no resolvable move-quality loss at d7.
# Stop: touch /mnt/e/chess_data/q239.stop (between sets).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r8a
STOP=/mnt/e/chess_data/q239.stop
GO=/mnt/e/chess_data/q238.go
A4E="ENABLE_ROOT_PRESEARCH=0 ENABLE_ROOT_RAZOR=0 ENABLE_PRESEARCH_OFF_SORT=1 PRESEARCH_OFF_SORT_MODE=2 ENABLE_ROOT_TABLE=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
echo "[q239] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #238 to reach its go wait"
until grep -qE 'waiting for q238.go|QUEUE 238 DONE|☠' /mnt/e/chess_data/q238.log; do [ -e "$STOP" ] && exit 0; sleep 30; done
grep -qE '☠' /mnt/e/chess_data/q238.log && { echo "[q239] #238 aborted — check the production module before running; QUEUE 239 ABORTED"; exit 1; }
f=$(bash "$R" wac r8a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q239] module (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q239] ☠️ the module is not production — QUEUE 239 ABORTED"; exit 1; }
for s in game_regret_set game_regret_set_v2; do
	[ -e "$STOP" ] && { echo "[q239] stop file — QUEUE 239 DONE"; exit 0; }
	[ -e "$GO" ] && { echo "[q239] q238.go present (timed games next) — QUEUE 239 DONE before $s"; exit 0; }
	echo "[q239] $(date '+%H:%M') == $s"
	env SET=ks_sets/$s.csv DEPTH=7 JOBS=2 BY_PHASE=0 BY_PS=0 BASE_KNOBS="V2_PRESET=shipped" \
		CAND_KNOBS="$A4E PLY1_NODE_PRUNE=3;$A4E;ASPIRATION_DELTA=300" CAND_NAME="H2;A4;neutral" DUMP=$W/fp_$s.csv \
		nice -n 10 bash "$R" pyrun diagnostics/_ks_footprint_regret.py > "$W/fp_$s.out" 2>&1
	grep -vE '^\s*$' "$W/fp_$s.out" | grep -iE 'arm|changed|win%|mean|crit|H2|A4|neutral' | head -40 | sed 's/^/[q239]   /'
done
echo "[q239] $(date) QUEUE 239 DONE"
