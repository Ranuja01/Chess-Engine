#!/bin/bash
# 2026-11-18 ~19:45, queue #277: size the 11-18 sweep's confirmed findings with count-only counters (the [sweep2] stats line;
# dev_notes/active/SEARCH-DEFECT-REGISTER-2026-11-18.md §5). Runs beside #274's node-limited SF18 legs (a build and
# fixed-depth benches change no result there); never beside timed games.
#  1 plain UCI test build chess_engine_uci_cnt18c from the current sources. GATES: default 246 / 35,138,434; seed 254 /
#    35,922,538; v1 250 / 35,310,775-778 (the counters must not move a node).
#  2 [sweep2] sums over the WAC bench (300, d10), game12 d10 (12 game positions) and eg60 d10 (60 endgames: 30 pawn / one-minor
#    from selfplay/eg_balanced_fens.txt, 30 from diagnostics/ks_sets/control_eg.txt), default and with the seed.
# REGISTERED before the run ("material" = fires on >= 1% of its own opportunities on game12 or eg60):
#  S2-1 checks dropped by the SEE filter: prediction 40-80% of checks; of the drops, unattacked checkers 60-90%.
#  S2-2 static horizons: eg60 20-60% of horizon nodes; WAC / game12 under 5% (middlegame roots).
#  S2-3 fail-low exits with a futility gap: 30-80% of exits that pruned a move; mean gap 300-1,500 mp.
#  S2-4 pre-search EXACT mislabels: 0.1-3% of pre_minimizer stores.
#  S2-9 lazy returns: 30-70% of RFP returns; lazy q-root stores 5-30% of q-cache stores.
#  S2-6 move-0 fail-highs: 5-20% of root passes.
#  S2-5 decay_stale: WAC searches after the first endgame root in file order (> 50% of the 300 if one comes early).
# Stop: touch /mnt/e/chess_data/q277.stop. Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/q277
G=/mnt/e/chess_data/sv2_speed/game12.txt
E=/mnt/e/chess_data/sv2_speed/eg60.txt
NEW="$ND/build_uci/chess_engine_uci_cnt18c"
KS="ENABLE_QSTANDPAT_SEED=1"
STOP=/mnt/e/chess_data/q277.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q277]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q277] $(date) stop file before $1 — QUEUE 277 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
sw() {  # binary, knobs, bench command → the summed [sweep2] counters with the derived rates
	local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"
	printf 'setoption name Debug value true\n%s\n%s\nquit\n' "$k" "$3" | "$1" 2>&1 | grep -E '^\[sweep2\]' \
		| awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); s[kv[1]] += kv[2] } n++ }
		       function r(a, b) { return b > 0 ? sprintf("%.2f%%", 100 * a / b) : "-" }
		       END {
		         printf "searches %d · S2-1 checks %d, dropped %d (%s), unattacked %d (%s of drops)", n, s["cx_checks"], s["cx_drop"], r(s["cx_drop"], s["cx_checks"]), s["cx_drop_noatt"], r(s["cx_drop_noatt"], s["cx_drop"])
		         printf " · S2-2 horizons %d, static %d (%s), in check %d, in null %d, qprec searches %d", s["hz_total"], s["hz_static"], r(s["hz_static"], s["hz_total"]), s["hz_static_check"], s["hz_static_null"], s["qprec"]
		         printf " · S2-3 exits %d, gap %d (%s), mean gap %.0f mp", s["fut_exit"], s["fut_gap"], r(s["fut_gap"], s["fut_exit"]), s["fut_gap"] ? s["fut_gap_sum"] / s["fut_gap"] : 0
		         printf " · S2-4 pre-search stores %d, EXACT mislabels %d (%s)", s["ps_store"], s["ps_exact_mislabel"], r(s["ps_exact_mislabel"], s["ps_store"])
		         printf " · S2-9 RFP returns %d, lazy %d (%s); q-cache stores %d, lazy %d (%s)", s["rfp_ret"], s["rfp_lazy_ret"], r(s["rfp_lazy_ret"], s["rfp_ret"]), s["qstore"], s["lazy_qstore"], r(s["lazy_qstore"], s["qstore"])
		         printf " · S2-6 root passes %d, move-0 fail-highs %d (%s)", s["root_pass"], s["root_m0_fh"], r(s["root_m0_fh"], s["root_pass"])
		         printf " · S2-5 decay_stale %d · S2-24 mate-range ends %d\n", s["decay_stale"], s["id_mate_stop"] }'
}
echo "[q277] $(date '+%Y-%m-%d %H:%M:%S') start"
pgrep -f "[b]uild_uci.sh|[s]ts_test|[v]algrind" >/dev/null && { echo "[q277] ☠️ another build or diagnostic is running — QUEUE 277 ABORTED"; exit 1; }
grep -q 'SF18 legs (node-limited)' /mnt/e/chess_data/q274.log && ! grep -qE 'self-play seed: 2000 finished|QUEUE 274|Mediocre' <(sed -n '/SF18 legs (node-limited)/,$p' /mnt/e/chess_data/q274.log) \
	|| { echo "[q277] ☠️ #274 is not in its node-limited SF18 legs — QUEUE 277 ABORTED (not run beside timed games)"; exit 1; }
[ -e /mnt/e/chess_data/q274.go ] && { echo "[q277] ☠️ q274.go is set (timed work may resume) — QUEUE 277 ABORTED"; exit 1; }

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_cnt18c nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 \
	|| { echo "[q277] ☠️ UCI build failed — QUEUE 277 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q277] $(date '+%H:%M') build done — the sources are free to edit"
a=$(fp "$NEW" ""); sd=$(fp "$NEW" "$KS")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q277] cnt18c default [$a] (246 / 35138434) · seed [$sd] (254 / 35922538) · v1 [$v1] (250 / 35310775-778)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$sd" = "solved 254/300 nodes 35922538" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' \
	|| { echo "[q277] ☠️ a gate failed — QUEUE 277 ABORTED"; exit 1; }
echo "[q277] ✅ every gate equal"

# ── 2 counts ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
for arm in default seed; do
	k=""; [ $arm = seed ] && k="$KS"
	stopped "the $arm counts"
	echo "[q277] WAC d10 $arm:    $(sw "$NEW" "$k" bench)"
	echo "[q277] game12 d10 $arm: $(sw "$NEW" "$k" "bench $G 10")"
	echo "[q277] eg60 d10 $arm:   $(sw "$NEW" "$k" "bench $E 10")"
done
echo "[q277] registered: material = fires on >= 1% of its own opportunities on game12 or eg60; predictions in the header"
echo "[q277] $(date) QUEUE 277 DONE"
