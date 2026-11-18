#!/bin/bash
# 2026-11-17 ~23:00, queue #268: SORT_KEYED TIME re-read on a quiet box (owner not gaming, asked to leave the PC idle), 15 rounds.
# #267 (same binaries) was DISTURBED: rounds drifted 26 → 29 s, arms ±6% within a round; skon / g76 0.999 (5 of 9), unreadable.
# Reuses #267's PGO build chess_engine_uci_skpgo (no rebuild); its gates are re-checked first.
# Pinned bench ×15 on core 5, three arms rotating: g76 (production), skon (skpgo + SORT_KEYED=1), skoff (skpgo default).
# REGISTERED before the bench (as #267, scaled to 15 rounds): skon / g76 within-round time median < 0.990 AND skon faster in
# ≥ 11 of 15 = a speed-ship candidate (owner's call); prediction −0.5 … −2%. Disturbance check: the spread of g76's own times.
# Stop: touch /mnt/e/chess_data/q268.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
PROD="$ND/build_uci/chess_engine_uci"
NEW="$ND/build_uci/chess_engine_uci_skpgo"
LOG=/mnt/e/chess_data/q268.log
STOP=/mnt/e/chess_data/q268.stop
cd "$ND" || exit 1
rm -f "$STOP"
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[b]uild_uci.sh|[_]depth_residual|[_]sf18_wdl|[t]rap_probe|[p]rune_ablation|[r]ook_trap" >/dev/null; }
echo "[q268] $(date '+%Y-%m-%d %H:%M:%S') start"
busy && { echo "[q268] ☠️ another job is running — QUEUE 268 ABORTED"; exit 1; }
a=$(fp "$PROD" ""); k=$(fp "$NEW" "SORT_KEYED=1"); o=$(fp "$NEW" "")
echo "[q268] g76 [$a] · skon [$k] · skoff [$o] (all 246 / 35138434)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$k" = "$a" ] && [ "$o" = "$a" ] || { echo "[q268] ☠️ a fingerprint differs — QUEUE 268 ABORTED"; exit 1; }
bench_one() {  # round, arm
	local out
	case $2 in
		g76) out=$(printf 'bench\nquit\n' | taskset -c 5 "$PROD" 2>/dev/null | grep '^bench') ;;
		skon) out=$(printf 'setoption name Knobs value SORT_KEYED=1\nbench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
		skoff) out=$(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
	esac
	echo "[q268]   r$1 $2 $out"
}
bt() { grep -E "^\[q268\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q268\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
echo "[q268] $(date '+%H:%M') == pinned bench x15 on core 5, g76 / skon / skoff rotating"
for i in $(seq 1 15); do
	[ -e "$STOP" ] && { echo "[q268] stop file at round $i — QUEUE 268 DONE"; exit 0; }
	case $((i % 3)) in 1) o="g76 skon skoff" ;; 2) o="skon skoff g76" ;; 0) o="skoff g76 skon" ;; esac
	for x in $o; do bench_one $i $x; done
done
for x in g76 skon skoff; do echo "[q268] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) · range $(bt $x | awk '{print $2}' | sort -n | sed -n '1p;$p' | tr '\n' ' ')over $(bt $x | wc -l) runs"; done
pair() { echo "[q268] within-round $1 / $2: time median $(join <(bt $1) <(bt $2) | awk '{print $2 / $3}' | med) ($1 faster in $(join <(bt $1) <(bt $2) | awk '$2 < $3 {n++} END {print n + 0}') of 15) · nps median $(join <(bn $1) <(bn $2) | awk '{print $2 / $3}' | med)"; }
pair skon g76; pair skon skoff; pair skoff g76
echo "[q268] registered: skon / g76 median < 0.990 and faster in ≥ 11 of 15 = speed-ship candidate (prediction −0.5 … −2%)"
echo "[q268] $(date) QUEUE 268 DONE"
