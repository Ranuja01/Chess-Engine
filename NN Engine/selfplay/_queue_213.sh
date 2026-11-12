#!/bin/bash
# 2026-11-12 ~12:30, queue #213: the gen 7.3 ship sheet's bench rows (owner 11-12: "when looking to ship, give the NPS, and bench
# results as well as potential elo gain as info"). SEE_PER_MOVE=3 vs production behaviour on the gate binary
# (build_uci/sv2_pgo1pe; ctl = no knobs = gen 7.2's fingerprint). A read, not a gate.
#  1 pinned bench, alternating ctl / see3, 4 rounds each, one core (taskset -c 5): WAC d10 solved / nodes / time / nps
#  2 STS300 d10 and WAC mirrored d10, both arms
# Needs a quiet box: waits for #211 and #212 to finish; starts only before 15:20 (the owner games from 15:30), otherwise waits
# for the go file /mnt/e/chess_data/q213.go (written when the owner says the box is free). Stop: touch /mnt/e/chess_data/q213.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/sv2_pgo1pe"
S=/mnt/e/chess_data/sv2_id
STOP=/mnt/e/chess_data/q213.stop
GO=/mnt/e/chess_data/q213.go
CUT=$(date -d '2026-11-12 15:20' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"
echo "[q213] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #211 and #212"
until grep -qE "QUEUE 211 (DONE|ABORTED)" /mnt/e/chess_data/q211.log 2>/dev/null && grep -qE "QUEUE 212 (DONE|ABORTED)" /mnt/e/chess_data/q212.log 2>/dev/null; do
	[ -e "$STOP" ] && { echo "[q213] stopped while waiting"; exit 0; }
	sleep 30
done
if [ $(date +%s) -ge $CUT ]; then
	echo "[q213] $(date '+%H:%M') past 15:20 — waiting for the go file (the owner's gaming)"
	until [ -e "$GO" ]; do [ -e "$STOP" ] && { echo "[q213] stopped while waiting"; exit 0; }; sleep 30; done
fi
echo "[q213] $(date '+%H:%M:%S') start"
kn() { case $1 in ctl) echo "" ;; see3) echo "SEE_PER_MOVE=3" ;; esac; }
run() {  # arm, bench args
	local k cmd="bench"
	k=$(kn $1); [ -n "$k" ] && k="setoption name Knobs value $k"
	[ -n "$2" ] && cmd="bench $2"
	printf '%s\n%s\nquit\n' "$k" "$cmd" | taskset -c 5 "$B" 2>/dev/null | grep '^bench'
}
for i in 1 2 3 4; do
	for a in ctl see3; do
		[ -e "$STOP" ] && { echo "[q213] stopped"; exit 0; }
		echo "[q213] pinned WAC d10 round $i $a: $(run $a "")"
	done
done
for f in sts300 wac_mirror; do
	for a in ctl see3; do
		[ -e "$STOP" ] && { echo "[q213] stopped"; exit 0; }
		echo "[q213] $f d10 $a: $(run $a "$S/$f.txt 10")"
	done
done
echo "[q213] $(date) QUEUE 213 DONE"
