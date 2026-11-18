#!/bin/bash
# 2026-11-17 ~21:35, queue #267: SORT_KEYED's TIME read (H-S speed item 1). #266 (plain build, game12 d10): instructions +0.38%,
# mispredictions −9.69% (89.89 → 81.18 M); every gate equal, the std::stable_sort check clean.
#  1 PGO build chess_engine_uci_skpgo from the current sources (gen 7.6; SORT_KEYED default 0), the profile trained with
#    SORT_KEYED=1 (/mnt/e/chess_data/pgo_sk). Production untouched. Gates: default 246 / 35,138,434 == SORT_KEYED=1 == the
#    SCRATCH_REUSE=2 check; v1 250 / 35,310,775-778; game12 d10 default == SORT_KEYED=1.
#  2 WAITS for /mnt/e/chess_data/q267.go (touched when the owner's gaming ends; the build above is fixed work), latest 10:00.
#  3 pinned bench ×9 on core 5, three arms rotating: g76 (production), skon (skpgo + SORT_KEYED=1), skoff (skpgo default).
# REGISTERED before the bench: skon / g76 within-round time median < 0.990 AND skon faster in ≥ 7 of 9 rounds = a speed-ship
# candidate (owner's call); prediction −0.5 … −2%. skoff / g76 reads the build's own layout and profile.
# Stop: touch /mnt/e/chess_data/q267.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q267
P=/mnt/e/chess_data/pgo_sk
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
NEW="$ND/build_uci/chess_engine_uci_skpgo"
LOG=/mnt/e/chess_data/q267.log
STOP=/mnt/e/chess_data/q267.stop
GO=/mnt/e/chess_data/q267.go
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q267]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q267] $(date) stop file before $1 — QUEUE 267 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*|abort.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[b]uild_uci.sh|[_]depth_residual|[_]sf18_wdl|[t]rap_probe" >/dev/null; }
echo "[q267] $(date '+%Y-%m-%d %H:%M:%S') start"
busy && { echo "[q267] ☠️ another job is running — QUEUE 267 ABORTED"; exit 1; }
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_skgen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_skgen" name=a option.OwnBook=false "option.Knobs=SORT_KEYED=1" st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_skgen" name=b option.OwnBook=false "option.Knobs=SORT_KEYED=1" st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 767 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_skpgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q267] ☠️ PGO use build failed — QUEUE 267 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q267] ☠️ PGO gen build failed — QUEUE 267 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q267] $(date '+%H:%M') build done — the sources are free to edit; training: $(grep -E '^Games:' "$W/train.log" | tail -1) · warnings $(grep -c 'have option' "$W/train.log")"
a=$(fp "$NEW" ""); k=$(fp "$NEW" "SORT_KEYED=1"); v=$(fp "$NEW" "SORT_KEYED=1,SCRATCH_REUSE=2")
v1=$(printf 'setoption name EvalArm value 0\nsetoption name Knobs value SORT_KEYED=1\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ga=$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
gk=$(printf "setoption name Knobs value SORT_KEYED=1\nbench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q267] skpgo default [$a] (246 / 35138434) · SORT_KEYED=1 [$k] · check [$v] · v1 [$v1] · game12 [$ga] / [$gk]"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$k" = "$a" ] && [ "$v" = "$a" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' \
	&& [ -n "$ga" ] && [ "$gk" = "$ga" ] || { echo "[q267] ☠️ a gate differs — no bench — QUEUE 267 DONE"; exit 1; }
echo "[q267] ✅ every gate equal; waiting for q267.go (the owner's gaming over) before the pinned bench"
LATEST=$(date -d 'tomorrow 10:00' +%s); [ "$(date +%H)" -lt 10 ] && LATEST=$(date -d 'today 10:00' +%s)
until [ -e "$GO" ]; do stopped "the bench"; [ "$(date +%s)" -ge "$LATEST" ] && { echo "[q267] $(date) no go by 10:00 — bench skipped — QUEUE 267 DONE"; exit 0; }; sleep 30; done
busy && echo "[q267] ☠️ another job is running — the pinned bench below is disturbed"
bench_one() {  # round, arm
	local out
	case $2 in
		g76) out=$(printf 'bench\nquit\n' | taskset -c 5 "$PROD" 2>/dev/null | grep '^bench') ;;
		skon) out=$(printf 'setoption name Knobs value SORT_KEYED=1\nbench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
		skoff) out=$(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
	esac
	echo "[q267]   r$1 $2 $out"
}
bt() { grep -E "^\[q267\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q267\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
echo "[q267] $(date '+%H:%M') == pinned bench x9 on core 5, g76 / skon / skoff rotating"
for i in 1 2 3 4 5 6 7 8 9; do
	stopped "bench round $i"
	case $((i % 3)) in 1) o="g76 skon skoff" ;; 2) o="skon skoff g76" ;; 0) o="skoff g76 skon" ;; esac
	for x in $o; do bench_one $i $x; done
done
for x in g76 skon skoff; do echo "[q267] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
pair() { echo "[q267] within-round $1 / $2: time median $(join <(bt $1) <(bt $2) | awk '{print $2 / $3}' | med) ($1 faster in $(join <(bt $1) <(bt $2) | awk '$2 < $3 {n++} END {print n + 0}') of 9) · nps median $(join <(bn $1) <(bn $2) | awk '{print $2 / $3}' | med)"; }
pair skon g76; pair skon skoff; pair skoff g76
echo "[q267] registered: skon / g76 median < 0.990 and faster in ≥ 7 of 9 = speed-ship candidate (prediction −0.5 … −2%)"
echo "[q267] $(date) QUEUE 267 DONE"
