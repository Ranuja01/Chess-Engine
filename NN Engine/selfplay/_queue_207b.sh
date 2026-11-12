#!/bin/bash
# 2026-11-11 23:50, queue #207b: queue #207's timed steps (2 speed, 3 games smoke), unchanged, moved behind the owner's go.
# The owner may game past 00:30 (#207's fixed start), so #207 was stopped while it waited (its builds sv2_pgo0 / sv2_pgo1 were
# done and gated at 20:12) and this queue runs its steps 2 and 3 on those binaries, with #207's registered rules (see its
# header), once the go file exists: touch /mnt/e/chess_data/q207.go (written when the owner says it is safe). Fallback start
# 03:00 if no go arrives. At the end it appends "QUEUE 207 DONE" to q207.log, which is what queue #209 waits for.
# Stop: touch /mnt/e/chess_data/q207b.stop (checked while waiting and between steps; a stop leaves #209 waiting).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/sv2_sw3
STOP=/mnt/e/chess_data/q207b.stop
GO=/mnt/e/chess_data/q207.go
L207=/mnt/e/chess_data/q207.log
FALLBACK=$(date -d '2026-11-12 03:00' +%s)
LATEST=$(date -d '2026-11-12 09:30' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"
show() { sed "s/^/[q207b]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q207b] $(date) stop file present before $1 — stopped (#209 left waiting)"; exit 0; }; }
done207() { echo "[q207b] $(date) $1"; echo "[q207] $(date) (via #207b) $1" >> "$L207"; exit 0; }
echo "[q207b] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #207 to exit, then for the go file (fallback 03:00)"
until grep -q "stopped while waiting" "$L207" 2>/dev/null; do stopped "#207 exited"; sleep 30; done
[ -x "$ND/build_uci/sv2_pgo0" ] && [ -x "$ND/build_uci/sv2_pgo1" ] || { echo "[q207b] ☠️ #207's binaries missing — stopped"; exit 1; }
echo "[q207b] $(date '+%H:%M') #207 exited; binaries present; waiting for the go"
until [ -e "$GO" ] || [ $(date +%s) -ge $FALLBACK ]; do stopped "the go"; sleep 30; done
echo "[q207b] $(date '+%Y-%m-%d %H:%M:%S') timed steps start ($([ -e "$GO" ] && echo 'go file' || echo 'fallback time'))"

# ── 2 real-play speed, alternated (as #207) ──────────────────────────────────────────────────────────────────────────
for rep in 1 2 3; do
	for arm in 0 1; do
		stopped "speed rep $rep arm $arm"
		[ $(date +%s) -lt $LATEST ] || done207 "past 09:30 — QUEUE 207 DONE"
		timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=arm$arm="$ND/build_uci/sv2_pgo$arm" \
			O1="OwnBook:false;Knobs:SEARCH_ARM=$arm" D1=13 N=40 SEED=1 OUT="$W/speed_r${rep}_arm$arm" > "$W/speed_r${rep}_arm$arm.log" 2>&1
		echo "[q207b] speed rep $rep arm $arm: $(grep -iE 'time to depth 13|EBF over' "$W/speed_r${rep}_arm$arm.log" | head -3 | tr '\n' ' ')"
	done
done
bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$W" > "$W/speed_read.txt" 2>&1
show < "$W/speed_read.txt"

# ── 3 games smoke (as #207) ──────────────────────────────────────────────────────────────────────────────────────────
stopped "the games smoke"
[ $(date +%s) -lt $LATEST ] || done207 "past 09:30 — QUEUE 207 DONE"
echo "[q207b] $(date '+%H:%M') games smoke: arm 1 vs arm 0 at 10+0.1, 300 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$ND/build_uci/sv2_pgo1" name=arm1 option.OwnBook=false option.Knobs=SEARCH_ARM=1 tc=10+0.1 timemargin=250 \
	-engine "cmd=$ND/build_uci/sv2_pgo0" name=arm0 option.OwnBook=false option.Knobs=SEARCH_ARM=0 tc=10+0.1 timemargin=250 \
	$COMMON -srand 207 -rounds 150 -games 2 -pgnout file=$W/smoke.pgn notation=san nodes=true append=false > $W/smoke.log 2>&1
echo "[q207b] smoke: $(grep -E '^Games:' $W/smoke.log | tail -1) · warnings $(grep -c 'have option' $W/smoke.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/smoke.log) · time forfeits $(grep -c 'loses on time' $W/smoke.pgn)"
grep -E 'Elo:|Ptnml' $W/smoke.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/smoke.pgn --p1 arm1 2>&1 | sed -n '/per engine/,$p' | show
done207 "QUEUE 207 DONE"
