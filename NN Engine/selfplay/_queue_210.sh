#!/bin/bash
# 2026-11-11 ~22:00, queue #210 (morning, after #209; owner asked to use the 08:00-10:30 window): a second batch of #209's
# SEE_PER_MOVE match, so the one behaviour question of the night has more power.
#   A = SEE_PER_MOVE=3, B = SEE_PER_MOVE=0, the same binary (build_uci/sv2_pgo1pe) and knobs as #209, 10+0.1, UHO, seed 210
#   (different openings from #209's seed 209), 800 games.
# REGISTERED 11-11 ~22:00, BEFORE ANY GAME OF #209 OR #210: the verdict is read on #209 + #210 COMBINED (up to 2,800 games),
# by #209's rule (A − B ≥ +2σ pentanomial = KEEP as S3's default; within ±2σ = NEUTRAL, kept as a knob; ≤ −2σ = REJECT for
# ordering). This supersedes #209's 2,000-game sample size; #209's own figure is reported but is not the verdict.
# Starts only if #209 played its games and it is before 09:15; ends by about 10:30. Stop: touch /mnt/e/chess_data/q210.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/s3_a
B="$ND/build_uci/sv2_pgo1pe"
STOP=/mnt/e/chess_data/q210.stop
LATEST_START=$(date -d '2026-11-12 09:15' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q210]   /"; }
echo "[q210] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #209"
until grep -qE "QUEUE 209 (DONE|ABORTED)|skipped" /mnt/e/chess_data/q209.log 2>/dev/null; do
	[ -e "$STOP" ] && { echo "[q210] stopped while waiting"; exit 0; }
	sleep 60
done
[ -s "$W/see.pgn" ] || { echo "[q210] #209 played no SEE games — skipped"; exit 0; }
[ $(date +%s) -lt $LATEST_START ] || { echo "[q210] $(date) past 09:15 — skipped"; exit 0; }
PE=$(grep -q "all SAME → SV2_PRUNE_EARLY=1" /mnt/e/chess_data/q209.log && echo 1 || echo 0)
echo "[q210] $(date '+%H:%M') start: SEE_PER_MOVE=3 vs 0, 800 games, SV2_PRUNE_EARLY=$PE (as #209)"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$B" name=see3 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE,SEE_PER_MOVE=3 tc=10+0.1 timemargin=250 \
	-engine "cmd=$B" name=see0 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE,SEE_PER_MOVE=0 tc=10+0.1 timemargin=250 \
	$COMMON -srand 210 -rounds 400 -games 2 -pgnout file=$W/see2.pgn notation=san nodes=true append=false > $W/see2.log 2>&1
echo "[q210] batch 2: $(grep -E '^Games:' $W/see2.log | tail -1) · warnings $(grep -c 'have option' $W/see2.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/see2.log) · time forfeits $(grep -c 'loses on time' $W/see2.pgn)"
cat "$W/see.pgn" "$W/see2.pgn" > "$W/see_combined.pgn"
echo "[q210] == COMBINED (#209 + #210), the registered verdict:"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/see_combined.pgn --p1 see3 2>&1 | sed -n '/per engine/,$p' | show
echo "[q210] $(date) QUEUE 210 DONE"
