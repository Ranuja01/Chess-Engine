#!/bin/bash
# 2026-10-21, queue #91 (owner: "spend the night efficiently") — ONE job at a time, bench guard first, games hard-stop EXIT_AT.
#   CORR   corrhist Phase-0 offline gate on v2 (gen 5): corrlog over 200 game_regret_set positions at d10 → corrsignal.
#          The 08-24 closure (keyed table loses to one GLOBAL slot; residual position-local) was measured on v1's residual.
#   PXV    PE_PASSER_EXT=1 vs gen 5, st=1, 400 g from eg_critical starts (the venue where it fires; move test +0.012 ± 0.011)
#   PXU    same, 200 g from UHO (narrow-item rule: elsewhere only "not harmful")
#   REC    gen 5 v2 bench record (WAC d10 · STS300 · STS full · WAC LIGHTNING depth · WAC speed); v1 and static eval unchanged
# Registered: PXV ≈ +0 … +20 (fires in most endgame games, small per-move gain); PXU ≈ 0; corrhist on v2 — GLOBAL still beats
# keyed (the starvation cause is structural), i.e. the closure holds; REC STS within ±30 of gen 4's 2115.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
B="$ND/build_uci/chess_engine_uci"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-10:00}
cd "$ND" || exit 1
echo "[q91] $(date) start"
FP=$(printf 'bench\nquit\n' | "$B" 2>/dev/null | grep "^bench")
echo "[q91] $FP"
echo "$FP" | grep -q "solved 258/300 nodes 47689586" || { echo "[q91] ☠️ BENCH MISMATCH (expected gen 5) — aborting"; exit 1; }

echo "[q91] $(date) CORR corrlog (v2 gen 5, 200 positions, d10)"
V2_PRESET=shipped bash "$R" corrlog diagnostics/ks_sets/game_regret_set.csv 200 /mnt/e/chess_data/corrlog_v2.txt V2_PRESET=shipped
bash "$R" corrsignal /mnt/e/chess_data/corrlog_v2.txt > /mnt/e/chess_data/corrsignal_v2.txt 2>&1
echo "[q91] $(date) CORR done: $(grep -ciE 'global|replay|oracle' /mnt/e/chess_data/corrsignal_v2.txt) summary lines → corrsignal_v2.txt"

# eg_critical as clean 4-field EPD for fastchess (the source has '#' headers and '; comment' tails)
grep -v '^#' selfplay/eg_critical_fens.txt | sed 's/ *;.*$//' | awk 'NF>=4{print $1" "$2" "$3" "$4}' > $O/eg_critical.epd
echo "[q91] eg_critical.epd: $(wc -l < $O/eg_critical.epd) openings"

( while true; do sleep 1800; echo "[q91] $(date +%H:%M) mem: $(free -g | awk '/Mem:/{print "used " $3 "G avail " $7 "G"}') swap $(free -g | awk '/Swap:/{print $3 "G"}')"; done ) &
MEMPID=$!
OURS="option.OwnBook=false st=1 timemargin=250 restart=on"
ADJ="-repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
run() {   # $1 tag, $2 seed, $3 rounds, $4 openings file, $5... engine args
  local tag=$1 seed=$2 rounds=$3 book=$4; shift 4
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 900 ] && { echo "[q91] $(date) skip $tag: past EXIT_AT"; return; }
  echo "[q91] $(date) $tag ($(( rounds * 2 )) games), up to ${secs}s"
  timeout $secs nice -n 5 $FC "$@" -openings file=$book format=epd order=random $ADJ -srand $seed -rounds $rounds -games 2 \
    -pgnout file=$O/$tag.pgn notation=san nodes=true append=false > $O/$tag.log 2>&1
  echo "[q91] $(date) $tag done (rc $?): $(grep -E '^Games:|^Elo:' $O/$tag.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$tag.log) unresponsive"
}
run pxext_eg 911 200 $O/eg_critical.epd -engine cmd="$B" name=pxext $OURS option.Knob_PE_PASSER_EXT=1 -engine cmd="$B" name=gen5 $OURS
run pxext_uho 912 100 $O/openings_uho.epd -engine cmd="$B" name=pxext $OURS option.Knob_PE_PASSER_EXT=1 -engine cmd="$B" name=gen5 $OURS
kill $MEMPID 2>/dev/null

echo "[q91] $(date) REC gen 5 v2 benches"
K="V2_PRESET=shipped"
echo "[q91] == WAC d10";        bash "$R" wac rec5_v2 $K | head -3
echo "[q91] == STS300 d10";     bash "$R" sts rec5_v2 $K | tail -2
echo "[q91] == STS full d10";   bash "$R" sts_full recf5_v2 $K | tail -1
echo "[q91] == WAC LIGHTNING";  bash "$R" wac_timed_depth rect5_v2 $K | head -6
echo "[q91] == WAC speed";      bash "$R" wac_speed recs5_v2 3 $K | grep -E "NODES:|MAX_NPS|MEDIAN|SPREAD"
echo "[q91] $(date) QUEUE 91 DONE"
