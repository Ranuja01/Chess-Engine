#!/bin/bash
# 2026-10-23, queue #95 (owner: run in this window) — which table's size carries the −30 of #94's small footprint?
#  1 eval-cache identity: v2 bench at EVAL_LOG2=20 (identical nodes => v2's eval cache can shrink for free)
#  2 TT speed sweep (fixed-depth bench, pinned core 4, x2): TT_LOG2 24..20
#  3 one table at a time vs current, same binary via option.Knobs, st=1, 400 g each: TT_LOG2=21 · MGCACHE_LOG2=20
# Registered: (1) identical; (3) the move-gen cache arm carries most of the loss (its lists hold move-ordering state); TT-only ≈ 0.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
EXIT_AT=${EXIT_AT:-20:30}
cd "$ND" || exit 1
echo "[q95] $(date) start"
b() { printf "$1bench\nquit\n" | taskset -c 4 "$B" 2>/dev/null | grep '^bench'; }
echo "[q95] default v2:        $(b '')"
echo "[q95] EVAL_LOG2=20 v2:   $(b 'setoption name Knobs value EVAL_LOG2=20\n')   (identical = 47689586)"
for t in 24 23 22 21 20; do
  for r in 1 2; do echo "[q95] sweep TT_LOG2=$t #$r: $(b "setoption name Knobs value TT_LOG2=$t\n")"; done
done
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
match() {   # $1 tag, $2 seed, $3 knobs
  local secs=$(( $(date -d "$EXIT_AT" +%s) - $(date +%s) )); [ $secs -lt 0 ] && secs=$(( secs + 86400 ))
  [ $secs -lt 900 ] && { echo "[q95] $(date) skip $1: past EXIT_AT"; return; }
  timeout 120 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$3 st=0.05 -engine cmd="$B" name=cur option.OwnBook=false st=0.05 \
    -openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
    -log file=/tmp/q95_pre.engine.log level=info engine=true > /tmp/q95_pre.log 2>&1
  local W=$(grep -c "have option" /tmp/q95_pre.log) E=$(grep -c "knobs $3" /tmp/q95_pre.engine.log)
  echo "[q95] $1 preflight: warnings $W, echoes $E"
  [ "$W" = "0" ] && [ "$E" -ge 1 ] || { echo "[q95] ☠️ $1 KNOB CHANNEL CHECK FAILED — skipped"; return; }
  echo "[q95] $(date) $1 (400 games), up to ${secs}s"
  timeout $secs nice -n 5 "$FC" -engine cmd="$B" name=arm option.OwnBook=false option.Knobs=$3 st=1 timemargin=250 restart=on \
    -engine cmd="$B" name=cur option.OwnBook=false st=1 timemargin=250 restart=on $COMMON -srand $2 -rounds 200 -games 2 \
    -pgnout file=$O/$1.pgn notation=san nodes=true append=false > $O/$1.log 2>&1
  echo "[q95] $(date) $1 done (rc $?): $(grep -E '^Games:|^Elo:' $O/$1.log | tail -2 | tr '\n' ' ') $(grep -c 'not responsive' $O/$1.log) unresponsive"
}
match fp_tt21 951 TT_LOG2=21
match fp_mg20 952 MGCACHE_LOG2=20
echo "[q95] $(date) QUEUE 95 DONE"
