#!/bin/bash
# 2026-11-05 afternoon, queue #161 (owner 11-05: spend today / tonight on high-quality data, tune tomorrow): three new corpus
# slices for the retune, filling coverage gaps of corpus_v2 (odds starts absent; the 960 slice VAR is 99.8% middlegame;
# K+pawns is the only strip-down stress). Runs BESIDE #160 (fixed work, not timed): at most 2 processes of its own.
#  1 games, node-limited (fixed work, deterministic): production gen 6.7, side a 25k nodes vs side b 40k nodes (different
#    budgets so a colour-swapped pair is two different games), played to the end: no draw adjudication before move 60, resign
#    only at 25 pawns for 6 moves, draw at 150 moves. 960-style starts (openings_variant, 2,376, one game each) and odds starts
#    (openings_odds, 480, both colours = 960 games). Concurrency 2.
#  2 sampling (_build_game_regret_set.py GAMES_PGN=, by-game ids for the splits, EG_QUOTA 0.4): VARG 20,000, ODDS 8,000.
#    STRIP 12,000 from gen_kp_fens.py MIX=nopawn,heavy,mixed SEED=91.
#  3 SF18 d14 single-best labels (K=1, the corpus contract), 2 processes.
# Output on /mnt/e/chess_data/retune (a few MB). Stop: touch /mnt/e/chess_data/q161.stop (between steps). Never edit while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
D=/mnt/e/chess_data/retune
G=/mnt/e/chess_data/datagen
PROD="$ND/build_uci/chess_engine_uci"
STOP=/mnt/e/chess_data/q161.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$G"
show() { sed "s/^/[q161]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q161] $(date) stop file present before $1 — QUEUE 161 DONE"; exit 0; }; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q161] $START start"
cmp -s "$PROD" "$ND/build_uci/chess_engine_uci_g67pgo" && echo "[q161] production == chess_engine_uci_g67pgo" \
  || { echo "[q161] ☠️ production is not gen 6.7 — QUEUE 161 ABORTED"; exit 1; }

# ── 1 games ───────────────────────────────────────────────────────────────────────────────────────────────────────────
grep -v '^#' selfplay/openings_variant.txt | sed 's/ *;.*$//' | grep -v '^$' > $G/variant.epd
grep -v '^#' selfplay/openings_odds.txt    | sed 's/ *;.*$//' | grep -v '^$' > $G/odds.epd
NV=$(wc -l < $G/variant.epd); NO=$(wc -l < $G/odds.epd)
echo "[q161] starts: variant $NV, odds $NO"
ADJ="-draw movenumber=60 movecount=10 score=5 -resign movecount=6 score=2500 -maxmoves 150 -recover"
E1="-engine cmd=$PROD name=a option.OwnBook=false nodes=25000"
E2="-engine cmd=$PROD name=b option.OwnBook=false nodes=40000"
t0=$(date +%s)
timeout 600 "$FC" $E1 $E2 -openings file=$G/variant.epd format=epd order=sequential -rounds 4 -games 1 -concurrency 1 $ADJ \
  -pgnout file=$G/pilot.pgn notation=san append=false > $G/pilot.log 2>&1
n=$(grep -c '^\[Event' $G/pilot.pgn 2>/dev/null)
echo "[q161] pilot: $n games in $(( $(date +%s) - t0 )) s · $(grep -E '^Games:' $G/pilot.log | tail -1) · warnings $(grep -c 'have option' $G/pilot.log) · plies $(grep -oE '\[PlyCount "[0-9]+' $G/pilot.pgn | grep -oE '[0-9]+$' | tr '\n' ' ')"
[ "${n:-0}" -ge 4 ] || { echo "[q161] ☠️ pilot failed — QUEUE 161 ABORTED"; tail -5 $G/pilot.log | show; exit 1; }
stopped "the variant games"
echo "[q161] $(date) variant games ($NV)"
nice -n 5 "$FC" $E1 $E2 -openings file=$G/variant.epd format=epd order=sequential -rounds $NV -games 1 -concurrency 2 $ADJ \
  -pgnout file=$G/varg.pgn notation=san append=false > $G/varg.log 2>&1
echo "[q161] $(date) variant games done: $(grep -c '^\[Event' $G/varg.pgn) · $(grep -E '^Games:' $G/varg.log | tail -1)"
stopped "the odds games"
echo "[q161] $(date) odds games ($NO x 2 colours)"
nice -n 5 "$FC" $E1 $E2 -openings file=$G/odds.epd format=epd order=sequential -rounds $NO -games 2 -repeat -concurrency 2 $ADJ \
  -pgnout file=$G/odds.pgn notation=san append=false > $G/odds.log 2>&1
echo "[q161] $(date) odds games done: $(grep -c '^\[Event' $G/odds.pgn) · $(grep -E '^Games:' $G/odds.log | tail -1)"

# ── 2 sampling ────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "sampling"
EXC=$D/A_ownplay_fens.csv,$D/A2_fens.csv,$D/V1_fens.csv,$D/KP_fens.csv,$D/GAUNT_fens.csv
[ -s $D/VARG_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py "GAMES_PGN=$G/varg.pgn" N=20000 PLY_STRIDE=8 \
    EG_QUOTA=0.4 MAX_GAMES=100000 EXCLUDE_EXTRA=$EXC SAMPLE_ONLY=$D/VARG_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror" | show
[ -s $D/ODDS_fens.csv ] || bash "$R" pyrun diagnostics/_build_game_regret_set.py "GAMES_PGN=$G/odds.pgn" N=8000 PLY_STRIDE=6 \
    EG_QUOTA=0.4 MAX_GAMES=100000 EXCLUDE_EXTRA=$EXC,$D/VARG_fens.csv SAMPLE_ONLY=$D/ODDS_fens.csv 2>&1 | grep -E "SAMPLE_ONLY|rror" | show
if [ ! -s $D/STRIP_fens.csv ]; then
  bash "$R" pyrun diagnostics/gen_kp_fens.py N=12000 SEED=91 MIX=nopawn,heavy,mixed OUT=$G/strip.txt 2>&1 | show
  { echo "fen"; grep -v '^#' $G/strip.txt; } > $D/STRIP_fens.csv
fi
echo "[q161] sampled: VARG $(($(wc -l < $D/VARG_fens.csv) - 1)) · ODDS $(($(wc -l < $D/ODDS_fens.csv) - 1)) · STRIP $(($(wc -l < $D/STRIP_fens.csv) - 1))"
for s in VARG ODDS STRIP; do
  for i in 0 1; do head -1 $D/${s}_fens.csv > $D/${s}_fens_s$i.csv; done
  tail -n +2 $D/${s}_fens.csv | awk -v B="$D/${s}_fens" '{print >> (B "_s" (NR-1)%2 ".csv")}'
done

# ── 3 SF18 labels, 2 processes ────────────────────────────────────────────────────────────────────────────────────────
stopped "labelling"
echo "[q161] $(date) labelling (SF18 d14 K=1)"
for s in VARG ODDS STRIP; do
  for i in 0 1; do
    [ -e "$STOP" ] && break 2
    while [ $(jobs -rp | wc -l) -ge 2 ]; do wait -n; done
    ( bash "$R" pyrun diagnostics/_build_regret_set.py IN=$D/${s}_fens_s$i.csv K=1 SF_DEPTH=14 OUT=$D/${s}_fens_sf18_s$i.csv \
        > $G/label_${s}_$i.log 2>&1; echo "[q161] $(date) labelled $s s$i: $(grep -vc '^fen,' $D/${s}_fens_sf18_s$i.csv)" ) &
  done
done
wait
echo "[q161] $(date) labelled: VARG $(cat $D/VARG_fens_sf18_s*.csv | grep -vc '^fen,') · ODDS $(cat $D/ODDS_fens_sf18_s*.csv | grep -vc '^fen,') · STRIP $(cat $D/STRIP_fens_sf18_s*.csv | grep -vc '^fen,')"
echo "[q161] $(date) QUEUE 161 DONE"
