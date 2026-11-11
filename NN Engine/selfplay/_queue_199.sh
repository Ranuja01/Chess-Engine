#!/bin/bash
# 2026-11-11 ~12:30, queue #199 (owner 11-11: "sure" to building gen 7.2): gen 7.2 = gen 7.1 + MOPUP_V2 100 in the gen 7 block of
# the v2 shipped preset, plus no resignation with a lone king against no pawns (search_engine.cpp resign check).
# Evidence (OPTIMIZATION_LOG 11-11 11:10, queue #198): K+B+N vs K 30 / 30 at 1M nodes, 27 / 30 at 100k (were 1 / 30); two
# bishops 30 / 30 at 100k (15); mirror test 0 violations; cachegrind +0.0009% instructions with identical trees.
#  1 PGO build chess_engine_uci_g72pgo from the repo sources. Gate: default bench 247 / 39,861,964 (#198's MOPUP_V2=100 reading,
#    expected), MOPUP_V2=0 must restore gen 7.1 (247 / 39,852,210), LAZY_V2=0,MOPUP_V2=0 gen 7 (246 / 36,509,830),
#    V2_SHIP_GEN=6 gen 6.7 (251 / 36,859,850), v1 250 / 35,310,775-778. Install as production; gen 7.1 stays as g71pgo.
#  2 Cython module rebuild (its .so backed up first; skipped while the UI runs) and its fingerprints.
#  3 eval symmetry gate, N=4000.
#  4 pinned bench, gen 7.1 and gen 7.2 production binaries alternating x6 on core 5.
#  5 production smoke at 10+0.1: 60 games gen 7.2 vs gen 7.1 (crashes, time forfeits; too few games for Elo).
# Stop: touch /mnt/e/chess_data/q199.stop (checked between steps). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g72
OLD="$ND/build_uci/chess_engine_uci_g71pgo"
NEW="$ND/build_uci/chess_engine_uci_g72pgo"
STOP=/mnt/e/chess_data/q199.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q199]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q199] $(date) stop file present before $1 — QUEUE 199 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|_mopup_conversion" >/dev/null; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q199] $START start"
busy && { echo "[q199] ☠️ a games or bench job is running — QUEUE 199 ABORTED"; exit 1; }
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q199] backup ok: production == chess_engine_uci_g71pgo" \
  || { echo "[q199] ☠️ production differs from chess_engine_uci_g71pgo — QUEUE 199 ABORTED"; exit 1; }
grep -q "Config::MOPUP_V2 = 100;" search_engine.cpp && grep -q "bare_king_defence" search_engine.cpp \
  && grep -q 'AlphaUX (v2 gen 7.2)' uci_main.cpp || { echo "[q199] ☠️ the gen 7.2 sources are missing — QUEUE 199 ABORTED"; exit 1; }

# ── 1 build, gate, install ────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g72gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q199] ☠️ gen build failed — QUEUE 199 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g72gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g72gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 72 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g72_train.log 2>&1
echo "[q199] training: $(grep -E '^Games:' /tmp/pgo_g72_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g72pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q199] ☠️ use build failed — QUEUE 199 ABORTED"; exit 1; }
d=$(fp "$NEW" "")
off=$(fp "$NEW" MOPUP_V2=0)
g7=$(fp "$NEW" LAZY_V2=0,MOPUP_V2=0)
g6=$(fp "$NEW" V2_SHIP_GEN=6)
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q199] gen 7.2 default (expected 247 / 39861964):           $d"
echo "[q199] MOPUP_V2=0 = gen 7.1 (want 247 / 39852210):          $off"
echo "[q199] LAZY_V2=0,MOPUP_V2=0 = gen 7 (want 246 / 36509830):  $g7"
echo "[q199] V2_SHIP_GEN=6 = gen 6.7 (want 251 / 36859850):       $g6"
echo "[q199] v1 EvalArm 0 (want 250 / 35310775-35310778):         $v1"
echo "[q199] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
[ "$d" = "solved 247/300 nodes 39861964" ] && [ "$off" = "solved 247/300 nodes 39852210" ] \
  && [ "$g7" = "solved 246/300 nodes 36509830" ] && [ "$g6" = "solved 251/300 nodes 36859850" ] \
  && echo "$v1" | grep -qE '^solved 250/300 nodes 3531077[5-8]$' \
  || { echo "[q199] ☠️ fingerprints differ — production NOT replaced, QUEUE 199 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
  && cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q199] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g72pgo" \
  || { echo "[q199] ☠️ install failed — QUEUE 199 ABORTED"; exit 1; }
echo "[q199] installed copy, default: $(fp build_uci/chess_engine_uci "")"

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython rebuild"
CY=0
if pgrep -f "chess_ui" >/dev/null; then
  echo "[q199] $(date) the UI is running — Cython rebuild skipped (rebuild later: bash selfplay/overnight_runner.sh build)"
else
  SO=$(ls ChessAI.*.so 2>/dev/null | head -1)
  [ -n "$SO" ] && cp "$SO" "$P/g71_$SO" && echo "[q199] backed up $SO to $P/g71_$SO"
  bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
  if ls ChessAI.*.so >/dev/null 2>&1; then
    CY=1
    echo "[q199] Cython gen 7.2 shipped (expected 247 / 39861964):  $(bash "$R" wac fp_g72 V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
    echo "[q199] Cython MOPUP_V2=0 = gen 7.1 (want 247 / 39852210): $(bash "$R" wac fp_g72off V2_PRESET=shipped MOPUP_V2=0 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
    echo "[q199] Cython v1 (want 250 / 35310778):                   $(bash "$R" wac fp_g72v1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
  else
    echo "[q199] ☠️ Cython build left no .so — restoring the gen 7.1 one"
    [ -n "$SO" ] && cp "$P/g71_$SO" "$SO"
  fi
fi

# ── 3 eval symmetry ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the symmetry gate"
if [ "$CY" = 1 ]; then
  echo "[q199] symmetry, gen 7.2:"
  env V2_PRESET=shipped bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations|clean" | show
else
  echo "[q199] symmetry gate skipped (no rebuilt Cython module)"
fi

# ── 4 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the pinned bench"
while busy; do sleep 30; done
echo "[q199] $(date) == TIMED: UCI bench, gen 7.1 and gen 7.2 production binaries, pinned to core 5, alternating x6"
rm -f /tmp/q199_bench.txt
for i in 1 2 3 4 5 6; do
  echo "g71 $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')" | tee -a /tmp/q199_bench.txt | show
  echo "g72 $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')" | tee -a /tmp/q199_bench.txt | show
done
awk '{ for (i = 1; i <= NF; i++) { if ($i == "time") t[$1] += $(i+1); if ($i == "nps") s[$1] += $(i+1) } n[$1]++ }
  END { printf "mean time g71 %.0f ms · g72 %.0f ms (%+.1f%%) · mean nps g71 %.0f · g72 %.0f (%+.1f%%)\n",
        t["g71"]/n["g71"], t["g72"]/n["g72"], 100*(t["g72"]/t["g71"]-1), s["g71"]/n["g71"], s["g72"]/n["g72"], 100*(s["g72"]/s["g71"]-1) }' /tmp/q199_bench.txt | show

# ── 5 production smoke ────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the production smoke"
while busy; do sleep 30; done
echo "[q199] $(date) production smoke: installed gen 7.2 vs gen 7.1 at 10+0.1, 60 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$ND/build_uci/chess_engine_uci" name=g72 option.OwnBook=false tc=10+0.1 timemargin=250 \
  -engine "cmd=$OLD" name=g71 option.OwnBook=false tc=10+0.1 timemargin=250 \
  $COMMON -srand 199111 -rounds 30 -games 2 -pgnout file=$O/g72_smoke.pgn notation=san nodes=true append=false > $O/g72_smoke.log 2>&1
echo "[q199] smoke: $(grep -E '^Games:' $O/g72_smoke.log | tail -1) · warnings $(grep -c 'have option' $O/g72_smoke.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/g72_smoke.log) · time forfeits $(grep -c 'loses on time' $O/g72_smoke.pgn)"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $O/g72_smoke.pgn --p1 g72 2>&1 | sed -n '/per engine/,$p' | show
echo "[q199] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q199] $(date) QUEUE 199 DONE"
