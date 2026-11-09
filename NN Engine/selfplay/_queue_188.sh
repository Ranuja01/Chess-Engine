#!/bin/bash
# 2026-11-09 ~11:45, queue #188 (owner 11-09: "given the evidence we can ship this, but it seems we must monitor this as we
# continue on, since there are some weaknesses"): gen 7.1 = gen 7 + lazy eval keep_gate (LAZY_V2 7, LAZY_V2_KEEP 1,
# LAZY_V2_GATE 3), set in the gen 7 block of the v2 shipped preset.
# Evidence (OPTIMIZATION_LOG 11-09): timed self-play +12.9 (-0.7 … +26.5) over 2,000 g (#185, z 1.86); Mediocre +4.0 pp (#186);
# SF18 at equal nodes -1.0 pp over 8 opening sets (#185 + #187; -2.1 pp on #187's fresh sets, z -1.24); 22-set grid 5.1% faster
# to depth 10 (#186).
#  1 PGO build chess_engine_uci_g71pgo from the repo sources. Gate: the default bench must read keep_gate's 247 / 39,852,210,
#    LAZY_V2=0 must restore gen 7 (246 / 36,509,830), V2_SHIP_GEN=6 gen 6.7 (251 / 36,859,850), v1 250 / 35,310,775-778 (v1's
#    last digits vary from run to run on one binary). Install as production; gen 7 stays as chess_engine_uci_g7pgo.
#  2 Cython module rebuild (its .so backed up first; skipped while the UI runs) and its fingerprints.
#  3 eval symmetry gate. It tests the static eval, which lazy eval does not change, so on and off must read the same.
#  4 pinned bench, gen 7 and gen 7.1 production binaries alternating x6 on core 5.
#  5 production smoke at 10+0.1: 60 games gen 7.1 vs gen 7 (crashes, time forfeits, real-play nps; too few games for Elo).
# Stop: touch /mnt/e/chess_data/q188.stop (checked between steps). Never edit this file or the sources while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
P=/mnt/e/chess_data/pgo_g71
OLD="$ND/build_uci/chess_engine_uci_g7pgo"
NEW="$ND/build_uci/chess_engine_uci_g71pgo"
LZ3="$ND/build_uci/chess_engine_uci_lazy3"
KG="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GATE=3"
STOP=/mnt/e/chess_data/q188.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q188]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q188] $(date) stop file present before $1 — QUEUE 188 DONE"; exit 0; }; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q188] $START start"
busy && { echo "[q188] ☠️ a games or bench job is running — QUEUE 188 ABORTED"; exit 1; }
cmp -s build_uci/chess_engine_uci "$OLD" && echo "[q188] backup ok: production == chess_engine_uci_g7pgo" \
  || { echo "[q188] ☠️ production differs from chess_engine_uci_g7pgo — QUEUE 188 ABORTED"; exit 1; }
grep -q "Config::LAZY_V2 = 7;  Config::LAZY_V2_KEEP = 1;  Config::LAZY_V2_GATE = 3;" search_engine.cpp \
  && grep -q 'AlphaUX (v2 gen 7.1)' uci_main.cpp || { echo "[q188] ☠️ the gen 7.1 sources are missing — QUEUE 188 ABORTED"; exit 1; }

# ── 1 build, gate, install ────────────────────────────────────────────────────────────────────────────────────────────
# SKIP_BUILD=1 reuses an existing chess_engine_uci_g71pgo (the 12:44 run built it, then stopped on a v1 gate too strict for v1's
# run-to-run wobble of the last digits, 35,310,775-778, seen on the untouched gen 7 binary too).
if [ "${SKIP_BUILD:-0}" = 1 ] && [ -x "$NEW" ]; then
  echo "[q188] SKIP_BUILD: reusing $(basename "$NEW") built $(date -r "$NEW" '+%Y-%m-%d %H:%M')"
else
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g71gen PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q188] ☠️ gen build failed — QUEUE 188 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g71gen" name=a option.OwnBook=false st=0.25 \
  -engine cmd="$ND/build_uci/chess_engine_uci_g71gen" name=b option.OwnBook=false st=0.25 \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 71 -rounds 4 -games 2 -concurrency 1 > /tmp/pgo_g71_train.log 2>&1
echo "[q188] training: $(grep -E '^Games:' /tmp/pgo_g71_train.log | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g71pgo PGO=use PGO_DIR=$P bash selfplay/build_uci.sh || { echo "[q188] ☠️ use build failed — QUEUE 188 ABORTED"; exit 1; }
fi
D=$(printf 'bench\nquit\n' | "$NEW" 2>/dev/null)
d=$(echo "$D" | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
off=$(fp "$NEW" LAZY_V2=0)
off3=$(fp "$NEW" LAZY_V2=0,LAZY_V2_KEEP=0,LAZY_V2_GATE=0)
g6=$(fp "$NEW" V2_SHIP_GEN=6)
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q188] gen 7.1 default (want 247 / 39852210):               $d"
echo "[q188]   $(echo "$D" | grep 'lazy eval')"
echo "[q188] LAZY_V2=0 = gen 7 (want 246 / 36509830):             $off"
echo "[q188] LAZY_V2=0,KEEP=0,GATE=0 (want 246 / 36509830):       $off3"
echo "[q188] V2_SHIP_GEN=6 = gen 6.7 (want 251 / 36859850):       $g6"
echo "[q188] V2_SHIP_GEN=5 (want 258 / 47689586):                 $(fp "$NEW" V2_SHIP_GEN=5)"
echo "[q188] v1 EvalArm 0 (want 250 / 35310775-35310778):         $v1"
echo "[q188] lazy3 binary, keep_gate knobs (on record 247 / 39852210): $(fp "$LZ3" "$KG")"
echo "[q188] gen 7 production, default (on record 246 / 36509830):    $(fp "$OLD" "")"
echo "[q188] id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name')"
[ "$d" = "solved 247/300 nodes 39852210" ] && [ "$off" = "solved 246/300 nodes 36509830" ] && [ "$off3" = "$off" ] \
  && [ "$g6" = "solved 251/300 nodes 36859850" ] \
  && echo "$v1" | grep -qE '^solved 250/300 nodes 3531077[5-8]$' \
  || { echo "[q188] ☠️ fingerprints differ — production NOT replaced, QUEUE 188 ABORTED"; exit 1; }
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp build_uci/chess_engine_uci \
  && cmp -s build_uci/chess_engine_uci "$NEW" && echo "[q188] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g71pgo" \
  || { echo "[q188] ☠️ install failed — QUEUE 188 ABORTED"; exit 1; }
echo "[q188] installed copy, default: $(fp build_uci/chess_engine_uci "")"

# ── 2 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython rebuild"
CY=0
if pgrep -f "chess_ui" >/dev/null; then
  echo "[q188] $(date) the UI is running — Cython rebuild skipped (rebuild later: bash selfplay/overnight_runner.sh build)"
else
  SO=$(ls ChessAI.*.so 2>/dev/null | head -1)
  [ -n "$SO" ] && cp "$SO" "$P/g7_$SO" && echo "[q188] backed up $SO to $P/g7_$SO"
  bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
  if ls ChessAI.*.so >/dev/null 2>&1; then
    CY=1
    echo "[q188] Cython gen 7.1 shipped (want 247 / 39852210):  $(bash "$R" wac fp_g71 V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
    echo "[q188] Cython LAZY_V2=0 = gen 7 (want 246 / 36509830): $(bash "$R" wac fp_g71off V2_PRESET=shipped LAZY_V2=0 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
    echo "[q188] Cython v1 (want 250 / 35310778):               $(bash "$R" wac fp_g71v1 2>&1 | grep -E '^SOLVED|^NODES|^EBF' | tr '\n' ' ')"
  else
    echo "[q188] ☠️ Cython build left no .so — restoring the gen 7 one"
    [ -n "$SO" ] && cp "$P/g7_$SO" "$SO"
  fi
fi

# ── 3 eval symmetry ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the symmetry gate"
if [ "$CY" = 1 ]; then
  S1=$(env V2_PRESET=shipped bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  S0=$(env V2_PRESET=shipped LAZY_V2=0 bash "$R" pyrun diagnostics/_eval_symmetry.py N=4000 2>&1 | grep -E "violations")
  echo "[q188] symmetry, gen 7.1:"; echo "$S1" | show
  [ "$S1" = "$S0" ] && echo "[q188] symmetry with LAZY_V2=0 identical (as it must be: the static eval is unchanged)" \
    || { echo "[q188] ☠️ symmetry differs with LAZY_V2=0:"; echo "$S0" | show; }
else
  echo "[q188] symmetry gate skipped (no rebuilt Cython module)"
fi

# ── 4 pinned bench ────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the pinned bench"
while busy; do sleep 30; done
echo "[q188] $(date) == TIMED: UCI bench, gen 7 and gen 7.1 production binaries, pinned to core 5, alternating x6"
rm -f /tmp/q188_bench.txt
for i in 1 2 3 4 5 6; do
  echo "g7  $(printf 'bench\nquit\n' | taskset -c 5 "$OLD" 2>/dev/null | grep '^bench')" | tee -a /tmp/q188_bench.txt | show
  echo "g71 $(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench')" | tee -a /tmp/q188_bench.txt | show
done
awk '{ for (i = 1; i <= NF; i++) { if ($i == "time") t[$1] += $(i+1); if ($i == "nps") s[$1] += $(i+1) } n[$1]++ }
  END { printf "mean time g7 %.0f ms · g71 %.0f ms (%+.1f%%) · mean nps g7 %.0f · g71 %.0f (%+.1f%%)\n",
        t["g7"]/n["g7"], t["g71"]/n["g71"], 100*(t["g71"]/t["g7"]-1), s["g7"]/n["g7"], s["g71"]/n["g71"], 100*(s["g71"]/s["g7"]-1) }' /tmp/q188_bench.txt | show

# ── 5 production smoke ────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the production smoke"
while busy; do sleep 30; done
echo "[q188] $(date) production smoke: installed gen 7.1 vs gen 7 at 10+0.1, 60 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$ND/build_uci/chess_engine_uci" name=g71 option.OwnBook=false tc=10+0.1 timemargin=250 \
  -engine "cmd=$OLD" name=g7 option.OwnBook=false tc=10+0.1 timemargin=250 \
  $COMMON -srand 188111 -rounds 30 -games 2 -pgnout file=$O/g71_smoke.pgn notation=san nodes=true append=false > $O/g71_smoke.log 2>&1
echo "[q188] smoke: $(grep -E '^Games:' $O/g71_smoke.log | tail -1) · warnings $(grep -c 'have option' $O/g71_smoke.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/g71_smoke.log) · time forfeits $(grep -c 'loses on time' $O/g71_smoke.pgn)"
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $O/g71_smoke.pgn --p1 g71 2>&1 | sed -n '/per engine/,$p' | show
echo "[q188] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q188] $(date) QUEUE 188 DONE"
