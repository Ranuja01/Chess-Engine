#!/bin/bash
# 2026-11-10 ~11:50, queue #192: resumes #190's watch item W1 after a WSL package re-registration at 06:38:22 shut the WSL VM
# down (no Windows reboot, no crash dump; #190 died in its second seed, #191 before starting). Seed 190111 finished in #190
# and is kept; this plays seeds 190112-190116 (190112 again from the start): SF18 at equal nodes, gen 7.1 (ctl) vs
# LAZY_V2=0 (off), 250 g per arm per seed, both arms of a seed before the next, on lazy4 (both fingerprints checked first).
# REGISTERED (as #190): W1, lazy on - off, over the 9 fresh sets of #189 (189111-189113) and #190 / #192 (190111-190116):
#   level if > -1 sigma (W1 closes); lazy eval costs strength at equal nodes if <= -2 sigma (raise with the owner); in
#   between, W1 stays open. Prediction about -1 pp. Only complete chunks (250 g) count.
# Node-limited, so safe while the owner uses the machine; no chunk starts after 16:30. Stop: touch /mnt/e/chess_data/q192.stop.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy4"
STOP=/mnt/e/chess_data/q192.stop
SEND=$(date -d '2026-11-10 16:30' +%s)
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q192]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() { case $1 in ctl) echo "" ;; off) echo "LAZY_V2=0" ;; esac; }
bn() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -E '^bench' | sed -E 's/ time [0-9]+ ms nps [0-9]+//'; }
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q192] $START start"
busy && { echo "[q192] ☠️ a games or bench job is running — QUEUE 192 ABORTED"; exit 1; }
F0=$(bn ""); F1=$(bn "LAZY_V2=0")
echo "[q192] ctl (want 247 / 39852210): $F0"
echo "[q192] off (want 246 / 36509830): $F1"
echo "$F0" | grep -q 'solved 247/300 nodes 39852210$' && echo "$F1" | grep -q 'solved 246/300 nodes 36509830$' \
  || { echo "[q192] ☠️ fingerprints — QUEUE 192 ABORTED"; exit 1; }

COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
sf_chunk() {  # seed, arm
  local seed=$1 a=$2 k f=$O/g18_q190_s$1_$2.log
  [ -e "$STOP" ] && return
  [ $(date +%s) -ge $SEND ] && { echo "[q192] $(date) skip SF18 seed $seed $a (past 16:30)"; return; }
  while busy; do sleep 30; done
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  [ "$a" = ctl ] || [ -n "$k" ] || { echo "[q192] ☠️ empty knobs for arm $a — chunk skipped"; return; }
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    -each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q190_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
  echo "[q192] $(date) SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
local_loop() {
  local s
  for s in 190112 190113 190114 190115 190116; do
    if [ $(( s % 2 )) = 1 ]; then sf_chunk $s ctl; sf_chunk $s off; else sf_chunk $s off; sf_chunk $s ctl; fi
  done
}
echo "[q192] $(date) W1, SF18 at equal nodes (node-limited), ctl vs off, seeds 190112-190116"
local_loop

row() {  # seed, arm, log: complete chunks (250 g) only
  grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n >= 250) print s, a, n, w, l, d }'
}
cmp() {  # A, B, rows file, label: A - B over the seeds both played
  awk -v A=$1 -v B=$2 -v lab="$4" '
    { if ($2 == A) { hA[$1] = 1; nA[$1] = $3; wA[$1] = $4; dA[$1] = $6 } if ($2 == B) { hB[$1] = 1; nB[$1] = $3; wB[$1] = $4; dB[$1] = $6 } }
    END {
      for (k in hA) if (hB[k]) { na += nA[k]; wa += wA[k]; da += dA[k]; nb += nB[k]; wb += wB[k]; db += dB[k]; seeds++ }
      if (na == 0 || nb == 0) { printf "  %s %-4s - %-4s: no shared seeds\n", lab, A, B; exit }
      sa = (wa + da / 2) / na; va = ((wa + da / 4) / na - sa * sa) / na
      sb = (wb + db / 2) / nb; vb = ((wb + db / 4) / nb - sb * sb) / nb
      d = sa - sb; se = sqrt(va + vb)
      printf "  %s %-4s - %-4s (%d seeds, %d / %d games): %.1f%% vs %.1f%%  %+.2f pp  ± %.2f (95%%)  z %+.2f\n", lab, A, B, seeds, na, nb, 100 * sa, 100 * sb, 100 * d, 196 * se, d / se
    }' $3
}
rows() {  # glob prefix, file
  local f b
  for f in $1; do [ -e "$f" ] || continue; b=$(basename "$f" .log); b=${b#g18_q*_s}; row ${b%%_*} ${b#*_} "$f"; done
}
{ rows "$O/g18_q189_s*_ctl.log"; rows "$O/g18_q189_s*_off.log"; rows "$O/g18_q190_s*_ctl.log"; rows "$O/g18_q190_s*_off.log"; } > /tmp/q192_sf.txt
{ rows "$O/g18_q190_s*_ctl.log"; rows "$O/g18_q190_s*_off.log"; } > /tmp/q192_sf_new.txt
echo "[q192] $(date) REGISTERED READ, W1 lazy on - off (level > -1 sigma; costs <= -2 sigma):"
{ cmp ctl off /tmp/q192_sf.txt "SF18 W1 (#189 + #190 / #192 fresh sets)"; cmp ctl off /tmp/q192_sf_new.txt "SF18 W1 (190111-190116 only)"; } | show
echo "[q192] per seed:"; sort /tmp/q192_sf.txt | show
echo "[q192] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q192] $(date) QUEUE 192 DONE"
