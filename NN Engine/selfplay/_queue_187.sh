#!/bin/bash
# 2026-11-08 night, queue #187 (owner 11-08 "sure", to use the idle box 05:00-10:00): the SF18 leg of #185 repeated on fresh
# openings, to answer the structure question: is the capture / null gate what keeps lazy eval level with off, and is KEEP harmful
# without it? Arms on build_uci/chess_engine_uci_lazy3: ctl (lazy off), gate (keep_gate), keep (KEEP, no gate), b1 (build 1).
# Ours at 250,000 nodes vs SF18 at 1,100 (node-limited, fastchess tc=inf), chunks of 250 games, concurrency 3. After queue #186.
# Five fresh seeds 187111-187115; every seed runs all four arms before the next seed starts (the order rotates by seed), so a
# partial run still compares arms on the same openings. No chunk starts after 09:45 on 11-09.
# Registered before it runs (fresh seeds only; #185's seeds generated the hypothesis):
#   THE GATE MATTERS if gate - keep >= +2 sigma on #187's seeds. Prediction: +2 to +4 pp.
#   KEEP WITHOUT THE GATE IS HARMFUL if keep - ctl <= -2 sigma on #187's seeds. Prediction: -1 to -4 pp, not resolved.
#   keep_gate stays level: gate - ctl above -1 sigma on #187's seeds, and on all shared seeds (#185 + #187). Prediction: within +-2 pp.
#   b1 is reported, no rule.
# Every comparison uses only the seeds both arms played (the 11-08 lesson: opening sets alone moved lazy off by 4.4 pp).
# Stop: touch /mnt/e/chess_data/q187.stop (checked while waiting and between chunks). Never edit this file while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
BIN="$ND/build_uci/chess_engine_uci_lazy3"
KG="LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GATE=3"
STOP=/mnt/e/chess_data/q187.stop
Q186=/mnt/e/chess_data/q186.log
LATEST=$(date -d '2026-11-09 09:45' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"
show() { sed "s/^/[q187]   /"; }
busy() { pgrep -f "fastchess-linux-x86-64/fastchess|vs_sf.py|_depth_residual_pass|_build_regret_set|_retune_fit|_ks_footprint_regret|sts_test|tactical_test|_theme_movetime|chess_engine_uci_lzprobe2" >/dev/null; }
kn() {
  case $1 in
    ctl) echo "" ;;
    gate) echo "$KG" ;;
    keep) echo "LAZY_V2=7,LAZY_V2_KEEP=1" ;;
    b1) echo "LAZY_V2=7" ;;
  esac
}
echo "[q187] $(date) waiting for queue 186's end line and a quiet box (stop: $STOP)"
until grep -qE 'QUEUE 186 (DONE|ABORTED)' "$Q186" 2>/dev/null && ! busy; do
  [ -e "$STOP" ] && { echo "[q187] $(date) stop file while waiting — QUEUE 187 DONE (nothing run)"; exit 0; }
  [ $(date +%s) -gt $LATEST ] && { echo "[q187] $(date) past 09:45 before starting — QUEUE 187 DONE (nothing run)"; exit 0; }
  sleep 60
done
sleep 30
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q187] $START start"
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$BIN" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
F0=$(fp ""); FG=$(fp "$KG")
echo "[q187] bench default: $F0 · keep_gate: $FG"
[ "$F0" = "solved 246/300 nodes 36509830" ] && [ "$FG" = "solved 247/300 nodes 39852210" ] || { echo "[q187] ☠️ fingerprints differ — QUEUE 187 ABORTED"; exit 1; }
COMMON="-each tc=inf -openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
T=g18_r187
chunk() {  # seed, arm
  local seed=$1 a=$2 k
  [ -e "$STOP" ] && return
  [ $(date +%s) -gt $LATEST ] && { echo "[q187] $(date) skip seed $seed $a (past 09:45)"; return; }
  k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
  nice -n 5 "$FC" -engine "cmd=$BIN" name=ours option.OwnBook=false $k nodes=250000 \
    -engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
    $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${T}_s${seed}_$a.pgn notation=san nodes=true append=false > $O/${T}_s${seed}_$a.log 2>&1
  echo "[q187] $(date) seed $seed $a: $(grep -E '^Games:' $O/${T}_s${seed}_$a.log | tail -1) · warnings $(grep -c 'have option' $O/${T}_s${seed}_$a.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $O/${T}_s${seed}_$a.log)"
}
rm -f $O/${T}_s*
for s in "187111 ctl gate keep b1" "187112 b1 keep gate ctl" "187113 gate ctl b1 keep" "187114 keep b1 ctl gate" "187115 ctl gate keep b1"; do
  set -- $s; seed=$1; shift
  for a in "$@"; do chunk $seed $a; done
done

# ── read: each pair of arms over the seeds both played ───────────────────────────────────────────────────────────────
# rows: source seed arm games wins losses draws (complete 250-game chunks only)
row() {  # source, seed, arm, log
  [ -e "$4" ] || return
  grep -E '^Games:' "$4" | tail -1 | awk -v src=$1 -v s=$2 -v a=$3 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n == 250) print src, s, a, n, w, l, d }'
}
{ for f in $O/${T}_s*_*.log; do b=$(basename $f .log); b=${b#${T}_s}; row new ${b%%_*} ${b#*_} $f; done
  for c in 1 2 3 4 5 6 7 8 9 10; do
    f=$(ls $O/g18_lzg_c${c}_*.log 2>/dev/null | head -1); [ -n "$f" ] || continue
    a=$(basename $f .log); a=${a#g18_lzg_c${c}_}
    seed=185111; [ $c -ge 5 ] && seed=185112; [ $c -ge 9 ] && seed=185113
    row old $seed $a $f
  done; } > /tmp/q187_rows.txt
echo "[q187] $(date) chunks read: $(grep -c '^new ' /tmp/q187_rows.txt) new, $(grep -c '^old ' /tmp/q187_rows.txt) from #185"
cmp() {  # A, B, scope (new | all): A - B over the seeds both played
  awk -v A=$1 -v B=$2 -v scope=$3 '
    scope == "new" && $1 != "new" { next }
    { key = $2; if ($3 == A) { has[key, "A"] = 1; nA[key] = $4; wA[key] = $5; dA[key] = $7 } if ($3 == B) { has[key, "B"] = 1; nB[key] = $4; wB[key] = $5; dB[key] = $7 } }
    END {
      for (k in nA) if (has[k, "B"]) { na += nA[k]; wa += wA[k]; da += dA[k]; nb += nB[k]; wb += wB[k]; db += dB[k]; seeds++ }
      if (na == 0 || nb == 0) { printf "  %-5s - %-5s (%s seeds): no shared seeds\n", A, B, scope; exit }
      sa = (wa + da / 2) / na; va = ((wa + da / 4) / na - sa * sa) / na
      sb = (wb + db / 2) / nb; vb = ((wb + db / 4) / nb - sb * sb) / nb
      d = sa - sb; se = sqrt(va + vb)
      printf "  %-5s - %-5s (%s seeds, %d shared, %d / %d games): %.1f%% vs %.1f%%  %+.2f pp  ± %.2f (95%%)  z %+.2f\n", A, B, scope, seeds, na, nb, 100 * sa, 100 * sb, 100 * d, 196 * se, d / se
    }' /tmp/q187_rows.txt
}
echo "[q187] REGISTERED READS (fresh seeds): the gate matters if gate - keep >= +2 sigma; keep harmful if keep - ctl <= -2 sigma; keep_gate level if gate - ctl > -1 sigma"
{ cmp gate keep new; cmp keep ctl new; cmp gate ctl new; cmp b1 ctl new; cmp gate b1 new; } | show
echo "[q187] with #185's shared seeds (secondary):"
{ cmp gate keep all; cmp keep ctl all; cmp gate ctl all; cmp b1 ctl all; cmp gate b1 all; } | show
echo "[q187] $(date) QUEUE 187 DONE"
