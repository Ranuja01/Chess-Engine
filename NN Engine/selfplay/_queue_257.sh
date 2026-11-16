#!/bin/bash
# 2026-11-16 ~13:50, queue #257: GEN 7.5 = the eval speed preset + c2c (the two halves of the search made symmetric), then
# PROMOTE re-tested on gen 7.5. Replaces #256, which stopped at its step 1 because #255 did not install the speed build.
# OWNER 11-16 ~13:45 ("Ship both now"): #255's pinned bench was disturbed by outside load (6 of 36 runs 25-40% slow, on every
# arm; default / g74 median 0.972 but faster in 5 of 9, below the registered 7 of 9), while #252's quiet bench read -4.7% (9 of
# 9) and cachegrind -9.6% instructions. So gen 7.5 ships with the speed preset already in the sources (EVAL_MAT_CACHE 1 +
# EVAL_FAST 31, result-identical) and a three-arm pinned bench (g75 / g74e / g74) goes in the ledger without gating the install.
# OWNER 11-16: ship c2c "as long as the mediocre numbers make sense". #252 (vs gen 7.4, 2,000 g per arm, 1,000 pairs): c2c =
# FIX_RESEARCH=1,FIX_MIN_SCOUT=1,FIX_MIN_RD1=1 +76.4 (95% +64.4 … +88.6); SF18 at equal nodes, 6 seeds: +9.63 pp (z +6.07);
# Mediocre 10+0.1, 2 seeds: +7.00 pp (z +2.84). c2a (FIX_RESEARCH alone) +41.4 / +2.8 pp / +4.5 pp.
#  0 c2c's registered reads from #252 (pass at 13:19; #253 already stopped by #256).
#  1 Production must still be g74pgo, #253 and #255 done, nothing else running.
#  2 Shipped preset += FIX_RESEARCH 1, FIX_MIN_SCOUT 1, FIX_MIN_RD1 1 (search_engine.cpp, after EVAL_FAST 31); id "AlphaUX (v2
#    gen 7.5)". PGO build chess_engine_uci_g75pgo (profile /mnt/e/chess_data/pgo_g75). GATES: default bench 252 / 35,235,460
#    (= c2c's knobs on g74pgo) == the c2c knobs on g75pgo == the c2c knobs on g74epgo; FIX_RESEARCH=0,FIX_MIN_SCOUT=0,FIX_MIN_RD1=0
#    = gen 7.4 250 / 37,320,382; the eval verify mode EVAL_MAT_CACHE=2 == default; v1 250 / 35,310,775-778; the id; game12 d10
#    verify == default.
#  3 INSTALL (backup = g74pgo, gen 7.4 as it played; g74epgo = gen 7.4 + the speed preset), Cython rebuild (backup
#    /mnt/e/chess_data/so_backup_g75) with its WAC fingerprint and EBF, then a pinned bench x9 on core 5, three arms in rotating
#    order: g75 (shipped), g74e (the speed build alone), g74 (gen 7.4 production). Ledger only. g75 searches 5.6% fewer nodes on
#    the bench than the other two, so its time ratio is time to the same depth and its NPS ratio is the per-node speed.
#  4 PROMOTE on gen 7.5 (SV2_VERIFY_PROMOTE=1 against the gen 7.5 default), as #252: timed self-play 10+0.1 in 150-pair chunks,
#    seeds from 2571, to 2,000 games (no timed work 20:50 to 00:15 unless q257.go; q257.pause holds; cutoff Tue 20:50); SF18 at
#    equal nodes (ours 250k, SF18 1,100) ctl + prom, seeds 257111-257116, in the owner's evening; the rest of the self-play;
#    Mediocre 10+0.1 ctl + prom, seeds 257311-257312.
# REGISTERED before any game (ship rule): self-play lower 95% bound > 0 = pass, within ±2σ = NEUTRAL; SF18 prom − ctl z > −1;
# Mediocre prom − ctl >= 0 pp. Prediction −5 to +10: PROMOTE re-searches at full depth a shallower verify that crosses a bound,
# and A1 now re-searches at full depth every reduced move that beats the bound, so much of what PROMOTE caught on gen 7.4
# (+13.6) is already caught.
# Stop: touch /mnt/e/chess_data/q257.stop (between steps). Never edit this file, the binaries or the sources while it runs; the
# sources are free after "Cython build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
MED="$ND/selfplay/mediocre_uci.sh"
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q257
GW=/mnt/e/chess_data/r16b
P=/mnt/e/chess_data/pgo_g75
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G74="$ND/build_uci/chess_engine_uci_g74pgo"
G74E="$ND/build_uci/chess_engine_uci_g74epgo"
NEW="$ND/build_uci/chess_engine_uci_g75pgo"
C2C="FIX_RESEARCH=1,FIX_MIN_SCOUT=1,FIX_MIN_RD1=1"
C2OFF="FIX_RESEARCH=0,FIX_MIN_SCOUT=0,FIX_MIN_RD1=0"
KP="SV2_VERIFY_PROMOTE=1"
BK=/mnt/e/chess_data/so_backup_g75
LOG=/mnt/e/chess_data/q257.log
Q252=/mnt/e/chess_data/q252.log
Q255=/mnt/e/chess_data/q255.log
STOP=/mnt/e/chess_data/q257.stop
PAUSE=/mnt/e/chess_data/q257.pause
GO=/mnt/e/chess_data/q257.go
DUMPS=/mnt/c/Users/Kumodth/AppData/Local/Temp/wsl-crashes
SE="$ND/search_engine.cpp"
UM="$ND/uci_main.cpp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$PAUSE" "$GO"; mkdir -p "$W" "$GW"
show() { sed "s/^/[q257]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q257] $(date) stop file before $1 — QUEUE 257 DONE"; exit 0; }; }
kn() { case $1 in ctl) echo "" ;; prom) echo "$KP" ;; esac; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
timed_ok() { local t; t=$(date +%s); [ ! -e "$PAUSE" ] && [ "$t" -lt "$CUT" ] && ! { [ "$t" -ge "$GS" ] && [ "$t" -lt "$GE" ] && [ ! -e "$GO" ]; }; }
guard() { ( until ! timed_ok; do sleep 15; done; pkill -INT -f "[f]astchess-linux-x86-64/fastchess" && echo "[q257] $(date '+%H:%M') guard: games interrupted" ) & GUARD=$!; }
finished() { local c; c=$(grep -cE '^\[Result "(1-0|0-1|1/2-1/2)"\]' "$1" 2>/dev/null); echo "${c:-0}"; }
sp_count() { local n=0 f; for f in "$GW/$1"_s*.pgn; do [ -e "$f" ] && n=$((n + $(finished "$f"))); done; echo $n; }

# ── 0 c2c's registered reads from #252 ────────────────────────────────────────────────────────────────────────────────────
echo "[q257] $(date '+%Y-%m-%d %H:%M:%S') start"
grep -qE 'QUEUE 252 DONE' "$Q252" || { echo "[q257] ☠️ #252 has not finished — QUEUE 257 ABORTED"; exit 1; }
MLINE=$(grep -E 'Mediocre c2c +- ctl' "$Q252" | tail -1)
SLINE=$(grep -E 'SF18 c2c +- ctl' "$Q252" | tail -1)
MS=$(echo "$MLINE" | grep -oE '\([0-9]+ seeds' | grep -oE '[0-9]+')
MPP=$(echo "$MLINE" | grep -oE '[+-][0-9]+\.[0-9]+ pp' | head -1 | grep -oE '^[+-][0-9.]+')
SPP=$(echo "$SLINE" | grep -oE '[+-][0-9]+\.[0-9]+ pp' | head -1 | grep -oE '^[+-][0-9.]+')
SZ=$(echo "$SLINE" | grep -oE 'z [+-][0-9.]+' | grep -oE '[+-][0-9.]+')
echo "[q257] #252 reads for c2c:"; { echo "$SLINE"; echo "$MLINE"; } | sed 's/^\[q252\] *//' | show
if [ "$MS" = 2 ] && awk -v p="$MPP" -v z="$SZ" 'BEGIN { exit !(p != "" && z != "" && p + 0 >= 0 && z + 0 > -1) }'; then
	echo "[q257] c2c passes every registered read (Mediocre ${MPP} pp over 2 seeds, SF18 ${SPP} pp z ${SZ})"
else
	echo "[q257] ☠️ c2c's reads do not all pass or are incomplete (Mediocre seeds [$MS] [$MPP] pp, SF18 z [$SZ]) — no ship — QUEUE 257 DONE"
	exit 0
fi

# ── 1 the machine and production as expected ─────────────────────────────────────────────────────────────────────────────
grep -qE 'QUEUE 253 (DONE|ABORTED)' /mnt/e/chess_data/q253.log && grep -qE 'QUEUE 255 (DONE|ABORTED)' "$Q255" \
	&& grep -qE 'QUEUE 256' /mnt/e/chess_data/q256.log || { echo "[q257] ☠️ #253, #255 or #256 has not finished — QUEUE 257 ABORTED"; exit 1; }
cmp -s "$PROD" "$G74" || { echo "[q257] ☠️ production is not g74pgo — QUEUE 257 ABORTED (nothing changed)"; exit 1; }
[ -x "$G74E" ] || { echo "[q257] ☠️ g74epgo is missing — QUEUE 257 ABORTED"; exit 1; }
stopped "the build"
busy && { echo "[q257] ☠️ another job is running — QUEUE 257 ABORTED (nothing changed)"; exit 1; }

# ── 2 the gen 7.5 preset, build, gates ────────────────────────────────────────────────────────────────────────────────────
[ "$(grep -c '^ *Config::EVAL_FAST = 31;$' "$SE")" = 1 ] && [ "$(grep -c 'Config::FIX_RESEARCH = 1;' "$SE")" = 0 ] \
	&& [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.4)");' "$UM")" = 1 ] || { echo "[q257] ☠️ the sources are not as expected — QUEUE 257 ABORTED (nothing changed)"; exit 1; }
cp "$SE" "$W/search_engine.cpp.g74e" && cp "$UM" "$W/uci_main.cpp.g74e" || { echo "[q257] ☠️ source backup failed — QUEUE 257 ABORTED"; exit 1; }
restore_src() { cp "$W/search_engine.cpp.g74e" "$SE"; cp "$W/uci_main.cpp.g74e" "$UM"; echo "[q257] sources restored to gen 7.4 + the speed preset"; }
cat > "$W/preset.txt" <<EOF
                    // ── GEN 7.5: the two halves of the search made symmetric. A reduced move that beats the node's bound is
                    // re-searched at full depth on both sides (A1), the minimizer scouts at its own bound (A2), and the
                    // minimizer-only full window at remaining depth 1 is dropped (A3). Self-play +76.4 (95% +64.4 … +88.6) over
                    // 2,000 games, SF18 at equal nodes ${SPP} pp (z ${SZ}), Mediocre ${MPP} pp (OPTIMIZATION_LOG). Set before the
                    // start-up read, so FIX_RESEARCH=0,FIX_MIN_SCOUT=0,FIX_MIN_RD1=0 restores gen 7.4 (250 / 37,320,382).
                    Config::FIX_RESEARCH = 1;
                    Config::FIX_MIN_SCOUT = 1;
                    Config::FIX_MIN_RD1 = 1;
EOF
sed -i "/^ *Config::EVAL_FAST = 31;\$/r $W/preset.txt" "$SE"
sed -i 's/say("id name AlphaUX (v2 gen 7\.4)");/say("id name AlphaUX (v2 gen 7.5)");/' "$UM"
[ "$(grep -c '^ *Config::FIX_RESEARCH = 1;$' "$SE")" = 1 ] && [ "$(grep -c '^ *Config::FIX_MIN_RD1 = 1;$' "$SE")" = 1 ] \
	&& [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.5)");' "$UM")" = 1 ] || { echo "[q257] ☠️ the preset edit did not apply — QUEUE 257 ABORTED"; restore_src; exit 1; }
echo "[q257] $(date '+%H:%M') preset written (search_engine.cpp +8 lines after EVAL_FAST 31, id gen 7.5); PGO build"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_g75gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g75gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g75gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 756 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_g75pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q257] ☠️ PGO use build failed — QUEUE 257 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; restore_src; exit 1; }
else
	echo "[q257] ☠️ PGO gen build failed — QUEUE 257 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; restore_src; exit 1
fi
echo "[q257] $(date '+%H:%M') UCI build done; training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
ok=1
a=$(fp "$NEW" ""); k=$(fp "$NEW" "$C2C"); e=$(fp "$G74E" "$C2C"); off=$(fp "$NEW" "$C2OFF"); v=$(fp "$NEW" "EVAL_MAT_CACHE=2")
echo "[q257] g75pgo default [$a] (want 252 / 35235460) · c2c knobs [$k] · g74epgo + c2c knobs [$e] (both == default)"
echo "[q257] g75pgo three off [$off] (= gen 7.4, 250 / 37320382) · eval verify EVAL_MAT_CACHE=2 [$v] (== default)"
[ "$a" = "solved 252/300 nodes 35235460" ] && [ "$k" = "$a" ] && [ "$e" = "$a" ] && [ "$v" = "$a" ] && [ "$off" = "solved 250/300 nodes 37320382" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ID=$(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep -m1 '^id name')
echo "[q257] v1 [$v1] (250 / 35310775-778) · $ID"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
echo "$ID" | grep -qF 'AlphaUX (v2 gen 7.5)' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2\nbench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q257] game12 d10: verify [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
[ $ok = 1 ] || { echo "[q257] ☠️ a gate failed — no install — QUEUE 257 ABORTED"; restore_src; exit 1; }

# ── 3 install, Cython, pinned bench ───────────────────────────────────────────────────────────────────────────────────────
stopped "the install"
cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" || { echo "[q257] ☠️ install copy failed — restoring g74pgo"; cp "$G74" "$PROD"; restore_src; exit 1; }
echo "[q257] $(date '+%H:%M') INSTALLED: production == g75pgo: bench [$(fp "$PROD" "")] · $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name')"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q257] .so backed up to $BK" || { echo "[q257] ☠️ .so backup failed — Cython not rebuilt (UCI is gen 7.5)"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q257] ☠️ Cython build left no .so — restoring the gen 7.4 .so"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac q257_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac q257_fp_74 V2_PRESET=shipped FIX_RESEARCH=0 FIX_MIN_SCOUT=0 FIX_MIN_RD1=0 > "$W/cy_74.txt" 2>&1
bash "$R" wac q257_fp_v1 > "$W/cy_v1.txt" 2>&1
echo "[q257] $(date '+%H:%M') Cython build done — the sources are free to edit"
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
echo "[q257] Cython shipped [$(cyr "$W/cy_default.txt")] (want 252 / 35235460) · three off [$(cyr "$W/cy_74.txt")] (gen 7.4 250 / 37320382 / EBF 3.743) · v1 [$(cyr "$W/cy_v1.txt")] (250 / 35310778)"
grep -q 'Solved 252/300' "$W/cy_default.txt" && grep -q '^NODES: 35235460' "$W/cy_default.txt" || echo "[q257] ☠️ the Cython module's shipped fingerprint differs from the UCI binary's — tell the owner (games use the UCI binary)"
bench_one() {  # round, arm
	local out b
	case $2 in g75) b="$NEW" ;; g74e) b="$G74E" ;; g74) b="$G74" ;; esac
	out=$(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')
	echo "[q257]   r$1 $2 $out"
}
bt() { grep -E "^\[q257\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q257\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
busy && echo "[q257] ☠️ another job is running — the pinned bench below is disturbed"
echo "[q257] $(date '+%H:%M') == pinned bench x9 on core 5, g75 / g74e / g74 in rotating order (ledger)"
for i in 1 2 3 4 5 6 7 8 9; do
	stopped "bench round $i"
	case $((i % 3)) in 1) o="g75 g74e g74" ;; 2) o="g74e g74 g75" ;; 0) o="g74 g75 g74e" ;; esac
	for x in $o; do bench_one $i $x; done
done
for x in g75 g74e g74; do echo "[q257] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
pair() {  # A, B: within-round time and nps ratios A / B
	echo "[q257] within-round $1 / $2: time median $(join <(bt $1) <(bt $2) | awk '{print $2 / $3}' | med) ($1 faster in $(join <(bt $1) <(bt $2) | awk '$2 < $3 {n++} END {print n + 0}') of 9) · nps median $(join <(bn $1) <(bn $2) | awk '{print $2 / $3}' | med)"
}
pair g75 g74; pair g74e g74; pair g75 g74e
echo "[q257] $(date '+%H:%M') GEN 7.5 SHIPPED"

# ── 4 PROMOTE on gen 7.5 ──────────────────────────────────────────────────────────────────────────────────────────────────
START=$(date '+%Y-%m-%d %H:%M:%S')
if [ "$(date +%H)" -ge 12 ]; then
	GS=$(date -d 'today 20:50' +%s); GE=$(date -d "$(date -d tomorrow +%F) 00:15" +%s); CUT=$(date -d "$(date -d tomorrow +%F) 20:50" +%s)
else
	GS=$(date +%s); GE=$GS; CUT=$(date -d 'today 20:50' +%s)
fi
echo "[q257] $START PROMOTE on gen 7.5 (no timed work $(date -d @$GS '+%a %H:%M') to $(date -d @$GE '+%a %H:%M') unless q257.go; cutoff $(date -d @$CUT '+%a %H:%M'))"
F0=$(fp "$PROD" ""); FPR=$(fp "$PROD" "$KP")
echo "[q257] gen 7.5 [$F0] · prom [$FPR] (must differ)"
[ -n "$FPR" ] && [ "$FPR" != "$F0" ] || { echo "[q257] ☠️ SV2_VERIFY_PROMOTE does not move the gen 7.5 bench — no PROMOTE games — QUEUE 257 DONE"; exit 1; }
rm -f /tmp/q257_pre.engine.log
timeout 180 "$FC" -engine "cmd=$PROD" name=prom option.OwnBook=false "option.Knobs=$KP" st=0.05 \
	-engine "cmd=$PROD" name=ctl option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 \
	-log file=/tmp/q257_pre.engine.log level=info engine=true > /tmp/q257_pre.log 2>&1
WN=$(grep -c 'have option' /tmp/q257_pre.log)
echo "[q257] preflight: $(grep -E '^Games:' /tmp/q257_pre.log | tail -1) · warnings $WN · echoes $KP $(grep -cF "$KP" /tmp/q257_pre.engine.log)"
[ "$WN" = 0 ] || { echo "[q257] ☠️ preflight warnings — no PROMOTE games — QUEUE 257 DONE"; exit 1; }
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
SEQ=0
sp_chunk() {  # arm; returns 1 when the arm already has 2,000
	local a=$1 n left pairs seed f
	n=$(sp_count $a); left=$(( (2000 - n + 1) / 2 )); [ $left -le 0 ] && return 1
	pairs=$(( left < 150 ? left : 150 )); SEQ=$((SEQ + 1)); seed=$((2570 + SEQ)); f=$GW/${a}_s$seed
	echo "[q257] $(date '+%H:%M') self-play $a: $n finished, chunk seed $seed, $pairs pairs"
	guard
	nice -n 5 "$FC" -engine "cmd=$PROD" name=$a option.OwnBook=false "option.Knobs=$(kn $a)" tc=10+0.1 timemargin=250 \
		-engine "cmd=$PROD" name=g75 option.OwnBook=false tc=10+0.1 timemargin=250 \
		$COMMON -srand $seed -rounds $pairs -games 2 -pgnout file=$f.pgn notation=san nodes=true append=false > $f.log 2>&1
	kill $GUARD 2>/dev/null
	echo "[q257] $(date '+%H:%M') $a s$seed: $(grep -E '^Games:' $f.log | tail -1) · warnings $(grep -c 'have option' $f.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $f.log) · time forfeits $(grep -c 'loses on time' $f.pgn) · unfinished $(grep -c '^\[Result "\*"' $f.pgn)"
	return 0
}
sp_loop() { while timed_ok; do stopped "a self-play chunk"; sp_chunk prom || break; done; }
pooled() {
	[ "$(sp_count prom)" -gt 0 ] || return
	echo "[q257] pooled self-play prom vs gen 7.5 ($(sp_count prom) finished):"
	timeout 900 bash "$R" pyrun selfplay/fastchess_report.py "$GW/prom"_s*.pgn --p1 prom 2>&1 | sed -n '/vs the field/,$p' | show
}
sp_loop
pooled

rm -f $O/g18_q257_s* $O/med_q257_s*
sf_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/g18_q257_s$1_$2.log
	stopped "SF18 $seed $a"
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false $k nodes=250000 \
		-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100 \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/g18_q257_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[q257] $(date '+%H:%M') SF18 seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f")"
}
echo "[q257] $(date '+%H:%M') SF18 legs (node-limited)"
for s in 1 2 3 4 5 6; do
	if [ $((s % 2)) = 1 ]; then o="ctl prom"; else o="prom ctl"; fi
	for x in $o; do sf_chunk $((257110 + s)) $x; done
done

until timed_ok || [ "$(date +%s)" -ge "$CUT" ]; do stopped "the rest of the self-play"; sleep 30; done
sp_loop
pooled

med_chunk() {  # seed, arm
	local seed=$1 a=$2 k f=$O/med_q257_s$1_$2.log
	stopped "Mediocre $seed $a"
	timed_ok || { echo "[q257] $(date) skip Mediocre $seed $a (cutoff or pause)"; return; }
	k=$(kn $a); [ -n "$k" ] && k="option.Knobs=$k"
	guard
	nice -n 5 "$FC" -engine "cmd=$PROD" name=ours option.OwnBook=false $k tc=10+0.1 timemargin=250 \
		-engine "cmd=$MED" name=mediocre tc=10+0.1 timemargin=1000 \
		$COMMON -srand $seed -rounds 75 -games 2 -pgnout file=$O/med_q257_s${seed}_$a.pgn notation=san append=false > "$f" 2>&1
	kill $GUARD 2>/dev/null
	echo "[q257] $(date '+%H:%M') Mediocre seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · time losses $(grep -c 'loses on time' "$f")"
}
for x in ctl prom; do med_chunk 257311 $x; done
for x in prom ctl; do med_chunk 257312 $x; done

# ── reads ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
row() {  # seed, arm, log: complete chunks only
	grep -E '^Games:' "$3" 2>/dev/null | tail -1 | awk -v s=$1 -v a=$2 '{ gsub(/,/, ""); for (i = 1; i <= NF; i++) { if ($i == "Games:") n = $(i+1); if ($i == "Wins:") w = $(i+1); if ($i == "Losses:") l = $(i+1); if ($i == "Draws:") d = $(i+1) } if (n > 0) print s, a, n, w, l, d }'
}
cmpr() {  # A, B, rows file, label: A - B over the seeds both played
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
for f in $O/g18_q257_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#g18_q257_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q257_sf.txt
for f in $O/med_q257_s*_*.log; do [ -e "$f" ] || continue; b=$(basename $f .log); b=${b#med_q257_s}; row ${b%%_*} ${b#*_} $f; done > /tmp/q257_med.txt
echo "[q257] $(date) per chunk (seed arm games W L D):"; { sed 's/^/Mediocre /' /tmp/q257_med.txt; sed 's/^/SF18 /' /tmp/q257_sf.txt; } | show
echo "[q257] REGISTERED READS (SF18: prom - ctl z > -1 = pass; Mediocre: prom - ctl >= 0 = pass):"
{ cmpr prom ctl /tmp/q257_sf.txt SF18; cmpr prom ctl /tmp/q257_med.txt Mediocre; } | show
echo "[q257] new WSL crash dumps since $START:"
find "$DUMPS" -newermt "$START" -name '*.dmp' 2>/dev/null | show
echo "[q257] $(date) QUEUE 257 DONE"
