#!/bin/bash
# 2026-11-18 ~11:00, queue #271: the SORT_KEYED speed build of gen 7.6 (owner 11-18: "if its a pure speed upgrade we can go
# for it"). SORT_KEYED (commits 363d797 / c91b34c) sorts the move-ordering scores on unique 64-bit keys with a compare-exchange
# network and branch-free merges: exactly the stable sort's order, so the moves and fingerprints are unchanged. #268 (quiet box,
# 15 rounds): skon / g76 time 0.982 (faster in 12 of 15).
#  1 Production must be g76pgo; nothing else running.
#  2 Shipped preset += SORT_KEYED 1 (search_engine.cpp, after SV2_VERIFY_PROMOTE 1); the id stays "AlphaUX (v2 gen 7.6)".
#    PGO build chess_engine_uci_g76spgo (profile /mnt/e/chess_data/pgo_g76s). GATES: default bench 246 / 35,138,434 ==
#    SORT_KEYED=0 == g76pgo; SCRATCH_REUSE=2 (every list checked against std::stable_sort, abort on a difference) == default;
#    SV2_VERIFY_PROMOTE=0 = gen 7.5 252 / 35,235,460; v1 250 / 35,310,775-778; the id; game12 d10 default == SORT_KEYED=0 == g76pgo.
#  3 Pinned bench x15 on core 5, g76s / g76 in alternating order. REGISTERED: install only if the within-round time median
#    is < 0.995; prediction 0.975 … 0.990 (#268: 0.982). Otherwise the sources are restored and production stays g76pgo.
#  4 INSTALL (backup = g76pgo), Cython rebuild (.so backed up to /mnt/e/chess_data/so_backup_g76s), WAC fingerprints.
# Stop: touch /mnt/e/chess_data/q271.stop (between steps). Never edit this file, the binaries or the sources while it runs; the
# sources are free after "Cython build done" (or after "sources restored").
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q271
P=/mnt/e/chess_data/pgo_g76s
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G76="$ND/build_uci/chess_engine_uci_g76pgo"
NEW="$ND/build_uci/chess_engine_uci_g76spgo"
BK=/mnt/e/chess_data/so_backup_g76s
LOG=/mnt/e/chess_data/q271.log
STOP=/mnt/e/chess_data/q271.stop
SE="$ND/search_engine.cpp"
UM="$ND/uci_main.cpp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q271]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q271] $(date) stop file before $1 — QUEUE 271 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }

# ── 1 the machine and production as expected ─────────────────────────────────────────────────────────────────────────────
echo "[q271] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" "$G76" || { echo "[q271] ☠️ production is not g76pgo — QUEUE 271 ABORTED (nothing changed)"; exit 1; }
busy && { echo "[q271] ☠️ another job is running — QUEUE 271 ABORTED (nothing changed)"; exit 1; }

# ── 2 the preset, build, gates ───────────────────────────────────────────────────────────────────────────────────────────
[ "$(grep -c '^ *Config::SV2_VERIFY_PROMOTE = 1;$' "$SE")" = 1 ] && [ "$(grep -c 'Config::SORT_KEYED = 1;' "$SE")" = 0 ] \
	&& [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.6)");' "$UM")" = 1 ] || { echo "[q271] ☠️ the sources are not as expected — QUEUE 271 ABORTED (nothing changed)"; exit 1; }
cp "$SE" "$W/search_engine.cpp.g76" || { echo "[q271] ☠️ source backup failed — QUEUE 271 ABORTED"; exit 1; }
restore_src() { cp "$W/search_engine.cpp.g76" "$SE"; echo "[q271] sources restored to gen 7.6 without SORT_KEYED — the sources are free to edit"; }
cat > "$W/preset.txt" <<'EOF'
                    // Move-ordering sort on unique 64-bit keys (score, then build order) with a compare-exchange network and
                    // branch-free merges: exactly the stable sort's order, so the same moves; about 10% fewer branch
                    // mispredictions and 1.8% less bench time (BASELINE_PERF). Set before the start-up read, so SORT_KEYED=0
                    // restores the previous sort.
                    Config::SORT_KEYED = 1;
EOF
sed -i "/^ *Config::SV2_VERIFY_PROMOTE = 1;\$/r $W/preset.txt" "$SE"
[ "$(grep -c '^ *Config::SORT_KEYED = 1;$' "$SE")" = 1 ] || { echo "[q271] ☠️ the preset edit did not apply — QUEUE 271 ABORTED"; restore_src; exit 1; }
echo "[q271] $(date '+%H:%M') preset written (search_engine.cpp +5 lines after SV2_VERIFY_PROMOTE 1); PGO build"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_g76sgen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g76sgen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g76sgen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 771 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_g76spgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q271] ☠️ PGO use build failed — QUEUE 271 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; restore_src; exit 1; }
else
	echo "[q271] ☠️ PGO gen build failed — QUEUE 271 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; restore_src; exit 1
fi
echo "[q271] $(date '+%H:%M') UCI build done; training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
ok=1
a=$(fp "$NEW" ""); s0=$(fp "$NEW" "SORT_KEYED=0"); sr=$(fp "$NEW" "SCRATCH_REUSE=2"); b=$(fp "$G76" ""); off=$(fp "$NEW" "SV2_VERIFY_PROMOTE=0")
echo "[q271] g76spgo default [$a] (want 246 / 35138434) · SORT_KEYED=0 [$s0] · SCRATCH_REUSE=2 check [$sr] · g76pgo [$b] (all == default)"
echo "[q271] g76spgo SV2_VERIFY_PROMOTE=0 [$off] (= gen 7.5, 252 / 35235460)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$s0" = "$a" ] && [ "$sr" = "$a" ] && [ "$b" = "$a" ] && [ "$off" = "solved 252/300 nodes 35235460" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ID=$(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep -m1 '^id name')
echo "[q271] v1 [$v1] (250 / 35310775-778) · $ID"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
echo "$ID" | grep -qF 'AlphaUX (v2 gen 7.6)' || ok=0
ga=$(gfp "$NEW" ""); g0=$(gfp "$NEW" "SORT_KEYED=0"); gr=$(gfp "$NEW" "SCRATCH_REUSE=2"); gb=$(gfp "$G76" "")
echo "[q271] game12 d10: default [$ga] · SORT_KEYED=0 [$g0] · SCRATCH_REUSE=2 [$gr] · g76pgo [$gb]"
[ -n "$ga" ] && [ "$g0" = "$ga" ] && [ "$gr" = "$ga" ] && [ "$gb" = "$ga" ] || ok=0
[ $ok = 1 ] || { echo "[q271] ☠️ a gate failed — no install — QUEUE 271 ABORTED"; restore_src; exit 1; }
echo "[q271] ✅ every gate equal"

# ── 3 pinned bench ───────────────────────────────────────────────────────────────────────────────────────────────────────
bench_one() {  # round, arm
	local out b
	case $2 in g76s) b="$NEW" ;; g76) b="$G76" ;; esac
	out=$(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')
	echo "[q271]   r$1 $2 $out"
}
bt() { grep -E "^\[q271\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q271\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
busy && echo "[q271] ☠️ another job is running — the pinned bench below is disturbed"
echo "[q271] $(date '+%H:%M') == pinned bench x15 on core 5, g76s / g76 in alternating order"
for i in $(seq 1 15); do
	stopped "bench round $i"
	if [ $((i % 2)) = 1 ]; then o="g76s g76"; else o="g76 g76s"; fi
	for x in $o; do bench_one $i $x; done
done
for x in g76s g76; do echo "[q271] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
RT=$(join <(bt g76s) <(bt g76) | awk '{print $2 / $3}' | med)
NF=$(join <(bt g76s) <(bt g76) | awk '$2 < $3 {n++} END {print n + 0}')
echo "[q271] within-round g76s / g76: time median $RT (g76s faster in $NF of 15) · nps median $(join <(bn g76s) <(bn g76) | awk '{print $2 / $3}' | med)"
echo "[q271] registered: install if the time median < 0.995; prediction 0.975 … 0.990"
awk -v r="$RT" 'BEGIN { exit !(r != "" && r < 0.995) }' || { echo "[q271] ✋ not faster by the registered rule — no install, production stays g76pgo — QUEUE 271 DONE"; restore_src; exit 0; }

# ── 4 install, Cython ────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the install"
cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" || { echo "[q271] ☠️ install copy failed — restoring g76pgo"; cp "$G76" "$PROD"; restore_src; exit 1; }
echo "[q271] $(date '+%H:%M') INSTALLED: production == g76spgo: bench [$(fp "$PROD" "")] · $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name')"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q271] previous .so backed up to $BK" || { echo "[q271] ☠️ .so backup failed — Cython not rebuilt (UCI is the speed build)"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q271] ☠️ Cython build left no .so — restoring the previous .so"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac q271_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac q271_fp_s0 V2_PRESET=shipped SORT_KEYED=0 > "$W/cy_s0.txt" 2>&1
bash "$R" wac q271_fp_v1 > "$W/cy_v1.txt" 2>&1
echo "[q271] $(date '+%H:%M') Cython build done — the sources are free to edit"
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
echo "[q271] Cython shipped [$(cyr "$W/cy_default.txt")] (want 246 / 35138434 / EBF 3.698) · SORT_KEYED=0 [$(cyr "$W/cy_s0.txt")] (same) · v1 [$(cyr "$W/cy_v1.txt")] (250 / 35310778)"
grep -q 'Solved 246/300' "$W/cy_default.txt" && grep -q '^NODES: 35138434' "$W/cy_default.txt" || echo "[q271] ☠️ the Cython module's shipped fingerprint differs from the UCI binary's — tell the owner (games use the UCI binary)"
echo "[q271] $(date '+%H:%M') GEN 7.6 SPEED BUILD (SORT_KEYED) SHIPPED"
echo "[q271] $(date) QUEUE 271 DONE"
