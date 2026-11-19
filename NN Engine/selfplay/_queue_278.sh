#!/bin/bash
# 2026-11-19 ~10:40, queue #278: GEN 7.7 = gen 7.6 + the qsearch stand-pat seed (ENABLE_QSTANDPAT_SEED) + SORT_KEYED.
# OWNER 11-19: "i reckon we can ship and continue on" (and Mediocre retired as the third judge, SF1.1 replaces it).
# #274 on gen 7.6: self-play 2,000 g +21.9 (95% +10.4 … +33.5); SF18 at equal nodes +3.50 pp (z +2.26) over 1,500 g per arm;
# Mediocre −0.50 pp (z −0.27) over 300 g per arm. #275: SORT_KEYED pinned bench x21 time median 0.989 (faster in 19 of 21).
# The sources were edited before this queue (nothing was running): the shipped preset sets ENABLE_QSTANDPAT_SEED and
# SORT_KEYED after SV2_VERIFY_PROMOTE 1, the seed is read again after the preset, the id reads gen 7.7, and uci_main.cpp's
# ensure_init keeps the shipped preset under an explicit `setoption name EvalArm value 1` (it used to give the bare skeleton).
#  1 Production must be g76pgo; nothing else running; the sources as described.
#  2 PGO build chess_engine_uci_g77pgo (profile /mnt/e/chess_data/pgo_g77). GATES: default bench 254 / 35,922,538 == the seed
#    knob on g76pgo (#274's number) == SORT_KEYED=0 == SCRATCH_REUSE=2 (every sorted list checked against std::stable_sort)
#    == EVAL_MAT_CACHE=2 (eval verify) == explicit EvalArm 1; ENABLE_QSTANDPAT_SEED=0 = gen 7.6 246 / 35,138,434;
#    + SV2_VERIFY_PROMOTE=0 = gen 7.5 252 / 35,235,460; v1 250 / 35,310,775-778; the id; game12 d10 default == g76pgo + seed,
#    seed off == g76pgo. EvalArm 1 on g76pgo is recorded (the old trap, expected 238 / 66,189,654).
#  3 INSTALL (backup = g76pgo), Cython rebuild (.so backed up to /mnt/e/chess_data/so_backup_g77) with WAC fingerprints and
#    EBF, then a pinned bench x11 on core 5, g77 / g76 alternating (ledger: time AND nps, the node counts differ by +2.2%).
# Stop: touch /mnt/e/chess_data/q278.stop (between steps). Never edit this file, the binaries or the sources while it runs; the
# sources are free after "Cython build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q278
P=/mnt/e/chess_data/pgo_g77
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G76="$ND/build_uci/chess_engine_uci_g76pgo"
NEW="$ND/build_uci/chess_engine_uci_g77pgo"
BK=/mnt/e/chess_data/so_backup_g77
LOG=/mnt/e/chess_data/q278.log
STOP=/mnt/e/chess_data/q278.stop
GO=/mnt/e/chess_data/q278.go
SE="$ND/search_engine.cpp"
UM="$ND/uci_main.cpp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q278]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q278] $(date) stop file before $1 — QUEUE 278 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
gfp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf "%s\nbench $G 10\nquit\n" "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
evening() { local n; n=$((10#$(date +%H%M))); [ ! -e "$GO" ] && { [ $n -ge 2050 ] || [ $n -lt 15 ]; }; }

# ── 1 the machine, production and the sources as expected ───────────────────────────────────────────────────────────────
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[q278] $START start"
cmp -s "$PROD" "$G76" || { echo "[q278] ☠️ production is not g76pgo — QUEUE 278 ABORTED (nothing changed)"; exit 1; }
busy && { echo "[q278] ☠️ another job is running — QUEUE 278 ABORTED (nothing changed)"; exit 1; }
[ "$(grep -c '^ *Config::ENABLE_QSTANDPAT_SEED = true;$' "$SE")" = 1 ] && [ "$(grep -c '^ *Config::SORT_KEYED = 1;$' "$SE")" = 1 ] \
	&& [ "$(grep -c 'Config::ENABLE_QSTANDPAT_SEED = env_flag("ENABLE_QSTANDPAT_SEED", Config::ENABLE_QSTANDPAT_SEED);' "$SE")" = 1 ] \
	&& [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.7)");' "$UM")" = 1 ] && [ "$(grep -c 'std::strcmp(arm, "0") != 0' "$UM")" = 1 ] \
	|| { echo "[q278] ☠️ the sources are not the gen 7.7 edit — QUEUE 278 ABORTED (nothing changed)"; exit 1; }

# ── 2 build, gates ───────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the build"
echo "[q278] $(date '+%H:%M') PGO build"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_g77gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g77gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g77gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 778 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_g77pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q278] ☠️ PGO use build failed — QUEUE 278 ABORTED (production untouched)"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q278] ☠️ PGO gen build failed — QUEUE 278 ABORTED (production untouched)"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q278] $(date '+%H:%M') UCI build done; training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
ok=1
a=$(fp "$NEW" ""); ks=$(fp "$G76" "ENABLE_QSTANDPAT_SEED=1"); s0=$(fp "$NEW" "SORT_KEYED=0"); sr=$(fp "$NEW" "SCRATCH_REUSE=2"); ev=$(fp "$NEW" "EVAL_MAT_CACHE=2")
off=$(fp "$NEW" "ENABLE_QSTANDPAT_SEED=0"); off2=$(fp "$NEW" "ENABLE_QSTANDPAT_SEED=0,SV2_VERIFY_PROMOTE=0")
ea=$(printf 'setoption name EvalArm value 1\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ea76=$(printf 'setoption name EvalArm value 1\nbench\nquit\n' | "$G76" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q278] g77pgo default [$a] (want 254 / 35922538) · seed knob on g76pgo [$ks] · SORT_KEYED=0 [$s0] · SCRATCH_REUSE=2 check [$sr] · EVAL_MAT_CACHE=2 [$ev] · explicit EvalArm 1 [$ea] (all == default)"
echo "[q278] g77pgo ENABLE_QSTANDPAT_SEED=0 [$off] (= gen 7.6, 246 / 35138434) · + SV2_VERIFY_PROMOTE=0 [$off2] (= gen 7.5, 252 / 35235460) · EvalArm 1 on g76pgo [$ea76] (the old trap, recorded)"
[ "$a" = "solved 254/300 nodes 35922538" ] && [ "$ks" = "$a" ] && [ "$s0" = "$a" ] && [ "$sr" = "$a" ] && [ "$ev" = "$a" ] && [ "$ea" = "$a" ] \
	&& [ "$off" = "solved 246/300 nodes 35138434" ] && [ "$off2" = "solved 252/300 nodes 35235460" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ID=$(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep -m1 '^id name')
echo "[q278] v1 [$v1] (250 / 35310775-778) · $ID"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
echo "$ID" | grep -qF 'AlphaUX (v2 gen 7.7)' || ok=0
ga=$(gfp "$NEW" ""); gk=$(gfp "$G76" "ENABLE_QSTANDPAT_SEED=1"); g0=$(gfp "$NEW" "ENABLE_QSTANDPAT_SEED=0"); gb=$(gfp "$G76" "")
echo "[q278] game12 d10: g77 default [$ga] == g76 + seed [$gk] · g77 seed off [$g0] == g76 default [$gb] (8/12 2272158)"
[ -n "$ga" ] && [ "$ga" = "$gk" ] && [ -n "$gb" ] && [ "$g0" = "$gb" ] || ok=0
[ $ok = 1 ] || { echo "[q278] ☠️ a gate failed — no install (production stays g76pgo) — QUEUE 278 ABORTED"; exit 1; }
echo "[q278] ✅ every gate equal"

# ── 3 install, Cython, pinned bench ──────────────────────────────────────────────────────────────────────────────────────
stopped "the install"
cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" || { echo "[q278] ☠️ install copy failed — restoring g76pgo"; cp "$G76" "$PROD"; exit 1; }
echo "[q278] $(date '+%H:%M') INSTALLED: production == g77pgo: bench [$(fp "$PROD" "")] · $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name')"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q278] pre-7.7 .so backed up to $BK" || { echo "[q278] ☠️ .so backup failed — Cython not rebuilt (UCI is gen 7.7)"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q278] ☠️ Cython build left no .so — restoring the pre-7.7 .so"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac q278_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac q278_fp_76 V2_PRESET=shipped ENABLE_QSTANDPAT_SEED=0 > "$W/cy_76.txt" 2>&1
bash "$R" wac q278_fp_v1 > "$W/cy_v1.txt" 2>&1
echo "[q278] $(date '+%H:%M') Cython build done — the sources are free to edit"
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
echo "[q278] Cython shipped [$(cyr "$W/cy_default.txt")] (want 254 / 35922538) · seed off [$(cyr "$W/cy_76.txt")] (gen 7.6 246 / 35138434 / EBF 3.698) · v1 [$(cyr "$W/cy_v1.txt")] (250 / 35310778)"
grep -q 'Solved 254/300' "$W/cy_default.txt" && grep -q '^NODES: 35922538' "$W/cy_default.txt" || echo "[q278] ☠️ the Cython module's shipped fingerprint differs from the UCI binary's — tell the owner (games use the UCI binary)"
until ! evening; do stopped "the pinned bench"; sleep 60; done
bench_one() {  # round, arm
	local out b
	case $2 in g77) b="$NEW" ;; g76) b="$G76" ;; esac
	out=$(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')
	echo "[q278]   r$1 $2 $out"
}
bt() { grep -E "^\[q278\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q278\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
busy && echo "[q278] ☠️ another job is running — the pinned bench below is disturbed"
echo "[q278] $(date '+%H:%M') == pinned bench x11 on core 5, g77 / g76 alternating (ledger)"
for i in $(seq 1 11); do
	stopped "bench round $i"
	if [ $((i % 2)) = 1 ]; then o="g77 g76"; else o="g76 g77"; fi
	for x in $o; do bench_one $i $x; done
done
for x in g77 g76; do echo "[q278] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
echo "[q278] within-round g77 / g76: time median $(join <(bt g77) <(bt g76) | awk '{print $2 / $3}' | med) (g77 faster in $(join <(bt g77) <(bt g76) | awk '$2 < $3 {n++} END {print n + 0}') of 11) · nps median $(join <(bn g77) <(bn g76) | awk '{print $2 / $3}' | med) (bench nodes +2.2%)"
echo "[q278] $(date '+%H:%M') GEN 7.7 SHIPPED"
echo "[q278] $(date) QUEUE 278 DONE"
