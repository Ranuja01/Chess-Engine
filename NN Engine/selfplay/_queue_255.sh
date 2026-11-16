#!/bin/bash
# 2026-11-15 ~21:10, queue #255: the eval speed ship (owner 11-15: "if its looking clean then we can ship"; log bench / NPS).
# Shipped preset: EVAL_MAT_CACHE 1 + EVAL_FAST 31 (batch 1: material cache, rings, direct calls · batch 2: per-type attack build,
# PX no-passer skip · batch 3: bit 16 placement lean, −0.247% instructions; bit 32 packed PST −0.106% did NOT count, left off).
# Same score for every position: the bench stays 250 / 37,320,382. The id stays "AlphaUX (v2 gen 7.4)" (a speed build keeps its
# gen, as g73fpgo did); binary chess_engine_uci_g74epgo.
#  A (tonight, after #254, beside #252's node-limited SF18 step; fixed work only, and only while >= 3 SF18 chunks remain):
#    PGO build (profile /mnt/e/chess_data/pgo_g74e, trained on the default path). GATES: bench 250 / 37,320,382 by default, with
#    EVAL_MAT_CACHE=0,EVAL_FAST=0 and with the verify mode EVAL_MAT_CACHE=2; SV2_VERIFY_DEEPER=0 249 / 37,279,382; v1 250 /
#    35,310,775-778; `bench game12 10` default == verify. LIVENESS (the bench cannot show a result-identical default): cachegrind
#    instructions on `bench game12 10`, default vs EVAL_MAT_CACHE=0,EVAL_FAST=0: the default must be at least 5% fewer
#    (batches 1-3 measured ≈ −9.8% on the plain build).
#  B (Monday, after #253 ends, quiet box): pinned bench x9 on core 5, arm order rotating: production g74pgo · g74epgo default ·
#    g74epgo EVAL_FAST=15 (bit 16 off: batch 3's own share) · g74epgo EVAL_MAT_CACHE=0,EVAL_FAST=0 (the build with every switch
#    off). REGISTERED: install when default / g74 within-round median < 0.99 and default is faster in >= 7 of 9 rounds;
#    prediction −4 to −6%. Bit 16 alone (default / EVAL_FAST=15) is expected below resolution (~ −0.1 to −0.2%): reported only.
#    INSTALL: copy g74epgo over build_uci/chess_engine_uci (g74pgo stays as the backup), rebuild the Cython module (backup
#    /mnt/e/chess_data/so_backup_g74e) and fingerprint it, print the ledger numbers.
# Stop: touch /mnt/e/chess_data/q255.stop (between steps). Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q255
P=/mnt/e/chess_data/pgo_g74e
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G74="$ND/build_uci/chess_engine_uci_g74pgo"
NEW="$ND/build_uci/chess_engine_uci_g74epgo"
OFF="EVAL_MAT_CACHE=0,EVAL_FAST=0"
BK=/mnt/e/chess_data/so_backup_g74e
LOG=/mnt/e/chess_data/q255.log
Q252=/mnt/e/chess_data/q252.log
STOP=/mnt/e/chess_data/q255.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q255]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q255] $(date) stop file before $1 — QUEUE 255 DONE"; exit 0; }; }
room() { [ "$(grep -c 'SF18 seed' "$Q252")" -le 15 ] && ! grep -qE 'QUEUE 252 (DONE|ABORTED)' "$Q252"; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
echo "[q255] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #254 to end"
until grep -qE 'QUEUE 254 (DONE|ABORTED)' /mnt/e/chess_data/q254.log; do stopped "the start"; sleep 30; done
sleep 30
stopped "the build"
room || { echo "[q255] ☠️ fewer than 3 SF18 chunks left in #252 — the build waits for a later slot — QUEUE 255 DONE"; exit 0; }
cmp -s "$PROD" "$G74" || { echo "[q255] ☠️ production is not g74pgo — QUEUE 255 ABORTED"; exit 1; }

# ── A build, gates, liveness ──────────────────────────────────────────────────────────────────────────────────────────────────
echo "[q255] $(date '+%H:%M') PGO build"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_g74egen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g74egen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g74egen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 755 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_g74epgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q255] ☠️ PGO use build failed — QUEUE 255 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; exit 1; }
else
	echo "[q255] ☠️ PGO gen build failed — QUEUE 255 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; exit 1
fi
echo "[q255] $(date '+%H:%M') build done — the sources are free to edit"
echo "[q255] training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
ok=1
a=$(fp "$NEW" ""); b=$(fp "$NEW" "$OFF"); v=$(fp "$NEW" "EVAL_MAT_CACHE=2"); d0=$(fp "$NEW" "SV2_VERIFY_DEEPER=0")
echo "[q255] g74epgo default [$a] · all off [$b] · verify [$v] (want 250 / 37320382 each) · SV2_VERIFY_DEEPER=0 [$d0] (249 / 37279382)"
[ "$a" = "solved 250/300 nodes 37320382" ] && [ "$b" = "$a" ] && [ "$v" = "$a" ] && [ "$d0" = "solved 249/300 nodes 37279382" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q255] v1 [$v1] (250 / 35310775-778) · id: $(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep -m1 '^id name')"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2\nbench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q255] game12 d10: verify [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
[ $ok = 1 ] || { echo "[q255] ☠️ a gate failed — no install — QUEUE 255 ABORTED"; exit 1; }
for arm in def:"" off:"$OFF"; do
	n=${arm%%:*}; k=${arm#*:}; kl=""; [ -n "$k" ] && kl="setoption name Knobs value $k"
	stopped "cachegrind $n"
	printf "%s\nbench $G 10\nquit\n" "$kl" | nice -n 10 valgrind --tool=cachegrind --cache-sim=no \
		--cachegrind-out-file="$W/cg_$n.out" "$NEW" > "$W/bench_$n.txt" 2> "$W/cg_$n.log"
	echo "[q255] cachegrind $n [${k:-default}]: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/bench_$n.txt") · Ir $(irefs "$W/cg_$n.log")"
done
lv=$(awk -v a="$(irefs "$W/cg_off.log")" -v x="$(irefs "$W/cg_def.log")" 'BEGIN { printf "%.3f", 100 * (x - a) / a }')
echo "[q255] LIVENESS: default vs all off $lv% instructions (want <= -5)"
awk -v d="$lv" 'BEGIN { exit !(d <= -5) }' || { echo "[q255] ☠️ the shipped default does not run the fast paths — no install — QUEUE 255 ABORTED"; exit 1; }

# ── B pinned bench and install (after #253) ───────────────────────────────────────────────────────────────────────────────────
echo "[q255] $(date '+%H:%M') waiting for #253 to end"
until grep -qE 'QUEUE 253 (DONE|ABORTED)|☠' /mnt/e/chess_data/q253.log; do stopped "the bench"; sleep 60; done
sleep 60
busy && { echo "[q255] ☠️ another job is running — no bench — QUEUE 255 DONE (re-run step B by hand)"; exit 1; }
cmp -s "$PROD" "$G74" || { echo "[q255] ☠️ production changed meanwhile — QUEUE 255 ABORTED"; exit 1; }
bench_one() {  # round, arm
	local out
	case $2 in
		g74) out=$(printf 'bench\nquit\n' | taskset -c 5 "$G74" 2>/dev/null | grep '^bench') ;;
		def) out=$(printf 'bench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
		b15) out=$(printf 'setoption name Knobs value EVAL_FAST=15\nbench\nquit\n' | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
		off) out=$(printf 'setoption name Knobs value %s\nbench\nquit\n' "$OFF" | taskset -c 5 "$NEW" 2>/dev/null | grep '^bench') ;;
	esac
	echo "[q255]   r$1 $2 $out"
}
bt() { grep -E "^\[q255\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
echo "[q255] $(date '+%H:%M') == TIMED: pinned bench x9 on core 5, arm order rotating"
for i in 1 2 3 4 5 6 7 8 9; do
	stopped "bench round $i"
	case $((i % 4)) in 1) o="g74 def b15 off" ;; 2) o="def b15 off g74" ;; 3) o="b15 off g74 def" ;; 0) o="off g74 def b15" ;; esac
	for a in $o; do bench_one $i $a; done
done
for a in g74 def b15 off; do echo "[q255] median $a: $(bt $a | awk '{print $2}' | med) ms · nps $(grep -E "^\[q255\]   r[0-9]+ $a bench" "$LOG" | awk '{print $13}' | med) over $(bt $a | wc -l) runs"; done
R1=$(join <(bt def) <(bt g74) | awk '{print $2 / $3}' | med); F1=$(join <(bt def) <(bt g74) | awk '$2 < $3 {n++} END {print n + 0}')
echo "[q255] within-round default / g74: median $R1 · default faster in $F1 of 9"
echo "[q255] within-round default / b15 (bit 16's share): median $(join <(bt def) <(bt b15) | awk '{print $2 / $3}' | med) · off / g74 (the build alone): median $(join <(bt off) <(bt g74) | awk '{print $2 / $3}' | med)"
if awk -v r="$R1" -v f="$F1" 'BEGIN { exit !(r < 0.99 && f >= 7) }'; then
	echo "[q255] REGISTERED: PASS — installing"
	cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" || { echo "[q255] ☠️ install copy failed — restoring g74pgo"; cp "$G74" "$PROD"; exit 1; }
	echo "[q255] production == g74epgo: bench [$(fp "$PROD" "")] · $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name')"
	mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q255] .so backed up to $BK" || { echo "[q255] ☠️ .so backup failed — Cython not rebuilt"; exit 1; }
	nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
	ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q255] ☠️ Cython build left no .so — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
	bash "$R" wac q255_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
	bash "$R" wac q255_fp_v1 > "$W/cy_v1.txt" 2>&1
	echo "[q255] Cython shipped [$(grep -E '^SOLVED|^NODES' "$W/cy_default.txt" | tr '\n' ' ')] (want 250 / 37320382) · v1 [$(grep -E '^SOLVED|^NODES' "$W/cy_v1.txt" | tr '\n' ' ')] (250 / 35310778)"
else
	echo "[q255] REGISTERED: not faster by the rule — NOT installed (production stays g74pgo)"
fi
echo "[q255] $(date) QUEUE 255 DONE"
