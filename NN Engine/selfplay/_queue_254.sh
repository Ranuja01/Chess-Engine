#!/bin/bash
# 2026-11-15 ~20:00, queue #254: light fixed-work jobs beside #252's NODE-LIMITED SF18 step (the owner's evening; node-limited
# games do not depend on machine load). One process at a time. Each step starts only while at least 3 of #252's 18 SF18 chunks
# are still to come (~39 min of margin), so nothing here overlaps #252's timed games; whatever is left is deferred.
#  1 Test build chess_engine_uci_e3test (plain) with eval speed batch 3, result-identical, each bit defaulting to off:
#      EVAL_FAST bit 16  the search-path placement detector skips the two counts the C1 scorer never reads (reach_n, longdiag)
#      EVAL_FAST bit 32  the tapered PST summed from one packed table (eg * 2^32 + mg)
#    GATES: bench 250 / 37,320,382 for EVAL_FAST 16, 32, 48 and with batches 1-3 (EVAL_MAT_CACHE=1,EVAL_FAST=63; verify mode
#    EVAL_MAT_CACHE=2,EVAL_FAST=63); gen 7.3 knob 249 / 37,279,382 and v1 250 / 35,310,775-778 with everything on; `bench game12
#    10` equal to default with verify + everything on. C3b fingerprints printed (SV2_STATIC_CUT_STORE 1 / 2 / 3).
#  2 cachegrind instructions on `bench game12 10`, base = batches 1 + 2 on (EVAL_MAT_CACHE=1,EVAL_FAST=15): +16, +32, +both.
#    REGISTERED: an item counts when it removes more than 0.2% with the same nodes. Predictions: bit 16 −0.2 to −0.8%,
#    bit 32 −0.3 to −1%.
#  3 Cython module rebuilt (backup /mnt/e/chess_data/so_backup_q254); shipped 250 / 37,320,382, gen 7.3 knob 249 / 37,279,382,
#    v1 250 / 35,310,778, else the backup is restored and step 4 is skipped.
#  4 C3b screens, STS1500 d10 + WAC, against gen 7.4 STS1500 9,945 / WAC 250 (#245's rule: a real move is > 70 from the base
#    AND a per-position sign test past 2σ): SV2_STATIC_CUT_STORE=2 (the static cut stored at depth 1) and =3 (one ply
#    shallower). C3 (=1, no store) read −307, z −2.26. Prediction: =2 close to C3 (a depth-1 entry rarely answers a probe),
#    =3 between C3 and the base.
# Stop: touch /mnt/e/chess_data/q254.stop (between steps). Never edit the sources before "build done" AND "Cython build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/q254
G=/mnt/e/chess_data/sv2_speed/game12.txt
T="$ND/build_uci/chess_engine_uci_e3test"
BK=/mnt/e/chess_data/so_backup_q254
Q252=/mnt/e/chess_data/q252.log
STOP=/mnt/e/chess_data/q254.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q254]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q254] $(date) stop file before $1 — QUEUE 254 DONE"; exit 0; }; }
room() { [ "$(grep -c 'SF18 seed' "$Q252")" -le 15 ] && ! grep -qE 'QUEUE 252 (DONE|ABORTED)' "$Q252"; }
step() { stopped "$1"; room || { echo "[q254] $(date '+%H:%M') fewer than 3 SF18 chunks left in #252 — $1 and the rest deferred — QUEUE 254 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$1" ] && k="setoption name Knobs value $1"; printf '%s\nbench\nquit\n' "$k" | "$T" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
irefs() { grep -E 'I *refs' "$1" | grep -oE '[0-9,]+$' | tr -d ,; }
echo "[q254] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #252's SF18 step"
until grep -qE 'SF18 legs|QUEUE 252 (DONE|ABORTED)' "$Q252"; do stopped "the start"; sleep 30; done
grep -qE 'QUEUE 252 (DONE|ABORTED)' "$Q252" && ! grep -q 'SF18 legs' "$Q252" && { echo "[q254] ☠️ #252 ended before its SF18 step — QUEUE 254 ABORTED"; exit 1; }
sleep 60
echo "[q254] $(date '+%H:%M') start"

# ── 1 build and gates ─────────────────────────────────────────────────────────────────────────────────────────────────────────
step "the build"
OUT=chess_engine_uci_e3test nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 || { echo "[q254] ☠️ build failed — QUEUE 254 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q254] $(date '+%H:%M') build done"
ok=1
for k in "" "EVAL_FAST=16" "EVAL_FAST=32" "EVAL_FAST=48" "EVAL_MAT_CACHE=1,EVAL_FAST=63" "EVAL_MAT_CACHE=2,EVAL_FAST=63"; do
	f=$(fp "$k"); echo "[q254] bench [${k:-default}]: $f"
	[ "$f" = "solved 250/300 nodes 37320382" ] || ok=0
done
d0=$(fp "EVAL_MAT_CACHE=1,EVAL_FAST=63,SV2_VERIFY_DEEPER=0"); echo "[q254] gen 7.3 knob, all on (249 / 37279382): $d0"
[ "$d0" = "solved 249/300 nodes 37279382" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nsetoption name Knobs value EVAL_MAT_CACHE=1,EVAL_FAST=63\nbench\nquit\n' | "$T" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q254] v1, all on (250 / 35310775-778): $v1"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2,EVAL_FAST=63\nbench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$T" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q254] game12 d10: verify + all on [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
echo "[q254] C3b fingerprints: =1 [$(fp SV2_STATIC_CUT_STORE=1)] · =2 [$(fp SV2_STATIC_CUT_STORE=2)] · =3 [$(fp SV2_STATIC_CUT_STORE=3)]"
[ $ok = 1 ] || echo "[q254] ☠️ a speed gate failed — no instruction counts (the C3b steps still run)"

# ── 2 instructions ────────────────────────────────────────────────────────────────────────────────────────────────────────────
if [ $ok = 1 ]; then
	for arm in b15:"EVAL_MAT_CACHE=1,EVAL_FAST=15" b31:"EVAL_MAT_CACHE=1,EVAL_FAST=31" b47:"EVAL_MAT_CACHE=1,EVAL_FAST=47" b63:"EVAL_MAT_CACHE=1,EVAL_FAST=63"; do
		n=${arm%%:*}; k=${arm#*:}
		step "cachegrind $n"
		printf "setoption name Knobs value %s\nbench $G 10\nquit\n" "$k" | nice -n 10 valgrind --tool=cachegrind --cache-sim=no \
			--cachegrind-out-file="$W/cg_$n.out" "$T" > "$W/bench_$n.txt" 2> "$W/cg_$n.log"
		echo "[q254] cachegrind $n [$k]: $(grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' "$W/bench_$n.txt") · Ir $(irefs "$W/cg_$n.log")"
	done
	b=$(irefs "$W/cg_b15.log")
	for n in b31 b47 b63; do echo "[q254]   $n vs batches 1 + 2: $(awk -v a="$b" -v x="$(irefs "$W/cg_$n.log")" 'BEGIN { printf "%+.3f%%", 100 * (x - a) / a }')"; done
fi

# ── 3 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────────────
step "the Cython build"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q254] .so backed up to $BK" || { echo "[q254] ☠️ .so backup failed — QUEUE 254 ABORTED"; exit 1; }
nice -n 10 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q254] ☠️ Cython build left no .so — restoring the backup — QUEUE 254 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac q254_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac q254_fp_d0 V2_PRESET=shipped SV2_VERIFY_DEEPER=0 > "$W/cy_d0.txt" 2>&1
bash "$R" wac q254_fp_v1 > "$W/cy_v1.txt" 2>&1
c0=$(grep -E '^SOLVED|^NODES' "$W/cy_default.txt" | tr '\n' ' '); c1=$(grep -E '^SOLVED|^NODES' "$W/cy_d0.txt" | tr '\n' ' '); c2=$(grep -E '^SOLVED|^NODES' "$W/cy_v1.txt" | tr '\n' ' ')
echo "[q254] Cython shipped [$c0] (want 250 / 37320382) · d0 [$c1] (249 / 37279382) · v1 [$c2] (250 / 35310778)"
cy_ok=1
echo "$c0" | grep -q '250/300' && echo "$c0" | grep -q '37320382' || cy_ok=0
echo "$c1" | grep -q '249/300' && echo "$c1" | grep -q '37279382' || cy_ok=0
echo "$c2" | grep -q '250/300' && echo "$c2" | grep -qE '3531077[5-8]' || cy_ok=0
if [ $cy_ok = 0 ]; then echo "[q254] ☠️ Cython fingerprints — restoring the backup — QUEUE 254 ABORTED"; cp "$BK"/ChessAI.*.so .; exit 1; fi
echo "[q254] $(date '+%H:%M') Cython build done — the sources are free to edit"

# ── 4 C3b screens ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
for m in 2 3; do
	step "C3b mode $m"
	w=$(bash "$R" wac q254_wac_c3b$m V2_PRESET=shipped SV2_STATIC_CUT_STORE=$m 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
	env $BASE SV2_STATIC_CUT_STORE=$m nice -n 10 bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q254_full_c3b$m > "$W/full_c3b$m.out" 2> "$W/full_c3b$m.err"
	echo "[q254] $(date '+%H:%M') C3b SV2_STATIC_CUT_STORE=$m: WAC $w· $(grep -E 'STS score' "$W/full_c3b$m.out" | tail -1)"
done
echo "[q254] base for the screens: gen 7.4 STS1500 9,945 (r12a_full_D7), WAC 250 / 37,320,382; C3 (=1) 9,638"
echo "[q254] $(date) QUEUE 254 DONE"
