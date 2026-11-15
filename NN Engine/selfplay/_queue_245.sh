#!/bin/bash
# 2026-11-14 ~23:30, queue #245: light fixed-depth work beside queue #241's NODE-LIMITED SF18 step (node-limited games do not
# depend on machine load; nothing here runs during #241's timed steps). Starts when #241 logs its first SF18 chunk or ends;
# stops between runs when /mnt/e/chess_data/q245.stop exists (queue #244 writes it before its timed games) and restores the
# production module.
#  1 Cython module rebuilt from today's sources (production backed up to /mnt/e/chess_data/so_backup_q245/), gated.
#  2 The owner's STS question (11-14: "understand the STS lost so we can eventually recover it"): is D7's STS −148 real?
#    a sts300 at d10 for two neutral arms, N1 ASPIRATION_DELTA=300 and N2 ASPIRATION_DELTA=800 (the measured-null aspiration
#      lane): their deltas vs production (tonight 2037) are the null spread on this set.
#    b the full STS (1,500 positions, five times sts300) at d10 for S0, D7, N1, then D7=2 and D7+PROMOTE.
#    REGISTERED: D7's loss is real if on STS1500 D7 - S0 is below N1 - S0 by more than the 1,500-position noise (taken as
#    150 / sqrt 5 ~ 70 points per arm pair) AND its per-position sign test (worse vs better) is past 2 sigma; else STS noise.
#  3 D7=2 move quality at d7, base = D7: _ks_footprint_regret.py, arms D7=2 and neutral (D7 + ASPIRATION_DELTA=300), on
#    game_regret_set then _v2. REGISTERED: within 2pp of neutral on either set = no resolvable move signal.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
W=/mnt/e/chess_data/r12a
BK=/mnt/e/chess_data/so_backup_q245
STOP=/mnt/e/chess_data/q245.stop
D7="SV2_VERIFY_DEEPER=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W" "$BK"
show() { sed "s/^/[q245]   /"; }
restore() { cp "$BK"/ChessAI.*.so . && echo "[q245] production module restored: $(bash "$R" wac r12a_fprest V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')"; }
stopped() { [ -e "$STOP" ] && { echo "[q245] $(date) stop file before $1"; restore; echo "[q245] QUEUE 245 DONE"; exit 0; }; }
echo "[q245] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #241's SF18 step"
until grep -qE 'SF18 seed|QUEUE 241|☠' /mnt/e/chess_data/q241.log; do [ -e "$STOP" ] && { echo "[q245] stop file — QUEUE 245 DONE"; exit 0; }; sleep 60; done
grep -qE 'Mediocre seed 241312 ctl' /mnt/e/chess_data/q241.log || grep -qE 'skip Mediocre|QUEUE 241' /mnt/e/chess_data/q241.log || { echo "[q245] ☠️ SF18 started before Mediocre finished? — QUEUE 245 ABORTED"; exit 1; }
echo "[q245] $(date '+%H:%M') start"

# ── 1 module ──────────────────────────────────────────────────────────────────────────────────────────────────────────
cp ChessAI.*.so "$BK"/ && echo "[q245] production module backed up to $BK" || { echo "[q245] ☠️ backup failed — QUEUE 245 ABORTED"; exit 1; }
nice -n 10 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q245] ☠️ no .so after the build — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
echo "[q245] $(date '+%H:%M') module built — the sources are free to edit"
f=$(bash "$R" wac r12a_fp V2_PRESET=shipped 2>&1 | grep -E '^SOLVED|^NODES' | tr '\n' ' ')
echo "[q245] module shipped (want 249 / 37279382): $f"
echo "$f" | grep -q 'Solved 249/300 NODES: 37279382' || { echo "[q245] ☠️ the module moved a node"; restore; exit 1; }

# ── 2 STS ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASE="MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped"
declare -A ARM
ARM[S0]=""; ARM[D7]="$D7"; ARM[N1]="ASPIRATION_DELTA=300"; ARM[N2]="ASPIRATION_DELTA=800"
ARM[D72]="SV2_VERIFY_DEEPER=2"; ARM[PROM]="$D7 SV2_VERIFY_PROMOTE=1"
sts() {  # arm, epd, tag
	stopped "$3"
	env $BASE ${ARM[$1]} nice -n 10 bash "$R" pyrun diagnostics/sts_test.py $2 $3 > "$W/$3.out" 2> "$W/$3.err"
	echo "[q245] $(date '+%H:%M') $3 (${ARM[$1]:-production}): $(grep -E 'STS score' "$W/$3.out" | tail -1)"
}
sts N1 sts300.epd r12a_s3_N1
sts N2 sts300.epd r12a_s3_N2
for a in S0 D7 N1; do sts $a STS1-STS15_LAN_v3.epd r12a_full_$a; done

# ── 3 D7=2 footprint regret on a D7 base ──────────────────────────────────────────────────────────────────────────────
for s in game_regret_set game_regret_set_v2; do
	stopped "regret $s"
	echo "[q245] $(date '+%H:%M') == regret $s"
	env SET=ks_sets/$s.csv DEPTH=7 JOBS=2 BY_PHASE=0 BY_PS=0 BASE_KNOBS="V2_PRESET=shipped $D7" \
		CAND_KNOBS="SV2_VERIFY_DEEPER=2;$D7 ASPIRATION_DELTA=300" CAND_NAME="D7=2;neutral" DUMP=$W/fp_$s.csv \
		nice -n 10 bash "$R" pyrun diagnostics/_ks_footprint_regret.py > "$W/fp_$s.out" 2>&1
	grep -E '^\s+(candidate|D7=2|neutral|cr4)' "$W/fp_$s.out" | head -12 | show
done

for a in D72 PROM; do sts $a STS1-STS15_LAN_v3.epd r12a_full_$a; done

restore
echo "[q245] $(date) QUEUE 245 DONE"
