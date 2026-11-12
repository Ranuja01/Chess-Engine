#!/bin/bash
# 2026-11-11 ~20:50, queue #209 (overnight, after #207): search v2 S3, the first two items.
#  1 PGO build of today's sources (SV2_PRUNE_EARLY and see_move included), trained with SEARCH_ARM=1 SV2_PRUNE_EARLY=1 →
#    build_uci/sv2_pgo1pe. Gates: default bench 247 / 39,861,964; with SEARCH_ARM=1 SV2_PRUNE_EARLY=1 the same.
#  2 Speed of early pruning (only if queue #208's identity gates all read SAME): _ebf_per_iter.py, the #207 set (40 per phase,
#    seed 1, depth 13), sv2_pgo1 (S1, #207) vs sv2_pgo1pe with SV2_PRUNE_EARLY=1, alternated three times; same tree required.
#    REGISTERED: early pruning faster (prediction −3 to −8% time to depth 13; #208 reports its instruction count).
#  3 GAMES, the per-move exchange value (SEE_PER_MOVE=3: capture ordering + the qsearch filter; MOVEGEN-AUDIT C3, S3 step 0:
#    a cheaper attacker of ours hits the square on 15-17% of the scorer's exchange-branch captures). A = SEE_PER_MOVE=3,
#    B = SEE_PER_MOVE=0, both sv2_pgo1pe with SEARCH_ARM=1 (+ SV2_PRUNE_EARLY=1 if #208 passed). 2,000 games, 10+0.1, UHO,
#    seed 209. Fingerprints (WAC d10, SEARCH_ARM=1): 0 = 247 / 39,861,964; 3 = 249 / 37,279,382 (measured on SEARCH_ARM=0;
#    the arms are identical).
#    REGISTERED before any game: A − B ≥ +2σ (pentanomial) = KEEP as S3's default; within ±2σ = NEUTRAL (a knob; the S3 picker
#    still needs a per-move test for its good / bad split); ≤ −2σ = REJECT for ordering. Prediction: +0 to +8 Elo, likely
#    unresolved at 2,000 games.
# Waits for #207. Stop: touch /mnt/e/chess_data/q209.stop (checked between steps). ≤ 3 engine processes. Never edit this file,
# the tools or the sources while it builds.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/s3_a
P=/mnt/e/chess_data/pgo_sv2_pe
STOP=/mnt/e/chess_data/q209.stop
LATEST=$(date -d '2026-11-12 10:00' +%s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q209]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q209] $(date) stop file present before $1 — QUEUE 209 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q209] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #207"
until grep -qE "QUEUE 207 (DONE|ABORTED)|skipped" /mnt/e/chess_data/q207.log 2>/dev/null; do
	[ -e "$STOP" ] && { echo "[q209] stopped while waiting"; exit 0; }
	sleep 60
done
[ $(date +%s) -lt $LATEST ] || { echo "[q209] $(date) past 10:00 — skipped"; exit 0; }
echo "[q209] $(date '+%Y-%m-%d %H:%M:%S') start"
# Early pruning is used only if every identity gate of #208 read SAME.
PE=0
grep -q "identity: 0 set(s) differ" /mnt/e/chess_data/q208.log && grep -q "game identity: 0 of 12 games differ" /mnt/e/chess_data/q208.log && PE=1
echo "[q209] #208 identity gates: $([ $PE = 1 ] && echo 'all SAME → SV2_PRUNE_EARLY=1 in the games' || echo 'NOT all SAME → early pruning left off')"

# ── 1 PGO build ──────────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=sv2_gen1pe PGO=gen PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_gen.out" 2>&1 || { echo "[q209] ☠️ gen build failed — QUEUE 209 ABORTED"; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/sv2_gen1pe" name=a option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE st=0.25 \
	-engine cmd="$ND/build_uci/sv2_gen1pe" name=b option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 209 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
echo "[q209] training: $(grep -E '^Games:' "$W/train.log" | tail -1) · warnings $(grep -c 'have option' "$W/train.log") · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=sv2_pgo1pe PGO=use PGO_DIR=$P bash selfplay/build_uci.sh > "$W/build_use.out" 2>&1 || { echo "[q209] ☠️ use build failed — QUEUE 209 ABORTED"; exit 1; }
B="$ND/build_uci/sv2_pgo1pe"
d0=$(fp "$B" ""); d1=$(fp "$B" "SEARCH_ARM=1,SV2_PRUNE_EARLY=1"); s3=$(fp "$B" "SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE,SEE_PER_MOVE=3")
echo "[q209] sv2_pgo1pe bench: default [$d0] · arm 1 + early [$d1] (want 247 / 39861964 both) · SEE_PER_MOVE=3 [$s3] (want 249 / 37279382)"
[ "$d0" = "solved 247/300 nodes 39861964" ] && [ "$d1" = "solved 247/300 nodes 39861964" ] || { echo "[q209] ☠️ fingerprint moved — QUEUE 209 ABORTED"; exit 1; }

# ── 2 speed of early pruning ─────────────────────────────────────────────────────────────────────────────────────────
if [ $PE = 1 ]; then
	for rep in 1 2 3; do
		for arm in s1 pe; do
			stopped "speed rep $rep $arm"
			[ $(date +%s) -lt $LATEST ] || { echo "[q209] $(date) past 10:00 — QUEUE 209 DONE"; exit 0; }
			if [ $arm = s1 ]; then BIN="$ND/build_uci/sv2_pgo1"; K="SEARCH_ARM=1"; else BIN="$B"; K="SEARCH_ARM=1,SV2_PRUNE_EARLY=1"; fi
			a=$([ $arm = s1 ] && echo 0 || echo 1)
			timeout 3000 bash "$R" pyrun diagnostics/_ebf_per_iter.py E1=$arm="$BIN" O1="OwnBook:false;Knobs:$K" D1=13 N=40 SEED=1 \
				OUT="$W/speed_r${rep}_arm$a" > "$W/speed_r${rep}_arm$a.log" 2>&1
		done
	done
	bash "$R" pyrun diagnostics/_sv2_speed_read.py DIR="$W" > "$W/speed_read.txt" 2>&1
	echo "[q209] speed, arm 0 = S1 (sv2_pgo1), arm 1 = early pruning (sv2_pgo1pe):"
	show < "$W/speed_read.txt"
fi

# ── 3 games: the per-move exchange value ─────────────────────────────────────────────────────────────────────────────
stopped "the SEE_PER_MOVE games"
[ $(date +%s) -lt $LATEST ] || { echo "[q209] $(date) past 10:00 — QUEUE 209 DONE"; exit 0; }
echo "[q209] $(date '+%H:%M') games: SEE_PER_MOVE=3 vs 0 at 10+0.1, 2,000 games"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$B" name=see3 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE,SEE_PER_MOVE=3 tc=10+0.1 timemargin=250 \
	-engine "cmd=$B" name=see0 option.OwnBook=false option.Knobs=SEARCH_ARM=1,SV2_PRUNE_EARLY=$PE,SEE_PER_MOVE=0 tc=10+0.1 timemargin=250 \
	$COMMON -srand 209 -rounds 1000 -games 2 -pgnout file=$W/see.pgn notation=san nodes=true append=false > $W/see.log 2>&1
echo "[q209] games: $(grep -E '^Games:' $W/see.log | tail -1) · warnings $(grep -c 'have option' $W/see.log) · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' $W/see.log) · time forfeits $(grep -c 'loses on time' $W/see.pgn)"
grep -E 'Elo:|Ptnml' $W/see.log | tail -2 | show
timeout 900 bash "$R" pyrun selfplay/fastchess_report.py $W/see.pgn --p1 see3 2>&1 | sed -n '/per engine/,$p' | show
echo "[q209] $(date) QUEUE 209 DONE"
