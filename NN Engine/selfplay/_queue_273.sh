#!/bin/bash
# 2026-11-18 ~12:45, queue #273: phase H re-sweep counts + the stand-pat seed's screen. Chained after #272.
#  1 plain UCI test build chess_engine_uci_cnt18 (two count-only counters: q_phantom = search v2 qsearch nodes that return the
#    mated sentinel because every capture was delta-pruned and no stand-pat seed replaced it; root_clock_m0 / root_clock_later =
#    root passes ended by the root's own clock check with time_up unset, which the driver then adopts as complete).
#    GATES: default 246 / 35,138,434; RFP_THREAT_GATE=1 257 / 38,712,861 (knobs pass); v1 250 / 35,310,775-778;
#    ENABLE_QSTANDPAT_SEED=1 must move the bench (recorded).
#  2 q_phantom on the bench (WAC 300) and game12 d10, default and with the seed (the seed must read 0: the counter's check).
#  3 TIMED, 100 g at 10+0.1, the same binary on both sides: cntd (Debug=true + FIX_COUNT_LOG=1, stderr to /tmp, Linux disk)
#    vs cnt (no Debug). Reads the TMCNT sums (root clock exits per 1,000 moves) and the score, which ablates #270's 9%
#    (that leg wrote its Debug stderr to a file on /mnt/e).
#  4 STS1500 at equal nodes for ENABLE_QSTANDPAT_SEED=1 at #272's three budgets, paired against #272's base runs.
# REGISTERED:
#  - q_phantom: "material" if >= 0.1% of qsearch visits on game12 (a fix worth games on its own).
#  - root clock exits: "worth a fix knob" if root_clock_m0 + root_clock_later >= 1 per 1,000 moves.
#  - the Debug ablation: cntd >= 40% ⇒ #270's 9% was the /mnt/e stderr writes; < 20% ⇒ Debug output itself is the cost.
#  - the seed (read with #272's verdict on the screen): prediction STS1500 vs base −40 … +60 at each budget.
# Stop: touch /mnt/e/chess_data/q273.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q273
T=/tmp/q273
G=/mnt/e/chess_data/sv2_speed/game12.txt
NEW="$ND/build_uci/chess_engine_uci_cnt18"
STOP=/mnt/e/chess_data/q273.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"; rm -rf "$T"; mkdir -p "$T"
show() { sed "s/^/[q273]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q273] $(date) stop file before $1 — QUEUE 273 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+' | head -1; }
qc() {  # binary, knobs, bench command → summed q_phantom / qnodes over every search's stats line
	local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"
	printf 'setoption name Debug value true\n%s\n%s\nquit\n' "$k" "$3" | "$1" 2>&1 | grep -oE 'q_phantom=[0-9]+ qnodes=[0-9]+' \
		| awk -F'[= ]' '{p += $2; q += $4} END {printf "q_phantom %d of %d qsearch visits (%.4f%%) over %d searches", p, q, 100 * p / (q ? q : 1), NR}'
}
echo "[q273] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #272"
until grep -q 'QUEUE 272' /mnt/e/chess_data/q272.log 2>/dev/null || ! pgrep -f "[_]queue_272.sh" >/dev/null; do stopped "the start"; sleep 30; done
sleep 20
pgrep -f "[f]astchess-linux|[b]uild_uci.sh|[s]ts_test|[v]algrind" >/dev/null && { echo "[q273] ☠️ another job is running — QUEUE 273 ABORTED"; exit 1; }
echo "[q273] $(date '+%H:%M') start"

# ── 1 build + gates ──────────────────────────────────────────────────────────────────────────────────────────────────────
OUT=chess_engine_uci_cnt18 nice -n 10 bash selfplay/build_uci.sh > "$W/build.out" 2>&1 \
	|| { echo "[q273] ☠️ UCI build failed — QUEUE 273 ABORTED"; grep -iE 'error' "$W/build.out" | head -8 | show; exit 1; }
echo "[q273] $(date '+%H:%M') build done — the sources are free to edit"
a=$(fp "$NEW" ""); g1=$(fp "$NEW" "RFP_THREAT_GATE=1"); sd=$(fp "$NEW" "ENABLE_QSTANDPAT_SEED=1")
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q273] cnt18 default [$a] (246 / 35138434) · RFP_THREAT_GATE=1 [$g1] (257 / 38712861) · v1 [$v1] · ENABLE_QSTANDPAT_SEED=1 [$sd] (must differ)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$g1" = "solved 257/300 nodes 38712861" ] && echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' \
	&& [ -n "$sd" ] && [ "$sd" != "$a" ] || { echo "[q273] ☠️ a gate failed — QUEUE 273 ABORTED"; exit 1; }
echo "[q273] ✅ every gate equal"

# ── 2 phantom counts ─────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the counts"
echo "[q273] WAC bench default: $(qc "$NEW" "" bench)"
echo "[q273] WAC bench seed:    $(qc "$NEW" "ENABLE_QSTANDPAT_SEED=1" bench) (must be 0)"
echo "[q273] game12 d10 default: $(qc "$NEW" "" "bench $G 10")"
echo "[q273] game12 d10 seed:    $(qc "$NEW" "ENABLE_QSTANDPAT_SEED=1" "bench $G 10") (must be 0)"
echo "[q273] game12 d10 seed bench: [$(printf "setoption name Knobs value ENABLE_QSTANDPAT_SEED=1\nbench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)] · default [$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)]"
echo "[q273] registered: material if >= 0.1% of qsearch visits on game12"

# ── 3 timed: root clock exits + the Debug ablation ───────────────────────────────────────────────────────────────────────
stopped "the timed leg"
pgrep -f "[f]astchess-linux|[s]ts_test" >/dev/null && echo "[q273] ☠️ another job is running — the timed leg below is disturbed"
printf '#!/bin/bash\nexec "%s" 2>>"%s/err_$$.log"\n' "$NEW" "$T" > "$T/wrap.sh"; chmod +x "$T/wrap.sh"
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
nice -n 5 "$FC" -engine "cmd=$T/wrap.sh" name=cntd option.OwnBook=false option.Debug=true "option.Knobs=FIX_COUNT_LOG=1" tc=10+0.1 timemargin=250 \
	-engine "cmd=$NEW" name=cnt option.OwnBook=false tc=10+0.1 timemargin=250 \
	$COMMON -srand 2730 -rounds 50 -games 2 -pgnout file=$W/cntd.pgn notation=san append=false > $W/cntd.log 2>&1
echo "[q273] $(date '+%H:%M') timed leg: $(grep -E '^Games:' $W/cntd.log | tail -1) · warnings $(grep -c 'have option' $W/cntd.log)"
for f in "$T"/err_*.log; do grep -E '^TMCNT' "$f" | tail -1; done > "$W/tmcnt.txt"
du -sh "$T" | show
awk '{ for (i = 1; i <= NF; i++) { split($i, kv, "="); s[kv[1]] += kv[2] } } END { r = s["root_clock_m0"] + s["root_clock_later"]; printf "moves %d · time cut %d (differs %d) · root clock exits m0 %d, later %d · per 1,000 moves: cut %.2f, root exits %.2f\n", s["moves"], s["cut"], s["cut_differs"], s["root_clock_m0"], s["root_clock_later"], 1000 * s["cut"] / (s["moves"] ? s["moves"] : 1), 1000 * r / (s["moves"] ? s["moves"] : 1) }' "$W/tmcnt.txt" | sed 's/^/[q273] TMCNT (engine processes: '"$(wc -l < "$W/tmcnt.txt")"'): /'
echo "[q273] registered: root exits worth a fix knob if >= 1 per 1,000 moves; Debug ablation: cntd >= 40% ⇒ #270's 9% was the /mnt/e writes, < 20% ⇒ Debug itself"

# ── 4 the seed at equal nodes ────────────────────────────────────────────────────────────────────────────────────────────
stopped "the seed STS"
echo "[q273] $(date '+%H:%M') STS1500 equal nodes, ENABLE_QSTANDPAT_SEED=1 at 125000 / 249014 / 500000 in parallel"
for N in 125000 249014 500000; do
	env MAX_DEPTH=64 NODE_LIMIT=$N USE_OPENING_BOOK=0 PRESET=LONG_FORMAT V2_PRESET=shipped ENABLE_QSTANDPAT_SEED=1 nice -n 10 \
		bash "$R" pyrun diagnostics/sts_test.py STS1-STS15_LAN_v3.epd q273_n${N}_seed > "$W/n${N}_seed.out" 2> "$W/n${N}_seed.err" &
done
wait
for N in 125000 249014 500000; do
	echo "[q273] $(date '+%H:%M') @$N seed: $(grep -E 'STS score' "$W/n${N}_seed.out" | tail -1) · base $(grep -E "@$N base" /mnt/e/chess_data/q272.log | grep -oE 'STS score: [0-9]+')"
	bash "$R" pyrun diagnostics/_sts_paired.py q273_n${N}_seed q272_n${N}_base 2>&1 | show
done
echo "[q273] registered: seed prediction STS1500 vs base −40 … +60 at each budget (read with #272's screen verdict)"
echo "[q273] $(date) QUEUE 273 DONE"
