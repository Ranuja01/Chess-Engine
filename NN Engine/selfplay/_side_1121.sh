#!/bin/bash
# 2026-11-21 ~14:05: non-timed work for the owner's gaming window (owner 14:00: "what can run now until about 4"), after the
# s7 build (_build_s7.sh) finishes. Everything here is fixed-depth or node-limited, and it holds #287's timed legs with
# q287.pause while it runs (removed at the end), so no timed game ever overlaps it. A new step starts only while
# gaming.pause exists; once the owner is done, the running step finishes and the script ends.
#  1 S2-4's three fixed-depth checks (dev_notes/active/S2-4-INVESTIGATION-2026-11-21.md §c, predictions registered there):
#    s6pgo, BENCH_PER_POS=1 + Debug, arms PS_TT_FLAG_FIX 0 / 1 on WAC d10 (A), WAC d6 (B), game12 d10 (C). Raw outputs in
#    /mnt/e/chess_data/side_1121/S24_<set><arm>.out; read afterwards.
#  2 First counts for the two count-first hygiene bits (s7pgo, only if its gates passed; ENABLE_LMR_COUNTERS=1 + Debug):
#    default, HYG_FIX=1 (S2-8), HYG_FIX=16 (S2-32), on WAC d10 and game12 d10: per-search [sweep2] qdelta_check and
#    q_underpromo summed, with the [qdelta_permove] fires and qnodes. REGISTERED: S2-8 — checking captures are 2-10% of
#    per-capture delta prunes; S2-32 — bishop / rook promotions are < 0.1% of qnodes on WAC (rare outside endgames).
#    ENABLE_LMR_COUNTERS=1 must leave the default bench unchanged (count-only).
#  3 Pre-played outside-judge legs for the next candidate, #288 = S2-1's CHECK_EXT_FORM=3 vs gen 7.7 on s7pgo (only if its
#    gates passed), node-limited exactly as the queues play them: ours 250k nodes vs SF18 1,100 nodes (seeds 288111-3) and
#    SF1.1 320k nodes (seeds 288211-3), 250 g per leg, UHO, files $O/{g18,s11}_q288_s<seed>_{ctl,cx3}.{log,pgn}, in the
#    queue's leg order. #288 must reuse complete legs (Games: 250) and not delete them. They count only after #288's
#    self-play passes (streamlined judging); otherwise information only.
# Stop: touch /mnt/e/chess_data/side_1121.stop (between steps).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
SF18="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_18_linux/stockfish-ubuntu-x86-64-avx2"
SF1=/mnt/e/chess_data/engines/sf1.1/stockfish-1.1
O=/mnt/e/chess_data/fastchess
OUT=/mnt/e/chess_data/side_1121
G=/mnt/e/chess_data/sv2_speed/game12.txt
S6="$ND/build_uci/chess_engine_uci_s6pgo"
S7="$ND/build_uci/chess_engine_uci_s7pgo"
BLOG=/mnt/e/chess_data/build_s7.log
HOLD=/mnt/e/chess_data/q287.pause
GPAUSE=/mnt/e/chess_data/gaming.pause
STOP=/mnt/e/chess_data/side_1121.stop
N11=320000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"; cd "$ND" || exit 1; rm -f "$STOP"
timed_running() { pgrep -f '[f]astchess.*tc=[0-9]' >/dev/null; }
may_start() { [ -e "$GPAUSE" ] && [ ! -e "$STOP" ] && ! timed_running; }
finish() { rm -f "$HOLD"; echo "[side] $(date '+%H:%M') $1 — q287.pause removed · SIDE 1121 DONE"; exit 0; }
echo "[side] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for BUILD S7 to end"
until grep -qE 'BUILD S7 (DONE|FAILED|NOT RUN)' "$BLOG" 2>/dev/null; do [ -e "$STOP" ] && exit 0; sleep 30; done
S7OK=0; grep -q 'every gate passed — s7pgo usable' "$BLOG" && S7OK=1
echo "[side] $(date '+%H:%M') build s7 ended (usable: $S7OK)"
may_start || finish "the gaming window is over (or a stop file) before step 1"
touch "$HOLD"; sleep 20
timed_running && finish "a timed game is running — nothing run"

# ── 1 S2-4's fixed-depth checks (s6pgo) ──────────────────────────────────────────────────────────────────────────────────
s24() {  # set label, bench command, knobs
	local f="$OUT/S24_$1.out" t=$(date +%s)
	printf 'setoption name Knobs value %s\nsetoption name Debug value true\n%s\nquit\n' "$3" "$2" | timeout 900 "$S6" > "$f" 2>&1
	echo "[side] $(date '+%H:%M') S2-4 check $1 [$3] $2: $(grep -oE '^bench: solved [0-9]+/[0-9]+ nodes [0-9]+' "$f" | tail -1) · $(( $(date +%s) - t )) s"
}
s24 A0 "bench" "BENCH_PER_POS=1"; s24 A1 "bench" "BENCH_PER_POS=1,PS_TT_FLAG_FIX=1"
s24 B0 "bench 6" "BENCH_PER_POS=1"; s24 B1 "bench 6" "BENCH_PER_POS=1,PS_TT_FLAG_FIX=1"
s24 C0 "bench $G 10" "BENCH_PER_POS=1"; s24 C1 "bench $G 10" "BENCH_PER_POS=1,PS_TT_FLAG_FIX=1"

# ── 2 counts for HYG_FIX bits 1 and 16 (s7pgo) ───────────────────────────────────────────────────────────────────────────
[ "$S7OK" = 1 ] || finish "s7pgo not usable — steps 2 and 3 skipped"
cnt() {  # label, bench command, extra knobs
	local f="$OUT/HYG_$1.out" k="ENABLE_LMR_COUNTERS=1${3:+,$3}"
	printf 'setoption name Knobs value %s\nsetoption name Debug value true\n%s\nquit\n' "$k" "$2" | timeout 900 "$S7" > "$f" 2>&1
	echo "[side] $(date '+%H:%M') count $1 [$k] $2: $(grep -oE '^bench: solved [0-9]+/[0-9]+ nodes [0-9]+' "$f" | tail -1) · $(awk '
		/^\[sweep2\]/ { for (i = 1; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "qdelta_check") qc += kv[2]; if (kv[1] == "q_underpromo") up += kv[2] } }
		/\[qdelta_permove\]/ { on = 0; for (i = 1; i <= NF; i++) { if ($i == "[qdelta_permove]") on = 1; split($i, kv, "="); if (on && kv[1] == "fires") df += kv[2]; if (kv[1] == "qnodes") qn += kv[2] } }
		END { printf "delta prunes %d, of them checking %d (%.2f%%) · q_underpromo %d of qnodes %d (%.4f%%)", df, qc, df ? 100 * qc / df : 0, up, qn, qn ? 100 * up / qn : 0 }' "$f")"
}
may_start || finish "the gaming window is over before step 2"
cnt W0 "bench" ""; cnt W1 "bench" "HYG_FIX=1"; cnt W16 "bench" "HYG_FIX=16"
cnt G0 "bench $G 10" ""; cnt G1 "bench $G 10" "HYG_FIX=1"; cnt G16 "bench $G 10" "HYG_FIX=16"

# ── 3 pre-played outside-judge legs for #288 (CHECK_EXT_FORM=3, s7pgo) ───────────────────────────────────────────────────
COMMON="-openings file=$O/openings_uho.epd format=epd order=random -repeat -concurrency 3 -recover -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10"
LEGS="g18:1:ctl g18:1:cx3 s11:1:cx3 s11:1:ctl g18:2:cx3 g18:2:ctl s11:2:ctl s11:2:cx3 g18:3:ctl g18:3:cx3 s11:3:cx3 s11:3:ctl"
for L in $LEGS; do
	IFS=: read -r j s a <<< "$L"
	case $j in g18) seed=$((288110 + s)); opp=(-engine "cmd=$SF18" name=sf18 option.Threads=1 option.Hash=16 nodes=1100) ;;
	            s11) seed=$((288210 + s)); opp=(-engine "cmd=$SF1" name=sf1.1 option.Threads=1 option.Ponder=false option.OwnBook=false option.Hash=16 nodes=$N11) ;; esac
	f=$O/${j}_q288_s${seed}_$a.log
	grep -qE '^Games: 250,' "$f" 2>/dev/null && continue
	may_start || finish "the gaming window is over"
	k=""; [ "$a" = cx3 ] && k="option.Knobs=CHECK_EXT_FORM=3"
	t=$(date +%s)
	nice -n 5 "$FC" -engine "cmd=$S7" name=ours option.OwnBook=false $k nodes=250000 "${opp[@]}" \
		-each tc=inf $COMMON -srand $seed -rounds 125 -games 2 -pgnout file=$O/${j}_q288_s${seed}_$a.pgn notation=san nodes=true append=false > "$f" 2>&1
	echo "[side] $(date '+%H:%M') judge leg $j seed $seed $a: $(grep -E '^Games:' "$f" | tail -1) · warnings $(grep -c 'have option' "$f") · disconnects/stalls $(grep -ciE 'disconnect|stall|not responsive' "$f") · $(( $(date +%s) - t ))s"
done
finish "every #288 judge leg is played"
