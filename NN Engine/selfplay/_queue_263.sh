#!/bin/bash
# 2026-11-17 ~11:00, queue #263: GEN 7.6 = gen 7.5 + PROMOTE (SV2_VERIFY_PROMOTE 1), chained after #262 (C4's self-play).
# OWNER 11-17 ~10:55: "Ship after C4". #257 on gen 7.5 (registered reads, all pass): self-play 2,001 g +18.1 (95% +6.3 … +30.0);
# side check with the pairs slowed by the owner's gaming dropped (00:15-00:50, 01:05-02:15), 1,391 g +20.0 (+6.0 … +34.1);
# SF18 at equal nodes +1.40 pp (z +0.90); Mediocre +1.17 pp (z +0.59). On gen 7.4 (#248 + #251) it was +13.6 (+0.5 … +26.7).
#  1 Wait for #262 to end. Production must be g75pgo; nothing else running.
#  2 Shipped preset += SV2_VERIFY_PROMOTE 1 (search_engine.cpp, after FIX_MIN_RD1 1); id "AlphaUX (v2 gen 7.6)". PGO build
#    chess_engine_uci_g76pgo (profile /mnt/e/chess_data/pgo_g76). GATES: default bench 246 / 35,138,434 (= #257's PROMOTE knob
#    on g75pgo) == the knob on g76pgo == the knob on g75pgo == the knob on c4pgo; SV2_VERIFY_PROMOTE=0 = gen 7.5 252 / 35,235,460;
#    eval verify EVAL_MAT_CACHE=2 == default; v1 250 / 35,310,775-778; the id; game12 d10 verify == default. C4's knobs on
#    gen 7.6 are recorded (not gated) for C4's confirmation run.
#  3 INSTALL (backup = g75pgo), Cython rebuild (pre-7.6 .so backed up to /mnt/e/chess_data/so_backup_g76) with its WAC
#    fingerprint and EBF, then a pinned bench x9 on core 5, g76 / g75 in alternating order (ledger). The bench waits out the
#    20:50-00:15 gaming window unless q263.go.
# Stop: touch /mnt/e/chess_data/q263.stop (between steps). Never edit this file, the binaries or the sources while it runs; the
# sources are free after "Cython build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q263
P=/mnt/e/chess_data/pgo_g76
G=/mnt/e/chess_data/sv2_speed/game12.txt
PROD="$ND/build_uci/chess_engine_uci"
G75="$ND/build_uci/chess_engine_uci_g75pgo"
C4="$ND/build_uci/chess_engine_uci_c4pgo"
NEW="$ND/build_uci/chess_engine_uci_g76pgo"
KP="SV2_VERIFY_PROMOTE=1"
BK=/mnt/e/chess_data/so_backup_g76
LOG=/mnt/e/chess_data/q263.log
Q262=/mnt/e/chess_data/q262.log
STOP=/mnt/e/chess_data/q263.stop
GO=/mnt/e/chess_data/q263.go
SE="$ND/search_engine.cpp"
UM="$ND/uci_main.cpp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP" "$GO"; mkdir -p "$W"
show() { sed "s/^/[q263]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q263] $(date) stop file before $1 — QUEUE 263 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>&1 | grep -oE 'solved [0-9]+/300 nodes [0-9]+|mismatch.*' | head -1; }
busy() { pgrep -f "[f]astchess-linux-x86-64/fastchess" >/dev/null || pgrep -f "[s]ts_test|[t]actical_test|[v]algrind|[_]ks_footprint|[b]uild_uci.sh" >/dev/null; }
evening() { local n; n=$((10#$(date +%H%M))); [ ! -e "$GO" ] && { [ $n -ge 2050 ] || [ $n -lt 15 ]; }; }

# ── 1 wait for #262, then the machine and production as expected ─────────────────────────────────────────────────────────
echo "[q263] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #262 to end"
until grep -qE 'QUEUE 262 (DONE|ABORTED)' "$Q262" || ! pgrep -f "[_]queue_262.sh" >/dev/null; do stopped "the start"; sleep 60; done
sleep 60
stopped "the build"
echo "[q263] $(date '+%H:%M') start · #262 ended: $(grep -E 'QUEUE 262' "$Q262" | tail -1 | sed 's/^\[q262\] //')"
cmp -s "$PROD" "$G75" || { echo "[q263] ☠️ production is not g75pgo — QUEUE 263 ABORTED (nothing changed)"; exit 1; }
[ -x "$C4" ] || { echo "[q263] ☠️ c4pgo is missing — QUEUE 263 ABORTED"; exit 1; }
busy && { echo "[q263] ☠️ another job is running — QUEUE 263 ABORTED (nothing changed)"; exit 1; }

# ── 2 the gen 7.6 preset, build, gates ────────────────────────────────────────────────────────────────────────────────────
[ "$(grep -c '^ *Config::FIX_MIN_RD1 = 1;$' "$SE")" = 1 ] && [ "$(grep -c 'Config::SV2_VERIFY_PROMOTE = 1;' "$SE")" = 0 ] \
	&& [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.5)");' "$UM")" = 1 ] || { echo "[q263] ☠️ the sources are not as expected — QUEUE 263 ABORTED (nothing changed)"; exit 1; }
cp "$SE" "$W/search_engine.cpp.g75" && cp "$UM" "$W/uci_main.cpp.g75" || { echo "[q263] ☠️ source backup failed — QUEUE 263 ABORTED"; exit 1; }
restore_src() { cp "$W/search_engine.cpp.g75" "$SE"; cp "$W/uci_main.cpp.g75" "$UM"; echo "[q263] sources restored to gen 7.5"; }
cat > "$W/preset.txt" <<'EOF'
                    // ── GEN 7.6: a shallower near-miss re-search that moves the result to the other side of the node's bound
                    // is settled by one search at full depth, so a shallower result never decides the move alone. Self-play
                    // vs gen 7.5 +18.1 (95% +6.3 … +30.0) over 2,001 games, SF18 at equal nodes +1.40 pp (z +0.90), Mediocre
                    // +1.17 pp (OPTIMIZATION_LOG). Set before the start-up read, so SV2_VERIFY_PROMOTE=0 restores gen 7.5.
                    Config::SV2_VERIFY_PROMOTE = 1;
EOF
sed -i "/^ *Config::FIX_MIN_RD1 = 1;\$/r $W/preset.txt" "$SE"
sed -i 's/say("id name AlphaUX (v2 gen 7\.5)");/say("id name AlphaUX (v2 gen 7.6)");/' "$UM"
[ "$(grep -c '^ *Config::SV2_VERIFY_PROMOTE = 1;$' "$SE")" = 1 ] && [ "$(grep -c 'say("id name AlphaUX (v2 gen 7.6)");' "$UM")" = 1 ] \
	|| { echo "[q263] ☠️ the preset edit did not apply — QUEUE 263 ABORTED"; restore_src; exit 1; }
echo "[q263] $(date '+%H:%M') preset written (search_engine.cpp +5 lines after FIX_MIN_RD1 1, id gen 7.6); PGO build"
rm -rf "$P"; mkdir -p "$P"
if OUT=chess_engine_uci_g76gen PGO=gen PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1; then
	nice -n 10 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g76gen" name=a option.OwnBook=false st=0.25 \
		-engine cmd="$ND/build_uci/chess_engine_uci_g76gen" name=b option.OwnBook=false st=0.25 \
		-openings file=$O/openings_uho.epd format=epd order=random -srand 763 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
	OUT=chess_engine_uci_g76pgo PGO=use PGO_DIR=$P nice -n 10 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q263] ☠️ PGO use build failed — QUEUE 263 ABORTED"; grep -iE 'error' "$W/use.out" | head -8 | show; restore_src; exit 1; }
else
	echo "[q263] ☠️ PGO gen build failed — QUEUE 263 ABORTED"; grep -iE 'error' "$W/gen.out" | head -8 | show; restore_src; exit 1
fi
echo "[q263] $(date '+%H:%M') UCI build done; training: $(grep -E '^Games:' "$W/train.log" | tail -1)"
ok=1
a=$(fp "$NEW" ""); k=$(fp "$NEW" "$KP"); e=$(fp "$G75" "$KP"); c=$(fp "$C4" "$KP"); off=$(fp "$NEW" "SV2_VERIFY_PROMOTE=0"); v=$(fp "$NEW" "EVAL_MAT_CACHE=2")
echo "[q263] g76pgo default [$a] (want 246 / 35138434) · knob on g76 [$k] · on g75pgo [$e] · on c4pgo [$c] (all == default)"
echo "[q263] g76pgo SV2_VERIFY_PROMOTE=0 [$off] (= gen 7.5, 252 / 35235460) · eval verify EVAL_MAT_CACHE=2 [$v] (== default)"
[ "$a" = "solved 246/300 nodes 35138434" ] && [ "$k" = "$a" ] && [ "$e" = "$a" ] && [ "$c" = "$a" ] && [ "$v" = "$a" ] && [ "$off" = "solved 252/300 nodes 35235460" ] || ok=0
v1=$(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
ID=$(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep -m1 '^id name')
echo "[q263] v1 [$v1] (250 / 35310775-778) · $ID"
echo "$v1" | grep -qE 'solved 250/300 nodes 3531077[5-8]' || ok=0
echo "$ID" | grep -qF 'AlphaUX (v2 gen 7.6)' || ok=0
gv=$(printf "setoption name Knobs value EVAL_MAT_CACHE=2\nbench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+|mismatch.*' | head -1)
gb=$(printf "bench $G 10\nquit\n" | "$NEW" 2>&1 | grep -oE 'solved [0-9]+/[0-9]+ nodes [0-9]+' | head -1)
echo "[q263] game12 d10: verify [$gv] · default [$gb]"
[ -n "$gb" ] && [ "$gv" = "$gb" ] || ok=0
echo "[q263] C4 on gen 7.6 (recorded, not gated): SV2_HIST_ALLNODE_GATE=1 [$(fp "$NEW" "SV2_HIST_ALLNODE_GATE=1")] · =2 [$(fp "$NEW" "SV2_HIST_ALLNODE_GATE=2")]"
[ $ok = 1 ] || { echo "[q263] ☠️ a gate failed — no install — QUEUE 263 ABORTED"; restore_src; exit 1; }

# ── 3 install, Cython, pinned bench ───────────────────────────────────────────────────────────────────────────────────────
stopped "the install"
cp "$NEW" "$PROD" && cmp -s "$NEW" "$PROD" || { echo "[q263] ☠️ install copy failed — restoring g75pgo"; cp "$G75" "$PROD"; restore_src; exit 1; }
echo "[q263] $(date '+%H:%M') INSTALLED: production == g76pgo: bench [$(fp "$PROD" "")] · $(printf 'uci\nquit\n' | "$PROD" 2>/dev/null | grep -m1 '^id name')"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q263] pre-7.6 .so backed up to $BK" || { echo "[q263] ☠️ .so backup failed — Cython not rebuilt (UCI is gen 7.6)"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q263] ☠️ Cython build left no .so — restoring the pre-7.6 .so"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac q263_fp V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac q263_fp_75 V2_PRESET=shipped SV2_VERIFY_PROMOTE=0 > "$W/cy_75.txt" 2>&1
bash "$R" wac q263_fp_v1 > "$W/cy_v1.txt" 2>&1
echo "[q263] $(date '+%H:%M') Cython build done — the sources are free to edit"
cyr() { grep -E '^SOLVED|^NODES|^EBF' "$1" | tr '\n' ' '; }
echo "[q263] Cython shipped [$(cyr "$W/cy_default.txt")] (want 246 / 35138434) · PROMOTE off [$(cyr "$W/cy_75.txt")] (gen 7.5 252 / 35235460 / EBF 3.701) · v1 [$(cyr "$W/cy_v1.txt")] (250 / 35310778)"
grep -q 'Solved 246/300' "$W/cy_default.txt" && grep -q '^NODES: 35138434' "$W/cy_default.txt" || echo "[q263] ☠️ the Cython module's shipped fingerprint differs from the UCI binary's — tell the owner (games use the UCI binary)"
until ! evening; do stopped "the pinned bench"; sleep 60; done
bench_one() {  # round, arm
	local out b
	case $2 in g76) b="$NEW" ;; g75) b="$G75" ;; esac
	out=$(printf 'bench\nquit\n' | taskset -c 5 "$b" 2>/dev/null | grep '^bench')
	echo "[q263]   r$1 $2 $out"
}
bt() { grep -E "^\[q263\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $10}' | sort; }
bn() { grep -E "^\[q263\]   r[0-9]+ $1 bench" "$LOG" | awk '{print $2, $13}' | sort; }
med() { sort -g | awk '{a[NR] = $1} END { if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2 }'; }
busy && echo "[q263] ☠️ another job is running — the pinned bench below is disturbed"
echo "[q263] $(date '+%H:%M') == pinned bench x9 on core 5, g76 / g75 in alternating order (ledger)"
for i in 1 2 3 4 5 6 7 8 9; do
	stopped "bench round $i"
	if [ $((i % 2)) = 1 ]; then o="g76 g75"; else o="g75 g76"; fi
	for x in $o; do bench_one $i $x; done
done
for x in g76 g75; do echo "[q263] median $x: $(bt $x | awk '{print $2}' | med) ms · nps $(bn $x | awk '{print $2}' | med) over $(bt $x | wc -l) runs"; done
echo "[q263] within-round g76 / g75: time median $(join <(bt g76) <(bt g75) | awk '{print $2 / $3}' | med) (g76 faster in $(join <(bt g76) <(bt g75) | awk '$2 < $3 {n++} END {print n + 0}') of 9) · nps median $(join <(bn g76) <(bn g75) | awk '{print $2 / $3}' | med)"
echo "[q263] $(date '+%H:%M') GEN 7.6 SHIPPED"
echo "[q263] $(date) QUEUE 263 DONE"
