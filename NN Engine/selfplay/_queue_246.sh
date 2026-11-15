#!/bin/bash
# 2026-11-15 ~11:00, queue #246: ship gen 7.4 = gen 7.3 + D7 (SV2_VERIFY_DEEPER 1 in the shipped preset). Owner 11-15: "it seems D7
# can ship yes" (#241: self-play +38.4, 95% +25.3 … +51.6 over 2,001 g; SF18 equal nodes z +0.12; Mediocre z +0.50). No timed work
# (weekend afternoon): builds, fixed-node benches and a 2-game smoke only.
#  1 PGO chess_engine_uci_g74pgo trained on its own default, profile /mnt/e/chess_data/pgo_g74. Gates: default 250 / 37,320,382
#    (the default bench CHANGES, as a ship must show); SV2_VERIFY_DEEPER=0 = gen 7.3 249 / 37,279,382; SEARCH_ARM=0 the same
#    (D7 lives in search v2 only); + SEE_PER_MOVE=0 = gen 7.2 247 / 39,861,964; + MOPUP_V2=0 = gen 7.1 247 / 39,852,210;
#    V2_SHIP_GEN=6 = gen 6.7 251 / 36,859,850; v1 250 / 35,310,775-778; id "gen 7.4".
#  2 Install as build_uci/chess_engine_uci (production g73spgo stays as the backup); a 2-game smoke with 0 option warnings.
#  3 Cython module rebuilt from the committed sources (backup /mnt/e/chess_data/so_backup_g74/): shipped 250 / 37,320,382 with
#    SV2_VERIFY_DEEPER=1 in its toggles line; SV2_VERIFY_DEEPER=0 249 / 37,279,382; v1 250 / 35,310,778.
# Stop: touch /mnt/e/chess_data/q246.stop (between steps). Never edit the sources before "build done".
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
W=/mnt/e/chess_data/q246
P=/mnt/e/chess_data/pgo_g74
BK=/mnt/e/chess_data/so_backup_g74
PROD="$ND/build_uci/chess_engine_uci"
PRODS="$ND/build_uci/chess_engine_uci_g73spgo"
NEW="$ND/build_uci/chess_engine_uci_g74pgo"
STOP=/mnt/e/chess_data/q246.stop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q246]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q246] $(date) stop file present before $1 — QUEUE 246 DONE"; exit 0; }; }
fp() { local k=""; [ -n "$2" ] && k="setoption name Knobs value $2"; printf '%s\nbench\nquit\n' "$k" | "$1" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+'; }
echo "[q246] $(date '+%Y-%m-%d %H:%M:%S') start"
cmp -s "$PROD" "$PRODS" && echo "[q246] production == chess_engine_uci_g73spgo" || { echo "[q246] ☠️ production is not g73spgo — QUEUE 246 ABORTED"; exit 1; }

# ── 1 PGO build and gates ─────────────────────────────────────────────────────────────────────────────────────────────────
rm -rf "$P"; mkdir -p "$P"
OUT=chess_engine_uci_g74gen PGO=gen PGO_DIR=$P nice -n 5 bash selfplay/build_uci.sh > "$W/gen.out" 2>&1 || { echo "[q246] ☠️ gen build failed — QUEUE 246 ABORTED"; tail -5 "$W/gen.out" | show; exit 1; }
nice -n 5 "$FC" -engine cmd="$ND/build_uci/chess_engine_uci_g74gen" name=a option.OwnBook=false st=0.25 \
	-engine cmd="$ND/build_uci/chess_engine_uci_g74gen" name=b option.OwnBook=false st=0.25 \
	-openings file=$O/openings_uho.epd format=epd order=random -srand 746 -rounds 4 -games 2 -concurrency 1 > "$W/train.log" 2>&1
echo "[q246] training: $(grep -E '^Games:' "$W/train.log" | tail -1) · profile files $(find "$P" -name '*.gcda' | wc -l)"
OUT=chess_engine_uci_g74pgo PGO=use PGO_DIR=$P nice -n 5 bash selfplay/build_uci.sh > "$W/use.out" 2>&1 || { echo "[q246] ☠️ use build failed — QUEUE 246 ABORTED"; tail -5 "$W/use.out" | show; exit 1; }
echo "[q246] $(date '+%H:%M') build done"
d=$(fp "$NEW" ""); d0=$(fp "$NEW" SV2_VERIFY_DEEPER=0); a0=$(fp "$NEW" SEARCH_ARM=0)
g72=$(fp "$NEW" SV2_VERIFY_DEEPER=0,SEE_PER_MOVE=0); g71=$(fp "$NEW" SV2_VERIFY_DEEPER=0,SEE_PER_MOVE=0,MOPUP_V2=0); g67=$(fp "$NEW" V2_SHIP_GEN=6)
echo "[q246] default = gen 7.4 (want 250 / 37320382):      [$d]"
echo "[q246] SV2_VERIFY_DEEPER=0 = gen 7.3 (249 / 37279382): [$d0]"
echo "[q246] SEARCH_ARM=0 (249 / 37279382):               [$a0]"
echo "[q246] + SEE_PER_MOVE=0 = gen 7.2 (247 / 39861964):   [$g72]"
echo "[q246] + MOPUP_V2=0 = gen 7.1 (247 / 39852210):       [$g71]"
echo "[q246] V2_SHIP_GEN=6 = gen 6.7 (251 / 36859850):      [$g67]"
echo "[q246] v1 EvalArm 0 (250 / 35310775-778):             $(printf 'setoption name EvalArm value 0\nbench\nquit\n' | "$NEW" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')"
echo "[q246] default bench line: $(printf 'bench\nquit\n' | "$NEW" 2>/dev/null | grep '^bench')"
id=$(printf 'uci\nquit\n' | "$NEW" 2>/dev/null | grep '^id name'); echo "[q246] $id"
if ! { [ "$d" = "solved 250/300 nodes 37320382" ] && [ "$d0" = "solved 249/300 nodes 37279382" ] && [ "$a0" = "$d0" ] \
	&& [ "$g72" = "solved 247/300 nodes 39861964" ] && [ "$g71" = "solved 247/300 nodes 39852210" ] \
	&& [ "$g67" = "solved 251/300 nodes 36859850" ] && echo "$id" | grep -q 'gen 7.4'; }; then
	echo "[q246] ☠️ gates failed — production NOT replaced, QUEUE 246 ABORTED"; exit 1
fi

# ── 2 install and smoke ───────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the install"
cp "$NEW" build_uci/.chess_engine_uci.tmp && mv -f build_uci/.chess_engine_uci.tmp "$PROD" && cmp -s "$PROD" "$NEW" \
	&& echo "[q246] $(date) INSTALLED: build_uci/chess_engine_uci == chess_engine_uci_g74pgo (backup g73spgo)" \
	|| { echo "[q246] ☠️ install failed — QUEUE 246 ABORTED"; exit 1; }
echo "[q246] installed bench: $(printf 'bench\nquit\n' | "$PROD" 2>/dev/null | grep '^bench')"
timeout 180 nice -n 5 "$FC" -engine "cmd=$PROD" name=g74 option.OwnBook=false st=0.05 -engine "cmd=$PRODS" name=g73s option.OwnBook=false st=0.05 \
	-openings file=$O/openings_uho.epd format=epd order=random -rounds 1 -games 2 -concurrency 1 > "$W/smoke.log" 2>&1
echo "[q246] smoke: $(grep -E '^Games:' "$W/smoke.log" | tail -1) · warnings $(grep -c 'have option' "$W/smoke.log")"

# ── 3 Cython module ───────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the Cython build"
mkdir -p "$BK" && cp ChessAI.*.so "$BK"/ && echo "[q246] .so backed up to $BK" || { echo "[q246] ☠️ .so backup failed — QUEUE 246 ABORTED"; exit 1; }
nice -n 5 bash "$R" build 2>&1 | grep -iE 'error|copying|FAILED' | tail -n 3 | show
ls ChessAI.*.so >/dev/null 2>&1 || { echo "[q246] ☠️ Cython build left no .so — restoring"; cp "$BK"/ChessAI.*.so .; exit 1; }
bash "$R" wac fp_g74 V2_PRESET=shipped > "$W/cy_default.txt" 2>&1
bash "$R" wac fp_g74_d0 V2_PRESET=shipped SV2_VERIFY_DEEPER=0 > "$W/cy_d0.txt" 2>&1
bash "$R" wac fp_g74_v1 > "$W/cy_v1.txt" 2>&1
echo "[q246] Cython shipped (want 250 / 37320382):              $(grep -E '^SOLVED|^NODES|^EBF' "$W/cy_default.txt" | tr '\n' ' ')· $(grep -oE ' SV2_VERIFY_DEEPER=[0-9]+' "$W/cy_default.txt" | head -1)"
echo "[q246] Cython SV2_VERIFY_DEEPER=0 (want 249 / 37279382):  $(grep -E '^SOLVED|^NODES|^EBF' "$W/cy_d0.txt" | tr '\n' ' ')"
echo "[q246] Cython v1 (want 250 / 35310778):                   $(grep -E '^SOLVED|^NODES|^EBF' "$W/cy_v1.txt" | tr '\n' ' ')"
echo "[q246] $(date) QUEUE 246 DONE"
