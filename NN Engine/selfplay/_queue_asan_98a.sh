#!/bin/bash
# 2026-10-24, queue #98a: AddressSanitizer + UBSan crash hunt on the depth-cap fix source (self-play, st=1, 4 games at a time,
# -recover so a crash does not end the run). Every sanitizer report lands in its own file under E:\chess_data\asan\.
# Registered: 0 ASan reports (the depth cap closes the known path); a report = a further out-of-bounds bug, with its stack.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
FC=/mnt/e/chess_data/engines/fastchess/fastchess-linux-x86-64/fastchess
O=/mnt/e/chess_data/fastchess
A=/mnt/e/chess_data/asan
cd "$ND" || exit 1
mkdir -p "$A"
B="$ND/build_uci/chess_engine_uci_asan2"
echo "[q98a] $(date) build"
EXTRA_FLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -D_GLIBCXX_ASSERTIONS' OUT=chess_engine_uci_asan2 bash selfplay/build_uci.sh || exit 1
export ASAN_OPTIONS=log_path=$A/asan:detect_leaks=0:halt_on_error=1
export UBSAN_OPTIONS=print_stacktrace=1:log_path=$A/ubsan:halt_on_error=0
echo "[q98a] $(date) self-play, 160 games"
timeout 14400 nice -n 5 "$FC" -engine cmd="$B" name=a option.OwnBook=false st=1 timemargin=500 restart=on \
  -engine cmd="$B" name=b option.OwnBook=false st=1 timemargin=500 restart=on \
  -openings file=$O/openings_uho.epd format=epd order=random -srand 981 -repeat -rounds 80 -games 2 -concurrency 4 -recover \
  -resign movecount=3 score=1000 -draw movenumber=40 movecount=8 score=10 \
  -pgnout file=$O/asan_98a.pgn notation=san append=false > $O/asan_98a.log 2>&1
echo "[q98a] $(date) done (rc $?): $(grep -E '^Games:' $O/asan_98a.log | tail -1)  disconnects/stalls: $(grep -ciE 'disconnect|stall|not responsive' $O/asan_98a.log)"
echo "[q98a] sanitizer reports: $(ls $A 2>/dev/null | wc -l) files"
for f in $A/*; do [ -f "$f" ] && { echo "--- $f"; grep -E 'ERROR|runtime error|#[0-7] ' "$f" | head -10; }; done
echo "[q98a] $(date) QUEUE 98a DONE"
