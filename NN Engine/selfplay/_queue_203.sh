#!/bin/bash
# 2026-11-11 ~15:00, queue #203: owner-cpp-ideas §15.2 one-off (search v2 plan §7, "next idle hour"). v1's UCI bench wobbles
# 35,310,775-778 from run to run at fixed depth with tables cleared per position; a time abort would move thousands of nodes,
# so a 1-3 node wobble is a defect signature (an uninitialised read, or ordering by an address).
#  1 v1 bench x3 with ASLR on, x3 with ASLR off (setarch -R): if the wobble stops with ASLR off, the cause is
#    address-dependent, which Memcheck cannot see.
#  2 one v1 bench under Memcheck (--track-origins=yes): any uninitialised-value error is the candidate.
# Binary: chess_engine_uci_prof (production flags + the gen 7.1 PGO profile + -g -fno-omit-frame-pointer; the v1 path is the
# same code in gen 7.1 and gen 7.2). Fixed depth, one process at a time, nothing timed. Stop: touch /mnt/e/chess_data/q203.stop
# (checked between steps). Never edit this file or the binary while it runs.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
B="$ND/build_uci/chess_engine_uci_prof"
O=/mnt/e/chess_data/memcheck_v1
STOP=/mnt/e/chess_data/q203.stop
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$O"
stopped() { [ -e "$STOP" ] && { echo "[q203] $(date) stop file present before $1 — QUEUE 203 DONE"; exit 0; }; }
echo "[q203] $(date '+%Y-%m-%d %H:%M:%S') start"
command -v valgrind >/dev/null || { echo "[q203] ☠️ valgrind not found — QUEUE 203 ABORTED"; exit 1; }
command -v setarch >/dev/null || { echo "[q203] ☠️ setarch not found — QUEUE 203 ABORTED"; exit 1; }
[ -x "$B" ] || { echo "[q203] ☠️ no binary $B — QUEUE 203 ABORTED"; exit 1; }
v1bench() { printf 'bench\nquit\n' | env EVAL_ARM=0 "$@" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+ time [0-9]+'; }

# ── 1 the wobble, ASLR on and off ───────────────────────────────────────────────────────────────────────────────────────
for r in 1 2 3; do echo "[q203] v1 bench, ASLR on,  run $r: $(v1bench "$B")"; done
stopped "the ASLR-off runs"
for r in 1 2 3; do echo "[q203] v1 bench, ASLR off, run $r: $(v1bench setarch x86_64 -R "$B")"; done

# ── 2 Memcheck ──────────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "Memcheck"
echo "[q203] $(date '+%H:%M:%S') Memcheck start (expect 1-2 h)"
printf 'bench\nquit\n' | env EVAL_ARM=0 timeout 5h valgrind --tool=memcheck --track-origins=yes --error-limit=no \
	--num-callers=24 --log-file="$O/memcheck.txt" "$B" > "$O/memcheck_out.txt" 2>&1
echo "[q203] $(date '+%H:%M:%S') Memcheck exit $?: $(grep -oE 'solved [0-9]+/300 nodes [0-9]+' "$O/memcheck_out.txt")"
grep -E "ERROR SUMMARY|definitely lost|in use at exit" "$O/memcheck.txt" | sed "s/^/[q203]   /"
echo "[q203] distinct error kinds:"
grep -E "^==[0-9]+== [A-Z][a-z].*(uninitialised|Invalid|Mismatched|overlap)" "$O/memcheck.txt" | sed 's/^==[0-9]*== //' | sort | uniq -c | sed "s/^/[q203]   /"
echo "[q203] $(date) QUEUE 203 DONE"
