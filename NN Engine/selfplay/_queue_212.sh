#!/bin/bash
# 2026-11-12 ~12:20, queue #212: MOVEGEN_FAST correctness (S3, a pure speed item: the generator's piece-type passes with each
# piece's attacks computed once per list, legality by the pin test per piece, moves written straight into the list; check
# positions keep today's path). Starts after #211's Mediocre leg (timed); runs beside its SF18 leg (node-limited, unaffected).
# Builds TEST copies only (build_uci/mgf_verify, build_uci/mgf); production and the UI module are not touched.
#  1 verify build (-DMOVEGEN_FAST_VERIFY: every fast pass is also run the normal way and compared move for move)
#  2 benches on it with MOVEGEN_FAST=1: WAC d10 default, SEARCH_ARM=1, SEE_PER_MOVE=3; the identity sets d10; game24 d12
#  3 plain build, the identity sets with MOVEGEN_FAST=1 on both search arms (diagnostics/_sv2_identity.sh)
# REGISTERED (before any run): every verify line "mismatches 0" with checked > 0; every node count equal to the knob-off
# fingerprint: WAC 247 / 39,861,964 (SEE_PER_MOVE=3: 249 / 37,279,382) · WAC mirrored 38,059,595 · STS300 100,753,502 ·
# STS300 mirrored 98,825,166 · bare kings 6,715,062 · perpetual 41,426 · game24 d12 17,622,774. All pass = result-identical;
# its speed read (PGO, quiet box, alternated) comes later, prediction -2 to -5% time to depth 13 (the 08-25 direct-emit
# attempt alone gave 0-3% nps).
# Stop: touch /mnt/e/chess_data/q212.stop (checked between steps). Never edit this file or the sources while it builds.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
W=/mnt/e/chess_data/mgf
STOP=/mnt/e/chess_data/q212.stop
S=/mnt/e/chess_data/sv2_id
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ND" || exit 1
rm -f "$STOP"; mkdir -p "$W"
show() { sed "s/^/[q212]   /"; }
stopped() { [ -e "$STOP" ] && { echo "[q212] $(date) stop file present before $1 — QUEUE 212 DONE"; exit 0; }; }
echo "[q212] $(date '+%Y-%m-%d %H:%M:%S') queued; waiting for #211's Mediocre leg to finish"
until grep -qE "Mediocre seed 211312 ctl|QUEUE 211 (DONE|ABORTED)" /mnt/e/chess_data/q211.log 2>/dev/null; do stopped "the start"; sleep 30; done
echo "[q212] $(date '+%H:%M') start"

# ── 1 verify build ──────────────────────────────────────────────────────────────────────────────────────────────────────
stopped "the verify build"
OUT=mgf_verify EXTRA_FLAGS="-DMOVEGEN_FAST_VERIFY" bash selfplay/build_uci.sh > "$W/build_verify.out" 2>&1 || { echo "[q212] ☠️ verify build failed — QUEUE 212 ABORTED"; tail -20 "$W/build_verify.out" | show; exit 1; }
V="$ND/build_uci/mgf_verify"

# ── 2 benches on the verify build ───────────────────────────────────────────────────────────────────────────────────────
vb() {  # label, env knobs, bench args
	local out cmd="bench"
	[ -n "$3" ] && cmd="bench $3"
	out=$(printf '%s\nquit\n' "$cmd" | env $2 "$V" 2>/dev/null | grep -E '^bench|movegen_fast verify' | sed -E 's/ time [0-9]+ ms nps [0-9]+//' | tr '\n' ' ')
	echo "[q212] $1: $out"
}
vb "verify, knob off (want 247 / 39861964, checked 0)" "MOVEGEN_FAST=0" ""
for k in "MOVEGEN_FAST=1" "MOVEGEN_FAST=1 SEARCH_ARM=1" "MOVEGEN_FAST=1 SEE_PER_MOVE=3"; do
	stopped "verify $k"
	vb "verify WAC d10 [$k]" "$k" ""
done
for f in wac_mirror sts300 sts300_mirror barekings perpetual_probe; do
	stopped "verify $f"
	vb "verify $f d10 [MOVEGEN_FAST=1]" "MOVEGEN_FAST=1" "$S/$f.txt 10"
done
vb "verify game24 d12 [MOVEGEN_FAST=1]" "MOVEGEN_FAST=1" "/mnt/e/chess_data/sv2_profile/game24.txt 12"

# ── 3 plain build + the identity sets on both arms ──────────────────────────────────────────────────────────────────────
stopped "the plain build"
OUT=mgf bash selfplay/build_uci.sh > "$W/build_plain.out" 2>&1 || { echo "[q212] ☠️ plain build failed — QUEUE 212 ABORTED"; tail -20 "$W/build_plain.out" | show; exit 1; }
f0=$(printf 'bench\nquit\n' | "$ND/build_uci/mgf" 2>/dev/null | grep -oE 'solved [0-9]+/300 nodes [0-9]+')
echo "[q212] plain build, default (want 247 / 39861964): [$f0]"
echo "[q212] == identity sets, MOVEGEN_FAST=1 on both search arms (compare with the registered fingerprints)"
MOVEGEN_FAST=1 bash diagnostics/_sv2_identity.sh "$ND/build_uci/mgf" "$S" 10 2>&1 | show
echo "[q212] $(date) QUEUE 212 DONE"
