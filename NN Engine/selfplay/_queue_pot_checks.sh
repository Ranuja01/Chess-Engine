#!/bin/bash
# 2026-10-16 afternoon, queue #73 — the pre-game POT checks. ≤ 3 processes. Fixed depth (safe any hour).
#  1. d10 on 2,000 ONE-MINOR endings (EGX, SF18 d14; the ONEMINOR cell touches ~3× more game positions than pure pawn endings
#     and had only a static check): base vs the simplified set S (no WIN_RACE) vs cells only.
#     PREDICTION (registered): S improves one-minor MSE a little (−2…−6%), bias toward 0; never worse than base.
#  2. d10 on the 2,313 pawn rows for S (WIN_RACE dropped: static fit +0.9% only). PREDICTION: within 3% of arm A (68.0 own).
#  3. SPEED, solo, 600 pawn rows: NPS base vs S (race + cells cost in pawn endings). PREDICTION: NPS within 3%.
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
P=/mnt/e/chess_data/pe
cd "$ND" || exit 1
echo "[q73] $(date) start"
FP=$(env V2_PRESET=shipped bash "$R" wac fp_v2 2>&1 | grep -E "SOLVED|NODES" | tr '\n' ' ')
echo "$FP" | grep -q "259/300.*47869841" || { echo "[q73] ☠️ FINGERPRINT MISMATCH (expected gen 3) — aborting"; exit 1; }
"/home/ranuja/anaconda3/bin/python" - <<'EOF'
import csv, random
R = [r for r in csv.DictReader(open("/mnt/e/chess_data/pe/potfit_dump.csv")) if r["cls"] == "minor"]
random.Random(1016).shuffle(R)
with open("/mnt/e/chess_data/pe/pe_minor_in.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["fen", "best_uci", "best_cp", "stratum"])
    for r in R[:2000]:
        w.writerow([r["fen"], "", r["sf_cp"], "egx"])
print("pe_minor_in.csv: 2000 one-minor rows")
EOF
NR="RESIGN_THRESHOLD=-99999999"
S="POT_V2_RACE=1 POT_V2_WIN_PAWNEND=61 POT_V2_WIN_ONEMINOR=16 POT_V2_RACE_W_BEST=97 POT_V2_RACE_W_PROMO=210 POT_V2_RACE_W_TEMPI=206 POT_V2_RACE_W_DECISIVE=99"
depth_arm() {   # $1 = input csv, $2 = arm name, $3... = knobs
  in=$1; arm=$2; shift 2
  for s in 0 1 2; do
    ( until timeout 5400 env PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped $NR "$@" bash "$R" pyrun \
          diagnostics/_depth_residual_pass.py IN=$in OUT=$P/d10_$arm.csv SHARD=$s/3 CHUNK=500 > /dev/null 2>&1; \
        [ $? -ne 3 ]; do :; done ) &
  done
  wait
  echo "[q73] $(date) arm $arm: $(cat $P/d10_${arm}_s*of3.csv 2>/dev/null | grep -vc '^fen') rows"
}
depth_arm $P/pe_minor_in.csv mbase
depth_arm $P/pe_minor_in.csv mS $S
depth_arm $P/pe_minor_in.csv mC POT_V2_WIN_PAWNEND=61 POT_V2_WIN_ONEMINOR=16
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read PEIN=$P/pe_minor_in.csv ARMS=mbase,mS,mC
depth_arm $P/pe_in.csv potS $S
bash "$R" pyrun diagnostics/_pe_arms.py MODE=read ARMS=base,potAr,potS
for arm in base S; do
  K=""; [ $arm = S ] && K="$S"
  env DIAG_NO_CLEAR=1 PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 V2_PRESET=shipped $NR $K bash "$R" pyrun diagnostics/_pot_speed.py \
    IN=$P/pe_in.csv N=600 TAG=$arm 2>/dev/null | grep '^SPEED'
done
echo "[q73] $(date) QUEUE 73 DONE"
