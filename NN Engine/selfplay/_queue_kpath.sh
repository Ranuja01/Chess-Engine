#!/bin/bash
# 2026-10-18, queue #79 — KING-PATH race discrimination test (POT proper), on the CANDIDATE base (gen 3 + C64).
# _pot_race_proto.py MODE=moves SET=light: 5,974 SF18 multi-PV light endgames, 5 game folds; static vs static + race.
#   KPATH=1 race distances along real king routes (no own pawns, no enemy-pawn-attacked squares, not next to the enemy king)
#   KPATH=0 straight-line distances (control on the same base)
# KEEP (pre-registered, memory pot-may-not-work-use-stopping-points): KPATH=1's paired keep test z ≥ +2 on critical pairs
# (race fixes vs breaks the static ordering), else PARK. PREDICTION: KPATH=0 z ≈ 0; KPATH=1 z +1…+3 (uncertain).
# 2 processes, Python + static evals only (light).
ND="/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine"
R="$ND/selfplay/overnight_runner.sh"
cd "$ND" || exit 1
echo "[q79] $(date) start"
for k in 0 1; do
  ( bash "$R" pyrun diagnostics/_pot_race_proto.py V2_PRESET=candidate SET=light MODE=moves KPATH=$k 2>&1 \
      | grep -vE '^\[(toggles|c1|c3|kauf)\]|ABSENT keys' > /mnt/e/chess_data/q79_kpath$k.out ) &
done
wait
for k in 0 1; do echo "[q79] KPATH=$k:"; cat /mnt/e/chess_data/q79_kpath$k.out; done
echo "[q79] $(date) QUEUE 79 DONE"
