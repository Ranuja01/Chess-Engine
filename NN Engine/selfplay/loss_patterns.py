# -*- coding: utf-8 -*-
"""Classify one arm's LOSSES in a head-to-head tournament by start-of-play eval + trajectory.

Reads each game.pgn (which carries per-move `{ base|fast +x.xx/dN }` comments, White-POV) and, for every
game the 'fast' arm lost, reconstructs fast's OWN-POV eval trajectory (flip the White-POV comment when fast
is Black) to bucket the loss:
  - already_losing      : fast's own eval was already bad at the FIRST post-book move (a losing start)
  - squandered_winning  : fast's own eval peaked clearly winning (>=+1.5) yet it still lost
  - collapse_from_okayish: never clearly winning, but a sharp single-window drop (>= -3.0 over <=3 ply) sank it
                           (the "didn't see it coming" tactical-blindness signature)
  - slow_grind          : none of the above (gradual)
Also reports the start-of-play eval distribution and loss-length stats.

Run (from NN Engine/):  python selfplay/loss_patterns.py [tag]   (default tag overnight_speed)
"""

import os, re, glob, sys, statistics

TAG = sys.argv[1] if len(sys.argv) > 1 else 'overnight_speed'
THIS = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(THIS, 'games', TAG)

evre = re.compile(r'\{\s*(base|fast)\s+([+-]?\d+(?:\.\d+)?)/d')
hw = re.compile(r'\[White "(\w+)"\]')
hb = re.compile(r'\[Black "(\w+)"\]')
hr = re.compile(r'\[Result "([^"]+)"\]')

wins = losses = draws = 0
start_bucket = {'losing (<=-1.0)': 0, 'equalish (-1..+1)': 0, 'better (>=+1.0)': 0}
patt = {'already_losing': 0, 'squandered_winning': 0, 'collapse_from_okayish': 0, 'slow_grind': 0}
loss_plies = []

for d in sorted(glob.glob(os.path.join(G, 'game_*'))):
    p = os.path.join(d, 'game.pgn')
    if not os.path.exists(p):
        continue
    t = open(p).read()
    mw, mb, mr = hw.search(t), hb.search(t), hr.search(t)
    if not (mw and mb and mr):
        continue
    fw = (mw.group(1) == 'fast')          # is fast White?
    r = mr.group(1)
    fr = 'draw' if r == '1/2-1/2' else ('win' if ((r == '1-0') == fw) else 'loss')
    if fr == 'win':
        wins += 1
    elif fr == 'loss':
        losses += 1
    else:
        draws += 1
    if fr != 'loss':
        continue
    evs = []
    for side, val in evre.findall(t):
        v = float(val)
        own = v if fw else -v             # PGN comment is White-POV -> fast-own flips when fast is Black
        evs.append(max(-50.0, min(50.0, own)))
    if not evs:
        continue
    loss_plies.append(len(evs))
    start, peak = evs[0], max(evs)
    if start <= -1.0:
        start_bucket['losing (<=-1.0)'] += 1
    elif start >= 1.0:
        start_bucket['better (>=+1.0)'] += 1
    else:
        start_bucket['equalish (-1..+1)'] += 1
    worst = 0.0
    for i in range(len(evs)):
        for j in range(i + 1, min(i + 4, len(evs))):
            worst = min(worst, evs[j] - evs[i])
    if start <= -1.5:
        patt['already_losing'] += 1
    elif peak >= 1.5:
        patt['squandered_winning'] += 1
    elif worst <= -3.0:
        patt['collapse_from_okayish'] += 1
    else:
        patt['slow_grind'] += 1

n = max(1, losses)
print(f"fast: W={wins} L={losses} D={draws}")
print("--- fast LOSSES: start-of-play eval (fast POV, first post-book move) ---")
for k, v in start_bucket.items():
    print(f"  {k:20} {v:4}  ({100*v/n:.0f}%)")
print("--- fast LOSSES: pattern ---")
for k, v in patt.items():
    print(f"  {k:24} {v:4}  ({100*v/n:.0f}%)")
if loss_plies:
    short = sum(1 for x in loss_plies if x <= 30)
    print(f"--- loss length (plies): median {statistics.median(loss_plies):.0f}, mean {statistics.mean(loss_plies):.0f}, "
          f"min {min(loss_plies)}, max {max(loss_plies)};  short (<=30): {short} ({100*short/n:.0f}%)")
