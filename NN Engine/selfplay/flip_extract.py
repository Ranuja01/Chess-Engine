# -*- coding: utf-8 -*-
"""Mine eval-FLIP positions from one arm's LOSSES in a head-to-head tournament.

For each game the 'fast' arm lost, reconstruct fast's OWN-POV eval trajectory from `game.jsonl` (only
fast's own moves carry fast's eval; `eval_white_pov` is White-POV → flip when fast is Black), find the
SHARPEST drop window (fast-POV eval falls >= DROP pawns over <= WIN plies, starting from a not-already-lost
position), and emit the FEN at the window START — the position fast still thought was fine right before the
collapse (the blindness point). Writes games/<tag>/flips.csv ranked by drop, for downstream side-by-side
(our-engine-deep vs Stockfish) characterization. jsonl-only, no engine/SF — fast.

Run (from NN Engine/):  python selfplay/flip_extract.py [tag] [drop_pawns=3.0]
"""

import os, sys, json, glob, csv

TAG = sys.argv[1] if len(sys.argv) > 1 else 'overnight_speed'
DROP = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0   # pawns
WIN = 3                                                   # plies in the drop window
ARM = 'fast'
THIS = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(THIS, 'games', TAG)
OUT = os.path.join(G, 'flips.csv')


def piece_count(fen):
    return sum(1 for c in fen.split()[0] if c.isalpha())


rows = []
ngames = nloss = 0
for d in sorted(glob.glob(os.path.join(G, 'game_*'))):
    jl = os.path.join(d, 'game.jsonl')
    if not os.path.exists(jl):
        continue
    meta = result = None
    moves = []
    for line in open(jl):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        t = o.get('type')
        if t == 'meta':
            meta = o
        elif t == 'move':
            moves.append(o)
        elif t == 'result':
            result = o
    if not (meta and result):
        continue
    ngames += 1
    fast_white = (meta.get('white') == ARM)
    r = result.get('result')
    fr = 'draw' if r == '1/2-1/2' else ('win' if ((r == '1-0') == fast_white) else 'loss')
    if fr != 'loss':
        continue
    nloss += 1
    fast_color = 'white' if fast_white else 'black'
    traj = []   # (ply, fen, fast_eval_pawns, uci)
    for m in moves:
        if m.get('color') != fast_color:
            continue                       # only fast's own moves carry fast's eval
        ev = m.get('eval_white_pov')
        if ev is None:
            continue                       # opening/book move
        own = ev if fast_white else -ev
        traj.append((m['ply'], m.get('fen'), own / 1000.0, m.get('uci')))
    if len(traj) < 2:
        continue
    best = None  # (drop, i, j)
    for i in range(len(traj)):
        if traj[i][2] < -3.0:              # already lost at the window start -> not a "flip"
            continue
        for j in range(i + 1, min(i + 1 + WIN, len(traj))):
            drop = traj[j][2] - traj[i][2]
            if drop <= -DROP and (best is None or drop < best[0]):
                best = (drop, i, j)
    if best is None:
        continue
    drop, i, j = best
    ply_i, fen_i, ev_i, _ = traj[i]
    ply_j, _, ev_j, _ = traj[j]
    rows.append({'game': os.path.basename(d), 'ply_start': ply_i, 'ply_end': ply_j,
                 'ev_before': round(ev_i, 2), 'ev_after': round(ev_j, 2), 'drop': round(drop, 2),
                 'pieces': piece_count(fen_i),
                 'fast_move_next': traj[i + 1][3] if i + 1 < len(traj) else '',
                 'fen_start': fen_i})

rows.sort(key=lambda x: x['drop'])
os.makedirs(G, exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game', 'ply_start', 'ply_end', 'ev_before', 'ev_after',
                                      'drop', 'pieces', 'fast_move_next', 'fen_start'])
    w.writeheader()
    w.writerows(rows)

print(f"games={ngames}  {ARM}_losses={nloss}  flips_found={len(rows)}  (drop>={DROP}p over <={WIN} plies, not-already-lost)")
mid = sum(1 for r in rows if r['pieces'] > 12)
print(f"phase of flip: midgame-ish(>12 pieces)={mid}  endgame-ish(<=12)={len(rows)-mid}")
print(f"written -> {OUT}")
print("top 10 sharpest flips (game ply ev_before->ev_after fen):")
for r in rows[:10]:
    print(f"  {r['game']} p{r['ply_start']}->{r['ply_end']} {r['ev_before']:+.1f}->{r['ev_after']:+.1f} (drop {r['drop']:+.1f}, pc{r['pieces']})  {r['fen_start']}")
