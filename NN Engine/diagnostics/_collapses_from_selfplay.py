# -*- coding: utf-8 -*-
"""Extract collapses POST-HOC from already-played self-play games (gate/SPRT tags), which write game.jsonl
but no collapses.csv -- only the vs_sf harness detects collapses inline.

Replicates `selfplay/vs_sf.py::extract_collapse`: a collapse = a side's own eval peaked >= WIN_THRESH (it
believed it was clearly winning) but that side did NOT score 1. The decision_fen is the position it CHOSE
the peak move from, i.e. the position whose evaluation was confidently wrong.

⚠️ READ THIS BEFORE TRUSTING THE CORPUS. Self-play collapses are NOT equivalent to vs_sf collapses:
  - The opponent shares our blind spots, so positions our eval misjudges are not reliably PUNISHED the way
    a stronger engine punishes them. Expect under-sampling of exactly the errors we care most about.
  - In an A/B gate both engines are ~identical, so the sample is not biased toward the tested knob, but it
    IS biased toward positions where our shared eval is confidently wrong AND the game still turned.
  - Gate games run at LIGHTNING, so the peak evals are shallow and noisier than vs_sf's.
It is still a legitimate "our eval said winning and it wasn't" signal, and it is FREE (games already played).
Treat conclusions as hypotheses to confirm against a fresh vs_sf gather.

Emits a collapses.csv with the same columns classify_collapses.py / collapse_term_attribution.py expect.

Run: bash <runner> pyrun diagnostics/_collapses_from_selfplay.py TAG=sprt_onset6 [WIN_THRESH=2000] [SIDE=both]
"""
import os, sys, json, csv, glob

OPTS = {}
for a in sys.argv[1:]:
    if '=' in a:
        k, v = a.split('=', 1); OPTS[k] = v

TAG = OPTS.get('TAG', 'sprt_onset6')
WIN_THRESH = int(OPTS.get('WIN_THRESH', 2000))     # millipawns, same default as vs_sf --win-threshold
SIDE = OPTS.get('SIDE', 'both').lower()            # both | white | black | <label>

THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
GAMES = os.path.join(ENGINE, 'selfplay', 'games', TAG)
if not os.path.isdir(GAMES):
    print("!! no such tag dir: %s" % GAMES); sys.exit(1)

SCORE = {'1-0': (1.0, 0.0), '0-1': (0.0, 1.0), '1/2-1/2': (0.5, 0.5)}
rows, n_games, n_used = [], 0, 0

for d in sorted(glob.glob(os.path.join(GAMES, 'game_*'))):
    jp = os.path.join(d, 'game.jsonl')
    if not os.path.exists(jp):
        continue
    meta, moves, result = None, [], None
    with open(jp) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get('type')
            if t == 'meta':
                meta = o
            elif t == 'move':
                moves.append(o)
            elif t == 'result':
                result = o
    if not meta or not result or result.get('result') not in SCORE:
        continue
    n_games += 1
    ws, bs = SCORE[result['result']]

    for colour, score in (('white', ws), ('black', bs)):
        label = meta.get(colour, colour)
        if SIDE not in ('both', colour, label):
            continue
        if score == 1.0:
            continue                                   # won -> by definition not a collapse
        # peak of THIS side's own eval, from its own moves only (eval is side-to-move POV; `eval` field is
        # already own-POV, positive = good for the mover)
        peak = None
        for i, m in enumerate(moves):
            if m.get('color') != colour or m.get('booked') or 'eval' not in m:
                continue
            ev = m['eval']
            if not isinstance(ev, int):
                continue                               # null eval (mate score / unset) -> not a peak candidate
            if peak is None or ev > peak[1]:
                # decision_fen = the position this move was CHOSEN FROM = fen of the PREVIOUS ply
                prev_fen = moves[i - 1]['fen'] if i > 0 else meta.get('start_fen')
                peak = (m.get('ply'), ev, prev_fen, m.get('uci'))
        if peak is None or peak[1] < WIN_THRESH:
            continue
        last = moves[-1] if moves else {}
        rows.append({
            'family': TAG, 'seed': os.path.basename(d), 'game': os.path.basename(d),
            'our_color': colour, 'result': result['result'],
            'peak_ply': peak[0], 'peak_eval': peak[1], 'peak_move': peak[3],
            'drop_ply': last.get('ply', ''), 'drop_eval': last.get('eval', ''),
            'swing': (peak[1] - last.get('eval', 0)) if isinstance(last.get('eval'), int) else '',
            'decision_fen': peak[2], 'drop_fen': last.get('fen', ''),
            'src_dir': d, 'label': label,
        })
        n_used += 1

out = os.path.join(THIS, 'ks_sets', 'collapse_dataset_%s.csv' % TAG)
cols = ['family', 'seed', 'game', 'our_color', 'result', 'peak_ply', 'peak_eval', 'peak_move',
        'drop_ply', 'drop_eval', 'swing', 'decision_fen', 'drop_fen', 'src_dir', 'label']
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

print("tag=%s  games_parsed=%d  collapses=%d  (win_thresh=%+.1f, side=%s)"
      % (TAG, n_games, n_used, WIN_THRESH / 1000.0, SIDE))
by_label = {}
for r in rows:
    by_label[r['label']] = by_label.get(r['label'], 0) + 1
for k, v in sorted(by_label.items()):
    print("   %-12s %4d" % (k, v))
print("wrote %s" % out)
print("⚠️ self-play corpus: the opponent shares our blind spots -> treat as hypothesis-generating, confirm on vs_sf")
