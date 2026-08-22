# -*- coding: utf-8 -*-
"""Mine the intacc-vs-base SPRT games into HELPED vs HURT decision positions, for the SF counter-analysis.
- HURT  = intacc (the KS integration) reached a winning eval on its OWN move then did NOT win  -> a regression:
          a position the new stuff mis-handled.
- HELPED = base reached a winning eval on its OWN move then did NOT win (intacc drew/won from lost) -> the
          integration held a position baseline would have blown.
Decision FEN = the position at the collapsing side's PEAK own-eval (where it was most winning, before the throw).
Pure jsonl parse (no engine) -> fast, ~1 core. Writes two fen_start CSVs for downstream SF triangulation.

Run: bash <runner> pyrun diagnostics/_sprt_collapse_mine.py [games_dir]
"""
import os, sys, json, glob
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
GDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ENGINE_DIR, 'selfplay/games/sprt_intacc')
WIN = 2000   # millipawn peak threshold (same +2.0 as the vs_sf miner)

def outcome_for(result, color):
    if result == '1-0':   return 'win' if color == 'white' else 'loss'
    if result == '0-1':   return 'win' if color == 'black' else 'loss'
    return 'draw'

hurt, helped = [], []
n_games = 0
for jl in glob.glob(os.path.join(GDIR, 'game_*/game.jsonl')):
    try:
        lines = [json.loads(x) for x in open(jl) if x.strip()]
    except Exception:
        continue
    meta = next((l for l in lines if l.get('type') == 'meta'), None)
    res  = next((l for l in lines if l.get('type') == 'result'), None)
    if not meta or not res:
        continue
    n_games += 1
    intacc_color = 'white' if 'KS_FLANK_MODE' in (meta.get('config_white') or '') else 'black'
    base_color = 'black' if intacc_color == 'white' else 'white'
    result = res.get('result')
    intacc_out = outcome_for(result, intacc_color)
    base_out = outcome_for(result, base_color)

    # own-move eval traces (White-POV mp in eval_white_pov); convert to each engine's own POV
    def trace(label, color):
        out = []
        for l in lines:
            if l.get('type') == 'move' and l.get('label') == label and 'eval_white_pov' in l:
                e = l['eval_white_pov'] if color == 'white' else -l['eval_white_pov']
                out.append((l.get('ply'), l.get('fen'), e))
        return out

    it = trace('intacc', intacc_color)
    bt = trace('base', base_color)
    if it:
        ply, fen, pk = max(it, key=lambda t: t[2])
        if pk >= WIN and intacc_out != 'win':
            hurt.append((fen, pk, intacc_out, os.path.basename(os.path.dirname(jl))))
    if bt:
        ply, fen, pk = max(bt, key=lambda t: t[2])
        if pk >= WIN and base_out != 'win':
            helped.append((fen, pk, base_out, os.path.basename(os.path.dirname(jl))))

def dump(rows, path):
    with open(os.path.join(ENGINE_DIR, path), 'w') as f:
        f.write("fen_start,peak,outcome,game\n")
        for fen, pk, out, g in rows:
            f.write("%s,%d,%s,%s\n" % (fen, pk, out, g))

dump(hurt, 'diagnostics/_sprt_hurt.csv')
dump(helped, 'diagnostics/_sprt_helped.csv')
print("games parsed: %d" % n_games)
print("HURT   (intacc blew a win): %d  -> diagnostics/_sprt_hurt.csv" % len(hurt))
print("HELPED (base blew a win, intacc held): %d  -> diagnostics/_sprt_helped.csv" % len(helped))
print("net (hurt - helped) = %+d   [>0 => integration regressed net, consistent with the -20 Elo]" % (len(hurt) - len(helped)))
