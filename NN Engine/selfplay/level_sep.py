# -*- coding: utf-8 -*-
"""Separability diagnostic for the level-material collapse plurality.

Question: at midgame, LEVEL-material positions, does any eval TERM over-fire in the positions we
collapsed from (the flip blindness points) vs healthy level positions (SF says level)? A term that is
systematically larger in the lost set is a candidate DETECTOR/term for S4 placement-conditioning; if
nothing separates, the level scatter is genuine imprecision (precision/depth-bound, not knob-fixable).

Reads the stored (default) eval_breakdown — run a default `annotate <tag> --reeval` first. jsonl-only.
Run:  overnight_runner.sh pyrun selfplay/level_sep.py [tag=placement_bundle]
"""
import csv, json, os, glob, sys, random

THIS = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else 'placement_bundle'
G = os.path.join(THIS, 'games', TAG)
TERMS = ['pieces', 'capture_gains', 'passed_pawn_support', 'latent_threat', 'central',
         'imbalance_white', 'imbalance_black', 'pair_bonus', 'piece_value_boost',
         'pt_pawns', 'pt_knights', 'pt_bishops', 'pt_rooks', 'pt_queens', 'pt_kings']


def pcount(fen):
    return sum(1 for ch in fen.split()[0] if ch.isalpha())


# flip blindness points (midgame, by (game, ply_start)) from the candidate arm's losses
flip_keys = set()
fp = os.path.join(G, 'flips_fast.csv')
if os.path.exists(fp):
    for r in csv.DictReader(open(fp)):
        try:
            if int(r['pieces']) > 12:
                flip_keys.add((r['game'], int(float(r['ply_start']))))
        except Exception:
            pass

lost, healthy = [], []
for d in sorted(glob.glob(os.path.join(G, 'game_*'))):
    jf = os.path.join(d, 'game.annotated.jsonl')
    if not os.path.exists(jf):
        continue
    gname = os.path.basename(d)
    for ln in open(jf):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get('type') != 'move':
            continue
        eb = o.get('eval_breakdown'); fen = o.get('fen'); ply = o.get('ply'); sf = o.get('sf_cp')
        if not eb or not fen or pcount(fen) <= 12:
            continue
        if abs(eb.get('material', 0)) > 1000:          # level material only (<= 1 pawn)
            continue
        if (gname, ply) in flip_keys:
            lost.append(eb)
        elif sf is not None and abs(sf) < 50:           # SF says genuinely level
            healthy.append(eb)

random.seed(0)
if lost and len(healthy) > 5 * len(lost):
    healthy = random.sample(healthy, 5 * len(lost))
print(f'lost-level n={len(lost)}   healthy-level n={len(healthy)}')
if not lost or not healthy:
    print('insufficient data'); raise SystemExit


def mabs(group, t):
    return sum(abs(g.get(t, 0) or 0) for g in group) / len(group)


def msign(group, t):
    return sum((g.get(t, 0) or 0) for g in group) / len(group)


rows = []
for t in TERMS:
    lv, hv = mabs(lost, t), mabs(healthy, t)
    rows.append((abs(lv - hv), t, lv, hv, msign(lost, t), msign(healthy, t)))
print(f'{"term":22}{"lost|.|":>9}{"healthy|.|":>11}{"|.|diff":>9}{"lostMean":>10}{"healMean":>10}')
for _, t, lv, hv, ls, hs in sorted(rows, reverse=True):
    print(f'{t:22}{lv:9.0f}{hv:11.0f}{lv-hv:+9.0f}{ls:+10.0f}{hs:+10.0f}')
print('\n(term with large +|.|diff = over-fires in the lost positions = candidate S4 detector/term)')
