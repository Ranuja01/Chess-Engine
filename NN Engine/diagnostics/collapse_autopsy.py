# -*- coding: utf-8 -*-
"""Collapse autopsy — label each collapse blunder by MECHANISM via toggle-recovery.
Design: dev_notes/collapse-autopsy-design-2026-07-15.md.

For each collapse (from a vs_sf/Mediocre run's collapses.csv), localize the actual BLUNDER move
(max SF-eval swing across one of our moves in the peak->drop window, NOT the calm peak), get SF's
best move there, then re-search that position under a toggle matrix. The single change that makes
us play SF's move IS the culprit:
  our move == SF best              -> eval_calibration_or_ok (played the right move; loss is drift)
  a prune-off recovers SF best     -> prune:<which>  (RFP/razoring/futility = shallow; LMR = deep; LMP/null)
  only deeper recovers it          -> horizon (soft depth / EBF)
  nothing recovers it              -> eval_misrank (our eval prefers the losing move)

env is parsed once per engine process (search_engine.cpp initialize_engine, toggles_loaded), so each
toggle config runs as a fresh worker subprocess with a modified env.

Usage (dispatcher pyrun; STOCKFISH_PATH = SF18):
  pyrun diagnostics/collapse_autopsy.py run <collapses.csv> <game_dir> [max_collapses] [sf_movetime]
  pyrun diagnostics/collapse_autopsy.py worker <fenfile>     # internal (one engine pass under an env)
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
import csv
import json
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ENGINE_DIR)

PY = sys.executable
SF = os.environ.get('STOCKFISH_PATH')

# Common env for every engine pass; per-config overrides on top. 'deep16' tests the horizon.
COMMON = {'PRESET': 'LONG_FORMAT', 'USE_OPENING_BOOK': '0', 'OMP_NUM_THREADS': '1', 'MAX_DEPTH': '12'}
MATRIX = [
	('baseline', {}),
	('rfp_off', {'ENABLE_RFP': '0'}),
	('razoring_off', {'ENABLE_RAZORING': '0'}),
	('futility_off', {'ENABLE_FUTILITY': '0'}),
	('lmp_off', {'ENABLE_LMP': '0'}),
	('nullmove_off', {'ENABLE_NULLMOVE': '0'}),
	('lmr_off', {'ENABLE_LMR': '0'}),
	('deep16', {'MAX_DEPTH': '16'}),
]
PRUNE_LABELS = ['rfp_off', 'razoring_off', 'futility_off', 'lmp_off', 'nullmove_off', 'lmr_off']


def worker(fenfile):
	"""One engine pass (env fixed by the parent): run_one per FEN, emit fen<TAB>move<TAB>eval<TAB>depth."""
	from tactical_test import run_one
	for line in open(fenfile):
		fen = line.strip()
		if not fen:
			continue
		try:
			r = run_one(fen, set())
			print("%s\t%s\t%s\t%s" % (fen, r.get('uci'), r.get('eval'), r.get('depth')), flush=True)
		except Exception as e:
			print("%s\t%s\t%s\t%s" % (fen, None, None, "ERR:%s" % e), flush=True)


def run(collapses_csv, game_dir, max_collapses=20, sf_movetime=0.6):
	import chess
	import chess.engine
	rows = list(csv.DictReader(open(collapses_csv)))[:max_collapses]
	sf = chess.engine.SimpleEngine.popen_uci(SF)

	def sf_eval_wpov(fen):
		info = sf.analyse(chess.Board(fen), chess.engine.Limit(time=sf_movetime))
		return info['score'].white().score(mate_score=100000)

	def sf_best(fen):
		return sf.play(chess.Board(fen), chess.engine.Limit(time=max(1.0, sf_movetime))).move.uci()

	# 1) localize the blunder move (max our-POV SF swing) per collapse
	blunders = []
	for row in rows:
		g = row['game']
		color = row['our_color']
		jf = os.path.join(game_dir, "game_%03d.jsonl" % int(g))
		if not os.path.exists(jf):
			continue
		recs = [json.loads(l) for l in open(jf)]
		fen_by_ply = {r['ply']: r['fen'] for r in recs if 'fen' in r}
		move_by_ply = {r['ply']: r.get('uci') for r in recs}
		color_by_ply = {r['ply']: r.get('color') for r in recs}
		pp = int(float(row['peak_ply']))
		dp = int(float(row['drop_ply']))
		wsign = 1 if color == 'white' else -1
		best = None
		for ply in range(pp, dp + 1):
			if color_by_ply.get(ply) != color:
				continue
			fb = fen_by_ply.get(ply - 1)
			fa = fen_by_ply.get(ply)
			if not fb or not fa:
				continue
			eb = wsign * sf_eval_wpov(fb)
			ea = wsign * sf_eval_wpov(fa)
			swing = eb - ea             # positive = our move lost eval
			if best is None or swing > best[0]:
				best = (swing, ply, move_by_ply.get(ply), fb, eb, ea)
		if best is None:
			continue
		swing, ply, mv, fb, eb, ea = best
		blunders.append({'game': g, 'ply': ply, 'our_move': mv, 'fen': fb,
		                 'sf_before': eb, 'sf_after': ea, 'swing': swing, 'sf_best': sf_best(fb)})
	sf.quit()

	if not blunders:
		print("no blunders localized (empty collapses or missing jsonls)")
		return
	matrix_categorize(blunders, os.path.dirname(os.path.abspath(collapses_csv)), 'COLLAPSE AUTOPSY')


def evaldrop(game_dir, max_games=40, drop_thresh=500, sf_movetime=0.6, window=8):
	"""Broadened autopsy over ALL games. Two-stage localization: (1) OUR eval-drop finds the REGION cheaply
	(no full SF walk), (2) SF-swing WITHIN a lookback window [P-window, P] finds the TRUE blunder (our
	optimism lags, so the real error precedes where our eval finally falls). Keeps games whose our-drop
	exceeds drop_thresh. A window with no real SF-swing = genuine drift (no single blunder) -> tagged drift."""
	import glob
	import chess
	import chess.engine
	files = sorted(glob.glob(os.path.join(game_dir, 'game_*.jsonl')))[:max_games]
	# stage 1: our-eval-drop region per game
	regions = []
	for jf in files:
		recs = [json.loads(l) for l in open(jf)]
		fen_by_ply = {r['ply']: r['fen'] for r in recs if 'fen' in r}
		ours = [r for r in recs if not r.get('sf') and r.get('our_pov_eval') is not None
		        and abs(r['our_pov_eval']) < 900000]
		if len(ours) < 3:
			continue
		color = ours[0].get('color')
		best = None
		for a, b in zip(ours, ours[1:]):
			drop = a['our_pov_eval'] - b['our_pov_eval']
			if best is None or drop > best[0]:
				best = (drop, b['ply'])
		if best is None or best[0] < drop_thresh:
			continue
		regions.append({'game': os.path.basename(jf), 'color': color, 'drop': best[0], 'P': best[1],
		                'our_moves': [(r['ply'], r.get('uci')) for r in ours], 'fen_by_ply': fen_by_ply})
	if not regions:
		print("no eval-drop regions above threshold %d" % drop_thresh)
		return
	# stage 2: SF-swing within the lookback window -> true blunder
	sf = chess.engine.SimpleEngine.popen_uci(SF)

	def sf_wpov(fen):
		return sf.analyse(chess.Board(fen), chess.engine.Limit(time=sf_movetime))['score'].white().score(mate_score=100000)

	blunders = []
	for rg in regions:
		wsign = 1 if rg['color'] == 'white' else -1
		P = rg['P']
		fbp = rg['fen_by_ply']
		best = None
		for ply, mv in rg['our_moves']:
			if ply < P - window or ply > P:
				continue
			fb = fbp.get(ply - 1)
			fa = fbp.get(ply)
			if not fb or not fa:
				continue
			swing = wsign * sf_wpov(fb) - wsign * sf_wpov(fa)   # SF-confirmed our-POV loss across our move
			if best is None or swing > best[0]:
				best = (swing, ply, mv, fb)
		if best is None or not best[3]:
			continue
		swing, ply, mv, fb = best
		b = {'game': rg['game'], 'ply': ply, 'our_move': mv, 'fen': fb, 'swing': swing,
		     'drift': swing < 120}          # SF centipawns: <120cp single-move swing => drift/accumulation
		b['sf_best'] = sf.play(chess.Board(fb), chess.engine.Limit(time=max(1.0, sf_movetime))).move.uci()
		blunders.append(b)
	sf.quit()
	if not blunders:
		print("no blunders localized")
		return
	matrix_categorize(blunders, game_dir, 'EVAL-DROP+SF AUTOPSY (drop>=%d win=%d)' % (drop_thresh, window))


def matrix_categorize(blunders, out_dir, title):
	"""Shared: run the toggle matrix over blunder FENs (fresh process per config) + categorize + histogram.
	Each blunder needs 'fen' and 'sf_best'."""
	fenfile = os.path.join(out_dir, 'autopsy_fens.txt')
	open(fenfile, 'w').write("\n".join(b['fen'] for b in blunders) + "\n")
	results = {}
	for label, ov in MATRIX:
		env = os.environ.copy()
		env.update(COMMON)
		env.update(ov)
		p = subprocess.run([PY, os.path.abspath(__file__), 'worker', fenfile],
		                   env=env, capture_output=True, text=True)
		d = {}
		for line in p.stdout.splitlines():
			parts = line.split('\t')
			if len(parts) >= 2:
				d[parts[0]] = parts[1]
		results[label] = d
		print("[matrix] %-14s positions=%d" % (label, len(d)), flush=True)
	hist = {}
	print("\n=== %s (n=%d) ===" % (title, len(blunders)))
	for b in blunders:
		fen = b['fen']
		sfbest = b['sf_best']
		base_mv = results['baseline'].get(fen)
		if b.get('drift'):
			cat = 'drift_no_single_blunder'
		elif base_mv == sfbest:
			cat = 'eval_calibration_matched_SF'
		else:
			cat = None
			for label in PRUNE_LABELS:
				if results[label].get(fen) == sfbest:
					cat = 'prune:' + label
					break
			if cat is None:
				cat = 'horizon' if results['deep16'].get(fen) == sfbest else 'eval_misrank'
		b['category'] = cat
		hist[cat] = hist.get(cat, 0) + 1
		print("%s ply %s swing %+d | our %s vs SF %s | %s"
		      % (b['game'], b['ply'], b['swing'], base_mv, sfbest, cat))
	print("\n=== HISTOGRAM ===")
	for k, v in sorted(hist.items(), key=lambda x: -x[1]):
		print("%3d  %s" % (v, k))
	# Labeled corpus for the plug-the-gaps loop (probe candidate settings against it, no game noise).
	corpus = os.path.join(out_dir, 'autopsy_corpus.csv')
	with open(corpus, 'w', newline='') as f:
		w = csv.writer(f)
		w.writerow(['game', 'ply', 'fen', 'base_move', 'sf_best', 'category'])
		for b in blunders:
			w.writerow([b.get('game'), b.get('ply'), b['fen'], results['baseline'].get(b['fen']),
			            b['sf_best'], b['category']])
	print("corpus -> %s" % corpus)


def probe(corpus_csv, knobs):
	"""Test a candidate setting against the labeled gap corpus (autopsy_corpus.csv): run the candidate
	config on each gap FEN and report, per category, how many now MATCH SF's best move (i.e. the setting
	'plugs' that gap). Deterministic — no game-sample noise. knobs = ['ENABLE_X=1', 'SCALE_Y=130', ...]."""
	rows = list(csv.DictReader(open(corpus_csv)))
	fenfile = os.path.join(os.path.dirname(os.path.abspath(corpus_csv)), 'probe_fens.txt')
	open(fenfile, 'w').write("\n".join(r['fen'] for r in rows) + "\n")
	env = os.environ.copy()
	env.update(COMMON)
	for kv in knobs:
		if '=' in kv:
			k, v = kv.split('=', 1)
			env[k] = v
	p = subprocess.run([PY, os.path.abspath(__file__), 'worker', fenfile], env=env, capture_output=True, text=True)
	cand = {}
	for line in p.stdout.splitlines():
		parts = line.split('\t')
		if len(parts) >= 2:
			cand[parts[0]] = parts[1]
	from collections import defaultdict
	tot = defaultdict(int)
	recov = defaultdict(int)      # candidate now plays SF-best
	changed = defaultdict(int)    # candidate differs from baseline (any change)
	for r in rows:
		c = r['category']
		tot[c] += 1
		cm = cand.get(r['fen'])
		if cm == r['sf_best']:
			recov[c] += 1
		if cm != r['base_move']:
			changed[c] += 1
	print("=== PROBE [%s] ===" % ' '.join(knobs))
	print("category                     n   ->SFbest   changed")
	for c in sorted(tot, key=lambda x: -tot[x]):
		print("%-26s %4d   %4d      %4d" % (c, tot[c], recov[c], changed[c]))


if __name__ == '__main__':
	mode = sys.argv[1] if len(sys.argv) > 1 else ''
	if mode == 'worker':
		worker(sys.argv[2])
	elif mode == 'run':
		run(sys.argv[2], sys.argv[3],
		    int(sys.argv[4]) if len(sys.argv) > 4 else 20,
		    float(sys.argv[5]) if len(sys.argv) > 5 else 0.6)
	elif mode == 'evaldrop':
		evaldrop(sys.argv[2],
		         int(sys.argv[3]) if len(sys.argv) > 3 else 40,
		         int(sys.argv[4]) if len(sys.argv) > 4 else 500,
		         float(sys.argv[5]) if len(sys.argv) > 5 else 0.6)
	elif mode == 'probe':
		probe(sys.argv[2], sys.argv[3:])
	else:
		print(__doc__)
