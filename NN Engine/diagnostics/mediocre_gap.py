# -*- coding: utf-8 -*-
"""Stage-0 CONCENTRATED-vs-DIFFUSE diagnostic. For each autopsy gap position, get MEDIOCRE's move (the
reachable ~2319 HCE exemplar) and compare to OUR move (base_move) and SF's (sf_best):
  LEARNABLE            : mediocre == sf, we differ  -> an HCE concept a 2319 hobby engine has and we LACK
  shared_hce_ceiling   : mediocre == us, both != sf -> we + Mediocre both miss what SF sees (not our us->Med gap)
  all_differ           : all three differ           -> ambiguous / hard
  we_match_sf          : we already play SF's move   -> not a gap here (localization artifact)
Read: LEARNABLE high + clustered into a few themes => the us->Mediocre gap is CONCENTRATED (hand-fixable);
LEARNABLE low / scattered => DIFFUSE (needs full-fidelity data-fit / NNUE). LEARNABLE FENs dumped for naming.

Usage: pyrun diagnostics/mediocre_gap.py <autopsy_corpus.csv> [movetime=1.0]
"""
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "selfplay"))
import chess
import chess.engine
from raw_uci import RawUciEngine

WRAP = "/home/ranuja/mediocre_uci.sh"


def main():
	corpus = sys.argv[1]
	mt = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
	rows = list(csv.DictReader(open(corpus)))
	eng = RawUciEngine(WRAP)
	hist = {}
	learnable = []
	for r in rows:
		fen = r["fen"]
		ours = r.get("base_move")
		sfb = r.get("sf_best")
		try:
			res = eng.play(chess.Board(fen), chess.engine.Limit(time=mt))
			med = res.move.uci() if res.move else None
		except Exception:
			med = None
		if med is None:
			cat = "mediocre_no_move"
		elif ours == sfb:
			cat = "we_match_sf"
		elif med == sfb and ours != sfb:
			cat = "LEARNABLE"
			learnable.append((r.get("game"), r.get("ply"), fen, ours, sfb, r.get("category")))
		elif med == ours and med != sfb:
			cat = "shared_hce_ceiling"
		else:
			cat = "all_differ"
		hist[cat] = hist.get(cat, 0) + 1
	eng.quit()

	print("=== MEDIOCRE GAP (n=%d, movetime=%.1fs) ===" % (len(rows), mt))
	for k, v in sorted(hist.items(), key=lambda x: -x[1]):
		print("%4d  %s" % (v, k))
	real_gap = sum(v for k, v in hist.items() if k not in ("we_match_sf", "mediocre_no_move"))
	learn = hist.get("LEARNABLE", 0)
	print("\nLEARNABLE / real-gap = %d/%d (%.0f%%)   high+clustered=CONCENTRATED, low/scattered=DIFFUSE"
	      % (learn, real_gap, 100.0 * learn / max(1, real_gap)))
	print("\n-- LEARNABLE positions (mediocre==SF, we differ) — for concept-naming --")
	for g, p, fen, ours, sfb, ac in learnable[:50]:
		print("%s ply%s  our=%s  sf=med=%s  [%s]  %s" % (g, p, ours, sfb, ac, fen))


if __name__ == "__main__":
	main()
