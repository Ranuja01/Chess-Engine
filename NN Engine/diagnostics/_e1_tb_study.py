"""
E1 step 0b (2026-11-10): exact truth for "the side our eval favours cannot create a passer" in K+P endings, from the 7-piece
tablebase, and whether a simple king-race condition separates the draws from the king-raid wins.

Why: step 0 (_e1_nopot_precision.py) had no venue. The stress set is passer races (3 fires in 600), and the game set has 58
pure pawn endings. There, 2 of the 3 fires were WINS for the leader by king activity
(8/1kp5/p5K1/8/8/1P6/8/8 b, 8/6k1/8/1p6/1P6/5K2/8/8 w). So structure alone is not drawishness; the owner's 10-04 case was
drawn because neither king could get in. Design: dev_notes/active/research-2026-10-08/POT-DESIGN-PREPLAN.md, E1 (and E2,
king activity).

    pyrun diagnostics/_e1_tb_study.py V2_PRESET=shipped [N_FIRE=500] [N_OTHER=200] [SEED=11] [DELAY=0.25] [MAXTRY=400000]

Positions: random legal K+P endings (2 kings, 2-5 pawns, each side at least one; `_draw_oracle.random_position`), side to
move random. leader = the side our STATIC v2 eval favours; fires = the leader's `passer_potential` mask is empty
(ChessAI.pawn_masks "potential"). Truth = Lichess tablebase WDL (cached in ks_sets/tablebase_labels_draw.json, shared with
_draw_oracle.py), turned to the leader's view.
King race (tempi, side to move counted): for each defender pawn not protected by a defender pawn, race = (leader king
distance to a square next to it) − (defender king distance to that pawn) − (1 if the leader is to move else 0); KR = the
minimum over those pawns (lower = the leader's king gets in first). No unprotected defender pawn ⇒ KR = +9 (no target).

REGISTERED before the run (11-10 19:30):
  - E1 is designable if some cut-off c gives "fires and KR >= c" a tablebase draw share >= 85% on >= 50 positions with the
    leader winning <= 10% of them. Then the damp fires only there. Otherwise E1 as a structural input parks (the king race
    would need search, not a static rule) and the record says why.
  - prediction: draw share among all fires 40-60%; among fires with KR >= 2, >= 80%; among fires with KR <= 0, wins >= 60%.
RESULT (seed 11, 11-10 20:10): fires 500: draw 42.6% / leader wins 46.0%; non-fires 200: draw 12.5% / wins 77.0%. By KR:
  <= −1 449 positions draw 39.4% wins 49.0% · 0: 29, 58.6% / 31.0% · +1: 16, 81.2% / 6.2% · +2: 6, 100% / 0%. NOT
  designable by the registered rule: only 22 fires have KR >= 1 (needs 50). Random placement rarely seals a king out.
REPLICATION, registered before its run (11-10 20:15): KR_MIN=1 on a FRESH seed (12) keeps only fires with KR >= 1 and
  queries the tablebase only for those, until N_FIRE=80. Same bar: designable if draw >= 85% and leader wins <= 10%.
  Prediction: draw 80-90%, wins 3-8% (the seed-11 pool of 22 read 86% / 4.5%).
Caveat: uniform random placement is not the game distribution (more scattered kings and pawns); it measures the rule's
logic, not its frequency in play. Good citizen: modest N, a delay per new query, everything cached.
"""
import os
import random
import sys

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
for k, v in ARGS.items():
	if k not in ("N_FIRE", "N_OTHER", "SEED", "MAXTRY", "KR_MIN"):
		os.environ[k] = v
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import chess   # noqa: E402

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import ChessAI   # noqa: E402
import _draw_oracle as orc   # noqa: E402  (tb_lookup, save_cache, random_position; its main is guarded)

N_FIRE = int(ARGS.get("N_FIRE", "500"))
N_OTHER = int(ARGS.get("N_OTHER", "200"))
MAXTRY = int(ARGS.get("MAXTRY", "400000"))
KR_MIN = int(ARGS["KR_MIN"]) if "KR_MIN" in ARGS else None   # replication mode: keep only fires with KR >= KR_MIN
rng = random.Random(int(ARGS.get("SEED", "11")))
CONFIGS = [(w, b) for w in range(1, 5) for b in range(1, 5) if 2 <= w + b <= 5]

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)


def cheb(a, b):
	return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3)))


def king_race(bd, leader):
	defender = not leader
	lk, dk = bd.king(leader), bd.king(defender)
	dpawns = bd.pieces(chess.PAWN, defender)
	best = 9
	for p in dpawns:
		if bd.attackers(defender, p) & dpawns:
			continue   # protected by a pawn: not a raid target
		adj = [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[p]) if not bd.piece_at(s)]
		if not adj:
			continue
		d_lead = min(cheb(lk, s) for s in adj)
		d_def = cheb(dk, p)
		race = d_lead - d_def - (1 if bd.turn == leader else 0)
		best = min(best, race)
	return best


rows = []   # (fires, kr, outcome for the leader: win / draw / loss)
n_fire = n_other = tries = 0
while (n_fire < N_FIRE or n_other < N_OTHER) and tries < MAXTRY:
	tries += 1
	w, b = rng.choice(CONFIGS)
	bd = orc.random_position([chess.PAWN] * w, [chess.PAWN] * b, rng)
	if bd is None:
		continue
	static_black = ai.ev_breakdown(bd).get("total", 0)
	if static_black == 0:
		continue
	leader = chess.BLACK if static_black > 0 else chess.WHITE
	masks = ChessAI.pawn_masks(int(bd.pawns), int(bd.occupied_co[chess.WHITE]), int(bd.occupied_co[chess.BLACK]))
	fires = masks["potential"][1 if leader == chess.BLACK else 0] == 0
	if fires and n_fire >= N_FIRE:
		continue
	if not fires and n_other >= N_OTHER:
		continue
	if KR_MIN is not None and (not fires or king_race(bd, leader) < KR_MIN):
		continue   # replication mode: query the tablebase only for the subset under test
	cat, _ = orc.tb_lookup(bd.fen())
	if cat not in ("win", "loss", "draw", "cursed-win", "blessed-loss", "maybe-win", "maybe-loss"):
		continue
	stm_wins = cat in ("win", "maybe-win")
	stm_loses = cat in ("loss", "maybe-loss")
	if cat in ("cursed-win", "blessed-loss"):
		stm_wins = stm_loses = False   # drawn by the fifty-move rule
	if stm_wins:
		out = "win" if bd.turn == leader else "loss"
	elif stm_loses:
		out = "loss" if bd.turn == leader else "win"
	else:
		out = "draw"
	rows.append((fires, king_race(bd, leader), out))
	if fires:
		n_fire += 1
	else:
		n_other += 1
	if (n_fire + n_other) % 100 == 0:
		orc.save_cache()
orc.save_cache()


def share(sel):
	n = len(sel)
	if n == 0:
		return "  0 positions"
	d = sum(o == "draw" for _, _, o in sel); wn = sum(o == "win" for _, _, o in sel); ls = n - d - wn
	return f"{n:4d} positions · draw {100 * d / n:5.1f}% · leader wins {100 * wn / n:5.1f}% · leader loses {100 * ls / n:5.1f}%"


fires = [r for r in rows if r[0]]
others = [r for r in rows if not r[0]]
print(f"tried {tries} random K+P placements; kept {len(fires)} fires and {len(others)} non-fires")
print(f"(new tablebase queries this run: {orc._net[0]})")
print("all fires     :", share(fires))
print("all non-fires :", share(others))
print("fires by king race KR (leader's king gets in first when KR is low):")
for lo, hi in ((-9, -1), (0, 0), (1, 1), (2, 2), (3, 8), (9, 9)):
	sel = [r for r in fires if lo <= r[1] <= hi]
	print(f"  KR {lo:+d}..{hi:+d}:", share(sel))
best = None
for c in range(-2, 10):
	sel = [r for r in fires if r[1] >= c]
	n = len(sel)
	if n < 50:
		continue
	d = sum(o == "draw" for _, _, o in sel) / n; wn = sum(o == "win" for _, _, o in sel) / n
	if d >= 0.85 and wn <= 0.10 and best is None:
		best = (c, n, d, wn)
print("REGISTERED: " + (f"designable — fires and KR >= {best[0]}: {best[1]} positions, draw {100 * best[2]:.1f}%, "
                        f"leader wins {100 * best[3]:.1f}%" if best else "NOT designable at any cut-off (E1 as a structural input parks)"))
