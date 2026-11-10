"""
E1 step 0 (2026-11-10): how precise is "the side our eval favours cannot create a passer" as a drawishness signal in
pawn endings, BEFORE any eval change? Transition close-out item; design in
dev_notes/active/research-2026-10-08/POT-DESIGN-PREPLAN.md (E1) and the owner's 10-04 UI loss (a dead-drawn K+P ending
read +0.5..+0.9 because the extra pawn was a doubled g-pawn).

    pyrun diagnostics/_e1_nopot_precision.py V2_PRESET=shipped [CSV=diagnostics/ks_sets/kp_stress_sf18.csv] [DRAW=60] [WIN=200]

Per pawn-ending row (no pieces but kings and pawns):
  - leader = the side our STATIC eval favours (v2 shipped; the winnability factor scales the edge of that side);
  - fires  = the leader's passer_potential mask is empty (engine's own detector, ChessAI.pawn_masks "potential":
             no front-most passer and no classic candidate);
  - truth  = SF18's score in the CSV (White-relative), turned to the leader's view: draw if |cp| <= DRAW,
             leader wins if >= WIN, defender wins if <= -WIN, otherwise unclear.
Reads: fire rate; truth among fires vs non-fires; our static edge where it fires and SF calls it a draw (the false edge
E1 would damp); and the won rows among fires (the king-raid risk: a damp there would throw a win away).

REGISTERED before the run (11-10 18:20):
  - go to the build if, among fires, the SF draw share is at least 2x the non-fire draw share AND leader-wins among
    fires is at most 25% (beyond that, a damp costs too many real wins);
  - prediction: fire rate 10-25% of rows with a leader; draw share among fires 50-70% vs about 30% elsewhere;
    leader wins among fires 15-25% (king raids).
Caveat: SF's cp is material-scaled in general (memory sf-cp-labels-are-material-scaled-use-wdl); in pawn endings the
material spread is small, so a cp threshold is a usable first cut, not a final label. Tablebase rows come later
(_draw_oracle.py) for the <= 7-piece cases.
"""
import csv
import os
import sys

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
for k, v in ARGS.items():
	if k not in ("CSV", "DRAW", "WIN"):
		os.environ[k] = v   # engine knobs before the engine is built
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import chess   # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))
import ChessAI   # noqa: E402

DRAW = int(ARGS.get("DRAW", "60"))
WIN = int(ARGS.get("WIN", "200"))
path = ARGS.get("CSV", "diagnostics/ks_sets/kp_stress_sf18.csv")

seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)

stats = {True: {"draw": 0, "lead": 0, "def": 0, "unclear": 0}, False: {"draw": 0, "lead": 0, "def": 0, "unclear": 0}}
false_edge = []   # our static edge (pawns) on fired rows SF calls drawn
won_fired = []    # (fen, our edge, SF leader cp) on fired rows the leader wins
skipped = 0
with open(path, newline="") as f:
	for row in csv.DictReader(f):
		b = chess.Board(row["fen"])
		if b.occupied & ~(b.pawns | b.kings):
			skipped += 1   # not a pure pawn ending
			continue
		ev = ai.ev_breakdown(b)
		static_black = ev.get("total", 0)   # Black-positive, millipawns
		if static_black == 0:
			skipped += 1
			continue
		leader_black = static_black > 0
		masks = ChessAI.pawn_masks(int(b.pawns), int(b.occupied_co[chess.WHITE]), int(b.occupied_co[chess.BLACK]))
		pot_w, pot_b = masks["potential"]
		fires = (pot_b if leader_black else pot_w) == 0
		# ☠️ best_cp in kp_stress_sf18.csv is WHITE-relative, not side-to-move relative: checked 11-10 on its row 4 (Black
		# to move, stored −312; SF18 d22 gives +452 for Black). Reading it as stm-relative flipped every Black-to-move label.
		cp_white = int(row["best_cp"])
		cp_lead = -cp_white if leader_black else cp_white
		if abs(cp_lead) <= DRAW:
			t = "draw"
		elif cp_lead >= WIN:
			t = "lead"
		elif cp_lead <= -WIN:
			t = "def"
		else:
			t = "unclear"
		stats[fires][t] += 1
		edge = abs(static_black) / 1000.0
		if fires and t == "draw":
			false_edge.append(edge)
		if fires and t == "lead":
			won_fired.append((row["fen"], edge, cp_lead))


def line(name, s):
	n = sum(s.values())
	if n == 0:
		return f"  {name}: no rows"
	return (f"  {name}: {n} rows · SF draw {s['draw']} ({100 * s['draw'] / n:.1f}%) · leader wins {s['lead']} "
	        f"({100 * s['lead'] / n:.1f}%) · defender wins {s['def']} ({100 * s['def'] / n:.1f}%) · unclear {s['unclear']}")


n_f, n_n = sum(stats[True].values()), sum(stats[False].values())
print(f"pawn-ending rows with a static leader: {n_f + n_n} (skipped {skipped}); DRAW <= {DRAW} cp, WIN >= {WIN} cp")
print(f"fires (leader has no passer potential): {n_f} = {100 * n_f / max(1, n_f + n_n):.1f}%")
print(line("fires    ", stats[True]))
print(line("non-fires", stats[False]))
if false_edge:
	false_edge.sort()
	print(f"our static edge where it fires and SF says draw: mean {sum(false_edge) / len(false_edge):.2f} pawns, "
	      f"median {false_edge[len(false_edge) // 2]:.2f}, >= 0.5 pawns on {sum(e >= 0.5 for e in false_edge)} rows")
dr_f = stats[True]["draw"] / max(1, n_f)
dr_n = stats[False]["draw"] / max(1, n_n)
lw_f = stats[True]["lead"] / max(1, n_f)
ok = n_f >= 50 and dr_f >= 2 * dr_n and lw_f <= 0.25
print(f"REGISTERED: draw share fires {100 * dr_f:.1f}% vs non-fires {100 * dr_n:.1f}% (need >= 2x); leader wins among "
      f"fires {100 * lw_f:.1f}% (need <= 25%); fires {n_f} (need >= 50) ⇒ {'GO to the build' if ok else 'NO GO'}")
print("won rows among fires (the king-raid risk), first 8:")
for fen, edge, cp in won_fired[:8]:
	print(f"  {fen}  ours {edge:+.2f}  SF leader {cp:+d}")
