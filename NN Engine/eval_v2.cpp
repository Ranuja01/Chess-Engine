/* eval_v2.cpp

@author: Ranuja Pinnaduwage

The ground-up second static evaluation, built one feature at a time. Selected at runtime by
Config::EVAL_ARM; the shipped eval (placement_and_piece_eval_v1 in cpp_bitboard.cpp) is untouched and
remains the byte-identical control arm inside the same binary.

WHY THIS FILE EXISTS. Term-at-a-time screening was closed by arithmetic -- a whole eval is worth ~7pp of
win%-regret and our cross-set resolution is ~2-2.5pp, so a single term would have to carry a third of a
whole eval to be visible, and none does. The one route left on record was "bundle disjoint terms", and on
2026-09-10 that was refuted by measurement: three individually sign-consistent components combined to
+0.47pp -- BELOW every one of them alone -- while cancelling ~26% of each other's move changes, replicated
on all three corpora. The shipped eval carries ~30 names over ~2 signals and its components overlap 50-64%,
so nothing can be improved additively inside it.
★ A rebuild is different in kind from an edit because a term's value is `constant x mechanism`, and every
constant in v1 was fitted while its neighbours were wrong. A term there can be load-bearing BY ACCIDENT:
removing a bad one measures worse than keeping it. Nothing here inherits a constant fitted around it.

THE MEASUREMENT DIRECTION IS THE POINT. Ablating a feature out of a full eval measures it in the regime
where it is MAXIMALLY redundant -- 29 other terms stand ready to cover for it. Adding a feature to a
minimal core measures it where it is load-bearing. That is why this is a build-up ladder and not a cleanup.

THE DESIGN CONTRACT -- every rung must hold all of it:
  1. OUTPUT: absolute Black-positive milli-pawns (pawn = 1000) for a NON-TERMINAL position. White-favouring
     terms SUBTRACT, Black-favouring terms ADD. The side-to-move flip happens once, outside, at
     search_engine.cpp:8910 -- never here.
  2. PURE: a function of the 12 arguments plus data frozen at init (constexpr tables, the BB_* masks built
     by initialize_attack_tables, the placement layers rebuilt by rebuild_scaled_placement, Config::*).
     ☠️ v2 WRITES NO GLOBAL. cpp_bitboard.h exposes every one of v1's -- pawns, knights, occupied*,
     attackingLayer, attack_bitmasks, pieceTypeLookUp, white/blackOffensiveScore, white/blackDefensiveScore,
     white/blackPieceVal, central_score, g_castling_rights, g_rook_file_bonus, g_ks_units_*, the attack-layer
     caches -- and the compiler will not stop us touching them. The enforcement is empirical: EVAL_ARM=2
     runs both evals in the same node and returns v1's value, so any write v1 reads moves v1's own result
     and the bench falls off 250 / 35,310,778.
  3. SYMMETRIC BY CONSTRUCTION: one parameterised body per concept with the side as a parameter, never two
     mirrored copies. v1's 11 historical colour defects all came from hand-mirrored twins.
  4. DETECTOR AND TRANSFORMATION SEPARATE: every feature publishes its detector output into EvalBreakdown so
     "does it fire on the right positions" (a cheap classification question, no search, no null band) can be
     audited apart from "does firing produce the right ordering" (the regret question).
  5. HONEST BREAKDOWN: publish only terms actually computed, and set the matching EB_* bit. Unpublished
     keys are OMITTED by ChessAI.ev_breakdown, so a consumer raises KeyError rather than silently reading 0
     and reporting "no gap". ~172 files read that struct.

WHAT IS DELIBERATELY ABSENT, and why (the inverse of an optimization log is the valuable half of it):
  - THE HEAT MAP / attackingLayer. It is a 5-in-1 -- king attack, king defence, central control,
    mobility/space, xrays -- and the giants carry none, because they have a dedicated subsystem for each
    job. Our own measurement supports decomposing it rather than keeping it: its three consumer channels
    read as NOT collinear (max r=0.42), which is what you expect when one mechanism is doing three different
    jobs. It is retired decomposed, not discredited, and it returns as a testable ADDITION at a later rung:
    if it still pays once dedicated KS/central/mobility/space exist, it was carrying something they miss.
    ⚠️ Targeted attacker and xray queries are still needed -- what is retired is the general per-piece sweep.
  - CAPTURE GAINS. No giant computes a static SEE pre-booking of pending exchanges, and it is expensive.
    Standing puzzle it leaves behind: with capgains working we should rarely be sitting in the non-quiet
    positions where it is load-bearing, which points at either qsearch leaving tension unresolved or capgains
    compensating for something else. Building v2 without it converts that question into a result.
  - OvD (offensive-vs-defensive pressure). It drifted into being a second king-safety term and fought the
    first. ★ The concept it was reaching for is real and distinct -- LONG-TERM, prophylactic pressure (an
    oncoming pawn storm making castling to that side bad) as opposed to KS's IMMEDIATE pressure from
    attacking pieces -- so it returns as a late rung designed to be harmonious with KS, not as a port.

THE LADDER. Config::EVAL_V2_RUNG selects how far up to evaluate; each rung is read against the PREVIOUS
rung, which is a candidate-vs-candidate comparison and therefore null-independent -- the one comparison our
instruments resolve well (the SF11/SF15c gap read 0.08 on both corpora, first try).
  rung 0: material + piece-square tables.

*/

#include "eval_v2.h"
#include "cpp_bitboard.h"
#include "move_gen.h"
#include "search_engine.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

// ===================================================================================================
// SHARED PER-EVAL CONTEXT
// ===================================================================================================

/*
	Everything derived from the board that more than one term wants, computed EXACTLY ONCE and passed by
	reference thereafter.

	WHY this exists before there is anything to share: v1 computes its material census SIX different ways --
	side-effect accumulation inside the 24 per-piece evaluator sites, a full bitboard recompute duplicated
	verbatim at four places, plus private recounts in advanced_endgame_eval, endgame_convertibility_scale,
	the Kaufman census and the NPEDGE block -- and one of those paths double-counts pawns and rooks whenever
	phase_score > 40 (cpp_bitboard.cpp:7709-7713). That did not arrive as a bad decision; it accumulated
	because there was never one place to read the census from. This is that place, from the first rung.

	Counts are int8_t because they index and compare, never accumulate -- narrow the STORAGE, compute in int
	(an int8_t arithmetic chain costs sign-extension and blocks vectorisation, so it is a false economy on
	anything hot).
*/
struct V2Context {
	// Raw board, as handed in.
	uint64_t pawns, knights, bishops, rooks, queens, kings;
	uint64_t white, black, occupied;
	uint64_t castling_rights;
	bool     turn;
	int      moveNum;

	// Material census -- THE single source, in milli-pawns, as magnitudes (not yet Black-positive).
	int   mat_white, mat_black;
	int   npm_white, npm_black;          // non-pawn material, for phase and endgame gating
	int8_t cnt_white[6], cnt_black[6];   // per type, indexed 0=pawn .. 5=king (matches whitePlacementLayer)

	// Game phase, 256 = full midgame .. 0 = deep endgame. ⚠️ v2 owns this and it is the OPPOSITE
	// orientation to v1's phase_score (0 = opening, 128 = bare kings, consumed as a 3-way boolean).
	// Deliberate: v2 does not inherit a convention it does not use, and mixing the two silently is a
	// sign-error waiting to happen. Continuous, so no phase cliff exists to tune around.
	int   phase256;
};

/*
	v2's own piece value, phase-tapered, in millipawns. ☠️ Separate from Config `values[]` BY DESIGN --
	that table feeds see(), move ordering and the null-move material threshold, so retuning it would be a
	search change. See the EVAL_V2_PAWN_MG comment in search_engine.h for why only the pawn tapers.

	At the default EVAL_V2_PAWN_MG = 1000 this returns values[] unchanged for every piece in every phase,
	so rung 0.5 defaults to rung 0 exactly.
*/
inline int v2_piece_value(int t, int phase256)
{
	if (t != 0){
		// Non-pawn pieces. ★ The PIECE-SIDE taper: relative to PAWNS, pieces are DEARER in the midgame
		// (SF's knight is 781/128 = 6.10 pawns mg, 854/213 = 4.01 eg). Applying the shift HERE instead of
		// to the pawn keeps the pawn at 1000 as the UNIT OF ACCOUNT, so every positional constant keeps
		// its meaning. ☠️ Tapering the pawn instead inflates them all in the midgame: STS measured
		// 1698 flat -> 1614 at PAWN_MG 550 -> 1522 at 700, monotone.
		const int peg = values[t + 1];
		if (Config::EVAL_V2_PIECE_MG_PCT == 100) return peg;   // identity short-circuit
		const int pmg = peg * Config::EVAL_V2_PIECE_MG_PCT / 100;
		return (pmg * phase256 + peg * (256 - phase256)) >> 8;
	}
	const int mg = Config::EVAL_V2_PAWN_MG;
	const int eg = values[1];
	if (mg == eg) return eg;                           // identity short-circuit: the default path is free
	return (mg * phase256 + eg * (256 - phase256)) >> 8;
}

/*
	Build the context. One pass per piece type per side; popcount rather than a bit loop because we want the
	COUNT here and the per-square work happens later in the terms that actually need squares.
*/
inline void build_context(V2Context &c, int moveNum, bool turn,
                          uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
                          uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,
                          uint64_t whiteMask, uint64_t blackMask, uint64_t occupiedMask,
                          uint64_t castlingRights)
{
	c.pawns = pawnsMask;   c.knights = knightsMask; c.bishops = bishopsMask;
	c.rooks = rooksMask;   c.queens  = queensMask;  c.kings   = kingsMask;
	c.white = whiteMask;   c.black   = blackMask;   c.occupied = occupiedMask;
	c.castling_rights = castlingRights;
	c.turn = turn;
	c.moveNum = moveNum;

	const uint64_t typeMasks[6] = {pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask};
	c.npm_white = c.npm_black = 0;
	for (int t = 0; t < 6; ++t){
		const int w = __builtin_popcountll(typeMasks[t] & whiteMask);
		const int b = __builtin_popcountll(typeMasks[t] & blackMask);
		c.cnt_white[t] = (int8_t)w;
		c.cnt_black[t] = (int8_t)b;
		if (t != 0 && t != 5){                       // non-pawn, non-king
			c.npm_white += w * values[t + 1];
			c.npm_black += b * values[t + 1];
		}
	}

	// Phase from non-pawn material over BOTH sides -- pawns excluded so that trading pawns never moves
	// the phase, and so the pawn's own taper cannot feed back into the phase that computes it.
	{
		const int lo = Config::EVAL_V2_EG_LIMIT, hi = Config::EVAL_V2_MG_LIMIT;
		int npm = c.npm_white + c.npm_black;
		if (npm <= lo)      c.phase256 = 0;
		else if (npm >= hi) c.phase256 = 256;
		else                c.phase256 = (256 * (npm - lo)) / (hi - lo);
	}

	// Material census, ONCE, with the phase already known.
	c.mat_white = c.mat_black = 0;
	for (int t = 0; t < 6; ++t){
		const int v = v2_piece_value(t, c.phase256);
		c.mat_white += (int)c.cnt_white[t] * v;
		c.mat_black += (int)c.cnt_black[t] * v;
	}
}

// ===================================================================================================
// RUNG 0 -- MATERIAL + PIECE-SQUARE TABLES
// ===================================================================================================

/*
	The irreducible core: what each side owns, and where it stands.

	Returns Black-positive milli-pawns and also hands back the four per-side magnitudes, because the colour
	ship-gate needs SIGNED PER-SIDE terms to be non-vacuous. A bare-material rung is colour-symmetric by
	construction and would let diagnostics/_eval_symmetry.py report "clean" having discriminated nothing --
	and we would then trust that gate for the rest of the project. Including placement, and publishing the
	per-side sums, gives all three of the gate's invariant classes something real to check from commit one.

	Placement comes from whitePlacementLayer / blackPlacementLayer, which are rebuilt once at init by
	rebuild_scaled_placement() and are READ-ONLY here -- they are also consulted by move ordering
	(move_gen.h:91), so v2 must never write them.

	Cost: two popcount-driven bit loops per piece type, no allocation, no globals.
*/
inline int rung0_material_and_placement(const V2Context &c,
                                        int &w_mat, int &b_mat, int &w_pst, int &b_pst)
{
	const uint64_t typeMasks[6] = {c.pawns, c.knights, c.bishops, c.rooks, c.queens, c.kings};

	w_mat = c.mat_white;                 // already censused exactly once; never recount
	b_mat = c.mat_black;
	w_pst = 0;
	b_pst = 0;

	for (int t = 0; t < 6; ++t){
		uint64_t wb = typeMasks[t] & c.white;
		while (wb){
			const uint8_t sq = __builtin_ctzll(wb);
			wb &= wb - 1;
			w_pst += whitePlacementLayer[t][sq & 7][sq >> 3];
		}
		uint64_t bb = typeMasks[t] & c.black;
		while (bb){
			const uint8_t sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			b_pst += blackPlacementLayer[t][sq & 7][sq >> 3];
		}
	}

	// Black-positive: White's holdings subtract, Black's add.
	return (b_mat + b_pst) - (w_mat + w_pst);
}

// ===================================================================================================
// RUNG 1 -- KING SAFETY (KS-A)
// ===================================================================================================

/*
	Per-side attack maps, built from the 12 arguments and the init-frozen BB_* tables. Nothing global is
	read or written -- attacks_mask(colour, occupied, square, pieceType) is fully parameterised.

	`all` = every square this side attacks · `dbl` = squares it attacks at least TWICE · the per-type masks
	are needed because the weak-square and safe-check tests are type-specific. `dbl` costs one AND per
	piece, which is why it is accumulated here rather than recomputed later.
*/
struct SideAttacks {
	uint64_t all, dbl;
	uint64_t by[7];      // indexed PAWN..KING (1..6); [0] unused
};

inline void build_side_attacks(SideAttacks &sa, const V2Context &c, bool white)
{
	sa.all = sa.dbl = 0;
	for (int t = 0; t < 7; ++t) sa.by[t] = 0;

	const uint64_t own = white ? c.white : c.black;
	const uint64_t typeMasks[6] = {c.pawns, c.knights, c.bishops, c.rooks, c.queens, c.kings};
	for (int t = 0; t < 6; ++t){
		const uint8_t pt = (uint8_t)(t + 1);
		uint64_t bb = typeMasks[t] & own;
		while (bb){
			const uint8_t sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			// ★ SF11:268-271 computes slider attacks THROUGH queens (and, for rooks, through own rooks)
			// so BATTERIES register: a queen behind a bishop, or doubled rooks, still bear on the zone.
			uint64_t occ_for = c.occupied;
			if (Config::KS_V2_XRAY){
				if (pt == BISHOP)    occ_for = c.occupied ^ c.queens;
				else if (pt == ROOK) occ_for = c.occupied ^ c.queens ^ (c.rooks & own);
			}
			const uint64_t a = attacks_mask(white, occ_for, sq, pt);
			sa.dbl |= sa.all & a;
			sa.all |= a;
			sa.by[pt] |= a;
		}
	}
}

/*
	The king zone: the king square, its ring, and one rank forward toward the enemy.

	⚠️ The FILE is clamped to b..g before building, as SF does. A king on the a-file otherwise gets a ring
	that falls off the board edge and under-counts its exposure by a third; clamping keeps the zone the same
	SIZE everywhere, so attacker counts are comparable across king placements instead of silently rewarding
	a corner king.
	⚠️ v1 carries THREE zone tables (king_zones, _lean, _clamped) of which one is live. v2 has one.
*/
inline uint64_t ks_zone(uint8_t ksq, bool white, uint64_t own_pawns)
{
	int f = ksq & 7;
	int r = ksq >> 3;
	if (f < 1) f = 1; else if (f > 6) f = 6;

	if (Config::KS_V2_ZONE_SF){
		// ★ SF11:239-247 -- clamp BOTH axes, ring + centre, NO forward extension, then remove squares
		// defended by two of our own pawns (a square two pawns hold is not a hole).
		if (r < 1) r = 1; else if (r > 6) r = 6;
		const uint8_t cs = (uint8_t)((r << 3) | f);
		uint64_t z = BB_KING_ATTACKS[cs] | (1ULL << cs);
		uint64_t l, rr;
		if (white){ l = (own_pawns & ~BB_FILE_A) << 7; rr = (own_pawns & ~BB_FILE_H) << 9; }
		else      { l = (own_pawns & ~BB_FILE_A) >> 9; rr = (own_pawns & ~BB_FILE_H) >> 7; }
		return z & ~(l & rr);
	}

	const uint8_t cs = (uint8_t)((r << 3) | f);
	uint64_t z = BB_KING_ATTACKS[cs] | (1ULL << cs);
	z |= white ? (z << 8) : (z >> 8);         // one rank toward the enemy
	return z;
}

/*
	Attack units borne against ONE king. Returns SF-scale units (0 .. ~2000), NOT millipawns.

	`white_king` selects whose king is examined; the ATTACKER is the other side. See
	dev_notes/EVAL-V2-RUNG1-KS-DESIGN.md for the full 14-item audit of SF's kingDanger, why 7 components
	are deferred, and why one (mobility -> kingDanger) is EXCLUDED as already refuted for us across a 4x
	range including cranked.
*/
inline int ks_units(const V2Context &c, const SideAttacks &wa, const SideAttacks &ba, bool white_king)
{
	const uint64_t kbb = c.kings & (white_king ? c.white : c.black);
	if (!kbb) return 0;
	const uint8_t ksq = __builtin_ctzll(kbb);

	const SideAttacks &att = white_king ? ba : wa;   // the ATTACKING side
	const SideAttacks &def = white_king ? wa : ba;   // the king's OWN side
	const uint64_t enemy = white_king ? c.black : c.white;
	const uint64_t zone  = ks_zone(ksq, white_king, c.pawns & (white_king ? c.white : c.black));

	// ── attacker count and weight ────────────────────────────────────────────────────────────────
	// ⚠️ PROFILE 0 keeps OUR ordering (queen-high, shared with Weiss); PROFILE 1 is the knight-high shape
	// SF and Ethereal share. A design split ⇒ both are first-class; neither is "the fix".
	// ☠️ SCALE, NOT RATIO. v1's raw weights are {N2, B2, R3, Q5} in v1's OWN unit scale, where KS_FLOOR=13
	// and KS_CAP=80 -- units run 0..80. Every other constant here is SF-scale (WEAK 185, ADJ 69, checks
	// 635-1080), where units run 0..2000+. Dropping the raw weights in unchanged would contribute ~14 units
	// against ~370 from weak squares and ~780 from one safe check, i.e. the attacker channel would be
	// SWITCHED OFF -- and profile 1 would then win for reasons having nothing to do with shape, confounding
	// the one comparison this rung exists to make.
	// ⇒ Our ordering is preserved EXACTLY, rescaled to SF magnitude: 2:2:3:5 x (187/12) = 31:31:47:78,
	// matching SF's weight sum of 187 so the two profiles are compared at equal total scale.
	static constexpr int W_OURS[7]   = {0, 0, 31, 31, 47, 78, 0};   // -, P, N, B, R, Q, K  (queen-high, ours)
	static constexpr int W_KNIGHT[7] = {0, 0, 81, 52, 44, 10, 0};   // knight-high (SF11)
	const int *W = Config::KS_V2_ATT_PROFILE ? W_KNIGHT : W_OURS;

	int n_att = 0, w_att = 0;
	const uint64_t typeMasks[4] = {c.knights, c.bishops, c.rooks, c.queens};
	for (int i = 0; i < 4; ++i){
		const uint8_t pt = (uint8_t)(i + 2);              // KNIGHT..QUEEN
		uint64_t bb = typeMasks[i] & enemy;
		while (bb){
			const uint8_t sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			if (attacks_mask(!white_king, c.occupied, sq, pt) & zone){
				++n_att;
				w_att += W[pt];
			}
		}
	}

	// ★ COORDINATION, as a CONTINUOUS parameter rather than a sum/product switch:
	//     COORD_MUL[n] = 256 + (n-1) * KS_V2_COORD   ⇒   0 = pure sum · 256 = pure product
	// A lone heavy attacker scores little until a second piece joins, and HOW MUCH that matters is the
	// knob. ⚠️ v1 has this idea twice (KS_ATT_PRODUCT, KS_COORD_GATE_MODE), both default-off, and its
	// record says the product is "necessary-but-insufficient, not a solo ship" -- so the interior of this
	// range is where the answer plausibly lives.
	// ★ SF seeds kingAttackersCount with enemy PAWN attacks on the ring: a pawn bearing on the zone
	// counts toward COORDINATION even though its weight is zero.
	if (Config::KS_V2_PAWN_ATT){
		const uint64_t ep = c.pawns & enemy;
		uint64_t l, rr;
		if (white_king){ l = (ep & ~BB_FILE_A) >> 9; rr = (ep & ~BB_FILE_H) >> 7; }
		else           { l = (ep & ~BB_FILE_A) << 7; rr = (ep & ~BB_FILE_H) << 9; }
		n_att += __builtin_popcountll(zone & (l | rr)) > 0 ? 1 : 0;
	}

	int u = 0;
	if (n_att > 0){
		const long long mul = 256LL + (long long)(n_att - 1) * (long long)Config::KS_V2_COORD;
		u = (int)(((long long)w_att * mul) >> 8);
	}

	// ── weak squares in the zone ─────────────────────────────────────────────────────────────────
	// SF's definition: attacked by them, NOT defended twice by us, and either undefended or defended only
	// by our king or queen -- squares whose defence collapses the moment the defender is deflected.
	const uint64_t weak = att.all & ~def.dbl & (~def.all | def.by[KING] | def.by[QUEEN]);
	u += Config::KS_V2_WEAK * __builtin_popcountll(zone & weak);

	// ── attacks landing on squares adjacent to the king ──────────────────────────────────────────
	u += Config::KS_V2_ADJ * __builtin_popcountll(att.all & BB_KING_ATTACKS[ksq]);

	// ── safe checks, a SEPARATE channel ──────────────────────────────────────────────────────────
	// A check square is safe if we do not defend it, or if it is weak and they attack it twice. Sliding
	// checks are traced THROUGH our own queen (SF does the same): a queen that can be deflected is not a
	// blocker worth trusting.
	const uint64_t occ_x      = c.occupied ^ (c.queens & (white_king ? c.white : c.black));
	const uint64_t rookRays   = attacks_mask(white_king, occ_x, ksq, ROOK);
	const uint64_t bishopRays = attacks_mask(white_king, occ_x, ksq, BISHOP);
	const uint64_t knightRays = attacks_mask(white_king, c.occupied, ksq, KNIGHT);
	const uint64_t safe       = ~enemy & (~def.all | (weak & att.dbl));

	// ⚠️ Ordering is a DESIGN SPLIT: SF ranks the rook clearly highest, v1 ties queen and rook. Profile 1
	// is v1's ordering rescaled to SF magnitude, so the two are compared at equal total weight (3285).
	// ☠️ Neither ordering has been tested; profile 0 is default only because it is what rung 1 was measured with.
	int chk_q = Config::KS_V2_CHK_Q, chk_r = Config::KS_V2_CHK_R;
	int chk_b = Config::KS_V2_CHK_B, chk_n = Config::KS_V2_CHK_N;
	if (Config::KS_V2_CHK_PROFILE == 1){ chk_q = 1046; chk_r = 1046; chk_b = 523; chk_n = 672; }

	// ⚠️ FORM is a knob because it is COUPLED to the magnitudes: SF11 fires once, Ethereal adds per
	// square. Running one engine's constants in the other's form mis-weights the whole channel.
	if (Config::KS_V2_CHK_COUNT){
		u += chk_r * __builtin_popcountll(rookRays                & safe & att.by[ROOK]);
		u += chk_q * __builtin_popcountll((rookRays | bishopRays)  & safe & att.by[QUEEN]);
		u += chk_b * __builtin_popcountll(bishopRays               & safe & att.by[BISHOP]);
		u += chk_n * __builtin_popcountll(knightRays               & safe & att.by[KNIGHT]);
	} else {
		if (rookRays                & safe & att.by[ROOK])   u += chk_r;
		if ((rookRays | bishopRays) & safe & att.by[QUEEN])  u += chk_q;
		if (bishopRays              & safe & att.by[BISHOP]) u += chk_b;
		if (knightRays              & safe & att.by[KNIGHT]) u += chk_n;
	}

	// ── no queen ─────────────────────────────────────────────────────────────────────────────────
	// ★ The largest single term in SF's sum (-873), and correct to real chess: without a queen an attack
	// usually cannot be converted. v1 has its own version (KS_NO_QUEEN / KS_NQ_SUP).
	if (!(c.queens & enemy)) u -= Config::KS_V2_NO_QUEEN;

	// ★ ONSET: quiet positions must contribute EXACTLY zero, not a small positive. Both references do
	// this (SF `if (kingDanger > 100)`, Ethereal `SafetyAdjustment -74` + `MAX(0,...)`). Without it the
	// curve returns 400mp at u=100/HALF=300, which is pure error across the many quiet positions that
	// dominate a representative corpus.
	u -= Config::KS_V2_ONSET;
	return u > 0 ? u : 0;
}

/*
	Attack units -> millipawns, bounded BY CONSTRUCTION:

	    danger(u) = MAX * u^2 / (u^2 + HALF^2)

	Zero at u=0, quadratic onset, saturating at MAX as a LIMIT rather than a std::min. ⚠️ There is
	deliberately NO phase gate -- danger decays on its own in the endgame because there are fewer attackers,
	and the PRODUCT form makes that decay steeper still. v1 hard-zeroed KS above KS_PHASE_ZERO, which blinds
	it to back-rank mates and mating nets, and our 3-way phase cliffs at ONE minor trade so that gate fires
	far earlier than "deep endgame" suggests.

	64-bit arithmetic: u reaches ~2000 so u*u ~4e6, and MAX*u*u can reach ~1.6e10 -- past 32 bits.
*/
inline int ks_danger_mp(int u)
{
	if (u <= 0 || Config::KS_V2_MAX <= 0) return 0;
	const long long uu = (long long)u * (long long)u;
	const long long hh = (long long)Config::KS_V2_HALF * (long long)Config::KS_V2_HALF;
	return (int)(((long long)Config::KS_V2_MAX * uu) / (uu + hh));
}

// ===================================================================================================
// RUNG 2a -- PAWN STRUCTURE
// ===================================================================================================

/* LAYER A output. Every mask is a pure function of the two pawn bitboards -- no piece, no king, no side
 * to move is read to produce any of it.
 *
 * ★ That purity is not incidental. It is simultaneously (a) what will make this layer cacheable on
 * pawnKey, and (b) the owner's DETECTOR stage, separable from scoring. Those two boundaries turning out
 * to be the same line is the structural gift of this rung: the split that makes it testable is the split
 * that makes it fast. Index 0 = White, 1 = Black throughout.
 */
struct PawnEntry {
	uint64_t passed[2], candidate[2];
	uint64_t isolated[2], doubled[2], backward[2];
	uint64_t phalanx[2], supported[2], opposed[2], lever[2];
	uint64_t blocked[2], stop_held[2];
	uint64_t attacks[2], attacks2[2];
	uint8_t  openFiles, halfOpen[2];
};

/* Set-wise helpers. Deliberately tiny and side-agnostic: every predicate below is a fill or a shift, not
 * a per-square loop, which is what keeps the detector a pure function of two bitboards.
 */
static inline uint64_t ps_nfill(uint64_t b) noexcept { b |= b << 8; b |= b << 16; b |= b << 32; return b; }
static inline uint64_t ps_sfill(uint64_t b) noexcept { b |= b >> 8; b |= b >> 16; b |= b >> 32; return b; }
static inline uint64_t ps_east(uint64_t b)  noexcept { return (b & ~BB_FILE_H) << 1; }
static inline uint64_t ps_west(uint64_t b)  noexcept { return (b & ~BB_FILE_A) >> 1; }
static inline uint64_t ps_watt(uint64_t p)  noexcept { return ((p & ~BB_FILE_A) << 7) | ((p & ~BB_FILE_H) << 9); }
static inline uint64_t ps_batt(uint64_t p)  noexcept { return ((p & ~BB_FILE_A) >> 9) | ((p & ~BB_FILE_H) >> 7); }

/* LAYER A -- the DETECTOR.
 *
 * ☠️ The detector RETURNS its masks and must NEVER publish them as a side effect. v1's getPPIncrement
 * writes white_passed_pawns / black_passed_pawns / candidate_passed_pawns as it goes, which is exactly
 * what makes a pawn hash unsafe there: a cache hit would skip the write and every downstream piece
 * evaluator -- all ten of them take the passer masks as parameters -- would read stale state. Same failure
 * family as eval-global-side-effects-are-skipped-by-a-cache-hit.
 *
 * ✅ Every predicate here is mirrored in diagnostics/_pawn_term_overlap.py, which is validated 8/8 on
 * hand-checked positions and colour-symmetric 3/3. That file is this function's ORACLE -- compare
 * mask-for-mask over the corpus before trusting any score built on top, because a detector bug and a
 * scoring bug are indistinguishable from outside.
 *
 * @param e  output; fully overwritten
 * @param c  context; only c.pawns, c.white, c.black are read
 *
 * Gating: none -- the caller gates on PS_V2_MAG before invoking.
 * Cost: ~30 shift/fill ops, no loops over squares, no allocation.
 */
static inline void build_pawn_entry(PawnEntry &e, const V2Context &c) noexcept
{
	const uint64_t wp = c.pawns & c.white;
	const uint64_t bp = c.pawns & c.black;

	e.attacks[0]  = ps_watt(wp);
	e.attacks[1]  = ps_batt(bp);
	e.attacks2[0] = ((wp & ~BB_FILE_A) << 7) & ((wp & ~BB_FILE_H) << 9);
	e.attacks2[1] = ((bp & ~BB_FILE_A) >> 9) & ((bp & ~BB_FILE_H) >> 7);

	for (int s = 0; s < 2; ++s){
		const bool     white = (s == 0);
		const uint64_t own   = white ? wp : bp;
		const uint64_t enemy = white ? bp : wp;
		const uint64_t adj   = ps_east(own) | ps_west(own);

		// isolated: no own pawn anywhere on an adjacent file
		e.isolated[s]  = own & ~ps_nfill(ps_sfill(adj));
		// doubled: an own pawn directly AHEAD on the same file (the rear pawn of the pair carries it)
		e.doubled[s]   = own & (white ? (own >> 8) : (own << 8));
		// phalanx: an own pawn beside it on the same rank.  ★ This is v1's "pawn wall".
		e.phalanx[s]   = own & adj;
		// supported: defended by one of our own pawns.      ★ This is v1's "chain".
		e.supported[s] = own & e.attacks[s];
		// opposed: an enemy pawn anywhere ahead on our OWN file
		e.opposed[s]   = own & (white ? ps_sfill(enemy) : ps_nfill(enemy));
		// lever: WE attack an enemy pawn. A white pawn on sq attacks sq+7/sq+9, so the squares attacking
		// an enemy pawn at X are exactly {X-9, X-7} = ps_batt(X). Mirrored for Black.
		e.lever[s]     = own & (white ? ps_batt(enemy) : ps_watt(enemy));

		const uint64_t stop = white ? (own << 8) : (own >> 8);
		const uint64_t eatt = white ? e.attacks[1] : e.attacks[0];
		e.blocked[s]   = own & (white ? ((stop & enemy) >> 8) : ((stop & enemy) << 8));
		e.stop_held[s] = own & (white ? ((stop & eatt)  >> 8) : ((stop & eatt)  << 8));

		// backward (SF's condition): NO friendly neighbour on an adjacent file at or BEHIND our rank, and
		// the push is blocked or contested. ⚠️ ps_nfill smears NORTH, so bit (f,r) is set iff a source sits
		// at r' <= r -- "a neighbour at or behind us" for White. Mirrored for Black. Getting this backwards
		// produces entirely plausible numbers, which is why the oracle checks it rather than review alone.
		const uint64_t rear_nb = own & (white ? ps_nfill(adj) : ps_sfill(adj));
		e.backward[s]  = own & ~rear_nb & (e.blocked[s] | e.stop_held[s]);

		// ── RUNG 2b: passed and candidate ────────────────────────────────────────────────────────
		// passed: no enemy pawn anywhere in the forward three-file span. `passed_span_*` is precomputed
		// at init and is the same mask shape getPPIncrement uses, so this is one lookup per pawn.
		uint64_t pb = 0, cb = 0;
		uint64_t it = own;
		while (it){
			const uint8_t sq = (uint8_t)__builtin_ctzll(it);
			it &= it - 1;
			const uint64_t m    = 1ULL << sq;
			const uint64_t span = white ? passed_span_white[sq] : passed_span_black[sq];
			const uint64_t st   = span & enemy;            // SF's `stoppers`
			if (!st){ pb |= m; continue; }

			// ★ SF CANDIDATE passers -- the block v1 hides behind ENABLE_PASSER_DETECT_SF, which is FALSE,
			// which is why we miss ~14% of SF's passers. A pawn with stoppers still counts as passed when
			// every stopper is a pawn WE attack (lever), or every stopper is a pawn our PUSHED self would
			// attack and our phalanx is at least as large (leverPush), or the only stopper is a same-file
			// blocker we out-support from the 5th rank up (blocked).
			uint64_t lev, sup, front, lpush, sps, rear;
			int rank_owner;
			const uint64_t phal = (ps_east(m) | ps_west(m)) & own;
			if (white){
				lev   = (((m & ~BB_FILE_A) << 7) | ((m & ~BB_FILE_H) << 9)) & enemy;
				sup   = (((m & ~BB_FILE_H) >> 7) | ((m & ~BB_FILE_A) >> 9)) & own;
				front = m << 8;
				lpush = (((front & ~BB_FILE_A) << 7) | ((front & ~BB_FILE_H) << 9)) & enemy;
				sps   = (sup << 8) & ~enemy;
				rear  = ps_nfill(front) & ps_nfill(ps_sfill(m)) & own;
				rank_owner = (sq >> 3) + 1;
			} else {
				lev   = (((m & ~BB_FILE_A) >> 9) | ((m & ~BB_FILE_H) >> 7)) & enemy;
				sup   = (((m & ~BB_FILE_H) << 9) | ((m & ~BB_FILE_A) << 7)) & own;
				front = m >> 8;
				lpush = (((front & ~BB_FILE_A) >> 9) | ((front & ~BB_FILE_H) >> 7)) & enemy;
				sps   = (sup >> 8) & ~enemy;
				rear  = ps_sfill(front) & ps_nfill(ps_sfill(m)) & own;
				rank_owner = 8 - (sq >> 3);
			}
			const uint64_t blk = front & enemy;
			const bool ok = ((st ^ lev) == 0)
			             || (((st ^ lpush) == 0) && __builtin_popcountll(phal) >= __builtin_popcountll(lpush))
			             || (st == blk && st != 0 && rank_owner >= 5 && sps != 0);
			// ⚠️ A REAR-DOUBLED pawn (a friendly pawn AHEAD on its own file) can never promote, so it is
			// never a candidate however favourable its stoppers look.
			if (ok && !rear) cb |= m;
		}
		e.passed[s]    = pb;
		e.candidate[s] = cb;
	}

	// File occupancy: detected HERE, consumed at rung 6 (rook files). Two fills we are already paying for,
	// and it saves the rook pass re-deriving what the pawn pass already knows.
	const uint64_t wfiles = ps_nfill(ps_sfill(wp));
	const uint64_t bfiles = ps_nfill(ps_sfill(bp));
	e.openFiles   = (uint8_t)(~(wfiles | bfiles) & 0xFFULL);
	e.halfOpen[0] = (uint8_t)(~wfiles & 0xFFULL);
	e.halfOpen[1] = (uint8_t)(~bfiles & 0xFFULL);
}

// SF11 Connected[] = {0,7,8,12,29,48,86} in SF units, pre-multiplied by its neutral modifier (2) and
// converted at mg x7.81. Indexed by RELATIVE rank, 0-based, so index 6 is the 7th rank.
static constexpr int PS_CONN_RANK_MP[8] = {0, 109, 125, 187, 453, 750, 1343, 0};

// Ethereal PawnConnected32's rank-7 file shape {108,214,216,233}, mirrored and normalised so the mean is
// 256 (= neutral). ★ Its centre:edge ratio is 2.16x. v1's pawn_chain_file_bonus runs 15.0x AND applies at
// every rank -- the outlier against BOTH references, on both axes independently.
static constexpr int PS_CONN_FILE_256[8] = {143, 284, 287, 309, 309, 287, 284, 143};

// Ethereal PawnIsolated[FILE], converted (mg x12.20, eg x6.94). ☠️ POSITIVE = a BONUS to the owner.
// Note the midgame row is positive in the CENTRE: Ethereal PAYS for a central isolated pawn and charges
// only on the wings, where SF11 charges -39mp flat everywhere. That sign disagreement between two tuned
// references is why PS_V2_ISOLATED_MG defaults to 0.
//
// ☠️ MIRROR-SYMMETRISED, and this was caught by the gate, not by review. Ethereal's raw table is itself
// file-ASYMMETRIC (a-file -13 vs h-file -4 in mg; -83 vs -118 in eg). Transcribed verbatim it made an
// a-file isolated pawn worth something different from an h-file one, which has no justification in a game
// whose rules are file-mirror symmetric -- and _eval_symmetry.py's file-mirror check went from 21 to 344
// violations (52.8%) the moment the rung was switched on. Each file is now averaged with its mirror.
// ★ Doing so also shows the ENDGAME row is essentially FLAT once symmetrised (-101..-129): Ethereal's
// apparent endgame file structure was tuner noise, and only the MIDGAME row carries real shape.
static constexpr int PS_ISO_FILE_MG[8] = {-104,  -31,   25,   61,   61,   25,  -31, -104};
static constexpr int PS_ISO_FILE_EG[8] = {-101, -104, -108, -129, -129, -108, -104, -101};

/* LAYER B -- the structure SCORE. Reads ONLY Layer A's masks, so it caches alongside them.
 * Returns Black-positive milli-pawns: White's structure subtracts, Black's adds.
 *
 * WHY THIS SHAPE. v1 scores the same information as `chain_file[f] * support + wall_file[f'] * phalanx` --
 * two separate FILE-keyed terms, no rank term anywhere. SF11 (pawns.cpp:43/:135) uses ONE RANK-keyed term
 * with phalanx and support as MODIFIERS; Ethereal uses one rank-AND-file table whose tilt is a rank-6/7
 * phenomenon. ★ v1's form is additive and therefore SEPARABLE: it structurally cannot express "file
 * matters at rank 7 but not rank 3", which is precisely what Ethereal's table says. Merging the two terms
 * into one 2D-keyed term is what makes that expressible -- the owner's file concept survives, its
 * representation changes.
 * ⚠️ Measured 2026-09-12: v1's file-keyed chain bonus is harmful on 6/6 corpora once PAWN_CLAMP stops
 * masking it (-2.29%), as is the post-hoc opposed multiplier (-1.38%).
 *
 * @param e  detector output
 * @param c  context; only c.pawns/white/black and c.phase256 are read
 * @return   Black-positive milli-pawns, phase-blended and scaled by PS_V2_MAG
 *
 * Gating: PS_V2_MAG == 0 returns 0 => byte-identical to the rung-1 baseline.
 * Cost: one bit loop per side over connected pawns plus one over isolated; three popcounts. NO CLAMP.
 */
static inline int pawn_structure_mp(const PawnEntry &e, const V2Context &c)
{
	if (Config::PS_V2_MAG == 0) return 0;

	int side_mg[2] = {0, 0};
	int side_eg[2] = {0, 0};

	for (int s = 0; s < 2; ++s){
		const bool     white = (s == 0);
		const uint64_t own   = c.pawns & (white ? c.white : c.black);

		// --- connected: ONE term, rank-primary, phalanx/opposed/support as modifiers ------------------
		// ★ Ethereal applies connected as an else-if AFTER backward, so the two are mutually exclusive;
		// SF stacks them. PS_V2_CONN_EXCL selects which, because the references disagree.
		uint64_t bb = own & (e.phalanx[s] | e.supported[s]);
		if (Config::PS_V2_CONN_EXCL >= 1) bb &= ~e.backward[s];
		while (bb){
			const uint8_t  sq = (uint8_t)__builtin_ctzll(bb);
			bb &= bb - 1;
			const uint64_t m  = 1ULL << sq;
			const int      f  = sq & 7;
			const int      r  = white ? (sq >> 3) : (7 - (sq >> 3));   // relative rank, 0-based

			// SF's (2 + phalanx - opposed), expressed in /256 so the neutral case is exactly 256.
			int mod = 256;
			if (m & e.phalanx[s]) mod += 128;
			if (m & e.opposed[s]) mod -= 128;

			// FORM 0 = 2D rank x file (target) - FORM 1 = rank only (SF11 exactly).
			// FORM 2 (v1's additive rank+file) is reserved and currently behaves as FORM 1; it exists so
			// the refuted form can be reinstated deliberately rather than reconstructed from memory.
			// FORM 2 is RANK-FLAT: the PST owns rank, this owns only "is it connected".
			int base = (Config::PS_V2_CONN_FORM == 2) ? Config::PS_V2_CONN_FLAT : PS_CONN_RANK_MP[r];
			if (Config::PS_V2_CONN_FORM == 0 && r >= Config::PS_V2_TILT_MIN_RANK){
				const int tilt = 256 + ((PS_CONN_FILE_256[f] - 256) * Config::PS_V2_FILE_TILT) / 256;
				base = (base * tilt) >> 8;
			}

			int v = (base * mod) >> 8;

			// Supporter COUNT, not a boolean: the squares from which our pawns defend sq are exactly
			// ps_batt(sq) for White (mirrored for Black) -- the same inversion trick as `lever` above.
			const uint64_t sup = own & (white ? ps_batt(m) : ps_watt(m));
			if (sup) v += Config::PS_V2_SUPPORT * __builtin_popcountll(sup);

			if (Config::PS_V2_CONN_MAG != 100) v = v * Config::PS_V2_CONN_MAG / 100;
			side_mg[s] += v;
			// SF derives its endgame leg from the SAME v, as `v * (r - 2) / 4`. ⚠️ We cannot reuse its
			// constant directly because our mg and eg conversions differ (x7.81 vs x4.69), so
			// PS_V2_EG_RATIO carries that difference EXPLICITLY instead of burying it in a table.
			// ☠️ All three unit-scale errors during rung 1 came from mixing conversion bases silently.
			// ⚠️ FORM 2's endgame leg must be flat too. Deriving it as v*(r-2)/4 would smuggle the rank
			// dependence straight back in through the eg half, which is the very thing form 2 removes.
			side_eg[s] += (Config::PS_V2_CONN_FORM == 2)
			              ? v
			              : ((v * (r - 2)) / 4) * Config::PS_V2_EG_RATIO / 100;
		}

		// --- doubled: v1 HAS this (hardcoded 125 mg / 150 eg); the defect was the FLAT taper -----------
		const int nd = __builtin_popcountll(e.doubled[s]);
		side_mg[s] -= nd * Config::PS_V2_DOUBLED_MG;
		side_eg[s] -= nd * Config::PS_V2_DOUBLED_EG;

		// --- isolated: FILE table, because a scalar cannot express wing-penalty / centre-bonus ---------
		uint64_t ib = e.isolated[s];
		while (ib){
			const int f = (int)(__builtin_ctzll(ib) & 7);
			ib &= ib - 1;
			side_mg[s] += PS_ISO_FILE_MG[f] * Config::PS_V2_ISOLATED_MG / 100;
			side_eg[s] += PS_ISO_FILE_EG[f] * Config::PS_V2_ISOLATED_EG / 100;
		}

		// --- backward: both references agree the ENDGAME leg is negative; the midgame is contested -----
		const int nb = __builtin_popcountll(e.backward[s]);
		side_mg[s] -= nb * Config::PS_V2_BACKWARD_MG;
		side_eg[s] -= nb * Config::PS_V2_BACKWARD_EG;
	}

	// Phase-blend each side, then combine Black-positive. c.phase256: 256 = full midgame, 0 = deep endgame.
	const int w = (side_mg[0] * c.phase256 + side_eg[0] * (256 - c.phase256)) >> 8;
	const int b = (side_mg[1] * c.phase256 + side_eg[1] * (256 - c.phase256)) >> 8;
	return ((b - w) * Config::PS_V2_MAG) / 100;
}


// ===================================================================================================
// BREAKDOWN PUBLICATION
// ===================================================================================================

/*
	Publish what this rung genuinely computed, and set exactly those EB_* bits.

	⚠️ Terms v2 does not compute are left unset, NOT zeroed-and-claimed: eval_breakdown_capture already
	clears the struct, and ChessAI.ev_breakdown omits any key whose bit is clear so a consumer asking for it
	raises KeyError. A silent 0 on 172 diagnostic files is the expensive failure; a KeyError is a cheap one.

	phase_score / is_endgame are deliberately NOT published at rung 0. v2 has no phase model yet, and
	publishing v1's convention (inverted, 0 = opening, 128 = bare kings, consumed as a 3-way boolean) would
	invite the BY_PS strata to be read as if they still meant what their labels say.
*/
inline void publish_rung0(int total, const V2Context &c, int w_mat, int b_mat, int w_pst, int b_pst)
{
	EvalBreakdown &b = g_eval_breakdown;

	b.total    = total;
	b.material = b_mat - w_mat;          // Black-positive, matching `total`'s convention
	b.pieces   = b_pst - w_pst;

	// Signed per-side holdings. SIGNED_SWAP class in the symmetry gate: mirroring must negate AND swap
	// these, which simultaneously checks that whitePlacementLayer and blackPlacementLayer really are
	// mirror images of one another. total == imbalance_white + imbalance_black is an exact partition.
	b.imbalance_white = -(w_mat + w_pst);
	b.imbalance_black =  (b_mat + b_pst);

	// Magnitudes. Plain-SWAP class in the symmetry gate.
	b.det_w_pieceval = w_mat;
	b.det_b_pieceval = b_mat;

	b.det_pawn_count = __builtin_popcountll(c.pawns);

	b.arm = Config::EVAL_ARM;
	b.terms_valid = (1ULL << EB_TOTAL)           | (1ULL << EB_MATERIAL)
	              | (1ULL << EB_PIECES)          | (1ULL << EB_IMBALANCE_WHITE)
	              | (1ULL << EB_IMBALANCE_BLACK) | (1ULL << EB_DET_W_PIECEVAL)
	              | (1ULL << EB_DET_B_PIECEVAL)  | (1ULL << EB_DET_PAWN_COUNT);
}

// ===================================================================================================
// SHADOW-ARM ACCUMULATORS (Config::EVAL_ARM == 2 only)
// ===================================================================================================

/*
	The v2-v1 disagreement distribution over the real search distribution.

	⚠️ Plain counters, not atomics: this is a diagnostic arm and the bench lanes that use it run
	single-threaded. Under a threaded run the totals are approximate. They are never read by search.
*/
long long g_shadow_n      = 0;
long long g_shadow_sum    = 0;          // signed sum of (v2 - v1)
long long g_shadow_absum  = 0;
int       g_shadow_min    = 0;
int       g_shadow_max    = 0;
long long g_shadow_bucket[6] = {0,0,0,0,0,0};   // |delta| < 100, 300, 1000, 3000, 10000, rest

constexpr long long SHADOW_REPORT_STRIDE = 1LL << 20;

} // anonymous namespace

// ⚠️ pawn_entry_probe lives OUTSIDE the anonymous namespace deliberately. Everything above is
// internal by construction -- that is what enforces the zero-global contract -- but this one is
// called from Cython, so internal linkage would (and did) fail the link with an undefined symbol.

// ===================================================================================================
// RUNG 2b -- PASSED PAWNS
// ===================================================================================================

// SF11 PassedRank[] converted to our milli-pawns (mg x7.81, eg x4.69), indexed by RELATIVE rank, 0-based,
// so index 6 is the 7th rank. ⚠️ SF's own pawn PST (PBonus) has essentially NO rank ramp -- tiny and
// non-monotonic -- and neither does ours (flat ~25mp from rank 3, ZERO at rank 7). So this table is the
// WHOLE advancement price in both engines, and adding it over our PST is structurally correct rather than
// a double-count. That was verified before a constant was chosen; it is the check skipped at rung 2a.
static constexpr int PS_PASSED_MG[8] = {0,  78, 133, 117, 484, 1312, 2156, 0};
static constexpr int PS_PASSED_EG[8] = {0, 131, 155, 192, 338,  830, 1219, 0};

/* Chebyshev (king-move) distance between two squares, capped at 5 as SF caps king_proximity.
 * @return 0..5
 */
static inline int ps_kdist(int a, int b) noexcept
{
	const int dx = ((a & 7) > (b & 7)) ? (a & 7) - (b & 7) : (b & 7) - (a & 7);
	const int dy = ((a >> 3) > (b >> 3)) ? (a >> 3) - (b >> 3) : (b >> 3) - (a >> 3);
	const int d  = dx > dy ? dx : dy;
	return d > 5 ? 5 : d;
}

/* LAYER C -- passed-pawn value. Piece/king dependent, so NOT part of the cacheable pawn entry.
 * Returns Black-positive milli-pawns: White's passers subtract, Black's add.
 *
 * ★ THE FORM IS THE POINT. The rank table is granted UNCONDITIONALLY and every modifier is ADDITIVE.
 * All five references agree on this (SF1, SF11, SF15.1, Ethereal, Weiss) and NOT ONE multiplies by a
 * realizability factor. v1 computes `mag * R / 256` with R in [0,320], and v1's own comment records the
 * measured consequence: below R ~21/256 a passer collects LESS than the same pawn would have earned for
 * NOT being passed -- 6mp where the ordinary rank bonus is 90. Being recognised as passed must never make
 * a pawn worth less. There is no R here, and no clamp chain.
 *
 * ⚠️ The EXTRAS are rank-gated at PASSER_V2_MIN_RANK (default 3 = the 4th rank), which is universal:
 * SF11 gates on `r > RANK_3`, Weiss on `if (rank < RANK_4) continue`. Below it a passer gets the table and
 * nothing else. v1 runs its whole machinery at every rank.
 *
 * @param e  detector output; reads passed[] and candidate[] only
 * @param c  context; kings, occupancy and phase256
 * @return   Black-positive milli-pawns, phase-blended and scaled by PASSER_V2_MAG
 *
 * Gating: PASSER_V2_MAG == 0 returns 0 => byte-identical to the rung-2a baseline.
 * Cost: one bit loop per side over passers only (11.72% of pawns), two distance computations each.
 */
static inline int passer_value_mp(const PawnEntry &e, const V2Context &c)
{
	if (Config::PASSER_V2_MAG == 0) return 0;

	const int wk = (c.kings & c.white) ? __builtin_ctzll(c.kings & c.white) : 0;
	const int bk = (c.kings & c.black) ? __builtin_ctzll(c.kings & c.black) : 0;

	int side_mg[2] = {0, 0};
	int side_eg[2] = {0, 0};

	for (int s = 0; s < 2; ++s){
		const bool white  = (s == 0);
		const int  ourK   = white ? wk : bk;
		const int  theirK = white ? bk : wk;

		uint64_t bb = e.passed[s] | e.candidate[s];
		while (bb){
			const int sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			const uint64_t m = 1ULL << sq;
			const int r = white ? (sq >> 3) : (7 - (sq >> 3));

			int mg = PS_PASSED_MG[r];
			int eg = PS_PASSED_EG[r];

			if (r >= Config::PASSER_V2_MIN_RANK){
				// SF's rank weight: king proximity matters more the further advanced the pawn.
				int w = 5 * r - 13;
				if (w < 0) w = 0;
				const int stop = white ? (sq + 8) : (sq - 8);
				if (stop >= 0 && stop < 64){
					// ENDGAME leg only, as SF: make_score(0, ...). ★ Their king's distance outweighs ours
					// ~2.4x in SF and ~2.7x in v1 -- the one passer ratio v1 already has right.
					eg += (ps_kdist(theirK, stop) * Config::PASSER_V2_KING_THEM
					     - ps_kdist(ourK,   stop) * Config::PASSER_V2_KING_US) * w / 100;
				}
			}

			// SF scales candidates down: they still face a stopper and need more than one push.
			if (m & e.candidate[s]){
				mg = mg * Config::PASSER_V2_CAND_PCT / 100;
				eg = eg * Config::PASSER_V2_CAND_PCT / 100;
			}

			side_mg[s] += mg * Config::PASSER_V2_MG_PCT / 100;
			side_eg[s] += eg * Config::PASSER_V2_EG_PCT / 100;
		}
	}

	const int w = (side_mg[0] * c.phase256 + side_eg[0] * (256 - c.phase256)) >> 8;
	const int b = (side_mg[1] * c.phase256 + side_eg[1] * (256 - c.phase256)) >> 8;
	return ((b - w) * Config::PASSER_V2_MAG) / 100;
}

/* DETECTOR ORACLE PROBE -- exports Layer A's raw masks so they can be compared against the independent
 * Python implementation in diagnostics/_pawn_term_overlap.py, which is validated 8/8 on hand-checked
 * positions and colour-symmetric 3/3.
 *
 * ★ WHY A PROBE AND NOT A SCORE CHECK. A detector bug and a scoring bug are indistinguishable from
 * outside: both show up as "the eval moved". Comparing MASKS against an independently-written reference
 * separates them completely, and it is the one correctness check available to us that does not depend on
 * any constant being right. ⚠️ It exists because of the 2026-09-12 near-miss where a transcription defect
 * (Ethereal's file-asymmetric isolated table) was caught by a symmetry gate rather than by reading code.
 *
 * Layout, 2 entries per predicate, [White, Black]:
 *   0-1 isolated - 2-3 doubled - 4-5 backward - 6-7 phalanx - 8-9 supported - 10-11 opposed
 *   12-13 lever - 14-15 blocked - 16-17 stop_held - 18-19 attacks - 20 openFiles - 21-22 halfOpen
 *   23-24 passed - 25-26 candidate
 *
 * @param pawnsMask  all pawns
 * @param whiteMask  all White occupancy
 * @param blackMask  all Black occupancy
 * @param out        caller-provided, at least 27 entries; fully written
 *
 * Gating: none -- diagnostic only, never called from search.
 * Cost: one detector build. Not on any hot path.
 */
void pawn_entry_probe(uint64_t pawnsMask, uint64_t whiteMask, uint64_t blackMask, uint64_t *out)
{
	V2Context c{};
	c.pawns = pawnsMask;
	c.white = whiteMask;
	c.black = blackMask;

	PawnEntry e;
	build_pawn_entry(e, c);

	for (int s = 0; s < 2; ++s){
		out[0  + s] = e.isolated[s];
		out[2  + s] = e.doubled[s];
		out[4  + s] = e.backward[s];
		out[6  + s] = e.phalanx[s];
		out[8  + s] = e.supported[s];
		out[10 + s] = e.opposed[s];
		out[12 + s] = e.lever[s];
		out[14 + s] = e.blocked[s];
		out[16 + s] = e.stop_held[s];
		out[18 + s] = e.attacks[s];
		out[21 + s] = (uint64_t)e.halfOpen[s];
		out[23 + s] = e.passed[s];
		out[25 + s] = e.candidate[s];
	}
	out[20] = (uint64_t)e.openFiles;
}


void eval_v2_shadow_record(int v1, int v2)
{
	const int d = v2 - v1;
	const int a = d < 0 ? -d : d;

	if (g_shadow_n == 0){ g_shadow_min = d; g_shadow_max = d; }
	else { if (d < g_shadow_min) g_shadow_min = d; if (d > g_shadow_max) g_shadow_max = d; }

	g_shadow_n++;
	g_shadow_sum   += d;
	g_shadow_absum += a;
	g_shadow_bucket[a < 100 ? 0 : a < 300 ? 1 : a < 1000 ? 2 : a < 3000 ? 3 : a < 10000 ? 4 : 5]++;

	if ((g_shadow_n % SHADOW_REPORT_STRIDE) == 0)
		eval_v2_shadow_report();
}

void eval_v2_shadow_report()
{
	if (g_shadow_n == 0) return;
	std::cerr << "[eval_v2 shadow] n=" << g_shadow_n
	          << " mean=" << (double)g_shadow_sum / (double)g_shadow_n
	          << " mean|d|=" << (double)g_shadow_absum / (double)g_shadow_n
	          << " min=" << g_shadow_min << " max=" << g_shadow_max
	          << " |d|<100=" << g_shadow_bucket[0]
	          << " <300=" << g_shadow_bucket[1]
	          << " <1000=" << g_shadow_bucket[2]
	          << " <3000=" << g_shadow_bucket[3]
	          << " <10000=" << g_shadow_bucket[4]
	          << " >=10000=" << g_shadow_bucket[5]
	          << std::endl;
}

// ===================================================================================================
// ENTRY POINT -- RUNG DISPATCH
// ===================================================================================================

/*
	See the file header for the output and purity contract. Rung selection is Config::EVAL_V2_RUNG; rungs
	are cumulative, so a rung adds to everything below it rather than replacing it.
*/
int placement_and_piece_eval_v2(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t castlingRights)
{
	V2Context c;
	build_context(c, moveNum, turn, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              occupied_whiteMask, occupied_blackMask, occupiedMask, castlingRights);

	int w_mat = 0, b_mat = 0, w_pst = 0, b_pst = 0;
	int total = rung0_material_and_placement(c, w_mat, b_mat, w_pst, b_pst);

	// ── rung 1: king safety (KS-A) ───────────────────────────────────────────────────────────────
	// Gated on KS_V2_MAX, so 0 = the rung is absent and this is byte-identical to rung 0.5. The attack maps
	// are built ONCE here and handed to both kings by reference -- they are the expensive part, and each
	// king needs the other side as attacker and its own as defender, so one build serves both.
	int ks_w = 0, ks_b = 0, ks_mp = 0;
	if (Config::KS_V2_MAX > 0){
		SideAttacks wa, ba;
		build_side_attacks(wa, c, true);
		build_side_attacks(ba, c, false);
		ks_w = ks_units(c, wa, ba, true);
		ks_b = ks_units(c, wa, ba, false);
		// Black-positive, matching `total`: a dangerous WHITE king favours Black (+), a dangerous BLACK
		// king favours White (-). ⚠️ Getting this backwards still produces entirely plausible numbers --
		// it is the historic failure mode -- so the colour ship-gate runs on every KS-A build.
		ks_mp = ks_danger_mp(ks_w) - ks_danger_mp(ks_b);
		total += ks_mp;
	}

	// ── rung 2a: pawn structure ──────────────────────────────────────────────────────────────────
	// Gated on PS_V2_MAG, so 0 = the rung is absent and this is byte-identical to the rung-1 result that
	// passed games at ~+101 Elo. ★ The detector is built here and RETURNS its masks; nothing about it is
	// published as a side effect, which is what will let the whole of Layer A+B move behind a pawnKey cache
	// at 2b without any downstream consumer reading stale state.
	// ★ ONE detector build serves both 2a and 2b -- the masks are a pure function of the two pawn
	// bitboards, so whichever rung is on, the entry is computed once and both scorers read it.
	int ps_mp = 0, pp_mp = 0;
	PawnEntry pe;
	if (Config::PS_V2_MAG != 0 || Config::PASSER_V2_MAG != 0){
		build_pawn_entry(pe, c);
		ps_mp = pawn_structure_mp(pe, c);   // returns 0 when PS_V2_MAG == 0
		pp_mp = passer_value_mp(pe, c);     // returns 0 when PASSER_V2_MAG == 0
		total += ps_mp + pp_mp;
	}

	// Later rungs accumulate here, each gated on its own knob.

	// ⚠️ Arm 2 excluded deliberately: in SHADOW the value search uses is v1's, and v1 has already published
	// its own breakdown for this position. Publishing here would leave a breakdown from one eval attached to
	// a score from the other -- the two would silently disagree and every term-attribution tool would be
	// reading a chimera.
	if (g_capture_eval_breakdown && Config::EVAL_ARM == 1){
		publish_rung0(total, c, w_mat, b_mat, w_pst, b_pst);
		if (Config::KS_V2_MAX > 0){
			// king_safety is signed Black-positive (NEGATE class in the symmetry gate); the unit counts are
			// per-side magnitudes (PLAIN-SWAP class). Publishing the units, not just the score, is the
			// detector/transformation split: "does it fire on the right kings" stays auditable apart from
			// "does firing produce the right ordering".
			g_eval_breakdown.king_safety    = ks_mp;
			g_eval_breakdown.det_ks_units_w = ks_w;
			g_eval_breakdown.det_ks_units_b = ks_b;
			g_eval_breakdown.terms_valid |= (1ULL << EB_KING_SAFETY)
			                              | (1ULL << EB_DET_KS_UNITS_W)
			                              | (1ULL << EB_DET_KS_UNITS_B);
		}
	}

	return total;
}
