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
	if (t != 0)                                        // pieces are flat for now; the endgame leg is
		return values[t + 1];                          // already the consensus endgame ratio
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
