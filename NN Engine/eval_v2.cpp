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
    ★ It returned (2026-09-29) as **POT (Potential) -- OvD reworked**: the owner's long-term-pressure concept, redesigned
    king-free as transformation potential (mg) + winning potential (eg winnability, `win_inputs` / `win_adjust`).
    Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §8, §11-13.

THE LADDER. Config::EVAL_V2_RUNG selects how far up to evaluate; each rung is read against the PREVIOUS
rung, which is a candidate-vs-candidate comparison and therefore null-independent -- the one comparison our
instruments resolve well (the SF11/SF15c gap read 0.08 on both corpora, first try).
  rung 0: material + piece-square tables.

*/

#include "eval_v2.h"
#include "pst_v2_fitted.h"
#include "ship_tables_v2.h"
#include "cpp_bitboard.h"
#include "move_gen.h"
#include "search_engine.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
	// Same census, unblended, for EVAL_V2_PAIR. ⚠️ PST is NOT tapered (one int per square), so its leg is
	// identical in both phases — which is exactly the gap a tapered PST would close (gap audit, L2/A3).
	int   mat_mg_white, mat_eg_white, mat_mg_black, mat_eg_black;
	int   npm_white, npm_black;          // non-pawn material, for phase and endgame gating
	int8_t cnt_white[6], cnt_black[6];   // per type, indexed 0=pawn .. 5=king (matches whitePlacementLayer)

	// Game phase, 256 = full midgame .. 0 = deep endgame. ⚠️ v2 owns this and it is the OPPOSITE
	// orientation to v1's phase_score (0 = opening, 128 = bare kings, consumed as a 3-way boolean).
	// Deliberate: v2 does not inherit a convention it does not use, and mixing the two silently is a
	// sign-error waiting to happen. Continuous, so no phase cliff exists to tune around.
	int   phase256;
};

/* ═══ (mg,eg) PAIR ACCUMULATOR — the structural prerequisite v2 has never had ═══════════════════════
 *
 * WHY (audited 2026-09-21, four-engine contrast): all 4 references accumulate a packed S(mg,eg) and
 * interpolate ONCE at the end. v2 blends inside every term and returns a scalar, so `total` is already
 * blended. That single absence is what blocks, all at once:
 *   - the ENDGAME SCALE FACTOR (4/4 universal; v2 can only say draw=0 or full value, never "drawish")
 *   - a TAPERED PST (4/4; our placement layer is one int per square, so no endgame king centralisation)
 *   - king safety's two-leg transform (3/4; ours is one curve for all phases)
 *   - any eg-LEG-ONLY term: SF's minPawnDist, and the winnability/complexity family the owner wants for
 *     a reworked OvD (Ethereal's evaluateComplexity is DEFINED on the eg leg and is king-free)
 *
 * ☠️☠️ INTERPOLATING ONCE CANNOT BE BYTE-IDENTICAL, and that is inherent, not a defect. Every `>> 8`
 * TRUNCATES, and the sum of N truncated blends != the truncation of one blended sum. v2 blends at THREE
 * different granularities -- per PIECE (v2_piece_value), per ROOK (rookfile_mp's loop), and per SIDE
 * (everything else) -- so a position carries ~40 truncations. Expect the two modes to differ by a few
 * tens of millipawns and NO MORE: that bound is the correctness test (see below), not a nuisance.
 *
 * ⇒ THE KNOB GATES *WHERE* THE BLEND HAPPENS, NOT A SECOND IMPLEMENTATION. Each scorer computes its
 * legs exactly once -- one source of truth -- and:
 *     Config::EVAL_V2_PAIR == 0  blend at the CURRENT site and granularity  => BYTE-IDENTICAL
 *     Config::EVAL_V2_PAIR == 1  return the legs, accumulate, blend ONCE at the end
 *
 * VERIFICATION LADDER (byte-identity alone cannot prove this one, so the bound does the work):
 *   1. mode 0 must reproduce the shipped fingerprint `250 / 49,440,513 / EBF 4.031` EXACTLY. That proves
 *      every leg was extracted correctly -- a mis-converted term shows up here immediately.
 *   2. mode 1: per position, |total_1 - total_0| < the blend-op count (~50 mp). A larger delta is a REAL
 *      BUG, not rounding. This is the decisive mechanical check; run it over thousands of positions.
 *   3. colour symmetry 0/800 in BOTH modes.
 *   4. per-term isolation: one term on at a time, confirm the bound per term, so a violation localises.
 *   5. only then measure mode 1 on the normal instruments -- it IS a behaviour change, however small.
 *
 * ⚠️ Mode 1 will EVENTUALLY become the shipped path (every payoff item above needs it), at which point the
 * fingerprint moves and gets re-recorded. That is expected, not a regression.
 * ⚠️ ks_danger_mp has no phase at all; in pair mode it contributes (x, x), preserving today's behaviour.
 * Giving king safety real phase legs is a SEPARATE later item, not part of this plumbing.
 */
struct EvalPair {
	int mg = 0;
	int eg = 0;
};

static inline EvalPair& operator+=(EvalPair &a, const EvalPair &b) { a.mg += b.mg; a.eg += b.eg; return a; }
static inline EvalPair& operator-=(EvalPair &a, const EvalPair &b) { a.mg -= b.mg; a.eg -= b.eg; return a; }
static inline EvalPair  operator-(const EvalPair &a, const EvalPair &b) { return EvalPair{a.mg - b.mg, a.eg - b.eg}; }

/* The ONE interpolation. phase256: 256 = full midgame, 0 = deep endgame (v2's convention).
   ☠️ NOT v1's `phase_score`, which is INVERTED and half-scaled -- aliasing them silently inverts every
   phase-conditioned reading.

   ☠️☠️ `/ 256`, NOT `>> 8` — AND THE COLOUR-SYMMETRY GATE IS WHAT FOUND THIS. An arithmetic shift rounds
   toward NEGATIVE INFINITY, so `(-x) >> 8 != -(x >> 8)` whenever there is a remainder. The per-term sites
   blend each SIDE separately (non-negative magnitudes) and difference afterwards, so the bias cancels
   there and `>> 8` is correct and must stay. Here we blend the already-DIFFERENCED, signed value, so the
   shift makes eval(mirror(b)) != -eval(b): first run of pair mode scored 474/800 violations, every one
   exactly 1 mp. Integer division truncates toward ZERO and is therefore sign-symmetric. */
static inline int eval_blend(const EvalPair &p, int phase256)
{
	return (p.mg * phase256 + p.eg * (256 - phase256)) / 256;
}

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

/* The same value, UNBLENDED, for EVAL_V2_PAIR. ☠️ Must mirror v2_piece_value exactly or mode 1 diverges
   from mode 0 by more than rounding — including the identity short-circuits, where mg == eg. */
static inline void v2_piece_value_legs(int t, int &pmg, int &peg)
{
	if (t != 0){
		peg = values[t + 1];
		pmg = (Config::EVAL_V2_PIECE_MG_PCT == 100) ? peg
		                                            : peg * Config::EVAL_V2_PIECE_MG_PCT / 100;
		return;
	}
	pmg = Config::EVAL_V2_PAWN_MG;
	peg = values[1];
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
	// ★ Note the granularity: v2_piece_value is called PER TYPE (6 blends), not per piece — so material
	// contributes 6 truncations to a position, not 32. The legs below are the same census unblended, for
	// EVAL_V2_PAIR; they cost two adds per type and are computed unconditionally so there is no second path.
	c.mat_white = c.mat_black = 0;
	c.mat_mg_white = c.mat_eg_white = c.mat_mg_black = c.mat_eg_black = 0;
	for (int t = 0; t < 6; ++t){
		const int v = v2_piece_value(t, c.phase256);
		c.mat_white += (int)c.cnt_white[t] * v;
		c.mat_black += (int)c.cnt_black[t] * v;
		int pmg, peg;
		v2_piece_value_legs(t, pmg, peg);
		c.mat_mg_white += (int)c.cnt_white[t] * pmg;  c.mat_eg_white += (int)c.cnt_white[t] * peg;
		c.mat_mg_black += (int)c.cnt_black[t] * pmg;  c.mat_eg_black += (int)c.cnt_black[t] * peg;
	}
}

// ===================================================================================================
// RUNG 0 -- MATERIAL + PIECE-SQUARE TABLES
// ===================================================================================================

// v2's own tapered piece-square tables (Config::PST_V2_TAPERED). White's point of view, [type][square] with
// square = rank*8 + file (a1 = 0); Black reads the rank-mirrored square (sq ^ 56). Written only by
// v2_pst_init() at engine init, read-only afterwards.
static int v2PstMg[6][64];
static int v2PstEg[6][64];

// Texel fit C1 (Config::C1_V2_FIT): fitted values the four C1 scorers read INSTEAD of their constants, in
// millipawns with every percent / magnitude knob already folded in. Filled once by v2_c1_init(); unread while
// g_c1_fit is false, which keeps the shipped path byte-identical. Layout mirrors v2_features (eval_v2.h).
static bool   g_c1_fit = false;
static int    c1_mob[2][4][28];                     // [leg][N, B, R, Q][move count]
static int    c1_ps_doubled[2], c1_ps_iso[2][8], c1_ps_back[2], c1_ps_wu[2];
static int    c1_pass[2][2][8];                     // [candidate][leg][relative rank]
static double c1_kd[2][2];                          // [candidate][them, us]: eg mp per unit of kdist x rank weight
static int    c1_place[2][9];                       // [leg][outpost N, outpost B, behind, bad bishop x4, trapped rook, weak queen]

// C3-a king shelter + pawn storm (Config::KSB_V2): the cell values, [leg][cell] in millipawns per count, as value to
// the king's OWN side. Cell layout: see ksb_cells. Filled once by v2_ksb_init(); unread while g_ksb_on is false, which
// keeps the shipped path byte-identical (the detector is not even run).
static constexpr int KSB_CELLS = 56;                // shelter 4 file classes x 6 states + storm 4 x 8
static bool   g_ksb_on = false;
static int    ksb_w[2][KSB_CELLS];
// C3-b pawnless flank + king-to-pawn distance (Config::KFL_V2) and C3-c KingProtector (Config::KPROT_V2): same
// contract as the shelter table -- [leg][cell], value to the king's / minor's OWN side, unread while off.
static constexpr int KFL_CELLS = 10;                // nearest own pawn d 2/3/4/>=5 · nearest enemy pawn d 2/3/4/>=5 · flank empty · flank only enemy
static constexpr int KPROT_CELLS = 12;              // knight d 1..6+ · bishop d 1..6+ (Chebyshev to our king)
static bool   g_kfl_on = false, g_kprot_on = false;
static int    kfl_w[2][KFL_CELLS], kprot_w[2][KPROT_CELLS];
// PX: the PASSER SYSTEM cells (2026-10-03, dev_notes/PASSER-SYSTEM-DESIGN-2026-10-03.md; layout at px_counts). Loaded
// from PX_V2_FILE under PX_V2=1 (c3_load_table contract: a missing/malformed file leaves the block OFF).
static constexpr int PX_CELLS = 51;
static int    px_w[2][PX_CELLS];
static bool   g_px_on = false;

static constexpr int V2_PST_VALUES = 6 * 2 * 64;

/*
	Build the tapered tables. Defaults copy the shared placement layer so the shipped eval is reproduced
	exactly: mg == eg == today's value for every piece, and mg = 0 for the king (the shipped
	PST_V2_KING_EG_ONLY=1, whose table is an endgame centralisation table). Then optionally load fitted values
	(PST_V2_FILE) and dump the active tables (PST_V2_DUMP). A malformed file is reported and ignored, leaving
	the defaults in place, so a bad path can never silently yield a half-loaded table.
	Called through the external v2_pst_init() defined after this anonymous namespace.
*/
void v2_pst_init_impl()
{
	for (int t = 0; t < 6; ++t)
		for (int sq = 0; sq < 64; ++sq){
			const int v = whitePlacementLayer[t][sq & 7][sq >> 3];
			v2PstEg[t][sq] = v;
			v2PstMg[t][sq] = (t == 5) ? 0 : v;
		}

	// Mode 2 = the shipped Texel-fitted tables compiled in (pst_v2_fitted.h), so the shipped eval never
	// depends on an external file. A PST_V2_FILE below still overrides either mode.
	if (Config::PST_V2_TAPERED == 2)
		for (int t = 0; t < 6; ++t)
			for (int sq = 0; sq < 64; ++sq){
				v2PstMg[t][sq] = V2_PST_FITTED[t][0][sq];
				v2PstEg[t][sq] = V2_PST_FITTED[t][1][sq];
			}

	if (const char *path = std::getenv("PST_V2_FILE"); path && *path){
		std::ifstream in(path);
		std::vector<int> vals;
		vals.reserve(V2_PST_VALUES);
		std::string tok;
		while (in >> tok){
			if (tok[0] == '#'){ std::string rest; std::getline(in, rest); continue; }
			try { vals.push_back(std::stoi(tok)); } catch (...) { vals.clear(); break; }
		}
		if (vals.size() != (size_t)V2_PST_VALUES){
			std::cerr << "☠️ PST_V2_FILE=" << path << " holds " << vals.size() << " values, expected "
			          << V2_PST_VALUES << " -- IGNORED, defaults kept." << std::endl;
		}
		else {
			size_t i = 0;
			for (int t = 0; t < 6; ++t){
				for (int sq = 0; sq < 64; ++sq) v2PstMg[t][sq] = vals[i++];
				for (int sq = 0; sq < 64; ++sq) v2PstEg[t][sq] = vals[i++];
			}
		}
	}

	if (const char *path = std::getenv("PST_V2_DUMP"); path && *path){
		std::ofstream out(path);
		static const char *names[6] = {"pawn", "knight", "bishop", "rook", "queen", "king"};
		out << "# v2 tapered PST, millipawns, White POV; per piece: 64 mg then 64 eg, rows = ranks 1..8, cols = a..h\n";
		for (int t = 0; t < 6; ++t)
			for (int leg = 0; leg < 2; ++leg){
				out << "# " << names[t] << (leg ? " eg" : " mg") << "\n";
				const int (*tab)[64] = leg ? v2PstEg : v2PstMg;
				for (int r = 0; r < 8; ++r){
					for (int f = 0; f < 8; ++f) out << tab[t][r * 8 + f] << (f < 7 ? " " : "\n");
				}
			}
	}
}

/*
	Rung 0's placement through the tapered tables. Each side's mg and eg sums are blended separately and then
	differenced -- the same per-side order as every other v2 term, so eval(mirror(b)) == -eval(b) holds even
	when fitted cells are negative (both sides round the same way). Division, not a shift, so a negative
	per-side sum truncates exactly as its mirror does.
*/
static inline void rung0_tapered_pst(const V2Context &c, int &w_pst, int &b_pst, int &w_mg, int &w_eg,
                                    int &b_mg, int &b_eg)
{
	const uint64_t typeMasks[6] = {c.pawns, c.knights, c.bishops, c.rooks, c.queens, c.kings};
	w_mg = w_eg = b_mg = b_eg = 0;
	if (!Config::PST_V2_ZERO){
		for (int t = 0; t < 6; ++t){
			uint64_t wb = typeMasks[t] & c.white;
			while (wb){
				const int sq = __builtin_ctzll(wb);
				wb &= wb - 1;
				w_mg += v2PstMg[t][sq];
				w_eg += v2PstEg[t][sq];
			}
			uint64_t bb = typeMasks[t] & c.black;
			while (bb){
				const int sq = __builtin_ctzll(bb) ^ 56;
				bb &= bb - 1;
				b_mg += v2PstMg[t][sq];
				b_eg += v2PstEg[t][sq];
			}
		}
	}
	const int eg = 256 - c.phase256;
	w_pst = (w_mg * c.phase256 + w_eg * eg) / 256;
	b_pst = (b_mg * c.phase256 + b_eg * eg) / 256;
}

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
                                        int &w_mat, int &b_mat, int &w_pst, int &b_pst,
                                        EvalPair *legs = nullptr)
{
	const uint64_t typeMasks[6] = {c.pawns, c.knights, c.bishops, c.rooks, c.queens, c.kings};

	w_mat = c.mat_white;                 // already censused exactly once; never recount
	b_mat = c.mat_black;
	w_pst = 0;
	b_pst = 0;

	if (Config::PST_V2_TAPERED){
		int w_mg, w_eg, b_mg, b_eg;
		rung0_tapered_pst(c, w_pst, b_pst, w_mg, w_eg, b_mg, b_eg);
		if (legs){
			legs->mg = (c.mat_mg_black + b_mg) - (c.mat_mg_white + w_mg);
			legs->eg = (c.mat_eg_black + b_eg) - (c.mat_eg_white + w_eg);
		}
		return (b_mat + b_pst) - (w_mat + w_pst);
	}

	// ── gap-audit A3, the KING half: the sixth placement table is an ENDGAME table ────────────────
	// ☠️ `whitePlacementLayerBase`'s last slot is commented "Kings - Endgame" and is a CENTRALISING table
	// (edge 0..5, centre 35). v1 reads it ONLY inside evaluate_kings_endgame (cpp_bitboard.cpp:4699);
	// evaluate_kings_midgame does not read it at all. This census loop reads t = 0..5 with NO phase gate,
	// so v2 pays a king-centralisation bonus in the OPENING AND MIDGAME, where v1 pays none and where the
	// king wants to be tucked away instead. It is a v2-only, wrong-SIGNED term.
	// ★ This is the cheap half of the tapered PST (A3) and it needs no (mg,eg) accumulator: the king's
	// contribution is simply blended eg-only at this site, exactly as every other v2 term blends at its own.
	// ⚠️ The table is SHARED with v1 and with move ordering (move_gen.h), so it must never be rewritten --
	// only how WE read it may change. Knob default 0 = today's behaviour = byte-identical.
	const bool king_eg_only = (Config::PST_V2_KING_EG_ONLY != 0);
	const int  n_types = king_eg_only ? 5 : 6;   // 5 => the king table is handled separately below
	for (int t = 0; t < n_types; ++t){
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

	// The king's placement, kept apart so it can carry a phase of its own. Non-negative per side, so the
	// per-side blend below may use >> 8 (the shift is only unsafe once a quantity is SIGNED).
	int w_kpst = 0, b_kpst = 0;
	if (king_eg_only){
		uint64_t wk = c.kings & c.white;
		while (wk){ const uint8_t sq = __builtin_ctzll(wk); wk &= wk - 1; w_kpst += whitePlacementLayer[5][sq & 7][sq >> 3]; }
		uint64_t bk = c.kings & c.black;
		while (bk){ const uint8_t sq = __builtin_ctzll(bk); bk &= bk - 1; b_kpst += blackPlacementLayer[5][sq & 7][sq >> 3]; }
		// ⚠️ Fold the BLENDED king back into the reported PST sums. `w_pst`/`b_pst` are published to
		// ev_breakdown, and consumers reconstruct total as (b_mat+b_pst)-(w_mat+w_pst); leaving the king
		// outside them would break that identity and every term-attribution tool would read a chimera.
		const int eg = 256 - c.phase256;
		w_pst += (w_kpst * eg) >> 8;
		b_pst += (b_kpst * eg) >> 8;
	}

	// Black-positive: White's holdings subtract, Black's add.
	if (legs){
		// PST is untapered, so it enters BOTH legs identically; only material differs by phase.
		// ★ EXCEPT the king, when king_eg_only is on: then it is an eg-leg-only term, which is what the
		// table always was. This is the one genuinely tapered PST entry v2 has.
		// ★ In PAIR mode the king takes its UNBLENDED value on the eg leg only -- the single interpolation
		// at the end then applies the phase. b_pst/w_pst above already carry the mode-0 blended king, so
		// subtract it back out here to avoid paying it twice.
		const int eg = 256 - c.phase256;
		const int w_folded = (w_kpst * eg) >> 8, b_folded = (b_kpst * eg) >> 8;
		legs->mg = (c.mat_mg_black + b_pst - b_folded) - (c.mat_mg_white + w_pst - w_folded);
		legs->eg = (c.mat_eg_black + b_pst - b_folded + b_kpst) - (c.mat_eg_white + w_pst - w_folded + w_kpst);
	}
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

/*
	SF11 MobilityBonus (evaluate.cpp:93-107), RAW SF units, indexed [KNIGHT..QUEEN][area-filtered count].
	Kept raw on purpose: they are summed raw per side and scaled ONCE at the end (mobility_mp), so the scale
	knob costs one multiply per eval and no rounding accumulates per piece. Unused tail entries are 0 and
	unreachable (max counts 8 / 13 / 14 / 27 on an empty board; x-ray cannot exceed them).
*/
/*
	MOB_V2_TABLE form bake-off (2026-09-15): the same [KNIGHT..QUEEN][count] layout for four references, indexed
	[table][type][count]. 0 SF11 (shipped; evaluate.cpp:93-107) · 1 SF15.1 (evaluate.cpp:213-227) · 2 Ethereal
	(master 0e47e9b, src/evaluate.c Knight/Bishop/Rook/QueenMobility) · 3 Weiss (master c735b8f, src/evaluate.c Mobility).
	Each engine's own units; mobility_mp rescales every table by ITS OWN knight mg range and pawn pair, so only the
	SHAPE competes. Ethereal/Weiss digits were fetched twice independently and agreed entry-for-entry.
*/
static constexpr int MOB_TAB_MG[4][4][28] = {
	{ // SF11
	{-62,-53,-12, -4,  3, 13, 22, 28, 33},
	{-48,-20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98},
	{-58,-27,-15,-10, -5, -2,  9, 16, 30, 29, 32, 38, 46, 48, 58},
	{-39,-21,  3,  3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66, 67, 70, 71, 73, 79, 88, 88, 99,102,102,106,109,113,116},
	},
	{ // SF15.1
	{-62,-53,-12, -3,  3, 12, 21, 28, 37},
	{-47,-20, 14, 29, 39, 53, 53, 60, 62, 69, 78, 83, 91, 96},
	{-60,-24,  0,  3,  4, 14, 20, 30, 41, 41, 41, 45, 57, 58, 67},
	{-29,-16, -8, -8, 18, 25, 23, 37, 41, 54, 65, 68, 69, 70, 70, 70, 71, 72, 74, 76, 90,104,105,106,112,114,114,119},
	},
	{ // Ethereal
	{-104,-45,-22, -8,  6, 11, 19, 30, 43},
	{ -99,-46,-16, -4,  6, 14, 17, 19, 19, 27, 26, 52, 55, 83},
	{-127,-56,-25,-12,-10,-12,-11, -4,  4,  9, 11, 19, 19, 37, 97},
	{-111,-253,-127,-46,-20,-9,-1, 2,  8, 10, 15, 17, 20, 23, 22, 21, 24, 16, 13, 18, 25, 38, 34, 28, 10,  7,-42,-23},
	},
	{ // Weiss
	{ -44,-31,-10,  0, 13, 22, 32, 43, 54},
	{ -51,-26,-11, -3,  9, 21, 26, 32, 32, 35, 41, 57, 50,100},
	{-105,-15, -1,  5,  2,  6,  5, 12, 14, 19, 24, 23, 25, 36, 72},
	{ -63,-97,-89,-17,  0, -8, -2, -2,  1,  5,  7,  9, 15, 15, 16, 18, 16, 16, 12, 13, 23, 22, 48, 58,122,135,146,125},
	},
};
static constexpr int MOB_TAB_EG[4][4][28] = {
	{ // SF11
	{-81,-56,-30,-14,  8, 15, 23, 27, 33},
	{-59,-23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97},
	{-76,-18, 28, 55, 69, 82,112,118,132,142,155,165,166,169,171},
	{-36,-15,  8, 18, 34, 54, 61, 73, 79, 92, 94,104,113,120,123,126,133,136,140,143,148,166,170,175,184,191,206,212},
	},
	{ // SF15.1
	{-79,-57,-31,-17,  7, 13, 16, 21, 26},
	{-59,-25, -8, 12, 21, 40, 56, 58, 65, 72, 78, 87, 88, 98},
	{-82,-15, 17, 43, 72,100,102,122,133,139,153,160,165,170,175},
	{-49,-29, -8, 17, 39, 54, 59, 73, 76, 95, 95,101,124,128,132,133,136,140,147,149,153,169,171,171,178,185,187,221},
	},
	{ // Ethereal
	{-139,-114,-37,  3, 15, 34, 38, 37, 17},
	{-186,-124,-54,-14,  1, 20, 35, 39, 49, 48, 48, 32, 47,  2},
	{-148,-127,-85,-28,  2, 27, 42, 46, 52, 55, 64, 68, 73, 60, 15},
	{-273,-401,-228,-236,-173,-86,-35,-1, 8, 31, 37, 55, 46, 57, 58, 64, 62, 65, 63, 48, 30,  8,-12,-29,-44,-79,-30,-50},
	},
	{ // Weiss
	{-139, -7, 60, 86, 89,102,102,101, 81},
	{ -81,-40, 19, 53, 65, 83, 98,104,114,116,115,106,115, 76},
	{-146, 18, 82, 88,121,133,144,146,152,157,164,171,177,177,154},
	{ -48,-54,-107,-127,-52,72,142,184,215,230,243,254,255,268,279,283,294,302,313,321,314,318,298,279,221,193,166,162},
	},
};
// Per-table scale: knight mg range (max - min) and the engine's own pawn mg / eg, all in that engine's units.
static constexpr int MOB_TAB_N_RANGE[4] = { 95,  99, 147,  98};
static constexpr int MOB_TAB_PAWN_MG[4] = {128, 126,  82, 104};
static constexpr int MOB_TAB_PAWN_EG[4] = {213, 208, 144, 204};

/*
	Optional mobility accumulator for build_side_attacks. `area` is an INPUT (squares that count); everything
	else is output. A null pointer means mobility is off and the attack build is exactly the rung-1 build.
*/
struct MobAcc {
	uint64_t area;
	// MOB_V2_PIN only (else pinned == 0): our king-blockers, restricted to their line through our king at ksq.
	uint64_t pinned;
	uint8_t  ksq;
	int      cnt[4];       // area-filtered squares summed per type, KNIGHT..QUEEN (the detector output)
	int      raw_mg;       // SF11 table sums, raw SF units
	int      raw_eg;
	// Per-ROOK area counts, kept so trapped-rook reads them instead of recomputing rook attacks and the area.
	// 10 is the legal maximum rooks per side (2 + 8 promotions).
	int      n_rooks;
	uint8_t  rook_sq[10];
	uint8_t  rook_mob[10];
	// MOB_V2_SAFE only: each N/B/R/Q attack mask and its type index, stored in the one attack pass so the safe-square
	// count can run once BOTH sides' attack maps exist. 15 is the legal maximum per side (7 + 8 promotions).
	int      n_pieces;
	uint64_t pmask[15];
	uint8_t  ptype[15];
};

/*
	Per-side attack maps, and optionally per-piece mobility, in ONE pass over the pieces.

	★ Every reference computes mobility in the same loop that fills its attack maps (SF11 pieces(), Ethereal,
	Weiss), because the attack mask per piece is the expensive part and both consumers want it. Doing it here
	rather than in a second loop is the whole point of reusing SideAttacks.

	@param sa     output attack maps; fully overwritten
	@param c      context
	@param white  which side's pieces
	@param mob    optional; when non-null, its cnt/raw_mg/raw_eg are ACCUMULATED (caller zeroes them)

	Gating: mob == nullptr leaves the loop identical to rung 1. Cost with mob: one AND + popcount + two table
	reads per non-pawn, non-king piece.
*/
inline void build_side_attacks(SideAttacks &sa, const V2Context &c, bool white, MobAcc *mob = nullptr,
                               uint64_t pin_restrict = 0, uint8_t pin_ksq = 0)
{
	// pin_restrict (KS_V2_PIN_DEF only): pieces of this side pinned to their king at pin_ksq attack only along the
	// pin line, as SF restricts its defender maps. 0 = the shared maps every other consumer reads, unchanged.
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
			uint64_t a = attacks_mask(white, occ_for, sq, pt);
			if (pin_restrict && ((pin_restrict >> sq) & 1) && pt != PAWN && pt != KING) a &= ray(pin_ksq, sq);
			sa.dbl |= sa.all & a;
			sa.all |= a;
			sa.by[pt] |= a;
			if (mob && pt >= KNIGHT && pt <= QUEEN){
				// MOB_V2_PIN (SF11:273-274): a pinned piece of ours counts only its pin line; a pinned knight's targets are
				// never on it. The KS maps above keep the full mask -- the knob changes mobility only.
				const uint64_t am = ((mob->pinned >> sq) & 1) ? (a & ray(mob->ksq, sq)) : a;
				const int n = __builtin_popcountll(am & mob->area);
				const int i = pt - KNIGHT;
				if (Config::MOB_V2_SAFE){
					// Deferred: mob_count_safe() needs the ENEMY's maps, which are not built yet.
					if (mob->n_pieces < 15){
						mob->pmask[mob->n_pieces] = am;
						mob->ptype[mob->n_pieces] = (uint8_t)i;
						++mob->n_pieces;
					}
				} else if (g_c1_fit){
					// Fitted cells are already millipawns: mobility_mp skips the SF-unit conversion.
					mob->cnt[i] += n;
					mob->raw_mg += c1_mob[0][i][n];
					mob->raw_eg += c1_mob[1][i][n];
				} else {
					mob->cnt[i] += n;
					mob->raw_mg += MOB_TAB_MG[Config::MOB_V2_TABLE][i][n];
					mob->raw_eg += MOB_TAB_EG[Config::MOB_V2_TABLE][i][n];
				}
				// ★ Trapped rook's contract is the PLAIN area count, whatever MOB_V2_SAFE does.
				if (pt == ROOK && mob->n_rooks < 10){
					mob->rook_sq[mob->n_rooks]  = sq;
					mob->rook_mob[mob->n_rooks] = (uint8_t)n;
					++mob->n_rooks;
				}
			}
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

/* Pieces of EITHER colour that are the single piece between `white`'s king and an enemy slider (SF blockers_for_king).
 * Snipers are enemy R/Q on the king's orthogonals and B/Q on its diagonals on an EMPTY board; occupancy with the snipers
 * removed; exactly one piece between => blocker. The one definition both MOB_V2_PIN (mob_king_blockers) and KS read. */
static inline uint64_t v2_king_blockers(const V2Context &c, bool white) noexcept
{
	const uint64_t own  = white ? c.white : c.black;
	const uint64_t them = white ? c.black : c.white;
	const uint8_t  ksq  = (uint8_t)__builtin_ctzll(c.kings & own);
	const uint64_t snipers = ((attacks_mask(white, 0, ksq, ROOK)   & (c.rooks   | c.queens))
	                        | (attacks_mask(white, 0, ksq, BISHOP) & (c.bishops | c.queens))) & them;
	const uint64_t occ = c.occupied ^ snipers;
	uint64_t blockers = 0, sn = snipers;
	while (sn){
		const uint8_t r = (uint8_t)__builtin_ctzll(sn); sn &= sn - 1;
		const uint64_t b = betweenPieces(ksq, r) & occ;
		if (b && !(b & (b - 1))) blockers |= b;
	}
	return blockers;
}

/*
	Every king-safety CHANNEL for one king, computed ONCE and shared by the scorer (ks_units) and the probe (ks_probe),
	so the two can never drift apart (they used to duplicate this logic line by line).

	The rung-1 channels (attacker count and weight, weak squares, adjacent squares, safe checks, enemy queen) are computed
	exactly as before, so the shipped path is byte-identical. The 2026-09-27 BALANCE channels are computed only when
	`full` is set -- the probe always sets it; the scorer sets it only when one of their knobs is live:
	  att_xray     attacker count / weight with the x-ray occupancy the shared maps already use (KS_V2_ATT_XRAY)
	  adj_inst     attack INSTANCES on the king ring: per counted attacker, how many ring squares it hits -- convergence,
	               where adj_sq measures breadth (SF counts instances)
	  unsafe       check squares of R/B/N that exist but are not safe (SF unsafeChecks)
	  blockers     pieces pinned or blocking against this king (SF blockersForKing)
	  flank_att/def  attacks / defence on the king's flank inside its camp (SF kingFlankAttacks / Defense)
	  knight_def   one of our knights guards the king ring (SF)
	  contest_*    OUR per-square balance: on each zone square, enemy attack count vs own defence count (pawn defenders
	               counted twice, king excluded); excess = sum of positive margins, sq = squares with >= 2 attackers
	               that outnumber the defence
	  w_att_contest  defaware-v2: each attacker's weight scaled by the share of its zone footprint that is contested
	  gate         the universal attacker gate: >= 2 attackers, or 1 with an enemy queen (SF1.1, Ethereal, Weiss)
*/
struct KsChannels {
	int  n_att, w_att, weak, adj_sq, n_att_x, w_att_x, adj_inst, unsafe, blockers, flank_att, flank_def, knight_def,
	     contest_excess, contest_sq, w_att_contest;
	// Per attacker TYPE (N, B, R, Q), added 2026-09-28 for Fit K2 so the fit can price each type's weight: attackers
	// of the zone (plain occupancy), the same with x-ray occupancy, and the sum over attackers of 256 x the contested
	// share of their zone footprint (DEFAWARE's per-attacker factor before the weight is applied).
	int  att_t[4], att_x_t[4], share_t[4];
	uint64_t chk_r, chk_q, chk_b, chk_n;       // safe-check squares per type (masks)
	bool enemy_queen, gate;
};

static constexpr int KS_W_OURS[7]   = {0, 0, 31, 31, 47, 78, 0};   // -, P, N, B, R, Q, K  (queen-high, ours)
static constexpr int KS_W_KNIGHT[7] = {0, 0, 81, 52, 44, 10, 0};   // knight-high (SF11)
// The LIVE "ours" profile: KS_W_OURS unless KS_V2_W_N/B/R/Q override it (Fit K2 prices each type). Statically the
// shipped values, so a probe called before any engine init reads the real weights, never zeros; v2_c3_init rewrites
// it from the knobs.
static int KS_W_LIVE[7] = {0, 0, 31, 31, 47, 78, 0};
// SF KingFlank: the king's file and its neighbours, 3 files at the edge, 4 inside.
static constexpr uint64_t KS_FILE_BB(int f) { return 0x0101010101010101ULL << f; }

inline void ks_channels(KsChannels &ch, const V2Context &c, const SideAttacks &wa, const SideAttacks &ba,
                        bool white_king, bool full)
{
	ch = KsChannels{};
	const uint64_t kbb = c.kings & (white_king ? c.white : c.black);
	if (!kbb) return;
	const uint8_t ksq = __builtin_ctzll(kbb);
	const SideAttacks &att = white_king ? ba : wa;   // the ATTACKING side
	const SideAttacks &def0 = white_king ? wa : ba;  // the king's OWN side
	const uint64_t own   = white_king ? c.white : c.black;
	const uint64_t enemy = white_king ? c.black : c.white;
	const uint64_t zone  = ks_zone(ksq, white_king, c.pawns & own);
	ch.enemy_queen = (c.queens & enemy) != 0;

	// KS_V2_PIN_DEF: our pinned pieces defend only along their pin line (their own maps, never the shared ones).
	SideAttacks defp;
	const SideAttacks *defs = &def0;
	uint64_t blk = 0;
	if (full || Config::KS_V2_BLOCKERS || Config::KS_V2_PIN_DEF) blk = v2_king_blockers(c, white_king);
	if (Config::KS_V2_PIN_DEF){
		build_side_attacks(defp, c, white_king, nullptr, blk & own, ksq);
		defs = &defp;
	}
	const SideAttacks &def = *defs;
	ch.blockers = __builtin_popcountll(blk);

	const int *W = Config::KS_V2_ATT_PROFILE ? KS_W_KNIGHT : KS_W_LIVE;

	// Contest planes (only when needed): per zone square, how many enemy men attack it and how many of ours defend it.
	const bool want_contest = full || Config::KS_V2_DEFAWARE || Config::KS_V2_CONTEST_EXCESS
	                       || Config::KS_V2_CONTEST_SQ || Config::KS_V2_CONTEST_SQ_Q;
	uint64_t contested = 0;
	if (want_contest){
		int na[64] = {0}, nd[64] = {0};
		const uint64_t typeAll[5] = {c.pawns, c.knights, c.bishops, c.rooks, c.queens};
		for (int side = 0; side < 2; ++side){
			const bool     w     = side == 0;
			const uint64_t men   = w ? c.white : c.black;
			const bool     is_att = (men == enemy);
			for (int t = 0; t < 5; ++t){
				const uint8_t pt = (uint8_t)(t + 1);
				uint64_t bb = typeAll[t] & men;
				while (bb){
					const uint8_t sq = (uint8_t)__builtin_ctzll(bb);
					bb &= bb - 1;
					uint64_t occ_for = c.occupied;
					if (Config::KS_V2_XRAY){
						if (pt == BISHOP)    occ_for = c.occupied ^ c.queens;
						else if (pt == ROOK) occ_for = c.occupied ^ c.queens ^ (c.rooks & men);
					}
					uint64_t a = attacks_mask(w, occ_for, sq, pt) & zone;
					if (!is_att && Config::KS_V2_PIN_DEF && ((blk & own) >> sq & 1) && pt != PAWN) a &= ray(ksq, sq);
					const int mult = (!is_att && pt == PAWN) ? 2 : 1;   // a pawn defender is the strongest guard
					while (a){
						const int z = __builtin_ctzll(a); a &= a - 1;
						if (is_att) ++na[z]; else nd[z] += mult;
					}
				}
			}
		}
		uint64_t zz = zone;
		while (zz){
			const int z = __builtin_ctzll(zz); zz &= zz - 1;
			if (na[z] > nd[z]){
				contested |= 1ULL << z;
				ch.contest_excess += na[z] - nd[z];
				if (na[z] >= 2) ++ch.contest_sq;
			}
		}
	}

	// ── attacker count and weight: plain occupancy (shipped), x-ray (KS_V2_ATT_XRAY), contest-scaled (DEFAWARE) ─────
	const uint64_t typeMasks[4] = {c.knights, c.bishops, c.rooks, c.queens};
	const uint64_t ring = BB_KING_ATTACKS[ksq];
	long long w_contest256 = 0;
	for (int i = 0; i < 4; ++i){
		const uint8_t pt = (uint8_t)(i + 2);              // KNIGHT..QUEEN
		uint64_t bb = typeMasks[i] & enemy;
		while (bb){
			const uint8_t sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			const uint64_t a = attacks_mask(!white_king, c.occupied, sq, pt);
			if (a & zone){
				++ch.n_att;
				ch.w_att += W[pt];
				++ch.att_t[i];
			}
			if (full || Config::KS_V2_ATT_XRAY || Config::KS_V2_ADJ_INST || Config::KS_V2_DEFAWARE){
				uint64_t ax = a;
				if (Config::KS_V2_XRAY){
					if (pt == BISHOP)    ax = attacks_mask(!white_king, c.occupied ^ c.queens, sq, pt);
					else if (pt == ROOK) ax = attacks_mask(!white_king, c.occupied ^ c.queens ^ (c.rooks & enemy), sq, pt);
				}
				const uint64_t cnt_mask = Config::KS_V2_ATT_XRAY ? ax : a;
				if (ax & zone){ ++ch.n_att_x; ch.w_att_x += W[pt]; ++ch.att_x_t[i]; }
				if (cnt_mask & zone){
					ch.adj_inst += __builtin_popcountll(cnt_mask & ring);
					const int foot = __builtin_popcountll(cnt_mask & zone);
					const int hit  = __builtin_popcountll(cnt_mask & zone & contested);
					w_contest256 += (long long)W[pt] * 256 * hit / foot;
					ch.share_t[i] += 256 * hit / foot;
				}
			}
		}
	}
	ch.w_att_contest = (int)(w_contest256 / 256);

	// ★ SF seeds kingAttackersCount with enemy PAWN attacks on the ring: a pawn bearing on the zone counts toward
	// COORDINATION even though its weight is zero.
	if (Config::KS_V2_PAWN_ATT){
		const uint64_t ep = c.pawns & enemy;
		uint64_t l, rr;
		if (white_king){ l = (ep & ~BB_FILE_A) >> 9; rr = (ep & ~BB_FILE_H) >> 7; }
		else           { l = (ep & ~BB_FILE_A) << 7; rr = (ep & ~BB_FILE_H) << 9; }
		const int seed = __builtin_popcountll(zone & (l | rr)) > 0 ? 1 : 0;
		ch.n_att += seed;
		ch.n_att_x += seed;
	}
	ch.gate = (ch.n_att >= 2) || (ch.n_att >= 1 && ch.enemy_queen);

	// ── weak squares: SF's definition (attacked by them, not defended twice by us, undefended or only by K/Q) ──────
	const uint64_t weak = att.all & ~def.dbl & (~def.all | def.by[KING] | def.by[QUEEN]);
	ch.weak   = __builtin_popcountll(zone & weak);
	ch.adj_sq = __builtin_popcountll(att.all & ring);

	// ── checks: a check square is SAFE if we do not defend it, or if it is weak and they attack it twice. Sliding
	// checks are traced THROUGH our own queen (SF does the same).
	const uint64_t occ_x      = c.occupied ^ (c.queens & own);
	const uint64_t rookRays   = attacks_mask(white_king, occ_x, ksq, ROOK);
	const uint64_t bishopRays = attacks_mask(white_king, occ_x, ksq, BISHOP);
	const uint64_t knightRays = attacks_mask(white_king, c.occupied, ksq, KNIGHT);
	const uint64_t safe       = ~enemy & (~def.all | (weak & att.dbl));
	ch.chk_r = rookRays & safe & att.by[ROOK];
	ch.chk_q = (rookRays | bishopRays) & safe & att.by[QUEEN];
	ch.chk_b = bishopRays & safe & att.by[BISHOP];
	ch.chk_n = knightRays & safe & att.by[KNIGHT];
	if (full || Config::KS_V2_UNSAFE){
		// SF unsafeChecks: a type with NO safe check contributes the check squares it does have. The queen is excluded.
		uint64_t unsafe = 0;
		if (!ch.chk_r) unsafe |= rookRays & att.by[ROOK] & ~enemy;
		if (!ch.chk_b) unsafe |= bishopRays & att.by[BISHOP] & ~enemy;
		if (!ch.chk_n) unsafe |= knightRays & att.by[KNIGHT] & ~enemy;
		ch.unsafe = __builtin_popcountll(unsafe);
	}
	if (full || Config::KS_V2_FLANK_ATT || Config::KS_V2_FLANK_ATT2 || Config::KS_V2_FLANK_DEF || Config::KS_V2_KNIGHT_DEF){
		const int f = ksq & 7;
		const int lo = f == 0 ? 0 : (f <= 2 ? 0 : (f <= 4 ? 2 : (f <= 6 ? 4 : 5)));
		const int hi = f == 7 ? 7 : (f >= 5 ? 7 : (f >= 3 ? 5 : (f >= 1 ? 3 : 2)));
		uint64_t flank = 0;
		for (int x = lo; x <= hi; ++x) flank |= KS_FILE_BB(x);
		const uint64_t camp = white_king ? 0x000000FFFFFFFFFFULL : 0xFFFFFFFFFF000000ULL;   // our 5 ranks
		ch.flank_att  = __builtin_popcountll(att.all & flank & camp) + __builtin_popcountll(att.dbl & flank & camp);
		ch.flank_def  = __builtin_popcountll(def.all & flank & camp);
		ch.knight_def = (def.by[KNIGHT] & ring) ? 1 : 0;
	}
}

/*
	Attack units borne against ONE king. Returns SF-scale units (0 .. ~2000), NOT millipawns.

	`white_king` selects whose king is examined; the ATTACKER is the other side. See
	dev_notes/EVAL-V2-RUNG1-KS-DESIGN.md for the full 14-item audit of SF's kingDanger, and
	dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md for the 2026-09-27 balance channels (every one 0 by default).
*/
inline int ks_units(const V2Context &c, const SideAttacks &wa, const SideAttacks &ba, bool white_king)
{
	const bool full = Config::KS_V2_ATT_XRAY || Config::KS_V2_ADJ_INST || Config::KS_V2_UNSAFE || Config::KS_V2_BLOCKERS
	               || Config::KS_V2_FLANK_ATT || Config::KS_V2_FLANK_ATT2 || Config::KS_V2_FLANK_DEF
	               || Config::KS_V2_KNIGHT_DEF || Config::KS_V2_CONTEST_EXCESS || Config::KS_V2_CONTEST_SQ
	               || Config::KS_V2_CONTEST_SQ_Q || Config::KS_V2_DEFAWARE || Config::KS_V2_GATE || Config::KS_V2_PIN_DEF;
	KsChannels ch;
	ks_channels(ch, c, wa, ba, white_king, full);
	if (!(c.kings & (white_king ? c.white : c.black))) return 0;
	if (Config::KS_V2_GATE && !ch.gate) return 0;

	// ★ COORDINATION as a CONTINUOUS parameter: COORD_MUL[n] = 256 + (n-1) * KS_V2_COORD (0 = sum, 256 = product).
	const int n_att = Config::KS_V2_ATT_XRAY ? ch.n_att_x : ch.n_att;
	const int w_att = Config::KS_V2_DEFAWARE ? ch.w_att_contest : (Config::KS_V2_ATT_XRAY ? ch.w_att_x : ch.w_att);
	int u = 0;
	if (n_att > 0){
		const long long mul = 256LL + (long long)(n_att - 1) * (long long)Config::KS_V2_COORD;
		u = (int)(((long long)w_att * mul) >> 8);
	}
	u += Config::KS_V2_WEAK * ch.weak;
	u += Config::KS_V2_ADJ * ch.adj_sq;

	// ⚠️ FORM is a knob because it is COUPLED to the magnitudes: SF11 fires once, Ethereal adds per square.
	int chk_q = Config::KS_V2_CHK_Q, chk_r = Config::KS_V2_CHK_R;
	int chk_b = Config::KS_V2_CHK_B, chk_n = Config::KS_V2_CHK_N;
	if (Config::KS_V2_CHK_PROFILE == 1){ chk_q = 1046; chk_r = 1046; chk_b = 523; chk_n = 672; }
	if (Config::KS_V2_CHK_COUNT){
		u += chk_r * __builtin_popcountll(ch.chk_r);
		u += chk_q * __builtin_popcountll(ch.chk_q);
		u += chk_b * __builtin_popcountll(ch.chk_b);
		u += chk_n * __builtin_popcountll(ch.chk_n);
	} else {
		if (ch.chk_r) u += chk_r;
		if (ch.chk_q) u += chk_q;
		if (ch.chk_b) u += chk_b;
		if (ch.chk_n) u += chk_n;
	}

	// ★ The largest single term in SF's sum (-873), and correct to real chess: without a queen an attack usually
	// cannot be converted.
	if (!ch.enemy_queen) u -= Config::KS_V2_NO_QUEEN;

	// ── the 2026-09-27 balance channels: every weight 0 by default, so the shipped units are unchanged ───────────
	if (full){
		u += Config::KS_V2_ADJ_INST * ch.adj_inst;
		u += Config::KS_V2_UNSAFE * ch.unsafe;
		u += Config::KS_V2_BLOCKERS * ch.blockers;
		u += Config::KS_V2_FLANK_ATT * ch.flank_att;
		u += Config::KS_V2_FLANK_ATT2 * ch.flank_att * ch.flank_att / 8;
		u -= Config::KS_V2_FLANK_DEF * ch.flank_def;
		u -= Config::KS_V2_KNIGHT_DEF * ch.knight_def;
		u += Config::KS_V2_CONTEST_EXCESS * ch.contest_excess;
		u += Config::KS_V2_CONTEST_SQ * ch.contest_sq;
		if (ch.enemy_queen) u += Config::KS_V2_CONTEST_SQ_Q * ch.contest_sq;
	}

	// ★ ONSET: quiet positions must contribute EXACTLY zero, not a small positive (SF `kingDanger > 100`, Ethereal
	// `SafetyAdjustment` + `MAX(0,...)`).
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
			if (!st){
				// ── gap-audit P4: REAR-DOUBLED PASSER OVER-CREDIT ────────────────────────────────
				// ☠️ A DEFECT, not a coverage gap. `passed` is flagged on a clear ENEMY span alone --
				// nothing asks whether one of OUR OWN pawns is ahead on the same file. Two stacked own
				// pawns on a clear file are therefore BOTH flagged passed and BOTH paid in full by
				// passer_value_mp, which is one passer counted twice: only the front pawn can actually
				// promote, and the rear one is a liability the front one blocks.
				// ★ Ethereal has exactly this as an anti-double-count, not a bonus:
				//     if (several(forwardFileMasks(US, sq) & myPassers)) continue;
				// ⚠️ `ps_nfill`/`ps_sfill` shift by whole ranks, so they stay on the pawn's own file --
				// no file mask is needed, and this is the same quantity the CANDIDATE branch below
				// already computes as `rear` and already excludes on. The bug is that the true-passer
				// branch never got the same test.
				// Modes: 0 = today's behaviour (byte-identical) · 1 = demote the rear pawn to CANDIDATE,
				// so it is priced through the existing PASSER_V2_CAND_PCT path · 2 = Ethereal, drop it.
				if (Config::PS_V2_REAR_DOUBLED){
					const uint64_t own_ahead = (white ? ps_nfill(m << 8) : ps_sfill(m >> 8)) & own;
					if (own_ahead){
						if (Config::PS_V2_REAR_DOUBLED == 1) cb |= m;   // demote to candidate
						continue;                                       // mode 2: no credit at all
					}
				}
				pb |= m; continue;
			}

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
static inline int pawn_structure_mp(const PawnEntry &e, const V2Context &c, EvalPair *legs = nullptr)
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
		const int nb = __builtin_popcountll(e.backward[s]);
		if (g_c1_fit){
			// Texel C1: fitted, signed values with PS_V2_MAG folded in (so the final scale below is skipped).
			side_mg[s] += nd * c1_ps_doubled[0] + nb * c1_ps_back[0];
			side_eg[s] += nd * c1_ps_doubled[1] + nb * c1_ps_back[1];
			uint64_t ibf = e.isolated[s];
			while (ibf){
				const int f = (int)(__builtin_ctzll(ibf) & 7);
				ibf &= ibf - 1;
				side_mg[s] += c1_ps_iso[0][f];
				side_eg[s] += c1_ps_iso[1][f];
			}
			const int nwu = __builtin_popcountll((e.isolated[s] | e.backward[s]) & ~e.opposed[s]);
			side_mg[s] += nwu * c1_ps_wu[0];
			side_eg[s] += nwu * c1_ps_wu[1];
			continue;
		}
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
		side_mg[s] -= nb * Config::PS_V2_BACKWARD_MG;
		side_eg[s] -= nb * Config::PS_V2_BACKWARD_EG;

		// --- weak-unopposed: an isolated-or-backward pawn on a HALF-OPEN file (SF WeakUnopposed) -------
		// ☠️ Deliberately an EXTRA charge on pawns the two legs above have ALREADY penalised, not a
		// separate term: the firing set is 100% contained in `isolated | backward`
		// (_pawn_term_overlap.py, 2026-09-20), so a second owner would be two names for one signal.
		// SF stacks it the same way -- it pays WeakUnopposed ON TOP of its own isolated/backward charges,
		// and on top of the passer bonus when the pawn is both.
		// ⚠️ `opposed` is "an enemy pawn anywhere AHEAD on our own file", so ~opposed is exactly "no enemy
		// pawn will ever trade this one off, and a rook can sit in front of it" -- the mechanism the four
		// references agree on. Gated: both knobs 0 => this block contributes nothing.
		if (Config::PS_V2_WEAKUNOPP_MG != 0 || Config::PS_V2_WEAKUNOPP_EG != 0){
			const int nwu = __builtin_popcountll((e.isolated[s] | e.backward[s]) & ~e.opposed[s]);
			side_mg[s] -= nwu * Config::PS_V2_WEAKUNOPP_MG;
			side_eg[s] -= nwu * Config::PS_V2_WEAKUNOPP_EG;
		}
	}

	// Phase-blend each side, then combine Black-positive. c.phase256: 256 = full midgame, 0 = deep endgame.
	// `legs` non-null ⇒ PAIR MODE: difference and scale the legs WITHOUT blending, and let the caller do
	// the one interpolation. ⚠️ Mode 0 blends per side and THEN scales by MAG/100; mode 1 scales the legs
	// and blends after, so the two truncate at different points. Bounded, expected.
	if (legs){
		legs->mg = ((side_mg[1] - side_mg[0]) * Config::PS_V2_MAG) / 100;
		legs->eg = ((side_eg[1] - side_eg[0]) * Config::PS_V2_MAG) / 100;
	}
	const int w = (side_mg[0] * c.phase256 + side_eg[0] * (256 - c.phase256)) >> 8;
	const int b = (side_mg[1] * c.phase256 + side_eg[1] * (256 - c.phase256)) >> 8;
	if (g_c1_fit) return b - w;
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

	// PROVENANCE, not an eval term: every corpus/analysis consumer needs the phase, and v2 published none,
	// which crashed `tune_corpus.py` with KeyError('phase_score') -- the THIRD v1-era tool to assume v1's
	// partition. ☠️ Published under its OWN name because v1's `phase_score` is the INVERSE on HALF the
	// scale (v1: 0 = opening, 128 = endgame; v2: 256 = opening, 0 = endgame).
	b.v2_phase256 = c.phase256;

	b.arm = Config::EVAL_ARM;
	b.terms_valid = (1ULL << EB_TOTAL)           | (1ULL << EB_MATERIAL)
	              | (1ULL << EB_PIECES)          | (1ULL << EB_IMBALANCE_WHITE)
	              | (1ULL << EB_IMBALANCE_BLACK) | (1ULL << EB_DET_W_PIECEVAL)
	              | (1ULL << EB_DET_B_PIECEVAL)  | (1ULL << EB_DET_PAWN_COUNT)
	              | (1ULL << EB_V2_PHASE256);
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

// Called from engine init (search_engine.cpp), so it needs external linkage -- see eval_v2.h.
void v2_pst_init()
{
	v2_pst_init_impl();
}

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

/* SF11's PATH-SAFETY LADDER (evaluate.cpp:626-635), in SF's OWN score units, before conversion.
 * Indexed by how far the enemy's attacks reach up the pawn's forward span:
 *   0 span entirely unattacked · 1 span attacked but the queening file clear · 2 file attacked but the
 *   stop square clear · 3 the stop square itself is attacked.
 * ★ These are the rungs `w = 5r - 13` was always meant to multiply. v2 shipped w without them.
 */
static constexpr int PS_PATH_K[4]       = {35, 20, 9, 0};
static constexpr int PS_PATH_K_DEFENDED = 5;    // stop square defended, or our own R/Q behind the pawn

/* SF11 score units -> millipawns. ★ This is the SAME conversion PS_PASSED_* above already used, verified
 * entry-for-entry against SF11's PassedRank: mg = SF.mg * 1000 / 128 reproduces {78,133,117,484,1312,2156}
 * and eg = SF.eg * 1000 / 213 reproduces {131,155,192,338,830,1219}. So the ladder needs no new convention
 * -- `k*w` is ONE SF number added to both legs, and each leg converts by its own phase's SF pawn.
 * ⚠️ This is the by-the-PAWN conversion, which is correct HERE because the rank table it extends used it;
 * it is NOT the positional-scale rule that governs terms with no such anchor.
 */
static constexpr int PS_SF_PAWN_MG = 128;
static constexpr int PS_SF_PAWN_EG = 213;

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
/* @param wa,ba  the shared attack maps, or nullptr when the attack build did not run. Only the
 *               PATH-SAFETY LADDER reads them; every other rung here is pawn- and king-only, so passing
 *               nullptr reproduces the pre-ladder scorer exactly.
 * ☠️ Taking piece attacks here does NOT cost the pawn cache: the cacheable block is the DETECTOR
 * (build_pawn_entry), not this scorer, and this scorer already read c.kings for king distance.
 */
/* SF11's PATH-SAFETY LADDER for one passer (evaluate.cpp:626-635), in SF units × the rank weight w × PASSER_V2_PATH_PCT.
 * Shared by the constant and the C1 paths (2026-10-09: it used to live inside the constant path only, so ANY C1 table made
 * PASSER_V2_PATH_PCT silently dead — caught when a "+path" stack read byte-identical to the stack without it).
 * Gated on an EMPTY stop square exactly as SF is. Returns 0 when the stop square is off-board or occupied.
 * ⚠️ KNOWN DEVIATION FROM SF, deliberate: SF intersects with a PLAIN attack set; ours is the SHARED map (x-ray under
 * KS_V2_XRAY), so `unsafe` is strictly larger and k is, if anything, one rung more pessimistic.
 * ☠️ SF's asymmetry (the enemy half of rook-behind-passer): with an enemy R/Q behind the pawn the span stays MAXIMALLY unsafe. */
static inline int passer_path_kw(const V2Context &c, const SideAttacks &wa, const SideAttacks &ba, bool white, int sq,
                                 int stop, int w) noexcept
{
	if (stop < 0 || stop >= 64 || (c.occupied & (1ULL << stop))) return 0;
	const uint64_t spanf   = white ? passed_span_white[sq] : passed_span_black[sq];
	const uint64_t toQueen = spanf & BB_FILES[sq & 7];
	// Squares BEHIND the pawn on its own file = the OTHER colour's forward span, same file (SF's forward_file_bb(Them, s)).
	const uint64_t behind  = (white ? passed_span_black[sq] : passed_span_white[sq]) & BB_FILES[sq & 7];
	const uint64_t rq      = (c.rooks | c.queens) & behind;
	const uint64_t theirs  = white ? c.black : c.white;
	const uint64_t ours    = white ? c.white : c.black;
	const uint64_t their_att = white ? ba.all : wa.all;
	const uint64_t our_att   = white ? wa.all : ba.all;
	uint64_t unsafe_sq = spanf;
	if (!(theirs & rq)) unsafe_sq &= their_att;
	int k = !unsafe_sq                        ? PS_PATH_K[0]
	      : !(unsafe_sq & toQueen)            ? PS_PATH_K[1]
	      : !(unsafe_sq & (1ULL << stop))     ? PS_PATH_K[2]
	      :                                     PS_PATH_K[3];
	// Our own R/Q behind, or a defended stop square -- SF's other half of P3.
	if ((ours & rq) || (our_att & (1ULL << stop))) k += PS_PATH_K_DEFENDED;
	return k * w * Config::PASSER_V2_PATH_PCT / 100;
}

static inline int passer_value_mp(const PawnEntry &e, const V2Context &c,
                                  const SideAttacks *wa, const SideAttacks *ba, EvalPair *legs = nullptr)
{
	if (Config::PASSER_V2_MAG == 0) return 0;
	// The ladder needs both maps; with the shared attack build off it is absent, not half-applied.
	const bool path_on = (Config::PASSER_V2_PATH_PCT != 0) && wa && ba;

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

			if (g_c1_fit){
				// Texel C1: fitted rank values per candidate class, with the percent knobs and PASSER_V2_MAG folded
				// in; the king-distance terms stay eg-only, as in the constant path.
				const int ci = (m & e.candidate[s]) ? 1 : 0;
				int fmg = c1_pass[ci][0][r], feg = c1_pass[ci][1][r];
				if (r >= Config::PASSER_V2_MIN_RANK){
					int w = 5 * r - 13;
					if (w < 0) w = 0;
					const int stop = white ? (sq + 8) : (sq - 8);
					if (stop >= 0 && stop < 64)
						feg += (int)std::lround((ps_kdist(theirK, stop) * c1_kd[ci][0]
						                       + ps_kdist(ourK, stop) * c1_kd[ci][1]) * w);
					// The path ladder is NOT a C1 column, so it is not folded into the fitted values: apply it with the
					// same candidate / phase-percent / magnitude scaling the constant path gives it.
					if (path_on){
						const int kw = passer_path_kw(c, *wa, *ba, white, sq, stop, w);
						if (kw){
							long long lm = (long long)kw * 1000 / PS_SF_PAWN_MG, le = (long long)kw * 1000 / PS_SF_PAWN_EG;
							if (ci){ lm = lm * Config::PASSER_V2_CAND_PCT / 100; le = le * Config::PASSER_V2_CAND_PCT / 100; }
							fmg += (int)(lm * Config::PASSER_V2_MG_PCT / 100 * Config::PASSER_V2_MAG / 100);
							feg += (int)(le * Config::PASSER_V2_EG_PCT / 100 * Config::PASSER_V2_MAG / 100);
						}
					}
				}
				side_mg[s] += fmg;
				side_eg[s] += feg;
				continue;
			}

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

				// ── gap-audit P1+P2: SF's PATH-SAFETY LADDER (evaluate.cpp:626-635) — see passer_path_kw ──
				if (path_on){
					const int kw = passer_path_kw(c, *wa, *ba, white, sq, stop, w);
					mg += kw * 1000 / PS_SF_PAWN_MG;
					eg += kw * 1000 / PS_SF_PAWN_EG;
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

	if (legs){
		legs->mg = ((side_mg[1] - side_mg[0]) * Config::PASSER_V2_MAG) / 100;
		legs->eg = ((side_eg[1] - side_eg[0]) * Config::PASSER_V2_MAG) / 100;
	}
	const int w = (side_mg[0] * c.phase256 + side_eg[0] * (256 - c.phase256)) >> 8;
	const int b = (side_mg[1] * c.phase256 + side_eg[1] * (256 - c.phase256)) >> 8;
	if (g_c1_fit) return b - w;
	return ((b - w) * Config::PASSER_V2_MAG) / 100;
}

// ===================================================================================================
// SLICE 2 -- MOBILITY + ROOK FILES
// ===================================================================================================

static constexpr uint64_t MOB_LOW_RANKS_W = 0x0000000000FFFF00ULL;   // ranks 2-3
static constexpr uint64_t MOB_LOW_RANKS_B = 0x00FFFF0000000000ULL;   // ranks 7-6
// SF11 RookOnFile {semi S(21,4), open S(47,25)}: the eg leg as a percent of the mg leg, IN PAWN TERMS.
static constexpr int      ROOKFILE_OPEN_EG_PCT = 32;               // (25/213) / (47/128)
static constexpr int      ROOKFILE_SEMI_EG_PCT = 11;               // ( 4/213) / (21/128)

/* The squares whose control counts as mobility for one side.
 *
 * Core (4/5 references): NOT attacked by an enemy pawn, NOT one of our blocked pawns (any piece directly in
 * front -- SF11's `shift<Down>(pos.pieces())`, evaluate.cpp:226), NOT our king. Own minor/rook squares DO
 * count (4/5; only SF1.1 excludes them): defending a piece is activity.
 * Candidates where the references split: our queen (SF11/15) and our pawns still on ranks 2-3 (SF11/15, Weiss).
 * ⚠️ SF11 also removes our PINNED pieces' squares and restricts a pinned piece to its pin line. That is one
 * lineage only and is not built yet (`slider_blockers` would provide it purely).
 *
 * @return the area bitboard
 * Gating: none; the caller gates on MOB_V2_MAG. Cost: ~6 shifts.
 */
static inline uint64_t mob_area(const V2Context &c, bool white) noexcept
{
	const uint64_t own     = white ? c.white : c.black;
	const uint64_t own_p   = c.pawns & own;
	const uint64_t enemy_p = c.pawns & ~own;
	const uint64_t e_att   = white ? ps_batt(enemy_p) : ps_watt(enemy_p);
	const uint64_t blocked = own_p & (white ? (c.occupied >> 8) : (c.occupied << 8));
	uint64_t excl = blocked | (c.kings & own) | e_att;
	if (Config::MOB_V2_EXCL_QUEEN)   excl |= c.queens & own;
	if (Config::MOB_V2_EXCL_LOWRANK) excl |= own_p & (white ? MOB_LOW_RANKS_W : MOB_LOW_RANKS_B);
	return ~excl;
}

/* OURS-FIRST safe-square mobility count (MOB_V2_SAFE), from the masks build_side_attacks stored.
 *
 * What: per stored N/B/R/Q mask, counts area squares NOT attacked by an enemy piece of lower (1) or lower-or-equal (2)
 * value, and accumulates cnt / raw_mg / raw_eg exactly as the shipped path does. Units: raw SF table units.
 * Why: v1's knight and queen activity (cpp_bitboard.cpp:1436-1480, :2881-2896) counted a square only if no lower-value
 * piece attacked it; SF and v2 exclude only enemy PAWN attacks. The one v1 mobility idea not yet measured in v2
 * (record-check 2026-09-15). Pawn attacks are already outside the area, so level 1 changes only rooks and queens.
 * Gating: called only when MOB_V2_SAFE != 0; the shipped path never reaches it.
 * Cost: one AND + popcount + two table reads per piece; the masks came from the one attack pass.
 */
static inline void mob_count_safe(MobAcc &m, const SideAttacks &e) noexcept
{
	const bool     le     = (Config::MOB_V2_SAFE == 2);
	const uint64_t minors = e.by[KNIGHT] | e.by[BISHOP];
	const uint64_t excl[4] = {
		le ? minors : 0,
		le ? minors : 0,
		minors | (le ? e.by[ROOK] : 0),
		minors | e.by[ROOK] | (le ? e.by[QUEEN] : 0),
	};
	for (int k = 0; k < m.n_pieces; ++k){
		const int i = m.ptype[k];
		const int n = __builtin_popcountll(m.pmask[k] & m.area & ~excl[i]);
		m.cnt[i]  += n;
		m.raw_mg  += MOB_TAB_MG[Config::MOB_V2_TABLE][i][n];
		m.raw_eg  += MOB_TAB_EG[Config::MOB_V2_TABLE][i][n];
	}
}

/* Resets a mobility accumulator's counters (not its per-piece arrays, which are only read up to their counts). */
static inline void mob_reset(MobAcc &m) noexcept
{
	m.cnt[0] = m.cnt[1] = m.cnt[2] = m.cnt[3] = 0;
	m.raw_mg = m.raw_eg = 0;
	m.n_rooks = m.n_pieces = 0;
	m.pinned = 0;
	m.ksq = 0;
}

/* Blockers for one side's king: pieces of EITHER colour that are the single piece between that king and an enemy slider.
 *
 * What: SF's blockers_for_king (position.cpp slider_blockers): snipers are enemy R/Q on the king's orthogonals and B/Q on
 * its diagonals on an EMPTY board; occupancy with the snipers removed; exactly one piece between => blocker.
 * Why: MOB_V2_PIN (SF11 evaluate.cpp:230, :273-274) removes these squares from the area and restricts our own ones to
 * their line. cpp_bitboard.h's slider_blockers returns only the current side's blockers without removing snipers, so it
 * is not the same set (same reason weak queen computes its own).
 * Gating: called only when MOB_V2_PIN. Cost: two empty-board slider masks + one between-mask per sniper.
 */
static inline uint64_t mob_king_blockers(const V2Context &c, bool white) noexcept
{
	return v2_king_blockers(c, white);
}

/* Both sides' attack maps plus mobility accumulators -- the ONE setup every mobility consumer uses.
 *
 * What: zeroes the accumulators, sets each side's area, runs the single attack pass per side, then any count that needs
 * both sides' maps (MOB_V2_SAFE). Why: the eval dispatch and both oracle probes previously each repeated this; one helper
 * guarantees the probes exercise the exact path search runs, and gives later form knobs one place to hook in.
 * Gating: with every MOB_V2_* form knob at default it performs exactly the shipped steps (byte-identical).
 * Cost: at or below the shipped build -- only the scalar counters are reset; the per-piece arrays are read only up to
 * n_rooks / n_pieces, so zeroing them (the old `MobAcc{}`) was wasted work.
 */
static inline void mobility_build(const V2Context &c, SideAttacks &wa, SideAttacks &ba, MobAcc &mw, MobAcc &mb) noexcept
{
	mob_reset(mw);
	mob_reset(mb);
	mw.area = mob_area(c, true);
	mb.area = mob_area(c, false);
	if (Config::MOB_V2_PIN){
		const uint64_t kw = mob_king_blockers(c, true);
		const uint64_t kb = mob_king_blockers(c, false);
		mw.area &= ~kw;
		mb.area &= ~kb;
		mw.pinned = kw & c.white;
		mb.pinned = kb & c.black;
		mw.ksq = (uint8_t)__builtin_ctzll(c.kings & c.white);
		mb.ksq = (uint8_t)__builtin_ctzll(c.kings & c.black);
	}
	build_side_attacks(wa, c, true,  &mw);
	build_side_attacks(ba, c, false, &mb);
	if (Config::MOB_V2_SAFE){
		mob_count_safe(mw, ba);
		mob_count_safe(mb, wa);
	}
}

/* Mobility in millipawns, Black-positive.
 *
 * ★ SCALE: MOB_V2_MAG is the knight MIDGAME table range in millipawns. mg leg = raw * MAG / 95; eg leg = raw *
 * MAG * 128 / (95 * 213) -- i.e. each SF11 leg is first converted by ITS OWN pawn, so an entry keeps its meaning
 * as a fraction of a pawn in its phase, and then the whole set is shrunk by one common factor. ☠️ Converting
 * the eg leg by the MIDGAME pawn would inflate it 1.66x -- the PASSER_V2_MG_PCT trap in the other direction.
 * Per-side blend, then difference, so the mirror swaps two identical computations exactly.
 *
 * @return Black-positive millipawns
 * Gating: caller calls only when MOB_V2_MAG > 0. Cost: a few multiplies. NO CLAMP.
 */
static inline int mobility_mp(const MobAcc &w, const MobAcc &b, const V2Context &c, EvalPair *legs = nullptr) noexcept
{
	if (g_c1_fit){
		// Texel C1: the accumulators already hold fitted millipawns per leg. Per-side blend with a sign-symmetric
		// divide, because fitted sums can be negative.
		if (legs){ legs->mg = b.raw_mg - w.raw_mg; legs->eg = b.raw_eg - w.raw_eg; }
		const int ws = (w.raw_mg * c.phase256 + w.raw_eg * (256 - c.phase256)) / 256;
		const int bs = (b.raw_mg * c.phase256 + b.raw_eg * (256 - c.phase256)) / 256;
		return bs - ws;
	}
	const int mag = Config::MOB_V2_MAG;
	// MOB_V2_TABLE: each table by ITS OWN knight range and pawn pair (table 0 = SF11's 95 / 128 / 213, the shipped numbers).
	const int t      = Config::MOB_V2_TABLE;
	const int nr     = MOB_TAB_N_RANGE[t];
	const int eg_den = nr * MOB_TAB_PAWN_EG[t];
	const int wmg = w.raw_mg * mag / nr;
	const int bmg = b.raw_mg * mag / nr;
	int weg = w.raw_eg * mag * MOB_TAB_PAWN_MG[t] / eg_den;
	int beg = b.raw_eg * mag * MOB_TAB_PAWN_MG[t] / eg_den;
	// MOB_V2_EG_PCT (form bake-off): scales the endgame leg only. Applied AFTER the conversion so the shipped 100 takes
	// no extra division (byte-identical) and the intermediate product cannot overflow.
	if (Config::MOB_V2_EG_PCT != 100){
		weg = weg * Config::MOB_V2_EG_PCT / 100;
		beg = beg * Config::MOB_V2_EG_PCT / 100;
	}
	if (legs){ legs->mg = bmg - wmg; legs->eg = beg - weg; }
	const int ws = (wmg * c.phase256 + weg * (256 - c.phase256)) >> 8;
	const int bs = (bmg * c.phase256 + beg * (256 - c.phase256)) >> 8;
	return bs - ws;
}

/*
	Slice 3 -- THREATS. SF11 evaluate.cpp:116-121 and :133-147, each leg pawn-converted by ITS OWN phase's pawn
	(mg /128, eg /213, x1000), so a constant keeps its meaning as a fraction of a pawn in its phase -- our pawn is flat.
	Victim index 0=PAWN 1=KNIGHT 2=BISHOP 3=ROOK 4=QUEEN.
	★ The transferable fact from the five-engine contrast: "a pawn attacks a piece" is the LARGEST single constant in
	4/4 references (here 1352 mp mg). The ratios transfer; the absolute scales never do, which is what THREAT_V2_PCT is for.
*/
static constexpr int TH_MINOR_MG[5] = {  47, 461, 617, 703, 617};
static constexpr int TH_MINOR_EG[5] = { 150, 192, 263, 559, 756};
static constexpr int TH_ROOK_MG[5]  = {  23, 297, 297,   0, 398};
static constexpr int TH_ROOK_EG[5]  = { 207, 333, 286, 178, 178};
static constexpr int TH_KING_MG     = 187, TH_KING_EG     = 418;   // ThreatByKing S(24,89)
static constexpr int TH_HANG_MG     = 539, TH_HANG_EG     = 169;   // Hanging S(69,36)
static constexpr int TH_SAFEPAWN_MG = 1352, TH_SAFEPAWN_EG = 441;  // ThreatBySafePawn S(173,94)
static constexpr int TH_PUSH_MG     = 375, TH_PUSH_EG     = 183;   // ThreatByPawnPush S(48,39)
static constexpr int TH_RESTRICT_MG =  55, TH_RESTRICT_EG =  33;   // RestrictedPiece S(7,7)

/* Which piece type stands on `bit` (0=PAWN..4=QUEEN, 5=KING); -1 if empty. Used only to index the victim tables. */
static inline int th_victim(const V2Context &c, uint64_t bit) noexcept
{
	if (bit & c.pawns)   return 0;
	if (bit & c.knights) return 1;
	if (bit & c.bishops) return 2;
	if (bit & c.rooks)   return 3;
	if (bit & c.queens)  return 4;
	return (bit & c.kings) ? 5 : -1;
}

/* Threats, Black-positive millipawns (slice 3).
 *
 * What: SF's threat family over the attack maps KS/mobility/space already built -- no second attack pass. Per side, then
 * differenced, so the colour mirror swaps two identical computations.
 * Legs: core (always, 4/4 references) = pawn-attacks-a-piece on SAFE pawns + target-indexed minor and rook threats;
 * switchable (references split) = pawn PUSH threat, KING attacker, HANGING, RESTRICTED squares, PAWN victims.
 * GATE form 0 = SF's `stronglyProtected` (enemy's view: their pawn attacks, or their double-attacks we do not match);
 * form 1 = Ethereal's `poorlyDefended` (victim's view, pawn support overrides).
 * ☠️ Hanging's v1 ownership argument ("~87% a subset of capture_gains") does NOT transfer: v2 has no capture-gains term.
 * ⚠️ RESTRICTED reads the same attack maps as mobility's area -- run the collinearity gate before laddering it.
 * Gating: caller calls only when THREAT_V2_PCT > 0. Cost: a few masks plus one bit-loop per threatened set.
 */
static inline int threats_mp(const V2Context &c, const SideAttacks &wa, const SideAttacks &ba, EvalPair *legs = nullptr) noexcept
{
	const int pct = Config::THREAT_V2_PCT;
	int side[2] = {0, 0};
	int lg_mg[2] = {0, 0}, lg_eg[2] = {0, 0};   // unblended legs for EVAL_V2_PAIR
	for (int s = 0; s < 2; ++s){
		const bool     white = (s == 0);
		const uint64_t own   = white ? c.white : c.black;
		const uint64_t them  = white ? c.black : c.white;
		const SideAttacks &us = white ? wa : ba;
		const SideAttacks &th = white ? ba : wa;
		const uint64_t their_np = them & ~c.pawns;            // their non-pawn pieces (king included, as in SF)
		long long mg = 0, eg = 0;

		// The defence gate: which of their pieces count as adequately protected.
		uint64_t protected_set, weak;
		if (Config::THREAT_V2_GATE == 1){
			// Ethereal: victim's view, and PAWN support overrides everything.
			const uint64_t poorly = (th.all & ~us.all) | (th.dbl & ~us.dbl & ~us.by[PAWN]);
			weak          = them & poorly & us.all;
			protected_set = them & ~poorly;
		} else {
			protected_set = th.by[PAWN] | (th.dbl & ~us.dbl);
			weak          = them & ~protected_set & us.all;
		}
		const uint64_t defended = their_np & protected_set;

		// Minor threats: SF pays `defended | weak` (a pawn-defended target still counts). Rook threats: `weak` only.
		uint64_t set = (defended | weak) & (us.by[KNIGHT] | us.by[BISHOP]);
		while (set){
			const uint64_t bit = set & -set; set ^= bit;
			const int v = th_victim(c, bit);
			if (v < 0 || v > 4) continue;
			if (v == 0 && !Config::THREAT_V2_PAWN_TARGETS) continue;
			mg += TH_MINOR_MG[v]; eg += TH_MINOR_EG[v];
		}
		set = weak & us.by[ROOK];
		while (set){
			const uint64_t bit = set & -set; set ^= bit;
			const int v = th_victim(c, bit);
			if (v < 0 || v > 4) continue;
			if (v == 0 && !Config::THREAT_V2_PAWN_TARGETS) continue;
			mg += TH_ROOK_MG[v]; eg += TH_ROOK_EG[v];
		}
		if (Config::THREAT_V2_KING){
			const int n = __builtin_popcountll(weak & us.by[KING] & ~c.pawns);
			mg += (long long)n * TH_KING_MG; eg += (long long)n * TH_KING_EG;
		}
		if (Config::THREAT_V2_HANGING){
			const uint64_t hanging = weak & (~th.all | (their_np & us.dbl));
			const int n = __builtin_popcountll(hanging);
			mg += (long long)n * TH_HANG_MG; eg += (long long)n * TH_HANG_EG;
		}
		if (Config::THREAT_V2_RESTRICT){
			const int n = __builtin_popcountll(th.all & ~protected_set & us.all);
			mg += (long long)n * TH_RESTRICT_MG; eg += (long long)n * TH_RESTRICT_EG;
		}
		// Pawn threats. `safe` follows SF: a square they do not attack, or one we also attack.
		const uint64_t safe   = ~th.all | us.all;
		const uint64_t our_p  = c.pawns & own;
		const uint64_t safe_p = our_p & safe;
		const uint64_t patt   = white ? ps_watt(safe_p) : ps_batt(safe_p);
		{
			const int n = __builtin_popcountll(patt & their_np);
			mg += (long long)n * TH_SAFEPAWN_MG; eg += (long long)n * TH_SAFEPAWN_EG;
		}
		if (Config::THREAT_V2_PUSH){
			// One and two-square pushes to EMPTY squares that their pawns do not attack and that are safe.
			const uint64_t empty = ~c.occupied;
			const uint64_t tp    = c.pawns & them;
			const uint64_t p_att = white ? ps_batt(tp) : ps_watt(tp);
			uint64_t push = (white ? (our_p << 8) : (our_p >> 8)) & empty;
			const uint64_t rank3 = white ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
			push |= (white ? ((push & rank3) << 8) : ((push & rank3) >> 8)) & empty;
			push &= ~p_att & safe;
			const uint64_t hit = (white ? ps_watt(push) : ps_batt(push)) & their_np;
			const int n = __builtin_popcountll(hit);
			mg += (long long)n * TH_PUSH_MG; eg += (long long)n * TH_PUSH_EG;
		}
		const int m = (int)(mg * pct / 100), g = (int)(eg * pct / 100);
		lg_mg[s] = m; lg_eg[s] = g;
		side[s] = (m * c.phase256 + g * (256 - c.phase256)) >> 8;
	}
	if (legs){ legs->mg = lg_mg[1] - lg_mg[0]; legs->eg = lg_eg[1] - lg_eg[0]; }
	return side[1] - side[0];
}

/* Bishop pair, Black-positive millipawns (slice 3).
 *
 * What: BPAIR_V2_MAG is the MIDGAME value in mp for holding two or more bishops; the endgame leg and any census
 * coupling follow BPAIR_V2_FORM. Per side, then the difference, so the colour mirror swaps identical computations.
 * Why: 5/5 references pay a bishop pair, at ~1-2x that engine's own knight PST rim-vs-centre mg spread -- the one
 * quantity that transfers across engines (EVAL-V2-SLICE3-DESIGN.md §1.4). v2 has had no pair term at all.
 *   FORM 0 flat (eg == mg; the SF lineage applies one value to both phases).
 *   FORM 1 endgame-heavy, eg = 3.5x mg (Ethereal 4:1, Weiss 3.3:1).
 *   FORM 2 flat + SF's OWN-pawn coupling: +2.8% of the pair per own pawn (SF11 pair x own-pawn cell, POSITIVE sign --
 *          the opposite of the "bishops like open boards" folklore, which no reference implements).
 * ⚠️ Two or more bishops of ANY colour complexion counts (4/5; only Ethereal requires opposite colours) -- so a
 * promoted third bishop counts, exactly as SF's piece_count test does.
 * Gating: caller calls only when BPAIR_V2_MAG > 0. Cost: two popcounts (plus one more per side under FORM 2).
 */
static inline int bishop_pair_mp(const V2Context &c, EvalPair *legs = nullptr) noexcept
{
	const int mag  = Config::BPAIR_V2_MAG;
	const int form = Config::BPAIR_V2_FORM;
	int side[2] = {0, 0};
	int lg_mg[2] = {0, 0}, lg_eg[2] = {0, 0};   // unblended legs for EVAL_V2_PAIR
	for (int s = 0; s < 2; ++s){
		const uint64_t own = s == 0 ? c.white : c.black;
		if (__builtin_popcountll(c.bishops & own) < 2) continue;
		int mg = mag;
		int eg = mag;
		if (form == 1)      eg = mag * 350 / 100;
		else if (form == 2){
			const int np = __builtin_popcountll(c.pawns & own);
			mg = eg = mag * (1000 + 28 * np) / 1000;
		}
		lg_mg[s] = mg; lg_eg[s] = eg;
		side[s] = (mg * c.phase256 + eg * (256 - c.phase256)) >> 8;
	}
	if (legs){ legs->mg = lg_mg[1] - lg_mg[0]; legs->eg = lg_eg[1] - lg_eg[0]; }
	return side[1] - side[0];
}

/* Slice 3 -- KAUFMAN / polynomial MATERIAL IMBALANCE, Black-positive millipawns.
 *
 * What: ONE scalar re-pricing ALL material by the whole piece census. SF's quadratic form, tables transcribed
 * VERBATIM from stockfish_11/src/material.cpp:33-53, converted to our units once at the end:
 *   white_pov = SUM over pt2 <= pt1 of OURS[pt1][pt2]*(cw1*cw2 - cb1*cb2) + THEIRS[pt1][pt2]*(cw1*cb2 - cb1*cw2)
 * ★ ANTISYMMETRIC BY CONSTRUCTION -- swapping the colours negates the sum exactly, so this term cannot break the
 * colour-symmetry gate whatever the coefficients are. (v2 is currently 0/4000; keep it that way.)
 *
 * Why: 2 of 4 references carry a census re-pricing (SF11 + SF15.1; Ethereal substitutes a closedness index and
 * Weiss has none), and it is the ONLY mechanism in any reference that prices piece REDUNDANCY. Its two largest
 * cells are R x R -208 and Q x enemy-R +268 -- neither expressible anywhere in v2 today.
 * ☠️ The TABLES ARE SF'S, NOT v1'S. v1's fitted cells (`cpp_bitboard.cpp:8269-8284`) CONTRADICT SF's sign in
 * B x own-pawn, N x enemy-pawn and 3 of 5 pair-vs-enemy cells, and price the bishop pair at only ~0.13 pawns.
 * ⚠️ **B x own-pawn is POSITIVE (+104) in SF and must STAY positive here.** "Bishops dislike own pawns" is NOT a
 * census term in ANY reference -- a COUNT cannot see square COLOUR -- so SF puts it in pieces() as a per-piece
 * colour-complex penalty. v2 already owns that as SF15.1's bad-bishop form, shipped in placement bundle E. A
 * negative cell here would both DOUBLE-OWN that term and mis-state the mechanism.
 *
 * ☠️ INDEX CONVENTION, the transcription hazard: SF is 0=bishop-pair pseudo-piece, 1=P, 2=N, 3=B, 4=R, 5=Q.
 * `V2Context::cnt_white/cnt_black` is 0=P .. 5=K. The local cw/cb vectors are built in SF ORDER deliberately --
 * never index cnt_* against these tables directly.
 *
 * UNITS -- ★ the one place [[convert-reference-constants-by-positional-scale-not-by-the-pawn]] INVERTS: this
 * re-prices MATERIAL, so the PAWN is the correct anchor, not v2's 5-35 mp positional spread. SF's cells are its own
 * mg units and SF divides the side difference by 16; SF11's mg pawn is 128 (`types.h:182`) against our 1000, so one
 * SF cell unit = 1000/(16*128) = 0.488 mp. ⇒ **KAUF_V2_MAG = 1000 means "exactly SF's scale in our millipawns"**,
 * and the ladder moves that one number rather than 36 coefficients (the pattern that won mobility: reference SHAPE,
 * our MAGNITUDE). Fitting the cells is explicitly NOT the plan -- raw-corpus fits are 5-for-5 bench-negative.
 *
 * KAUF_V2_PAIR decides WHO OWNS THE BISHOP PAIR: 1 = this term (SF's structure, where the pair is a pseudo-piece
 * whose value rises with own pawns, falls with own queen, and falls with EVERY enemy unit), 0 = the pair row and
 * column are zeroed so the standalone `bishop_pair_mp` can own it. ⚠️ Exactly one of the two should be non-zero.
 * ★ This directly tests the open question: v2's measurement that the pair is "already owned by PST + mobility"
 * refuted a FLAT pair -- it never tested SF's CONDITIONED one.
 *
 * Gating: caller calls only when KAUF_V2_MAG != 0, so 0 = absent = byte-identical.
 * Cost: counts are already in the context; 21 + 15 multiply-adds once per eval. No loop over squares, no attack map.
 */
static constexpr int KAUF_OURS[6][6] = {
	{ 1438,    0,    0,    0,    0,   0 },   // bishop pair
	{   40,   38,    0,    0,    0,   0 },   // pawn
	{   32,  255,  -62,    0,    0,   0 },   // knight
	{    0,  104,    4,    0,    0,   0 },   // bishop
	{  -26,   -2,   47,  105, -208,   0 },   // rook
	{ -189,   24,  117,  133, -134,  -6 },   // queen
};
static constexpr int KAUF_THEIRS[6][6] = {
	{    0,    0,    0,    0,    0,   0 },   // bishop pair
	{   36,    0,    0,    0,    0,   0 },   // pawn
	{    9,   63,    0,    0,    0,   0 },   // knight
	{   59,   65,   42,    0,    0,   0 },   // bishop
	{   46,   39,   24,  -24,    0,   0 },   // rook
	{   97,  100,  -42,  137,  268,   0 },   // queen
};

/* ☠️ DIAGNOSTIC FORM 1 -- v1's FITTED cells (`cpp_bitboard.cpp:8269-8284`), carried here ONLY to test one
 * hypothesis, and NOT as a ship candidate. The 2026-09-18 ladder found SF's tables (FORM 0) monotonically WORSE
 * on all six corpora at every magnitude, while the SAME FORM with v1's fitted cells is recorded as HELPING v1 by
 * ~7% on the same instrument. Sign and scale were both ruled out (v1 and this both do `total -= white_pov_sum`;
 * a hand-checked pair+minor-swap position reads +0.29 pawns at FORM 0 MAG=1000).
 * ⇒ HYPOTHESIS: the failure is BASIS, not scale. Imbalance cells are corrections layered on the PIECE VALUES they
 * correct, and SF11's midgame pieces are ~2x steeper than ours (mg knight 6.10 pawns vs our 3.25). ★ v1 and v2
 * SHARE `Config::values[]`, so if the basis story holds, v1's cells should help v2 where SF's hurt it.
 * ⚠️ UNIT CONVENTION DIFFERS: v1 applies its sum as millipawns DIRECTLY (`total -= kauf*SCALE/100`, SCALE 100),
 * with no /16 and no pawn conversion -- so its cells are ~2x SF's in effect. Hence FORM 1 divides by 1000, not
 * 2048, keeping "MAG = 1000 means THIS form's own native scale" true for both forms and the ladder comparable.
 * ☠️ If FORM 1 also fails, the shared-piece-values argument dies with it and the census form itself is the problem.
 */
static constexpr int KAUF_V1_OURS[6][6] = {
	{   26,    0,    0,    0,    0,   0 },   // bishop pair
	{   -3,   13,    0,    0,    0,   0 },   // pawn
	{    4,  178,  -28,    0,    0,   0 },   // knight
	{   53, -106,  162,   29,    0,   0 },   // bishop
	{   11,  -80, -108,   98,  -90,   0 },   // rook
	{  -23,  163,  131,  152,   -3,  38 },   // queen
};
static constexpr int KAUF_V1_THEIRS[6][6] = {
	{    0,    0,    0,    0,    0,   0 },   // bishop pair
	{  -89,    0,    0,    0,    0,   0 },   // pawn
	{   54, -104,    0,    0,    0,   0 },   // knight
	{  -47,    7,  120,    0,    0,   0 },   // bishop
	{  -29,  226,  -81, -102,    0,   0 },   // rook
	{   39,  120,  -11,  -88,  113,   0 },   // queen
};

/* FORM 2 -- the DERIVED rescale, ZERO free parameters, and the one arm the basis hypothesis actually implies.
 * ☠️ A GLOBAL rescale of SF's cells is already REFUTED: the 09-18 ladder swept MAG 250..2000 and every point was
 * monotonically worse, so no single scalar on SF's tables can work. What the hypothesis implies instead is a
 * DIFFERENTIAL rescale: if a cell is a correction proportional to the VALUE of the pieces it corrects, it should
 * scale by our-value/SF-value for BOTH of its indices.
 *   ratio = ours(pawns) / SF11-mg(pawns):  P 1.00/1.00 = 1.00 · N 3.25/6.10 = 0.53 · B 3.45/6.45 = 0.53
 *                                          R 5.00/9.97 = 0.50 · Q 10.0/19.83 = 0.50
 * ⇒ Non-pawn pieces are uniformly ~0.52 and pawns are 1.00, so this is NOT a rescale -- it REWEIGHTS pawn-interaction
 * cells UP relative to piece-interaction ones (piece x piece -> ~0.27, piece x pawn -> ~0.52, pawn x pawn -> 1.00).
 * ⚠️ JUDGEMENT CALL, stated because it is not forced by the derivation: the PAIR slot keeps ratio 1.00. It is a 0/1
 * INDICATOR, not a piece count, so a piece-value ratio has no meaning for it; its cells are rescaled only through
 * their OTHER index. Scaling it as a bishop instead would put pair x pair at ~0.20 pawns, below every reference's
 * pair value -- if FORM 2 fails, this is one of the two places to look (the other is that the cells may simply not
 * be separable into per-piece factors at all).
 * Ratios are /256 fixed point; the product of two needs >> 16.
 */
static constexpr int KAUF_VAL_RATIO[6] = { 256, 256, 136, 137, 129, 129 };  // pair, P, N, B, R, Q

/* FORM 3 (2026-09-30) -- TEXEL-FITTED cells on SF18 SEARCH labels (`diagnostics/_texel_kauf_fit.py`; C3 doc §18c),
 * loaded from KAUF_V2_FILE. Never tried before: 09-18 only swept SF's fixed tables x one scalar on §I, and its "fitting
 * is NOT the plan" was a rule about d6-OUTCOME corpus fits (5/5 bench-negative) -- these labels are depth-independent.
 * Cells are in MILLIPAWNS per unit, White-POV, same index order as the tables above (0=pair 1=P 2=N 3=B 4=R 5=Q), so
 * MAG = 1000 means "exactly as fitted". File lines: `O a b value` (OURS, b <= a) / `T a b value` (THEIRS, b < a).
 * A missing/malformed file leaves the form OFF (every cell 0, reported) -- never a half-loaded table. */
static int KAUF_FIT_OURS[6][6], KAUF_FIT_THEIRS[6][6];

static void kauf_fit_load()
{
	for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) KAUF_FIT_OURS[a][b] = KAUF_FIT_THEIRS[a][b] = 0;
	const char *path = std::getenv("KAUF_V2_FILE");
	if (!path || !*path){
		// No file: the SHIPPED fitted cells, compiled in (ship_tables_v2.h, 2026-10-03) — V2_PRESET=shipped needs no file.
		for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b){
			KAUF_FIT_OURS[a][b] = KAUF_SHIP_OURS[a][b]; KAUF_FIT_THEIRS[a][b] = KAUF_SHIP_THEIRS[a][b];
		}
		std::cerr << "[kauf] KAUF_V2_FORM=3 ON: compiled shipped cells (ship_tables_v2.h)" << '\n';
		return;
	}
	std::ifstream in(path);
	if (!in){
		std::cerr << "☠️ KAUF_V2_FILE=" << path << " cannot be opened -- the term stays at 0." << '\n';
		return;
	}
	int o[6][6] = {}, t[6][6] = {}, n_set = 0;
	std::string line;
	while (std::getline(in, line)){
		if (line.empty() || line[0] == '#') continue;
		char k; int a, b; double v;
		if (std::sscanf(line.c_str(), " %c %d %d %lf", &k, &a, &b, &v) != 4 || a < 0 || a > 5 || b < 0 || b > a
		    || (k != 'O' && k != 'T') || (k == 'T' && b == a)){
			std::cerr << "☠️ KAUF_V2_FILE malformed line '" << line << "' -- the term stays at 0." << '\n';
			return;
		}
		(k == 'O' ? o : t)[a][b] = (int)std::lround(v);
		++n_set;
	}
	for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b){ KAUF_FIT_OURS[a][b] = o[a][b]; KAUF_FIT_THEIRS[a][b] = t[a][b]; }
	std::cerr << "[kauf] KAUF_V2_FORM=3 ON: " << n_set << " fitted cells from " << path << '\n';
}

/* ═══ NARROW MATERIAL CLASSES (2026-10-01; C3 doc §18f-g, `diagnostics/_material_class_fit.py`) ═══════════════════════
 * WHY: SF18's d14 labels and OUR d10 search still disagree on four material signatures (depth residual, 9,827 rows):
 * queen vs no queen −6.3pp toward the queen side · minor vs ≥2 pawns +4.7 · rook vs two minors −2.7 · bishop pair +1.8.
 * The Texel-fitted Kaufman census (FORM 3) attacked the same errors but fired on 52-85% of all positions and lost at 250k
 * (−30 / −20) ⇒ these terms fire ONLY inside their own class. A = the side named first; d = the OTHER side's surplus:
 *   QUEEN  A has a queen, the other side none:  Q0 + QR·dR + QM·dM + QP·dP     (dR/dM/dP = rooks / minors / pawns)
 *   R2M    A +1 rook, −2 minors, queens equal:  R2M
 *   MINOR  A +1 minor, ≥2 fewer pawns, queens + rooks equal:  MP0 + MPP·(pawn deficit − 2)
 *   PAIR   A has the bishop pair, the other side not; minors, queens, rooks equal:  PAIR
 * Values in mp per unit, A-oriented, scaled by phase256/256 (fades into the endgame, which POT winnability owns).
 * Antisymmetric by construction (each class is evaluated for both sides with opposite sign); truncation toward zero. */
static inline int mcl_white_mp(const V2Context &c) noexcept      // White-oriented, unscaled (mp)
{
	int s = 0;
	for (int side = 0; side < 2; ++side){
		const int8_t *a = side == 0 ? c.cnt_white : c.cnt_black;   // 0=P 1=N 2=B 3=R 4=Q
		const int8_t *o = side == 0 ? c.cnt_black : c.cnt_white;
		const int sg = side == 0 ? 1 : -1;
		const int dR = o[3] - a[3], dM = (o[1] + o[2]) - (a[1] + a[2]), dP = o[0] - a[0];
		int v = 0;
		if (a[4] > 0 && o[4] == 0)
			v += Config::MCL_V2_Q0 + Config::MCL_V2_QR * dR + Config::MCL_V2_QM * dM + Config::MCL_V2_QP * dP;
		if (a[4] == o[4] && a[3] - o[3] == 1 && dM == 2)
			v += Config::MCL_V2_R2M;
		if (a[4] == o[4] && a[3] == o[3] && dM == -1 && dP >= 2)
			v += Config::MCL_V2_MP0 + Config::MCL_V2_MPP * (dP - 2);
		if (dM == 0 && a[4] == o[4] && a[3] == o[3] && a[2] >= 2 && o[2] < 2)
			v += Config::MCL_V2_PAIR;
		s += sg * v;
	}
	return s;
}

/* Black-positive, mg-weighted contribution to the total (non-pair mode). */
static inline int mcl_mp(const V2Context &c) noexcept
{
	return -(mcl_white_mp(c) * c.phase256 / 256);
}

static inline int kaufman_mp(const V2Context &c) noexcept
{
	const bool pair_here = Config::KAUF_V2_PAIR != 0;
	const bool v1_tables = Config::KAUF_V2_FORM == 1;
	int cw[6], cb[6];
	cw[0] = (pair_here && __builtin_popcountll(c.bishops & c.white) >= 2) ? 1 : 0;
	cb[0] = (pair_here && __builtin_popcountll(c.bishops & c.black) >= 2) ? 1 : 0;
	for (int t = 0; t < 5; ++t){            // v2 order 0=P 1=N 2=B 3=R 4=Q  ->  SF slots 1..5
		cw[t + 1] = (int)c.cnt_white[t];
		cb[t + 1] = (int)c.cnt_black[t];
	}
	const bool fit_tables = Config::KAUF_V2_FORM == 3;
	const int (*OURS)[6]   = fit_tables ? KAUF_FIT_OURS   : v1_tables ? KAUF_V1_OURS   : KAUF_OURS;
	const int (*THEIRS)[6] = fit_tables ? KAUF_FIT_THEIRS : v1_tables ? KAUF_V1_THEIRS : KAUF_THEIRS;
	const bool ratio_scale = Config::KAUF_V2_FORM == 2;
	long long sum = 0;                      // White-POV, in the selected form's own cell units
	for (int pt1 = 0; pt1 < 6; ++pt1)
		for (int pt2 = 0; pt2 <= pt1; ++pt2){
			long long o = OURS[pt1][pt2], t = THEIRS[pt1][pt2];
			if (ratio_scale){               // FORM 2: derived per-piece value-ratio rescale, /256 fixed point
				o = o * KAUF_VAL_RATIO[pt1] * KAUF_VAL_RATIO[pt2] >> 16;
				t = t * KAUF_VAL_RATIO[pt1] * KAUF_VAL_RATIO[pt2] >> 16;
			}
			sum += o * (cw[pt1] * cw[pt2] - cb[pt1] * cb[pt2])
			     + t * (cw[pt1] * cb[pt2] - cb[pt1] * cw[pt2]);
		}
	// White-POV -> Black-positive (our convention), and cell units -> our millipawns in one step.
	// ⚠️ The divisor is the FORM's native convention: SF divides its side-difference by 16 on a 128-mg pawn
	// (=> 2048 against our 1000), while v1 applies its sum as millipawns directly (=> 1000). Keeping both at
	// "MAG 1000 == this form's own native scale" is what makes the two ladders comparable.
	// FORM 3's cells are already millipawns (as fitted) => divisor 1000, like FORM 1.
	return (int)(-sum * (long long)Config::KAUF_V2_MAG / ((v1_tables || fit_tables) ? 1000 : 2048));
}

// Slice 3 -- SPACE region masks. SF counts OWN-CAMP development room; Ethereal counts a shared centre block.
static constexpr uint64_t SPACE_RANKS_W = 0x00000000FFFFFF00ULL;   // relative ranks 2-4 for White
static constexpr uint64_t SPACE_RANKS_B = 0x00FFFFFF00000000ULL;   // relative ranks 2-4 for Black (7-5)
static constexpr uint64_t SPACE_ETH_BIG = 0x00003C3C3C3C0000ULL;   // c3-f6, shared (Ethereal CENTER_BIG)
// Files c-f. ⚠️ Same value as the placement block's PL_CENTRE_FILES, which is declared BELOW this point in the file
// (the slice-2 placement constants sit after the slice-3 space term), so space owns its own copy rather than
// forward-declaring or reordering a shipped, oracle-verified block.
static constexpr uint64_t SPACE_CENTRE_FILES = 0x3C3C3C3C3C3C3C3CULL;
// Start-position total non-pawn material in OUR values (2N+2B+2R+Q per side = 33,400 each): the gate's denominator.
static constexpr int      SPACE_START_NPM = 66800;
// One reference unit of space: SF11's shape at the start weight (15^2/16) over 12 counted squares = 169 raw.
static constexpr int      SPACE_RAW_REF = 169;

/* Space, Black-positive millipawns (slice 3).
 *
 * What: per side, counts SAFE squares in a region and weights them by piece count, exactly SF's shape; the result is
 * applied MIDGAME-ONLY (all 3/3 references that have a space term give it a zero endgame leg), then differenced.
 * Why: 3/5 references carry it and v2 has nothing of this shape. ☠️ v1's flat `SPACE_MAG` form read net-flat and
 * NON-MONOTONIC on the pre-08-14 contaminated harness -- unreadable, not refuted -- and SF's gated form was never
 * built here (EVAL-V2-SLICE3-DESIGN.md §0.1).
 * Forms (the references split, so each is a knob): REGION own-camp c-f x ranks 2-4 (SF) or c3-f6 (Ethereal) ·
 * SAFE `~own pawns & ~enemy pawn attacks` (SF) or `~all enemy attacks & (we attack or occupy)` (Ethereal) ·
 * WEIGHT (pieces-1)^2/16 (SF11) or linear (Ethereal) · BEHIND = SF's double count of un-attacked squares behind
 * our pawns · GATE_PCT = SF's non-pawn-material gate.
 * ⚠️ Rides the attack maps KS/mobility already build -- no second attack pass. Symmetric positions cancel exactly.
 * Gating: caller calls only when SPACE_V2_MAG > 0. Cost: a few masks, two popcounts, one multiply per side.
 */
static inline int space_mp(const V2Context &c, const SideAttacks &wa, const SideAttacks &ba, EvalPair *legs = nullptr) noexcept
{
	if (Config::SPACE_V2_GATE_PCT > 0
	    && (c.npm_white + c.npm_black) * 100 < SPACE_START_NPM * Config::SPACE_V2_GATE_PCT)
		return 0;
	int side[2] = {0, 0};
	int lg_mg[2] = {0, 0};   // eg leg is identically 0 for space (see below)
	for (int s = 0; s < 2; ++s){
		const bool     white   = (s == 0);
		const uint64_t own     = white ? c.white : c.black;
		const uint64_t own_p   = c.pawns & own;
		const uint64_t enemy_p = c.pawns & ~own;
		const SideAttacks &us   = white ? wa : ba;
		const SideAttacks &them = white ? ba : wa;
		const uint64_t region = Config::SPACE_V2_REGION == 1
		                      ? SPACE_ETH_BIG
		                      : (SPACE_CENTRE_FILES & (white ? SPACE_RANKS_W : SPACE_RANKS_B));
		const uint64_t safe = Config::SPACE_V2_SAFE == 1
		                    ? (region & ~them.all & (us.all | own))
		                    : (region & ~own_p & ~(white ? ps_batt(enemy_p) : ps_watt(enemy_p)));
		int count = __builtin_popcountll(safe);
		if (Config::SPACE_V2_BEHIND){
			uint64_t behind = own_p;
			if (white){ behind |= behind >> 8; behind |= behind >> 16; }
			else      { behind |= behind << 8; behind |= behind << 16; }
			count += __builtin_popcountll(safe & behind & ~them.all);
		}
		int pieces = 0;
		for (int t = 0; t < 6; ++t) pieces += white ? c.cnt_white[t] : c.cnt_black[t];
		const int w = Config::SPACE_V2_WEIGHT == 1 ? 16 : (pieces - 1) * (pieces - 1);
		const long long raw = (long long)count * w / 16;
		// Midgame only: phase256 = 256 in the full midgame, 0 in the deep endgame.
		// ★ Space is MIDGAME-ONLY BY CONSTRUCTION — the phase factor multiplies the whole term, so its eg
		// leg is exactly 0 and the mg leg is the unscaled value. No rounding difference between modes here.
		lg_mg[s] = (int)((long long)Config::SPACE_V2_MAG * raw / (long long)SPACE_RAW_REF);
		side[s] = (int)((long long)Config::SPACE_V2_MAG * raw * c.phase256 / ((long long)SPACE_RAW_REF * 256));
	}
	if (legs){ legs->mg = lg_mg[1] - lg_mg[0]; legs->eg = 0; }
	return side[1] - side[0];
}

/* Rook on an open or semi-open file, Black-positive millipawns.
 *
 * Reads PawnEntry.openFiles / halfOpen, built at rung 2 for exactly this consumer. ★ No 7th-rank term: SF11+
 * and Weiss dropped it (their eg-heavy rook mobility carries it); no cap: v1's 300 cap absorbed its own file
 * penalties.
 *
 * @return Black-positive millipawns
 * Gating: caller calls only when ROOKFILE_V2_OPEN or _SEMI is non-zero. Cost: one bit loop over rooks.
 */
/* `legs` non-null ⇒ PAIR MODE: fill the unblended (mg,eg) and return 0; the caller blends once.
   ⚠️ This term blends PER ROOK, so the two modes differ by up to one millipawn per rook on the board —
   the largest per-term truncation gap in v2 after v2_piece_value. That is the expected bound, not a bug. */
static inline int rookfile_mp(const PawnEntry &e, const V2Context &c, EvalPair *legs = nullptr) noexcept
{
	int side[2] = {0, 0};
	int leg_mg[2] = {0, 0}, leg_eg[2] = {0, 0};
	for (int s = 0; s < 2; ++s){
		uint64_t rb = c.rooks & (s == 0 ? c.white : c.black);
		while (rb){
			const int f = __builtin_ctzll(rb) & 7;
			rb &= rb - 1;
			if (!((e.halfOpen[s] >> f) & 1)) continue;                 // one of our own pawns is on the file
			const bool open = (e.openFiles >> f) & 1;
			const int  mg   = open ? Config::ROOKFILE_V2_OPEN : Config::ROOKFILE_V2_SEMI;
			const int  eg   = mg * (open ? ROOKFILE_OPEN_EG_PCT : ROOKFILE_SEMI_EG_PCT) / 100;
			leg_mg[s] += mg; leg_eg[s] += eg;
			side[s] += (mg * c.phase256 + eg * (256 - c.phase256)) >> 8;
		}
	}
	if (legs){ legs->mg = leg_mg[1] - leg_mg[0]; legs->eg = leg_eg[1] - leg_eg[0]; }
	return side[1] - side[0];
}

// ─── PER-PIECE PLACEMENT (SF11 evaluate.cpp:291-361) ───────────────────────────────────────────────

static constexpr uint64_t PL_OUTPOST_RANKS_W = 0x0000FFFFFF000000ULL;   // ranks 4-6
static constexpr uint64_t PL_OUTPOST_RANKS_B = 0x000000FFFFFF0000ULL;   // ranks 5-3
static constexpr uint64_t PL_DARK_SQUARES    = 0xAA55AA55AA55AA55ULL;   // a1 is dark
static constexpr uint64_t PL_CENTRE_FILES    = 0x3C3C3C3C3C3C3C3CULL;   // files c-f
static constexpr uint64_t PL_CENTRE          = 0x0000001818000000ULL;   // d4 e4 d5 e5
static constexpr uint64_t PL_RANK_1 = 0x00000000000000FFULL, PL_RANK_8 = 0xFF00000000000000ULL;

/* SF11 values converted by the pawn (mg x1000/128, eg x1000/213), millipawns. The *_V2_PCT knobs scale these, so
 * 100 = the plain pawn conversion. Penalties are stored positive and subtracted at the use site. */
static constexpr int PL_OUTPOST_MG = 234, PL_OUTPOST_EG = 99;     // S(30,21)
static constexpr int PL_REACH_MG   = 250, PL_REACH_EG   = 47;     // S(32,10)
static constexpr int PL_BEHIND_MG  = 141, PL_BEHIND_EG  = 14;     // S(18,3)
static constexpr int PL_BADB_MG    =  23, PL_BADB_EG    = 33;     // S(3,7) per unit
static constexpr int PL_LONGD_MG   = 352, PL_LONGD_EG   =  0;     // S(45,0)
static constexpr int PL_TRAPR_MG   = 406, PL_TRAPR_EG   = 47;     // S(52,10) per unit
static constexpr int PL_WEAKQ_MG   = 383, PL_WEAKQ_EG   = 70;     // S(49,15)

// ── FORM alternatives (per-term reference forms from source, 2026-09-14). Each leg converted by ITS OWN engine's pawn. ──
// OUTPOST_V2_FORM 1 -- Ethereal KnightOutpost / BishopOutpost[outside][defended], pawn 82 mg / 144 eg.
// Index = outside*2 + defended, where outside = a- or h-file.
static constexpr int PL_ETH_N_MG[4] = { 146, 488,  85, 256};   // S(12,-32) S(40,0) S(7,-24) S(21,-3)
static constexpr int PL_ETH_N_EG[4] = {-222,   0,-167, -21};
static constexpr int PL_ETH_B_MG[4] = { 195, 610, 110, -49};   // S(16,-16) S(50,-3) S(9,-9) S(-4,-4)
static constexpr int PL_ETH_B_EG[4] = {-111, -21, -63, -28};
// OUTPOST_V2_FORM 2 -- SF15.1 Outpost[knight] S(54,34), Outpost[bishop] S(31,25), pawn 126/208 (no knight x2).
static constexpr int PL_SF15_OUT_N_MG = 429, PL_SF15_OUT_N_EG = 163;
static constexpr int PL_SF15_OUT_B_MG = 246, PL_SF15_OUT_B_EG = 120;
// BADB_V2_FORM 1 -- SF15.1 BishopPawns by file edge-distance {a/h, b/g, c/f, d/e} = S(3,8) S(3,9) S(2,7) S(3,7), pawn 126/208.
static constexpr int PL_SF15_BADB_MG[4] = {24, 24, 16, 24};
static constexpr int PL_SF15_BADB_EG[4] = {38, 43, 34, 34};
// BADB_V2_FORM 2 -- Weiss BishopBadP S(-1,-5), pawn 104/204. BADB_V2_FORM 3 -- Ethereal BishopRammedPawns S(-8,-17), pawn 82/144.
static constexpr int PL_WEISS_BADB_MG = 10, PL_WEISS_BADB_EG = 25;
static constexpr int PL_ETH_BADB_MG   = 98, PL_ETH_BADB_EG   = 118;
// LATENT_V2 (ours) -- v1's latent pawn-pressure increments (BISHOP_MOB_PAWN_ATTACK 15, the rook literal 10), mp, midgame only.
static constexpr int PL_LATENT_B = 15, PL_LATENT_R = 10;

/* Detector output, index 0 = White, 1 = Black. Counts and "units" (the multiplier already applied), never scores,
 * so the probe can compare them one-for-one against the python-chess oracle. */
struct PlaceCounts {
	int outpost_n[2], outpost_b[2], reach_n[2], behind[2], badb_units[2], longdiag[2], traprook_units[2], weakq[2];
	int latent_b[2], latent_r[2];      // OURS: latent squares (behind own blockers) attacking an enemy pawn
	int eth_n[2][4], eth_b[2][4];      // OUTPOST_V2_FORM 1: counts by [outside*2 + defended]
	int badb_cls[2][4];                // BADB_V2_FORM 1: units by the bishop's file edge-distance class (a/h, b/g, c/f, d/e)
};

/* Trapped-rook units for one rook, in the active TRAPROOK_V2_FORM.
 *
 * Common to both forms: the rook is NOT on our semi-open file, and it is on the king's EDGE side -- file-symmetrised.
 * ☠️ SF11's `(kf < FILE_E) == (file_of(s) < kf)` is NOT file-mirror symmetric when the rook shares the king's file
 * (Kh1+Rh2 trapped, mirrored Ka1+Ra2 not) -- caught by _eval_symmetry.py's file-mirror check as a 576 mp violation,
 * 2026-09-14. The test below is identical to SF for every kingside king; queenside is its exact mirror.
 *
 * FORM 0 (SF11): area mobility <= 3 -> 1 unit, 2 without castling rights.
 * FORM 1 (SF1.1 evaluate.cpp:650-674): mobility <= 6, the king on its back rank or the rook's rank, and NO half-open file
 * of ours between the king and the edge; value 180 - 16*mob (SF1.1 units, midgame only), halved while castling is still
 * possible. ⚠️ Uses our AREA count rather than SF1.1's own-occupied-excluded count, so the reuse path stays free.
 *
 * @return units in the form's own scale (see placement_mp)
 */
static inline int trap_rook_units(int f, int rrank, int mob, int ksq, bool white, uint8_t half_open, bool can_castle) noexcept
{
	const int kf = ksq & 7;
	if ((half_open >> f) & 1) return 0;                                       // on our semi-open file: not trapped
	if (!(kf < 4 ? (f <= kf) : (f >= kf))) return 0;                          // not on the king's edge side
	if (Config::TRAPROOK_V2_FORM == 1){
		if (mob > 6) return 0;
		const int krank = ksq >> 3;
		if (krank != (white ? 0 : 7) && krank != rrank) return 0;
		const unsigned edge = kf < 4 ? ((unsigned)half_open & ((1u << kf) - 1u)) : ((unsigned)half_open >> (kf + 1));
		if (edge) return 0;
		const int v = 180 - 16 * mob;
		return can_castle ? v / 2 : v;
	}
	if (mob > 3) return 0;
	return can_castle ? 1 : 2;
}

/* The per-piece placement DETECTOR. Pure; reads the context and Layer A's pawn masks.
 *
 * ★ Outposts use SF's pawn_attacks_span (pawns.cpp:88,114-115): the enemy's pawn attacks PLUS the forward adjacent-file
 *   span of every enemy pawn that is neither backward nor blocked -- a square is still an outpost if the only enemy pawn
 *   able to challenge it cannot advance. `backward` and `blocked` are Layer A's, which are SF's definitions.
 * ★ Trapped rook is the ELSE-branch of "own semi-open file" (SF: `if (is_on_semiopen_file) ... else if (mob <= 3)`),
 *   independent of whether the rook-file knob is on. Its mobility is the SAME area count mobility uses.
 * ☠️ Weak queen is computed here, NOT with cpp_bitboard.h's slider_blockers: that helper returns only the CURRENT
 *   side's blockers and does not remove snipers from the occupancy, while SF counts a single blocker of EITHER colour
 *   with snipers removed (position.cpp slider_blockers).
 * ⚠️ Not ported: SF's pinned-piece handling of the mobility area; KingProtector (overlaps the KS zone); Chess960
 *   cornered bishop.
 */
static inline void placement_detect(PlaceCounts &pc, const V2Context &c, const PawnEntry &e,
                                    const MobAcc *acc_w = nullptr, const MobAcc *acc_b = nullptr,
                                    bool want_latent = false) noexcept
{
	for (int s = 0; s < 2; ++s){
		const bool     white = (s == 0);
		const int      t     = 1 - s;
		const uint64_t own   = white ? c.white : c.black;
		const uint64_t them  = white ? c.black : c.white;
		const uint64_t own_p = c.pawns & own;
		const uint64_t tp    = c.pawns & them;

		pc.outpost_n[s] = pc.outpost_b[s] = pc.reach_n[s] = pc.behind[s] = 0;
		pc.badb_units[s] = pc.longdiag[s] = pc.traprook_units[s] = pc.weakq[s] = 0;
		pc.latent_b[s] = pc.latent_r[s] = 0;
		for (int k = 0; k < 4; ++k){ pc.eth_n[s][k] = pc.eth_b[s][k] = pc.badb_cls[s][k] = 0; }

		// Enemy pawn attack span: attacks + forward adjacent-file span of their non-backward, non-blocked pawns.
		const uint64_t elig = tp & ~e.backward[t] & ~e.blocked[t];
		const uint64_t eadj = ps_east(elig) | ps_west(elig);
		const uint64_t span = e.attacks[t] | (white ? ps_sfill(eadj >> 8) : ps_nfill(eadj << 8));
		// A pawn of EITHER colour directly in front of the square.
		const uint64_t pawn_in_front = white ? (c.pawns >> 8) : (c.pawns << 8);
		const uint64_t ranks = white ? PL_OUTPOST_RANKS_W : PL_OUTPOST_RANKS_B;
		const int      ofrm  = Config::OUTPOST_V2_FORM;
		// FORM 0 SF11: pawn-defended, outside the refined span. FORM 2 SF15.1: pawn-defended OR a pawn directly in front.
		// FORM 1 Ethereal: outside the RAW span (every enemy pawn ahead on an adjacent file counts); defence is an INDEX.
		const uint64_t outposts = (ofrm == 2) ? (ranks & (e.attacks[s] | pawn_in_front) & ~span)
		                                      : (ranks & e.attacks[s] & ~span);
		const uint64_t tadj     = ps_east(tp) | ps_west(tp);
		const uint64_t raw_safe = ranks & ~(white ? ps_sfill(tadj >> 8) : ps_nfill(tadj << 8));

		uint64_t nb = c.knights & own;
		while (nb){
			const uint8_t sq = (uint8_t)__builtin_ctzll(nb); nb &= nb - 1;
			const uint64_t m = 1ULL << sq;
			if (ofrm == 1){
				if (raw_safe & m) ++pc.eth_n[s][((m & (BB_FILE_A | BB_FILE_H)) ? 2 : 0) + ((e.attacks[s] & m) ? 1 : 0)];
			} else if (outposts & m) ++pc.outpost_n[s];
			else if (outposts & attacks_mask(white, c.occupied, sq, KNIGHT) & ~own) ++pc.reach_n[s];
			if (pawn_in_front & m) ++pc.behind[s];
		}

		const uint64_t blocked_any = own_p & (white ? (c.occupied >> 8) : (c.occupied << 8));
		const int      centre_blk  = __builtin_popcountll(blocked_any & PL_CENTRE_FILES);
		const int bfrm = Config::BADB_V2_FORM;
		uint64_t bb = c.bishops & own;
		while (bb){
			const uint8_t sq = (uint8_t)__builtin_ctzll(bb); bb &= bb - 1;
			const uint64_t m = 1ULL << sq;
			if (ofrm == 1){
				if (raw_safe & m) ++pc.eth_b[s][((m & (BB_FILE_A | BB_FILE_H)) ? 2 : 0) + ((e.attacks[s] & m) ? 1 : 0)];
			} else if (outposts & m) ++pc.outpost_b[s];
			if (pawn_in_front & m) ++pc.behind[s];
			const uint64_t colour = (PL_DARK_SQUARES & m) ? PL_DARK_SQUARES : ~PL_DARK_SQUARES;
			const int      same   = __builtin_popcountll(own_p & colour);
			// BAD BISHOP forms, where the references split: 0 SF11 N·(1+blk) · 1 SF15.1 N·(!pawnDefended + blk) with a
			// file-class table · 2 Weiss N·blk (zero while the centre is open) · 3 Ethereal: same-colour pawns RAMMED by an
			// enemy pawn only, no multiplier.
			if (bfrm == 1){
				const int f = sq & 7;
				const int u = same * (((e.attacks[s] & m) ? 0 : 1) + centre_blk);
				pc.badb_cls[s][f < 7 - f ? f : 7 - f] += u;
				pc.badb_units[s] += u;
			} else if (bfrm == 2) pc.badb_units[s] += same * centre_blk;
			else if (bfrm == 3)   pc.badb_units[s] += __builtin_popcountll(own_p & colour & e.blocked[s]);
			else                  pc.badb_units[s] += same * (1 + centre_blk);
			if (__builtin_popcountll(attacks_mask(white, c.pawns, sq, BISHOP) & PL_CENTRE) > 1) ++pc.longdiag[s];
		}

		uint64_t rb = c.rooks & own;
		const MobAcc *acc = white ? acc_w : acc_b;
		if (rb && (c.kings & own) && acc){
			// ★ REUSE PATH: the mobility loop already computed each rook's area count from the SAME occupancy (KS_V2_XRAY)
			// and the SAME area (mob_area), so reading it is byte-identical to recomputing it and costs nothing.
			const int ksq = __builtin_ctzll(c.kings & own);
			const bool can_castle = (c.castling_rights & (white ? PL_RANK_1 : PL_RANK_8)) != 0;
			for (int i = 0; i < acc->n_rooks; ++i)
				pc.traprook_units[s] += trap_rook_units(acc->rook_sq[i] & 7, acc->rook_sq[i] >> 3, acc->rook_mob[i],
				                                        ksq, white, e.halfOpen[s], can_castle);
		} else if (rb && (c.kings & own)){
			// Fallback when mobility is off (no accumulator was built): compute the counts locally.
			const uint64_t area = mob_area(c, white);
			const uint64_t occ  = Config::KS_V2_XRAY ? (c.occupied ^ c.queens ^ (c.rooks & own)) : c.occupied;
			const int ksq = __builtin_ctzll(c.kings & own);
			const bool can_castle = (c.castling_rights & (white ? PL_RANK_1 : PL_RANK_8)) != 0;
			while (rb){
				const uint8_t sq = (uint8_t)__builtin_ctzll(rb); rb &= rb - 1;
				const int mob = __builtin_popcountll(attacks_mask(white, occ, sq, ROOK) & area);
				pc.traprook_units[s] += trap_rook_units(sq & 7, sq >> 3, mob, ksq, white, e.halfOpen[s], can_castle);
			}
		}

		uint64_t qb = c.queens & own;
		while (qb){
			const uint8_t sq = (uint8_t)__builtin_ctzll(qb); qb &= qb - 1;
			const uint64_t snipers = (attacks_mask(white, 0, sq, ROOK) & c.rooks & them)
			                       | (attacks_mask(white, 0, sq, BISHOP) & c.bishops & them);
			const uint64_t occ = c.occupied ^ snipers;
			uint64_t sn = snipers;
			while (sn){
				const uint8_t r = (uint8_t)__builtin_ctzll(sn); sn &= sn - 1;
				const uint64_t b = betweenPieces(sq, r) & occ;
				if (b && !(b & (b - 1))){ ++pc.weakq[s]; break; }
			}
		}

		// OURS -- LATENT PAWN PRESSURE (v1 get_latent_bishop/rook_activity_score, without the retired heat map): squares a
		// slider would reach if its OWN blockers were removed (bishop: all own pieces; rook: own non-pawns), minus squares
		// it already reaches, that are not ours and from which a pawn of ours would attack an enemy pawn.
		if (want_latent){
			const uint64_t targets = white ? ps_batt(tp) : ps_watt(tp);
			uint64_t lb = c.bishops & own;
			while (lb){
				const uint8_t sq = (uint8_t)__builtin_ctzll(lb); lb &= lb - 1;
				const uint64_t a   = attacks_mask(white, c.occupied, sq, BISHOP);
				const uint64_t lat = attacks_mask(white, c.occupied & ~(a & own), sq, BISHOP) & ~a;
				pc.latent_b[s] += __builtin_popcountll(lat & ~own & targets);
			}
			uint64_t lr = c.rooks & own;
			while (lr){
				const uint8_t sq = (uint8_t)__builtin_ctzll(lr); lr &= lr - 1;
				const uint64_t a   = attacks_mask(white, c.occupied, sq, ROOK);
				const uint64_t lat = attacks_mask(white, c.occupied & ~(a & own & ~c.pawns), sq, ROOK) & ~a;
				pc.latent_r[s] += __builtin_popcountll(lat & ~own & targets);
			}
		}
	}
}

/* Placement score, Black-positive millipawns. Each term is its SF11 pawn-converted value x its percent knob.
 * Gating: the caller calls only when at least one *_V2_PCT is non-zero. NO CLAMP. */
static inline int placement_mp(const PlaceCounts &pc, const V2Context &c, EvalPair *legs = nullptr) noexcept
{
	int side[2] = {0, 0};
	int lg_mg[2] = {0, 0}, lg_eg[2] = {0, 0};   // unblended legs for EVAL_V2_PAIR
	for (int s = 0; s < 2; ++s){
		if (g_c1_fit){
			// Texel C1: fitted per-count values, already divided by 100 and scaled by their percent knobs.
			const int cnt[9] = {pc.outpost_n[s], pc.outpost_b[s], pc.behind[s], pc.badb_cls[s][0], pc.badb_cls[s][1],
			                    pc.badb_cls[s][2], pc.badb_cls[s][3], pc.traprook_units[s], pc.weakq[s]};
			int fm = 0, fg = 0;
			for (int j = 0; j < 9; ++j){ fm += cnt[j] * c1_place[0][j]; fg += cnt[j] * c1_place[1][j]; }
			lg_mg[s] = fm; lg_eg[s] = fg;
			side[s] = (fm * c.phase256 + fg * (256 - c.phase256)) / 256;
			continue;
		}
		long long mg = 0, eg = 0;
		const int ofrm = Config::OUTPOST_V2_FORM, opct = Config::OUTPOST_V2_PCT;
		if (ofrm == 1){
			for (int k = 0; k < 4; ++k){
				mg += ((long long)pc.eth_n[s][k] * PL_ETH_N_MG[k] + (long long)pc.eth_b[s][k] * PL_ETH_B_MG[k]) * opct;
				eg += ((long long)pc.eth_n[s][k] * PL_ETH_N_EG[k] + (long long)pc.eth_b[s][k] * PL_ETH_B_EG[k]) * opct;
			}
		} else if (ofrm == 2){
			mg += ((long long)pc.outpost_n[s] * PL_SF15_OUT_N_MG + (long long)pc.outpost_b[s] * PL_SF15_OUT_B_MG) * opct;
			eg += ((long long)pc.outpost_n[s] * PL_SF15_OUT_N_EG + (long long)pc.outpost_b[s] * PL_SF15_OUT_B_EG) * opct;
		} else {
			const int on = 2 * pc.outpost_n[s] + pc.outpost_b[s];             // SF11: knights count double
			mg += (long long)on * PL_OUTPOST_MG * opct;  eg += (long long)on * PL_OUTPOST_EG * opct;
		}
		mg += (long long)pc.reach_n[s] * PL_REACH_MG * Config::REACH_V2_PCT;   eg += (long long)pc.reach_n[s] * PL_REACH_EG * Config::REACH_V2_PCT;
		// FORM 1 = Weiss NBBehindPawn S(9,32) at Weiss's pawn (104 mg / 204 eg): 87 / 157 mp -- the endgame-heavy shape.
		const int bh_mg = Config::BEHIND_V2_FORM == 1 ? 87  : PL_BEHIND_MG;
		const int bh_eg = Config::BEHIND_V2_FORM == 1 ? 157 : PL_BEHIND_EG;
		mg += (long long)pc.behind[s] * bh_mg * Config::BEHIND_V2_PCT;  eg += (long long)pc.behind[s] * bh_eg * Config::BEHIND_V2_PCT;
		mg += (long long)pc.longdiag[s] * PL_LONGD_MG * Config::LONGDIAG_V2_PCT; eg += (long long)pc.longdiag[s] * PL_LONGD_EG * Config::LONGDIAG_V2_PCT;
		const int bfrm = Config::BADB_V2_FORM, bpct = Config::BADB_V2_PCT;
		if (bfrm == 1){
			for (int k = 0; k < 4; ++k){
				mg -= (long long)pc.badb_cls[s][k] * PL_SF15_BADB_MG[k] * bpct;
				eg -= (long long)pc.badb_cls[s][k] * PL_SF15_BADB_EG[k] * bpct;
			}
		} else {
			const int bmg = bfrm == 2 ? PL_WEISS_BADB_MG : bfrm == 3 ? PL_ETH_BADB_MG : PL_BADB_MG;
			const int beg = bfrm == 2 ? PL_WEISS_BADB_EG : bfrm == 3 ? PL_ETH_BADB_EG : PL_BADB_EG;
			mg -= (long long)pc.badb_units[s] * bmg * bpct;  eg -= (long long)pc.badb_units[s] * beg * bpct;
		}
		if (Config::TRAPROOK_V2_FORM == 1)
			mg -= (long long)pc.traprook_units[s] * 1000 * Config::TRAPROOK_V2_PCT / 204;   // SF1.1 raw units, pawn mg 204, mg only
		else {
			mg -= (long long)pc.traprook_units[s] * PL_TRAPR_MG * Config::TRAPROOK_V2_PCT;
			eg -= (long long)pc.traprook_units[s] * PL_TRAPR_EG * Config::TRAPROOK_V2_PCT;
		}
		mg -= (long long)pc.weakq[s] * PL_WEAKQ_MG * Config::WEAKQ_V2_PCT;     eg -= (long long)pc.weakq[s] * PL_WEAKQ_EG * Config::WEAKQ_V2_PCT;
		mg += ((long long)pc.latent_b[s] * PL_LATENT_B + (long long)pc.latent_r[s] * PL_LATENT_R) * Config::LATENT_V2_PCT;
		const int m = (int)(mg / 100), g = (int)(eg / 100);
		lg_mg[s] = m; lg_eg[s] = g;
		side[s] = (m * c.phase256 + g * (256 - c.phase256)) >> 8;
	}
	if (legs){ legs->mg = lg_mg[1] - lg_mg[0]; legs->eg = lg_eg[1] - lg_eg[0]; }
	return side[1] - side[0];
}

/* DETECTOR ORACLE PROBE for the placement sub-terms (diagnostic; never called from search).
 * Layout (long long, 20): per side [outpost_n, outpost_b, reach_n, behind, badb_units, longdiag, traprook_units, weakq,
 * latent_b, latent_r], White 0-9, Black 10-19. Form-dependent packing: OUTPOST_V2_FORM 1 packs the four Ethereal cells
 * into outpost_n / outpost_b, 8 bits each; BADB_V2_FORM 1 packs the four SF15.1 file classes into badb_units, 12 bits each. ⚠️ Reads KS_V2_XRAY (trapped-rook occupancy) and MOB_V2_EXCL_* (area): build an engine under
 * the arm's environment first. Compared against diagnostics/_placement_detector_oracle.py.
 */
void placement_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
                     uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask,
                     uint64_t castlingRights, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, castlingRights);
	PawnEntry pe;
	build_pawn_entry(pe, c);
	// Exercise the REUSE path (the one search runs when mobility is on), so the oracle validates it directly.
	MobAcc mw, mb;
	SideAttacks wa, ba;
	mobility_build(c, wa, ba, mw, mb);
	PlaceCounts pc;
	placement_detect(pc, c, pe, &mw, &mb, true);
	for (int s = 0; s < 2; ++s){
		const int o = 10 * s;
		long long on = pc.outpost_n[s], ob = pc.outpost_b[s], bu = pc.badb_units[s];
		if (Config::OUTPOST_V2_FORM == 1){
			on = ob = 0;
			for (int k = 0; k < 4; ++k){ on |= (long long)pc.eth_n[s][k] << (8 * k); ob |= (long long)pc.eth_b[s][k] << (8 * k); }
		}
		if (Config::BADB_V2_FORM == 1){
			bu = 0;
			for (int k = 0; k < 4; ++k) bu |= (long long)pc.badb_cls[s][k] << (12 * k);
		}
		out[o]   = on; out[o+1] = ob; out[o+2] = pc.reach_n[s]; out[o+3] = pc.behind[s];
		out[o+4] = bu; out[o+5] = pc.longdiag[s]; out[o+6] = pc.traprook_units[s]; out[o+7] = pc.weakq[s];
		out[o+8] = pc.latent_b[s]; out[o+9] = pc.latent_r[s];
	}
}

/* KING-SAFETY COUNT PROBE (diagnostic; never called from search).
 *
 * Layout (long long, 36), indexed by the KING examined (0 = White's, 1 = Black's), two slots per channel:
 *   0 attacker count · 2 weighted attacker sum · 4 weak zone squares · 6 king-adjacent attacked squares · 8 safe-check
 *   squares (all four types, summed per square) · 10 the scored unit total from ks_units (onset already subtracted)
 *   -- the rung-1 layout, unchanged -- then the 2026-09-27 balance channels:
 *   12 x-ray attacker count · 14 ring attack instances · 16 unsafe checks · 18 blockers · 20 flank attacks ·
 *   22 flank defence · 24 knight guards the ring · 26 contest excess · 28 contested squares (>= 2 attackers) ·
 *   30 enemy queen present · 32 contest-scaled attacker weight (defaware-v2) · 34 attacker gate passes
 * ★ Computed by ks_channels, the SAME function ks_units scores from, so probe and scorer cannot disagree.
 * ☠️ Channel counts are UNCONDITIONAL (every balance channel is computed); only out[10-11] respects the knobs.
 */
void ks_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
              uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, 0);
	SideAttacks wa, ba;
	build_side_attacks(wa, c, true);
	build_side_attacks(ba, c, false);
	for (int s = 0; s < 2; ++s){
		const bool white_king = (s == 0);
		KsChannels ch;
		ks_channels(ch, c, wa, ba, white_king, true);
		out[s]      = ch.n_att;
		out[2 + s]  = ch.w_att;
		out[4 + s]  = ch.weak;
		out[6 + s]  = ch.adj_sq;
		out[8 + s]  = __builtin_popcountll(ch.chk_r) + __builtin_popcountll(ch.chk_q)
		            + __builtin_popcountll(ch.chk_b) + __builtin_popcountll(ch.chk_n);
		out[10 + s] = (c.kings & (white_king ? c.white : c.black)) ? ks_units(c, wa, ba, white_king) : 0;
		out[12 + s] = ch.n_att_x;
		out[14 + s] = ch.adj_inst;
		out[16 + s] = ch.unsafe;
		out[18 + s] = ch.blockers;
		out[20 + s] = ch.flank_att;
		out[22 + s] = ch.flank_def;
		out[24 + s] = ch.knight_def;
		out[26 + s] = ch.contest_excess;
		out[28 + s] = ch.contest_sq;
		out[30 + s] = ch.enemy_queen ? 1 : 0;
		out[32 + s] = ch.w_att_contest;
		out[34 + s] = ch.gate ? 1 : 0;
		// Per-type SAFE-CHECK square counts (appended 2026-09-28 for the joint KS fit): out[8] sums the four types,
		// but the scorer prices each type separately, so a Python reproduction of `units` needs them apart.
		out[36 + s] = __builtin_popcountll(ch.chk_r);
		out[38 + s] = __builtin_popcountll(ch.chk_q);
		out[40 + s] = __builtin_popcountll(ch.chk_b);
		out[42 + s] = __builtin_popcountll(ch.chk_n);
		// Per attacker TYPE (appended 2026-09-28 for Fit K2), pairs from 44: att_n/b/r/q, att_x_n/b/r/q, share_n/b/r/q
		// (sum of 256 x contested footprint share), then w_att_x. With these, w_att == sum W[t] x att_t exactly, so a
		// fit can free each type's weight, and the x-ray / defence-aware modes can be fitted from counts alone.
		for (int t = 0; t < 4; ++t){
			out[44 + 2 * t + s]      = ch.att_t[t];
			out[44 + 2 * (4 + t) + s] = ch.att_x_t[t];
			out[44 + 2 * (8 + t) + s] = ch.share_t[t];
		}
		out[44 + 2 * 12 + s] = ch.w_att_x;
	}
}

/* DETECTOR ORACLE PROBE for slice-3 threats (diagnostic; never called from search).
 *
 * Layout (long long, 16), per side W,B: 0-1 minor victims · 2-3 rook victims · 4-5 king victims · 6-7 hanging ·
 * 8-9 restricted · 10-11 safe-pawn · 12-13 pawn-push · 14 score (Black-positive mp) · 15 phase256.
 * ⚠️ READS every THREAT_V2_* knob and KS_V2_XRAY. ★ Recomputes the counts the same way threats_mp does rather than
 * instrumenting it: the hot path carries no diagnostic branch, and a divergence between the two is itself a finding.
 * ☠️ CONTRACT (documented properly 2026-09-17): leg counts are UNCONDITIONAL -- filled whether or not the leg's knob is
 * on, so a disabled leg still reports what it would contribute; only out[14] (the score) respects the knobs. PAWN_TARGETS
 * is the exception, because it is part of the minor/rook VICTIM definition rather than an on/off leg.
 * ⚠️ An oracle that fills a count only when the knob is on will false-mismatch on every position where a disabled leg has
 * a non-zero detector -- which is exactly what happened on the first run (scores matched, counts did not).
 */
void threats_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
                   uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, 0);
	SideAttacks wa, ba;
	build_side_attacks(wa, c, true);
	build_side_attacks(ba, c, false);
	for (int s = 0; s < 2; ++s){
		const bool     white = (s == 0);
		const uint64_t own   = white ? c.white : c.black;
		const uint64_t them  = white ? c.black : c.white;
		const SideAttacks &us = white ? wa : ba;
		const SideAttacks &th = white ? ba : wa;
		const uint64_t their_np = them & ~c.pawns;
		uint64_t protected_set, weak;
		if (Config::THREAT_V2_GATE == 1){
			const uint64_t poorly = (th.all & ~us.all) | (th.dbl & ~us.dbl & ~us.by[PAWN]);
			weak          = them & poorly & us.all;
			protected_set = them & ~poorly;
		} else {
			protected_set = th.by[PAWN] | (th.dbl & ~us.dbl);
			weak          = them & ~protected_set & us.all;
		}
		const uint64_t defended = their_np & protected_set;
		int n_minor = 0, n_rook = 0;
		uint64_t set = (defended | weak) & (us.by[KNIGHT] | us.by[BISHOP]);
		while (set){
			const uint64_t bit = set & -set; set ^= bit;
			const int v = th_victim(c, bit);
			if (v < 0 || v > 4) continue;
			if (v == 0 && !Config::THREAT_V2_PAWN_TARGETS) continue;
			++n_minor;
		}
		set = weak & us.by[ROOK];
		while (set){
			const uint64_t bit = set & -set; set ^= bit;
			const int v = th_victim(c, bit);
			if (v < 0 || v > 4) continue;
			if (v == 0 && !Config::THREAT_V2_PAWN_TARGETS) continue;
			++n_rook;
		}
		const uint64_t safe   = ~th.all | us.all;
		const uint64_t our_p  = c.pawns & own;
		const uint64_t safe_p = our_p & safe;
		const uint64_t patt   = white ? ps_watt(safe_p) : ps_batt(safe_p);
		const uint64_t empty  = ~c.occupied;
		const uint64_t tp     = c.pawns & them;
		const uint64_t p_att  = white ? ps_batt(tp) : ps_watt(tp);
		uint64_t push = (white ? (our_p << 8) : (our_p >> 8)) & empty;
		const uint64_t rank3 = white ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
		push |= (white ? ((push & rank3) << 8) : ((push & rank3) >> 8)) & empty;
		push &= ~p_att & safe;
		out[s]      = n_minor;
		out[2 + s]  = n_rook;
		out[4 + s]  = __builtin_popcountll(weak & us.by[KING] & ~c.pawns);
		out[6 + s]  = __builtin_popcountll(weak & (~th.all | (their_np & us.dbl)));
		out[8 + s]  = __builtin_popcountll(th.all & ~protected_set & us.all);
		out[10 + s] = __builtin_popcountll(patt & their_np);
		out[12 + s] = __builtin_popcountll((white ? ps_watt(push) : ps_batt(push)) & their_np);
	}
	out[14] = threats_mp(c, wa, ba);
	out[15] = c.phase256;
}

/* DETECTOR ORACLE PROBE for slice-3 space (diagnostic; never called from search).
 *
 * Layout (long long, 6): 0-1 safe-square counts W,B (including the BEHIND double count) · 2-3 the piece counts the
 * weight uses · 4 the Black-positive millipawn score · 5 phase256.
 * ⚠️ READS every SPACE_V2_* knob and KS_V2_XRAY, so the caller must have built an engine under the arm's env.
 * ★ Recomputes the counts the same way space_mp does rather than instrumenting it: space_mp stays branch-free of
 * any diagnostic code, and a divergence between the two is itself a finding the oracle will surface.
 */
void space_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
                 uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, 0);
	SideAttacks wa, ba;
	build_side_attacks(wa, c, true);
	build_side_attacks(ba, c, false);
	for (int s = 0; s < 2; ++s){
		const bool     white   = (s == 0);
		const uint64_t own     = white ? c.white : c.black;
		const uint64_t own_p   = c.pawns & own;
		const uint64_t enemy_p = c.pawns & ~own;
		const SideAttacks &us   = white ? wa : ba;
		const SideAttacks &them = white ? ba : wa;
		const uint64_t region = Config::SPACE_V2_REGION == 1
		                      ? SPACE_ETH_BIG
		                      : (SPACE_CENTRE_FILES & (white ? SPACE_RANKS_W : SPACE_RANKS_B));
		const uint64_t safe = Config::SPACE_V2_SAFE == 1
		                    ? (region & ~them.all & (us.all | own))
		                    : (region & ~own_p & ~(white ? ps_batt(enemy_p) : ps_watt(enemy_p)));
		int count = __builtin_popcountll(safe);
		if (Config::SPACE_V2_BEHIND){
			uint64_t behind = own_p;
			if (white){ behind |= behind >> 8; behind |= behind >> 16; }
			else      { behind |= behind << 8; behind |= behind << 16; }
			count += __builtin_popcountll(safe & behind & ~them.all);
		}
		int pieces = 0;
		for (int t = 0; t < 6; ++t) pieces += white ? c.cnt_white[t] : c.cnt_black[t];
		out[s]     = count;
		out[2 + s] = pieces;
	}
	out[4] = space_mp(c, wa, ba);
	out[5] = c.phase256;
}

/* DETECTOR ORACLE PROBE for slice-2 mobility (diagnostic; never called from search).
 *
 * Layout (long long, 14 entries): 0-3 White area-filtered counts N,B,R,Q · 4-7 Black · 8-9 raw mg W,B ·
 * 10-11 raw eg W,B · 12-13 area masks W,B (as signed 64-bit).
 * ⚠️ READS KNOBS -- KS_V2_XRAY (attack occupancy), MOB_V2_EXCL_* and MOB_V2_PIN (area + pin line), MOB_V2_SAFE (counts)
 * and MOB_V2_TABLE (raw sums) -- so the caller must have built an
 * engine under the arm's environment first. Compared against diagnostics/_mobility_detector_oracle.py.
 */
void mobility_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
                    uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, 0);
	MobAcc mw, mb;
	SideAttacks wa, ba;
	mobility_build(c, wa, ba, mw, mb);
	for (int i = 0; i < 4; ++i){ out[i] = mw.cnt[i]; out[4 + i] = mb.cnt[i]; }
	out[8]  = mw.raw_mg; out[9]  = mb.raw_mg;
	out[10] = mw.raw_eg; out[11] = mb.raw_eg;
	out[12] = (long long)mw.area; out[13] = (long long)mb.area;
}

/* PASSER-CREATION POTENTIAL ("can this side ever make a passer?") — the input POT winnability needs on pawn-only boards
 * (owner, 2026-10-07: the 10-04 loss read +0.7 in a dead draw whose extra pawn was a doubled g-pawn).
 *
 * WHY NOT `candidate[]`. Layer A's candidate is SF11's NARROW test: a pawn ONE exchange from passing (every stopper a
 * lever, or out-supported head-on). The winnability question is structural and longer-range, so this is the CLASSIC
 * candidate (SF 1.x–5 `candidate`, the textbook definition): a pawn with NO enemy pawn ahead on its own file whose
 * HELPERS (own pawns on the adjacent files, at most one rank ahead of it — they can advance level and trade) are at
 * least as many as its SENTRIES (enemy pawns on the adjacent files ahead of it). Equal numbers trade off and leave
 * the pawn free. A pawn with an own pawn ahead on its file (rear-doubled) is never one: only the front pawn can run.
 *
 * Returns, for side s (0 White, 1 Black): bit 0..63 mask of pawns that are a FRONT-MOST passer or a classic candidate.
 * Empty ⇒ this side cannot create a passer from the pawn structure alone (king raids can still change the structure —
 * that is the king-activity input's job, not this one's).
 * Pure, pawn-only; not on any search path today (diagnostic probe), so the engine is byte-identical.
 */
static uint64_t passer_potential(const PawnEntry &e, const V2Context &c, int s) noexcept
{
	const bool white = (s == 0);
	const uint64_t own   = c.pawns & (white ? c.white : c.black);
	const uint64_t enemy = c.pawns & (white ? c.black : c.white);
	uint64_t out = 0;
	for (uint64_t it = own; it; it &= it - 1){
		const int sq = __builtin_ctzll(it);
		const uint64_t m = 1ULL << sq;
		const int f = sq & 7, r = sq >> 3;
		const uint64_t file  = BB_FILES[f];
		const uint64_t ahead = white ? ps_nfill(m << 8) : ps_sfill(m >> 8);       // own file, strictly ahead
		if (ahead & own) continue;                                                 // rear-doubled: only the front runs
		if (e.passed[s] & m){ out |= m; continue; }
		if (ahead & enemy) continue;                                               // opposed: not a candidate
		const uint64_t adj = (f > 0 ? BB_FILES[f - 1] : 0) | (f < 7 ? BB_FILES[f + 1] : 0);
		const uint64_t span = white ? passed_span_white[sq] : passed_span_black[sq];
		const int sentries = __builtin_popcountll(span & ~file & enemy);
		// helpers: adjacent-file own pawns on ranks up to ONE ahead of this pawn (relative), i.e. able to come level
		const uint64_t upto = white ? (r + 2 >= 8 ? ~0ULL : (1ULL << ((r + 2) * 8)) - 1)
		                            : (r - 1 <= 0 ? ~0ULL : ~((1ULL << ((r - 1) * 8)) - 1));
		const int helpers = __builtin_popcountll(adj & upto & own);
		if (helpers >= sentries) out |= m;
	}
	return out;
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
 *   23-24 passed - 25-26 candidate - 27-28 passer_potential (classic candidate | front-most passer)
 *
 * @param pawnsMask  all pawns
 * @param whiteMask  all White occupancy
 * @param blackMask  all Black occupancy
 * @param out        caller-provided, at least 29 entries; fully written
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
		out[27 + s] = passer_potential(e, c, s);
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

// ═════════════════════════════════════════════════════════════════════════════════════════════════
// SLICE 1 -- BINARY DRAW CLASSIFIER
// ═════════════════════════════════════════════════════════════════════════════════════════════════

/* Uncapped Chebyshev (king-move) distance between two squares.
 * ⚠️ Deliberately NOT ps_kdist, which CAPS at 5 because SF caps king_proximity there. The rook-pawn
 * opposition test compares true distances to a promotion square and needs the full 0..7 range; reusing
 * the capped helper would silently equalise every distance above 5 and flag won positions as drawn.
 */
static inline int dv_dist(int a, int b) noexcept
{
	const int dx = ((a & 7) > (b & 7)) ? (a & 7) - (b & 7) : (b & 7) - (a & 7);
	const int dy = ((a >> 3) > (b >> 3)) ? (a >> 3) - (b >> 3) : (b >> 3) - (a >> 3);
	return dx > dy ? dx : dy;
}

/* True iff `sq` is a light square. Matches v1's is_white_square (cpp_bitboard.cpp:1665) exactly. */
static inline bool dv_light(int sq) noexcept { return (((sq & 7) + (sq >> 3)) & 1) != 0; }

// ─── EXACT KPK BITBASE ─────────────────────────────────────────────────────────────────────────────
// SF11 bitbase.cpp's retrograde classification, re-expressed in our square convention (a1 = 0, h8 = 63).
// ★ WHY EXACT, NOT A BETTER HEURISTIC: every lone-pawn race rule we wrote failed the oracle (v1's opposition
// test 6.2% false draws, the tempo-corrected one still 0.6%). A retrograde fixpoint over all 196,608 indices has
// zero false positives BY CONSTRUCTION -- it is the game tree, not an approximation of it.
// ☠️ Uses NO runtime table. BB_KING_ATTACKS is filled by initialize_attack_tables() at engine construction; a
// lazily-built static that read it before then would latch a garbage bitbase for the life of the process
// (memory runtime-tables-are-empty-outside-an-engine-instance). King and pawn attacks are local shifts instead.

static constexpr unsigned KPK_MAX_INDEX = 2 * 24 * 64 * 64;   // stm x pawn(a-d, ranks 2-7) x wk x bk
static constexpr uint8_t  KPK_INVALID = 0, KPK_UNKNOWN = 1, KPK_DRAW = 2, KPK_WIN = 4;
static constexpr uint64_t KPK_FILE_A = 0x0101010101010101ULL, KPK_FILE_H = 0x8080808080808080ULL;

static inline uint64_t kpk_king_att(int sq) noexcept
{
	const uint64_t b = 1ULL << sq;
	const uint64_t e = b & ~KPK_FILE_H, w = b & ~KPK_FILE_A;
	return (b << 8) | (b >> 8) | (e << 1) | (e << 9) | (e >> 7) | (w >> 1) | (w >> 9) | (w << 7);
}

/* White pawn attacks from sq (the strong side is always normalised to White). */
static inline uint64_t kpk_pawn_att(int sq) noexcept
{
	const uint64_t b = 1ULL << sq;
	return ((b & ~KPK_FILE_A) << 7) | ((b & ~KPK_FILE_H) << 9);
}

/* Index layout (SF11 bitbase.cpp:41-47): bits 0-5 wk · 6-11 bk · 12 side to move (0 = strong side) · 13-14 pawn
 * file (a-d) · 15-17 (RANK_7 - pawn rank). `psq` must be on files a-d, ranks 2-7. */
static inline unsigned kpk_index(int us, int bk, int wk, int psq) noexcept
{
	return (unsigned)wk | ((unsigned)bk << 6) | ((unsigned)us << 12)
	     | ((unsigned)(psq & 7) << 13) | ((unsigned)(6 - (psq >> 3)) << 15);
}

struct KpkTable { uint32_t bits[KPK_MAX_INDEX / 32]; };

/* Build the bitbase. ~15 sweeps over 196,608 entries; milliseconds, once per process.
 * Mirrors SF11 KPKPosition::KPKPosition (initial classification) and ::classify (the sweep). */
static void kpk_build(KpkTable &t)
{
	std::vector<uint8_t> db(KPK_MAX_INDEX);

	for (unsigned idx = 0; idx < KPK_MAX_INDEX; ++idx){
		const int wk  = (int)(idx & 0x3F);
		const int bk  = (int)((idx >> 6) & 0x3F);
		const int us  = (int)((idx >> 12) & 1);
		const int psq = (6 - (int)((idx >> 15) & 0x7)) * 8 + (int)((idx >> 13) & 0x3);
		const uint64_t patt = kpk_pawn_att(psq);

		if (dv_dist(wk, bk) <= 1 || wk == psq || bk == psq || (us == 0 && (patt & (1ULL << bk))))
			db[idx] = KPK_INVALID;
		// Immediate win: the pawn promotes and the new queen cannot be taken.
		else if (us == 0 && (psq >> 3) == 6 && wk != psq + 8
		         && (dv_dist(bk, psq + 8) > 1 || (kpk_king_att(wk) & (1ULL << (psq + 8)))))
			db[idx] = KPK_WIN;
		// Immediate draw: the weak side is stalemated, or its king takes an undefended pawn.
		else if (us == 1
		         && (!(kpk_king_att(bk) & ~(kpk_king_att(wk) | patt))
		             || (kpk_king_att(bk) & (1ULL << psq) & ~kpk_king_att(wk))))
			db[idx] = KPK_DRAW;
		else
			db[idx] = KPK_UNKNOWN;
	}

	bool repeat = true;
	while (repeat){
		repeat = false;
		for (unsigned idx = 0; idx < KPK_MAX_INDEX; ++idx){
			if (db[idx] != KPK_UNKNOWN) continue;
			const int wk  = (int)(idx & 0x3F);
			const int bk  = (int)((idx >> 6) & 0x3F);
			const int us  = (int)((idx >> 12) & 1);
			const int psq = (6 - (int)((idx >> 15) & 0x7)) * 8 + (int)((idx >> 13) & 0x3);

			uint8_t r = KPK_INVALID;
			if (us == 0){
				uint64_t b = kpk_king_att(wk);
				while (b){ const int to = __builtin_ctzll(b); b &= b - 1; r |= db[kpk_index(1, bk, to, psq)]; }
				if ((psq >> 3) < 6)                                             // single push
					r |= db[kpk_index(1, bk, wk, psq + 8)];
				if ((psq >> 3) == 1 && psq + 8 != wk && psq + 8 != bk)          // double push
					r |= db[kpk_index(1, bk, wk, psq + 16)];
				db[idx] = (r & KPK_WIN) ? KPK_WIN : (r & KPK_UNKNOWN) ? KPK_UNKNOWN : KPK_DRAW;
			} else {
				uint64_t b = kpk_king_att(bk);
				while (b){ const int to = __builtin_ctzll(b); b &= b - 1; r |= db[kpk_index(0, to, wk, psq)]; }
				db[idx] = (r & KPK_DRAW) ? KPK_DRAW : (r & KPK_UNKNOWN) ? KPK_UNKNOWN : KPK_WIN;
			}
			if (db[idx] != KPK_UNKNOWN) repeat = true;
		}
	}

	for (unsigned idx = 0; idx < KPK_MAX_INDEX; ++idx)
		if (db[idx] == KPK_WIN) t.bits[idx >> 5] |= 1u << (idx & 0x1F);
}

/* True iff the normalised position (strong side White, pawn on files a-d, ranks 2-7) is a WIN.
 * Built on first call through a thread-safe function-local static; never reads Config or a runtime table. */
static bool kpk_is_win(int wk, int psq, int bk, int us)
{
	static const KpkTable *const tbl = []{ KpkTable *t = new KpkTable{}; kpk_build(*t); return t; }();
	const unsigned idx = kpk_index(us, bk, wk, psq);
	return (tbl->bits[idx >> 5] >> (idx & 0x1F)) & 1u;
}

/* K + P vs K, any file: true iff it is a DRAW. Normalises colour (rank flip) and file (a-d) first. */
static bool kpk_drawn(const V2Context &c) noexcept
{
	const bool sw = (c.pawns & c.white) != 0;                       // strong side is White?
	int psq = __builtin_ctzll(c.pawns);
	int sk  = __builtin_ctzll(c.kings & (sw ? c.white : c.black));
	int wkq = __builtin_ctzll(c.kings & (sw ? c.black : c.white));
	if (!sw){ psq ^= 56; sk ^= 56; wkq ^= 56; }                     // mirror ranks so the pawn moves up
	if ((psq & 7) > 3){ psq ^= 7; sk ^= 7; wkq ^= 7; }              // mirror files onto a-d
	const int r = psq >> 3;
	if (r < 1 || r > 6) return false;                               // not a legal pawn -- never flag
	const int us = (sw == c.turn) ? 0 : 1;                          // c.turn true = White to move
	return !kpk_is_win(sk, psq, wkq, us);
}

/* KPK BITBASE PROBE (diagnostic). Normalised inputs: strong side White, pawn on files a-d, ranks 2-7.
 * @param strong_to_move  1 if the pawn's side is on move
 * @return 1 = win, 0 = draw, -1 = inputs outside the normalised domain
 */
int kpk_probe(int wksq, int wpsq, int bksq, int strong_to_move)
{
	if (wksq < 0 || wksq > 63 || bksq < 0 || bksq > 63 || wpsq < 0 || wpsq > 63
	    || (wpsq & 7) > 3 || (wpsq >> 3) < 1 || (wpsq >> 3) > 6)
		return -1;
	return kpk_is_win(wksq, wpsq, bksq, strong_to_move ? 0 : 1) ? 1 : 0;
}

/* Binary draw classifier: true => this position is a dead draw and the eval must return 0 outright.
 *
 * ☠️ MEMBERSHIP HERE IS MEASURED, NOT REASONED. v1's is_practically_drawn carries ten cases; five of them
 * flag positions that are FORCED WINS, at rates of 10-28% (🧰 diagnostics/_draw_oracle.py vs the Lichess
 * 7-piece tablebase, two seeds). Only these survived at 0 false positives in 62 samples each, and they
 * are the only ones here:
 *     KvK / KBvK / KNvK   -- insufficient material, provable by argument
 *     KBvKB, KNvKN        -- equal count, only minors of one type
 *     K + lone rook pawn vs K -- ALSO validated on _kpk_oracle.py (83,238 states, 0 false draws)
 *     bishop vs a lone rook pawn it cannot stop
 * ☠️ NOT here, and deliberately: R+B-vs-R (28% FP) · R+N-vs-R (22%) · bare-R-vs-bare-minor (24/28%) ·
 * wrong-coloured-bishop + rook pawn (10%). Those are "usually drawn, sometimes won" -- a MAGNITUDE, which
 * a bool cannot express. They belong in the convertibility scale. See EVAL-V2-SLICE1-DRAW-DESIGN.md §1.
 * ★ The rule this obeys is the owner's, from June: a won position flagged drawn is CATASTROPHIC, while a
 * missed draw only forfeits an opportunity -- so the target is NO FALSE POSITIVES, never coverage.
 *
 * ⚠️ NARROWER THAN v1 ON PURPOSE: the `n_nk > 2` early-out means multi-minor equal-material endings
 * (KBBvKBB, KNNvKNN) are NOT flagged, where v1's equal-count rule would flag them. The oracle sweep never
 * generated those, so they are unmeasured -- and an unmeasured case must fail in the direction that only
 * forfeits a draw, never the direction that discards a win. Widen it only with oracle evidence.
 *
 * Pure; reads only the context's bitboards. Cost: a handful of popcounts, and every case is guarded by a
 * piece-count test that fails immediately in any normal position.
 */
static bool draw_class(const V2Context &c) noexcept
{
	const uint64_t nk = c.occupied & ~c.kings;

	if (nk == 0) return true;                                              // KvK

	const int n_nk = __builtin_popcountll(nk);
	if (n_nk > 2) return false;                                            // nothing below can fire

	const int n_b = __builtin_popcountll(c.bishops);
	const int n_n = __builtin_popcountll(c.knights);
	const int n_p = __builtin_popcountll(c.pawns);

	if (n_b == 1 && nk == c.bishops) return true;                          // KBvK
	if (n_n == 1 && nk == c.knights) return true;                          // KNvK

	// Equal total piece count with only bishops, or only knights, off the board: KBvKB / KNvKN.
	if (__builtin_popcountll(c.white) == __builtin_popcountll(c.black) &&
	    (nk == c.bishops || nk == c.knights))
		return true;

	// ── Reference-derived additions (2026-09-13), each oracle-checked BEFORE this code was written ─────
	// Tablebase sampling, uniform + corner-biased (_draw_oracle.py EDGE=1), two seeds each:
	//     KBvKN 0/400 · KNNvK 0/400 · SF fortress wrong-bishop 0/286 -- zero short AND zero long false positives.
	// ⚠️ These pass the PROPOSED DTM-weighted gate (EVAL-V2-SLICE1-DRAW-DESIGN.md §2d), not literal zero-FP: a
	// constructed KNvKN mate-in-1 (6nk/8/6K1/4N3/8/8/8/8 w) proves rare short wins exist that sampling misses,
	// and search was shown to still find that mate with this rule scoring 0.

	// KB vs KN, opposite sides, nothing else (Weiss TrivialDraw; SF's generic material rule also zeroes it).
	if (n_b == 1 && n_n == 1 && nk == (c.bishops | c.knights) &&
	    ((c.bishops & c.white) != 0) != ((c.knights & c.white) != 0))
		return true;

	// KNN vs bare K. Every reference draws it: SF's named handler `Endgame<KNNK>` returns VALUE_DRAW (SF11
	// endgame.cpp:329, SF15.1 :313) and named handlers override the material scale factor; Weiss TrivialDraw.
	// Tablebase 0/400, and two knights cannot force mate against correct defence.
	// ⚠️ Corrected 2026-09-13: an earlier comment here said SF "scales it to 4/64" -- that read only the generic
	// material rule and missed that the named KNNK handler takes precedence.
	if (n_n == 2 && nk == c.knights &&
	    ((c.knights & c.white) == 0 || (c.knights & c.white) == c.knights))
		return true;

	// Wrong-coloured bishop + rook pawn, in SF KBPsK's FORTRESS form (stockfish_11 endgame.cpp:356): the
	// defending king is ALREADY within one square of the queening corner. ☠️ Deliberately NOT v1's race test
	// (defender_dist <= min(...)), which measured 10% false positives -- a race can be lost to tempo and
	// interference; a fortress already reached cannot. Static condition over dynamic race, always.
	if (n_b == 1 && n_p == 1 && nk == (c.bishops | c.pawns) &&
	    ((c.bishops & c.white) != 0) == ((c.pawns & c.white) != 0) &&
	    (c.pawns & (BB_FILE_A | BB_FILE_H))) {
		const bool sw  = (c.pawns & c.white) != 0;
		const int  psq = __builtin_ctzll(c.pawns);
		const int  qsq = (sw ? 56 : 0) + (psq & 7);
		const int  bsq = __builtin_ctzll(c.bishops);
		const int  dks = __builtin_ctzll(c.kings & (sw ? c.black : c.white));
		if (dv_light(bsq) != dv_light(qsq) && dv_dist(dks, qsq) <= 1)
			return true;
	}

	// ── lone-pawn cases: SEPARATE KNOB, DEFAULT OFF ─────────────────────────────────────────────
	// ☠️ These are NOT clean and must not ride on DRAW_V2_CLASS. v1's chebyshev-opposition test ignores
	// WHOSE MOVE IT IS and measured 6.2% false positives on KPvK (🧰 _draw_oracle.py). Adding the tempo
	// term below cuts that to 0.6% over 320 samples across 4 seeds -- a 10x improvement, and still not
	// zero (`8/8/8/8/8/PK1k4/8/8 b`: defender equidistant AND on move, and still lost).
	// ⚠️ The June note "validated against a full KPvK retrograde oracle: no won position is flagged drawn"
	// does NOT hold for the rule as shipped. Whatever that oracle checked, it was not this condition.
	// ★ THE REAL FIX IS AN EXACT KPK BITBASE, not a better heuristic -- zero false positives by
	// construction, and it covers ALL of KPvK rather than just rook pawns. SF ships one
	// (stockfish_11/src/bitbase.cpp, ~24KB packed, built at init in milliseconds) and we already own the
	// retrograde tooling (diagnostics/_kpk_oracle.py, 83,238 states). Until that exists this stays OFF.
	// K + P vs K from the EXACT bitbase. Checked before the heuristic so that, when on, it owns this case outright.
	if (Config::DRAW_V2_KPK_EXACT && n_p == 1 && nk == c.pawns) return kpk_drawn(c);

	if (!Config::DRAW_V2_KPK) return false;
	if (n_p != 1) return false;

	const int  psq   = __builtin_ctzll(c.pawns);
	const bool p_w   = (c.pawns & c.white) != 0;
	const bool file_a = (c.pawns & BB_FILE_A) != 0;
	if (!file_a && !(c.pawns & BB_FILE_H)) return false;                   // rook pawns only

	const int promo = file_a ? (p_w ? 56 : 0) : (p_w ? 63 : 7);
	const int wk    = __builtin_ctzll(c.kings & c.white);
	const int bk    = __builtin_ctzll(c.kings & c.black);
	const int atk   = p_w ? wk : bk;                                       // the pawn's side
	const int dfd   = p_w ? bk : wk;
	const int d_atk = dv_dist(atk, promo), d_p = dv_dist(psq, promo);
	// ☠️ TEMPO TERM -- absent from v1, and the single largest false-positive source there. If the PAWN'S
	// side is on move it gains a tempo, so the defender needs one square more in hand. Measured: 6.2% ->
	// 0.6% false positives. This is the one place the eval genuinely needs `turn`.
	const bool attacker_to_move = (p_w == c.turn);
	const int  d_dfd = dv_dist(dfd, promo) + (attacker_to_move ? 1 : 0);
	const bool holds = d_dfd <= (d_p < d_atk ? d_p : d_atk);

	// K + lone rook pawn vs K, defender holding the promotion corner.
	if (nk == c.pawns) return holds;

	// Bishop vs a lone rook pawn: the bishop's side defends, and only when it is the pawn's opponent.
	if (n_nk == 2 && n_b == 1 && ((c.bishops & c.white) != 0) != p_w) {
		const int bsq = __builtin_ctzll(c.bishops);
		if (dv_light(bsq) != dv_light(promo) && holds) return true;
	}
	return false;
}

/* Slice 4 tier-2b -- TECHNIQUE VALUE for K+R vs K+minor (pawnless), Black-positive millipawns.
 *
 * What: for K+R vs K+B and K+R vs K+N with no pawns, DISCARD the material lead and return a pure technique
 * gradient: drive the weak king to the edge, and for K+R vs K+N also drive it away from its own knight.
 * Returns true and sets `out` when it applies; false otherwise. A REPLACEMENT, not a term added to a sum.
 *
 * Why: ★ these endings are normally DRAWN with correct defence, yet v2 currently returns the ordinary eval,
 * which reads a rook up as roughly +1550 mp. **That is an OVER-READ, and over-reads are how an engine trades
 * INTO a dead ending believing it is winning** -- the exact failure that motivated the KPvK rook-pawn fix
 * (a real loss where we evaluated a dead draw at +4870). ⚠️ NOT the same defect as v1's: v1 hard-ZEROES these
 * (`is_practically_drawn` cases 8-10) at 22-28% tablebase FALSE POSITIVES; v2 dropped those rules, so v2 has
 * no false draw here -- it has no technique gradient and an inflated magnitude.
 * SF discards the material for exactly this reason (`stockfish_11/src/endgame.cpp:241-263`): a rook up is worth
 * nothing here EXCEPT the ability to drive the king to the edge. ⚠️ Single-lineage (SF only) -- Ethereal and
 * Weiss leave these to search entirely -- so this is a CANDIDATE, not a consensus adoption.
 *
 * Form: SF15.1's formulas rather than SF11's tables (`evaluate`-side, endgame.cpp:32-45):
 *   push_to_edge(s) = 90 - (7*fd*fd/2 + 7*rd*rd/2), fd/rd = distance from the NEAREST edge => corner 90, centre 28
 *   push_close(d)   = 140 - 20*d ; push_away(d) = 120 - push_close(d)   [K+R vs K+N only]
 * UNITS: SF endgame units, its eg pawn = 213 (types.h) against our 1000 => 1 unit ~ 4.7 mp. TIER2_V2_MAG is a
 * PERCENT of SF's own scale, so 100 == SF's magnitude expressed in our millipawns (corner ~423 mp, centre ~131).
 * ★ MATERIAL-anchored like Kaufman, so the pawn is the right unit -- not v2's 5-35 mp positional spread.
 *
 * Gating: caller checks TIER2_V2_MAG > 0, so 0 = absent = byte-identical.
 * ⚠️ Deliberately NOT extended to K+R+B vs K+R / K+R+N vs K+R: SF handles THOSE with a SCALE (~14/64), a
 * different mechanism, and they are step two.
 * Cost: a handful of popcounts on a path already guarded by draw_class's piece-count early-outs.
 */
static inline int t2_edge_dist(int x) noexcept { return x < 7 - x ? x : 7 - x; }

static bool tier2b_value_mp(const V2Context &c, int &out) noexcept
{
	if (c.pawns) return false;
	const uint64_t minors = c.knights | c.bishops;
	if (c.queens || (c.rooks & c.knights) || __builtin_popcountll(c.rooks) != 1) return false;
	if (__builtin_popcountll(minors) != 1) return false;

	const bool rook_white = (c.rooks & c.white) != 0;
	// The rook side must have ONLY the rook; the other side ONLY the minor (plus kings).
	const uint64_t strong = rook_white ? c.white : c.black;
	const uint64_t weak   = rook_white ? c.black : c.white;
	if ((strong & ~(c.kings | c.rooks)) || (weak & ~(c.kings | minors))) return false;

	const int weak_ksq = __builtin_ctzll(c.kings & weak);
	const int fd = t2_edge_dist(weak_ksq & 7), rd = t2_edge_dist(weak_ksq >> 3);
	int units = 90 - (7 * fd * fd / 2 + 7 * rd * rd / 2);
	if (c.knights){                                   // K+R vs K+N: also split the king from its knight
		const int nsq  = __builtin_ctzll(c.knights);
		const int dfile = (nsq & 7) - (weak_ksq & 7), drank = (nsq >> 3) - (weak_ksq >> 3);
		const int af = dfile < 0 ? -dfile : dfile, ar = drank < 0 ? -drank : drank;
		units += 120 - (140 - 20 * (af > ar ? af : ar));
	}
	// SF units -> our millipawns, as a percent of SF's own scale. Black-positive: negate when White is strong.
	const int mp = units * Config::TIER2_V2_MAG * 10 / 213;
	out = rook_white ? -mp : mp;
	return true;
}

/*
	See the file header for the output and purity contract. Rung selection is Config::EVAL_V2_RUNG; rungs
	are cumulative, so a rung adds to everything below it rather than replacing it.
*/
/* ═══ C3-a: KING SHELTER + PAWN STORM (KS-B) ═══════════════════════════════════════════════════════════════════════
 * Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §3 (cells) and §8 (ownership: storm-against-a-king lives
 * HERE, never in OvD, which is king-free).
 *
 * WHAT: for one king, the pawns of a 3-file window in front of it, as one-hot CELLS -- a direct score, separate from
 * ks_units. Per file of the window (centre clamped to b..g, so an edge king still gets three files):
 *   shelter = our pawn NEAREST the king at or ahead of its rank, as the rank gap d:
 *             beside (d 0) · 1 · 2 · 3 · >= 4 · lever-attacked (any d; an enemy pawn attacks it)
 *   storm   = their pawn NEAREST the king at or ahead of its rank, as the rank gap d:
 *             unblocked d <= 1 · 2 · 3 · 4 · >= 5 · blocked (our pawn directly in front of it) d <= 2 · 3 · >= 4
 *   "no pawn" is the reference state and has no cell (pinned to 0).
 * Cell = shelter F*6 + state (0..23), storm 24 + F*8 + state (24..55), F = min(file, 7 - file). Value is to the king's
 * OWN side, so a good shelter is a positive cell value.
 *
 * WHY: 5/5 references have shelter and 4/5 a storm, every one of them as a king-safety component beside the king
 * (SF11 pawns.cpp:186-215, SF15.1 pawns.cpp:236-260, Ethereal evaluateKingsPawns). v2 had neither -- the recall study
 * (§6a) found KS misses are structural: shelter <= 1 (x2.2), >= 2 semi-open files (x3.3), storm (x2.1).
 * Divergences chosen here, each for the mirror gate or the fit: the file index is the edge-distance CLASS (Ethereal's
 * absolute file would fail our file-mirror gate); an attacked shelter pawn is its OWN state rather than silently
 * dropped (SF15.1 drops it); the cells are left free for the Texel fit rather than hand-set.
 *
 * Gating: the whole term, detector included, runs only when g_ksb_on (Config::KSB_V2 = 1 with a loaded
 * KSB_V2_FILE); otherwise byte-identical. v2_features reports the cells UNCONDITIONALLY, which is what the fit reads.
 * Cost: 3 files x two masked bit-scans per king; pawn- and king-only, so a future pawn-king cache could hold it.
 *
 * @param own, enemy  the king's side's pawns and the other side's pawns
 * @param white       the king's colour
 * @param ksq         the king square (or a castling target square, for the castling max)
 * @param cells       output, room for 6
 * @return            the number of cells written
 */
[[nodiscard]] static inline int ksb_cells(uint64_t own, uint64_t enemy, bool white, int ksq, int *cells) noexcept
{
	const int kf = ksq & 7;
	const int krank = ksq >> 3;
	const int kr = white ? krank : 7 - krank;                         // relative rank, 0 = own back rank
	const int centre = kf < 1 ? 1 : (kf > 6 ? 6 : kf);
	const uint64_t eatt  = white ? ps_batt(enemy) : ps_watt(enemy);
	// Ranks at or AHEAD of the king, from its own side's view.
	const uint64_t ahead = white ? (~0ULL << (8 * krank)) : (~0ULL >> (8 * (7 - krank)));
	int n = 0;
	for (int f = centre - 1; f <= centre + 1; ++f){
		const int F = f < 7 - f ? f : 7 - f;
		const uint64_t fm = BB_FILE_A << f;
		const uint64_t o = own & fm & ahead, t = enemy & fm & ahead;
		int orr = -1;
		if (o){
			// Nearest to our king: the LOWEST square for White, the HIGHEST for Black (one file, so rank order).
			const int sq = white ? __builtin_ctzll(o) : 63 - __builtin_clzll(o);
			orr = white ? (sq >> 3) : 7 - (sq >> 3);
			const int d = orr - kr;
			const int st = ((eatt >> sq) & 1) ? 5 : (d >= 4 ? 4 : d);
			cells[n++] = F * 6 + st;
		}
		if (t){
			const int sq = white ? __builtin_ctzll(t) : 63 - __builtin_clzll(t);
			const int trr = white ? (sq >> 3) : 7 - (sq >> 3);
			const int d = trr - kr;
			int st;
			if (orr >= 0 && orr == trr - 1) st = d <= 2 ? 5 : (d == 3 ? 6 : 7);     // blocked (rammed)
			else                            st = d <= 1 ? 0 : (d >= 5 ? 4 : d - 1);  // unblocked
			cells[n++] = 24 + F * 8 + st;
		}
	}
	return n;
}

/* One side's shelter + storm legs at one king square, (mg, eg) millipawns, value to that side. */
static inline void ksb_legs(uint64_t own, uint64_t enemy, bool white, int ksq, int &mg, int &eg) noexcept
{
	int cells[6];
	const int n = ksb_cells(own, enemy, white, ksq, cells);
	mg = eg = 0;
	for (int i = 0; i < n; ++i){ mg += ksb_w[0][cells[i]]; eg += ksb_w[1][cells[i]]; }
}

/* One side's shelter + storm, (mg, eg) legs at the square it is scored at. With KSB_V2_CASTLE = 1 a king that still has
 * a castling right is scored at the BEST of its square and the castling targets it may still reach (SF11
 * pawns.cpp:233-237 takes the max by mg; here the max is by the phase-blended value, since v2 blends per term).
 * ⚠️ The feature extractor reports the ACTUAL square only, so it flags mode 1 as unmodelled (flag 4). */
static inline void ksb_side(const V2Context &c, bool white, int &mg, int &eg) noexcept
{
	const uint64_t kb = c.kings & (white ? c.white : c.black);
	mg = eg = 0;
	if (!kb) return;
	const int ksq = __builtin_ctzll(kb);
	const uint64_t own = c.pawns & (white ? c.white : c.black), enemy = c.pawns & (white ? c.black : c.white);
	ksb_legs(own, enemy, white, ksq, mg, eg);
	if (Config::KSB_V2_CASTLE == 0) return;
	const int back = white ? 0 : 56;
	const bool ks = (c.castling_rights >> (back + 7)) & 1, qs = (c.castling_rights >> back) & 1;
	if (!ks && !qs) return;
	int best = (mg * c.phase256 + eg * (256 - c.phase256)) >> 8;
	for (int k = 0; k < 2; ++k){
		if (!(k == 0 ? ks : qs)) continue;
		int m2, e2;
		ksb_legs(own, enemy, white, back + (k == 0 ? 6 : 2), m2, e2);
		const int v = (m2 * c.phase256 + e2 * (256 - c.phase256)) >> 8;
		if (v > best){ best = v; mg = m2; eg = e2; }
	}
}

/* Chebyshev (king-move) distance, UNCAPPED (ps_kdist caps at 5; KingProtector bins run to 6+). */
[[nodiscard]] static inline int c3_cheb(int a, int b) noexcept
{
	const int dx = (a & 7) - (b & 7), dy = (a >> 3) - (b >> 3);
	const int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
	return ax > ay ? ax : ay;
}

/* SF's KingFlank: the files a king on file f "lives on" (SF11 bitboard.h; QueenSide = a-d, CenterFiles = c-f,
 * KingSide = e-h, the edge files trimmed by one). File-mirror symmetric: flank[7 - f] is flank[f] reflected. */
static constexpr uint64_t C3_FILE_A = 0x0101010101010101ULL;
static constexpr uint64_t C3_FLANK[8] = {
	C3_FILE_A * 0x07, C3_FILE_A * 0x0F, C3_FILE_A * 0x0F, C3_FILE_A * 0x3C,
	C3_FILE_A * 0x3C, C3_FILE_A * 0xF0, C3_FILE_A * 0xF0, C3_FILE_A * 0xE0};

/* ═══ C3-b: PAWNLESS FLANK + KING-TO-PAWN DISTANCE ══════════════════════════════════════════════════════════════════
 * Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §3.
 * WHAT: for one king, one-hot cells (value to the king's own side):
 *   0-3  Chebyshev distance to the NEAREST OWN pawn: 2 · 3 · 4 · >= 5   (1 is the reference; no own pawn = no cell)
 *   4-7  the same to the nearest ENEMY pawn
 *   8    no pawn of EITHER colour on the king's flank (SF PawnlessFlank)
 *   9    only ENEMY pawns on the king's flank
 * WHY: king races with NO passer -- v2 prices king-to-pawn distance only through passers. SF11 has minPawnDist (own
 * pawns, eg) and PawnlessFlank (both colours, mg+eg); Ethereal and Weiss price king-pawn distance in the endgame too.
 * Splitting own from enemy is ours: it separates "guarding my pawns" from "attacking theirs". The fit is expected to
 * leave the mg leg near 0 for 0-7; both legs are exposed so the fit, not us, decides.
 * Gating: runs only when g_kfl_on (KFL_V2 = 1 with a loaded KFL_V2_FILE). Cost: one pass over each side's pawns.
 * @return the number of cells written (<= 4)
 */
[[nodiscard]] static inline int kfl_cells(const V2Context &c, bool white, int *cells) noexcept
{
	const uint64_t kb = c.kings & (white ? c.white : c.black);
	if (!kb) return 0;
	const int ksq = __builtin_ctzll(kb);
	const uint64_t own = c.pawns & (white ? c.white : c.black), enemy = c.pawns & (white ? c.black : c.white);
	int n = 0;
	for (int e = 0; e < 2; ++e){
		uint64_t it = e ? enemy : own;
		if (!it) continue;
		int d = 8;
		for (; it; it &= it - 1){
			const int dd = c3_cheb(ksq, __builtin_ctzll(it));
			if (dd < d) d = dd;
		}
		if (d >= 2) cells[n++] = e * 4 + (d >= 5 ? 3 : d - 2);
	}
	const uint64_t flank = C3_FLANK[ksq & 7];
	if (!(c.pawns & flank))  cells[n++] = 8;
	else if (!(own & flank)) cells[n++] = 9;
	return n;
}

/* ═══ C3-c: KINGPROTECTOR (minors only) ═════════════════════════════════════════════════════════════════════════════
 * Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §3; owner call: MINORS ONLY (rooks and queens defend from
 * range, so their defence belongs to the KS defender channels).
 * WHAT: per own knight and bishop, its Chebyshev distance to OUR king, one-hot 1..6+ (cells 0-5 knight, 6-11 bishop),
 * value to the minor's own side. Counts are per piece, so two knights at distance 2 put 2 in one cell.
 * WHY: every reference that has it prices minors only (SF11 KingProtector, linear in distance from 1; Ethereal nothing
 * below 4, to the nearer king). One free cell per distance lets the fit choose between the two shapes.
 * Gating: runs only when g_kprot_on (KPROT_V2 = 1 with a loaded KPROT_V2_FILE). Cost: one pass over the minors.
 * @param cnt  output, KPROT_CELLS counts, overwritten
 */
static inline void kprot_counts(const V2Context &c, bool white, int *cnt) noexcept
{
	for (int i = 0; i < KPROT_CELLS; ++i) cnt[i] = 0;
	const uint64_t side = white ? c.white : c.black;
	const uint64_t kb = c.kings & side;
	if (!kb) return;
	const int ksq = __builtin_ctzll(kb);
	for (int t = 0; t < 2; ++t)
		for (uint64_t it = (t ? c.bishops : c.knights) & side; it; it &= it - 1){
			const int d = c3_cheb(ksq, __builtin_ctzll(it));
			++cnt[t * 6 + (d >= 6 ? 5 : d - 1)];
		}
}

/* One side's C3-b legs, (mg, eg) millipawns, value to that side. */
static inline void kfl_side(const V2Context &c, bool white, int &mg, int &eg) noexcept
{
	int cells[4];
	const int n = kfl_cells(c, white, cells);
	mg = eg = 0;
	for (int i = 0; i < n; ++i){ mg += kfl_w[0][cells[i]]; eg += kfl_w[1][cells[i]]; }
}

/* One side's C3-c legs, (mg, eg) millipawns, value to that side. */
static inline void kprot_side(const V2Context &c, bool white, int &mg, int &eg) noexcept
{
	int cnt[KPROT_CELLS];
	kprot_counts(c, white, cnt);
	mg = eg = 0;
	for (int i = 0; i < KPROT_CELLS; ++i){ mg += cnt[i] * kprot_w[0][i]; eg += cnt[i] * kprot_w[1][i]; }
}

/* Black-positive blended value of a per-side (mg, eg) scorer, plus its raw legs for pair mode. Blended PER SIDE, as
 * KS-A and C3-a are, so the colour mirror sees the same truncation on both sides. */
/* ═══ PX: THE PASSER SYSTEM (2026-10-03; dev_notes/PASSER-SYSTEM-DESIGN-2026-10-03.md) ════════════════════════════
 * The passer DYNAMICS every reference prices and v2 did not (11 engines / 7 lineages): stop-square state, path, support,
 * king escort, pieces behind, file, square rule. Counts only — every cell is a Texel-fitted (mg, eg) pair, so the fit
 * decides sign and size (owner: "find what works for OUR engine"); all start at 0. Applies to TRUE passers (candidates
 * stay with passer_value_mp). Rank cells cover relative ranks 4-7 (0-based r = 3..6, ri = r - 3).
 *   0-3   blocked by an OWN piece [ri]      4-7   blocked by an enemy MINOR [ri]   8-11  blocked by an enemy R/Q [ri]
 *   12-15 blocked by the enemy KING [ri]    16-19 stop free AND not enemy-attacked [ri]
 *   20-23 whole path to promotion free of enemy pieces and enemy attacks [ri] (nested in 16-19)
 *   24-27 stop square attacked by us [ri]   28-31 passer defended by an own pawn [ri]  32-35 phalanx passer [ri]
 *   36-39 OUR king's distance to the stop [ri]   40-43 THEIR king's distance to the stop [ri]   (ps_kdist, capped 5)
 *   44-46 our king's distance to the square AFTER the stop (SF's second push), r = 3..5
 *   47 own R/Q behind (line of sight)   48 enemy R/Q behind (line of sight)
 *   49 file distance from the nearest edge (0 = a/h)   50 square rule: the defender has NO non-pawn material and its
 *      king (exact Chebyshev) is more than one step outside the pawn's square — stm-free, so it never needs the tempo.
 * Blockade by piece TYPE (4-15) is the owner's invented split (no reference has it) — at 0 like everything else.
 * Attack maps: the SAME shared maps passer_value_mp and mobility use (x-ray occupancy, see the ladder note above).
 * Mirror-safe: every count is relative to the side, so a colour mirror swaps the per-side vectors exactly. */
static constexpr int PX_BLK_OWN = 0, PX_BLK_MINOR = 4, PX_BLK_HEAVY = 8, PX_BLK_KING = 12, PX_FREE_SAFE = 16,
                     PX_PATH_FREE = 20, PX_STOP_DEF = 24, PX_PAWN_DEF = 28, PX_PHALANX = 32, PX_KD_US = 36,
                     PX_KD_THEM = 40, PX_KD2_US = 44, PX_RQB_OWN = 47, PX_RQB_THEM = 48, PX_FILE = 49, PX_SQUARE = 50;
static_assert(PX_SQUARE + 1 == PX_CELLS, "PX layout must fill PX_CELLS");

static inline void px_counts(const V2Context &c, const PawnEntry &pe, const SideAttacks &wa, const SideAttacks &ba,
                             bool white, int *cnt) noexcept
{
	for (int i = 0; i < PX_CELLS; ++i) cnt[i] = 0;
	const int s = white ? 0 : 1;
	const uint64_t own = white ? c.white : c.black, theirs = white ? c.black : c.white;
	const uint64_t our_att = white ? wa.all : ba.all, their_att = white ? ba.all : wa.all;
	const uint64_t ownPawns = c.pawns & own;
	const uint64_t okb = c.kings & own, tkb = c.kings & theirs;
	const int ourK = okb ? __builtin_ctzll(okb) : -1, theirK = tkb ? __builtin_ctzll(tkb) : -1;
	const int their_npm = white ? c.npm_black : c.npm_white;
	const int step = white ? 8 : -8;
	uint64_t bb = pe.passed[s];
	while (bb){
		const int sq = __builtin_ctzll(bb);
		bb &= bb - 1;
		const uint64_t m = 1ULL << sq;
		const int f = sq & 7;
		const int r = white ? (sq >> 3) : (7 - (sq >> 3));
		cnt[PX_FILE] += f < 7 - f ? f : 7 - f;
		// pieces behind on the file, first one met (line of sight)
		for (int t = sq - step; t >= 0 && t < 64; t -= step){
			const uint64_t tb = 1ULL << t;
			if (!(c.occupied & tb)) continue;
			if ((c.rooks | c.queens) & tb) ++cnt[(own & tb) ? PX_RQB_OWN : PX_RQB_THEM];
			break;
		}
		// square rule (defender without pieces); exact Chebyshev to the promotion square
		if (their_npm == 0 && theirK >= 0){
			const int promo = white ? (56 + f) : f;
			const int dx = (theirK & 7) > f ? (theirK & 7) - f : f - (theirK & 7);
			const int dy = (theirK >> 3) > (promo >> 3) ? (theirK >> 3) - (promo >> 3) : (promo >> 3) - (theirK >> 3);
			const int kd = dx > dy ? dx : dy;
			const int steps = (r == 1) ? 5 : 7 - r;              // a pawn on its 2nd rank may double-push
			if (kd - 1 > steps) ++cnt[PX_SQUARE];
		}
		if (r < 3) continue;
		const int ri = r - 3;
		const int stop = sq + step;
		const uint64_t sb = 1ULL << stop;
		if (c.occupied & sb){
			if (own & sb)                           ++cnt[PX_BLK_OWN + ri];
			else if (c.kings & sb)                  ++cnt[PX_BLK_KING + ri];
			else if ((c.knights | c.bishops) & sb)  ++cnt[PX_BLK_MINOR + ri];
			else                                    ++cnt[PX_BLK_HEAVY + ri];
		} else if (!(their_att & sb)){
			++cnt[PX_FREE_SAFE + ri];
			const uint64_t toQueen = (white ? passed_span_white[sq] : passed_span_black[sq]) & BB_FILES[f];
			if (!(toQueen & theirs) && !(toQueen & their_att)) ++cnt[PX_PATH_FREE + ri];
		}
		if (our_att & sb) ++cnt[PX_STOP_DEF + ri];
		if (ownPawns & (white ? ps_batt(m) : ps_watt(m))) ++cnt[PX_PAWN_DEF + ri];
		if (ownPawns & (ps_east(m) | ps_west(m)))         ++cnt[PX_PHALANX + ri];
		if (ourK >= 0)   cnt[PX_KD_US + ri]   += ps_kdist(ourK, stop);
		if (theirK >= 0) cnt[PX_KD_THEM + ri] += ps_kdist(theirK, stop);
		if (r <= 5 && ourK >= 0) cnt[PX_KD2_US + ri] += ps_kdist(ourK, stop + step);
	}
}

template <typename SideFn>
static inline int c3_block_mp(const V2Context &c, SideFn fn, int &dmg, int &deg) noexcept
{
	int wmg, weg, bmg, beg;
	fn(c, true, wmg, weg);
	fn(c, false, bmg, beg);
	dmg = bmg - wmg;
	deg = beg - weg;
	return ((bmg * c.phase256 + beg * (256 - c.phase256)) >> 8)
	     - ((wmg * c.phase256 + weg * (256 - c.phase256)) >> 8);
}

/* ═══ OvD eg leg: WINNABILITY inputs ════════════════════════════════════════════════════════════════════════════════
 * Design: dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md §8 + the pilot in §11 (low-complexity endgames: the leader
 * scores 72.6% where the eval predicts ~80%; v2 had no term for it -- the endgame could only say "draw" or full value).
 * WHAT: SF11's complexity inputs, in its colour-SYMMETRIC form (SF15.1's signed outflanking would fail our mirror gate):
 *   in[0] passed pawns (both sides, v2's own passer masks) · in[1] pawns · in[2] outflanking = file distance − rank
 *   distance of the kings · in[3] infiltration (a king past the middle) · in[4] pawns on both flanks · in[5] pure pawn
 *   ending · in[6] almost unwinnable (no passers, outflanking < 0, one flank).
 * Purely board-derived and symmetric, so C = Σ w·in is colour-invariant and the applied adjustment (× sign(total)) is
 * antisymmetric. Cost: a handful of popcounts, once per eval, only when WIN_V2 is on.
 */
static constexpr uint64_t WIN_QS = 0x0F0F0F0F0F0F0F0FULL, WIN_KS = 0xF0F0F0F0F0F0F0F0ULL;

static inline void win_inputs(const V2Context &c, const PawnEntry &pe, int *in) noexcept
{
	const int wk = (c.kings & c.white) ? __builtin_ctzll(c.kings & c.white) : 0;
	const int bk = (c.kings & c.black) ? __builtin_ctzll(c.kings & c.black) : 0;
	const int fd = (wk & 7) > (bk & 7) ? (wk & 7) - (bk & 7) : (bk & 7) - (wk & 7);
	const int rd = (wk >> 3) > (bk >> 3) ? (wk >> 3) - (bk >> 3) : (bk >> 3) - (wk >> 3);
	in[0] = __builtin_popcountll(pe.passed[0] | pe.passed[1]);
	in[1] = __builtin_popcountll(c.pawns);
	in[2] = fd - rd;
	in[3] = ((wk >> 3) > 3 || (bk >> 3) < 4) ? 1 : 0;
	in[4] = ((c.pawns & WIN_QS) && (c.pawns & WIN_KS)) ? 1 : 0;
	in[5] = (c.npm_white + c.npm_black == 0) ? 1 : 0;
	in[6] = (in[0] == 0 && in[2] < 0 && !in[4]) ? 1 : 0;
}

/* The sign-preserving winnability adjustment for a Black-positive total: C = Σ w·in + BASE (mp), endgame-weighted by
 * phase; the leader's score moves by max(C·eg, −|total|), so it can reach 0 but never flip. */
static inline int win_adjust(int total, const int *in, int phase256) noexcept
{
	if (total == 0) return 0;
	const int C = Config::WIN_V2_PASSED * in[0] + Config::WIN_V2_PAWNS * in[1] + Config::WIN_V2_OUTFLANK * in[2]
	            + Config::WIN_V2_INFILT * in[3] + Config::WIN_V2_FLANKS * in[4] + Config::WIN_V2_PAWN_END * in[5]
	            + Config::WIN_V2_UNWIN * in[6] + Config::WIN_V2_BASE;
	const int v = C * (256 - phase256) / 256;
	const int mag = total > 0 ? total : -total;
	int d = v > -mag ? v : -mag;
	// WIN_V2_CAP (2026-09-30): bound the adjustment to ±CAP mp. The d6-outcome fit (swings to ~1.7 pawns) hurt at depth;
	// the SF18-label fit is priced under a ½-pawn cap. 0 = uncapped = the previous behaviour.
	if (Config::WIN_V2_CAP > 0){
		if (d > Config::WIN_V2_CAP) d = Config::WIN_V2_CAP;
		if (d < -Config::WIN_V2_CAP) d = -Config::WIN_V2_CAP;
	}
	return total > 0 ? d : -d;
}

/* ═══ POT winnability, REFERENCE FORM: ENDGAME SCALE FACTOR (2026-09-30; C3 doc §16) ═══════════════════════════════
 * WHY: the additive form above is sign(T)·C — discontinuous at a level score (+1 mp becomes +C) — and two very
 * different fits of it both lost ~−24 Elo at depth. SF11/15, Ethereal and Weiss all scale the endgame MULTIPLICATIVELY
 * (eg·sf/64), which is continuous at 0 (T·f → 0 from both sides) and can never manufacture an edge from noise.
 * WHAT: f = clamp(64 + BASE + SP·strong pawns + ONEFLANK·[pawns on ≤1 flank] + OCB·[each side one bishop, opposite
 * colours, no other pieces] + PASSED·strong passers, 0, 64); strong = the leader, sign(total). v2 ships in non-pair
 * mode, so the scale enters as total·(1 + eg·(f−64)/64) with eg = (256−phase)/256 — exact on the eg share when
 * mg = eg, and the mg share is untouched at phase 256. Inputs are leader-relative and board-derived, so a colour
 * mirror swaps the leader together with the sign: antisymmetric. Division truncates toward zero on both signs.
 * Returns the adjustment (mp, Black-positive); |adjustment| ≤ |total| because f ∈ [0, 64].
 */
static inline int win_scale_adjust(const V2Context &c, const PawnEntry &pe, int total) noexcept
{
	if (total == 0) return 0;
	const int s = total > 0 ? 1 : 0;                    // Black-positive: a positive total means Black leads (s = 1)
	const uint64_t own = s ? c.black : c.white;
	const int sp = __builtin_popcountll(c.pawns & own);
	const int oneflank = ((c.pawns & WIN_QS) && (c.pawns & WIN_KS)) ? 0 : 1;
	const uint64_t wb = c.bishops & c.white, bb = c.bishops & c.black;
	int ocb = 0;
	if (__builtin_popcountll(wb) == 1 && __builtin_popcountll(bb) == 1 && !(c.knights | c.rooks | c.queens)){
		const int ws = __builtin_ctzll(wb), bs = __builtin_ctzll(bb);
		ocb = (((ws & 7) + (ws >> 3)) & 1) != (((bs & 7) + (bs >> 3)) & 1);
	}
	const int passed = __builtin_popcountll(pe.passed[s]);
	int f = 64 + Config::POT_V2_WIN_BASE + Config::POT_V2_WIN_SP * sp + Config::POT_V2_WIN_ONEFLANK * oneflank
	      + Config::POT_V2_WIN_OCB * ocb + Config::POT_V2_WIN_PASSED * passed;
	// ── MATERIAL-CLASS CAPS (2026-10-09; EVAL-NUANCES-VS-GIANTS doc; SF15.1 evaluate.cpp winnable() + material.cpp) ──────
	// SF's class rules REPLACE its pawn-count formula, so here each rule is a CAP: when its condition holds,
	// f = min(f, 64 + knob …). A knob at 0 caps at 64 = no change ⇒ byte-identical; the base knob of each rule enables it.
	// Leader-relative and board-derived ⇒ antisymmetric under the colour mirror, like every other input here.
	if (Config::POT_V2_WIN_OCBX | Config::POT_V2_WIN_ROOKE | Config::POT_V2_WIN_QNOQ | Config::POT_V2_WIN_LONEMINOR){
		const uint64_t them = s ? c.white : c.black;
		const int minors_own = __builtin_popcountll((c.knights | c.bishops) & own);
		const bool heavy_own = (c.rooks | c.queens) & own;
		auto cap = [&](int v){ if (v < f) f = v; };
		// Opposite-coloured bishops WITH other pieces (4/4 universal; ours above fires on pure OCB only):
		// SF15.1 sf = 22 + 3 · count<ALL_PIECES>(strong).
		if (Config::POT_V2_WIN_OCBX && !ocb && __builtin_popcountll(wb) == 1 && __builtin_popcountll(bb) == 1){
			const int ws = __builtin_ctzll(wb), bs = __builtin_ctzll(bb);
			if ((((ws & 7) + (ws >> 3)) & 1) != (((bs & 7) + (bs >> 3)) & 1))
				cap(64 + Config::POT_V2_WIN_OCBX + Config::POT_V2_WIN_OCBX_PC * __builtin_popcountll(own));
		}
		// Rook ending, one rook each and nothing else: leader at most one pawn up, its pawns on ONE flank, the defending king
		// touching one of its own pawns (SF15.1 sf = 36).
		if (Config::POT_V2_WIN_ROOKE && !(c.knights | c.bishops | c.queens)
		    && __builtin_popcountll(c.rooks & c.white) == 1 && __builtin_popcountll(c.rooks & c.black) == 1){
			const uint64_t sp_bb = c.pawns & own;
			const int diff = sp - __builtin_popcountll(c.pawns & them);
			const uint64_t tk = c.kings & them;
			if (diff <= 1 && sp_bb && (bool(sp_bb & WIN_KS) != bool(sp_bb & WIN_QS)) && tk
			    && (BB_KING_ATTACKS[__builtin_ctzll(tk)] & c.pawns & them))
				cap(64 + Config::POT_V2_WIN_ROOKE);
		}
		// Queen vs no queen (exactly one queen on the board): SF15.1 sf = 37 + 3 · minors of the side WITHOUT the queen.
		// The one persistent material misjudgement (v2 over-values queen-vs-minors compensation) was only ever attacked
		// additively; this is the multiplicative form.
		if (Config::POT_V2_WIN_QNOQ && __builtin_popcountll(c.queens) == 1){
			const uint64_t noq = (c.queens & c.white) ? c.black : c.white;
			cap(64 + Config::POT_V2_WIN_QNOQ + Config::POT_V2_WIN_QNOQ_MINOR * __builtin_popcountll((c.knights | c.bishops) & noq));
		}
		// Leader with NO pawns and at most a lone minor (no rook/queen): cannot win (SF material.cpp → SCALE_FACTOR_DRAW).
		// The shipped BASE leaves this at 27/64 ≈ 0.42.
		if (Config::POT_V2_WIN_LONEMINOR && sp == 0 && !heavy_own && minors_own <= 1)
			cap(64 + Config::POT_V2_WIN_LONEMINOR);
	}
	if (f > 64) f = 64;
	if (f < 0) f = 0;
	if (f == 64) return 0;
	const long long num = (long long)total * (256 - c.phase256) * (f - 64);
	return (int)(num / (256LL * 64));
}

/* Winnability input probe (diagnostic): out[0..6] = in[0..6], out[7] = phase256. */
void win_probe(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask,
               uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask, long long *out)
{
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, 0);
	PawnEntry pe;
	build_pawn_entry(pe, c);
	int in[7];
	win_inputs(c, pe, in);
	for (int i = 0; i < 7; ++i) out[i] = in[i];
	out[7] = c.phase256;
}

/* TEXEL FIT C1 feature extractor -- see eval_v2.h for the layout. Diagnostic only, never called from search.
 *
 * ★ It reuses the engine's own DETECTORS (build_pawn_entry, mobility_build, placement_detect) and re-derives only
 * the per-parameter COUNTS that each scorer multiplies by its constants. That duplication is checked, not trusted:
 * the fitter requires sum(count x v2_features_theta) to reproduce each block's published score (mobility,
 * pawn_struct, v2_passers, v2_placement) within its truncation budget on every row, so any drift from a scorer
 * shows up immediately as a residual.
 */
static constexpr int V2F_MOB = 0, V2F_DOUBLED = 66, V2F_ISO = 67, V2F_BACKWARD = 75, V2F_WU = 76,
                     V2F_PASSED = 77, V2F_CAND = 85, V2F_KD = 93, V2F_OUT_N = 97, V2F_OUT_B = 98,
                     V2F_BEHIND = 99, V2F_BADB = 100, V2F_TRAPR = 104, V2F_WEAKQ = 105,
                     V2F_C1_END = 106, V2F_KSB = 106, V2F_KFL = V2F_KSB + KSB_CELLS, V2F_KPROT = V2F_KFL + KFL_CELLS,
                     V2F_PX = V2F_KPROT + KPROT_CELLS;    // PX passer cells appended 2026-10-03
static_assert(V2F_PX + PX_CELLS == V2F_PER_SIDE, "v2_features layout: the PX cells must end the per-side block");
static constexpr int V2F_MOB_BASE[4] = {0, 9, 23, 38};   // offsets of N / B / R / Q move-count cells

void v2_features(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask,
                 uint64_t queensMask, uint64_t kingsMask, uint64_t whiteMask, uint64_t blackMask,
                 uint64_t castlingRights, long long *out)
{
	for (int k = 0; k < 2 * V2F_PER_SIDE + 1; ++k) out[k] = 0;
	V2Context c;
	build_context(c, 0, true, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
	              whiteMask, blackMask, whiteMask | blackMask, castlingRights);
	long long flags = 0;
	if (Config::DRAW_V2_CLASS && draw_class(c)) flags |= 1;
	int t2 = 0;
	if (Config::TIER2_V2_MAG > 0 && tier2b_value_mp(c, t2)) flags |= 2;
	// Knobs whose terms this extractor does not decompose: a live one would leave its value in the "fixed" part.
	if (Config::OUTPOST_V2_FORM != 0 || Config::BADB_V2_FORM != 1 || Config::TRAPROOK_V2_FORM != 0
	    || Config::MOB_V2_SAFE || Config::REACH_V2_PCT || Config::LONGDIAG_V2_PCT || Config::LATENT_V2_PCT
	    || Config::PASSER_V2_PATH_PCT || Config::PS_V2_CONN_MAG || (g_ksb_on && Config::KSB_V2_CASTLE)) flags |= 4;
	out[2 * V2F_PER_SIDE] = flags;

	// Mobility: the same area, pin restriction and x-ray occupancy the scorer's attack build uses.
	MobAcc mw, mb;
	SideAttacks wa, ba;
	mobility_build(c, wa, ba, mw, mb);
	const uint64_t typeMasks[4] = {c.knights, c.bishops, c.rooks, c.queens};
	for (int s = 0; s < 2; ++s){
		const bool white = (s == 0);
		const MobAcc &m = white ? mw : mb;
		const uint64_t own = white ? c.white : c.black;
		long long *o = out + s * V2F_PER_SIDE;
		for (int i = 0; i < 4; ++i){
			const uint8_t pt = (uint8_t)(KNIGHT + i);
			uint64_t bb = typeMasks[i] & own;
			while (bb){
				const uint8_t sq = (uint8_t)__builtin_ctzll(bb);
				bb &= bb - 1;
				uint64_t occ_for = c.occupied;
				if (Config::KS_V2_XRAY){
					if (pt == BISHOP)    occ_for = c.occupied ^ c.queens;
					else if (pt == ROOK) occ_for = c.occupied ^ c.queens ^ (c.rooks & own);
				}
				const uint64_t a  = attacks_mask(white, occ_for, sq, pt);
				const uint64_t am = ((m.pinned >> sq) & 1) ? (a & ray(m.ksq, sq)) : a;
				++o[V2F_MOB + V2F_MOB_BASE[i] + __builtin_popcountll(am & m.area)];
			}
		}
	}

	// Pawn structure and passers from the one detector build.
	PawnEntry pe;
	build_pawn_entry(pe, c);
	const int wk = (c.kings & c.white) ? __builtin_ctzll(c.kings & c.white) : 0;
	const int bk = (c.kings & c.black) ? __builtin_ctzll(c.kings & c.black) : 0;
	for (int s = 0; s < 2; ++s){
		const bool white = (s == 0);
		long long *o = out + s * V2F_PER_SIDE;
		o[V2F_DOUBLED] = __builtin_popcountll(pe.doubled[s]);
		uint64_t ib = pe.isolated[s];
		while (ib){ ++o[V2F_ISO + (__builtin_ctzll(ib) & 7)]; ib &= ib - 1; }
		o[V2F_BACKWARD] = __builtin_popcountll(pe.backward[s]);
		o[V2F_WU] = __builtin_popcountll((pe.isolated[s] | pe.backward[s]) & ~pe.opposed[s]);

		const int ourK = white ? wk : bk, theirK = white ? bk : wk;
		uint64_t bb = pe.passed[s] | pe.candidate[s];
		while (bb){
			const int sq = __builtin_ctzll(bb);
			bb &= bb - 1;
			const bool cand = ((1ULL << sq) & pe.candidate[s]) != 0;
			const int r = white ? (sq >> 3) : (7 - (sq >> 3));
			++o[(cand ? V2F_CAND : V2F_PASSED) + r];
			if (r >= Config::PASSER_V2_MIN_RANK){
				int w = 5 * r - 13;
				if (w < 0) w = 0;
				const int stop = white ? (sq + 8) : (sq - 8);
				if (stop >= 0 && stop < 64){
					o[V2F_KD + (cand ? 2 : 0)]     += (long long)ps_kdist(theirK, stop) * w;
					o[V2F_KD + (cand ? 3 : 1)]     += (long long)ps_kdist(ourK, stop) * w;
				}
			}
		}
	}

	// Placement counts from the shared detector, reusing the per-rook mobility counts as the scorer does.
	PlaceCounts pc;
	placement_detect(pc, c, pe, &mw, &mb);
	for (int s = 0; s < 2; ++s){
		long long *o = out + s * V2F_PER_SIDE;
		o[V2F_OUT_N]  = pc.outpost_n[s];
		o[V2F_OUT_B]  = pc.outpost_b[s];
		o[V2F_BEHIND] = pc.behind[s];
		for (int k = 0; k < 4; ++k) o[V2F_BADB + k] = pc.badb_cls[s][k];
		o[V2F_TRAPR] = pc.traprook_units[s];
		o[V2F_WEAKQ] = pc.weakq[s];
	}

	// C3-a shelter + storm cells at the king's ACTUAL square, unconditional (reported whether or not KSB_V2 is on).
	for (int s = 0; s < 2; ++s){
		const bool white = (s == 0);
		const uint64_t kb = c.kings & (white ? c.white : c.black);
		if (!kb) continue;
		int cells[6];
		const int n = ksb_cells(c.pawns & (white ? c.white : c.black), c.pawns & (white ? c.black : c.white), white,
		                        __builtin_ctzll(kb), cells);
		for (int i = 0; i < n; ++i) ++out[s * V2F_PER_SIDE + V2F_KSB + cells[i]];
		// C3-b and C3-c, also unconditional.
		int fc[4];
		const int nf = kfl_cells(c, white, fc);
		for (int i = 0; i < nf; ++i) ++out[s * V2F_PER_SIDE + V2F_KFL + fc[i]];
		int pc2[KPROT_CELLS];
		kprot_counts(c, white, pc2);
		for (int i = 0; i < KPROT_CELLS; ++i) out[s * V2F_PER_SIDE + V2F_KPROT + i] += pc2[i];
	}

	// PX passer cells (2026-10-03), unconditional, from the same pawn entry and shared attack maps the scorer uses.
	for (int s = 0; s < 2; ++s){
		int px[PX_CELLS];
		px_counts(c, pe, wa, ba, s == 0, px);
		for (int i = 0; i < PX_CELLS; ++i) out[s * V2F_PER_SIDE + V2F_PX + i] = px[i];
	}
}

/* Starting value of every C1 parameter, (mg, eg) millipawns per unit of its count, from the live Config: the
 * product of each scorer's table entry and its percent / magnitude knobs, before any per-site truncation. */
void v2_features_theta(double *mg, double *eg)
{
	for (int k = 0; k < V2F_PER_SIDE; ++k) mg[k] = eg[k] = 0.0;
	// C3-a cells: the loaded values when the term is on, else 0 (the term is absent, so its value IS 0).
	if (g_ksb_on)
		for (int i = 0; i < KSB_CELLS; ++i){ mg[V2F_KSB + i] = ksb_w[0][i]; eg[V2F_KSB + i] = ksb_w[1][i]; }
	if (g_kfl_on)
		for (int i = 0; i < KFL_CELLS; ++i){ mg[V2F_KFL + i] = kfl_w[0][i]; eg[V2F_KFL + i] = kfl_w[1][i]; }
	if (g_kprot_on)
		for (int i = 0; i < KPROT_CELLS; ++i){ mg[V2F_KPROT + i] = kprot_w[0][i]; eg[V2F_KPROT + i] = kprot_w[1][i]; }
	if (g_px_on)
		for (int i = 0; i < PX_CELLS; ++i){ mg[V2F_PX + i] = px_w[0][i]; eg[V2F_PX + i] = px_w[1][i]; }
	if (g_c1_fit){
		// The ACTIVE fitted values, so a pass under C1_V2_FIT=1 checks the engine against the fitted model (closure).
		for (int i = 0; i < 4; ++i)
			for (int n = 0; n < 28; ++n)
				if (V2F_MOB_BASE[i] + n < (i < 3 ? V2F_MOB_BASE[i + 1] : 66)){
					mg[V2F_MOB + V2F_MOB_BASE[i] + n] = c1_mob[0][i][n];
					eg[V2F_MOB + V2F_MOB_BASE[i] + n] = c1_mob[1][i][n];
				}
		mg[V2F_DOUBLED] = c1_ps_doubled[0];  eg[V2F_DOUBLED] = c1_ps_doubled[1];
		for (int f = 0; f < 8; ++f){ mg[V2F_ISO + f] = c1_ps_iso[0][f]; eg[V2F_ISO + f] = c1_ps_iso[1][f]; }
		mg[V2F_BACKWARD] = c1_ps_back[0];    eg[V2F_BACKWARD] = c1_ps_back[1];
		mg[V2F_WU] = c1_ps_wu[0];            eg[V2F_WU] = c1_ps_wu[1];
		for (int r = 0; r < 8; ++r){
			mg[V2F_PASSED + r] = c1_pass[0][0][r]; eg[V2F_PASSED + r] = c1_pass[0][1][r];
			mg[V2F_CAND + r]   = c1_pass[1][0][r]; eg[V2F_CAND + r]   = c1_pass[1][1][r];
		}
		eg[V2F_KD + 0] = c1_kd[0][0]; eg[V2F_KD + 1] = c1_kd[0][1];
		eg[V2F_KD + 2] = c1_kd[1][0]; eg[V2F_KD + 3] = c1_kd[1][1];
		for (int j = 0; j < 9; ++j){ mg[V2F_OUT_N + j] = c1_place[0][j]; eg[V2F_OUT_N + j] = c1_place[1][j]; }
		return;
	}
	// Mobility: raw * MAG / N_RANGE on the mg leg; raw * MAG * PAWN_MG / (N_RANGE * PAWN_EG) * EG_PCT / 100 on eg.
	const int t = Config::MOB_V2_TABLE;
	const double mag = Config::MOB_V2_MAG;
	const double nr = MOB_TAB_N_RANGE[t];
	const double eg_scale = mag * MOB_TAB_PAWN_MG[t] / (nr * MOB_TAB_PAWN_EG[t]) * Config::MOB_V2_EG_PCT / 100.0;
	const int n_max[4] = {9, 14, 15, 28};
	for (int i = 0; i < 4; ++i)
		for (int n = 0; n < n_max[i]; ++n){
			mg[V2F_MOB + V2F_MOB_BASE[i] + n] = MOB_TAB_MG[t][i][n] * mag / nr;
			eg[V2F_MOB + V2F_MOB_BASE[i] + n] = MOB_TAB_EG[t][i][n] * eg_scale;
		}
	// Pawn structure, all scaled by PS_V2_MAG / 100 after the blend.
	const double ps = Config::PS_V2_MAG / 100.0;
	mg[V2F_DOUBLED]  = -Config::PS_V2_DOUBLED_MG * ps;   eg[V2F_DOUBLED]  = -Config::PS_V2_DOUBLED_EG * ps;
	for (int f = 0; f < 8; ++f){
		mg[V2F_ISO + f] = PS_ISO_FILE_MG[f] * Config::PS_V2_ISOLATED_MG / 100.0 * ps;
		eg[V2F_ISO + f] = PS_ISO_FILE_EG[f] * Config::PS_V2_ISOLATED_EG / 100.0 * ps;
	}
	mg[V2F_BACKWARD] = -Config::PS_V2_BACKWARD_MG * ps;  eg[V2F_BACKWARD] = -Config::PS_V2_BACKWARD_EG * ps;
	mg[V2F_WU]       = -Config::PS_V2_WEAKUNOPP_MG * ps; eg[V2F_WU]       = -Config::PS_V2_WEAKUNOPP_EG * ps;
	// Passers: rank tables x MG/EG_PCT, candidates additionally x CAND_PCT; king terms are eg-only, x w / 100.
	const double pm = Config::PASSER_V2_MAG / 100.0, mgp = Config::PASSER_V2_MG_PCT / 100.0,
	             egp = Config::PASSER_V2_EG_PCT / 100.0, cp = Config::PASSER_V2_CAND_PCT / 100.0;
	for (int r = 0; r < 8; ++r){
		mg[V2F_PASSED + r] = PS_PASSED_MG[r] * mgp * pm;       eg[V2F_PASSED + r] = PS_PASSED_EG[r] * egp * pm;
		mg[V2F_CAND + r]   = PS_PASSED_MG[r] * cp * mgp * pm;  eg[V2F_CAND + r]   = PS_PASSED_EG[r] * cp * egp * pm;
	}
	eg[V2F_KD + 0] =  Config::PASSER_V2_KING_THEM / 100.0 * egp * pm;
	eg[V2F_KD + 1] = -Config::PASSER_V2_KING_US   / 100.0 * egp * pm;
	eg[V2F_KD + 2] =  Config::PASSER_V2_KING_THEM / 100.0 * cp * egp * pm;
	eg[V2F_KD + 3] = -Config::PASSER_V2_KING_US   / 100.0 * cp * egp * pm;
	// Placement: every product is divided by 100 once per side in placement_mp.
	const double op = Config::OUTPOST_V2_PCT / 100.0;
	mg[V2F_OUT_N]  = 2.0 * PL_OUTPOST_MG * op;   eg[V2F_OUT_N]  = 2.0 * PL_OUTPOST_EG * op;
	mg[V2F_OUT_B]  = PL_OUTPOST_MG * op;         eg[V2F_OUT_B]  = PL_OUTPOST_EG * op;
	const double bh = Config::BEHIND_V2_PCT / 100.0;
	mg[V2F_BEHIND] = (Config::BEHIND_V2_FORM == 1 ? 87 : PL_BEHIND_MG) * bh;
	eg[V2F_BEHIND] = (Config::BEHIND_V2_FORM == 1 ? 157 : PL_BEHIND_EG) * bh;
	for (int k = 0; k < 4; ++k){
		mg[V2F_BADB + k] = -PL_SF15_BADB_MG[k] * Config::BADB_V2_PCT / 100.0;
		eg[V2F_BADB + k] = -PL_SF15_BADB_EG[k] * Config::BADB_V2_PCT / 100.0;
	}
	mg[V2F_TRAPR] = -PL_TRAPR_MG * Config::TRAPROOK_V2_PCT / 100.0; eg[V2F_TRAPR] = -PL_TRAPR_EG * Config::TRAPROOK_V2_PCT / 100.0;
	mg[V2F_WEAKQ] = -PL_WEAKQ_MG * Config::WEAKQ_V2_PCT / 100.0;    eg[V2F_WEAKQ] = -PL_WEAKQ_EG * Config::WEAKQ_V2_PCT / 100.0;
}

/* Load the Texel C1 values (Config::C1_V2_FIT). Starts from the live constants (v2_features_theta), so a file
 * holding only some parameters leaves the rest at their shipped values, then applies C1_V2_FILE: lines of
 * `feature_k leg start fitted` as written by diagnostics/_texel_c1_fit.py. A malformed or missing file is
 * reported and the fit stays OFF -- the engine never runs on a half-loaded table. */
void v2_c1_init()
{
	g_c1_fit = false;
	if (Config::C1_V2_FIT == 0) return;
	if (Config::PS_V2_MAG != 100 || Config::PASSER_V2_MAG != 100)
		std::cerr << "⚠️ C1_V2_FIT folds PS_V2_MAG / PASSER_V2_MAG into the fitted values; they are not 100 here." << std::endl;
	double mg[V2F_PER_SIDE], eg[V2F_PER_SIDE];
	v2_features_theta(mg, eg);
	auto set = [&](int k, int leg, double v){ (leg ? eg : mg)[k] = v; };
	const char *path = std::getenv("C1_V2_FILE");
	if (!path || !*path){
		std::cerr << "☠️ C1_V2_FIT=1 needs C1_V2_FILE -- fit stays OFF." << std::endl;
		return;
	}
	std::ifstream in(path);
	if (!in){
		std::cerr << "☠️ C1_V2_FILE=" << path << " cannot be opened -- fit stays OFF." << std::endl;
		return;
	}
	std::string line;
	int n_set = 0;
	while (std::getline(in, line)){
		if (line.empty() || line[0] == '#') continue;
		int k, leg; double start, fitted;
		// ☠️ k is bounded by the C1 range, not V2F_PER_SIDE: the C3-a cells after it have their own loader, and
		// accepting them here would read as loaded while changing nothing.
		if (std::sscanf(line.c_str(), "%d %d %lf %lf", &k, &leg, &start, &fitted) != 4 || k < 0 || k >= V2F_C1_END
		    || leg < 0 || leg > 1){
			std::cerr << "☠️ C1_V2_FILE malformed line '" << line << "' -- fit stays OFF." << std::endl;
			return;
		}
		set(k, leg, fitted);
		++n_set;
	}
	for (int i = 0; i < 4; ++i){
		const int lim = (i < 3 ? V2F_MOB_BASE[i + 1] : 66) - V2F_MOB_BASE[i];
		for (int n = 0; n < 28; ++n){
			c1_mob[0][i][n] = n < lim ? (int)std::lround(mg[V2F_MOB_BASE[i] + n]) : 0;
			c1_mob[1][i][n] = n < lim ? (int)std::lround(eg[V2F_MOB_BASE[i] + n]) : 0;
		}
	}
	for (int leg = 0; leg < 2; ++leg){
		const double *t = leg ? eg : mg;
		c1_ps_doubled[leg] = (int)std::lround(t[V2F_DOUBLED]);
		for (int f = 0; f < 8; ++f) c1_ps_iso[leg][f] = (int)std::lround(t[V2F_ISO + f]);
		c1_ps_back[leg] = (int)std::lround(t[V2F_BACKWARD]);
		c1_ps_wu[leg]   = (int)std::lround(t[V2F_WU]);
		for (int r = 0; r < 8; ++r){
			c1_pass[0][leg][r] = (int)std::lround(t[V2F_PASSED + r]);
			c1_pass[1][leg][r] = (int)std::lround(t[V2F_CAND + r]);
		}
		for (int j = 0; j < 9; ++j) c1_place[leg][j] = (int)std::lround(t[V2F_OUT_N + j]);
	}
	c1_kd[0][0] = eg[V2F_KD + 0]; c1_kd[0][1] = eg[V2F_KD + 1];
	c1_kd[1][0] = eg[V2F_KD + 2]; c1_kd[1][1] = eg[V2F_KD + 3];
	g_c1_fit = true;
	std::cerr << "[c1] Texel C1 fit ON: " << n_set << " values from " << path << std::endl;
}

/* Load one C3 cell table from env `env_name`. Same line format as C1_V2_FILE -- `feature_k leg start fitted`, k in the
 * v2_features index space [k0, k0 + n) -- so one fitter output serves every loader. Cells a file does not name stay 0
 * (the reference). A missing or malformed file is reported and the block stays OFF (returns false, table zeroed): the
 * engine never runs on a half-loaded table.
 * @param w_mg, w_eg  the block's two legs, n entries each */
static bool c3_load_table(const char *knob, const char *env_name, int k0, int n, int *w_mg, int *w_eg)
{
	for (int i = 0; i < n; ++i) w_mg[i] = w_eg[i] = 0;
	const char *path = std::getenv(env_name);
	if (!path || !*path){
		std::cerr << "☠️ " << knob << "=1 needs " << env_name << " -- it stays OFF." << '\n';
		return false;
	}
	std::ifstream in(path);
	if (!in){
		std::cerr << "☠️ " << env_name << "=" << path << " cannot be opened -- " << knob << " stays OFF." << '\n';
		return false;
	}
	std::vector<int> mg(n, 0), eg(n, 0);
	std::string line;
	int n_set = 0;
	while (std::getline(in, line)){
		if (line.empty() || line[0] == '#') continue;
		int k, leg; double start, fitted;
		if (std::sscanf(line.c_str(), "%d %d %lf %lf", &k, &leg, &start, &fitted) != 4 || k < k0 || k >= k0 + n
		    || leg < 0 || leg > 1){
			std::cerr << "☠️ " << env_name << " malformed line '" << line << "' -- " << knob << " stays OFF." << '\n';
			return false;
		}
		(leg ? eg : mg)[k - k0] = (int)std::lround(fitted);
		++n_set;
	}
	for (int i = 0; i < n; ++i){ w_mg[i] = mg[i]; w_eg[i] = eg[i]; }
	std::cerr << "[c3] " << knob << " ON: " << n_set << " values from " << path << '\n';
	return true;
}

/* Load the C3 detector tables -- C3-a shelter/storm (KSB_V2), C3-b pawnless flank + king-pawn distance (KFL_V2), C3-c
 * KingProtector (KPROT_V2) -- once at engine init. Each block is independent, so the nested fits can switch them
 * separately. */
void v2_c3_init()
{
	// The live attacker-weight profile (Fit K2): the shipped KS_W_OURS unless KS_V2_W_N/B/R/Q override it.
	KS_W_LIVE[KNIGHT] = Config::KS_V2_W_N;
	KS_W_LIVE[BISHOP] = Config::KS_V2_W_B;
	KS_W_LIVE[ROOK]   = Config::KS_V2_W_R;
	KS_W_LIVE[QUEEN]  = Config::KS_V2_W_Q;
	// KS-B: an explicit KSB_V2_FILE wins; without one, the SHIPPED depth-fitted cells compiled in (ship_tables_v2.h,
	// 2026-10-03) — so V2_PRESET=shipped needs no file. (Before 10-03, KSB_V2=1 without a file stayed OFF.)
	const char *ksb_path = std::getenv("KSB_V2_FILE");
	if (Config::KSB_V2 && !(ksb_path && *ksb_path)){
		for (int i = 0; i < KSB_CELLS; ++i) ksb_w[0][i] = ksb_w[1][i] = 0;
		for (int i = 0; i < KSB_SHIP_N; ++i) ksb_w[KSB_SHIP_LEG[i]][KSB_SHIP_K[i] - V2F_KSB] = KSB_SHIP_V[i];
		g_ksb_on = true;
		std::cerr << "[c3] KSB_V2 ON: " << KSB_SHIP_N << " compiled shipped cells (ship_tables_v2.h)" << '\n';
	} else
		g_ksb_on = Config::KSB_V2 && c3_load_table("KSB_V2", "KSB_V2_FILE", V2F_KSB, KSB_CELLS, ksb_w[0], ksb_w[1]);
	g_kfl_on   = Config::KFL_V2   && c3_load_table("KFL_V2", "KFL_V2_FILE", V2F_KFL, KFL_CELLS, kfl_w[0], kfl_w[1]);
	g_kprot_on = Config::KPROT_V2 && c3_load_table("KPROT_V2", "KPROT_V2_FILE", V2F_KPROT, KPROT_CELLS,
	                                               kprot_w[0], kprot_w[1]);
	if (g_ksb_on && Config::KSB_V2_CASTLE) std::cerr << "[c3] KSB_V2_CASTLE: shelter scored at the castling max" << '\n';
	if (Config::KAUF_V2_MAG != 0 && Config::KAUF_V2_FORM == 3) kauf_fit_load();
	g_px_on = Config::PX_V2 && c3_load_table("PX_V2", "PX_V2_FILE", V2F_PX, PX_CELLS, px_w[0], px_w[1]);
}

int placement_and_piece_eval_v2(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t castlingRights)
{
	// Compile-gated cycle profiler (no-op unless built with PROFILE_EVAL=1; production stays
	// byte-identical -- verify with the WAC fingerprint after any production rebuild).
	PROF_BLOCK(PROF_V2_EVAL);
	V2Context c;
	{
		PROF_BLOCK(PROF_V2_CONTEXT);
		build_context(c, moveNum, turn, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
		              occupied_whiteMask, occupied_blackMask, occupiedMask, castlingRights);
	}

	// (tier-2b technique value is defined above draw_class's caller; see tier2b_value_mp)
	// ── slice 1: binary draw classifier ──────────────────────────────────────────────────────────
	// Unconditional, not gated on phase: every case needs <= 2 non-king pieces, which IS the endgame by
	// construction, so a phase gate would only add a test. ⚠️ v1 instead hides this behind its endgame
	// branch (cpp_bitboard.cpp:7941), which is why its cost is invisible there.
	if (Config::DRAW_V2_CLASS && draw_class(c)){
		// Publish exactly what is KNOWN: total = 0, which IS the eval here, and no term fields.
		// ☠️ The first version set terms_valid = 0 ("publish nothing, let consumers KeyError") and broke the very
		// gate it was meant to protect: eval_symmetry.py reads bd["total"] and died with KeyError the moment one
		// of its 600 positions was draw-flagged (2026-09-13). `total` was never unknown -- only the TERMS are.
		// Setting ONLY EB_TOTAL still hides the stale term fields (the chimera the arm-2 note warns about) while
		// every consumer that needs the score gets the true one.
		if (g_capture_eval_breakdown && Config::EVAL_ARM == 1){
			g_eval_breakdown.total       = 0;
			g_eval_breakdown.arm         = Config::EVAL_ARM;
			g_eval_breakdown.terms_valid = (1ULL << EB_TOTAL);
		}
		return 0;
	}

	// ── slice 4 tier-2b: TECHNIQUE VALUE for K+R vs K+minor ──────────────────────────────────────
	// Gated on TIER2_V2_MAG, so 0 = absent = byte-identical. Sits immediately after draw_class because it
	// is the same KIND of thing -- a material-configuration REPLACEMENT, not a term added to a sum.
	if (Config::TIER2_V2_MAG > 0){
		int t2 = 0;
		if (tier2b_value_mp(c, t2)){
			// ☠️ SAME CONTRACT AS draw_class ABOVE -- and omitting it was a real defect (found 2026-09-19).
			// This is a material-configuration REPLACEMENT, so `total` is KNOWN and the term fields are not.
			// Without publishing here the breakdown keeps the PREVIOUS position's values, and every static
			// instrument reads a chimera: `_eval_symmetry.py` takes bd["total"], NOT the return value, so it
			// would have compared one position's score against another's and reported the difference as an
			// asymmetry (or, worse, as a clean pass). Search was never affected -- it uses the return value.
			// Publishing only EB_TOTAL hides the stale terms while giving every consumer the true score.
			if (g_capture_eval_breakdown && Config::EVAL_ARM == 1){
				g_eval_breakdown.total       = t2;
				g_eval_breakdown.arm         = Config::EVAL_ARM;
				g_eval_breakdown.terms_valid = (1ULL << EB_TOTAL);
			}
			return t2;
		}
	}

	// ═══ EVAL_V2_PAIR: accumulate unblended legs and interpolate ONCE at the end ═══════════════════
	// `pair_mode` false = the shipped path: every term blends at its own site, exactly as before.
	// Each scorer fills `lp` AND returns its blended value, so there is ONE scoring path either way —
	// the mode only chooses which of the two outputs is consumed.
	const bool pair_mode = (Config::EVAL_V2_PAIR != 0);
	EvalPair acc;   // the single accumulator (pair mode only)
	EvalPair lp;    // scratch legs, refilled per term

	int w_mat = 0, b_mat = 0, w_pst = 0, b_pst = 0;
	int total = rung0_material_and_placement(c, w_mat, b_mat, w_pst, b_pst, &lp);
	if (pair_mode){ acc = lp; total = 0; }

	// ── slice 3: bishop pair ─────────────────────────────────────────────────────────────────────
	// Sits with material because that is what it is -- a census term, not placement. Gated on BPAIR_V2_MAG,
	// so 0 = absent = byte-identical.
	int bp_mp = 0;
	if (Config::BPAIR_V2_MAG > 0){
		bp_mp = bishop_pair_mp(c, &lp);
		if (pair_mode) acc += lp; else total += bp_mp;
	}

	// ── slice 3: Kaufman / polynomial material imbalance ────────────────────────────────────────
	// Also sits with material -- it IS a census re-pricing of material, which is why the pawn is its unit
	// anchor. Gated on KAUF_V2_MAG, so 0 = absent = byte-identical. ★ Antisymmetric by construction, so it
	// cannot move the colour-symmetry gate (v2 is 0/4000 and must stay there).
	int kauf_mp_v = 0;
	if (Config::KAUF_V2_MAG > 0){
		kauf_mp_v = kaufman_mp(c);
		// Kaufman is PHASE-FLAT (no blend site), so it enters both legs identically.
		if (pair_mode){ acc.mg += kauf_mp_v; acc.eg += kauf_mp_v; } else total += kauf_mp_v;
	}

	// ── NARROW MATERIAL CLASSES (2026-10-01; see mcl_mp). Middlegame-weighted, so it enters the mg leg only. Gated on
	// MCL_V2: 0 = absent = byte-identical.
	if (Config::MCL_V2){
		const int m = mcl_mp(c);
		if (pair_mode) acc.mg -= mcl_white_mp(c); else total += m;     // acc is Black-positive like the total
	}

	// ── rung 1: king safety (KS-A) ───────────────────────────────────────────────────────────────
	// Gated on KS_V2_MAX, so 0 = the rung is absent and this is byte-identical to rung 0.5. The attack maps
	// are built ONCE here and handed to both kings by reference -- they are the expensive part, and each
	// king needs the other side as attacker and its own as defender, so one build serves both.
	// ── slice 2: mobility rides on the SAME attack build ─────────────────────────────────────────
	// The build now runs when EITHER consumer is on. With MOB_V2_MAG == 0 the call is the exact rung-1 build
	// (null accumulator), so KS output is unchanged.
	int ks_w = 0, ks_b = 0, ks_mp = 0;
	int mob_mp = 0, mob_cnt_w = 0, mob_cnt_b = 0;
	int sp_mp = 0;
	const bool mob_on = Config::MOB_V2_MAG > 0;
	// ── slice 3: space and threats ride the SAME attack build (both need the enemy's maps) ───────
	const bool space_on = Config::SPACE_V2_MAG > 0;
	int th_mp = 0;
	const bool threats_on = Config::THREAT_V2_PCT > 0;
	// Hoisted so the placement pass can read the per-rook mobility counts. Uninitialised unless mobility runs.
	MobAcc mw, mb;
	bool acc_ready = false;
	// ⚠️ HOISTED out of the block below (2026-09-21) so the passer path-safety ladder can read them --
	// the same reason MobAcc was hoisted for the placement pass. They are UNINITIALISED unless the build
	// actually runs, so `atk_ready` is the guard and every consumer must take it, never the maps alone.
	SideAttacks wa, ba;
	bool atk_ready = false;
	if (Config::KS_V2_MAX > 0 || mob_on || space_on || threats_on){
		atk_ready = true;
		{
			// The attack build is shared by KS / mobility / space / threats and was v1's single most
			// expensive block, so it is scoped separately from the scoring that consumes it.
			PROF_BLOCK(PROF_V2_ATTACK);
			if (mob_on){
				acc_ready = true;
				mobility_build(c, wa, ba, mw, mb);
				mob_mp    = mobility_mp(mw, mb, c, &lp);
				mob_cnt_w = mw.cnt[0] + mw.cnt[1] + mw.cnt[2] + mw.cnt[3];
				mob_cnt_b = mb.cnt[0] + mb.cnt[1] + mb.cnt[2] + mb.cnt[3];
				if (pair_mode) acc += lp; else total += mob_mp;
			} else {
				build_side_attacks(wa, c, true);
				build_side_attacks(ba, c, false);
			}
		}
		// Space needs BOTH sides' maps (the enemy's full attack set gates its safe squares), so it sits here rather
		// than with the pawn terms. Gated on SPACE_V2_MAG, so 0 = absent = byte-identical.
		if (space_on){
			sp_mp = space_mp(c, wa, ba, &lp);
			if (pair_mode) acc += lp; else total += sp_mp;
		}
		if (threats_on){
			th_mp = threats_mp(c, wa, ba, &lp);
			if (pair_mode) acc += lp; else total += th_mp;
		}
		if (Config::KS_V2_MAX > 0){
			// ⚠️ The scope must START here: ks_units IS the expensive half (it walks both sides' attack
			// maps over the king zones), and ks_danger_mp is a table lookup. Scoping only the scorer
			// would report king safety as ~free and read entirely plausibly.
			PROF_BLOCK(PROF_V2_KS);
			ks_w = ks_units(c, wa, ba, true);
			ks_b = ks_units(c, wa, ba, false);
			// Black-positive, matching `total`: a dangerous WHITE king favours Black (+), a dangerous BLACK
			// king favours White (-). ⚠️ Getting this backwards still produces entirely plausible numbers --
			// it is the historic failure mode -- so the colour ship-gate runs on every KS-A build.
			// ── gap-audit K2/A8: give king safety an ENDGAME LEG ─────────────────────────────
			// KS-A is one saturating curve for every phase. 3 of 4 references give king danger two
			// legs -- SF uses S(kD^2/4096, kD/16), i.e. quadratic in the midgame and LINEAR and far
			// smaller in the endgame -- because with the queens and rooks gone a "dangerous" king is
			// mostly an active one.
            //
			// ★ This is SUBTRACTIVE: it removes king-danger credit in the endgame rather than adding
			// any. That matters, because additive KS changes in this engine are 0-for-11 and only
			// subtractive ones have ever won (ks-twelve-attempt-history-and-the-channel-law).
			// ★ Needs NO (mg,eg) accumulator -- the taper is applied per side at this site, exactly as
			// every other v2 term blends at its own. 100 = today's behaviour = byte-identical.
			// ⚠️ Per-side values are non-negative magnitudes, so >> 8 is safe here; the difference is
			// taken afterwards. Once a quantity is SIGNED, >> n stops being a safe divide.
			const int kdw = ks_danger_mp(ks_w), kdb = ks_danger_mp(ks_b);
			// At 100 the eg legs equal the mg legs, so the blend below is an identity for EVERY phase
			// and the result is byte-identical to the untapered path.
			const int kdw_eg = (Config::KS_V2_EG_PCT == 100) ? kdw : kdw * Config::KS_V2_EG_PCT / 100;
			const int kdb_eg = (Config::KS_V2_EG_PCT == 100) ? kdb : kdb * Config::KS_V2_EG_PCT / 100;
			const int ks_eg  = 256 - c.phase256;
			// ⚠️ ks_mp is computed UNCONDITIONALLY, including in pair mode, because it is what gets
			// published to ev_breakdown below. Zeroing it after accumulating would leave every
			// term-attribution tool reading 0 for king safety -- the chimera the arm-2 note warns about.
			ks_mp = (int)(((kdw * c.phase256 + kdw_eg * ks_eg) >> 8)
			            - ((kdb * c.phase256 + kdb_eg * ks_eg) >> 8));
			// In pair mode the single interpolation at the end applies the phase, so hand it raw legs.
			if (pair_mode){ acc.mg += kdw - kdb; acc.eg += kdw_eg - kdb_eg; } else total += ks_mp;
		}
	}

	// ── rung 2a: pawn structure ──────────────────────────────────────────────────────────────────
	// Gated on PS_V2_MAG, so 0 = the rung is absent and this is byte-identical to the rung-1 result that
	// passed games at ~+101 Elo. ★ The detector is built here and RETURNS its masks; nothing about it is
	// published as a side effect, which is what will let the whole of Layer A+B move behind a pawnKey cache
	// at 2b without any downstream consumer reading stale state.
	// ★ ONE detector build serves both 2a and 2b -- the masks are a pure function of the two pawn
	// bitboards, so whichever rung is on, the entry is computed once and both scorers read it.
	int ps_mp = 0, pp_mp = 0, rf_mp = 0;
	PawnEntry pe;
	const bool rf_on = Config::ROOKFILE_V2_OPEN != 0 || Config::ROOKFILE_V2_SEMI != 0;
	const bool pl_on = Config::OUTPOST_V2_PCT != 0 || Config::REACH_V2_PCT != 0 || Config::BEHIND_V2_PCT != 0
	                || Config::BADB_V2_PCT != 0 || Config::LONGDIAG_V2_PCT != 0 || Config::TRAPROOK_V2_PCT != 0
	                || Config::WEAKQ_V2_PCT != 0 || Config::LATENT_V2_PCT != 0;
	int pl_mp = 0;
	if (Config::PS_V2_MAG != 0 || Config::PASSER_V2_MAG != 0 || rf_on || pl_on){
		{
			// The pawn DETECTOR (Layer A). Scoped alone because it is the one block a pawnKey cache
			// could memoize -- its share IS the prize for building one.
			PROF_BLOCK(PROF_V2_PAWNENTRY);
			build_pawn_entry(pe, c);
		}
		// ⚠️ `lp` is RESET per term here: pawn_structure_mp and passer_value_mp return early (leaving lp
		// untouched) when their magnitude knob is 0, so a stale lp would be added as that term's legs.
		EvalPair lps, lpp, lpr;
		ps_mp = pawn_structure_mp(pe, c, &lps);   // returns 0 when PS_V2_MAG == 0
		// ⚠️ The maps are passed only when they were actually built; with the shared attack build off the
		// path-safety ladder is ABSENT rather than reading uninitialised stack.
		pp_mp = passer_value_mp(pe, c, atk_ready ? &wa : nullptr, atk_ready ? &ba : nullptr, &lpp);
		// slice 2: rook files read the file masks this detector already built
		if (rf_on) rf_mp = rookfile_mp(pe, c, &lpr);
		// slice 2: per-piece placement sub-terms, also reading this detector's masks
		if (pl_on){
			PROF_BLOCK(PROF_V2_PLACEMENT);
			PlaceCounts pc;
			placement_detect(pc, c, pe, acc_ready ? &mw : nullptr, acc_ready ? &mb : nullptr, Config::LATENT_V2_PCT != 0);
			pl_mp = placement_mp(pc, c, &lp);
		}
		if (pair_mode){ acc += lps; acc += lpp; acc += lpr; if (pl_on) acc += lp; }
		else           total += ps_mp + pp_mp + rf_mp + pl_mp;
	}

	// ── C3-a: king shelter + pawn storm (KS-B) ───────────────────────────────────────────────────────
	// A direct score beside KS-A, not a feeder into its units (a shelter->danger coupling is a separate, later test
	// as ONE scalar). Gated on g_ksb_on (KSB_V2 = 1 with a loaded table), so off = absent = byte-identical.
	// Each side's legs are that side's value; Black-positive, so Black's is added and White's subtracted.
	// ── C3-b pawnless flank + king-to-pawn distance, C3-c KingProtector (minors) ─────────────────────
	// Same contract, each on its own knob and table, each published separately so its closure is checked alone.
	int ksb_mp = 0, kfl_mp = 0, kprot_mp = 0;
	{
		int dmg, deg;
		if (g_ksb_on){
			ksb_mp = c3_block_mp(c, ksb_side, dmg, deg);
			if (pair_mode){ acc.mg += dmg; acc.eg += deg; } else total += ksb_mp;
		}
		if (g_kfl_on){
			kfl_mp = c3_block_mp(c, kfl_side, dmg, deg);
			if (pair_mode){ acc.mg += dmg; acc.eg += deg; } else total += kfl_mp;
		}
		if (g_kprot_on){
			kprot_mp = c3_block_mp(c, kprot_side, dmg, deg);
			if (pair_mode){ acc.mg += dmg; acc.eg += deg; } else total += kprot_mp;
		}
	}

	// ── PX: the passer system (2026-10-03; px_counts). Needs the shared attack maps; gated on g_px_on (PX_V2 = 1 with
	// a loaded PX_V2_FILE), so off = absent = byte-identical. Same contract as the C3 blocks.
	int px_mp = 0;
	if (g_px_on && atk_ready){
		if (!(Config::PS_V2_MAG != 0 || Config::PASSER_V2_MAG != 0 || rf_on || pl_on)) build_pawn_entry(pe, c);
		int dmg, deg;
		auto px_side = [&](const V2Context &cc, bool white, int &mg, int &eg){
			int cnt[PX_CELLS];
			px_counts(cc, pe, wa, ba, white, cnt);
			mg = eg = 0;
			for (int i = 0; i < PX_CELLS; ++i){ mg += cnt[i] * px_w[0][i]; eg += cnt[i] * px_w[1][i]; }
		};
		px_mp = c3_block_mp(c, px_side, dmg, deg);
		if (pair_mode){ acc.mg += dmg; acc.eg += deg; } else total += px_mp;
	}

	// ── slice 1 / component 1: tempo ─────────────────────────────────
	// A bonus for simply being the side to move. This is the ONLY place in v2 that reads c.turn, which is
	// what makes its gate an exact identity rather than a statistic: with v2 otherwise side-to-move-blind,
	// diagnostics/eval_symmetry.py's TEMPO swing must equal exactly 2*t on EVERY position, so one run
	// validates sign, magnitude and the phase curve at once.
	// ☠️ The mirror gate cannot check this: _eval_symmetry.py mirrors `turn` as well, so a term that
	// PENALISES the mover is still perfectly mirror-symmetric and passes clean.
	// Sign: `turn == true` is White to move (v1 establishes this independently at cpp_bitboard.cpp:5911 and
	// :5078), and this eval is Black-positive, so rewarding the mover means White-to-move SUBTRACTS.
	// ⚠️ Phased on purpose. SF11/Ethereal/Weiss write one FLAT constant but their pawn is dearer in the
	// endgame, so their tempo is ~1.7-2.0x more pawns in the midgame without them choosing it; our pawn is
	// flat at 1000, so we have to state the taper explicitly. SF1.1 -- the only reference that chose it --
	// used an explicit 50/20 pair. See dev_notes/EVAL-V2-SLICE1-TEMPO-DESIGN.md §1.
	if (Config::TEMPO_V2_MG != 0 || Config::TEMPO_V2_EG != 0){
		const int t = (Config::TEMPO_V2_MG * c.phase256
		             + Config::TEMPO_V2_EG * (256 - c.phase256)) >> 8;
		if (pair_mode){
			const int sgn = c.turn ? -1 : 1;
			acc.mg += sgn * Config::TEMPO_V2_MG;
			acc.eg += sgn * Config::TEMPO_V2_EG;
		} else total += c.turn ? -t : t;
	}

	// ═══ THE ONE INTERPOLATION ═════════════════════════════════════════════════════════════════════
	// Everything above contributed unblended legs; this is the single place phase is applied. It is also
	// where an ENDGAME SCALE FACTOR would multiply acc.eg, and where a tapered PST would finally have
	// somewhere to put its endgame leg (gap audit A2 / A3).
	if (pair_mode) total = eval_blend(acc, c.phase256);

	// ── OvD eg leg: WINNABILITY (see win_inputs). Applied ONCE to the finished total, as SF applies `winnable`, and
	// sign-preserving: the leader's score moves by max(C·eg, −|total|). Gated on WIN_V2, so 0 = absent = byte-identical.
	// The pawn entry is reused when the pawn rungs already built it (the shipped config), else built here.
	int win_mp = 0;
	if (Config::WIN_V2){
		if (!(Config::PS_V2_MAG != 0 || Config::PASSER_V2_MAG != 0 || rf_on || pl_on)) build_pawn_entry(pe, c);
		int in[7];
		win_inputs(c, pe, in);
		win_mp = win_adjust(total, in, c.phase256);
		total += win_mp;
	}
	// ── POT winnability, REFERENCE FORM (2026-09-30): the endgame scale factor (see win_scale_adjust). Replaces the
	// additive form above when on (running both is not a supported configuration). Gated on POT_V2_WIN: 0 = byte-identical.
	if (Config::POT_V2_WIN){
		if (!(Config::PS_V2_MAG != 0 || Config::PASSER_V2_MAG != 0 || rf_on || pl_on)) build_pawn_entry(pe, c);
		const int d = win_scale_adjust(c, pe, total);
		win_mp += d;
		total += d;
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
		if (mob_on){
			// ⚠️ ARM-1 MEANING of det_*_mobility: the AREA-FILTERED squares summed over N/B/R/Q. v1 writes a
			// different quantity into the same field (squares attacked that are not its own). Read the arm
			// before comparing across arms. Plain-SWAP class in the symmetry gate; `mobility` is NEGATE.
			g_eval_breakdown.mobility       = mob_mp;
			g_eval_breakdown.det_w_mobility = mob_cnt_w;
			g_eval_breakdown.det_b_mobility = mob_cnt_b;
			g_eval_breakdown.terms_valid |= (1ULL << EB_MOBILITY)
			                              | (1ULL << EB_DET_W_MOBILITY)
			                              | (1ULL << EB_DET_B_MOBILITY);
		}
		// ── TERM PUBLICATION for the ours-vs-SF11/SF15 triangulation (added 2026-09-19) ──────────────
		// ★ BYTE-IDENTICAL BY CONSTRUCTION: this whole block is inside `g_capture_eval_breakdown`, which is
		// false during search, so publishing costs play exactly nothing. That is what makes it safe to emit
		// the full partition here rather than the 14 fields v2 published before.
		// ★ WHY IT WAS NEEDED: of the subsystems we want to compare against SF's trace rows, only material,
		// king safety and mobility were visible. Pawn structure, passers, placement and rook files were all
		// folded into `total` with no way to read them out, so three of SF's rows had no counterpart to
		// compare against -- not because v2 lacks the terms, but because it never surfaced them.
		// ☠️ Each field is published ONLY when its owner actually ran, so an ABSENT key means "never
		// computed" rather than "computed and zero". That distinction is the entire contract of
		// terms_valid, and collapsing it is how a consumer ends up reporting "no gap" for a term that was
		// simply switched off.
		// ⚠️ Rook files now publish to the NEW `v2_rookfile`, never to v1's `rook_cond` -- that field is a
		// tension-conditioned rescale and writing a file bonus into it would be a read against its
		// writer's intent.
		if (Config::PS_V2_MAG != 0){
			g_eval_breakdown.pawn_struct = ps_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_PAWN_STRUCT);
		}
		if (Config::PASSER_V2_MAG != 0){
			g_eval_breakdown.v2_passers = pp_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_PASSERS);
		}
		if (pl_on){
			g_eval_breakdown.v2_placement = pl_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_PLACEMENT);
		}
		if (rf_on){
			g_eval_breakdown.v2_rookfile = rf_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_ROOKFILE);
		}
		// The four parked slice-3 concepts. They publish only when their knob is non-zero, i.e. never in the
		// shipped config -- so a triangulation run sees them ABSENT, not zero, and cannot mistake a term we
		// deliberately do not carry for a term that measured zero.
		if (space_on){
			g_eval_breakdown.space = sp_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_SPACE);
		}
		if (threats_on){
			g_eval_breakdown.threats = th_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_THREATS);
		}
		if (Config::BPAIR_V2_MAG > 0){
			g_eval_breakdown.pair_bonus = bp_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_PAIR_BONUS);
		}
		if (Config::KAUF_V2_MAG > 0){
			g_eval_breakdown.kaufman_imbalance = kauf_mp_v;
			g_eval_breakdown.terms_valid |= (1ULL << EB_KAUFMAN_IMBALANCE);
		}
		if (g_ksb_on){
			g_eval_breakdown.v2_shelter = ksb_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_SHELTER);
		}
		if (g_kfl_on){
			g_eval_breakdown.v2_kflank = kfl_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_KFLANK);
		}
		if (g_kprot_on){
			g_eval_breakdown.v2_kprot = kprot_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_KPROT);
		}
		if (Config::WIN_V2 || Config::POT_V2_WIN){
			g_eval_breakdown.v2_winnab = win_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_WINNAB);
		}
		if (g_px_on && atk_ready){
			g_eval_breakdown.v2_pxpass = px_mp;
			g_eval_breakdown.terms_valid |= (1ULL << EB_V2_PXPASS);
		}
	}

	return total;
}
