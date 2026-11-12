
#ifndef MOVE_GEN_H
#define MOVE_GEN_H

#include "cpp_bitboard.h"
#include "cache_management.h"
#include "sv2_probe.h"
#include <vector>

/*
 * The exchange value of ONE capture: the piece on from_square takes on to_square, then both sides recapture with their least
 * valuable attacker and may stop at any point. It is see_impl's incremental path with the first capture forced; see() asks
 * the same question for the square (its first capture is the side's least valuable attacker), so the two are equal when the
 * moving piece is that attacker. Used where a move's own value is wanted (SEE_PER_MOVE): with a cheaper attacker of ours on
 * the square, see() describes that piece's capture, not this one (15-17% of the scorer's exchange-branch captures, S3 step 0).
 */
inline int see_move(uint8_t from_square, uint8_t to_square, bool side_to_move, const BoardState &state)
{
	int gain[32];
	gain[0] = get_value_at(to_square, state);
	uint64_t occ = state.occupied;
	const uint64_t qr = state.queens | state.rooks;
	const uint64_t qb = state.queens | state.bishops;
	uint64_t att = (BB_KING_ATTACKS[to_square] & state.kings) |
	               (BB_KNIGHT_ATTACKS[to_square] & state.knights) |
	               (BB_RANK_ATTACKS[to_square][BB_RANK_MASKS[to_square] & occ] & qr) |
	               (BB_FILE_ATTACKS[to_square][BB_FILE_MASKS[to_square] & occ] & qr) |
	               (BB_DIAG_ATTACKS[to_square][BB_DIAG_MASKS[to_square] & occ] & qb) |
	               (BB_PAWN_ATTACKS[0][to_square] & state.pawns & state.occupied_colour[true]) |
	               (BB_PAWN_ATTACKS[1][to_square] & state.pawns & state.occupied_colour[false]);
	att &= occ;
	int d = 0;
	bool s = side_to_move;
	int next = from_square;   // the forced first capture, then the least valuable attacker each turn
	while (true)
	{
		++d;
		gain[d] = get_value_at(next, state) - gain[d - 1];
		occ &= ~(1ULL << next);
		att &= ~(1ULL << next);
		// Re-reveal x-ray sliders through the vacated square (leapers never change).
		att |= ((BB_RANK_ATTACKS[to_square][BB_RANK_MASKS[to_square] & occ] |
		         BB_FILE_ATTACKS[to_square][BB_FILE_MASKS[to_square] & occ]) & qr) |
		       (BB_DIAG_ATTACKS[to_square][BB_DIAG_MASKS[to_square] & occ] & qb);
		att &= occ;
		s = !s;
		const uint64_t side_att = att & state.occupied_colour[s] & occ;
		if (!side_att || d >= 31)
			break;
		next = get_least_valuable_attacker_static(side_att, state);
	}
	while (--d > 0)
		gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
	return gain[0];
}
#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <unordered_map>

// Define masks for move generation
extern std::array<uint64_t, NUM_SQUARES> BB_KNIGHT_ATTACKS;
extern std::array<uint64_t, NUM_SQUARES> BB_KING_ATTACKS;
extern std::array<std::array<uint64_t, NUM_SQUARES>, 2> BB_PAWN_ATTACKS;
extern std::vector<uint64_t> BB_DIAG_MASKS;
extern SlidingTable BB_DIAG_ATTACKS;
extern std::vector<uint64_t> BB_FILE_MASKS;
extern SlidingTable BB_FILE_ATTACKS;
extern std::vector<uint64_t> BB_RANK_MASKS;
extern SlidingTable BB_RANK_ATTACKS;
extern std::vector<std::vector<uint64_t>> BB_RAYS;

// Static-ordering instrumentation: how often the placement tiebreaker was CONSULTED for a quiet (the
// history-thin population it targets) and how often it actually returned a non-zero score. A neutral bench
// result is uninterpretable without these -- "the signal is useless" and "the signal never gets to speak"
// look identical from the outside. Defined in cpp_bitboard.cpp.
extern long g_static_order_eligible;
extern long g_static_order_fires;

// NOTE 2026-08-25: a per-node attacks_mask memo was built and REMOVED here. It was node-identical but
// measured −1.7% NPS (−3.6% with thread_local) -- attacks_mask is a bare table index for knights/kings and
// 2-3 ops for sliders, so the validity test + store cost more than the recompute. Full record:
// memory/movegen-is-36-percent-of-search-not-a-non-lever.md. Do not re-add without reading it.

// Runtime (SCALE_PLACE_*-scaled) placement tables, defined in cpp_bitboard.cpp and previously file-local
// there. Declared here so move ordering can score a quiet by the SAME table the eval uses -- an ordering
// tiebreaker built on different numbers than the eval would rank moves the search then disagrees with.
extern std::array<std::array<std::array<int, 8>, 8>, 6> whitePlacementLayer;
extern std::array<std::array<std::array<int, 8>, 8>, 6> blackPlacementLayer;

/*
    Static placement score for a QUIET move, used to break ties among moves the history tables have never
    seen. Quiets with no history all score identically today, so their relative order is whatever move
    generation happened to produce -- and LMP/LMR then prune and reduce by that arbitrary index.

    Indexing mirrors the eval exactly: [pieceType-1][file][rank] with file = sq & 7, rank = sq >> 3.
    There is ONE placement table per colour (no separate midgame/endgame set), so no phase blend is needed.

    Parameters:
        turn        - side to move; selects the white or black table
        from, to    - the move's origin and destination squares
        *Mask       - piece bitboards, used to identify the mover's type (king falls out by elimination)
    Returns: the scaled placement term, sign-oriented so that a HIGHER score is a better move for `turn`.
*/
inline int staticPlacementScore(bool turn, uint8_t from, uint8_t to,
								uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
								uint64_t rooksMask, uint64_t queensMask)
{
	uint64_t from_bb = BB_SQUARES[from];
	int pt;
	if (pawnsMask & from_bb)        pt = 0;
	else if (knightsMask & from_bb) pt = 1;
	else if (bishopsMask & from_bb) pt = 2;
	else if (rooksMask & from_bb)   pt = 3;
	else if (queensMask & from_bb)  pt = 4;
	else                            pt = 5;

	// Only score piece types whose table the EVAL actually reads. The rook PST is dead code there, and the
	// king PST is endgame-only (see rebuild_scaled_placement) -- ordering either by these tables ranks moves
	// on numbers the evaluation ignores or contradicts.
	if (!(Config::STATIC_ORDER_PIECES & (1 << pt)))
		return 0;

	if (pt == 5 && Config::STATIC_ORDER_KING_EG_ONLY)
	{
		// Same phase formula move generation already uses; computed here only for king moves so the
		// common case pays nothing.
		int phase = 4 * __builtin_popcountll(queensMask)
				  + 2 * __builtin_popcountll(rooksMask)
				  + __builtin_popcountll(bishopsMask | knightsMask);
		int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE;
		if (phase_score <= 62)
			return 0;
	}

	const std::array<std::array<std::array<int, 8>, 8>, 6> &layer =
		turn ? whitePlacementLayer : blackPlacementLayer;

	int dest = layer[pt][to & 7][to >> 3];
	int raw = (Config::STATIC_ORDER_MODE == Config::STATIC_ORDER_DEST)
				  ? dest
				  : dest - layer[pt][from & 7][from >> 3];
	return raw * Config::STATIC_ORDER_WEIGHT / 100;
}

// Union of all squares attacked by the given side's pieces (the "threats" bitboard for threat-conditioned
// quiet history). Computed once per node from the raw piece bitboards; the SAME helper is used at the ordering
// read (move_gen) and the history-write (search_engine) sites so both index threatHist identically. `oppColor`
// selects the pawn attack direction. Only called when Config::ENABLE_THREAT_HIST is on.
inline uint64_t opponent_threats(bool oppColor, uint64_t oppPawns, uint64_t oppKnights, uint64_t oppBishops,
								 uint64_t oppRooks, uint64_t oppQueens, uint64_t oppKing, uint64_t occ)
{
	uint64_t threats = 0, b;
	for (b = oppPawns;   b; b &= b - 1) threats |= BB_PAWN_ATTACKS[oppColor][__builtin_ctzll(b)];
	for (b = oppKnights; b; b &= b - 1) threats |= BB_KNIGHT_ATTACKS[__builtin_ctzll(b)];
	for (b = oppKing;    b; b &= b - 1) threats |= BB_KING_ATTACKS[__builtin_ctzll(b)];
	for (b = oppBishops; b; b &= b - 1) { int s = __builtin_ctzll(b); threats |= BB_DIAG_ATTACKS[s][BB_DIAG_MASKS[s] & occ]; }
	for (b = oppRooks;   b; b &= b - 1) { int s = __builtin_ctzll(b); threats |= BB_RANK_ATTACKS[s][BB_RANK_MASKS[s] & occ] | BB_FILE_ATTACKS[s][BB_FILE_MASKS[s] & occ]; }
	for (b = oppQueens;  b; b &= b - 1) { int s = __builtin_ctzll(b); threats |= BB_DIAG_ATTACKS[s][BB_DIAG_MASKS[s] & occ] | BB_RANK_ATTACKS[s][BB_RANK_MASKS[s] & occ] | BB_FILE_ATTACKS[s][BB_FILE_MASKS[s] & occ]; }
	return threats;
}

// Additive threat-conditioned history term for a quiet from->to given the node's threats bitboard.
inline int threat_hist_term(bool turn, uint64_t threats, uint8_t from, uint8_t to)
{
	int tf = (threats >> from) & 1;
	int tt = (threats >> to) & 1;
	return threatHist[turn][tf][tt][from][to] >> Config::THREAT_HIST_SHIFT;
}

// Node threats bitboard from a BoardState (used at the history-write sites in the search).
inline uint64_t node_threats_of(const BoardState &st)
{
	uint64_t opp = st.occupied_colour[!st.turn];
	return opponent_threats(!st.turn, st.pawns & opp, st.knights & opp, st.bishops & opp,
							st.rooks & opp, st.queens & opp, st.kings & opp, st.occupied);
}

// Bonus/malus write to threatHist for a quiet from->to (saturating or plain, mirroring the butterfly update).
inline void threat_hist_update(bool turn, uint64_t threats, uint8_t from, uint8_t to, int delta, bool saturate)
{
	int tf = (threats >> from) & 1;
	int tt = (threats >> to) & 1;
	if (saturate) hist_update(threatHist[turn][tf][tt][from][to], delta);
	else          threatHist[turn][tf][tt][from][to] += delta;
}

// One-call threatHist maintenance at a quiet beta-cutoff, mirroring the butterfly historyHeuristics update:
// bonus `b` to the cutoff move + `-b/malus_div` malus to every other tried quiet (when malus is enabled).
// Threats are computed once here; no-op unless ENABLE_THREAT_HIST. Root/qsearch cutoffs (no tried list, custom
// bonus) call threat_hist_update directly instead.
inline void threat_hist_on_cutoff(const BoardState &st, const Move &best, const std::vector<Move> &tried,
								  int b, bool saturate, int malus_div, bool malus)
{
	if (!Config::ENABLE_THREAT_HIST) return;
	uint64_t nt = node_threats_of(st);
	threat_hist_update(st.turn, nt, best.from_square, best.to_square, b, saturate);
	if (malus)
		for (const Move &q : tried)
			if (!(q == best))
				threat_hist_update(st.turn, nt, q.from_square, q.to_square, -b / malus_div, saturate);
}

/*
	Set of functions used to generate moves
*/

// A move paired with its ordering score, sorted in place so the final reorder needs no index
// indirection. score is the sort key; from/to/promo carry the move payload.
struct ScoredMove {
	int score;
	uint8_t from;
	uint8_t to;
	uint8_t promo;
};

/* Stable sort by descending score with a merge buffer kept between calls. A stable sort's result is fixed by the comparison
   alone, so this gives exactly std::stable_sort's order without the temporary buffer std::stable_sort allocates on every
   call: insertion-sorted runs of 8, then bottom-up merges that take the right element only when it scores strictly higher. */
inline void stableSortScoredDesc(std::vector<ScoredMove>& a)
{
	constexpr size_t RUN = 8;
	const size_t n = a.size();
	for (size_t lo = 0; lo < n; lo += RUN) {
		const size_t hi = std::min(lo + RUN, n);
		for (size_t i = lo + 1; i < hi; ++i) {
			const ScoredMove x = a[i];
			size_t j = i;
			while (j > lo && a[j - 1].score < x.score) {
				a[j] = a[j - 1];
				--j;
			}
			a[j] = x;
		}
	}
	if (n <= RUN)
		return;
	thread_local std::vector<ScoredMove> buffer;
	if (buffer.size() < n)
		buffer.resize(n);
	ScoredMove* src = a.data();
	ScoredMove* dst = buffer.data();
	for (size_t width = RUN; width < n; width *= 2) {
		for (size_t lo = 0; lo < n; lo += 2 * width) {
			const size_t mid = std::min(lo + width, n), hi = std::min(lo + 2 * width, n);
			size_t i = lo, j = mid, k = lo;
			while (i < mid && j < hi)
				dst[k++] = (src[j].score > src[i].score) ? src[j++] : src[i++];
			while (i < mid)
				dst[k++] = src[i++];
			while (j < hi)
				dst[k++] = src[j++];
		}
		std::swap(src, dst);
	}
	if (src != a.data())
		std::copy(src, src + n, a.data());
}

// Move-list body with the node-invariant king square, slider blockers and checker mask supplied by
// the caller. These depend only on the board, not on from_mask/to_mask, so the partitioned move-gen
// (processMaskPairs invokes this once per piece-type pair) computes them once per node instead of
// re-deriving the expensive slider_blockers / attackersMask on every call.
inline void generateLegalMovesPre(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn,
						uint8_t king, uint64_t blockers, uint64_t checkers){

	// NOTE 2026-08-25: a DIRECT-EMIT path (generate into the caller's arrays, compact survivors in place,
	// avoiding the scratch round-trip) was built and REMOVED here. Node-identical but only ~0-3% NPS, inside
	// the instrument noise. Same record as the attacks memo above.
	//
	// Per-call scratch reused across nodes (cleared, capacity kept) to avoid a fresh allocation on
	// every piece-type pair.
	thread_local std::vector<uint8_t> startPos;
	thread_local std::vector<uint8_t> endPos;
	thread_local std::vector<uint8_t> promotions;
	startPos.clear();
	endPos.clear();
	promotions.clear();

	if (checkers != 0){
		{
		PROF_BLOCK(PROF_MG_PSEUDO);
		generateEvasions(startPos, endPos, promotions, preliminary_castling_mask, king, checkers, from_mask, to_mask, occupiedMask,occupiedWhite,
			             opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
		}

		PROF_BLOCK(PROF_MG_ISSAFE);
		for (size_t i = 0; i < startPos.size(); i++){
			if (is_safe(king, blockers, startPos[i], endPos[i], occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask,
				knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn)){

				startPos_filtered.push_back(startPos[i]);
				endPos_filtered.push_back(endPos[i]);
				promotions_filtered.push_back(promotions[i]);

			}
		}
	} else {
		{
		PROF_BLOCK(PROF_MG_PSEUDO);
		generatePseudoLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask,
	 						     kingsMask & ourPieces, occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
							     rooksMask, queensMask, kingsMask, ep_square, turn);
		}

		PROF_BLOCK(PROF_MG_ISSAFE);
		for (size_t i = 0; i < startPos.size(); i++){
			if (is_safe(king, blockers, startPos[i], endPos[i], occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask,
				knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn)){

				startPos_filtered.push_back(startPos[i]);
				endPos_filtered.push_back(endPos[i]);
				promotions_filtered.push_back(promotions[i]);

			}
		}
	}

}

inline void generateLegalMoves(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){

	uint64_t king_mask = kingsMask & ourPieces;
	uint8_t king = 63 - __builtin_clzll(king_mask);
	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);
	uint64_t checkers = attackersMask(!turn, king, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask, pawnsMask, opposingPieces);

	generateLegalMovesPre(startPos_filtered, endPos_filtered, promotions_filtered, preliminary_castling_mask, from_mask, to_mask,
						  occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
						  rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
}

inline bool has_any_legal_move(uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){

	// Reused scratch (cleared, capacity kept): this runs on eval-cache misses, and fresh vectors allocated every time.
	// Not re-entrant, which is fine: nothing below calls back into has_any_legal_move.
	static thread_local std::vector<uint8_t> startPos, endPos, promotions;
	startPos.clear();
	endPos.clear();
	promotions.clear();

	uint64_t king_mask = kingsMask & ourPieces;
	uint8_t king = 63 - __builtin_clzll(king_mask);

	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);
	uint64_t checkers = attackersMask(!turn, king, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask, pawnsMask, opposingPieces);

	if (checkers != 0){
		generateEvasions(startPos, endPos, promotions, preliminary_castling_mask, king, checkers, from_mask, to_mask, occupiedMask, occupiedWhite,
			             opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
	} else {
		generatePseudoLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask,
	 						     king_mask, occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
							     rooksMask, queensMask, kingsMask, ep_square, turn);
	}

	// Same pseudo-legal/evasion set as generateLegalMoves, but stop at the first move that
	// survives is_safe -- the existence answer the checkmate/stalemate callers need.
	for (size_t i = 0; i < startPos.size(); i++){
		if (is_safe(king, blockers, startPos[i], endPos[i], occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask,
			knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn)){
			return true;
		}
	}
	return false;
}

inline void generatePseudoLegalMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 						  uint64_t king, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
							  uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){

	//uint64_t our_pieces = turn ? occupied_white : occupied_black;
	//uint64_t opposingPieces = turn ? occupied_black : occupied_white;

	// Call the function to generate piece moves.
    generatePieceMoves(startPos, endPos, promotions, ourPieces, from_mask, to_mask, occupiedMask, occupiedWhite, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask);
	
	if ((from_mask & kingsMask) != 0){
		generateCastlingMoves(startPos, endPos, promotions, preliminary_castling_mask, to_mask, king, opposingPieces, occupiedMask, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, turn);
		
	}

	uint64_t pawns_mask = pawnsMask & ourPieces & from_mask;
	if(pawns_mask == 0)
		return;

	generatePawnMoves(startPos, endPos, promotions, opposingPieces, turn, pawns_mask, occupiedMask, from_mask, to_mask);
	
	if (ep_square == -1)
		return;
	
	generateEnPassentMoves(startPos, endPos, promotions, from_mask, to_mask, ourPieces, occupiedMask, pawnsMask, ep_square, turn);
	
}

inline void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t our_pieces, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask,
	 					uint64_t occupiedWhite, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask){
    
	/*
		Function to generate moves for non-pawn pieces
		
		Parameters:
		- startPos: An empty vector to hold the starting positions, passed by reference 
		- endPos: An empty vector to hold the ending positions, passed by reference 
		- our_pieces: The mask containing only the pieces of the current side
		- from_mask: The mask of the possible starting positions for move generation
		- to_mask: The mask of the possible ending positions for move generation		
	*/	
	// Define mask of non pawn pieces
	uint64_t non_pawns = (our_pieces & ~pawnsMask) & from_mask;
		
	// Loop through the non pawn pieces
	uint8_t r = 0;
	uint64_t bb = non_pawns;
	while (bb) {
		r = __builtin_ctzll(bb);

		uint64_t mask = (BB_SQUARES[r]);
		
		uint8_t piece_type = 0;
		if (pawnsMask & mask) {
			piece_type = 1;
		} else if (knightsMask & mask){
			piece_type = 2;
		} else if (bishopsMask & mask){
			piece_type = 3;
		} else if (rooksMask & mask){
			piece_type = 4;
		} else if (queensMask & mask){
			piece_type = 5;
		} else if (kingsMask & mask){
			piece_type = 6;
		}
		
		
		// Define the moves as a bitwise and between the squares attacked from the starting square and the starting mask
		uint64_t moves = (attacks_mask(bool((mask) & occupiedWhite),occupiedMask,r,piece_type) & ~our_pieces) & to_mask;
		
		// Loop through the possible destinations
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {
			r_inner = __builtin_ctzll(bb_inner);
			
			// Push the starting and ending positions to their respective vectors
			startPos.push_back(r);
			endPos.push_back(r_inner);
			promotions.push_back(1);  
			bb_inner &= bb_inner - 1;
		}
		
		bb &= bb - 1;
	}
}

inline void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, bool colour, uint64_t pawnsMask, uint64_t occupiedMask,
					   uint64_t from_mask, uint64_t to_mask){    			
	
	/*
		Function to generate moves for pawn pieces
		
		Parameters:
		- startPos: An empty vector to hold the starting positions, passed by reference 
		- endPos: An empty vector to hold the ending positions, passed by reference 
		- promotions: An empty vector to hold the promotion status, passed by reference
		- opposingPieces: The mask containing only the pieces of the opposing side
		- occupied: The mask containing all pieces
		- colour: the colour of the current side
		- pawnsMask: The mask containing only pawns
		- from_mask: The mask of the possible starting positions for move generation
		- to_mask: The mask of the possible ending positions for move generation		
	*/
		
	/*
		This section of code is used for pawn captures
	*/		
	// Loop through the pawns
	uint8_t r = 0;
	uint64_t bb = pawnsMask;
	while (bb) {
		r = __builtin_ctzll(bb);
		
		// Acquire the destinations that follow pawn attacks and opposing pieces
		uint64_t moves = BB_PAWN_ATTACKS[colour][r] & opposingPieces & to_mask;
		
		// Loop through the destinations 
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {			
			r_inner = __builtin_ctzll(bb_inner);
			
			// Check if the rank suggests the move is a promotion
			uint8_t rank = r_inner / 8;			
			if (rank == 7 || rank == 0){
				
				// Loop through all possible promotions
				for (int k = 5; k > 1; k--){
					startPos.push_back(r);
					endPos.push_back(r_inner);
					promotions.push_back(k);
				}
			
			// Else the move is not a promotion
			} else{
				startPos.push_back(r);
				endPos.push_back(r_inner);
				promotions.push_back(1);
			}  
			bb_inner &= bb_inner - 1;
		}
		bb &= bb - 1;		
	}
	
	/*
		In this section, define single and double pawn pushes
	*/
	uint64_t single_moves, double_moves;
	if (colour){
        single_moves = pawnsMask << 8 & ~occupiedMask;
        double_moves = single_moves << 8 & ~occupiedMask & (BB_RANK_3 | BB_RANK_4);
    }else{
        single_moves = pawnsMask >> 8 & ~occupiedMask;
        double_moves = single_moves >> 8 & ~occupiedMask & (BB_RANK_6 | BB_RANK_5);
	}
    
	single_moves &= to_mask;
    double_moves &= to_mask;
	
	/*
		This section of code is used for single pawn pushes
	*/		
	// Loop through the pawns
	r = 0;
	bb = single_moves;
	while (bb) {
		r = __builtin_ctzll(bb);
		
		// Set the destination square as either one square up or down the board depending on the colour
		uint8_t from_square = r;
		if (colour){
			from_square -= 8;
		} else{
			from_square += 8;
		}
		
		// Check if the rank suggests the move is a promotion
		uint8_t rank = r / 8;
		if (rank == 7 || rank == 0){	

			// Loop through all possible promotions		
			for (int j = 5; j > 1; j--){
				startPos.push_back(from_square);
				endPos.push_back(r);
				promotions.push_back(j);
			}
		// Else the move is not a promotion
		} else{
			startPos.push_back(from_square);
			endPos.push_back(r);
			promotions.push_back(1);
		}
		bb &= bb - 1;
	}
	
	/*
		This section of code is used for double pawn pushes
	*/		
	// Loop through the pawns
	r = 0;
	bb = double_moves;
	while (bb) {
		r = __builtin_ctzll(bb);
		
		// Set the destination square as either two squares up or down the board depending on the colour
		uint8_t from_square = r;
		if (colour){
			from_square -= 16;
		} else{
			from_square += 16;
		}
		
		// Set the start and destination
		startPos.push_back(from_square);
		endPos.push_back(r);
		promotions.push_back(1);
		bb &= bb - 1;
	}
	
}

inline void generateCastlingMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint64_t to_mask, uint64_t king,
	                       uint64_t opposingPieces, uint64_t occupiedMask, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, bool turn){

	uint64_t backrank = turn ? BB_RANK_1 : BB_RANK_8;
	uint64_t candidates_mask = preliminary_castling_mask & backrank & to_mask;
	
	if (candidates_mask == 0)
		return;

	uint8_t king_square = __builtin_ctzll(king);

	uint64_t bb_c = BB_FILE_C & backrank;
    uint64_t bb_d = BB_FILE_D & backrank;
    uint64_t bb_f = BB_FILE_F & backrank;
    uint64_t bb_g = BB_FILE_G & backrank;

	while (candidates_mask) {
        // Get least significant bit index (square)
        uint8_t candidate = __builtin_ctzll(candidates_mask);  // GCC/Clang builtin: count trailing zeros

        // Clear the LSB from path
        candidates_mask &= candidates_mask - 1;

		uint64_t rook = BB_SQUARES[candidate];

		bool a_side = candidate < king_square;
		uint64_t king_to = a_side ? bb_c: bb_g;
		uint64_t rook_to = a_side ? bb_d: bb_f;

		uint64_t king_path = betweenPieces(king_square, __builtin_ctzll(king_to));
		uint64_t rook_path = betweenPieces(candidate, __builtin_ctzll(rook_to));

		if (!((occupiedMask ^ king ^ rook) & (king_path | rook_path | king_to | rook_to) || attackedForKing(!turn, king_path | king, occupiedMask ^ king, opposingPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask) || attackedForKing(!turn, king_to, occupiedMask ^ king ^ rook ^ rook_to, opposingPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask))){

			if (king_square == 4 && turn) { // White E1
                if (candidate == 7) {
                    startPos.push_back(4); endPos.push_back(6); promotions.push_back(1); // O-O
                } else if (candidate == 0) {
                    startPos.push_back(4); endPos.push_back(2); promotions.push_back(1); // O-O-O
                }
            } else if (king_square == 60 && !turn) { // Black E8
                if (candidate == 63) {
                    startPos.push_back(60); endPos.push_back(62); promotions.push_back(1);
                } else if (candidate == 56) {
                    startPos.push_back(60); endPos.push_back(58); promotions.push_back(1);
                }
            }
		}
	}
}

inline void generateEnPassentMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t from_mask, uint64_t to_mask, uint64_t our_pieces, uint64_t occupiedMask, uint64_t pawnsMask, int ep_square, bool turn){
	if (ep_square == -1 || (BB_SQUARES[ep_square] & to_mask) == 0)
	    return;

	if ((BB_SQUARES[ep_square] & occupiedMask) != 0)
	    return;

	uint64_t capturers = (
            pawnsMask & our_pieces & from_mask &
            BB_PAWN_ATTACKS[!turn][ep_square] &
            BB_RANKS[turn ? 4 : 3]);
	
	while (capturers) {
        // Get least significant bit index (square)
        uint8_t capturer = __builtin_ctzll(capturers);  // GCC/Clang builtin: count trailing zeros

        // Clear the LSB from path
        capturers &= capturers - 1;

		startPos.push_back(capturer); endPos.push_back(ep_square); promotions.push_back(1);
	}
}

inline void generateEvasions(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint8_t king, uint64_t checkers, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces,
					  uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){
	
    uint64_t king_mask = BB_SQUARES[king];

	// Define mask for sliding pieces which are also checkers
    uint64_t sliders = checkers & (bishopsMask | rooksMask | queensMask);
    
    // Define mask to hold ray attacks towards the king
    uint64_t attacked = 0;

	uint64_t bb = sliders;
	while (bb) {
		uint8_t r = __builtin_ctzll(bb);
		
		// Acquire ray attacks
		attacked |= ray(king, r) & ~BB_SQUARES[r];

		bb &= bb - 1;
	}

	if ((king_mask & from_mask) != 0){
		uint64_t bb = BB_KING_ATTACKS[king] & ~ourPieces & ~attacked & to_mask;
		while (bb) {
			uint8_t r = __builtin_ctzll(bb);
			
			// Add king evasion moves
			startPos.push_back(king); endPos.push_back(r); promotions.push_back(1);

			bb &= bb - 1;
		}
	}

	uint8_t checker = 63 - __builtin_clzll(checkers);

	if (BB_SQUARES[checker] == checkers){

		uint64_t target = betweenPieces(king, checker) | checkers;

		generatePseudoLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, ~kingsMask & from_mask, target & to_mask,
	 						  king_mask, occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
							  rooksMask, queensMask, kingsMask, ep_square, turn);
		
		if (ep_square != -1 && ((BB_SQUARES[ep_square] & target) == 0)){
			int last_double = ep_square + (turn ? -8 : 8);

			if (last_double == checker)
				generateEnPassentMoves(startPos, endPos, promotions, from_mask, to_mask, ourPieces, occupiedMask, pawnsMask, ep_square, turn);
		}
	}
}

void generateLegalCaptures(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){
	
	generateLegalMoves(startPos_filtered, endPos_filtered, promotions_filtered, 0, from_mask, to_mask & opposingPieces,
	 				   occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
					   rooksMask, queensMask, kingsMask, ep_square, turn);

	if (ep_square == -1)
		return;

	std::vector<uint8_t> startPos;
	std::vector<uint8_t> endPos;
	std::vector<uint8_t> promotions;	

	generateEnPassentMoves(startPos, endPos, promotions, from_mask, to_mask, ourPieces, occupiedMask, pawnsMask, ep_square, turn);

	for (size_t i = 0; i < startPos.size(); i++){
		if (!is_into_check(startPos[i], endPos[i], occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn)){
			
			startPos_filtered.push_back(startPos[i]);
			endPos_filtered.push_back(endPos[i]);
			promotions_filtered.push_back(promotions[i]);
		}
	}
}

inline void generateLegalMovesReordered(std::vector<Move>& converted_moves, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, int ply, Move prevMove,
								 bool noisy_only = false) {
	PROF_BLOCK(PROF_MOVEGEN);
	
	// Determine if the game is at the endgame phase as well as an advanced endgame phase
	bool isEndGame;
	// isNearGameEnd was left uninitialised for phase_score 63..96 and only ever assigned true. The compiled code has
	// always behaved as if it were true there: the bench reproduces with true and not with false. Initialised to true,
	// so every endgame uses the king-first quiet order below, as it always has; the "normal endgame" order (minors
	// first, king last) is unreachable. Giving 63..96 that order is a search change, to be tested on its own.
	bool isNearGameEnd = true;

	int phase = 0;
	phase += 4 * __builtin_popcountll(queensMask);
	phase += 2 * __builtin_popcountll(rooksMask);
	phase += 1 * __builtin_popcountll(bishopsMask | knightsMask);

	int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to 128

	if (phase_score <= 62) {
    	// Midgame
		isEndGame = false;
	} else if (phase_score <= 96) {
		// Normal endgame
		isEndGame = true;
	} else {
		isEndGame = true;
		isNearGameEnd = true;
	}

	// Per-node scratch reused across calls (cleared, capacity retained) to avoid a fresh
	// heap allocation every node. Single-threaded search; thread_local keeps it correct if
	// the work is ever parallelised.
	thread_local std::vector<uint8_t> startPos;
	thread_local std::vector<uint8_t> endPos;
	thread_local std::vector<uint8_t> promotions;

	startPos.clear();
	endPos.clear();
	promotions.clear();
	startPos.reserve(64);
	endPos.reserve(64);
	promotions.reserve(64);

	{
	PROF_BLOCK(PROF_MG_GEN);
	// Slider blockers + checker mask + king square are node-invariant (identical across the attack and
	// quiet gen passes below); compute once here and pass into every processMaskPairs call.
	uint8_t king = 63 - __builtin_clzll(kingsMask & ourPieces);
	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);
	uint64_t checkers = attackersMask(!turn, king, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask, pawnsMask, opposingPieces);

	std::array<MaskPair, 12> attack_pairs = {
		MaskPair(pawnsMask, (queensMask | rooksMask | bishopsMask | knightsMask) & opposingPieces),

		MaskPair(knightsMask | bishopsMask, (queensMask | rooksMask) & opposingPieces),

		MaskPair(rooksMask, queensMask & opposingPieces),

		MaskPair(knightsMask | bishopsMask, (knightsMask | bishopsMask) & opposingPieces),

		MaskPair(pawnsMask, pawnsMask & opposingPieces),
		MaskPair(rooksMask, rooksMask & opposingPieces),
		MaskPair(queensMask, queensMask & opposingPieces),

		MaskPair(knightsMask | bishopsMask, pawnsMask & opposingPieces),

		MaskPair(rooksMask, (bishopsMask | knightsMask) & opposingPieces),

		MaskPair(queensMask, (rooksMask | bishopsMask | knightsMask) & opposingPieces),

		MaskPair(rooksMask | queensMask, pawnsMask & opposingPieces),

		MaskPair(kingsMask, (queensMask | rooksMask | bishopsMask | knightsMask | pawnsMask) & opposingPieces)
	};

	processMaskPairs(attack_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
					 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
	
	/* generateLegalMoves(preliminaryStartPos, preliminaryEndPos, preliminaryPromotions, preliminary_castling_mask,
                           from_mask & ourPieces,
                           to_mask & opposingPieces,
                           occupiedMask, occupiedWhite, opposingPieces, ourPieces,
                           pawnsMask, knightsMask, bishopsMask, rooksMask,
                           queensMask, kingsMask, ep_square, turn); */
	/* generateLegalCaptures(preliminaryStartPos, preliminaryEndPos, preliminaryPromotions, from_mask, to_mask,
	 					  occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, 
						  bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn); */

	if (noisy_only){
		// Noisy-only list for qsearch: the attack passes above already hold every capture (incl. capture-promotions). The only
		// other noisy moves come from the pawn QUIET pass: pushes onto the promotion rank and en passant (its target square is
		// empty, so no attack pass sees it). Restricting that one pass to those squares yields exactly the noisy subset of the
		// full list in the same relative build order; scores are per-move and the sort below is stable, so the final order of
		// the noisy moves is identical to their order in the full list.
		uint64_t noisy_quiet_to = (turn ? 0xFF00000000000000ULL : 0x00000000000000FFULL) | (ep_square >= 0 ? BB_SQUARES[ep_square] : 0ULL);
		std::array<MaskPair, 1> promo_ep_pair = { MaskPair(pawnsMask, ~opposingPieces & noisy_quiet_to) };
		processMaskPairs(promo_ep_pair, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
						 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
	} else if (!isEndGame){
		std::array<MaskPair, 4> quiet_pairs = {
			MaskPair(knightsMask | bishopsMask, ~opposingPieces),
			MaskPair(pawnsMask, ~opposingPieces),
			MaskPair(queensMask, ~opposingPieces),
			MaskPair(rooksMask | kingsMask, ~opposingPieces)
		};

		processMaskPairs(quiet_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
						 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);

	} else{
		if (isNearGameEnd){
			std::array<MaskPair, 4> quiet_pairs = {

				// 2. King movement (positional)
				MaskPair(kingsMask, ~opposingPieces),

				// 4. Pawn pushes (promotion racing)
				MaskPair(pawnsMask, ~opposingPieces),

				// 5. Rook/queen quiet moves (only if still on board)
				MaskPair(rooksMask | queensMask, ~opposingPieces),

				// 6. Minor piece quiet moves
				MaskPair(knightsMask | bishopsMask, ~opposingPieces)
			};

			processMaskPairs(quiet_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
		} else {
			std::array<MaskPair, 4> quiet_pairs = {

				// 3. Minor piece quiet moves
				MaskPair(knightsMask | bishopsMask, ~opposingPieces),

				// 4. Pawn pushes (promotion racing)
				MaskPair(pawnsMask, ~opposingPieces),

				// 5. Rook/queen quiet moves (only if still on board)
				MaskPair(rooksMask | queensMask, ~opposingPieces),

				// 6. King movement (positional)
				MaskPair(kingsMask, ~opposingPieces),		

			};
			processMaskPairs(quiet_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
		}
	}
	} // PROF_MG_GEN

	BoardState state(
		pawnsMask,
		knightsMask,
		bishopsMask,
		rooksMask,
		queensMask,
		kingsMask,
		occupiedWhite,
		~occupiedWhite & occupiedMask,
		occupiedMask,
		1,
		turn,                  
		0,
		0,                          
		0,                           
		0    
	);

	// Enemy king square for the quiet-check ordering bonus (computed once; only when enabled).
	int enemy_king_sq = -1;
	if (Config::ENABLE_CHECK_ORDER)
	{
		uint64_t ek = kingsMask & opposingPieces;
		if (ek)
			enemy_king_sq = 63 - __builtin_clzll(ek);
	}

	// Opponent threats bitboard for threat-conditioned quiet history (computed once; only when enabled).
	uint64_t node_threats = 0;
	if (Config::ENABLE_THREAT_HIST)
		node_threats = opponent_threats(!turn, pawnsMask & opposingPieces, knightsMask & opposingPieces,
			bishopsMask & opposingPieces, rooksMask & opposingPieces, queensMask & opposingPieces,
			kingsMask & opposingPieces, occupiedMask);

	// Squares the opponent attacks with a pawn; with a pawn or minor; with a pawn, minor or rook. These are the
	// "cheaper attacker" sets for a minor, a rook and a queen in the threat-aware quiet ordering below.
	uint64_t threat_by_pawn = 0, threat_by_minor = 0, threat_by_rook = 0;
	if (Config::THREAT_ORDER_W > 0)
	{
		uint64_t b;
		for (b = pawnsMask & opposingPieces; b; b &= b - 1) threat_by_pawn |= BB_PAWN_ATTACKS[!turn][__builtin_ctzll(b)];
		threat_by_minor = threat_by_pawn;
		for (b = knightsMask & opposingPieces; b; b &= b - 1) threat_by_minor |= BB_KNIGHT_ATTACKS[__builtin_ctzll(b)];
		for (b = bishopsMask & opposingPieces; b; b &= b - 1) { int s = __builtin_ctzll(b); threat_by_minor |= BB_DIAG_ATTACKS[s][BB_DIAG_MASKS[s] & occupiedMask]; }
		threat_by_rook = threat_by_minor;
		for (b = rooksMask & opposingPieces; b; b &= b - 1) { int s = __builtin_ctzll(b); threat_by_rook |= BB_RANK_ATTACKS[s][BB_RANK_MASKS[s] & occupiedMask] | BB_FILE_ATTACKS[s][BB_FILE_MASKS[s] & occupiedMask]; }
	}

	auto score_move = [&](size_t i) -> int {
		uint8_t from = startPos[i];
		uint8_t to = endPos[i];

		if(BB_SQUARES[to] & opposingPieces){
			/* int score = get_see_score(preliminaryEndPos[i], turn, state);
			//int score = 100000;
			return score < 0? score + preliminaryPromotions[i] * 5000: 15000 + score + preliminaryPromotions[i] * 5000; */

			int value_captured = get_value_at(to, state);
			int value_attacker = get_value_at(from, state);
			int promo_bonus = (promotions[i] - 1) * 5000;
			int capture_base_value = 200000;
			int move_freq_bonus = mf_get(turn, from, to);
			// Capture history refines ordering WITHIN the static capture tiers; bad captures rarely
			// accumulate it, so they stay below quiets in practice. Gated, default off = byte-identical.
			int cap_hist = Config::ENABLE_CAPTURE_HIST ? capture_hist_ref_bb(turn, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask) : 0;
			//return 15000 + value_captured - value_attacker + promo_bonus;
			if (value_captured >= value_attacker) {
				// Clearly good capture, skip SEE
				return capture_base_value + value_captured - value_attacker + promo_bonus + move_freq_bonus + cap_hist;
			} else {
#ifdef SV2_MOVE_PROBE
				// S3 step 0: does a cheaper piece of ours also attack the square? Then the per-square exchange value below
				// describes that piece's capture, not this move's.
				++g_sv2_probe.see_branch;
				for (uint64_t ours = attackersMask(turn, to, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask,
				                                   pawnsMask, ourPieces) & ~BB_SQUARES[from]; ours; ours &= ours - 1)
					if (get_value_at(__builtin_ctzll(ours), state) < value_attacker)
					{
						++g_sv2_probe.see_cheaper_attacker;
						break;
					}
#endif
				// Unclear or losing capture, run SEE: the square's value, or this move's own (SEE_PER_MOVE bit 1)
				int see_score = (Config::SEE_PER_MOVE & 1) ? see_move(from, to, turn, state) : see(to, turn, state);
#ifdef SV2_SEE_VERIFY
				// When this piece is the side's least valuable attacker, the move's value must equal the square's.
				{
					const uint64_t ours = attackersMask(turn, to, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask,
					                                    pawnsMask, ourPieces);
					if (ours && get_least_valuable_attacker_static(ours, state) == (int)from)
					{
						++g_sv2_see_checked;
						g_sv2_see_mismatch += (see_move(from, to, turn, state) != see(to, turn, state));
					}
				}
#endif
				return see_score < 0 ? see_score + promo_bonus + move_freq_bonus + cap_hist : capture_base_value + see_score + promo_bonus + move_freq_bonus + cap_hist;
			}
		}

		Move cur(from, to, promotions[i]);
		int counterMoveBonus = 0;
		if(counterMoves[prevMove.from_square][prevMove.to_square] == cur){
			counterMoveBonus = 8000;
		}
		int promo_bonus = (promotions[i] - 1) * 5000;
		// 2-ply continuation history: keyed by the move 2 plies back (g_searchStack[ply-2]) x this move.
		// Gated, default off = byte-identical (the stack is neither pushed nor read when off).
		int cont2 = 0;
		if (Config::ENABLE_CONT_HIST_2PLY && ply >= 2)
		{
			Move p2 = g_searchStack[ply - 2];
			if (p2.from_square != p2.to_square)
				cont2 = ch2_ref_bb(turn, p2, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask);
		}
		// Ordering bonus for a quiet move that gives a DIRECT check (the mover, landing on `to`, attacks
		// the enemy king). Misses discovered checks. Gated, default off = byte-identical.
		int check_bonus = 0;
		if (Config::ENABLE_CHECK_ORDER && enemy_king_sq >= 0)
		{
			uint64_t fromBB = BB_SQUARES[from];
			uint64_t occ = occupiedMask & ~fromBB;   // the mover has vacated `from`
			uint64_t atk = 0;
			if (knightsMask & fromBB)
				atk = BB_KNIGHT_ATTACKS[to];
			else if (pawnsMask & fromBB)
				atk = BB_PAWN_ATTACKS[turn][to];
			else if (bishopsMask & fromBB)
				atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ];
			else if (rooksMask & fromBB)
				atk = BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
			else if (queensMask & fromBB)
				atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ] | BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
			if (atk & BB_SQUARES[enemy_king_sq])
				check_bonus = Config::CHECK_ORDER_BONUS;
		}
		// Escape from / step onto a square attacked by a cheaper enemy piece (pawns and the king are never scored).
		int threat_order = 0;
		if (Config::THREAT_ORDER_W > 0)
		{
			uint64_t fromBB = BB_SQUARES[from];
			uint64_t lesser = 0;
			int value = 0;
			if (knightsMask & fromBB)      { lesser = threat_by_pawn;  value = values[KNIGHT]; }
			else if (bishopsMask & fromBB) { lesser = threat_by_pawn;  value = values[BISHOP]; }
			else if (rooksMask & fromBB)   { lesser = threat_by_minor; value = values[ROOK]; }
			else if (queensMask & fromBB)  { lesser = threat_by_rook;  value = values[QUEEN]; }
			if (lesser & BB_SQUARES[to])
				threat_order = -value * Config::THREAT_ORDER_W * 19 / 200;
			else if (lesser & fromBB)
				threat_order = value * Config::THREAT_ORDER_W / 10;
		}
		int pcont = 0;
		if (Config::ENABLE_PIECE_CONTHIST)
			pcont = pieceContHist[turn][pcont_ctx_key_bb(prevMove, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)][pcont_ent_key_bb(from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)] >> Config::PIECE_CONTHIST_SHIFT;
		int bh = Config::ENABLE_THREAT_HIST ? threat_hist_term(turn, node_threats, from, to) : historyHeuristics[turn][from][to];
		int static_tb = 0;
		if (Config::ENABLE_STATIC_ORDER && (bh <= Config::STATIC_ORDER_HIST_MAX) && (bh >= -Config::STATIC_ORDER_HIST_MAX))
		{
			++g_static_order_eligible;
			static_tb = staticPlacementScore(turn, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask);
			if (static_tb != 0) ++g_static_order_fires;
		}
		return bh + static_tb + killerBonus(ply, cur)
			   + counterMoveBonus + cmh_get_bb(turn, prevMove, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)
			   + cont2 + check_bonus + threat_order + pcont
			   + promo_bonus + mf_get(turn, from, to);
	};

	// Score each move once up front — score_move runs SEE for unclear captures, so
	// calling it inside the comparator would repeat that work O(log n) times per move
	thread_local std::vector<ScoredMove> scored;
	size_t move_count = startPos.size();
	scored.resize(move_count);
	{
	PROF_BLOCK(PROF_MG_SCORE);
	for (size_t i = 0; i < move_count; ++i)
		scored[i] = ScoredMove{ score_move(i), startPos[i], endPos[i], promotions[i] };
	}

	{
	PROF_BLOCK(PROF_MG_SORT);
	// stable_sort over the 0..N-1 build order matches the previous index-based reorder exactly.
	if (Config::SCRATCH_REUSE) {
		thread_local std::vector<ScoredMove> expected;
		if (Config::SCRATCH_REUSE == 2) {
			expected = scored;
			std::stable_sort(expected.begin(), expected.end(), [](const ScoredMove& a, const ScoredMove& b) {
				return a.score > b.score;
			});
		}
		stableSortScoredDesc(scored);
		if (Config::SCRATCH_REUSE == 2) {
			for (size_t i = 0; i < move_count; ++i)
				if (scored[i].score != expected[i].score || scored[i].from != expected[i].from || scored[i].to != expected[i].to
				    || scored[i].promo != expected[i].promo) {
					std::cerr << "SCRATCH_REUSE sort mismatch at " << i << " of " << move_count << std::endl;
					std::abort();
				}
		}
	} else
		std::stable_sort(scored.begin(), scored.end(), [](const ScoredMove& a, const ScoredMove& b) {
			return a.score > b.score;
		});

	for (const ScoredMove& s : scored) {
		converted_moves.push_back(Move(s.from, s.to, s.promo));
	}
#ifdef SV2_MOVE_PROBE
	// S3 step 0, option B: does a non-capture sort above a capture in the good tier (score >= 200000)? Two-phase list
	// building (captures first) would order such a list differently from today.
	if (!noisy_only)
	{
		bool seen_quiet = false, quiet_above = false;
		for (const ScoredMove &s : scored)
		{
			if (!(BB_SQUARES[s.to] & opposingPieces))
				seen_quiet = true;
			else if (seen_quiet && s.score >= 200000)
			{
				quiet_above = true;
				break;
			}
		}
		++g_sv2_probe.sorted_lists;
		g_sv2_probe.quiet_above_good_capture += quiet_above;
	}
#endif
	}

}

/* Live ordering score for a QUIET move, mirroring the quiet branch of generateLegalMovesReordered's
   score_move lambda above (history + killer + counter + 2-ply continuation + check + promo + freq).
   Used by the lazy cached-tail re-sort in buildMoveListFromReordered to re-rank stale quiets against
   the CURRENT history tables. Keep the term list in sync with that lambda (except the threat-aware term,
   THREAT_ORDER_W, which needs the node's attack sets and is static per position). Captures are never passed
   here (the re-sort only touches the quiet tail). enemy_king_sq is precomputed by the caller (-1 when
   the check-ordering feature is off). */
inline int score_quiet(uint8_t from, uint8_t to, uint8_t promo, bool turn, int ply, Move prevMove,
					   uint64_t occupiedMask, uint64_t pawnsMask, uint64_t knightsMask,
					   uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, int enemy_king_sq,
					   uint64_t node_threats)
{
	Move cur(from, to, promo);
	// Kings mask isn't a parameter here; recover it by elimination for the piece-key helpers.
	uint64_t kingsMask = occupiedMask & ~(pawnsMask | knightsMask | bishopsMask | rooksMask | queensMask);
	int counterMoveBonus = 0;
	if (counterMoves[prevMove.from_square][prevMove.to_square] == cur)
		counterMoveBonus = 8000;
	int promo_bonus = (promo - 1) * 5000;
	int cont2 = 0;
	if (Config::ENABLE_CONT_HIST_2PLY && ply >= 2)
	{
		Move p2 = g_searchStack[ply - 2];
		if (p2.from_square != p2.to_square)
			cont2 = ch2_ref_bb(turn, p2, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask);
	}
	int check_bonus = 0;
	if (Config::ENABLE_CHECK_ORDER && enemy_king_sq >= 0)
	{
		uint64_t fromBB = BB_SQUARES[from];
		uint64_t occ = occupiedMask & ~fromBB;   // the mover has vacated `from`
		uint64_t atk = 0;
		if (knightsMask & fromBB)
			atk = BB_KNIGHT_ATTACKS[to];
		else if (pawnsMask & fromBB)
			atk = BB_PAWN_ATTACKS[turn][to];
		else if (bishopsMask & fromBB)
			atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ];
		else if (rooksMask & fromBB)
			atk = BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
		else if (queensMask & fromBB)
			atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ] | BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
		if (atk & BB_SQUARES[enemy_king_sq])
			check_bonus = Config::CHECK_ORDER_BONUS;
	}
	int pcont = 0;
	if (Config::ENABLE_PIECE_CONTHIST)
		pcont = pieceContHist[turn][pcont_ctx_key_bb(prevMove, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)][pcont_ent_key_bb(from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)] >> Config::PIECE_CONTHIST_SHIFT;
	int bh = Config::ENABLE_THREAT_HIST ? threat_hist_term(turn, node_threats, from, to) : historyHeuristics[turn][from][to];
	int static_tb = 0;
	if (Config::ENABLE_STATIC_ORDER && (bh <= Config::STATIC_ORDER_HIST_MAX) && (bh >= -Config::STATIC_ORDER_HIST_MAX))
	{
		++g_static_order_eligible;
		static_tb = staticPlacementScore(turn, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask);
		if (static_tb != 0) ++g_static_order_fires;
	}
	return bh + static_tb + killerBonus(ply, cur)
		   + counterMoveBonus + cmh_get_bb(turn, prevMove, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)
		   + cont2 + check_bonus + pcont
		   + promo_bonus + mf_get(turn, from, to);
}

inline void generateLegalMovesReordered1(std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, int ply, Move prevMove) {
	PROF_BLOCK(PROF_MOVEGEN);

	// Slider blockers + checker mask + king square are node-invariant (identical across the attack and
	// quiet gen passes below); compute once here and pass into every processMaskPairs call.
	uint8_t king = 63 - __builtin_clzll(kingsMask & ourPieces);
	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);
	uint64_t checkers = attackersMask(!turn, king, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask, pawnsMask, opposingPieces);

	// Determine if the game is at the endgame phase as well as an advanced endgame phase
	bool isEndGame;
	bool isNearGameEnd;

	int phase = 0;
	phase += 4 * __builtin_popcountll(queensMask);
	phase += 2 * __builtin_popcountll(rooksMask);
	phase += 1 * __builtin_popcountll(bishopsMask | knightsMask);

	int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to 128

	if (phase_score <= 62) {
    	// Midgame
		isEndGame = false;
	} else if (phase_score <= 96) {
		// Normal endgame
		isEndGame = true;
	} else {		
		isEndGame = true;
		isNearGameEnd = true;
	}
	
	std::vector<uint8_t> quietStartPos;
	std::vector<uint8_t> quietEndPos;
	std::vector<uint8_t> quietPromotions;
	

	std::array<MaskPair, 12> attack_pairs = {
		MaskPair(pawnsMask, (queensMask | rooksMask | bishopsMask | knightsMask) & opposingPieces),

		MaskPair(knightsMask | bishopsMask, (queensMask | rooksMask) & opposingPieces),

		MaskPair(rooksMask, queensMask & opposingPieces),

		MaskPair(knightsMask | bishopsMask, (knightsMask | bishopsMask) & opposingPieces),

		MaskPair(pawnsMask, pawnsMask & opposingPieces),
		MaskPair(rooksMask, rooksMask & opposingPieces),
		MaskPair(queensMask, queensMask & opposingPieces),

		MaskPair(knightsMask | bishopsMask, pawnsMask & opposingPieces),

		MaskPair(rooksMask, (bishopsMask | knightsMask) & opposingPieces),

		MaskPair(queensMask, (rooksMask | bishopsMask | knightsMask) & opposingPieces),

		MaskPair(rooksMask | queensMask, pawnsMask & opposingPieces),

		MaskPair(kingsMask, (queensMask | rooksMask | bishopsMask | knightsMask | pawnsMask) & opposingPieces)
	};

	processMaskPairs(attack_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
					 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);

	if (!isEndGame){
		std::array<MaskPair, 4> quiet_pairs = {
			MaskPair(knightsMask | bishopsMask, ~opposingPieces),
			MaskPair(pawnsMask, ~opposingPieces),
			MaskPair(queensMask, ~opposingPieces),
			MaskPair(rooksMask | kingsMask, ~opposingPieces)
		};

		processMaskPairs(quiet_pairs, quietStartPos, quietEndPos, quietPromotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
								opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);

	} else{
		if (isNearGameEnd){
			std::array<MaskPair, 4> quiet_pairs = {

				// 2. King movement (positional)
				MaskPair(kingsMask, ~opposingPieces),

				// 4. Pawn pushes (promotion racing)
				MaskPair(pawnsMask, ~opposingPieces),

				// 5. Rook/queen quiet moves (only if still on board)
				MaskPair(rooksMask | queensMask, ~opposingPieces),

				// 6. Minor piece quiet moves
				MaskPair(knightsMask | bishopsMask, ~opposingPieces)
			};

			processMaskPairs(quiet_pairs, quietStartPos, quietEndPos, quietPromotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
		} else {
			std::array<MaskPair, 4> quiet_pairs = {

				// 3. Minor piece quiet moves
				MaskPair(knightsMask | bishopsMask, ~opposingPieces),

				// 4. Pawn pushes (promotion racing)
				MaskPair(pawnsMask, ~opposingPieces),

				// 5. Rook/queen quiet moves (only if still on board)
				MaskPair(rooksMask | queensMask, ~opposingPieces),

				// 6. King movement (positional)
				MaskPair(kingsMask, ~opposingPieces),		

			};
			processMaskPairs(quiet_pairs, quietStartPos, quietEndPos, quietPromotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
		}
	}

	std::vector<size_t> indices(quietStartPos.size());
	std::iota(indices.begin(), indices.end(), 0); // Fill with 0..N-1

	uint64_t node_threats = 0;
	if (Config::ENABLE_THREAT_HIST)
		node_threats = opponent_threats(!turn, pawnsMask & opposingPieces, knightsMask & opposingPieces,
			bishopsMask & opposingPieces, rooksMask & opposingPieces, queensMask & opposingPieces,
			kingsMask & opposingPieces, occupiedMask);

	auto score_move = [&](size_t i) -> int {
		uint8_t from = quietStartPos[i];
		uint8_t to = quietEndPos[i];

		Move cur(quietStartPos[i], quietEndPos[i], quietPromotions[i]);
		int counterMoveBonus = 0;
		if(counterMoves[prevMove.from_square][prevMove.to_square] == cur){
			counterMoveBonus = 8000;
		}

		int pcont = 0;
		if (Config::ENABLE_PIECE_CONTHIST)
			pcont = pieceContHist[turn][pcont_ctx_key_bb(prevMove, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)][pcont_ent_key_bb(from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)] >> Config::PIECE_CONTHIST_SHIFT;
		int bh = Config::ENABLE_THREAT_HIST ? threat_hist_term(turn, node_threats, from, to) : historyHeuristics[turn][from][to];
		return bh + killerBonus(ply, Move(quietStartPos[i], quietEndPos[i], quietPromotions[i]))
			   + counterMoveBonus + cmh_get_bb(turn, prevMove, from, to, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask)
			   + pcont + quietPromotions[i] * 5000 + mf_get(turn, from, to);
	};

	std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return score_move(a) > score_move(b);
	});

	for (size_t i : indices) {
		startPos.push_back(quietStartPos[i]);
		endPos.push_back(quietEndPos[i]);
		promotions.push_back(quietPromotions[i]);
	}
}


template<std::size_t N>
void processMaskPairs(const std::array<MaskPair, N>& mask_pairs, std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask,
	                  uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
                      uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, uint8_t king, uint64_t blockers, uint64_t checkers){
    // King square, slider blockers and checker mask are invariant across the piece-type pairs of this
    // node AND across the attack/quiet gen split, so they are computed once by the caller and passed in.

    for (const MaskPair& pair : mask_pairs) {
        uint64_t from = pair.from_mask;
        uint64_t to = pair.to_mask;

        if ((from & ourPieces) == 0 || to == 0) continue;

        generateLegalMovesPre(startPos, endPos, promotions, preliminary_castling_mask,
                           from_mask & from & ourPieces,
                           to_mask & to,
                           occupiedMask, occupiedWhite, opposingPieces, ourPieces,
                           pawnsMask, knightsMask, bishopsMask, rooksMask,
                           queensMask, kingsMask, ep_square, turn, king, blockers, checkers);
    }
}
#endif