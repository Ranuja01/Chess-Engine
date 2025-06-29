
#ifndef MOVE_GEN_H
#define MOVE_GEN_H

#include "cpp_bitboard.h"
#include "cache_management.h"
#include <vector>
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
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_DIAG_ATTACKS;
extern std::vector<uint64_t> BB_FILE_MASKS;
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_FILE_ATTACKS;
extern std::vector<uint64_t> BB_RANK_MASKS;
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_RANK_ATTACKS;
extern std::vector<std::vector<uint64_t>> BB_RAYS;

/*
	Set of functions used to generate moves
*/

inline void generateLegalMoves(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){
    
	std::vector<uint8_t> startPos;
	std::vector<uint8_t> endPos;
	std::vector<uint8_t> promotions;

	uint64_t king_mask = kingsMask & ourPieces;
	uint8_t king = 63 - __builtin_clzll(king_mask);

    
	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);            
    uint64_t checkers = attackersMask(!turn, king, occupiedMask, queensMask | rooksMask, queensMask | bishopsMask, kingsMask, knightsMask, pawnsMask, opposingPieces);

	if (checkers != 0){
		generateEvasions(startPos, endPos, promotions, preliminary_castling_mask, king, checkers, from_mask, to_mask, occupiedMask,occupiedWhite,
			             opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);

		for (size_t i = 0; i < startPos.size(); i++){
			if (is_safe(king, blockers, startPos[i], endPos[i], occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask,
				knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn)){
				
				startPos_filtered.push_back(startPos[i]);
				endPos_filtered.push_back(endPos[i]);
				promotions_filtered.push_back(promotions[i]);

			}
		}
	} else {
		generatePseudoLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask,
	 						     king_mask, occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
							     rooksMask, queensMask, kingsMask, ep_square, turn);

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
		uint64_t moves = (attacks_mask(bool((1ULL<<r) & occupiedWhite),occupiedMask,r,piece_type) & ~our_pieces) & to_mask;		
		
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
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, int ply, Move prevMove) {
	
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

	std::vector<uint8_t> startPos;
	std::vector<uint8_t> endPos;
	std::vector<uint8_t> promotions;

	startPos.reserve(64);
	endPos.reserve(64);
	promotions.reserve(64);
	
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
					 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
	
	/* generateLegalMoves(preliminaryStartPos, preliminaryEndPos, preliminaryPromotions, preliminary_castling_mask,
                           from_mask & ourPieces,
                           to_mask & opposingPieces,
                           occupiedMask, occupiedWhite, opposingPieces, ourPieces,
                           pawnsMask, knightsMask, bishopsMask, rooksMask,
                           queensMask, kingsMask, ep_square, turn); */
	/* generateLegalCaptures(preliminaryStartPos, preliminaryEndPos, preliminaryPromotions, from_mask, to_mask,
	 					  occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, 
						  bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn); */

	if (!isEndGame){
		std::array<MaskPair, 4> quiet_pairs = {
			MaskPair(knightsMask | bishopsMask, ~opposingPieces),
			MaskPair(pawnsMask, ~opposingPieces),
			MaskPair(queensMask, ~opposingPieces),
			MaskPair(rooksMask | kingsMask, ~opposingPieces)
		};

		processMaskPairs(quiet_pairs, startPos, endPos, promotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
						 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);

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
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
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
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);	
		}
	}

	

	std::vector<size_t> indices(startPos.size());
	std::iota(indices.begin(), indices.end(), 0); // Fill with 0..N-1

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
			int move_freq_bonus = moveFrequency[turn][from][to];
			//return 15000 + value_captured - value_attacker + promo_bonus;
			if (value_captured >= value_attacker) {
				// Clearly good capture, skip SEE
				return capture_base_value + value_captured - value_attacker + promo_bonus + move_freq_bonus;
			} else {
				// Unclear or losing capture, run SEE
				int see_score = see(to, turn, state);
				return see_score < 0 ? see_score + promo_bonus + move_freq_bonus : capture_base_value + see_score + promo_bonus + move_freq_bonus;
			}

		}

		Move cur(from, to, promotions[i]);
		int counterMoveBonus = 0;
		if(counterMoves[prevMove.from_square][prevMove.to_square] == cur){
			counterMoveBonus = 8000;
		}
		int promo_bonus = (promotions[i] - 1) * 5000;
		return historyHeuristics[turn][from][to] + killerBonus(ply, cur)
			   + counterMoveBonus + counterMoveHeuristics[turn][prevMove.from_square * 64 + prevMove.to_square][from * 64 + to]
			   + promo_bonus + moveFrequency[turn][from][to];
	};

	std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return score_move(a) > score_move(b);
	});

	/* std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return score_move(a) > score_move(b);
	}); */
	for (size_t i : indices) {
		converted_moves.push_back(Move(startPos[i],endPos[i],promotions[i]));
	}	

}

inline void generateLegalMovesReordered1(std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, int ply, Move prevMove) {

	
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
					 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);

	if (!isEndGame){
		std::array<MaskPair, 4> quiet_pairs = {
			MaskPair(knightsMask | bishopsMask, ~opposingPieces),
			MaskPair(pawnsMask, ~opposingPieces),
			MaskPair(queensMask, ~opposingPieces),
			MaskPair(rooksMask | kingsMask, ~opposingPieces)
		};

		processMaskPairs(quiet_pairs, quietStartPos, quietEndPos, quietPromotions, preliminary_castling_mask, from_mask, to_mask, occupiedMask, occupiedWhite,
								opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);

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
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
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
							 opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);	
		}
	}

	std::vector<size_t> indices(quietStartPos.size());
	std::iota(indices.begin(), indices.end(), 0); // Fill with 0..N-1

	auto score_move = [&](size_t i) -> int {
		uint8_t from = quietStartPos[i];
		uint8_t to = quietEndPos[i];

		Move cur(quietStartPos[i], quietEndPos[i], quietPromotions[i]);
		int counterMoveBonus = 0;
		if(counterMoves[prevMove.from_square][prevMove.to_square] == cur){
			counterMoveBonus = 8000;
		}

		return historyHeuristics[turn][from][to] + killerBonus(ply, Move(quietStartPos[i], quietEndPos[i], quietPromotions[i]))
			   + counterMoveBonus + counterMoveHeuristics[turn][prevMove.from_square * 64 + prevMove.to_square][from * 64 + to]
			   + quietPromotions[i] * 5000 + moveFrequency[turn][from][to];
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
                      uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){
    for (const MaskPair& pair : mask_pairs) {
        uint64_t from = pair.from_mask;
        uint64_t to = pair.to_mask;

        if ((from & ourPieces) == 0 || to == 0) continue;

        generateLegalMoves(startPos, endPos, promotions, preliminary_castling_mask,
                           from_mask & from & ourPieces,
                           to_mask & to,
                           occupiedMask, occupiedWhite, opposingPieces, ourPieces,
                           pawnsMask, knightsMask, bishopsMask, rooksMask,
                           queensMask, kingsMask, ep_square, turn);
    }
}
#endif