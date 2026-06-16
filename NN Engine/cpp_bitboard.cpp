/* cpp_wrapper.cpp

@author: Ranuja Pinnaduwage

This file contains c++ code to emulate the python-chess components for generating legal moves as well as functions for evaluating a position

Code augmented from python-chess: https://github.com/niklasf/python-chess/tree/5826ef5dd1c463654d2479408a7ddf56a91603d6

*/

#include "cpp_bitboard.h"
#include "search_engine.h"
#include "cache_management.h"
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <omp.h>
#include <numeric>
#include <execution>
#include <random>
#include <deque>
#include <string>
#include <cstring>
#include <optional>
#include <sstream>
#include <immintrin.h>
#include <stdint.h>


// Define masks for move generation
std::array<uint64_t, NUM_SQUARES> BB_KNIGHT_ATTACKS;
std::array<uint64_t, NUM_SQUARES> BB_KING_ATTACKS;
// 2-ring around each king square = the exact set of squares the king-zone loops in setAttackingLayer
// read; used as the midgame attack-layer cache key (built in initialize_attack_tables).
std::array<uint64_t, NUM_SQUARES> king_ring2;
// Forward 3-file span ahead of a pawn on each square (its own file + both neighbours, all ranks toward
// promotion). A pawn is passed iff no enemy pawn occupies this span. Mirrors getPPIncrement's mask shape;
// built in initialize_attack_tables and consumed by the search's passed-pawn pruning exemption.
std::array<uint64_t, NUM_SQUARES> passed_span_white;
std::array<uint64_t, NUM_SQUARES> passed_span_black;
std::array<std::array<uint64_t, NUM_SQUARES>, 2> BB_PAWN_ATTACKS;
std::vector<uint64_t> BB_DIAG_MASKS;
std::vector<SlidingRow> BB_DIAG_ATTACKS;
std::vector<uint64_t> BB_FILE_MASKS;
std::vector<SlidingRow> BB_FILE_ATTACKS;
std::vector<uint64_t> BB_RANK_MASKS;
std::vector<SlidingRow> BB_RANK_ATTACKS;
std::vector<std::vector<uint64_t>> BB_RAYS;


// Define global masks for piece placement
uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

// Define global variables for offensive, defensive and piece value scores
int whiteOffensiveScore, blackOffensiveScore, whiteDefensiveScore, blackDefensiveScore;
int blackPieceVal, whitePieceVal;


int central_score;
/*
	Define a set of lookup tables
*/

// Define zobrist table, cache and insertion order for efficient hashing
uint64_t zobristTable[12][64];
uint64_t zobristTurn;

uint64_t castling_hash[4];
uint64_t ep_hash[65];


std::vector<EvalEntry> evalCacheNew(CACHE_SIZE);

std::unordered_map<uint64_t, int> evalCache;
std::deque<uint64_t> insertionOrder;

/* std::unordered_map<uint64_t, int> quiesceEvalCache;
std::deque<uint64_t> quiesceinsertionOrder; */


std::vector<MoveEntry> moveGenCache(CACHE_SIZE);
std::vector<QTTEntry> quiesceEvalCache(CACHE_SIZE);
/* std::unordered_map<uint64_t, std::vector<Move>> moveGenCache;
std::deque<uint64_t> moveGenInsertionOrder; */

std::vector<TTEntry> searchEvalCache(TT_CACHE_SIZE);

alignas(64) Move killerMoves[MAX_PLY][2];

alignas(64) Move counterMoves[64][64];

alignas(64) int counterMoveHeuristics[2][4096][4096] = {};

alignas(64) int historyHeuristics[2][64][64] = {};

alignas(64) int moveFrequency[2][64][64] = {};

alignas(64) int contHist2[2][4096][4096] = {};

alignas(64) int captureHistory[2][64][64] = {};

alignas(64) Move g_searchStack[MAX_PLY] = {};
alignas(64) int g_evalStack[MAX_PLY] = {};





// Define a heat map for attacks
alignas(64) std::array<std::array<std::array<int, 8>, 8>, 2> attackingLayer;

// Define heat maps for piece placement for both white and black
alignas(64) std::array<std::array<std::array<int, 8>, 8>, 6> whitePlacementLayer = {{
    {{ // Pawns
        {{0,20,7,7,10,15,20,0}},
        {{0,20,5,5,7,15,20,0}},
        {{0,15,15,30,25,35,30,0}},
        {{0,0,20,50,60,35,30,0}},
        {{0,0,20,50,60,35,30,0}},
        {{0,15,15,30,25,35,30,0}},
        {{0,20,5,5,7,15,20,0}},
        {{0,20,7,7,10,15,20,0}}
    }},
    {{ // Knights 
		{{  0,  5, 10, 15, 15, 10,  5,  0 }}, // File A
		{{  5, 10, 20, 25, 25, 20, 10,  5 }}, // File B
		{{ 10, 20, 30, 35, 35, 30, 20, 10 }}, // File C
		{{ 15, 25, 35, 40, 40, 35, 25, 15 }}, // File D (center)
		{{ 15, 25, 35, 40, 40, 35, 25, 15 }}, // File E (center)
		{{ 10, 20, 30, 35, 35, 30, 20, 10 }}, // File F
		{{  5, 10, 20, 25, 25, 20, 10,  5 }}, // File G
		{{  0,  5, 10, 15, 15, 10,  5,  0 }}  // File H
	}},
    {{ // Bishops
        {{10,10,20,20,25,25,15,20}},
        {{10,20,20,35,20,25,20,20}},
        {{10,20,25,35,35,25,20,20}},
        {{10,25,25,35,35,25,20,20}},
        {{10,25,25,35,35,25,20,20}},
        {{10,20,25,35,35,25,20,20}},
        {{10,20,20,35,20,25,20,20}},
        {{10,10,20,20,25,25,15,20}}
    }},
    {{ // Rooks
        {{0,0,0,0,0,0,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,0,0,0,0,0,0}}
    }},
    {{ // Queens
        {{40,20,40,40,35,20,15,20}},
		{{40,30,60,60,55,25,20,20}},
		{{40,40,65,65,60,20,20,20}},
		{{40,45,65,65,60,25,20,20}},
		{{40,45,65,65,60,25,20,20}},
		{{40,40,65,65,60,20,20,20}},
		{{40,30,60,60,55,20,20,20}},
		{{40,20,40,40,35,20,20,20}}
    }},
    {{ // Kings - Endgame
		{{  0,  5,  5,  5,  5,  5,  5,  0 }},
		{{  5, 10, 15, 20, 20, 15, 10,  5 }},
		{{  5, 15, 25, 30, 30, 25, 15,  5 }},
		{{  5, 20, 30, 35, 35, 30, 20,  5 }},
		{{  5, 20, 30, 35, 35, 30, 20,  5 }},
		{{  5, 15, 25, 30, 30, 25, 15,  5 }},
		{{  5, 10, 15, 20, 20, 15, 10,  5 }},
		{{  0,  5,  5,  5,  5,  5,  5,  0 }}
	}}
}};

alignas(64) std::array<std::array<std::array<int, 8>, 8>, 6> blackPlacementLayer = {{
    {{ // Pawns
		{{ 0, 20, 15, 10,  7,  7, 20,  0 }},
		{{ 0, 20, 15,  7,  5,  5, 20,  0 }},
		{{ 0, 30, 35, 25, 30, 15, 15,  0 }},
		{{ 0, 30, 35, 60, 50, 20,  0,  0 }},
		{{ 0, 30, 35, 60, 50, 20,  0,  0 }},
		{{ 0, 30, 35, 25, 30, 15, 15,  0 }},
		{{ 0, 20, 15,  7,  5,  5, 20,  0 }},
		{{ 0, 20, 15, 10,  7,  7, 20,  0 }}
	}},
    {{ // Knights 
		{{  0,  5, 10, 15, 15, 10,  5,  0 }}, // File A
		{{  5, 10, 20, 25, 25, 20, 10,  5 }}, // File B
		{{ 10, 20, 30, 35, 35, 30, 20, 10 }}, // File C
		{{ 15, 25, 35, 40, 40, 35, 25, 15 }}, // File D (center)
		{{ 15, 25, 35, 40, 40, 35, 25, 15 }}, // File E (center)
		{{ 10, 20, 30, 35, 35, 30, 20, 10 }}, // File F
		{{  5, 10, 20, 25, 25, 20, 10,  5 }}, // File G
		{{  0,  5, 10, 15, 15, 10,  5,  0 }}  // File H
	}},
    {{ // Bishops
        {{20,15,25,25,20,20,10,10}},
		{{20,20,25,20,35,20,20,10}},
		{{20,20,25,35,35,25,20,10}},
		{{20,20,25,35,35,25,25,10}},
		{{20,20,25,35,35,25,25,10}},
		{{20,20,25,35,35,25,20,10}},
		{{20,20,25,20,35,20,20,10}},
		{{20,15,25,25,20,20,10,10}}
    }},
    {{ // Rooks
        {{0,0,0,0,0,0,0,0}},
        {{0,0,2,10,10,3,0,0}},
        {{0,0,5,15,15,3,0,0}},
        {{0,0,5,25,20,3,0,0}},
        {{0,0,5,25,20,3,0,0}},
        {{0,0,5,15,15,3,0,0}},
        {{0,0,2,10,10,3,0,0}},
        {{0,0,0,0,0,0,0,0}}
    }},
    {{ // Queens
        {{20,15,20,35,40,40,20,40}},
		{{20,20,25,55,60,60,30,40}},
		{{20,20,20,60,65,65,40,40}},
		{{20,20,25,60,65,65,45,40}},
		{{20,20,25,60,65,65,45,40}},
		{{20,20,20,60,65,65,40,40}},
		{{20,20,20,55,60,60,30,40}},
		{{20,20,20,35,40,40,20,40}}
    }},
    {{ // Kings - Endgame
		{{  0,  5,  5,  5,  5,  5,  5,  0 }},
		{{  5, 10, 15, 20, 20, 15, 10,  5 }},
		{{  5, 15, 25, 30, 30, 25, 15,  5 }},
		{{  5, 20, 30, 35, 35, 30, 20,  5 }},
		{{  5, 20, 30, 35, 35, 30, 20,  5 }},
		{{  5, 15, 25, 30, 30, 25, 15,  5 }},
		{{  5, 10, 15, 20, 20, 15, 10,  5 }},
		{{  0,  5,  5,  5,  5,  5,  5,  0 }}
	}}

}};

// Define array to hold the piece type 
alignas(64) std::array<uint8_t, 64> pieceTypeLookUp = {};

std::array<uint64_t, 64> attack_bitmasks = {0ULL};


std::array<int, 64> pressure_white = {0};
std::array<int, 64> support_white = {0};
std::array<int, 64> pressure_black = {0};
std::array<int, 64> support_black = {0};

std::array<int, 64> num_attackers = {0};
std::array<int, 64> num_supporters = {0};

std::array<int, 64> square_values = {0};

// Diagnostic-only static-eval term attribution (see EvalBreakdown in cpp_bitboard.h). Off during search.
EvalBreakdown g_eval_breakdown = {};
bool g_capture_eval_breakdown = false;
// Scratch for advanced_endgame_eval's internal deltas (published into g_eval_breakdown only when capturing).
int g_ae_matedrive = 0;
int g_ae_passer = 0;

constexpr std::array<std::array<int, 7>, 7> support_weights = {{
    //             None  Pawn  Knight  Bishop  Rook  Queen  King
    /* None   */ {  0,    0,      0,      0,     0,     0,     0 },
    /* Pawn   */ {  0,   85,     20,     20,    15,     4,     0 }, 
    /* Knight */ {  0,   16,     13,     13,    11,     3,     0 }, 
    /* Bishop */ {  0,   15,     12,     12,    10,     3,     0 }, 
    /* Rook   */ {  0,   10,      8,     10,     7,     2,     0 }, 
    /* Queen  */ {  0,    5,      5,      5,     5,     1,     0 },
    /* King   */ {  0,    2,      1,      1,     1,     1,     0 }
}};

constexpr std::array<std::array<int, 7>, 7> pressure_weights = {{
    //             None  Pawn  Knight  Bishop  Rook  Queen  King
    /* None   */ {  0,    0,      0,      0,     0,     0,     0 },
    /* Pawn   */ {  0,   85,    100,    105,   125,   150,     0 }, 
    /* Knight */ {  0,   15,     13,     20,    50,   100,     0 }, 
    /* Bishop */ {  0,   14,     19,     12,    48,   100,     0 }, 
    /* Rook   */ {  0,   10,     10,     10,     7,    50,     0 }, 
    /* Queen  */ {  0,    5,      5,      5,     5,     1,     0 }, 
    /* King   */ {  0,    5,      5,      5,     5,     5,     0 } 
}};

constexpr int decrement_lookup[7] = {0,  5, 35, 35, 100, 150, 1000};  // [piece_type]
constexpr int pressure_increase_lookup[7] = {0,  5, 10, 10, 20,  30,  30};



bool horizon_mitigation_flag = false;
bool get_horizon_mitigation_flag(){return horizon_mitigation_flag;}

/*
	Set of functions to initialize masks for move generation
*/
void initialize_attack_tables() {
	
	/*
		Function to initialize attack tables for move generation
	*/
	
	// Define the position deltas for knights, kings and pawn moves
    std::vector<int8_t> knight_deltas = {17, 15, 10, 6, -17, -15, -10, -6};
    std::vector<int8_t> king_deltas = {9, 8, 7, 1, -9, -8, -7, -1};
    std::vector<int8_t> white_pawn_deltas = {-7, -9};
    std::vector<int8_t> black_pawn_deltas = {7, 9};
	
	// Fill up the tables for all possible knight, king and pawn moves
    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        BB_KNIGHT_ATTACKS[sq] = sliding_attacks(sq, ~0ULL, knight_deltas);
        BB_KING_ATTACKS[sq] = sliding_attacks(sq, ~0ULL, king_deltas);
		BB_PAWN_ATTACKS[0][sq] = sliding_attacks(sq, ~0ULL, white_pawn_deltas);
        BB_PAWN_ATTACKS[1][sq] = sliding_attacks(sq, ~0ULL, black_pawn_deltas);
    }

    // Build the king 2-ring (king move squares + their neighbours) = the squares setAttackingLayer's
    // king-zone loops touch, used as the midgame attack-layer cache key.
    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        uint64_t ring = BB_KING_ATTACKS[sq];
        uint64_t inner = BB_KING_ATTACKS[sq];
        while (inner) {
            ring |= BB_KING_ATTACKS[__builtin_ctzll(inner)];
            inner &= inner - 1;
        }
        king_ring2[sq] = ring;
    }

    // Build the passed-pawn forward spans: for each square, the union over its own file and both
    // neighbouring files of every rank strictly ahead of the pawn (toward promotion). A pawn is passed
    // iff the enemy-pawn bitboard does not intersect this span. Shape mirrors getPPIncrement.
    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        int x = sq & 7;
        int y = sq >> 3;
        uint64_t span_white = 0;
        uint64_t span_black = 0;
        for (int f = x - 1; f <= x + 1; ++f) {
            if (f < 0 || f > 7) continue;
            // White advances toward rank 8: every rank above y. y==7 has nothing ahead (avoid the UB shift).
            span_white |= BB_FILES[f] & (y < 7 ? ~((1ULL << ((y + 1) * 8)) - 1) : 0ULL);
            // Black advances toward rank 1: every rank below y. y==0 yields 0 naturally.
            span_black |= BB_FILES[f] & ((1ULL << (y * 8)) - 1);
        }
        passed_span_white[sq] = span_white;
        passed_span_black[sq] = span_black;
    }

	// Call the function to fill up the tables for all possible queen and rook moves
	attack_table({-9, -7, 7, 9},BB_DIAG_MASKS,BB_DIAG_ATTACKS);
	attack_table({-8, 8},BB_FILE_MASKS,BB_FILE_ATTACKS);
	attack_table({-1, 1},BB_RANK_MASKS,BB_RANK_ATTACKS);

#ifdef PEXT_SELFCHECK
	// Prove the PEXT row lookup returns identical attack sets to the reference
	// generator for every square and every masked-occupancy subset.
	{
		const std::vector<int8_t> diag_deltas = {-9, -7, 7, 9};
		const std::vector<int8_t> file_deltas = {-8, 8};
		const std::vector<int8_t> rank_deltas = {-1, 1};
		for (int sq = 0; sq < NUM_SQUARES; ++sq) {
			std::vector<uint64_t> subsets;
			carry_rippler(BB_DIAG_MASKS[sq], subsets);
			for (uint64_t s : subsets) assert(BB_DIAG_ATTACKS[sq][s] == sliding_attacks(sq, s, diag_deltas));
			subsets.clear();
			carry_rippler(BB_FILE_MASKS[sq], subsets);
			for (uint64_t s : subsets) assert(BB_FILE_ATTACKS[sq][s] == sliding_attacks(sq, s, file_deltas));
			subsets.clear();
			carry_rippler(BB_RANK_MASKS[sq], subsets);
			for (uint64_t s : subsets) assert(BB_RANK_ATTACKS[sq][s] == sliding_attacks(sq, s, rank_deltas));
		}
	}
#endif

	rays(BB_RAYS);
}

void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<SlidingRow> &attack_table) {

	/*
		Function to initialize attack mask tables for diagonal, file and rank attacks

		Parameters:
		- deltas: A vector of position delta values, passed by reference
		- mask_table: An empty vector to hold the sliding attacks masks, passed by reference
		- attack_table: An empty vector to hold the attack subsets of the mask, passed by reference

		Each square's attacks are stored in a PEXT (Parallel Bits Extract) row:
		the masked occupancy subset is compressed with _pext_u64 into a dense
		index into a flat array, replacing the prior unordered_map lookup.
	*/

	// Loop through all squares
    for (int square = 0; square < 64; ++square) {

		// Acquire sliding attacks mask for the given deltas
        uint64_t mask = sliding_attacks(square, 0ULL, deltas) & ~edges(square);

		// Build the dense PEXT-indexed attack array for this square
		SlidingRow row;
		row.mask = mask;
		row.data.assign(1ULL << __builtin_popcountll(mask), 0ULL);

		// Acquire subsets of attacks mask and loop through them to form the attack table
		std::vector<uint64_t> subsets;
		carry_rippler(mask,subsets);
        for (uint64_t subset : subsets) {
            row.data[_pext_u64(subset, mask)] = sliding_attacks(square, subset, deltas);
        }

		// Push the current mask and attack tables to the full set
        mask_table.push_back(mask);
        attack_table.push_back(std::move(row));
    }
}

uint64_t sliding_attacks(uint8_t square, uint64_t occupied, const std::vector<int8_t>& deltas) {
	
	/*
		Function to calculate sliding attacks
		
		Parameters:
		- square: The starting square
		- uint64_t: The mask of occupied pieces
		- deltas: A vector of position delta values, passed by reference
		
		Returns:
		A bitboard mask representing the sliding attacks possible from the given square with the given deltas
	*/
	
    uint64_t attacks = 0ULL;

	// Loop through the deltas
    for (int8_t delta : deltas) {
        uint8_t sq = square;

		// Keep applying the delta
        while (true) {
			
			// Check if the current square either wraps around or goes outside the board to break the loop
            sq += delta;
            if (!(0 <= sq && sq < 64) || square_distance(sq, sq - delta) > 2) {
                break;
            }

			// Add the square to the attacks mask
            attacks |= (1ULL << sq);

			// If the square is occupied, the attack stops there
            if (occupied & (1ULL << sq)) {
                break;
            }
        }
    }
    return attacks;
}

void carry_rippler(uint64_t mask, std::vector<uint64_t> &subsets) {
    
	/*
		Function to generate subsets of a given mask
		
		Parameters:
		- mask: The mask to create subsets of
		- subsets: An empty vector to hold the ssubsets, passed by reference		
	*/
	
	// Generates all subsets of the bitmask iteratively    
	uint64_t subset = 0ULL;
    do {
		// This operation flips bits in subset and ensures only bits set in mask are retained.
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset);
    
}

void rays(std::vector<std::vector<uint64_t>> &rays) {
    
	/*
		Function to attack rays
		
		Parameters:
		- rays: An empty vector to hold the vectors of squares representing each ray, passed by reference
	*/
	
	// Loop through all squares to represent starting points
    for (size_t a = 0; a < 64; ++a) {
        std::vector<uint64_t> rays_row;
        uint64_t bb_a = 1ULL << a;
		
		// Loop through all squares to represent ending points
        for (size_t b = 0; b < 64; ++b) {
            uint64_t bb_b = 1ULL << b;
			
			// Get all diagonal, rank and file attacks for the given points
            if (BB_DIAG_ATTACKS[a][0] & bb_b) {
                rays_row.push_back((BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | bb_a | bb_b);
            } else if (BB_RANK_ATTACKS[a][0] & bb_b) {
                rays_row.push_back(BB_RANK_ATTACKS[a][0] | bb_a);
            } else if (BB_FILE_ATTACKS[a][0] & bb_b) {
                rays_row.push_back(BB_FILE_ATTACKS[a][0] | bb_a);
            } else {
                rays_row.push_back(0ULL);
            }
        }
		
		// Push each ray to the vector
        rays.push_back(rays_row);
    }    
}

uint64_t edges(uint8_t square) {
	
	/*
		Function to get a bitmask of the edges
		
		Parameters:
		- rays: An empty vector to hold the vectors of squares representing each ray, passed by reference
	*/
	
    uint8_t rank = square >> 3;  
    uint8_t file = square & 7;      

    uint64_t rank_mask = (0xFFULL | 0xFF00000000000000ULL) & ~(0xFFULL << (8 * rank));
    uint64_t file_mask = (0x0101010101010101ULL | 0x8080808080808080ULL) & ~(0x0101010101010101ULL << file);

    return rank_mask | file_mask;
}


/*
	Set of functions directly used to evaluate the position
*/

inline int evaluate_pawns_midgame(uint8_t square, uint64_t& white_passed_pawns, uint64_t& black_passed_pawns, int& pawn_rank_bonus){
	// Initialize the evaluation
    int total = 0;
	int structural_bonus = 0;
	int positional_bonus = 0;
	
	// Initialize the maximum increment for pawn placement
	int ppIncrement = 200;
 	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
    
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value
        total -= values[PAWN];
        whitePieceVal += values[PAWN];

		// Subtract the placement layer for the given piece at that square
		positional_bonus += whitePlacementLayer[PAWN - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		positional_bonus += attackingLayer[0][x][y];
		positional_bonus += attackingLayer[1][x][y];
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
		whiteDefensiveScore += attackingLayer[1][x][y] >> 2;

		update_global_central_scores(-(whitePlacementLayer[PAWN - 1][x][y] << 1), BB_SQUARES[square]);
		//central_score -= whitePlacementLayer[PAWN - 1][x][y] << 2;

		// Lower white's score for more than one white pawn being on the same file
		total += 125 * (__builtin_popcountll(BB_FILES[x] & (occupied_white & pawns)) > 1);
		
		// Call the function to acquire an extra boost for passed and semi passed pawns
		ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white, white_passed_pawns, black_passed_pawns);
		ppIncrement = std::min(ppIncrement, 400); // cap runaway boosts

		int rank = y;
		//total -= ((rank * 15) * (ppIncrement < 200)) + (((rank * 50) + (rank * rank * 15) + (ppIncrement >> 3)) * (ppIncrement >= 200));
		pawn_rank_bonus = -(((default_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement < 100)) + ((passed_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement >= 100)));
		total += std::max(pawn_rank_bonus,-275);
		
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_white & pawns;
		uint64_t sw  = (BB_SQUARES[square] >> 9) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t se = (BB_SQUARES[square] >> 7) & ~BB_FILE_A & occupied_white & pawns;

		uint64_t latent_left_support_mask = latent_support_mask_left(square, colour);
		uint64_t latent_right_support_mask = latent_support_mask_right(square, colour);
		
		structural_bonus += pawn_wall_file_bonus[x]     * (left  != 0); // for pawn on file x-1
		structural_bonus += pawn_wall_file_bonus[x + 2] * (right != 0); // for pawn on file x+1
		
		structural_bonus += (pawn_chain_file_bonus[x] + 15) * (sw != 0);
		structural_bonus += (pawn_chain_file_bonus[x] + 15) * (se != 0);
		
		structural_bonus += 30 * (sw == 0) * ((latent_left_support_mask & occupied_white & pawns) != 0) * ((latent_left_support_mask & occupied_black) == 0);
		structural_bonus += 30 * (se == 0) * ((latent_right_support_mask & occupied_white & pawns) != 0) * ((latent_right_support_mask & occupied_black) == 0);  
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_PAWN_ATTACKS[colour][square];	
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (square_mask) & ~kings){				
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (square_mask)));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of white's own position
			positional_bonus += attackingLayer[0][x][y];
			positional_bonus += attackingLayer[1][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
			whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
			
			update_global_central_scores(-(attackingLayer[0][x][y] << 1), square_mask);

			if (square_mask & white_passed_pawns){
				total -= 100;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}

			//central_score -= attackingLayer[0][x][y] << 2;
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			structural_bonus += pawn_chain_file_bonus[x] * ((square_mask & occupied_white & pawns)  != 0);
								
			bb &= bb - 1;		
		}
		//std::cout << total << std::endl;
		total -= std::min(225, structural_bonus + positional_bonus);			
		//std::cout << structural_bonus << " | " << positional_bonus<< std::endl;
	}else{
		
		// First add the piece value
        total += values[PAWN];
		blackPieceVal += values[PAWN];
		            
		// Add the placement layer for the given piece at that square
		positional_bonus += blackPlacementLayer[PAWN - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		positional_bonus += attackingLayer[1][x][y];
		positional_bonus += attackingLayer[0][x][y];
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		blackOffensiveScore += attackingLayer[1][x][y] >> 1;
		blackDefensiveScore += attackingLayer[0][x][y] >> 2;

		update_global_central_scores((blackPlacementLayer[PAWN - 1][x][y] << 1), BB_SQUARES[square]);
		//central_score += blackPlacementLayer[PAWN - 1][x][y] << 2;
						
		// Lower black's score for more than one black pawn being on the same file						
		total -= 125 * (__builtin_popcountll(BB_FILES[x] & (occupied_black & pawns)) > 1);
		
		ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black, white_passed_pawns, black_passed_pawns);
		ppIncrement = std::min(ppIncrement, 400); // cap runaway boosts
		
		int rank = 7 - y;
		//total += ((rank * 15) * (ppIncrement < 200)) + (((rank * 50) + (rank * rank * 15) + (ppIncrement >> 3)) * (ppIncrement >= 200));		
		
		pawn_rank_bonus = ((default_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement < 100)) + ((passed_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement >= 100));
		total += std::min(pawn_rank_bonus,275);
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
				
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_black & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_black & pawns;
		uint64_t ne  = (BB_SQUARES[square] << 9) & ~BB_FILE_A & occupied_black & pawns;
		uint64_t nw = (BB_SQUARES[square] << 7) & ~BB_FILE_H & occupied_black & pawns;
		
		uint64_t latent_left_support_mask = latent_support_mask_left(square, colour);
		uint64_t latent_right_support_mask = latent_support_mask_right(square, colour);

		structural_bonus += pawn_wall_file_bonus[x]     * (left  != 0); // for pawn on file x-1
		structural_bonus += pawn_wall_file_bonus[x + 2] * (right != 0); // for pawn on file x+1
		
		structural_bonus += (pawn_chain_file_bonus[x] + 15) * (nw != 0);
		structural_bonus += (pawn_chain_file_bonus[x] + 15) * (ne != 0);

		structural_bonus += 30 * (nw == 0) * ((latent_left_support_mask & occupied_black & pawns) != 0) * ((latent_left_support_mask & occupied_white) == 0);
		structural_bonus += 30 * (ne == 0) * ((latent_right_support_mask & occupied_black & pawns) != 0) * ((latent_right_support_mask & occupied_white) == 0);  
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_PAWN_ATTACKS[colour][square];	
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (square_mask) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			positional_bonus += attackingLayer[1][x][y];
			positional_bonus += attackingLayer[0][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y] >> 1;
			blackDefensiveScore += attackingLayer[0][x][y] >> 2;
			
			update_global_central_scores((attackingLayer[1][x][y] << 1), square_mask);

			if (square_mask & black_passed_pawns){
				total += 100;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
			//central_score += attackingLayer[1][x][y] << 2;
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			
			// Increase the boost as the attacked pawn is closer to the center files			 
			structural_bonus += pawn_chain_file_bonus[x] * ((square_mask & occupied_black & pawns) != 0);

			bb &= bb - 1; 
		}
		//std::cout << total << std::endl;
		total += std::min(225, structural_bonus + positional_bonus);			
		//std::cout << structural_bonus << " | " << positional_bonus<< std::endl;
	}	
	return total;
}

/* inline int pawns_simd_initializer(uint64_t bb, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	uint8_t  x[16] = {};
    int32_t  rank[16] = {};
    int32_t  ppIncrement_simd[16]  = {0};
    int32_t  attack_score[16] = {0};
    //int32_t  left_mask[16];
    //int32_t  right_mask[16];
	int32_t pawn_val[16] = {0};
	int32_t placement_val[16] = {0};
	int32_t double_pawn_val[16] = {0};

	int32_t pawn_wall_file_bonus_val[16] = {0};	

    int32_t  is_white[16] = {0};
    //int32_t  pawn_wall_file_bonus[11];  // LUT
    int32_t  result[16] = {0};

	int ppIncrement = 200;
	int count = 0;

	int total = 0;

	// pawnsMask
	while (bb) {
		
		uint8_t r = __builtin_ctzll(bb);  
		
		uint8_t square_file = r & 7; 
		uint8_t square_rank = r >> 3;

		bool colour = bool(occupied_white & (BB_SQUARES[r]));

		x[count] = square_file;

		if (colour){
			
			ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, square_file, square_rank, occupied_white, occupied_black, white_passed_pawns, black_passed_pawns);
			ppIncrement = std::min(ppIncrement, 400);
			
			ppIncrement_simd[count] = ppIncrement;
			rank[count] = r >> 3;


			// First subtract the piece value
			pawn_val[count] = -values[PAWN];
			
			// Subtract the placement layer for the given piece at that square
			placement_val[count] = -whitePlacementLayer[PAWN - 1][square_file][square_rank];
						
			// Lower white's score for more than one white pawn being on the same file
			double_pawn_val[count] = 200 * (__builtin_popcountll(BB_FILES[square_file] & (occupied_white & pawns)) > 1);
									
			uint64_t left = ((BB_SQUARES[r]) >> 1) & ~BB_FILE_H & occupied_white & pawns;
			uint64_t right = ((BB_SQUARES[r]) << 1) & ~BB_FILE_A & occupied_white & pawns;	

			pawn_wall_file_bonus_val[count] -= pawn_wall_file_bonus[square_file]     * (left  != 0); // for pawn on file x-1
			pawn_wall_file_bonus_val[count] -= pawn_wall_file_bonus[square_file + 2] * (right != 0); // for pawn on file x-1

			is_white[count] = true;

		} else {
			ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, square_file, square_rank, occupied_black, occupied_white, white_passed_pawns, black_passed_pawns);
			ppIncrement = std::min(ppIncrement, 400);
			
			ppIncrement_simd[count] = ppIncrement;
			rank[count] = 7 - (r >> 3);


			// First subtract the piece value
			pawn_val[count] = values[PAWN];
			
			// Subtract the placement layer for the given piece at that square
			placement_val[count] = blackPlacementLayer[PAWN - 1][square_file][square_rank];
						
			// Lower black's score for more than one black pawn being on the same file		
			double_pawn_val[count] = -200 * (__builtin_popcountll(BB_FILES[square_file] & (occupied_black & pawns)) > 1);						
			
			uint64_t left = ((BB_SQUARES[r]) >> 1) & ~BB_FILE_H & occupied_black & pawns;
			uint64_t right = ((BB_SQUARES[r]) << 1) & ~BB_FILE_A & occupied_black & pawns;	

			pawn_wall_file_bonus_val[count] += pawn_wall_file_bonus[square_file]     * (left  != 0); // for pawn on file x-1
			pawn_wall_file_bonus_val[count] += pawn_wall_file_bonus[square_file + 2] * (right != 0); // for pawn on file x-1

			is_white[count] = false;
		}
		int dummy_int;		
		int result = evaluate_pawns_midgame(r, white_passed_pawns, black_passed_pawns,dummy_int);
		attack_score[count] = result;
		square_values[r] = 1000;
		total += result;

		// Clear the least significant set bit
		bb &= bb - 1; 
		count++; 
	}

	int result_sum = 0;
	int simd_iters = (count + 7) / 8;

	for (int i = 0; i < simd_iters; i++) {
		int offset = i * 8;

		__m256i x_vec          = _mm256_loadu_si256((__m256i*)(x + offset));
		__m256i rank_vec       = _mm256_loadu_si256((__m256i*)(rank + offset));
		__m256i ppi_vec        = _mm256_loadu_si256((__m256i*)(ppIncrement_simd + offset));

		__m256i pawn_val_vec        = _mm256_loadu_si256((__m256i*)(pawn_val + offset));
		__m256i placement_val_vec        = _mm256_loadu_si256((__m256i*)(placement_val + offset));
		__m256i double_pawn_val_vec        = _mm256_loadu_si256((__m256i*)(double_pawn_val + offset));
		__m256i pawn_wall_file_bonus_val_vec        = _mm256_loadu_si256((__m256i*)(pawn_wall_file_bonus_val + offset));

		__m256i attack_score_vec       = _mm256_loadu_si256((__m256i*)(attack_score + offset));
		__m256i white_vec      = _mm256_loadu_si256((__m256i*)(is_white + offset));



		__m256i wscale = _mm256_sub_epi32(_mm256_set1_epi32(1), _mm256_slli_epi32(white_vec, 1));

		__m256i rank_sq = _mm256_mullo_epi32(rank_vec, rank_vec);
		__m256i term1 = _mm256_mullo_epi32(rank_vec, _mm256_set1_epi32(50));
		__m256i term2 = _mm256_mullo_epi32(rank_sq, _mm256_set1_epi32(15));
		__m256i term3 = _mm256_srli_epi32(ppi_vec, 3);
		__m256i bonus = _mm256_add_epi32(_mm256_add_epi32(term1, term2), term3);
		__m256i fallback = _mm256_mullo_epi32(rank_vec, _mm256_set1_epi32(15));

		__m256i ppi_threshold = _mm256_cmpgt_epi32(ppi_vec, _mm256_set1_epi32(199));
		__m256i passed_term = _mm256_blendv_epi8(fallback, bonus, ppi_threshold);
		__m256i passed_term_scaled = _mm256_mullo_epi32(passed_term, wscale);


		__m256i intermediate_score_1 = _mm256_add_epi32(attack_score_vec, passed_term_scaled);
		__m256i intermediate_score_2 = _mm256_add_epi32(placement_val_vec, intermediate_score_1);
		__m256i intermediate_score_3 = _mm256_add_epi32(double_pawn_val_vec, intermediate_score_2);
		__m256i intermediate_score_4 = _mm256_add_epi32(pawn_wall_file_bonus_val_vec, intermediate_score_3);

		__m256i final_score = _mm256_add_epi32(pawn_val_vec, intermediate_score_4);

		__m128i low128 = _mm256_castsi256_si128(final_score);
		__m128i high128 = _mm256_extracti128_si256(final_score, 1);
		__m128i sum128 = _mm_add_epi32(low128, high128);
		__m128i shuffle1 = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
		__m128i sum2 = _mm_add_epi32(sum128, shuffle1);
		__m128i shuffle2 = _mm_shuffle_epi32(sum2, _MM_SHUFFLE(1, 0, 3, 2));
		__m128i final_sum = _mm_add_epi32(sum2, shuffle2);

		result_sum += _mm_cvtsi128_si32(final_sum);
	}
	total += result_sum;	
	return total;
} */

inline int evaluate_knights_midgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
    
	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
    
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value
        total -= values[KNIGHT];
        whitePieceVal += values[KNIGHT];
		     
		// Subtract the placement layer for the given piece at that square
		total -= whitePlacementLayer[KNIGHT - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of white's own position				
		total -= attackingLayer[0][x][y] >> 1;   
		total -= attackingLayer[1][x][y] >> 1;  
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		whiteOffensiveScore += attackingLayer[0][x][y];
		whiteDefensiveScore += attackingLayer[1][x][y];

		//std::cout << total << std::endl;
		update_global_central_scores(-(whitePlacementLayer[KNIGHT - 1][x][y]), BB_SQUARES[square]);
		
		// Subtract extra value for the existence of a bishop or knight in the midgame		
		//total -= 150;	

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){				
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y] >> 1;   
			total -= attackingLayer[1][x][y] >> 1;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];

			update_global_central_scores(-(attackingLayer[0][x][y] * 3) / 2, square_mask);

			if (square_mask & white_passed_pawns){
				total -= 100;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}

			// If each square doesn't contain a white piece, boost the score for mobility
			if (!Config::ENABLE_CHEAP_KNIGHT_MOBILITY && !((occupied_white & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black);

				if (!attacked_by_lower_value_piece) {
		
					total -= 20;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_KNIGHT_ATTACKS[r];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask							
							uint8_t secondary = __builtin_ctzll(bb_second);							
							uint64_t second_sq = BB_SQUARES[secondary];

							// If each square doesn't contain a white piece, boost the score for mobility
							if (!(simulatedOccupied & second_sq)){
								total -= 5;
								//std::cout << (int)secondary << " | " << BB_SQUARES[secondary] <<std::endl;								
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
			//std::cout << (int)r << " | " << total << std::endl;								
			bb &= bb - 1;		
		}
		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_KNIGHT_MOBILITY){
			total -= Config::CHEAP_KNIGHT_MOB * __builtin_popcountll(pieceAttackMask & ~occupied_white);
		}
		total = std::max(total, -3750);
	}else{
		// First add the piece value
        total += values[KNIGHT];
		blackPieceVal += values[KNIGHT];
		            
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[KNIGHT - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y] >> 1;
		total += attackingLayer[0][x][y] >> 1;
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		blackOffensiveScore += attackingLayer[1][x][y];
		blackDefensiveScore += attackingLayer[0][x][y];

		update_global_central_scores(blackPlacementLayer[KNIGHT - 1][x][y], BB_SQUARES[square]);
		
		// Add extra value for the existence of a bishop or knight in the midgame
		//total += 150;

		//std::cout << total << std::endl;	

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);
			
			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & square_mask));
			} */
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
						
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y] >> 1;
			total += attackingLayer[0][x][y] >> 1;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	

			update_global_central_scores((attackingLayer[1][x][y] * 3) / 2, square_mask);

			if (square_mask & black_passed_pawns){
				total += 100;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
								
			// If each square doesn't contain a black piece, boost the score for mobility  			
			if (!Config::ENABLE_CHEAP_KNIGHT_MOBILITY && !((occupied_black & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white);
				//std::cout << (int)r << " | " << attacked_by_lower_value_piece << " | " << total << std::endl;
				if (!attacked_by_lower_value_piece) {

					total += 20;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_KNIGHT_ATTACKS[r];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(bb_second);							
							uint64_t second_sq = BB_SQUARES[secondary];
							
							// If each square doesn't contain a white piece, boost the score for mobility							
							if (!(simulatedOccupied & second_sq)){
								total += 5;								
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}								
			//std::cout << (int)r << " | " << total << std::endl;	
			bb &= bb - 1; 
		}            
		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_KNIGHT_MOBILITY){
			total += Config::CHEAP_KNIGHT_MOB * __builtin_popcountll(pieceAttackMask & ~occupied_black);
		}
		total = std::min(total, 3750);
	}
	return total;
}

inline int get_latent_bishop_activity_score(uint64_t originalAttackMask, uint8_t square, bool colour, uint64_t opposingPieces, uint64_t ourPieces){
	int total = 0;

	// Loop through the attacks mask
	uint8_t r = 0;
	uint64_t bb = ~originalAttackMask & BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & (occupied & ~(originalAttackMask & ourPieces))];
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);		

		uint64_t square_mask = BB_SQUARES[r];		

		// Get the x and y coordinates for the given square
		uint8_t y = r >> 3;
		uint8_t x = r & 7;
					
		if(colour){
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y] / 3;   
			total -= attackingLayer[1][x][y] / 3;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
			whiteDefensiveScore += attackingLayer[1][x][y] >> 1;
			update_global_central_scores(-attackingLayer[0][x][y], square_mask);
		}else{
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y] / 3;
			total += attackingLayer[0][x][y] / 3;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y] >> 1;
			blackDefensiveScore += attackingLayer[0][x][y] >> 1;	
			update_global_central_scores(attackingLayer[1][x][y], square_mask);
		}
		
		int mobility_increment = 0;
		
		// If each square doesn't contain a white piece, boost the score for mobility
		if (!(ourPieces & square_mask) && (BB_PAWN_ATTACKS[colour][r] & pawns & opposingPieces)){
			mobility_increment += 15;

			if (bool(~opposingPieces & square_mask)){

				uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;
				uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
		
				while (secondAttackMask) {
		
					// Get the position of the least significant set bit of the mask
					uint8_t secondary = __builtin_ctzll(secondAttackMask);		
					
					// If each square doesn't contain a white piece, boost the score for mobility
					if (bool(~simulatedOccupied & (BB_SQUARES[secondary]))){
						mobility_increment += 5;
					}
					secondAttackMask &= secondAttackMask - 1;		
				}
			}
		}	
		total += (colour ? mobility_increment * -1 : mobility_increment);
		bb &= bb - 1;		
	}
	//std::cout << total << std::endl;
	return total;
}


inline bool is_white_square(int square) {
    int file = square % 8;
    int rank = square / 8;
    return (file + rank) % 2 == 1;
}

// Pick the next square so the scan order is colour-mirror-symmetric. White scans low square first;
// Black scans the vertically-mirrored order (byteswap flips ranks but preserves file order within a
// rank, ^56 maps the chosen bit back to its real square). This keeps the bishop reach/depth scan an
// exact mirror between colours, so eval(P) == -eval(mirror(P)) holds for the colour-complex term.
inline uint8_t mirror_aware_lsb(uint64_t bb, bool colour) {
    return colour ? static_cast<uint8_t>(__builtin_ctzll(bb))
                  : static_cast<uint8_t>(__builtin_ctzll(__builtin_bswap64(bb)) ^ 56);
}

inline uint64_t bishop_floodfill_fast(uint8_t start_sq, uint64_t occupied, uint64_t our_pawns, uint64_t our_pieces, std::array<uint8_t, 64>& depth_map, bool colour) {
    uint64_t visited = 0ULL;
    uint64_t queue = BB_SQUARES[start_sq];
    uint64_t result = 0ULL;

	std::fill(depth_map.begin(), depth_map.end(), 255); // 255 = unreachable

	depth_map[start_sq] = 0;

    while (queue) {
        uint8_t sq = mirror_aware_lsb(queue, colour);
        queue &= ~BB_SQUARES[sq]; // pop

        //if (visited & (1ULL << sq)) continue;
        visited |= BB_SQUARES[sq];
        result |= BB_SQUARES[sq];

        for (int delta : {9, 7, -7, -9}) {
            int dir_sq = sq;
            int blockers = 0;
			
			//std::cout << "AAAA: " << dir_sq << " | " << delta << " | " << queue << std::endl;
            while (true) {
				bool blocker_check_flag = false;
				
                int next = dir_sq + delta;

				if (next < 0 || next >= 64) break;
                if (abs((next % 8) - (dir_sq % 8)) != 1) break;

				if(depth_map[next] > depth_map[sq] + blockers + 1){
					blocker_check_flag = true;
					depth_map[next] = depth_map[sq] + blockers + 1;
				}
					
                //std::cout << "BBBB: " << dir_sq << " | " << next << " | " << queue << std::endl;
				

                uint64_t m = BB_SQUARES[next];                                                

                if (occupied & m) {
					if (our_pawns & m) {
						visited |= m;
						break; // ❌ Stop immediately — pawn blocks
					}else if (our_pieces & m) {
						// Non-pawn friendly blocker
						//std::cout << "DDDD: " << dir_sq << " | " << next << " | "  << blockers << " | " << queue << std::endl;
						if (blockers == 0) {
							++blockers;
							if (blocker_check_flag)
								depth_map[next]++;
							if (visited & m) break;
							queue |= m; // ✅ Allow going THROUGH the first non-pawn blocker
							result |= m;
							
							//std::cout << "EEE: " << dir_sq << " | " << next << " | "  << blockers << " | " << result << " | " << queue << std::endl;						
							// Also mark this square as reachable
						} else {
							visited |= m;
							break; // ❌ Second friendly blocker — stop
						}
					}else {
						// Opponent piece: we can take it, but can't go beyond
						//result |= m; // ✅ Mark square as reachable (capture)
						visited |= m;
						break;
					}
				}else {
					if (visited & m) break;
					queue |= m; // ✅ Empty square — 
					result |= m;
				}
				visited |= m;
                dir_sq = next;
            }
        }
    }

    return result;
}

inline int eval_bishop_influence_score(int increment) {
    

    if (increment <= BAD_THRESHOLD) {
        // Penalty: scale from 0 to -100
        return (increment - BAD_THRESHOLD) * 2;  // e.g. -50 at 75
    } else if (increment <= NEUTRAL_THRESHOLD) {
        return 0;  // Flat zone
    } else if (increment <= MAX_THRESHOLD) {
        // Scale bonus: 0 to +150
        return ((increment - NEUTRAL_THRESHOLD) * MAX_BONUS) / (MAX_THRESHOLD - NEUTRAL_THRESHOLD);
    } else {
        return MAX_BONUS;  // Cap
    }
}


inline int get_bishop_colour_complex_score(bool colour, uint8_t square, uint64_t attack_mask){
	
	int increment = 0;
	int same_half_count = 0;
	int other_half_count = 0;
	int king_zone_count = 0;

	bool is_light = is_white_square(square);

	if (Config::ENABLE_CHEAP_BISHOP_COMPLEX) {
		// Cheap popcount surrogate for the flood-fill below (see ENABLE_CHEAP_BISHOP_COMPLEX). Bad bishop =
		// own pawns on the bishop's colour; activity = the bishop's current diagonal scope, weighted extra
		// for squares in the enemy half and the enemy king zone. The flood-fill path has no global
		// side-effects, so returning a value here is a clean drop-in. Clamped to the term's usual envelope.
		uint64_t colour_mask = is_light ? LIGHT_SQUARES : DARK_SQUARES;
		uint64_t our_pieces = colour ? occupied_white : occupied_black;
		uint64_t far_half = colour ? (BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8)
		                           : (BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4);
		uint64_t enemy_king_zone = colour
		    ? black_king_zones[__builtin_ctzll(occupied_black & kings) & 7]
		    : white_king_zones[__builtin_ctzll(occupied_white & kings) & 7];

		int block = __builtin_popcountll(pawns & our_pieces & colour_mask);
		int mob   = __builtin_popcountll(attack_mask);
		int fwd   = __builtin_popcountll(attack_mask & far_half);
		int kingp = __builtin_popcountll(attack_mask & enemy_king_zone);

		int raw = Config::CHEAP_BISHOP_MOB * mob
		        + Config::CHEAP_BISHOP_FWD * fwd
		        + Config::CHEAP_BISHOP_KING * kingp
		        - Config::CHEAP_BISHOP_BLOCK * block;
		return std::max(-2 * BAD_THRESHOLD, std::min(MAX_BONUS, raw));
	}

	uint64_t white_king_zone = white_king_zones[__builtin_ctzll(occupied_white&kings) & 7] & ~BB_RANK_4 & ~BB_RANK_5;
	uint64_t black_king_zone = black_king_zones[__builtin_ctzll(occupied_black&kings) & 7] & ~BB_RANK_4 & ~BB_RANK_5;

	uint64_t base_zone = colour
    ? (is_light ? WHITE_LIGHT_BISHOP_ZONE : WHITE_DARK_BISHOP_ZONE)
    : (is_light ? BLACK_LIGHT_BISHOP_ZONE : BLACK_DARK_BISHOP_ZONE);

	uint64_t opposingPieces = colour ? occupied_black : occupied_white;
	uint64_t ourPieces = colour ? occupied_white : occupied_black;

	uint64_t included = 0;

	base_zone |= attack_mask | BB_SQUARES[square];

	std::array<uint8_t, 64> depth_map;
	uint64_t reachable = bishop_floodfill_fast(square, occupied, pawns & ourPieces, ourPieces, depth_map, colour);
	uint64_t bb = base_zone & reachable;

	//std::cout << bb << std::endl;
	//std::cout << reachable << std::endl;
	while (bb) {
		// Scan in colour-mirror order so the first-claim attribution below is colour-symmetric
		uint8_t r = mirror_aware_lsb(bb, colour);
		
		//uint64_t square_mask = BB_SQUARES[r];	

		uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
		bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & opposingPieces) ||
													(BB_KNIGHT_ATTACKS[r] & knights & opposingPieces) ||
													(BB_DIAG_ATTACKS[r][diag_pieces] & bishops & opposingPieces);
		//std::cout << (int)r << " | " << depth_map[r] << std::endl;
		/* std::cout << (occupied & square_mask) << std::endl;
		std::cout << attacked_by_lower_value_piece << std::endl;
		std::cout << std::endl; */
		if (!(/* (occupied & square_mask) || */ attacked_by_lower_value_piece)){

			
			uint64_t raw_attacks = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & (occupied & ~BB_SQUARES[square])];

			uint64_t forward_mask;
			if (colour) {
				// White — forward = upward ranks (higher square indices)
				forward_mask = (r < 63) ? (~0ULL << (r + 1)) : 0ULL;
			} else {
				// Black — forward = downward ranks (lower square indices)
				forward_mask = (r > 0) ? ((1ULL << r) - 1) : 0ULL;
			}


			uint64_t forward_attacks = raw_attacks & forward_mask & ~(ourPieces & ~BB_SQUARES[square]);
			
			//std::cout << "FORWARD: " << forward_attacks << std::endl;
			while (forward_attacks) {
				// Get the position of the least significant set bit of the mask
				
				uint8_t staging_square = __builtin_ctzll(forward_attacks);
				//std::cout << (int)r << " | " << (int)staging_square << " | " << depth_map[r] << std::endl;
				uint64_t staging_square_mask = BB_SQUARES[staging_square];
				if(!(staging_square_mask & included)){
					included |= staging_square_mask;	
					//increment += 5;
					int move_depth = depth_map[r];
					increment += std::max(0, 11 - 2 * move_depth);
					same_half_count++;
					if(colour){
						//std::cout << (int)r << " | "  << (int)staging_square << " | " << (occupied & ~square_mask) << " | " << raw_attacks << std::endl;
						if(staging_square_mask & (BB_RANK_8 | BB_RANK_7 | BB_RANK_6 | BB_RANK_5)){
							//increment += 5;
							increment += std::max(0, 11 - 2 * move_depth);
							same_half_count--;
							other_half_count++;
						}

						if(staging_square_mask & black_king_zone){
							//increment += 5;							
							increment += std::max(0, 11 - 2 * move_depth);
							king_zone_count++;
						}
							
					}else{
						if(staging_square_mask & (BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4)){
							//increment += 5;
							increment += std::max(0, 11 - 2 * move_depth);
							same_half_count--;
							other_half_count++;
						}

						if(staging_square_mask & white_king_zone){
							//increment += 5;
							increment += std::max(0, 11 - 2 * move_depth);
							king_zone_count++;
						}
					}
				}					
				forward_attacks &= forward_attacks - 1;
			}
		}
		bb &= ~BB_SQUARES[r];
	}
	/* std::cout << same_half_count << std::endl;
	std::cout << other_half_count << std::endl;
	std::cout << king_zone_count << std::endl; 
	std::cout << std::endl;
	std::cout << "Increment: " << increment << std::endl;*/
	
	int final_value = eval_bishop_influence_score(increment);
	//std::cout << "final value: " << final_value << std::endl;
	
	return final_value;
}


inline int evaluate_bishops_midgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
    
	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	if(colour){
		// First subtract the piece value
        total -= values[BISHOP];
        whitePieceVal += values[BISHOP];
		            
		// Subtract the placement layer for the given piece at that square
		total -= whitePlacementLayer[BISHOP - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of white's own position				
		total -= attackingLayer[0][x][y] >> 1;   
		total -= attackingLayer[1][x][y] >> 1;  
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		whiteOffensiveScore += attackingLayer[0][x][y];
		whiteDefensiveScore += attackingLayer[1][x][y];

		update_global_central_scores(-(whitePlacementLayer[BISHOP - 1][x][y]), BB_SQUARES[square]);
		
		// Subtract extra value for the existence of a bishop or knight in the midgame            
		//total -= 200;

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){				
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			 			
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y] >> 1;   
			total -= attackingLayer[1][x][y] >> 1;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];

			update_global_central_scores(-(attackingLayer[0][x][y] * 3) / 2, square_mask);
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);
			
			if (square_mask & white_passed_pawns){
				total -= 100;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}

			// If each square doesn't contain a white piece, boost the score for mobility
			if (!((occupied_white & square_mask)) && (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black)){
				total -= 20;

				if (bool(~occupied_black & square_mask)){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;
					uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~simulatedOccupied & (BB_SQUARES[secondary]))){
							total -= 5;
						}
						secondAttackMask &= secondAttackMask - 1;		
					}
				}
			}	
			bb &= bb - 1;		
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
			
		/* handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,BISHOP);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_white;

		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			total -= attackingLayer[0][x][y] >> 2;

			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 6;					
			}				
			bb &= bb - 1;
		}            
		{
			PROF_BLOCK(PROF_BISHOP_ACTIVITY);
			total += get_latent_bishop_activity_score(pieceAttackMask, square, colour, occupied_black, occupied_white);
		}
		total = std::max(total, -3850);
		{
			PROF_BLOCK(PROF_BISHOP_COLOUR);
			total -= get_bishop_colour_complex_score(colour, square, pieceAttackMask);
		}
		total = std::max(total, -4000);
	}else{
		// First add the piece value
        total += values[BISHOP];
		blackPieceVal += values[BISHOP];
		            
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[BISHOP - 1][x][y];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y] >> 1;
		total += attackingLayer[0][x][y] >> 1;
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		blackOffensiveScore += attackingLayer[1][x][y];
		blackDefensiveScore += attackingLayer[0][x][y];	

		update_global_central_scores(blackPlacementLayer[BISHOP - 1][x][y], BB_SQUARES[square]);
		
		// Add extra value for the existence of a bishop or knight in the midgame            
		//total += 200;

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;		
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & square_mask));
			} */
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
						
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y] >> 1;
			total += attackingLayer[0][x][y] >> 1;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	

			update_global_central_scores((attackingLayer[1][x][y] * 3) / 2, square_mask);
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);

			if (square_mask & black_passed_pawns){
				total += 100;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility											
			if (!((occupied_black & square_mask)) && (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white)){
				total += 20;

				if (bool(~occupied_white & square_mask)){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;

					uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~simulatedOccupied & (BB_SQUARES[secondary]))){
							total += 5;
						}
						secondAttackMask &= secondAttackMask - 1;		
					}
				}
			}
										
			bb &= bb - 1; 
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
					
		/* handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,BISHOP);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_black;
		
		// Loop through the xray attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece
			
			total += attackingLayer[1][x][y] >> 2;
						
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total += values[xRayPieceType] >> 6;								
			}
			bb &= bb - 1; 
		}			         
		{
			PROF_BLOCK(PROF_BISHOP_ACTIVITY);
			total += get_latent_bishop_activity_score(pieceAttackMask, square, colour, occupied_white, occupied_black);
		}
		total = std::min(total, 3850);
		{
			PROF_BLOCK(PROF_BISHOP_COLOUR);
			total += get_bishop_colour_complex_score(colour, square, pieceAttackMask);
		}
		total = std::min(total, 4000);
	}
	
	return total;
}

inline int get_latent_rook_activity_score(uint8_t square){
	int total = 0;

	bool colour = bool(occupied_white & (BB_SQUARES[square])); 

	uint64_t opposingPieces = colour ? occupied_black : occupied_white;
	uint64_t ourPieces = colour ? occupied_white : occupied_black;

	// Acquire the attacks mask for the current piece and make a copy of the occupied mask
	uint64_t originalAttackMask = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];

	// Loop through the attacks mask
	uint8_t r = 0;
	uint64_t hypothetical_occupied = (occupied & ~(originalAttackMask & ourPieces & ~pawns));
	uint64_t bb = ~originalAttackMask & (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & hypothetical_occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & hypothetical_occupied]);
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);		

		uint64_t square_mask = BB_SQUARES[r];		

		// Get the x and y coordinates for the given square
		uint8_t y = r >> 3;
		uint8_t x = r & 7;
					
		if(colour){
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y] / 3;   
			total -= attackingLayer[1][x][y] / 3;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
			whiteDefensiveScore += attackingLayer[1][x][y] >> 1;
			update_global_central_scores(-(attackingLayer[0][x][y]), square_mask);
		}else{
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y] / 3;
			total += attackingLayer[0][x][y] / 3;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y] >> 1;
			blackDefensiveScore += attackingLayer[0][x][y] >> 1;	
			update_global_central_scores((attackingLayer[1][x][y]), square_mask);
		}
		
		int mobility_increment = 0;
		
		// If each square doesn't contain a white piece, boost the score for mobility
		if (!(ourPieces & square_mask) && (BB_PAWN_ATTACKS[colour][r] & pawns & opposingPieces)){
			mobility_increment += 10;

			if (bool(~opposingPieces & square_mask)){

				uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;
				uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
		
				while (secondAttackMask) {
		
					// Get the position of the least significant set bit of the mask
					uint8_t secondary = __builtin_ctzll(secondAttackMask);		
					
					// If each square doesn't contain a white piece, boost the score for mobility
					if (bool(~simulatedOccupied & (BB_SQUARES[secondary]))){
						mobility_increment += 5;
					}
					secondAttackMask &= secondAttackMask - 1;		
				}
			}
		}	
		total += (colour ? mobility_increment * -1 : mobility_increment);
		bb &= bb - 1;		
	}
	//std::cout << total << std::endl;
	return total;
}


inline int evaluate_rooks_midgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	int mobility_bonus = 0;

	// Define the maximum increment for rook-file positioning and attack mask
	int rookIncrement = 250;
    uint64_t rooks_mask = 0ULL;
    
	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	if (colour){

		// First subtract the piece value
        total -= values[ROOK];
        whitePieceVal += values[ROOK];

		// Subtract the score based on the attack of the opposing position and defense of white's own position				
		total -= attackingLayer[0][x][y] >> 1;   
		total -= attackingLayer[1][x][y] >> 2;  
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		whiteOffensiveScore += attackingLayer[0][x][y];
		whiteDefensiveScore += attackingLayer[1][x][y];

		// Boost the score if a rook is placed on the 7th Rank
		if (y >= 6){
			rookIncrement += 150;
		}

		if (((BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied]) & (occupied_white & rooks)) != 0){
			rookIncrement += 150;
		} else if (((BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]) & (occupied_white & rooks)) != 0){
			rookIncrement += 125;
		}
		
		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces
		uint8_t r = 0;
		uint64_t bb = rooks_mask;
		while (bb) {
			
			// Get the current square as the max bit of the current mask
			r = __builtin_ctzll(bb);							
			uint8_t att_square = r;
			bb &= bb - 1;  
			
			// Check if the attacked square is up the board from the rook
			if (att_square > square){
				
				// Get the piece type and colour
				uint8_t temp_piece_type = pieceTypeLookUp [att_square];
				bool temp_colour = bool(occupied_white & (BB_SQUARES[att_square]));
									
				/*
					In this section, update the rook increment based on how open the file is
					This includes pieces and pawns in the way of both colours
				*/
				
				// Check if the occupied piece is white (same as the rook)
				if (temp_colour){
					
					// Check if the piece is the rook's own pawn
					if (temp_piece_type == 1){     

						if (white_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (att_square >> 3) * 50;
						}else{
							// If the pawn is within its own (first) half, lower the rook's increment and break the loop
							if ((att_square >> 3) < 5){
								rookIncrement -= (50 + ((3 - (att_square >> 3)) * 125));
								break;
							}
						}				
					
					// If a white knight or bishop is in the way, lower the rook increment
					} else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					}
				
				// Check if the occupied piece is black (opposite of the rook)
				}else{
					
					// Check if the piece is the opponents (black) pawn
					if (temp_piece_type == 1){
						
						if (black_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (7 - (att_square >> 3)) * 25;
						}else{
							// If the pawn is within the opponent's (second) half, lower the rook's increment
							if ((att_square >> 3) > 4){
								rookIncrement -= 50;
							}
						}	

					// If a black knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					// If a black rook is in the way, lower the rook increment
					}else if(temp_piece_type == 4){
						rookIncrement -= 35;
					}
				}
			}
		}
		// Finally use the rook increment
		total -= std::min(rookIncrement, 300);

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		r = 0;
		bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);	
			
			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){				
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y] >> 1;   
			total -= attackingLayer[1][x][y] >> 2;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];

			update_global_central_scores(-attackingLayer[0][x][y], square_mask);
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);
			
			if (square_mask & white_passed_pawns){
				total -= 25;
			} else if(square_mask & black_passed_pawns){
				total -= 25;
			}

			// If each square doesn't contain a white piece, boost the score for mobility
			if (!Config::ENABLE_CHEAP_ROOK_MOBILITY && !((occupied_white & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_black) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_black);

				if (!attacked_by_lower_value_piece) {

					mobility_bonus += 15;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;

						uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
				
						while (secondAttackMask) {
				
							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(secondAttackMask);		
							
							// If each square doesn't contain a white piece, boost the score for mobility
							if (!(simulatedOccupied & BB_SQUARES[secondary])){
								mobility_bonus += 10;
							}
							secondAttackMask &= secondAttackMask - 1;		
						}
					}
				}
			}
			bb &= bb - 1;		
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
		
					
		/* handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,ROOK);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_white;

		// Boost score for semi connected rooks			
		if ((unBlockedMask & (occupied_white & rooks)) != 0){
			total -= 125;
		}			

		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece
			total -= attackingLayer[0][x][y] >> 3;

			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r];
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 7;
			}
			bb &= bb - 1;
		}
		// Cheap mobility: scale popcounts of reachable non-own squares + forward zone (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_ROOK_MOBILITY){
			uint64_t mob_sq = pieceAttackMask & ~occupied_white;
			mobility_bonus = Config::CHEAP_ROOK_MOB * __builtin_popcountll(mob_sq)
						   + Config::CHEAP_ROOK_FWD * __builtin_popcountll(mob_sq & (BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8));
		}
		total -= std::min(mobility_bonus, 225);
		//std::cout << mobility_bonus  << std::endl;
	}else{
		// First add the piece value
        total += values[ROOK];
		blackPieceVal += values[ROOK];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y] >> 1;
		total += attackingLayer[0][x][y] >> 2;
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		blackOffensiveScore += attackingLayer[1][x][y];
		blackDefensiveScore += attackingLayer[0][x][y];
		
		// Boost the score if a rook is placed on the 2nd Rank
		if (y <= 1){
			rookIncrement += 150;
		}

		// Boost the score if rooks are connected			
		if (((BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied]) & (occupied_black & rooks)) != 0){
			rookIncrement += 150;
		} else if (((BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]) & (occupied_black & rooks)) != 0){
			rookIncrement += 125;
		}

		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces
		uint8_t r = 0;
		uint64_t bb = rooks_mask;
		while (bb) {
			
			// Get the current square as the max bit of the current mask
			r = 64 - __builtin_clzll(bb) - 1;
			uint8_t att_square = r;
			bb ^= (BB_SQUARES[r]);
			
			// Check if the attacked square is down the board from the rook
			if (att_square < square){
				
				// Get the piece type and colour
				uint8_t temp_piece_type = pieceTypeLookUp [att_square];
				bool temp_colour = bool(occupied_white & (BB_SQUARES[att_square]));
				
				/*
					In this section, update the rook increment based on how open the file is
					This includes pieces and pawns in the way of both colours
				*/
				
				// Check if the occupied piece is white (opposite of the rook)					
				if (temp_colour){
					
					// Check if the piece is the opponents (white) pawn
					if (temp_piece_type == 1){    
						 
						if (white_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (att_square >> 3) * 25;
						}else{
							// If the pawn is within the opponent's (first) half, lower the rook's increment
							if ((att_square >> 3) < 5){
								rookIncrement -= 50;
							}
						}	
												
					// If a white knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					// If a white rook is in the way, lower the rook increment
					}else if(temp_piece_type == 4){
						rookIncrement -= 35;
					}
				}else{
					
					// Check if the piece is the rook's own pawn
					if (temp_piece_type == 1){

						if (black_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (7 - (att_square >> 3)) * 50;
						}else{
							// If the pawn is within its own (second) half, lower the rook's increment and break the loop
							if ((att_square >> 3) > (Config::ENABLE_ROOK_RANKWIN_FIX ? 2 : 4)){
								rookIncrement -= (50 + (((att_square / 8) - 4) * 125));								
								break;
							}
						}
											
					// If a white knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					}
				}
			}
		}
		// Finally use the rook increment
		total += std::min(rookIncrement, 300);

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;		
		
		// Loop through the attacks mask
		r = 0;
		bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & square_mask & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & square_mask));
			}
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y] >> 1;
			total += attackingLayer[0][x][y] >> 2;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	

			update_global_central_scores(attackingLayer[1][x][y], square_mask);

			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & black_passed_pawns){
				total += 25;
			} else if(square_mask & white_passed_pawns){
				total += 25;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility
			if (!Config::ENABLE_CHEAP_ROOK_MOBILITY && !((occupied_black & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_white) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_white);

				if (!attacked_by_lower_value_piece) {

					mobility_bonus += 15;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;

						uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
				
						while (secondAttackMask) {
				
							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(secondAttackMask);		
							
							// If each square doesn't contain a white piece, boost the score for mobility
							
							if (!(simulatedOccupied & BB_SQUARES[secondary])){
								mobility_bonus += 10;
							}
							secondAttackMask &= secondAttackMask - 1;		
						}
					}
				}
			}			
			bb &= bb - 1; 
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
				
		/* handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,ROOK);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_black;

		// Boost score for semi connected rooks		
		if ((unBlockedMask & (occupied_black & rooks)) != 0){
			total += 125;
		}		
		
		// Loop through the xray attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece			
			total += attackingLayer[1][x][y] >> 3;			
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){				
				total += values[xRayPieceType] >> 7;								
			}
			bb &= bb - 1; 
		}			       
	
		// Cheap mobility: scale popcounts of reachable non-own squares + forward zone (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_ROOK_MOBILITY){
			uint64_t mob_sq = pieceAttackMask & ~occupied_black;
			mobility_bonus = Config::CHEAP_ROOK_MOB * __builtin_popcountll(mob_sq)
						   + Config::CHEAP_ROOK_FWD * __builtin_popcountll(mob_sq & (BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4));
		}
		total += std::min(mobility_bonus, 225);
		//std::cout << mobility_bonus  << std::endl;
	}
	return total;	
}

inline int evaluate_queens_midgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
    
	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
    
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	if(colour){
		// First subtract the piece value
        total -= values[QUEEN];
        whitePieceVal += values[QUEEN];
		
		// Subtract the placement layer for the given piece at that square
		total -= whitePlacementLayer[QUEEN - 1][x][y];

		// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
		total -= attackingLayer[0][x][y] >> 2;				
		total -= attackingLayer[1][x][y] >> 3;
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
		whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
            
        update_global_central_scores(-(whitePlacementLayer[QUEEN - 1][x][y]), BB_SQUARES[square]);
        //if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied] | BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);
			
			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){				
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
			// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
			total -= attackingLayer[0][x][y] >> 2;				
			total -= attackingLayer[1][x][y] >> 3;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
			whiteDefensiveScore += attackingLayer[1][x][y] >> 2;

			update_global_central_scores(-(attackingLayer[0][x][y] >> 1), square_mask);
			
			// Remove pieces from the copy of the occupied mask
			occupiedCopy &= ~(square_mask);

			if (square_mask & white_passed_pawns){
				total -= 25;
			} else if(square_mask & black_passed_pawns){
				total -= 25;
			}
			
			// If each square doesn't contain a white piece, boost the score for mobility			
			if (!Config::ENABLE_CHEAP_QUEEN_MOBILITY && !(occupied_white & square_mask)){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & (queens | bishops) & occupied_black) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & (queens | rooks) & occupied_black) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & (queens | rooks) & occupied_black);

				if (!attacked_by_lower_value_piece) {
					total -= 5;
				}	
			}

			bb &= bb - 1;		
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
			
		/* handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour); */

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test)
		if (Config::ENABLE_CHEAP_QUEEN_MOBILITY){
			total -= Config::CHEAP_QUEEN_MOB_MG * __builtin_popcountll(pieceAttackMask & ~occupied_white);
		}

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,QUEEN);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_white;

		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece			
			total -= attackingLayer[0][x][y] >> 3;

			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r];
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 8;
			}
			bb &= bb - 1;
		}
		
	}else{
		// First add the piece value
        total += values[QUEEN];
		blackPieceVal += values[QUEEN];
	
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[QUEEN - 1][x][y];

		// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
		total += attackingLayer[1][x][y] >> 2;
		total += attackingLayer[0][x][y] >> 3;
		
		// Similar to above, increment the absolute offensive and defensive scores
		// Bit shift to reduce global scores
		blackOffensiveScore += attackingLayer[1][x][y] >> 1;
		blackDefensiveScore += attackingLayer[0][x][y] >> 2;

		update_global_central_scores(blackPlacementLayer[QUEEN - 1][x][y], BB_SQUARES[square]);
		
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied] | BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;		
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 

			uint64_t square_mask = BB_SQUARES[r];
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & square_mask));
			} */
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
						
			// Add the score based on the attack of the opposing position and defense of black's own position
			// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
			total += attackingLayer[1][x][y] >> 2;
			total += attackingLayer[0][x][y] >> 3;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y] >> 1;
			blackDefensiveScore += attackingLayer[0][x][y] >> 2;	
			
			update_global_central_scores(attackingLayer[1][x][y] >> 1, square_mask);
			
			// Remove pieces from the copy of the occupied mask
			occupiedCopy &= ~(BB_SQUARES[r]);

			if (square_mask & black_passed_pawns){
				total += 25;
			} else if(square_mask & white_passed_pawns){
				total += 25;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility
						
			if (!Config::ENABLE_CHEAP_QUEEN_MOBILITY && !(occupied_black & square_mask)){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & (queens | bishops) & occupied_white) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & (queens | rooks) & occupied_white) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & (queens | rooks) & occupied_white);

				if (!attacked_by_lower_value_piece) {
					total += 5;
				}	
			}
			
			bb &= bb - 1; 
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
		
		/* handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour); */

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test)
		if (Config::ENABLE_CHEAP_QUEEN_MOBILITY){
			total += Config::CHEAP_QUEEN_MOB_MG * __builtin_popcountll(pieceAttackMask & ~occupied_black);
		}

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,QUEEN);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_black;

		
		// Loop through the xray attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece			
			total += attackingLayer[1][x][y] >> 3;
						
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){				
				total += values[xRayPieceType] >> 8;								
			}
			bb &= bb - 1; 
		}					
	}
	return total;
}

inline int evaluate_kings_midgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
        	
	bool colour = bool(occupied_white & (BB_SQUARES[square])); 
    
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;
    uint8_t x = square & 7;

	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value
        total -= values[KING];
        whitePieceVal += values[KING];

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KING_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
									
			uint8_t kingRank = square >> 3;
			uint64_t r_mask = BB_SQUARES[r];

			bool isWhitePawn = pieceTypeLookUp[r] == 1 && (occupied_white & r_mask);
			bool isInfront = y == kingRank + 1;
			bool isShielding = isWhitePawn && isInfront;

			// Subtract the score based on the attack of the opposing position and absolute offensive score				
			total -= attackingLayer[0][x][y];   
			whiteOffensiveScore += attackingLayer[0][x][y];
			
			// Boost the king score for having the protection of its own pawns
			// Otherwise keep the local and global defensive score as normal
			
			int baseIncrement = attackingLayer[1][x][y];
			if (kingRank == 0) {									
				bool isPartialShielding = (isInfront && (BB_SQUARES[r + 8] & occupied_white & pawns) != 0);					
				if (isShielding) {
					whiteDefensiveScore += (baseIncrement << 2);
					total -= (baseIncrement << 2) + 185;									
				}else if(isPartialShielding){
					whiteDefensiveScore += (baseIncrement << 1);
					total -= (baseIncrement << 1) + 75;
				}else {
					whiteDefensiveScore += baseIncrement;
					total += baseIncrement >> 2;
				}
				
			} else {
				if (isShielding) {
					whiteDefensiveScore += baseIncrement;
					total -= baseIncrement;				
				} else {
					whiteDefensiveScore -= (baseIncrement << 1) / 2;
					total += baseIncrement << 1;
				}
			}					
			bb &= bb - 1;		
		}
	} else{
		// First add the piece value
        total += values[KING];
		blackPieceVal += values[KING];

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_KING_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			uint8_t kingRank = square >> 3;
			uint64_t r_mask = BB_SQUARES[r];

			bool isBlackPawn = pieceTypeLookUp[r] == 1 && (occupied_black & r_mask);
			bool isInfront = y == kingRank - 1;
			bool isShielding = isBlackPawn && isInfront;			

			// Subtract the score based on the attack of the opposing position and absolute offensive score				
			total += attackingLayer[1][x][y];
			blackOffensiveScore += attackingLayer[1][x][y];
			
			int baseIncrement = attackingLayer[0][x][y];
			if (kingRank == 7) {
				bool isPartialShielding = (isInfront && (BB_SQUARES[r - 8] & occupied_black & pawns) != 0);					
				if (isShielding) {
					blackDefensiveScore += (baseIncrement << 2);
					total += (baseIncrement << 2) + 185;									
				}else if(isPartialShielding){
					blackDefensiveScore += (baseIncrement << 1);
					total += (baseIncrement << 1) + 75;
				}else {
					blackDefensiveScore += baseIncrement;
					total -= baseIncrement >> 2;
				}
			} else {
				if (isShielding) {
					blackDefensiveScore += baseIncrement;
					total += baseIncrement;				
				} else {
					blackDefensiveScore -= (baseIncrement << 1) / 2;
					total -= baseIncrement << 1;
				}
			}			
			bb &= bb - 1; 
		}
	}
	return total;
}


inline int evaluate_pawns_endgame(uint8_t square, uint64_t& white_passed_pawns, uint64_t& black_passed_pawns, int& pawn_rank_bonus){
	
	// Initialize the evaluation
    int total = 0;
	int structural_bonus = 0;
	
	// Initialize the maximum increment for pawn placement
    int ppIncrement = 400;	
    
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;

	if(colour){
		// First subtract the piece value and increment the global white piece value
        total -= values[PAWN];		
        whitePieceVal += values[PAWN];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		total -= attackingLayer[0][x][y];
		total -= attackingLayer[1][x][y] >> 1;

		// Lower white's score for more than one white pawn being on the same file
		total += 150 * (__builtin_popcountll(BB_FILES[x] & (occupied_white & pawns)) > 1);
			
		// Call the function to acquire an extra pawn squared based on the position of opposing pawns
		// Only consider this if the pawn is above the 3rd rank
		
		ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white, white_passed_pawns, black_passed_pawns);
		ppIncrement = std::min(ppIncrement, 600); // cap runaway boosts

		int rank = y;		
		
		pawn_rank_bonus = -((3 * default_midgame_pawn_rank_bonus[rank] * (ppIncrement < 300)) + ((endgame_pawn_rank_bonus[rank] + (ppIncrement >> 2)) * (ppIncrement >= 300)));
		total += pawn_rank_bonus;
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_white & pawns;
		uint64_t sw  = (BB_SQUARES[square] >> 9) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t se = (BB_SQUARES[square] >> 7) & ~BB_FILE_A & occupied_white & pawns;

		uint64_t latent_left_support_mask = latent_support_mask_left(square, colour);
		uint64_t latent_right_support_mask = latent_support_mask_right(square, colour);

		structural_bonus += 100 * (left != 0);
		structural_bonus += 100 * (right != 0);
		
		structural_bonus += 135 * (sw != 0);
		structural_bonus += 135 * (se != 0);

		structural_bonus += 50 * (sw == 0) * ((latent_left_support_mask & occupied_white & pawns) != 0) * ((latent_left_support_mask & occupied_black) == 0);
		structural_bonus += 50 * (se == 0) * ((latent_right_support_mask & occupied_white & pawns) != 0) * ((latent_right_support_mask & occupied_black) == 0);  
		//std::cout << total << std::endl;
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_PAWN_ATTACKS[colour][square];	
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									
			uint64_t square_mask = BB_SQUARES[r];
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;	
			
			if (square_mask & white_passed_pawns){
				total -= 200;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}			
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			structural_bonus += 115 * ((BB_SQUARES[r] & occupied_white & pawns) != 0);

			bb &= bb - 1;   	
		}
		total -= std::min(175, structural_bonus);
		//std::cout << structural_bonus<< std::endl;
	}else{
		// First subtract the piece value and increment the global white piece value
        total += values[PAWN];
		blackPieceVal += values[PAWN];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y];
		total += attackingLayer[0][x][y] >> 1;
		
		// Lower white's score for more than one white pawn being on the same file
		total -= 150 * (__builtin_popcountll(BB_FILES[x] & (occupied_black & pawns)) > 1);
		
		// Call the function to acquire an extra pawn squared based on the position of opposing pawns
		// Only consider this if the pawn is below the 6th rank
		ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black, white_passed_pawns, black_passed_pawns);
		ppIncrement = std::min(ppIncrement, 600); // cap runaway boosts

		int rank = 7 - y;
				
		pawn_rank_bonus = (3 * default_midgame_pawn_rank_bonus[rank] * (ppIncrement < 300)) + ((endgame_pawn_rank_bonus[rank] + (ppIncrement >> 2)) * (ppIncrement >= 300));
		total += pawn_rank_bonus;
		
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_black & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_black & pawns;
		uint64_t ne  = (BB_SQUARES[square] << 9) & ~BB_FILE_H & occupied_black & pawns;
		uint64_t nw = (BB_SQUARES[square] << 7) & ~BB_FILE_A & occupied_black & pawns;

		uint64_t latent_left_support_mask = latent_support_mask_left(square, colour);
		uint64_t latent_right_support_mask = latent_support_mask_right(square, colour);
		
		structural_bonus += 100 * (left != 0);
		structural_bonus += 100 * (right != 0);	
		structural_bonus += 135 * (nw != 0);
		structural_bonus += 135 * (ne != 0);
		structural_bonus += 50 * (nw == 0) * ((latent_left_support_mask & occupied_black & pawns) != 0) * ((latent_left_support_mask & occupied_white) == 0);
		structural_bonus += 50 * (ne == 0) * ((latent_right_support_mask & occupied_black & pawns) != 0) * ((latent_right_support_mask & occupied_white) == 0);  

		/*
			In this section, the scores for piece attacks are acquired
		*/
		//std::cout << total << std::endl;
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_PAWN_ATTACKS[colour][square];		
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									
			uint64_t square_mask = BB_SQUARES[r];
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;

			if (square_mask & black_passed_pawns){
				total += 200;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/											
			structural_bonus += 115 * ((BB_SQUARES[r] & occupied_black & pawns) != 0);
						
			bb &= bb - 1; 
		}
		total += std::min(175, structural_bonus);
		//std::cout << structural_bonus<< std::endl;
	}	
	return total;			        
}

inline int evaluate_knights_endgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	    
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     
	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;

	if(colour){
		// First subtract the piece value and increment the global white piece value
        total -= values[KNIGHT];		
        whitePieceVal += values[KNIGHT];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		total -= attackingLayer[0][x][y];
		total -= attackingLayer[1][x][y] >> 1;
				
		total -= 200;
		/*
			In this section, the scores for piece attacks are acquired
		*/

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_KNIGHT_MOBILITY){
			total -= Config::CHEAP_KNIGHT_MOB * __builtin_popcountll(pieceAttackMask & ~occupied_white);
		}
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			

			if (square_mask & white_passed_pawns){
				total -= 150;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}
				
			// If each square doesn't contain a white piece, boost the score for mobility			
			if (!Config::ENABLE_CHEAP_KNIGHT_MOBILITY && !((occupied_white & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black);

				if (!attacked_by_lower_value_piece) {

					total -= (Config::ENABLE_KNIGHT_MOB_SYM_UP ? 15 : 10);

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_KNIGHT_ATTACKS[r];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {

							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(bb_second);

							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility
							if (!(simulatedOccupied & second_sq)){
								total -= 10;
								//std::cout << (int)secondary << " | " << BB_SQUARES[secondary] <<std::endl;								
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}		
			
			bb &= bb - 1;   	
		}
		
	}else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[KNIGHT];
		blackPieceVal += values[KNIGHT];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y];
		total += attackingLayer[0][x][y] >> 1;

		total += 200;
		/*
			In this section, the scores for piece attacks are acquired
		*/

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_KNIGHT_MOBILITY){
			total += Config::CHEAP_KNIGHT_MOB * __builtin_popcountll(pieceAttackMask & ~occupied_black);
		}
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);	
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;

			if (square_mask & black_passed_pawns){
				total += 150;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility  			
			if (!Config::ENABLE_CHEAP_KNIGHT_MOBILITY && !((occupied_black & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				//uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				//uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white);

				if (!attacked_by_lower_value_piece) {

					total += (Config::ENABLE_KNIGHT_MOB_FIX ? 10 : 15);

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_KNIGHT_ATTACKS[r];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask							
							uint8_t secondary = __builtin_ctzll(bb_second);
															
							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility							
							if (!(simulatedOccupied & second_sq)){
								total += 10;								
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
						
			bb &= bb - 1; 
		}
	}
	return total;
}

inline int evaluate_bishops_endgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;
        
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value and increment the global white piece value
        total -= values[BISHOP];		
        whitePieceVal += values[BISHOP];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		total -= attackingLayer[0][x][y];
		total -= attackingLayer[1][x][y] >> 1;	
		
		// Boost the scores for the existence of a bishop or knight		
		total -= 250;
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
							
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);

			if (square_mask & white_passed_pawns){
				total -= 150;
			} else if(square_mask & black_passed_pawns){
				total -= 100;
			}
			
			// If each square doesn't contain a white piece, boost the score for mobility			
			if (!((occupied_white & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black);

				if (!attacked_by_lower_value_piece) {
		
					total -= 10;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask							
							uint8_t secondary = __builtin_ctzll(bb_second);
							
							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility
							if (!(simulatedOccupied & second_sq)){
								total -= 10;
								//std::cout << (int)secondary << " | " << BB_SQUARES[secondary] <<std::endl;
								
								if ((second_sq & BB_RANK_8) != 0 || (second_sq & BB_RANK_7) != 0) {
									total -= 5;
								}
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
			
			bb &= bb - 1;   	
		}		
			
		/* handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,BISHOP)) & ~occupied_white;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece				
			total -= attackingLayer[0][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 6;
			}
			bb &= bb - 1; 	
		}
		
	}else{
		// First subtract the piece value and increment the global white piece value
        total += values[BISHOP];
		blackPieceVal += values[BISHOP];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y];
		total += attackingLayer[0][x][y] >> 1;
		
		// Boost the scores for the existence of a bishop or knight
        total += 250;		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & black_passed_pawns){
				total += 150;
			} else if(square_mask & white_passed_pawns){
				total += 100;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility  			
			if (!((occupied_black & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white);

				if (!attacked_by_lower_value_piece) {

					total += 10;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask
							
							uint8_t secondary = __builtin_ctzll(bb_second);
							
								
							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility							
							if (!(simulatedOccupied & second_sq)){
								total += 10;
								
								if ((second_sq & BB_RANK_1) != 0 || (second_sq & BB_RANK_2) != 0) {
									total += 5;
								}
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
						
			bb &= bb - 1; 
		}		
			
		/* handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,BISHOP)) & ~occupied_black;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece
			total += attackingLayer[1][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total += values[xRayPieceType] >> 6;
			}
			bb &= bb - 1;  
		}		
	}
	return total;
}

inline int evaluate_rooks_endgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	int mobility_bonus = 0;
	
	// Define the maximum increment for rook-file positioning and attack mask
    int rookIncrement = 200;
	uint64_t rooks_mask = 0ULL;
	
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;
        
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value and increment the global white piece value
        total -= values[ROOK];		
        whitePieceVal += values[ROOK];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		total -= attackingLayer[0][x][y];
		total -= attackingLayer[1][x][y] >> 1;	
		
		// Boost the score for the existence of a rook in the endgame
		total -= 350;

		
		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces
		uint8_t r = 0;
		uint64_t bb = rooks_mask;
		while (bb) {
			
			// Get the current square as the max bit of the current mask
			r = __builtin_ctzll(bb);							
			uint8_t att_square = r;
			
			// Get the piece type and colour
			uint8_t temp_piece_type = pieceTypeLookUp [att_square];
			bool temp_colour = bool(occupied_white & (BB_SQUARES[att_square]));
			
			// Check if the piece is a pawn
			if (temp_piece_type == 1){
				
				// Up the board from the white rook
				if (att_square > square){ 
				
					// Check if the pawn is white 
					if (temp_colour){ 
						if (white_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (att_square / 8) * 75;
						}else{
							// Increment rook for supporting the white pawn
							rookIncrement += (att_square / 8) * 35; 
						}						
					
					// Check if the pawn is black
					}else{
						if (black_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (7 - (att_square / 8)) * 75;
						}else{
							// Increment rook for blockading black pawn  
							rookIncrement += (7 - (att_square / 8)) * 50; 
						}						
					}		
					break;		
				// Down the board from the white rook
				}else { 
					// Check if the pawn is white
					if (temp_colour){ 
						if (att_square / 8 > 3){
							
							// Decrement rook for blocking own pawn
							rookIncrement -= 50 + ((att_square / 8) - 3) * 50; 
						}
					
					// Check if the pawn is black
					}else{ 
					
						// Increment rook for attacking black pawn from behind

						rookIncrement += (Config::ENABLE_ROOK_DBLCOUNT_SYM_UP ? (7 - (att_square / 8)) * 35 : 0);

						if (black_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (7 - (att_square / 8)) * 75;
						}else{							
							rookIncrement += (7 - (att_square / 8)) * 35; 
						}
					}
				}
			}
			bb &= bb - 1;   
		}
		
		// Finally use the increment
		total -= (Config::ENABLE_ROOK_ENDGAME_CAP ? std::min(rookIncrement, Config::ROOK_ENDGAME_CAP) : rookIncrement);
        		
		/*
			In this section, the scores for piece attacks are acquired
		*/		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		r = 0;
		bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & white_passed_pawns){
				total -= 50;
			} else if(square_mask & black_passed_pawns){
				total -= 25;
			}
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (!Config::ENABLE_CHEAP_ROOK_MOBILITY && !((occupied_white & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_black) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_black);

				if (!attacked_by_lower_value_piece) {

					mobility_bonus += 5;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;

						uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
				
						while (secondAttackMask) {
				
							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(secondAttackMask);		
							
							
							// If each square doesn't contain a piece, boost the score for mobility
							uint64_t sq = BB_SQUARES[secondary];
							if (!(simulatedOccupied & sq)){
								mobility_bonus += 5;
								
								if ((sq & BB_RANK_8) || (sq & BB_RANK_7)) {
									mobility_bonus += 5;
								}
							}
							secondAttackMask &= secondAttackMask - 1;		
						}
					}
				}
			}
			
			bb &= bb - 1;   	
		}
							
		/* handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,ROOK)) & ~occupied_white;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece				
			total -= attackingLayer[0][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 6;
			}
			bb &= bb - 1; 	
		}
		
		// Cheap mobility: scale popcounts of reachable non-own squares + forward zone (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_ROOK_MOBILITY){
			uint64_t mob_sq = pieceAttackMask & ~occupied_white;
			mobility_bonus = Config::CHEAP_ROOK_MOB * __builtin_popcountll(mob_sq)
						   + Config::CHEAP_ROOK_FWD * __builtin_popcountll(mob_sq & (BB_RANK_7 | BB_RANK_8));
		}
		total -= std::min(mobility_bonus, 350);
		//std::cout << mobility_bonus  << std::endl;
	// Else the piece is black (positive values for evaluation)
    }else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[ROOK];
		blackPieceVal += values[ROOK];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y];
		total += attackingLayer[0][x][y] >> 1;
				
		// Boost the score for the existence of a rook in the endgame
		total += 350;
		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces		
		uint8_t r = 0;
		uint64_t bb = rooks_mask;
		while (bb) {
			
			// Get the current square as the max bit of the current mask
			r = 64 - __builtin_clzll(bb) - 1;			
			uint8_t att_square = r;
			
			// Get the piece type and colour
			uint8_t temp_piece_type = pieceTypeLookUp [att_square];
			bool temp_colour = bool(occupied_white & (BB_SQUARES[att_square]));
			
			// Check if the piece is a pawn
			if (temp_piece_type == 1){    
			
				// Down the board from the black rook
				if (att_square < square){ 
					
					// If the pawn is white
					if (temp_colour){ 
						if (white_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (att_square / 8) * 75;
						}else{
							// Increment rook for blockading white pawn
							rookIncrement += (att_square / 8) * 50;      
						}									                  
					// If the pawn is black
					}else{
						if (black_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (7 - (att_square / 8)) * 75;
						}else{
							// Increment rook for supporting the black pawn
							rookIncrement += (7 - (att_square / 8)) * 35; 
						}							
					}
					break;
				// Up the board from the black rook
				}else{ 
				
					// If the pawn is white
					if (temp_colour){ 
					
						// Increment rook for attacking white pawn from behind
						rookIncrement += (Config::ENABLE_ROOK_DBLCOUNT_FIX ? 0 : (att_square / 8) * 35); 

						if (white_passed_pawns & BB_SQUARES[att_square]){
							rookIncrement += (att_square / 8) * 75;
						}else{							
							rookIncrement += (att_square / 8) * 35;      
						}	
					// If the pawn is black
					}else{ 
						if (att_square / 8 < 4){
							
							// Decrement rook for blocking own pawn
							rookIncrement -= 50 + (4 - (att_square / 8)) * 50; 
						}
					}
				}
			}
			bb ^= (BB_SQUARES[r]);			
		}	

		// Finally use the increment
		total += (Config::ENABLE_ROOK_ENDGAME_CAP ? std::min(rookIncrement, Config::ROOK_ENDGAME_CAP) : rookIncrement);
		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
        		
		// Loop through the attacks mask
		r = 0;
		bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);	
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & black_passed_pawns){
				total += 50;
			} else if(square_mask & white_passed_pawns){
				total += 25;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility
			if (!Config::ENABLE_CHEAP_ROOK_MOBILITY && !((occupied_black & square_mask))){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_white) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_white);

				if (!attacked_by_lower_value_piece) {

					mobility_bonus += 5;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | square_mask;

						uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
				
						while (secondAttackMask) {
				
							// Get the position of the least significant set bit of the mask
							uint8_t secondary = __builtin_ctzll(secondAttackMask);		
							
							// If each square doesn't contain a white piece, boost the score for mobility
							uint64_t sq = BB_SQUARES[secondary];
							if (!(simulatedOccupied & sq)){
								mobility_bonus += 5;
								
								if ((sq & BB_RANK_1) || (sq & BB_RANK_2)) {
									mobility_bonus += 5;
								}
							}
							secondAttackMask &= secondAttackMask - 1;		
						}
					}
				}
			}
						
			bb &= bb - 1; 
		}
		
			
		/* handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour); */

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,ROOK)) & ~occupied_black;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece
			total += attackingLayer[1][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total += values[xRayPieceType] >> 6;
			}
			bb &= bb - 1;  
		}
    
		// Cheap mobility: scale popcounts of reachable non-own squares + forward zone (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_ROOK_MOBILITY){
			uint64_t mob_sq = pieceAttackMask & ~occupied_black;
			mobility_bonus = Config::CHEAP_ROOK_MOB * __builtin_popcountll(mob_sq)
						   + Config::CHEAP_ROOK_FWD * __builtin_popcountll(mob_sq & (BB_RANK_1 | BB_RANK_2));
		}
		total += std::min(mobility_bonus, 350);
		//std::cout << mobility_bonus  << std::endl;
	}
	//std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << std::endl;
	return total;
}

inline int evaluate_queens_endgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;
        
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value and increment the global white piece value
        total -= values[QUEEN];		
        whitePieceVal += values[QUEEN];

		// Subtract the score based on the attack of the opposing position and defense of white's own position
		total -= attackingLayer[0][x][y];
		total -= attackingLayer[1][x][y] >> 1;	

		// Boost the score for the existence of a queen in the endgame
		total -= 900;
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied] | BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & white_passed_pawns){
				total -= 100;
			} else if(square_mask & black_passed_pawns){
				total -= 50;
			}
			
			// If each square doesn't contain a white piece, boost the score for mobility			
			if (!Config::ENABLE_CHEAP_QUEEN_MOBILITY && !(occupied_white & square_mask)){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_black) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_black) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_black) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_black) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_black);

				if (!attacked_by_lower_value_piece) {
		
					total -= 5;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied] | BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask							
							uint8_t secondary = __builtin_ctzll(bb_second);
							
							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility
							if (!(simulatedOccupied & second_sq)){
								total -= 5;
								//std::cout << (int)secondary << " | " << BB_SQUARES[secondary] <<std::endl;
								
								if ((second_sq & BB_RANK_8) != 0 || (second_sq & BB_RANK_7) != 0) {
									total -= 5;
								}
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
			
			bb &= bb - 1;   	
		}
				
		/* handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour); */

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_QUEEN_MOBILITY){
			total -= Config::CHEAP_QUEEN_MOB_EG * __builtin_popcountll(pieceAttackMask & ~occupied_white);
		}

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,QUEEN)) & ~occupied_white;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece				
			total -= attackingLayer[0][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 6;
			}
			bb &= bb - 1; 	
		}
		
	// Else the piece is black (positive values for evaluation)
    }else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[QUEEN];
		blackPieceVal += values[QUEEN];

		// Subtract the score based on the attack of the opposing position and defense of black's own position
		total += attackingLayer[1][x][y];
		total += attackingLayer[0][x][y] >> 1;

		// Boost the score for the existence of a queen in the endgame
		total += 900;
		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied] | BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		uint64_t occupiedCopy = occupied;
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & square_mask & ~kings){
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & square_mask));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(square_mask);

			if (square_mask & black_passed_pawns){
				total += 100;
			} else if(square_mask & white_passed_pawns){
				total += 50;
			}
			
			// If each square doesn't contain a black piece, boost the score for mobility  			
			if (!Config::ENABLE_CHEAP_QUEEN_MOBILITY && !(occupied_black & square_mask)){

				uint64_t diag_pieces = BB_DIAG_MASKS[r] & occupied;
				uint64_t rank_pieces = BB_RANK_MASKS[r] & occupied;
				uint64_t file_pieces = BB_FILE_MASKS[r] & occupied;

				bool attacked_by_lower_value_piece = (BB_PAWN_ATTACKS[colour][r] & pawns & occupied_white) ||
													 (BB_KNIGHT_ATTACKS[r] & knights & occupied_white) ||
													 (BB_DIAG_ATTACKS[r][diag_pieces] & bishops & occupied_white) ||
													 (BB_RANK_ATTACKS[r][rank_pieces] & rooks & occupied_white) ||
     												 (BB_FILE_ATTACKS[r][file_pieces] & rooks & occupied_white);

				if (!attacked_by_lower_value_piece) {

					total += 5;

					if (!(occupied & square_mask)){

						uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) & ~BB_SQUARES[r]; // Remove piece from both square and r
						simulatedOccupied |= BB_SQUARES[r]; // Place piece at r

						uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied] | BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
						//uint64_t secondAttackMask = attacks_mask(colour,simulatedOccupied,r,QUEEN);
						uint64_t bb_second = secondAttackMask;
						while (bb_second) {
				
							// Get the position of the least significant set bit of the mask
							
							uint8_t secondary = __builtin_ctzll(bb_second);
							
								
							uint64_t second_sq = BB_SQUARES[secondary];
							// If each square doesn't contain a white piece, boost the score for mobility							
							if (!(simulatedOccupied & second_sq)){
								total += 5;
								
								if ((second_sq & BB_RANK_1) != 0 || (second_sq & BB_RANK_2) != 0) {
									total += 5;
								}
							}
							bb_second &= bb_second - 1;		
						}
					}
				}
			}
						
			bb &= bb - 1; 
		}
		
		/* handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour); */

		// Cheap mobility: scale the popcount of reachable non-own squares (skips the per-square attacker test + 2nd-order scan)
		if (Config::ENABLE_CHEAP_QUEEN_MOBILITY){
			total += Config::CHEAP_QUEEN_MOB_EG * __builtin_popcountll(pieceAttackMask & ~occupied_black);
		}

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,QUEEN)) & ~occupied_black;
		
		// Loop through the attacks mask
		r = 0;
		bb = xRayMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			// Get the x and y coordinates for the given square
			y = r >> 3;
			x = r & 7;
			
			// Subtract a reduced score for square attacks behind a piece
			total += attackingLayer[1][x][y] >> 1;
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total += values[xRayPieceType] >> 6;
			}
			bb &= bb - 1;  
		}		
    }
	//std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << std::endl;
	return total;
}

inline int evaluate_kings_endgame(uint8_t square, uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	// Initialize the evaluation
    int total = 0;
	
	bool colour = bool(occupied_white & (BB_SQUARES[square]));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square >> 3;	
    uint8_t x = square & 7;
        
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value and increment the global white piece value
        total -= values[KING];		
        whitePieceVal += values[KING];

		// Add the placement layer for the given piece at that square
		total -= whitePlacementLayer[KING - 1][x][y];

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KING_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									
			uint64_t square_mask = BB_SQUARES[r];
			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;

			if (square_mask & white_passed_pawns){
				total -= 150;
			} else if(square_mask & black_passed_pawns){
				total -= 125;
			}
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
			
			bb &= bb - 1;   	
		}
				
	// Else the piece is black (positive values for evaluation)
    }else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[KING];
		blackPieceVal += values[KING];

		total += blackPlacementLayer[KING - 1][x][y];
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KING_ATTACKS[square];
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									
			uint64_t square_mask = BB_SQUARES[r];

			attack_bitmasks[r] |= BB_SQUARES[square];

			/* if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			} */

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;

			if (square_mask & black_passed_pawns){
				total += 150;
			} else if(square_mask & white_passed_pawns){
				total += 125;
			}
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
					
			bb &= bb - 1; 
		}
	}
	//std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << std::endl;
	return total;
}


inline uint64_t latent_support_mask_left(int square, bool is_white) {
    int file = square % 8;
    int rank = square / 8;
    uint64_t mask = 0ULL;

    if (file > 0) {
        for (int dr = 1; dr <= 3; ++dr) {
            int r = is_white ? (rank - dr) : (rank + dr);
            if (r < 0 || r > 7) break;

            int sq = r * 8 + (file - 1);
            mask |= BB_SQUARES[sq];
        }
    }

    return mask;
}

inline uint64_t latent_support_mask_right(int square, bool is_white) {
    int file = square % 8;
    int rank = square / 8;
    uint64_t mask = 0ULL;

    if (file < 7) {
        for (int dr = 1; dr <= 3; ++dr) {
            int r = is_white ? (rank - dr) : (rank + dr);
            if (r < 0 || r > 7) break;

            int sq = r * 8 + (file + 1);
            mask |= BB_SQUARES[sq];
        }
    }

    return mask;
}


inline void update_global_central_scores(int base_increment, uint64_t square_mask){
	if (square_mask & central_squares){
		central_score += base_increment * 2;
	}else if(square_mask & extended_central_squares){
		central_score += (base_increment * 3) / 2;
	}
}

inline void update_pressure_and_support_tables(uint8_t current_square, uint8_t attacking_piece_type, uint8_t decrement, bool attacking_piece_colour, bool current_piece_colour){
	
	if (attacking_piece_colour){
		if (current_piece_colour){
			support_white[current_square] += support_weights[attacking_piece_type][pieceTypeLookUp[current_square]] - decrement;
			num_supporters[current_square] += 1;
		} else {
			pressure_white[current_square] += pressure_weights[attacking_piece_type][pieceTypeLookUp[current_square]] - decrement;
			num_attackers[current_square] += 1;
		}
	} else {
		if (current_piece_colour){
			pressure_black[current_square] += pressure_weights[attacking_piece_type][pieceTypeLookUp[current_square]] - decrement;
			num_attackers[current_square] += 1;			
		} else {
			support_black[current_square] += support_weights[attacking_piece_type][pieceTypeLookUp[current_square]] - decrement;
			num_supporters[current_square] += 1;
		}
	}		
}

inline void handle_batteries_for_pressure_and_support_tables(uint8_t attacking_piece_square, uint8_t attacking_piece_type, uint64_t prev_attack_mask, bool attacking_piece_colour){

	uint64_t file_and_rank_attacks = 0;
	uint64_t diagonal_attacks = 0;	
	uint64_t removal_set = 0;
	uint64_t current_colour_mask = attacking_piece_colour ? occupied_white
                         : occupied_black;
	
	while(true){
		if (attacking_piece_type == 5){

			diagonal_attacks = (~diagonal_attacks) & BB_DIAG_ATTACKS[attacking_piece_square][BB_DIAG_MASKS[attacking_piece_square] & (occupied & ~removal_set)];
			file_and_rank_attacks = (~file_and_rank_attacks) & (BB_RANK_ATTACKS[attacking_piece_square][BB_RANK_MASKS[attacking_piece_square] & (occupied & ~removal_set)] |
									BB_FILE_ATTACKS[attacking_piece_square][BB_FILE_MASKS[attacking_piece_square] & (occupied & ~removal_set)]);

			// Determine blockers: same-color bishops or queens on the diagonal
			uint64_t same_colour_diagonal_sliders = diagonal_attacks & (bishops | queens) & current_colour_mask & ~removal_set;
			uint64_t same_colour_file_and_rank_sliders = file_and_rank_attacks & (rooks | queens) & current_colour_mask & ~removal_set;
			
			if ((same_colour_diagonal_sliders | same_colour_file_and_rank_sliders) == 0) {
				break;
			}

			// Add them to the removal set
			removal_set |= (same_colour_diagonal_sliders | same_colour_file_and_rank_sliders);

			loop_and_update((diagonal_attacks | file_and_rank_attacks) & (occupied & ~removal_set) & (~prev_attack_mask), attacking_piece_type, attacking_piece_colour, 0);

		} else if (attacking_piece_type == 4){
			file_and_rank_attacks = (~file_and_rank_attacks) & (BB_RANK_ATTACKS[attacking_piece_square][BB_RANK_MASKS[attacking_piece_square] & (occupied & ~removal_set)] |
									BB_FILE_ATTACKS[attacking_piece_square][BB_FILE_MASKS[attacking_piece_square] & (occupied & ~removal_set)]);

			// Determine blockers: same-color bishops or queens on the diagonal
			uint64_t same_colour_file_and_rank_sliders = file_and_rank_attacks & (rooks | queens) & current_colour_mask & ~removal_set;
			
			if (same_colour_file_and_rank_sliders == 0) {
				break;
			}

			// Add them to the removal set
			removal_set |= same_colour_file_and_rank_sliders;

			loop_and_update(file_and_rank_attacks & (occupied & ~removal_set) & (~prev_attack_mask), attacking_piece_type, attacking_piece_colour, 5);
				
		} else if (attacking_piece_type == 3){
			diagonal_attacks = (~diagonal_attacks) & BB_DIAG_ATTACKS[attacking_piece_square][BB_DIAG_MASKS[attacking_piece_square] & (occupied & ~removal_set)];

			// Determine blockers: same-color bishops or queens on the diagonal
			uint64_t same_colour_diagonal_sliders = diagonal_attacks & (bishops | queens) & current_colour_mask & ~removal_set;
			
			if (same_colour_diagonal_sliders == 0) {
				break;
			}

			// Add them to the removal set
			removal_set |= same_colour_diagonal_sliders;

			loop_and_update(diagonal_attacks & (occupied & ~removal_set) & (~prev_attack_mask), attacking_piece_type, attacking_piece_colour, 5);			
		}
	}	
}

inline void loop_and_update(uint64_t bb, uint8_t attacking_piece_type, bool attacking_piece_colour, int decrement) {
    uint8_t r = 0;
	while (bb) {
        r = __builtin_ctzll(bb);
        update_pressure_and_support_tables(r, attacking_piece_type, decrement, attacking_piece_colour, bool(occupied_white & (BB_SQUARES[r])));
        bb &= bb - 1;
    }
}

inline void adjust_pressure_and_support_tables_for_pins(uint64_t bb){

	while (bb) {
        uint8_t current_square = __builtin_ctzll(bb);
    	bb &= bb - 1;
		
		bool current_square_colour = bool(occupied_white & (BB_SQUARES[current_square])); 
		uint8_t current_square_piece = pieceTypeLookUp[current_square];
		
		uint64_t currentSidePieces = current_square_colour ? occupied_white : occupied_black;
		uint64_t opposingPieces = current_square_colour ? occupied_black : occupied_white;
		
		// Acquire the masks of the pieces on the same rank, file and diagonal as the given square
		uint64_t rank_pieces = BB_RANK_MASKS[current_square] & occupied;
		uint64_t file_pieces = BB_FILE_MASKS[current_square] & occupied;
		uint64_t diag_pieces = BB_DIAG_MASKS[current_square] & occupied;

		// Acquire all attack masks for each piece type
		uint64_t attackers = (
			(BB_RANK_ATTACKS[current_square][rank_pieces] & (queens | rooks)) |
			(BB_FILE_ATTACKS[current_square][file_pieces] & (queens | rooks)) |
			(BB_DIAG_ATTACKS[current_square][diag_pieces] & (queens | bishops))
		);

		// Perform a bitwise and with the opposing pieces 
		uint64_t sliding_attackers_mask = attackers & opposingPieces;
		uint64_t sliding_attackers = sliding_attackers_mask & (queens | rooks | bishops);

		uint8_t sliding_attacker = 0;
		uint8_t target = 0;

		int decrement = 0;
		int pressure_increase = 0;

		while (sliding_attackers) {
			sliding_attacker = __builtin_ctzll(sliding_attackers);
			sliding_attackers &= sliding_attackers - 1;

			uint8_t pinning_piece_type = pieceTypeLookUp[sliding_attacker];
			uint64_t attacked_pieces_behind_target = ~(BB_SQUARES[current_square]) & attacks_mask(!current_square_colour,occupied & ~(BB_SQUARES[current_square]),sliding_attacker ,pinning_piece_type) & currentSidePieces;
			//std::cout << "Pinned: " << (int)current_square << " Pinner " << int(sliding_attacker) << " attacked_pieces_behind_target: " << attacked_pieces_behind_target <<" | " <<  sliding_attackers_mask << std::endl;
			while (attacked_pieces_behind_target) {
				target = __builtin_ctzll(attacked_pieces_behind_target);
				attacked_pieces_behind_target &= attacked_pieces_behind_target - 1;

				uint8_t pinned_to_piece_type = pieceTypeLookUp[target];		

				if (pinned_to_piece_type == 6){
					decrement += decrement_lookup[pinned_to_piece_type];
					pressure_increase += pressure_increase_lookup[pinned_to_piece_type];
					continue;
				}
				
				int new_pressure = pressure_weights[pinning_piece_type][pinned_to_piece_type];
				int pressure = current_square_colour ? pressure_black[target] + new_pressure: pressure_white[target] + new_pressure;
				int support = current_square_colour ? support_white[target] : support_black[target];
				bool high_pressure = pressure >= support;
				
				if (high_pressure){
					decrement += decrement_lookup[pinned_to_piece_type];
					pressure_increase += pressure_increase_lookup[pinned_to_piece_type];
				}
				//std::cout << "Pinned: " << (int)current_square << " Pinner " << int(sliding_attacker) << " target: " << (int)target << " Decrement: " << decrement << " P_increase: " << pressure_increase << std::endl;
			}
		}

		if (current_square_colour){
			pressure_black[current_square] += pressure_increase;
		} else {
			pressure_white[current_square] += pressure_increase;
		}	

		uint64_t current_piece_attacks = attacks_mask(current_square_colour,occupied,current_square,current_square_piece) & occupied;
		uint8_t attacked_square = 0;
		while (current_piece_attacks) {
			attacked_square = __builtin_ctzll(current_piece_attacks);
			current_piece_attacks &= current_piece_attacks - 1;

			uint8_t attacked_piece_type = pieceTypeLookUp[attacked_square];

			if (current_square_colour){
				if ((occupied_white & (BB_SQUARES[attacked_square])) != 0){
					support_white[attacked_square] += std::min(support_weights[current_square_piece][attacked_piece_type], decrement);				
				} else {
					pressure_white[attacked_square] += std::min(pressure_weights[current_square_piece][attacked_piece_type], decrement);
				}
			} else {
				if ((occupied_white & (BB_SQUARES[attacked_square])) != 0){
					pressure_black[attacked_square] += std::min(pressure_weights[current_square_piece][attacked_piece_type], decrement);	
				} else {
					support_black[attacked_square] += std::min(support_weights[current_square_piece][attacked_piece_type], decrement);
				}
			}	
		}
	}
}

// Mate-drive (king->enemy-king proximity + drive-to-edge) is scaled by the winner's MATERIAL margin:
// no drive below ~1.5 pawns, full by ~a rook. Material is the reliable winning signal (the positional
// total is inflated and must not gate its own amplifier); a defending queen zeroes it.
constexpr int MATE_DRIVE_LO = 1500;
constexpr int MATE_DRIVE_HI = 5000;

inline int advanced_endgame_eval(int total, bool turn){
	//std::cout << total <<std::endl;
	// Acquire the square positions of each king
	uint8_t whiteKingSquare = __builtin_ctzll(occupied_white&kings);
	uint8_t blackKingSquare = __builtin_ctzll(occupied_black&kings);
	
	// Acquire the separation between the kings
	uint8_t kingSeparation = square_distance(whiteKingSquare,blackKingSquare);

	// Pre-drive total, for the optional material-scaled mate-drive (ENABLE_MATE_DRIVE_SCALE) below.
	int mate_drive_before = total;
	
	// Check if the black side has a 2000 point advantage or greater
	if (total > 2000){
		
		// Increment black side for having the black king closer to the white king
		total += (7-kingSeparation)*200;
					
		// Get the x and y coordinates for the white king
		uint8_t y = whiteKingSquare >> 3;
		uint8_t x = whiteKingSquare & 7;
		
		/*
			In this code section, lower white's score if it's king is closer to the board's edge
		*/
		if (x >= 4){
			if (y >= 4){
				total += (x + y) * 45;
			}else{
				total += (x + (7 - y)) * 45;
			}					
		} else{
			if (y >= 4){
				total += ((7 - x) + y) * 45;
			}else{
				total += ((7 - x) + (7 - y)) * 45;
			}					
		}
		
	// Check if the white side has a 2000 point advantage or greater	
	}else if (total < -2000){
		
		// Increment white side for having the white king closer to the black king
		total -= (7-kingSeparation)*200;
		
		// Get the x and y coordinates for the black king
		uint8_t y = blackKingSquare >> 3;
		uint8_t x = blackKingSquare & 7;
		
		/*
			In this code section, lower black's score if it's king is closer to the board's edge
		*/
		if (x >= 4){
			if (y >= 4){
				total -= (x + y) * 45;
			}else{
				total -= (x + (7 - y)) * 45;
			}					
		} else{
			if (y >= 4){
				total -= ((7 - x) + y) * 45;
			}else{
				total -= ((7 - x) + (7 - y)) * 45;
			}					
		}
	}

	// Optionally scale the mate-drive just added by the winner's MATERIAL margin (env-gated, default
	// off = byte-identical). A defending queen zeroes it; keeps real KX-K mates at full strength, stops
	// amplifying sharp/compensated leads where driving the king to the edge is not a mating plan.
	if (Config::ENABLE_MATE_DRIVE_SCALE && (mate_drive_before > 2000 || mate_drive_before < -2000)){
		int whiteMat = __builtin_popcountll(occupied_white & pawns)   * values[PAWN]
		             + __builtin_popcountll(occupied_white & knights) * values[KNIGHT]
		             + __builtin_popcountll(occupied_white & bishops) * values[BISHOP]
		             + __builtin_popcountll(occupied_white & rooks)   * values[ROOK]
		             + __builtin_popcountll(occupied_white & queens)  * values[QUEEN];
		int blackMat = __builtin_popcountll(occupied_black & pawns)   * values[PAWN]
		             + __builtin_popcountll(occupied_black & knights) * values[KNIGHT]
		             + __builtin_popcountll(occupied_black & bishops) * values[BISHOP]
		             + __builtin_popcountll(occupied_black & rooks)   * values[ROOK]
		             + __builtin_popcountll(occupied_black & queens)  * values[QUEEN];
		bool black_winning = (mate_drive_before > 2000);
		int margin = black_winning ? (blackMat - whiteMat) : (whiteMat - blackMat);
		bool defender_has_queen = black_winning ? bool(occupied_white & queens) : bool(occupied_black & queens);
		double drive_scale = defender_has_queen ? 0.0 :
			std::max(0.0, std::min(1.0, double(margin - MATE_DRIVE_LO) / double(MATE_DRIVE_HI - MATE_DRIVE_LO)));
		total = mate_drive_before + (int)((total - mate_drive_before) * drive_scale);
	}
	//std::cout<< "Inner " << total << std::endl;
	if (g_capture_eval_breakdown) g_ae_matedrive = total - mate_drive_before;
	int ae_passer_start = total;
	// Create bitmasks for the first and second half of the board
	uint64_t firstHalf = BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4;
	uint64_t secondHalf = BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8;
	
	// Define variables for separation variables for each king and each coloured pawns and passed pawn bonuses
	int blackKing_pawnSeparation = 0;
	int whiteKing_pawnSeparation = 0;
	int kingDist = 0;
	int pawnDist = 0;

	bool kingCanCatch;
	
	int ppIncrement = 0;
	int blockModifier = 0;
	int passedBonus = 0;
	
	// Loop through the mask containing black pawns in the first half
	uint64_t dummy;

	uint8_t r = 0;
	uint64_t bb = firstHalf & occupied_black & pawns;
	//std::cout << total <<std::endl;
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);	

		uint8_t file = r & 7;
		uint8_t rank = r >> 3;

		// Black promotes on rank 0; square index = file (since rank * 8 + file = 0 * 8 + file)
		uint8_t promotionSquare = file;

		// Find the distance between each king and the black pawn
		blackKing_pawnSeparation = square_distance(r,blackKingSquare);
		whiteKing_pawnSeparation = square_distance(r,whiteKingSquare);
		
		ppIncrement = getPPIncrement(false, (occupied_white & pawns), 100, file, rank, occupied_white, occupied_black,dummy,dummy);
		
		kingDist = square_distance(whiteKingSquare, promotionSquare);
		pawnDist = rank;

		kingCanCatch = (turn) ? (kingDist <= pawnDist + 1) : (kingDist <= pawnDist);

		// Defensive penalty for not being able to catch enemy pawn
		blockModifier = 0;
		if (!kingCanCatch) {
			blockModifier = ppIncrement >> 1; // can't stop it, big problem
		} else {
			int diff = (turn) ? (pawnDist + 1 - kingDist) : (pawnDist - kingDist);
			blockModifier = -diff * (ppIncrement >> 2); // the closer we are, the better
		}

		passedBonus = ((7 - rank) * (ppIncrement + blockModifier)) >> 4;

		total += (7 - blackKing_pawnSeparation) * passedBonus;
		//std::cout << total << " | " << ppIncrement << " | " << passedBonus << " | "<< blockModifier << " | " << int(r) <<std::endl;
		total += whiteKing_pawnSeparation * passedBonus;
		//std::cout << total <<std::endl;	
		bb &= bb - 1;
	}
	//std::cout << total <<std::endl;	
	// Loop through the mask containing white pawns in the second half
	r = 0;
	bb = secondHalf & occupied_white & pawns;
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);	
		
		uint8_t file = r & 7;
		uint8_t rank = r >> 3;

		// White promotes on rank 7; square index = 56 + file = 7 * 8 + file
		uint8_t promotionSquare = 56 + file;
		
		// Find the distance between each king and the white pawns
		blackKing_pawnSeparation = square_distance(r,blackKingSquare);
		whiteKing_pawnSeparation = square_distance(r,whiteKingSquare);
		
		ppIncrement = getPPIncrement(true, (occupied_black & pawns), 100, file, rank, occupied_black, occupied_white,dummy,dummy);
		
		kingDist = square_distance(blackKingSquare, promotionSquare);

		pawnDist = 7 - rank;
		kingCanCatch = (!turn) ? (kingDist <= pawnDist + 1) : (kingDist <= pawnDist);

		// Defensive penalty for not being able to catch enemy pawn
		blockModifier = 0;
		if (!kingCanCatch) {
			blockModifier = ppIncrement >> 1; // can't stop it, big problem
		} else {
			int diff = (turn) ? (pawnDist + 1 - kingDist) : (pawnDist - kingDist);
			blockModifier = -diff * (ppIncrement >> 2); // the closer we are, the better
		}
		
		passedBonus = (rank * (ppIncrement + blockModifier)) >> 4;

		total -= blackKing_pawnSeparation * passedBonus;
		//std::cout << total << " | " << ppIncrement << " | " << passedBonus << " | "<< blockModifier << " | " << blackKing_pawnSeparation << " | " << int(r) <<std::endl;
		total -= (7 - whiteKing_pawnSeparation) * passedBonus;
		//std::cout << total <<std::endl;
		bb &= bb - 1;
	}
	//std::cout <<"Inner2 "<< total <<std::endl;
	if (g_capture_eval_breakdown) g_ae_passer = total - ae_passer_start;
	return total;

}



inline int get_latent_threat_score(uint8_t white_king_square, uint8_t black_king_square){
	
	uint64_t white_king_zone = white_king_zones[white_king_square & 7];
	uint64_t black_king_zone = black_king_zones[black_king_square & 7];

	int black_increment = 0;
	int white_increment = 0;
	
	int num_attackers_in_white_zone = 0;
	int num_attackers_in_black_zone = 0;

	int num_defenders_in_white_zone = 0;
	int num_defenders_in_black_zone = 0;

	int num_attacked_squares_in_white_zone = 0;
	int num_attacked_squares_in_black_zone = 0;

	int num_defended_squares_in_white_zone = 0;
	int num_defended_squares_in_black_zone = 0;

	int white_zone_attack_increment = 50;
	int white_zone_presence_increment = 80;

	int black_zone_attack_increment = 50;
	int black_zone_presence_increment = 80;

	uint64_t bb = white_king_zone;
	uint8_t r = 0;
	while (bb) {		
		r = __builtin_ctzll(bb);  

		uint64_t mask = BB_SQUARES[r];

		if(occupied_white & mask){
			num_defenders_in_white_zone++;

			if (pawns & mask) {
				white_zone_presence_increment -= 2;
			} else if (knights & mask){
				white_zone_presence_increment -= 5;
			} else if (bishops & mask){
				white_zone_presence_increment -= 3;
			} else if (rooks & mask){
				white_zone_presence_increment -= 1;
			} else if (queens & mask){
				white_zone_presence_increment -= 2;
			}

		}else if(occupied_black & BB_SQUARES[r]){
			num_attackers_in_white_zone++;

			if (pawns & mask) {
				white_zone_presence_increment += 10;
			} else if (knights & mask){
				white_zone_presence_increment += 7;
			} else if (bishops & mask){
				white_zone_presence_increment += 7;
			} else if (rooks & mask){
				white_zone_presence_increment += 3;
			} else if (queens & mask){
				white_zone_presence_increment += 7;
			}
		}

		uint64_t attack_mask = attack_bitmasks[r];
		while (attack_mask) {		
			uint8_t attack_square = __builtin_ctzll(attack_mask);  
			attack_mask &= attack_mask - 1;

			uint64_t attack_square_mask = BB_SQUARES[attack_square];
			if (attack_square_mask & occupied_white){
				num_defended_squares_in_white_zone++;

				if (pawns & attack_square_mask) {
					white_zone_attack_increment -= 2;
				} else if (knights & attack_square_mask){
					white_zone_attack_increment -= 10;
				} else if (bishops & attack_square_mask){
					white_zone_attack_increment -= 7;
				} else if (rooks & attack_square_mask){
					white_zone_attack_increment -= 1;
				} else if (queens & attack_square_mask){
					white_zone_attack_increment -= 2;
				}
			} else{
				num_attacked_squares_in_white_zone++;

				if (pawns & attack_square_mask) {
					white_zone_attack_increment += 10;
				} else if (knights & attack_square_mask){
					white_zone_attack_increment += 7;
				} else if (bishops & attack_square_mask){
					white_zone_attack_increment += 7;
				} else if (rooks & attack_square_mask){
					white_zone_attack_increment += 5;
				} else if (queens & attack_square_mask){
					white_zone_attack_increment += 5;
				}
			}
		}	
		
		bb &= bb - 1;  
	}

	bb = black_king_zone;
	r = 0;
	while (bb) {		
		r = __builtin_ctzll(bb);  

		uint64_t mask = BB_SQUARES[r];

		if(occupied_white & mask){
			num_attackers_in_black_zone++;

			if (pawns & mask) {
				black_zone_presence_increment += 10;
			} else if (knights & mask){
				black_zone_presence_increment += 7;
			} else if (bishops & mask){
				black_zone_presence_increment += 7;
			} else if (rooks & mask){
				black_zone_presence_increment += 3;
			} else if (queens & mask){
				black_zone_presence_increment += 7;
			}

		}else if(occupied_black & BB_SQUARES[r]){
			num_defenders_in_black_zone++;

			if (pawns & mask) {
				black_zone_presence_increment -= 2;
			} else if (knights & mask){
				black_zone_presence_increment -= 5;
			} else if (bishops & mask){
				black_zone_presence_increment -= 3;
			} else if (rooks & mask){
				black_zone_presence_increment -= 1;
			} else if (queens & mask){
				black_zone_presence_increment -= 2;
			}
		}

		uint64_t attack_mask = attack_bitmasks[r];
		while (attack_mask) {		
			uint8_t attack_square = __builtin_ctzll(attack_mask);  
			attack_mask &= attack_mask - 1;

			uint64_t attack_square_mask = BB_SQUARES[attack_square];
			if (attack_square_mask & occupied_white){
				num_attacked_squares_in_black_zone ++;

				if (pawns & attack_square_mask) {
					black_zone_attack_increment += 10;
				} else if (knights & attack_square_mask){
					black_zone_attack_increment += 7;
				} else if (bishops & attack_square_mask){
					black_zone_attack_increment += 7;
				} else if (rooks & attack_square_mask){
					black_zone_attack_increment += 5;
				} else if (queens & attack_square_mask){
					black_zone_attack_increment += 5;
				}
			} else{
				num_defended_squares_in_black_zone++;

				if (pawns & attack_square_mask) {
					black_zone_attack_increment -= 2;
				} else if (knights & attack_square_mask){
					black_zone_attack_increment -= 10;
				} else if (bishops & attack_square_mask){
					black_zone_attack_increment -= 7;
				} else if (rooks & attack_square_mask){
					black_zone_attack_increment -= 1;
				} else if (queens & attack_square_mask){
					black_zone_attack_increment -= 2;
				}
			}
		}	
		
		bb &= bb - 1;  
	}

	int white_defended_threshold = (num_defended_squares_in_white_zone >> 3) * 5;
	int black_defended_threshold = (num_defended_squares_in_black_zone >> 3) * 5;

	if (num_attacked_squares_in_white_zone > white_defended_threshold){
		black_increment += (num_attacked_squares_in_white_zone - white_defended_threshold) * white_zone_attack_increment;
	}
	//std::cout << "black_increment: " << black_increment << std::endl;
	black_increment += std::max(0, (num_attackers_in_white_zone - num_defenders_in_white_zone + 4) * white_zone_presence_increment);

	if (num_attacked_squares_in_black_zone > black_defended_threshold){
		white_increment += (num_attacked_squares_in_black_zone - black_defended_threshold) * black_zone_attack_increment;
	}

	white_increment += std::max(0, (num_attackers_in_black_zone - num_defenders_in_black_zone + 4) * black_zone_presence_increment); 

	//black_increment = (num_attacked_squares_in_white_zone - num_defended_squares_in_white_zone + 7) * 100 + (num_attackers_in_white_zone - num_defenders_in_white_zone + 6) * 100;
	//white_increment = (num_attacked_squares_in_black_zone - num_defended_squares_in_black_zone + 7) * 100 + (num_attackers_in_black_zone - num_defenders_in_black_zone + 6) * 100;

	/* std::cout << "black_increment: " << black_increment << std::endl;
	std::cout << "white_increment: " << white_increment << std::endl; */
	if(num_defenders_in_black_zone <= (num_defended_squares_in_black_zone - 4) / 4)
		white_increment *= 2;

	if(num_defenders_in_white_zone <= (num_defended_squares_in_white_zone - 4) / 4)
		black_increment *= 2;

	if(num_attackers_in_black_zone <= num_attacked_squares_in_black_zone / 4)
		white_increment /= 3;

	if(num_attackers_in_white_zone <= num_attacked_squares_in_white_zone / 4)
		black_increment /= 3;


	/* // For black_increment
std::cout << "num_attacked_squares_in_white_zone: " << num_attacked_squares_in_white_zone << std::endl;
std::cout << "num_defended_squares_in_white_zone: " << num_defended_squares_in_white_zone << std::endl;
std::cout << "num_attackers_in_white_zone: " << num_attackers_in_white_zone << std::endl;
std::cout << "num_defenders_in_white_zone: " << num_defenders_in_white_zone << std::endl;
std::cout << "black_increment: " << black_increment << std::endl;

// For white_increment
std::cout << "num_attacked_squares_in_black_zone: " << num_attacked_squares_in_black_zone << std::endl;
std::cout << "num_defended_squares_in_black_zone: " << num_defended_squares_in_black_zone << std::endl;
std::cout << "num_attackers_in_black_zone: " << num_attackers_in_black_zone << std::endl;
std::cout << "num_defenders_in_black_zone: " << num_defenders_in_black_zone << std::endl;
std::cout << "white_increment: " << white_increment << std::endl; */

	// Snapshot both increments first so the two king-file bonuses are order-independent
	// (otherwise the first one to fire changes the other's condition, breaking color symmetry).
	int white_increment_pre = white_increment;
	int black_increment_pre = black_increment;

	if((black_king_square & 7) == 3 || (black_king_square & 7) == 4){
		if (white_increment_pre <= 0){
			black_increment += 75;
		}
	}

	if((white_king_square & 7) == 3 || (white_king_square & 7) == 4){
		if (black_increment_pre <= 0){
			white_increment += 75;
		}
	}


	return black_increment - white_increment;
}

inline int boost_pieces_for_supporting_passed_pawns(uint64_t white_passed_pawns, uint64_t black_passed_pawns, const std::array<int, 64>& pawn_rank_bonuses, bool isEndGame){

	int black_adjustment = 0;
	int white_adjustment = 0;
	int adjustment = 0;
	if (white_passed_pawns){

		uint64_t bb = white_passed_pawns;
		int r = 0;

		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			black_adjustment = 0;
			white_adjustment = 0;

			int y = r >> 3;		
			int scale = 2;
			if(y >= 4 && isEndGame){
				scale = 1;
			}				

			if (y > 2){
				for (int i = r + 8; i <= 63; i += 8){
					uint64_t cur_mask = BB_SQUARES[i];

					if (cur_mask & occupied_white){
						white_adjustment += (y * 75) / scale;
						//square_values[i] -= (y * 100);
						
					}else if(cur_mask & occupied_black){
						black_adjustment += (y * 100) / scale;
						//square_values[i] += (y * 125);
						
					} else{

						// Acquire the masks of the pieces on the same rank, file and diagonal as the given square
						uint64_t rank_pieces = BB_RANK_MASKS[i] & occupied;
						uint64_t file_pieces = BB_FILE_MASKS[i] & occupied;
						uint64_t diag_pieces = BB_DIAG_MASKS[i] & occupied;

						// Acquire all attack masks for each piece type
						uint64_t attackers = (
							(BB_KING_ATTACKS[i] & kings) |
							(BB_KNIGHT_ATTACKS[i] & knights) |
							(BB_RANK_ATTACKS[i][rank_pieces] & (queens | rooks)) |
							(BB_FILE_ATTACKS[i][file_pieces] & (queens | rooks)) |
							(BB_DIAG_ATTACKS[i][diag_pieces] & (queens | bishops))
						);

						if (attackers){
							while (attackers) {			
								uint8_t attacker = __builtin_ctzll(attackers); 
								if (BB_SQUARES[attacker] & occupied_white){
									white_adjustment -= (y * 60) / scale;
									//std::cout << "WHITE: " << (int)attacker << std::endl;
									//square_values[attacker] += (y * 75);
								}else{
									black_adjustment += (y * 50) / scale;
									//std::cout << "BLACK: " << (int)attacker << std::endl;
									//square_values[attacker] += (y * 60);
								}
								attackers &= attackers - 1;  
							}
						}
					}					
				}
			}
			adjustment += white_adjustment + std::min(-pawn_rank_bonuses[r],black_adjustment); 
			//std::cout << "ADJUSTMENT for white pp: " << white_adjustment + std::min(-pawn_rank_bonuses[r],black_adjustment) << " | " << (int)r << " | " << (int)y << std::endl;
			bb &= bb - 1;  
		}
	}
	//std::cout << "ADJUSTMENT for white pp: " << adjustment << std::endl;
	if (black_passed_pawns){
		uint64_t bb = black_passed_pawns;
		int r = 0;

		while (bb) {
			
			r = __builtin_ctzll(bb);  

			black_adjustment = 0;
			white_adjustment = 0;
			
			int y = r >> 3;
			int scale = 2;
			if(y <= 3 && isEndGame){
				scale = 1;
			}			

			if (y < 5){
				for (int i = r - 8; i >= 0; i -= 8){
					uint64_t cur_mask = BB_SQUARES[i];

					if (cur_mask & occupied_white){
						white_adjustment -= ((7 - y) * 100) / scale;
						//square_values[i] += ((7 - y) * 125);					
					}else if(cur_mask & occupied_black){
						black_adjustment -= ((7 - y) * 75) / scale;
						//square_values[i] -= ((7 - y) * 100);						
					} else{

						// Acquire the masks of the pieces on the same rank, file and diagonal as the given square
						uint64_t rank_pieces = BB_RANK_MASKS[i] & occupied;
						uint64_t file_pieces = BB_FILE_MASKS[i] & occupied;
						uint64_t diag_pieces = BB_DIAG_MASKS[i] & occupied;

						// Acquire all attack masks for each piece type
						uint64_t attackers = (
							(BB_KING_ATTACKS[i] & kings) |
							(BB_KNIGHT_ATTACKS[i] & knights) |
							(BB_RANK_ATTACKS[i][rank_pieces] & (queens | rooks)) |
							(BB_FILE_ATTACKS[i][file_pieces] & (queens | rooks)) |
							(BB_DIAG_ATTACKS[i][diag_pieces] & (queens | bishops))
						);

						if (attackers){
							while (attackers) {			
								uint8_t attacker = __builtin_ctzll(attackers); 
								if (BB_SQUARES[attacker] & occupied_white){
									white_adjustment -= ((7 - y) * 50) / scale;
									//square_values[attacker] += ((7 - y) * 60);
								}else{
									black_adjustment += ((7 - y) * 60) / scale;
									//square_values[attacker] += ((7 - y) * 75);
								}
								attackers &= attackers - 1;  
							}
						}
					}
				}
			}
			adjustment += black_adjustment + std::max(-pawn_rank_bonuses[r],white_adjustment); 
			//std::cout << "ADJUSTMENT for black pp: " << black_adjustment + std::max(-pawn_rank_bonuses[r],white_adjustment) << " | " << (int)r << " | " << black_adjustment << " | " << -pawn_rank_bonuses[r] << " | " << white_adjustment << std::endl;
			bb &= bb - 1;  
		}
	}

	return adjustment;
}



inline int chebyshev_distance(int from_sq, int to_sq) {
    int fx = from_sq % 8, fy = from_sq / 8;
    int tx = to_sq % 8, ty = to_sq / 8;
    return std::max(abs(fx - tx), abs(fy - ty));
}

inline bool is_practically_drawn(int pieceNum) {
    uint64_t no_king_mask = occupied & ~kings;
	//std::cout << "AAA: " << pieceNum<< std::endl;
    
	// King vs. King, King+Bishop vs. King, King+Knight vs. King
	if (__builtin_popcountll(occupied_white) == __builtin_popcountll(occupied_black)) {
		if (no_king_mask == bishops || no_king_mask == knights)
			return true;
	}

	// King + one minor piece vs King
	if (__builtin_popcountll(bishops) == 1 && no_king_mask == bishops)
		return true;

	if (__builtin_popcountll(knights) == 1 && no_king_mask == knights)
		return true;

	/* bool white_has_minor = (__builtin_popcountll(knights & occupied_white) == 1 ||
                        __builtin_popcountll(bishops & occupied_white) == 1);
	bool black_has_minor = (__builtin_popcountll(knights & occupied_black) == 1 ||
							__builtin_popcountll(bishops & occupied_black) == 1);

	bool white_has_pawn = (__builtin_popcountll(pawns & occupied_white) == 1);
	bool black_has_pawn = (__builtin_popcountll(pawns & occupied_black) == 1);

	// No other material on the board
	bool only_minor_and_pawn_exist = __builtin_popcountll(occupied & ~(kings | pawns | knights | bishops)) == 0;

	// Now apply draw condition
	if (only_minor_and_pawn_exist) {
		if ((white_has_minor && black_has_pawn) ||
			(black_has_minor && white_has_pawn)) {
			return true; // minor piece vs lone pawn is practically drawn
		}
	} */


	if (no_king_mask == 0)
		return true;
    
	// Bishop + Rook Pawn special case
	//std::cout << "AAA" << std::endl;
	if ((no_king_mask == (bishops | pawns)) &&
		(__builtin_popcountll(bishops) == 1) &&
		(__builtin_popcountll(pawns) == 1)) {
		//std::cout << "BBB" << std::endl;
		//bool bishop_is_white = (bishops & occupied_white);
		bool pawn_is_white = (pawns & occupied_white);

		int bishop_square = __builtin_ctzll(bishops);
		bool bishop_square_color = is_white_square(bishop_square);

		int pawn_square = __builtin_ctzll(pawns);
		int promotion_square = (pawns & BB_FILE_A) ?
								(pawn_is_white ? 56 : 0) :
								(pawn_is_white ? 63 : 7);

		if (!((pawns & BB_FILE_A) || (pawns & BB_FILE_H)))
			return false; // Not a rook pawn
		//std::cout << "CCC" << std::endl;
		bool promotion_color = is_white_square(promotion_square);
		if (bishop_square_color != promotion_color) {
			// Now check king proximity
			uint64_t white_king_bb = kings & occupied_white;
			uint64_t black_king_bb = kings & occupied_black;

			int white_king_sq = __builtin_ctzll(white_king_bb);
			int black_king_sq = __builtin_ctzll(black_king_bb);

			int attacker_king_sq = pawn_is_white ? white_king_sq : black_king_sq;
			int defender_king_sq = pawn_is_white ? black_king_sq : white_king_sq;

			int defender_dist = chebyshev_distance(defender_king_sq, promotion_square);
			int attacker_dist = chebyshev_distance(attacker_king_sq, promotion_square);
			int pawn_dist = chebyshev_distance(pawn_square, promotion_square);
			//std::cout << "DDD" << std::endl;
			if (defender_dist <= std::min(pawn_dist, attacker_dist))
				return true; // Drawn by opposition
			//std::cout << "EEE" << std::endl;
		}
	}

	// Rook and Bishop vs Rook — known theoretical draw in most cases
	if (__builtin_popcountll(no_king_mask) == 3 &&
		__builtin_popcountll(rooks) == 2 &&
		__builtin_popcountll(bishops) == 1)
	{
		int white_rooks = __builtin_popcountll(rooks & occupied_white);
		int black_rooks = __builtin_popcountll(rooks & occupied_black);
		int white_bishops = __builtin_popcountll(bishops & occupied_white);
		int black_bishops = __builtin_popcountll(bishops & occupied_black);

		if ((white_rooks == 1 && white_bishops == 1 && black_rooks == 1 && black_bishops == 0) ||
			(black_rooks == 1 && black_bishops == 1 && white_rooks == 1 && white_bishops == 0))
		{
			return true;
		}
	}

	// Rook and Knight vs Rook — known theoretical draw in most cases
	if (__builtin_popcountll(no_king_mask) == 3 &&
		__builtin_popcountll(rooks) == 2 &&
		__builtin_popcountll(knights) == 1)
	{
		int white_rooks = __builtin_popcountll(rooks & occupied_white);
		int black_rooks = __builtin_popcountll(rooks & occupied_black);
		int white_knights = __builtin_popcountll(knights & occupied_white);
		int black_knights = __builtin_popcountll(knights & occupied_black);

		if ((white_rooks == 1 && white_knights == 1 && black_rooks == 1 && black_knights == 0) ||
			(black_rooks == 1 && black_knights == 1 && white_rooks == 1 && white_knights == 0))
		{
			return true;
		}
	}

	// Bare rook vs bare minor (KRKN / KRKB) — a theoretical draw; the lone minor + king holds
	if (__builtin_popcountll(no_king_mask) == 2 &&
		__builtin_popcountll(rooks) == 1 &&
		(__builtin_popcountll(knights) == 1 || __builtin_popcountll(bishops) == 1))
	{
		uint64_t minor = knights | bishops;
		bool rook_is_white  = (rooks & occupied_white) != 0;
		bool minor_is_white = (minor & occupied_white) != 0;
		// Opposite sides => one side has the lone rook, the other the lone minor (drawn). Same side
		// would be R+minor vs lone king (a win), which this guard correctly leaves un-flagged.
		if (rook_is_white != minor_is_white)
			return true;
	}

	// Bishop vs Pawn (where the pawn cannot promote)
	if (__builtin_popcountll(no_king_mask) == 2 &&
		__builtin_popcountll(bishops) == 1 &&
		__builtin_popcountll(pawns) == 1)
	{
		bool bishop_is_white = (bishops & occupied_white);
		bool pawn_is_white = (pawns & occupied_white);

		// Must be opposite colors for bishop vs pawn
		if (bishop_is_white != pawn_is_white)
		{
			int bishop_square = __builtin_ctzll(bishops);
			int pawn_square = __builtin_ctzll(pawns);
			bool bishop_square_color = is_white_square(bishop_square);

			// Check if pawn is on rook file (A or H)
			if ((pawns & BB_FILE_A) || (pawns & BB_FILE_H)) {
				int promotion_square = (pawns & BB_FILE_A) ?
										(pawn_is_white ? 56 : 0) :
										(pawn_is_white ? 63 : 7);

				bool promotion_square_color = is_white_square(promotion_square);

				// If bishop is on opposite color as promotion square
				if (bishop_square_color != promotion_square_color) {

					// Get king squares
					uint64_t white_king_bb = kings & occupied_white;
					uint64_t black_king_bb = kings & occupied_black;
					int white_king_sq = __builtin_ctzll(white_king_bb);
					int black_king_sq = __builtin_ctzll(black_king_bb);

					int attacker_king_sq = pawn_is_white ? white_king_sq : black_king_sq;
					int defender_king_sq = pawn_is_white ? black_king_sq : white_king_sq;

					int defender_dist = chebyshev_distance(defender_king_sq, promotion_square);
					int attacker_dist = chebyshev_distance(attacker_king_sq, promotion_square);
					int pawn_dist = chebyshev_distance(pawn_square, promotion_square);

					// Defender has opposition or equal distance to block promotion
					if (defender_dist <= std::min(pawn_dist, attacker_dist)) {
						return true; // Practically drawn
					}
				}
			}
		}
	}

	// Knight vs Pawn (where the pawn cannot promote)
	if (__builtin_popcountll(no_king_mask) == 2 &&
		__builtin_popcountll(knights) == 1 &&
		__builtin_popcountll(pawns) == 1)
	{
		bool knight_is_white = (knights & occupied_white);
		bool pawn_is_white = (pawns & occupied_white);

		// Must be opposite sides
		if (knight_is_white != pawn_is_white)
		{
			int knight_square = __builtin_ctzll(knights);
			int pawn_square = __builtin_ctzll(pawns);

			// Only consider rook pawns
			if ((pawns & BB_FILE_A) || (pawns & BB_FILE_H)) {
				int promotion_square = (pawns & BB_FILE_A) ?
										(pawn_is_white ? 56 : 0) :
										(pawn_is_white ? 63 : 7);

				// Get kings
				uint64_t white_king_bb = kings & occupied_white;
				uint64_t black_king_bb = kings & occupied_black;
				int white_king_sq = __builtin_ctzll(white_king_bb);
				int black_king_sq = __builtin_ctzll(black_king_bb);

				int attacker_king_sq = pawn_is_white ? white_king_sq : black_king_sq;
				int defender_king_sq = pawn_is_white ? black_king_sq : white_king_sq;

				int defender_dist = chebyshev_distance(defender_king_sq, promotion_square);
				int attacker_dist = chebyshev_distance(attacker_king_sq, promotion_square);
				int pawn_dist = chebyshev_distance(pawn_square, promotion_square);

				// If defender controls promotion square and attacker too far
				if (defender_dist <= std::min(pawn_dist, attacker_dist) &&
					chebyshev_distance(knight_square, promotion_square) > 2) {
					return true;
				}
			}
		}
	}


    
    return false;
}



inline void print_bitboard(uint64_t bb, const std::string& label) {
    std::cout << label << ":\n";
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            std::cout << ((bb >> sq) & 1ULL ? "1 " : ". ");
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


// Continuous endgame convertibility scale in [DRAW_SCALE_FLOOR, 1] — the graded companion to
// is_practically_drawn (env-gated; see ENABLE_ENDGAME_SCALE). Damps a material/placement lead toward
// draw when the remaining force can't realistically convert it (a bare minor, opposite-coloured
// bishops), and pulls back toward 1 when the winning side has an advanced passed pawn (the primary
// conversion mechanism). It is multiplied into `total`, so a balanced (~0) eval is untouched, and any
// edge above a minor (KQ/KR/KBN vs K) keeps s=1 — genuine wins are never damped.
constexpr double DRAW_SCALE_FLOOR = 0.25;

inline double endgame_convertibility_scale(uint64_t white_passed_pawns, uint64_t black_passed_pawns){
	int wNP = __builtin_popcountll(occupied_white & knights) * values[KNIGHT]
	        + __builtin_popcountll(occupied_white & bishops) * values[BISHOP]
	        + __builtin_popcountll(occupied_white & rooks)   * values[ROOK]
	        + __builtin_popcountll(occupied_white & queens)  * values[QUEEN];
	int bNP = __builtin_popcountll(occupied_black & knights) * values[KNIGHT]
	        + __builtin_popcountll(occupied_black & bishops) * values[BISHOP]
	        + __builtin_popcountll(occupied_black & rooks)   * values[ROOK]
	        + __builtin_popcountll(occupied_black & queens)  * values[QUEEN];
	int wP = __builtin_popcountll(occupied_white & pawns);
	int bP = __builtin_popcountll(occupied_black & pawns);

	// The materially-stronger side (whose lead we're judging); ties broken by pawn count.
	bool white_stronger = (wNP != bNP) ? (wNP > bNP) : (wP >= bP);
	int edge = wNP > bNP ? (wNP - bNP) : (bNP - wNP);
	int winnerPawns = white_stronger ? wP : bP;
	uint64_t winnerPassers = white_stronger ? white_passed_pawns : black_passed_pawns;
	int totalPawns = wP + bP;

	double s = 1.0;

	// A lead of at most one minor (but a real lead, not equal material), with little/no pawns: can't
	// mate, can't promote -> hard damp. Equal material (edge == 0) is left to the OCB rule below.
	if (edge > 0 && edge <= values[BISHOP]){
		if (winnerPawns == 0)
			s = std::min(s, DRAW_SCALE_FLOOR);
		else if (winnerPawns <= 2)
			s = std::min(s, 0.45 + 0.18 * winnerPawns);
	}

	// Opposite-coloured bishops (one each, opposite colours, no other pieces): drawish, climbs with pawns.
	if (wNP == values[BISHOP] && bNP == values[BISHOP] && __builtin_popcountll(bishops) == 2 &&
	    is_white_square(__builtin_ctzll(occupied_white & bishops)) !=
	    is_white_square(__builtin_ctzll(occupied_black & bishops)))
		s = std::min(s, 0.35 + 0.09 * totalPawns);

	// A winning passed pawn is the main conversion lever -> pull back toward 1 by its advancement.
	if (winnerPassers){
		int best = 0;
		uint64_t bb = winnerPassers;
		while (bb){
			int rank = __builtin_ctzll(bb) >> 3;
			int adv = white_stronger ? rank : (7 - rank);   // 0..6, 6 = one square from promotion
			if (adv > best) best = adv;
			bb &= bb - 1;
		}
		if (best >= 6) return 1.0;
		if (best >= 5) s = std::max(s, 0.85);
		else if (best >= 4) s = std::max(s, 0.65);
	}

	return s < DRAW_SCALE_FLOOR ? DRAW_SCALE_FLOOR : (s > 1.0 ? 1.0 : s);
}

int placement_and_piece_eval(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask){

	/*
		Function to acquire a positional evaluation
		
		Parameters:
		- moveNum: The current move number
		- turn: The current side who's turn it is to move
		- pawnsMask: The mask containing only pawns
		- knightsMask: The mask containing only knights
		- bishopsMask: The mask containing only bishops
		- rooksMask: The mask containing only rooks
		- queensMask: The mask containing only queens
		- kingsMask: The mask containing only kings
		- prevKingsMask: The mask containing the position of the kings at the previous move in-game
		- occupied_whiteMask: The mask containing only white pieces
		- occupied_blackMask: The mask containing only black pieces
		- occupiedMask: The mask containing all pieces
		
		Returns:
		A position evaluation
	*/
	
	// Define the total
	int total = 0;

	// Diagnostic-only term-attribution accumulators. These only ever READ `total` and write to locals/globals;
	// they never feed back into `total`, so the search tree is byte-identical whether capture is on or off.
	int br_run = 0;
	int br_pieces = 0, br_capture = 0, br_passed = 0, br_latent = 0, br_central = 0;
	int br_imbalance_white = 0, br_imbalance_black = 0, br_pairs = 0, br_pv_boost = 0;
	int br_advanced_total = 0;
	int br_ae_input = 0;
	bool br_advanced_fired = false;
	// Per-piece-type contribution to `pieces` (midgame path), for color-mirror localization.
	int br_pt_pawns = 0, br_pt_knights = 0, br_pt_bishops = 0, br_pt_rooks = 0, br_pt_queens = 0, br_pt_kings = 0;

	// Set the masks for each piece and for all white and black pieces globally
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	occupied = occupiedMask;
	
	// Initialize the global offensive and defensive scores
	whiteOffensiveScore = 0;
	blackOffensiveScore = 0;
	whiteDefensiveScore = 0;
	blackDefensiveScore = 0;	
	
	// Initialize the global piece values
	blackPieceVal = 0;
	whitePieceVal = 0;

	// Initialize central score
	central_score = 0;

	// Initialize piece attack arrays
	attack_bitmasks.fill(0ULL);

	uint64_t white_passed_pawns = 0;
	uint64_t black_passed_pawns = 0;

	pressure_white.fill(0);
	support_white.fill(0);
	pressure_black.fill(0);
	support_black.fill(0);

	num_attackers.fill(0);
	num_supporters.fill(0);

	square_values.fill(0);

	horizon_mitigation_flag = false;
	
	// Acquire the number of pieces on the board not including the kings
	int pieceNum = __builtin_popcountll(occupied) - 2;
	
	// Determine if the game is at the endgame phase as well as an advanced endgame phase
	bool isEndGame;
	bool isNearGameEnd;

	bool boost_white_for_piece_value_advantage = false;
	bool boost_black_for_piece_value_advantage = false;

	// Indexed by square (0-63); a zero entry means "no bonus" for that square,
	// matching the old unordered_map's value-initialized miss. Zero-initialized
	// per eval so unwritten squares read 0.
	std::array<int, 64> pawn_rank_bonuses{};

	// Call the function to initialize global piece values
	initializePieceValues(occupied);
	// If the queens are off the board, then it can be considered an endgame at a higher piece value
	/* if (queens == 0){
		isEndGame = pieceNum < 18;
		isNearGameEnd = pieceNum < 12;
	} */
	
	int phase = 0;
	phase += 4 * __builtin_popcountll(queensMask);
	phase += 2 * __builtin_popcountll(rooksMask);
	phase += 1 * __builtin_popcountll(bishopsMask | knightsMask);

	int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to 128

	if (phase_score <= 64) {
    	// Midgame
		isEndGame = false;
	} else if (phase_score <= 96) {
		// Normal endgame
		isEndGame = true;
	} else {
		// Advanced endgame
		//setAttackingLayer(5, true);
		isEndGame = true;
		isNearGameEnd = true;
	}
/* 	print_bitboard(WHITE_LIGHT_BISHOP_ZONE, "WHITE_LIGHT_BISHOP_ZONE");
	print_bitboard(WHITE_DARK_BISHOP_ZONE,  "WHITE_DARK_BISHOP_ZONE");
	print_bitboard(BLACK_LIGHT_BISHOP_ZONE, "BLACK_LIGHT_BISHOP_ZONE");
	print_bitboard(BLACK_DARK_BISHOP_ZONE,  "BLACK_DARK_BISHOP_ZONE"); */

	// If the game is not in endgame phase
	if (!isEndGame){

		// Update the attacking layer based on the position of the king
		{
			PROF_BLOCK(PROF_ATTACK_LAYER);
			setAttackingLayer(5, isEndGame);
		}
		//std::cout << total << std::endl;
		uint64_t bb = pawnsMask;
		uint8_t r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			bb &= bb - 1;  
			
			PROF_BLOCK(PROF_PAWNS);
			int blended_score;
			if (phase_score <= 40) {
				int pawn_rank_bonus = 0;
				int result_mid = evaluate_pawns_midgame(r, white_passed_pawns, black_passed_pawns, pawn_rank_bonus);
				pawn_rank_bonuses[r] = pawn_rank_bonus;
				blended_score = result_mid;
			} else {
				int pawn_rank_bonus_mid = 0;
				int pawn_rank_bonus_end = 0;

				int result_mid = evaluate_pawns_midgame(r, white_passed_pawns, black_passed_pawns, pawn_rank_bonus_mid);
				int result_end = evaluate_pawns_endgame(r, white_passed_pawns, black_passed_pawns, pawn_rank_bonus_end);

				int blend_range = 30; // 70 - 40
				int end_weight = phase_score - 40;
				int mid_weight = blend_range - end_weight;
				blended_score = (mid_weight * result_mid + end_weight * result_end) / blend_range;

				pawn_rank_bonuses[r] = (mid_weight * pawn_rank_bonus_mid + end_weight * pawn_rank_bonus_end) / blend_range;
			}

			total += blended_score;
			br_pt_pawns += blended_score;

			//std::cout << "PAWNS: " << int(r) << " | " << blended_score << std::endl;					
		}

		//total += pawns_simd_initializer(bb);
		//std::cout << total << std::endl;
		
		bb = knightsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function
			PROF_BLOCK(PROF_KNIGHTS);
			int result = evaluate_knights_midgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_knights += result;
			//std::cout << "KNIGHTS: " << int(r) << " | " << result << std::endl;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		
		//std::cout << total << std::endl;
		bb = bishopsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_BISHOPS);
			int result = evaluate_bishops_midgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_bishops += result;
			//std::cout << "BISHOPS: " << int(r) << " | " << result << std::endl;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		//std::cout << total << std::endl;
		bb = rooksMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			/* int result = evaluate_rooks_midgame(r);
			square_values[r] = abs(result);
			total += result;
			std::cout << "ROOKS: " << int(r) << " | " << result << std::endl; */

			PROF_BLOCK(PROF_ROOKS);
			int blended_score;
			if (phase_score <= 40) {
				int result_mid = evaluate_rooks_midgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_mid;
			} else if (phase_score >= 70) {
				int result_end = evaluate_rooks_endgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_end;
			} else {
				int result_mid = evaluate_rooks_midgame(r, white_passed_pawns, black_passed_pawns);
				int result_end = evaluate_rooks_endgame(r, white_passed_pawns, black_passed_pawns);

				int blend_range = 30; // 70 - 40
				int end_weight = phase_score - 40;
				int mid_weight = blend_range - end_weight;
				blended_score = (mid_weight * result_mid + end_weight * result_end) / blend_range;
			}

			{
				PROF_BLOCK(PROF_ROOK_ACTIVITY);
				blended_score += get_latent_rook_activity_score(r);
			}

			if(blended_score < 0){
				blended_score = std::max(blended_score, -5600);
			}else{
				blended_score = std::min(blended_score, 5600);
			}
			//square_values[r] = abs(blended_score);
			total += blended_score;
			br_pt_rooks += blended_score;
			//std::cout << "ROOKS: " << int(r) << " | " << blended_score << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;
		bb = queensMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_QUEENS);
			int result = evaluate_queens_midgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_queens += result;
			//std::cout << "QUEENS: " << int(r) << " | " << result << std::endl;

			/* int blended_score;
			if (phase_score <= 40) {
				int result_mid = evaluate_queens_midgame(r);
				blended_score = result_mid;
			} else if (phase_score >= 70) {
				int result_end = evaluate_queens_endgame(r);
				blended_score = result_end;
			} else {
				int result_mid = evaluate_queens_midgame(r);
				int result_end = evaluate_queens_endgame(r);

				int blend_range = 30; // 70 - 40
				int end_weight = phase_score - 40;
				int mid_weight = blend_range - end_weight;
				blended_score = (mid_weight * result_mid + end_weight * result_end) / blend_range;
			}

			//square_values[r] = abs(blended_score);
			total += blended_score;
			std::cout << "QUEENS: " << int(r) << " | " << blended_score << std::endl; */
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;
		bb = kingsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			
			PROF_BLOCK(PROF_KINGS);
			int blended_score;
			if (phase_score <= 40) {
				int result_mid = evaluate_kings_midgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_mid;
			} else if (phase_score >= 70) {
				int result_end = evaluate_kings_endgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_end;
			} else {
				int result_mid = evaluate_kings_midgame(r, white_passed_pawns, black_passed_pawns);
				int result_end = evaluate_kings_endgame(r, white_passed_pawns, black_passed_pawns);

				int blend_range = 30; // 70 - 40
				int end_weight = phase_score - 40;
				int mid_weight = blend_range - end_weight;
				blended_score = (mid_weight * result_mid + end_weight * result_end) / blend_range;
			}

			square_values[r] = abs(blended_score);
			total += blended_score;
			br_pt_kings += blended_score;
			//std::cout << "KINGS: " << int(r) << " | " << blended_score << std::endl;
			/* int result = evaluate_kings_midgame(r);
			square_values[r] = abs(result);
			total += result; */

			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;
		//adjust_pressure_and_support_tables_for_pins(occupied & ~kings);
		
		BoardState state(
			pawnsMask,
			knightsMask,
			bishopsMask,
			rooksMask,
			queensMask,
			kingsMask,
			occupied_whiteMask,
			occupied_blackMask,
			occupiedMask,
			1,
			turn,                  
			0,
			0,                          
			0,                           
			0    
		);
		//std::cout << total << std::endl;
		br_pieces = total; br_run = total;
		{
			PROF_BLOCK(PROF_CAPTURE_GAINS);
			total += Config::SCALE_CAPTURE_GAINS * approximate_capture_gains(occupied & ~kings, turn, state, pawn_rank_bonuses) / 100;
		}
		br_capture = total - br_run; br_run = total;
		//std::cout << total << std::endl;
		{
			PROF_BLOCK(PROF_PASSED_SUPPORT);
			total += Config::SCALE_PASSED_PAWN * boost_pieces_for_supporting_passed_pawns(white_passed_pawns, black_passed_pawns, pawn_rank_bonuses, isEndGame) / 100;
		}
		br_passed = total - br_run; br_run = total;
		//std::cout << total << std::endl;
		{
			PROF_BLOCK(PROF_LATENT_THREAT);
			total += Config::SCALE_LATENT_THREAT * get_latent_threat_score(__builtin_ctzll(occupied_white&kings), __builtin_ctzll(occupied_black&kings)) / 100;
		}
		br_latent = total - br_run; br_run = total;
		//std::cout << total << std::endl;

		//std::cout << phase_score << std::endl;
		int central_add;
		if(phase_score < 20){
			central_add = std::max(std::min((central_score * 3) / 2, 400), -400);
		} else if (phase_score < 31){
			central_add = std::max(std::min(central_score, 350), -350);
		} else if (phase_score < 45){
			central_add = std::max(std::min(central_score / 2, 300), -300);
		} else{
			central_add = std::max(std::min(central_score / 4, 300), -300);
		}
		total += Config::SCALE_CENTRAL * central_add / 100;
		br_central = total - br_run; br_run = total;

		if(total <= -1500){
			boost_white_for_piece_value_advantage = true;
		}else if(total >= 1500){
			boost_black_for_piece_value_advantage = true;
		}

		//std::cout << total << std::endl;


		// Boost the score for the side with more piece value proportional to how many pieces are on the board	    
		
		//std::cout << total << " black:" << blackPieceVal << "  white: " << whitePieceVal  << " diff: " << (int)(((whitePieceVal - blackPieceVal)/ (1.0 * whitePieceVal)) * 10000) << std::endl;
		

		
		// Boost the scores of both sides based on how poor the side's defense is relative to the opponent's offense
		
		//std::cout << occupied << " black:" << blackOffensiveScore << "  white: " << whiteDefensiveScore  << " diff: " << ((blackOffensiveScore - std::max(whiteDefensiveScore, 0)) * 3)<< std::endl;
		//std::cout << occupied << " black:" << blackDefensiveScore << "  white: " << whiteOffensiveScore  << " diff: " << ((whiteOffensiveScore - std::max(blackDefensiveScore, 0)) * 3) << std::endl;
		if (whiteOffensiveScore > blackDefensiveScore){
			total -= (whiteOffensiveScore - std::max(blackDefensiveScore, 0)) * 3;
		}
		br_imbalance_white = total - br_run; br_run = total;

		if (blackOffensiveScore > whiteDefensiveScore){
			total += (blackOffensiveScore - std::max(whiteDefensiveScore, 0)) * 3;
		}
		br_imbalance_black = total - br_run; br_run = total;

	// Else the game is in endgame phase
	}else{
		
		if (is_practically_drawn(pieceNum))
			return 0;

		// Update the attacking layer based on the position of the king
		{
			PROF_BLOCK(PROF_ATTACK_LAYER);
			setAttackingLayer(10, isEndGame);
		}
		//std::cout << total << std::endl;			
		uint64_t bb = pawnsMask;
		uint8_t r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			int pawn_rank_bonus = 0;
			// Call the midgame pawns evaluation function 
			PROF_BLOCK(PROF_PAWNS);
			int result = evaluate_pawns_endgame(r, white_passed_pawns, black_passed_pawns, pawn_rank_bonus);
			square_values[r] = abs(result);
			pawn_rank_bonuses[r] = pawn_rank_bonus;
			total += result;
			br_pt_pawns += result;
			//std::cout << "PAWNS: " << int(r) << " | " << result << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;		
		bb = knightsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_KNIGHTS);
			int result = evaluate_knights_endgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_knights += result;
			//std::cout << "KNIGHTS: " << int(r) << " | " << result << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;		
		bb = bishopsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_BISHOPS);
			int result = evaluate_bishops_endgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_bishops += result;
			//std::cout << "BISHOPS: " << int(r) << " | " << result << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;		
		bb = rooksMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_ROOKS);
			int result = evaluate_rooks_endgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_rooks += result;
			//std::cout << "ROOKS: " << int(r) << " | " << result << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;		
		bb = queensMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			PROF_BLOCK(PROF_QUEENS);
			int result = evaluate_queens_endgame(r, white_passed_pawns, black_passed_pawns);
			square_values[r] = abs(result);
			total += result;
			br_pt_queens += result;
			//std::cout << "QUEENS: " << int(r) << " | " << result << std::endl;
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		//std::cout << total << std::endl;		
		bb = kingsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			
			PROF_BLOCK(PROF_KINGS);
			int blended_score;
			if (phase_score <= 40) {
				int result_mid = evaluate_kings_midgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_mid;
			} else if (phase_score >= 70) {
				int result_end = evaluate_kings_endgame(r, white_passed_pawns, black_passed_pawns);
				blended_score = result_end;
			} else {
				int result_mid = evaluate_kings_midgame(r, white_passed_pawns, black_passed_pawns);
				int result_end = evaluate_kings_endgame(r, white_passed_pawns, black_passed_pawns);

				int blend_range = 30; // 70 - 40
				int end_weight = phase_score - 40;
				int mid_weight = blend_range - end_weight;
				blended_score = (mid_weight * result_mid + end_weight * result_end) / blend_range;
			}

			square_values[r] = abs(blended_score);
			total += blended_score;
			br_pt_kings += blended_score;

			// Clear the least significant set bit
			bb &= bb - 1;
		}

		//adjust_pressure_and_support_tables_for_pins(occupied & ~kings);
		
		BoardState state(
			pawnsMask,
			knightsMask,
			bishopsMask,
			rooksMask,
			queensMask,
			kingsMask,
			occupied_whiteMask,
			occupied_blackMask,
			occupiedMask,
			1,
			turn,                  
			0,
			0,                          
			0,                           
			0    
		);
		//std::cout << total << std::endl;
		br_pieces = total; br_run = total;
		{
			PROF_BLOCK(PROF_CAPTURE_GAINS);
			total += Config::SCALE_CAPTURE_GAINS * approximate_capture_gains(occupied & ~kings, turn, state, pawn_rank_bonuses) / 100;
		}
		br_capture = total - br_run; br_run = total;
		//std::cout << total << std::endl;
		{
			PROF_BLOCK(PROF_PASSED_SUPPORT);
			total += Config::SCALE_PASSED_PAWN * boost_pieces_for_supporting_passed_pawns(white_passed_pawns, black_passed_pawns, pawn_rank_bonuses, isEndGame) / 100;
		}
		br_passed = total - br_run; br_run = total;
		//std::cout << " after pp: " << total << std::endl;

		if(total <= -1500){
			boost_white_for_piece_value_advantage = true;
		}else if(total >= 1500){
			boost_black_for_piece_value_advantage = true;
		}

		// Check if the position is an advanced endgame
		if (isNearGameEnd){
			br_ae_input = total;
			{
				PROF_BLOCK(PROF_ADV_ENDGAME);
				total = advanced_endgame_eval(total, turn);
			}
			br_advanced_fired = true;
			br_advanced_total = total;
			br_run = total;
		}
	}
	
	/*
		In this code section, boost both white and blacks score based on the existence of bishop and knight pairs
	*/
	if (__builtin_popcountll(occupied_white&bishops) == 2){
		total -= 300;
	}
	if (__builtin_popcountll(occupied_white&knights) == 2){
		total -= 200;
	}
	if (__builtin_popcountll(occupied_black&bishops) == 2){
		total += 300;
	}
	if (__builtin_popcountll(occupied_black&knights) == 2){
		total += 200;
	}
	br_pairs = total - br_run; br_run = total;

	// Endgame convertibility scale (env-gated, default off = byte-identical). Damp an unconvertible
	// material/placement lead toward draw, and scale the piece-value boost below by the same factor so
	// the material-conversion bonus is damped coherently. A balanced (~0) total is unaffected; conv_s
	// stays 1.0 in the midgame and whenever the flag is off, so the boost expression is unchanged then.
	double conv_s = 1.0;
	if (Config::ENABLE_ENDGAME_SCALE && isEndGame){
		conv_s = endgame_convertibility_scale(white_passed_pawns, black_passed_pawns);
		total = (int)(total * conv_s);
	}
	//std::cout << " before boost: "<< total << std::endl;
	if (blackPieceVal > whitePieceVal){
		if(boost_black_for_piece_value_advantage){
			total += (int)(((blackPieceVal - whitePieceVal)/ (1.0 * blackPieceVal)) * 10000 * conv_s);
		}

	}else if (whitePieceVal > blackPieceVal){
		if(boost_white_for_piece_value_advantage){
			total -= (int)(((whitePieceVal - blackPieceVal)/ (1.0 * whitePieceVal)) * 10000 * conv_s);
		}
	}
	br_pv_boost = total - br_run; br_run = total;
	//std::cout << total << std::endl;
	//std::cout << "AAAA: " << approximate_capture_gains1(occupied & ~kings, turn) << " occupied: " << (occupied & ~kings) << " turn: " << turn <<  std::endl;
	//std::cout << total << " " << whiteOffensiveScore << " " << whiteDefensiveScore << " " <<  blackOffensiveScore << " " << blackDefensiveScore << " " <<std::endl;

	// Diagnostic-only: publish the per-term attribution (read-only w.r.t. `total`; see EvalBreakdown).
	if (g_capture_eval_breakdown){
		g_eval_breakdown.total = total;
		g_eval_breakdown.pieces = br_pieces;
		g_eval_breakdown.material = blackPieceVal - whitePieceVal;
		g_eval_breakdown.capture_gains = br_capture;
		g_eval_breakdown.passed_pawn_support = br_passed;
		g_eval_breakdown.latent_threat = br_latent;
		g_eval_breakdown.central = br_central;
		g_eval_breakdown.imbalance_white = br_imbalance_white;
		g_eval_breakdown.imbalance_black = br_imbalance_black;
		g_eval_breakdown.pair_bonus = br_pairs;
		g_eval_breakdown.piece_value_boost = br_pv_boost;
		g_eval_breakdown.phase_score = phase_score;
		g_eval_breakdown.is_endgame = isEndGame;
		g_eval_breakdown.advanced_endgame_fired = br_advanced_fired;
		g_eval_breakdown.advanced_endgame_total = br_advanced_total;
		g_eval_breakdown.pt_pawns = br_pt_pawns;
		g_eval_breakdown.pt_knights = br_pt_knights;
		g_eval_breakdown.pt_bishops = br_pt_bishops;
		g_eval_breakdown.pt_rooks = br_pt_rooks;
		g_eval_breakdown.pt_queens = br_pt_queens;
		g_eval_breakdown.pt_kings = br_pt_kings;
		g_eval_breakdown.ae_input     = br_advanced_fired ? br_ae_input   : 0;
		g_eval_breakdown.ae_matedrive = br_advanced_fired ? g_ae_matedrive : 0;
		g_eval_breakdown.ae_passer    = br_advanced_fired ? g_ae_passer    : 0;
	}
	return total;
}

EvalBreakdown eval_breakdown_capture(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask){

	/*
		Diagnostic wrapper: run the REAL static eval with term capture enabled and return the per-term
		breakdown. The global is zeroed first so the practically-drawn early-return (which skips the publish
		block) still yields an honest all-zero / total-0 reading. Not reentrant; single-threaded diagnostic use.
	*/
	g_eval_breakdown = {};
	g_capture_eval_breakdown = true;
	int total = placement_and_piece_eval(moveNum, turn, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, occupied_whiteMask, occupied_blackMask, occupiedMask);
	g_capture_eval_breakdown = false;
	g_eval_breakdown.total = total;
	return g_eval_breakdown;
}

uint8_t lowest_value_attacker(uint64_t attackers, bool attackedColour){
	uint64_t attackingSidePieces = (attackedColour ? occupied_black : occupied_white) & attackers;
	
	if ((attackingSidePieces & pawns) != 0)
        return __builtin_ctzll(attackingSidePieces & pawns); // pawn
    else if ((attackingSidePieces & knights) != 0)
        return __builtin_ctzll(attackingSidePieces & knights);  // knight
    else if ((attackingSidePieces & bishops) != 0)
        return __builtin_ctzll(attackingSidePieces & bishops);  // bishop
    else if ((attackingSidePieces & rooks) != 0)
        return __builtin_ctzll(attackingSidePieces & rooks);  // rook
    else if ((attackingSidePieces & queens) != 0)
        return __builtin_ctzll(attackingSidePieces & queens);  // queen
    else
        return 0; // or INT_MAX, or some sentinel for "no attacker"
}

void apply_basic_capture(uint8_t from, uint8_t to, uint64_t& white_pieces, uint64_t& black_pieces, bool white_to_move) {
	uint64_t from_mask = BB_SQUARES[from];
	uint64_t to_mask   = BB_SQUARES[to];

	if (white_to_move) {
		// Remove piece from 'from' square
		white_pieces &= ~from_mask;

		// Move it to 'to' square
		white_pieces |= to_mask;

		// Remove captured black piece
		black_pieces &= ~to_mask;
	} else {
		// Black's move
		black_pieces &= ~from_mask;
		black_pieces |= to_mask;
		white_pieces &= ~to_mask;
	}
}

inline CaptureInfo* find_last_viable_capture(CaptureStack& captures, uint64_t& white_pieces, uint64_t& black_pieces, bool captureColour) {
	
	uint64_t from_side = captureColour ? white_pieces : black_pieces;
    uint64_t to_side   = captureColour ? black_pieces : white_pieces;

    for (int i = static_cast<int>(captures.size()) - 1; i >= 0; --i) {
        CaptureInfo& cur = captures[i];

        uint8_t from = cur.from;
        uint8_t to   = cur.to;

        bool isValid = ((from_side & BB_SQUARES[from]) != 0) &&
                       ((to_side   & BB_SQUARES[to])   != 0);

        if (isValid) {
            return &cur;
        }
    }

    return nullptr;
}

inline std::optional<CaptureInfo> find_and_pop_last_viable_capture(CaptureStack& captures, uint64_t white_pieces, uint64_t black_pieces, bool captureColour) {
    uint64_t from_side = captureColour ? white_pieces : black_pieces;
    uint64_t to_side   = captureColour ? black_pieces : white_pieces;

    while (!captures.empty()) {
        CaptureInfo cur = captures.back();
        uint8_t from = cur.from;
        uint8_t to   = cur.to;

        bool isValid = ((from_side & BB_SQUARES[from]) != 0) &&
                       ((to_side   & BB_SQUARES[to])   != 0);

        captures.pop_back();  // Always pop, whether valid or not

        if (isValid) {
            return cur;  // Return the valid capture
        }
    }

    return std::nullopt;  // No valid capture found
}

inline bool can_evade(uint8_t target_square, bool target_colour){
	
	// Acquire the attacks mask for the current piece
	uint64_t ourPieces = target_colour ? occupied_white : occupied_black;

	uint64_t pieceAttackMask = attacks_mask(target_colour,occupied,target_square,pieceTypeLookUp[target_square]) & ~ourPieces;

	uint64_t opposingPieces = target_colour ? occupied_black : occupied_white;

	// Loop through the attacks mask
	uint8_t to_square = 0;
	uint64_t bb = pieceAttackMask;
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		to_square = __builtin_ctzll(bb);	
		bb &= bb - 1;
		//std::cout << int(to_square)<< " "<<  attack_bitmasks[to_square]<< std::endl;
		if ((attack_bitmasks[to_square] & opposingPieces) == 0){
			return true;
		}
	}
	return false;
}

inline int approximate_capture_gains1(uint64_t bb, bool turn) {
    int black_gains = 0;
    int white_gains = 0;

	CaptureStack white_captures;
	CaptureStack black_captures;

    while (bb) {
        uint8_t r = __builtin_ctzll(bb);
        bb &= bb - 1;

        bool current_colour = (occupied_white & (BB_SQUARES[r])) != 0;

        int attackers = num_attackers[r];
        int supporters = num_supporters[r];
        int pressure   = current_colour ? pressure_black[r] : pressure_white[r];
        int support    = current_colour ? support_white[r]  : support_black[r];

        if (attackers == 0)
            continue;

		if (pressure > support) {
			if (supporters == 0) {
				uint8_t from = __builtin_ctzll(attack_bitmasks[r]);
				CaptureInfo newCapture(from, r, square_values[r]);
				//std::cout << "AAFrom: " << (int)from << " to " << int(r) << " value: " << newCapture.value_gained << "  "<< attackers << "  " << supporters << std::endl;
				if (current_colour)
					black_captures.push_back(newCapture);
				else
					white_captures.push_back(newCapture);
			} else {
				uint8_t from = lowest_value_attacker(attack_bitmasks[r], current_colour);
				CaptureInfo newCapture(from, r,  std::max(square_values[r] - square_values[from], 0));
				//std::cout << "BBBFrom: " << (int)from << " to " << int(r) << " value: " << newCapture.value_gained<< "  " << attack_bitmasks[r] << "  "<< attackers << "  " << supporters << std::endl;
				if (current_colour)
					black_captures.push_back(newCapture);
				else
					white_captures.push_back(newCapture);
			}     
        } 
    }

	std::sort(black_captures.begin(), black_captures.end(), [](const CaptureInfo& a, const CaptureInfo& b) {
    	return a.value_gained < b.value_gained;
	});

	std::sort(white_captures.begin(), white_captures.end(), [](const CaptureInfo& a, const CaptureInfo& b) {
    	return a.value_gained < b.value_gained; 
	});

	bool current_turn = turn;
	uint64_t black_pieces = occupied_black;
	uint64_t white_pieces = occupied_white;
	
	while (!white_captures.empty() || !black_captures.empty()) {
		bool evading = false;

		CaptureStack& own_captures = current_turn ? white_captures : black_captures;
		CaptureStack& opp_captures = current_turn ? black_captures : white_captures;

		// Step 1: Evaluate evasion option
		if (!opp_captures.empty()) {
			CaptureInfo* cur_side_capture = find_last_viable_capture(own_captures, white_pieces, black_pieces, current_turn);
			
			
			if (cur_side_capture != nullptr){
				uint64_t black_pieces_copy = black_pieces;
				uint64_t white_pieces_copy = white_pieces;

				apply_basic_capture(cur_side_capture->from, cur_side_capture->to, white_pieces_copy, black_pieces_copy, current_turn);
				CaptureInfo* opp_side_capture = find_last_viable_capture(opp_captures, white_pieces_copy, black_pieces_copy, !current_turn);
				
				if (opp_side_capture != nullptr){
					if (opp_side_capture->value_gained > cur_side_capture->value_gained){
						//std::cout << "CCCFrom: " << (int)opp_side_capture->from << " to " << int(opp_side_capture->to) << " value: " << opp_side_capture->value_gained << std::endl;
						if (can_evade(opp_side_capture->to, current_turn)){				
							evading = true;
							find_and_pop_last_viable_capture(opp_captures, white_pieces_copy, black_pieces_copy, current_turn);										
						}
					}
				}
				
			} else {
				CaptureInfo* opp_side_capture = find_last_viable_capture(opp_captures, white_pieces, black_pieces, !current_turn);
				if (opp_side_capture != nullptr && can_evade(opp_side_capture->to, current_turn)){				
					evading = true;
					find_and_pop_last_viable_capture(opp_captures, white_pieces, black_pieces, current_turn);								
				}
			}
		}

		// Step 2: Perform a capture if not evading
		if (!evading) {
			std::optional<CaptureInfo> cur_side_capture = find_and_pop_last_viable_capture(own_captures, white_pieces, black_pieces, current_turn);
			if (cur_side_capture) {
				apply_basic_capture(cur_side_capture->from, cur_side_capture->to, white_pieces, black_pieces, current_turn);
				if (current_turn){
					white_gains += cur_side_capture->value_gained;
					blackPieceVal -= cur_side_capture->value_gained;
				} else {
					black_gains += cur_side_capture->value_gained;
					whitePieceVal -= cur_side_capture->value_gained;
				}
			}
		}

		// Flip the turn
		current_turn = !current_turn;
	}
    return black_gains - white_gains;
}

inline int approximate_capture_gains(uint64_t bb, bool turn, const BoardState& state, const std::array<int, 64>& pawn_rank_bonuses) {
    int black_gains = 0;
    int white_gains = 0;

	CaptureStack white_captures;
	CaptureStack black_captures;

    while (bb) {
        uint8_t r = __builtin_ctzll(bb);
        bb &= bb - 1;

        bool current_colour = (occupied_white & (BB_SQUARES[r])) != 0;

        int attackers = __builtin_popcountll(attack_bitmasks[r] & state.occupied_colour[!current_colour]);
        

        if (attackers == 0)
            continue;
		//std::cout << std::endl;
		//std::cout <<(int)r << std::endl;
		int static_exchange_eval;
		{
			PROF_BLOCK(PROF_SEE);
			static_exchange_eval = see (r, !current_colour, state);
		}
		//std::cout << static_exchange_eval << std::endl;
		if (static_exchange_eval >= 0) {

			uint8_t from = get_least_valuable_attacker(attack_bitmasks[r] & state.occupied_colour[!current_colour], state);
			CaptureInfo newCapture(from, r, static_exchange_eval);

			if (current_colour)
				black_captures.push_back(newCapture);
			else
				white_captures.push_back(newCapture);
		}			
    }

	std::sort(black_captures.begin(), black_captures.end(), [](const CaptureInfo& a, const CaptureInfo& b) {
    	return a.value_gained < b.value_gained;
	});

	std::sort(white_captures.begin(), white_captures.end(), [](const CaptureInfo& a, const CaptureInfo& b) {
    	return a.value_gained < b.value_gained; 
	});

	bool current_turn = turn;
	//std::cout << current_turn << std::endl;
	uint64_t black_pieces = occupied_black;
	uint64_t white_pieces = occupied_white;
	
	while (!white_captures.empty() || !black_captures.empty()) {
		bool evading = false;

		CaptureStack& own_captures = current_turn ? white_captures : black_captures;
		CaptureStack& opp_captures = current_turn ? black_captures : white_captures;

		// Step 1: Evaluate evasion option
		if (!opp_captures.empty()) {
			CaptureInfo* cur_side_capture = find_last_viable_capture(own_captures, white_pieces, black_pieces, current_turn);
			
			
			if (cur_side_capture != nullptr){

				uint64_t black_pieces_copy = black_pieces;
				uint64_t white_pieces_copy = white_pieces;

				apply_basic_capture(cur_side_capture->from, cur_side_capture->to, white_pieces_copy, black_pieces_copy, current_turn);
				CaptureInfo* opp_side_capture = find_last_viable_capture(opp_captures, white_pieces_copy, black_pieces_copy, !current_turn);
				
				if (opp_side_capture != nullptr && opp_side_capture->value_gained > cur_side_capture->value_gained && can_evade(opp_side_capture->to, current_turn)){								
					evading = true;
					find_and_pop_last_viable_capture(opp_captures, white_pieces_copy, black_pieces_copy, current_turn);															
				}
				
			} else {
				CaptureInfo* opp_side_capture = find_last_viable_capture(opp_captures, white_pieces, black_pieces, !current_turn);
				if (opp_side_capture != nullptr && can_evade(opp_side_capture->to, current_turn)){				
					evading = true;
					find_and_pop_last_viable_capture(opp_captures, white_pieces, black_pieces, current_turn);								
				}
			}
		}

		// Step 2: Perform a capture if not evading
		if (!evading) {
			std::optional<CaptureInfo> cur_side_capture = find_and_pop_last_viable_capture(own_captures, white_pieces, black_pieces, current_turn);
			if (cur_side_capture) {
				//std::cout << (int)cur_side_capture->from << " | " << (int)cur_side_capture->to << " | " << (int)cur_side_capture->value_gained << std::endl;
				apply_basic_capture(cur_side_capture->from, cur_side_capture->to, white_pieces, black_pieces, current_turn);
				if (current_turn){
					int value_gained = cur_side_capture->value_gained;
					if(pieceTypeLookUp[cur_side_capture->to] == PAWN && pieceTypeLookUp[cur_side_capture->from] != PAWN){
						value_gained += pawn_rank_bonuses[cur_side_capture->to];
					}
					white_gains += value_gained;
					blackPieceVal -= value_gained;

					
				} else {
					int value_gained = cur_side_capture->value_gained;
					if(pieceTypeLookUp[cur_side_capture->to] == PAWN && pieceTypeLookUp[cur_side_capture->from] != PAWN){
						value_gained += (Config::ENABLE_CAPGAIN_PAWN_FIX ? -pawn_rank_bonuses[cur_side_capture->to] : pawn_rank_bonuses[cur_side_capture->to]);
					}
					black_gains += value_gained;
					whitePieceVal -= value_gained;
				}
			}
		}

		// Flip the turn
		current_turn = !current_turn;
	}
    return black_gains - white_gains;
}

inline void initializePieceValues(uint64_t bb){
	
	/*
		Function to set piece types in a global array
		
		Parameters:
		- bb: The occupied piece mask		
	*/
	
	// Reset the global array as empty
	pieceTypeLookUp = {};
	
	// Loop through the mask
	uint8_t r = 0;
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);		
		
		// Call the piece type function to populate the array
		pieceTypeLookUp [r] = piece_type_at (r);
		bb &= bb - 1;			
	} 
}



// One-entry endgame cache for setAttackingLayer (see ENABLE_ATTACK_LAYER_CACHE). In the endgame the
// per-square open/pawn-shield branches are skipped, so the layer is a pure function of the two king
// squares; a (wk,bk)-keyed single entry is therefore always correct and can never go stale.
static struct {
	bool valid = false;
	uint8_t wk = 0, bk = 0;
	std::array<std::array<std::array<int, 8>, 8>, 2> table;
} g_attack_layer_eg_cache;

// Split midgame half-caches (see ENABLE_ATTACK_LAYER_CACHE_MIDGAME). The white king-loop writes only
// attackingLayer[1] from (white king sq + white pieces/pawns in its 2-ring); the black loop only [0].
// Caching the two halves independently lets each survive moves the other side makes away from its king.
static struct {
	bool valid = false;
	uint8_t k = 0;
	uint64_t occ = 0, pawns = 0;
	std::array<std::array<int, 8>, 8> layer;
} g_al_mg_white, g_al_mg_black;

inline void setAttackingLayer(int increment, bool isEndGame){

	/*
		Function to update the attacking layer relative to the king's positions
		
		Parameters:
		- square: The increment to be used to boost the required squares
		
		Returns:
		A unsigned char representing the piece type
	*/
	
	// Endgame layer depends only on the two king squares; reuse the cached one on a match (lossless)
	bool use_al_cache = Config::ENABLE_ATTACK_LAYER_CACHE && isEndGame;
	uint8_t al_wk = 0, al_bk = 0;
	if (use_al_cache){
		al_wk = 63 - __builtin_clzll(occupied_white & kings);
		al_bk = 63 - __builtin_clzll(occupied_black & kings);
		if (g_attack_layer_eg_cache.valid && g_attack_layer_eg_cache.wk == al_wk && g_attack_layer_eg_cache.bk == al_bk){
			attackingLayer = g_attack_layer_eg_cache.table;
			return;
		}
	}

	// Midgame layer also depends on own pieces/pawns in each king's 2-ring; cache the two halves independently
	bool use_mg_cache = Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME && !isEndGame;

	// Set the default attacking layer

	if (isEndGame){
		attackingLayer = {{
			{{
				{{0,3,3,3,3,3,3,10}},
				{{0,0,5,5,5,7,10,15}},
				{{0,0,5,15,20,30,10,15}},
				{{0,0,5,30,35,35,10,15}},
				{{0,0,5,30,35,35,10,15}},
				{{0,0,5,15,20,30,10,15}},
				{{0,0,5,5,5,7,10,15}},
				{{0,3,3,3,3,3,3,10}}
			}},
			{{
				{{10,3,3,3,3,3,3,0}},
				{{15,10,7,5,5,5,0,0}},
				{{15,10,30,20,15,5,0,0}},
				{{15,10,35,35,30,5,0,0}},
				{{15,10,35,35,30,5,0,0}},
				{{15,10,30,20,15,5,0,0}},
				{{15,10,7,5,5,5,0,0}},
				{{10,3,3,3,3,3,3,0}}
			}}
		}};
	} else {
		attackingLayer = {{
			{{
				{{0,3,3,3,5,7,10,10}},
				{{0,0,5,5,15,20,25,15}},
				{{0,0,5,30,45,40,25,15}},
				{{0,0,5,50,65,45,25,15}},
				{{0,0,5,50,65,45,25,15}},
				{{0,0,5,30,45,40,25,15}},
				{{0,0,5,5,15,20,25,15}},
				{{0,3,3,3,5,7,10,10}}
			}},
			{{
				{{10,10,7,5,3,3,3,0}},
				{{15,25,20,15,5,5,0,0}},
				{{15,25,40,45,30,5,0,0}},
				{{15,25,45,65,50,5,0,0}},
				{{15,25,45,65,50,5,0,0}},
				{{15,25,40,45,30,5,0,0}},
				{{15,25,20,15,5,5,0,0}},
				{{10,10,7,5,3,3,3,0}}
			}}
		}};
	}
	/* 
	
	// Acquire the number of pieces on the board excluding the kings
	int pieceNum = scan_reversed_size(occupied) - 2;
	
	// Determine if the position is in endgame based on the number 
	bool isEndGame = pieceNum < 16;
	if (queens == 0){
		isEndGame = pieceNum < 18;
	} */
	
	// Set variable for squares being open near the king
	bool squareOpen = true;

	bool pawnShield = false;
	
	// Set the multiplier for open square boosts
	int multiplier = 5;
	
	// Define the x and y coordinates for each square
	uint8_t x,y;
	
	// White king half-layer: reuse attackingLayer[1] if its key (king sq + white pieces/pawns in the 2-ring) matches
	bool mg_white_hit = false;
	uint8_t mg_wk = 0; uint64_t mg_w_occ = 0, mg_w_pawns = 0;
	if (use_mg_cache){
		mg_wk = 63 - __builtin_clzll(occupied_white & kings);
		mg_w_occ = occupied_white & king_ring2[mg_wk];
		mg_w_pawns = (occupied_white & pawns) & king_ring2[mg_wk];
		if (g_al_mg_white.valid && g_al_mg_white.k == mg_wk && g_al_mg_white.occ == mg_w_occ && g_al_mg_white.pawns == mg_w_pawns){
			attackingLayer[1] = g_al_mg_white.layer;
			mg_white_hit = true;
		}
	}
	if (!mg_white_hit){
	// Loop through the squares around the white king
	uint8_t r = 0;
	uint64_t bb = attacks_mask(true,0ULL,63 - __builtin_clzll(occupied_white&kings),6);
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);								

		// Get the x and y coordinates for the given square
		y = r >> 3;
		x = r & 7;
		
		// Increment the area around the king
        attackingLayer[1][x][y] += increment;
		
		// If the square is open around the king, boost the score further
        squareOpen = false;
		pawnShield = false;
		if (!isEndGame){
			if ((occupied_white & (BB_SQUARES[r])) == 0){
				attackingLayer[1][x][y] += increment * multiplier;
				squareOpen = true;
			} else if ((occupied_white & pawns & (BB_SQUARES[r])) != 0){
				attackingLayer[1][x][y] -= increment >> 1;
				pawnShield = true;
			}
		}
		
		// Loop through the squares around the current king move square
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			
			// Get the position of the least significant set bit of the mask			
			r_inner = __builtin_ctzll(bb_inner);
			
			// Get the x and y coordinates for the given square
			y = r_inner >> 3;
			x = r_inner & 7;
			
			// Increment the given square
			attackingLayer[1][x][y] += increment;
			
			// If the square is open around the king, boost the score further
			if (!isEndGame){
				if (squareOpen && (occupied_white & (BB_SQUARES[r_inner])) == 0){					
					attackingLayer[1][x][y] += increment * multiplier;					
				} else if (pawnShield){
					attackingLayer[1][x][y] -= increment >> 1;
				}				
			}
			bb_inner &= bb_inner - 1;
		}
		
		bb &= bb - 1;
	}
		if (use_mg_cache){
			g_al_mg_white.valid = true; g_al_mg_white.k = mg_wk;
			g_al_mg_white.occ = mg_w_occ; g_al_mg_white.pawns = mg_w_pawns;
			g_al_mg_white.layer = attackingLayer[1];
		}
	}

	// Black king half-layer: reuse attackingLayer[0] if its key (king sq + black pieces/pawns in the 2-ring) matches
	bool mg_black_hit = false;
	uint8_t mg_bk = 0; uint64_t mg_b_occ = 0, mg_b_pawns = 0;
	if (use_mg_cache){
		mg_bk = 63 - __builtin_clzll(occupied_black & kings);
		mg_b_occ = occupied_black & king_ring2[mg_bk];
		mg_b_pawns = (occupied_black & pawns) & king_ring2[mg_bk];
		if (g_al_mg_black.valid && g_al_mg_black.k == mg_bk && g_al_mg_black.occ == mg_b_occ && g_al_mg_black.pawns == mg_b_pawns){
			attackingLayer[0] = g_al_mg_black.layer;
			mg_black_hit = true;
		}
	}
	if (!mg_black_hit){
	// Loop through the squares around the black king
	uint8_t r = 0;
	uint64_t bb = attacks_mask(false,0ULL,63 - __builtin_clzll(occupied_black&kings),6);
	while (bb) {
		
		// Get the position of the least significant set bit of the mask
		r = __builtin_ctzll(bb);									

		// Get the x and y coordinates for the given square
		y = r >> 3;
		x = r & 7;
		
		// Increment the area around the king
        attackingLayer[0][x][y] += increment;
		
		// If the square is open around the king, boost the score further
        squareOpen = false;
		pawnShield = false;
		if (!isEndGame){
			if ((occupied_black & (BB_SQUARES[r])) == 0){
				attackingLayer[0][x][y] += increment * multiplier;
				squareOpen = true;
			} else if ((occupied_black & pawns & (BB_SQUARES[r])) != 0){
				attackingLayer[0][x][y] -= increment >> 1;
				pawnShield = true;
			}
		}		
		// Loop through the squares around the current king move square
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			
			// Get the position of the least significant set bit of the mask
			r_inner = __builtin_ctzll(bb_inner);
			
			// Get the x and y coordinates for the given square
			y = r_inner >> 3;
			x = r_inner & 7;
			
			// Increment the given square
			attackingLayer[0][x][y] += increment;
			
			// If the square is open around the king, boost the score further
						
			if (!isEndGame){
				if (squareOpen && (occupied_black & (BB_SQUARES[r_inner])) == 0){					
					attackingLayer[0][x][y] += increment * multiplier;					
				} else if (pawnShield){
					attackingLayer[0][x][y] -= increment >> 1;
				}				
			}			
			bb_inner &= bb_inner - 1;
		}		
		bb &= bb - 1;
	}
		if (use_mg_cache){
			g_al_mg_black.valid = true; g_al_mg_black.k = mg_bk;
			g_al_mg_black.occ = mg_b_occ; g_al_mg_black.pawns = mg_b_pawns;
			g_al_mg_black.layer = attackingLayer[0];
		}
	}

	// Store the freshly-computed endgame layer for the next (wk,bk)-matching call
	if (use_al_cache){
		g_attack_layer_eg_cache.valid = true;
		g_attack_layer_eg_cache.wk = al_wk;
		g_attack_layer_eg_cache.bk = al_bk;
		g_attack_layer_eg_cache.table = attackingLayer;
	}
}

void printLayers(){
	
	/*
		Function to print the attacking layers for testing purposes		
	*/
	
	for (int i = 0; i < 2; i++){
		std::cout << "Layer " << i << ": " << std::endl;
		for (int j = 0; j < 8; j++){
			for (int k = 0; k < 8; k++){
				std::cout << attackingLayer[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	
}

inline int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces, uint64_t& white_passed_pawns, uint64_t& black_passed_pawns) {
	
	/*
		Function to acquire the increment for a pawn being or having the potential to be a passed pawn
		
		Parameters:
		- colour: The colour of the current side
		- opposingPawnMask: The pawns of the opposing side
		- ppIncrement: The base increment to be adjusted
		- x: The starting x coordinate
		- y: The starting y coordinate
		- opposingPieces: The bit mask of the opposing side's pieces
		- curSidePieces: The bit mask of the current side's pieces		
		
		Returns:
		The final pawn increment
	*/
	
	// The file and rank if the given square
    uint8_t file = x;
    uint8_t rank = y;
		
	// Define a mask to hold the final mask to be analyzed
    uint64_t bitmask = 0;
	
	// Define a copy of the initial increment
	int incrementCopy = ppIncrement;
	
	// Define a mask that will represent the squares in the file directly in front of the pawn
	uint64_t infrontMask = 0;

	/*
		In this section, acquire all the squares in front of pawn including those on either side of it
	*/
	
	// If the current side is white
    if (colour) {
		
        // Iterate over the three relevant files
        for (int f = file - 1; f < file + 2; ++f) {
            
			// Check if the file is within bounds
			if (f >= 0 && f <= 7) {
				bitmask |= BB_FILES [f] & ~((1ULL << ((rank + 1) * 8)) - 1);
				if (f == file){
					infrontMask |= BB_FILES [f] & ~((1ULL << ((rank + 1) * 8)) - 1);
				}
            }
        }
	// Else the current side is black	
    } else {
        // Iterate over the three relevant files
        for (int f = file - 1; f < file + 2; ++f) {
			
			// Check if the file is within bounds
            if (f >= 0 && f <= 7) {  
				bitmask |= BB_FILES [f] & ((1ULL << (rank * 8)) - 1);
				if (f == file){
					infrontMask |= BB_FILES [f] & ((1ULL << (rank * 8)) - 1);
				}
            }
        }
    }

	// Of the squares in front of pawn, filter to only include opposing pawns
    bitmask &= opposingPawnMask;	
		
	//std::cout << "PPMASK: " << bitmask << std::endl;
	// Loop through the bitmask 
	uint8_t r = 0;
	uint64_t bb = bitmask;
	while (bb) {
		r = __builtin_ctzll(bb);
		
		// If there is a blocker directly in front of the pawn, then it has no potential to be a passed pawn
		if ((r & 7) == x){
			return 0;			
		}
		
		// Otherwise there is an opposing pawn defending the promotion path, thereby lowering the increment
		ppIncrement -= Config::PP_OPP_PAWN_PEN;
		bb &= bb - 1;
	}
	//if (y == 4 && x == 2){std::cout << bitmask  << " " << ppIncrement << " " << incrementCopy << std::endl;}
	// The minimum increment is 0
	if (ppIncrement < 0) {
		return 0;	
	// Otherwise check if the increment does not suffer a decrement, this suggests the pawn is a passed pawn
	} else if (ppIncrement == incrementCopy){
		
		if (colour){			
			white_passed_pawns |= BB_SQUARES[y * 8 + x];
		}else{			
			black_passed_pawns |= BB_SQUARES[y * 8 + x];
		}

		// Check if there exists a non-pawn blocker infront of the passed pawn
		if ((infrontMask & (opposingPieces | curSidePieces))){
			
			// If the piece is the that of the opponent, decrement the score as it is blockaded
			if ((infrontMask & opposingPieces)){
				ppIncrement -= Config::PP_BLOCKADE_PEN;
			}
		// Else no blocker exists, the passed pawn is un-impeded, earning a larger boost
		} else{
			ppIncrement += Config::PP_UNBLOCKED;
		}

		// Give more of a boost if the passed pawn has supporters on either side or is defended.
		uint8_t square = rank * 8 + file;
		uint64_t pawnBB = BB_SQUARES[square];

		uint64_t left = (pawnBB >> 1) & ~BB_FILE_H & pawns;
		uint64_t right = (pawnBB << 1) & ~BB_FILE_A & pawns;

		if (colour) {
			uint64_t sw = (pawnBB >> 9) & ~BB_FILE_H & pawns & occupied_white;
			uint64_t se = (pawnBB >> 7) & ~BB_FILE_A & pawns & occupied_white;
			//if (y == 4 && x == 2){std::cout << sw << " " << se <<  " " << left << " " << right << std::endl;}
			if (sw != 0){
				ppIncrement += Config::PP_DIAG_SUPPORT;
				if ((left & occupied_black) == 0){
					ppIncrement += Config::PP_FILE_CLEAR;
				}
			}

			if (se != 0){
				ppIncrement += Config::PP_DIAG_SUPPORT;
				if ((right & occupied_black) == 0){
					ppIncrement += Config::PP_FILE_CLEAR;
				}
			}

			if ((left & occupied_white) != 0){
					ppIncrement += Config::PP_HORIZ_SUPPORT;
			}

			if ((right & occupied_white) != 0){
					ppIncrement += Config::PP_HORIZ_SUPPORT;
			}				
		} else {
			uint64_t nw = (pawnBB << 7) & ~BB_FILE_H & pawns & occupied_black;
			uint64_t ne = (pawnBB << 9) & ~BB_FILE_A & pawns & occupied_black;
			
			if (nw != 0){
				ppIncrement += Config::PP_DIAG_SUPPORT;
				if ((left & occupied_white) == 0){
					ppIncrement += Config::PP_FILE_CLEAR;
				}
			}

			if (ne != 0){
				ppIncrement += Config::PP_DIAG_SUPPORT;
				if ((right & occupied_white) == 0){
					ppIncrement += Config::PP_FILE_CLEAR;
				}
			}

			if ((left & occupied_black) != 0){
					ppIncrement += Config::PP_HORIZ_SUPPORT;
			}

			if ((right & occupied_black) != 0){
					ppIncrement += Config::PP_HORIZ_SUPPORT;
			}	
		}
	}
	//std::cout << bitmask << " | " << ppIncrement << " | " << infrontMask << " | "<< incrementCopy << " x: " << (int)x << " y: " << (int)y << std::endl;
	return ppIncrement;
}

void update_bitmasks(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask){
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	occupied = occupiedMask;
}

/*
	Compile-gated eval profiler implementation. The three entry points are defined
	unconditionally so the Cython bridge always links; their bodies are #ifdef-walled
	and become no-ops in a production build (no EVAL_PROFILE → byte-identical binary).
*/
#ifdef EVAL_PROFILE
uint64_t g_prof_cycles[NUM_PROF_TERMS];
uint64_t g_prof_calls[NUM_PROF_TERMS];

static const char* PROF_TERM_NAMES[NUM_PROF_TERMS] = {
	"PAWNS", "KNIGHTS", "BISHOPS", "ROOKS", "ROOK_ACTIVITY",
	"QUEENS", "KINGS", "ATTACK_LAYER", "CAPTURE_GAINS",
	"PASSED_SUPPORT", "LATENT_THREAT", "ADV_ENDGAME",
	"SEE", "BISHOP_ACTIVITY", "BISHOP_COLOUR"
};

// Terms PROF_PAWNS..PROF_ADV_ENDGAME are the top-level, mutually-exclusive call
// sites whose cycles sum to ~the instrumented eval; the remainder (SEE, the bishop
// helpers, and ROOK_ACTIVITY) are nested subsets and excluded from the %-share base.
static const int PROF_NUM_EXCLUSIVE = PROF_ADV_ENDGAME + 1;
#endif

void eval_profile_reset()
{
#ifdef EVAL_PROFILE
	for (int i = 0; i < NUM_PROF_TERMS; ++i) {
		g_prof_cycles[i] = 0;
		g_prof_calls[i] = 0;
	}
#endif
}

void eval_profile_dump(const char* label)
{
#ifdef EVAL_PROFILE
	uint64_t base = 0;
	for (int i = 0; i < PROF_NUM_EXCLUSIVE; ++i)
		base += g_prof_cycles[i];
	if (base == 0)
		base = 1; // avoid divide-by-zero on an empty run

	std::cerr << "[eval_profile] " << label << "\n";
	std::cerr << "  term             cycles            calls       cyc/call   %share\n";
	for (int i = 0; i < NUM_PROF_TERMS; ++i) {
		double share = 100.0 * (double)g_prof_cycles[i] / (double)base;
		double per_call = g_prof_calls[i] ? (double)g_prof_cycles[i] / (double)g_prof_calls[i] : 0.0;
		const char* tag = (i < PROF_NUM_EXCLUSIVE) ? "" : "  (nested)";
		std::cerr << "  " << PROF_TERM_NAMES[i];
		for (int p = (int)std::strlen(PROF_TERM_NAMES[i]); p < 16; ++p) std::cerr << ' ';
		std::cerr << g_prof_cycles[i] << "  " << g_prof_calls[i]
		          << "  " << (uint64_t)per_call << "  " << share << "%" << tag << "\n";
	}
	std::cerr << "  (base = sum of the " << PROF_NUM_EXCLUSIVE << " exclusive terms)\n";
#else
	(void)label;
#endif
}

void eval_profile_run(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, int reps)
{
#ifdef EVAL_PROFILE
	// The rep loop sits outside every ProfScope, so only the per-term work inside
	// placement_and_piece_eval is timed. A single volatile store after the loop
	// defeats dead-store elimination without timing the loop's own bookkeeping.
	int acc = 0;
	for (int i = 0; i < reps; ++i) {
		acc += placement_and_piece_eval(moveNum, turn, pawnsMask, knightsMask, bishopsMask, rooksMask,
		                                queensMask, kingsMask, occupied_whiteMask, occupied_blackMask, occupiedMask);
	}
	volatile int sink = acc;
	(void)sink;
#else
	(void)moveNum; (void)turn; (void)pawnsMask; (void)knightsMask; (void)bishopsMask; (void)rooksMask;
	(void)queensMask; (void)kingsMask; (void)occupied_whiteMask; (void)occupied_blackMask; (void)occupiedMask; (void)reps;
#endif
}

int eval_profile_num_terms()
{
#ifdef EVAL_PROFILE
	return NUM_PROF_TERMS;
#else
	return 0;
#endif
}

unsigned long long eval_profile_cycles(int term)
{
#ifdef EVAL_PROFILE
	if (term < 0 || term >= NUM_PROF_TERMS) return 0;
	return g_prof_cycles[term];
#else
	(void)term;
	return 0;
#endif
}

unsigned long long eval_profile_calls(int term)
{
#ifdef EVAL_PROFILE
	if (term < 0 || term >= NUM_PROF_TERMS) return 0;
	return g_prof_calls[term];
#else
	(void)term;
	return 0;
#endif
}

const char* eval_profile_name(int term)
{
#ifdef EVAL_PROFILE
	if (term < 0 || term >= NUM_PROF_TERMS) return "";
	return PROF_TERM_NAMES[term];
#else
	(void)term;
	return "";
#endif
}

int eval_profile_num_exclusive()
{
#ifdef EVAL_PROFILE
	return PROF_NUM_EXCLUSIVE;
#else
	return 0;
#endif
}



