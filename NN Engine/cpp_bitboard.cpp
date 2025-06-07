/* cpp_wrapper.cpp

@author: Ranuja Pinnaduwage

This file contains c++ code to emulate the python-chess components for generating legal moves as well as functions for evaluating a position

Code augmented from python-chess: https://github.com/niklasf/python-chess/tree/5826ef5dd1c463654d2479408a7ddf56a91603d6

*/

#include "cpp_bitboard.h"
#include "search_engine.h"
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
std::array<std::array<uint64_t, NUM_SQUARES>, 2> BB_PAWN_ATTACKS;
std::vector<uint64_t> BB_DIAG_MASKS;
std::vector<std::unordered_map<uint64_t, uint64_t>> BB_DIAG_ATTACKS;
std::vector<uint64_t> BB_FILE_MASKS;
std::vector<std::unordered_map<uint64_t, uint64_t>> BB_FILE_ATTACKS;
std::vector<uint64_t> BB_RANK_MASKS;
std::vector<std::unordered_map<uint64_t, uint64_t>> BB_RANK_ATTACKS;
std::vector<std::vector<uint64_t>> BB_RAYS;


// Define global masks for piece placement
uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

// Define global variables for offensive, defensive and piece value scores
int whiteOffensiveScore, blackOffensiveScore, whiteDefensiveScore, blackDefensiveScore;
int blackPieceVal, whitePieceVal;

/*
	Define a set of lookup tables
*/

// Define zobrist table, cache and insertion order for efficient hashing
uint64_t zobristTable[12][64];
uint64_t zobristTurn;

uint64_t castling_hash[4];
uint64_t ep_hash[65];

std::unordered_map<uint64_t, int> evalCache;
std::deque<uint64_t> insertionOrder;

std::unordered_map<uint64_t, std::vector<Move>> moveGenCache;
std::deque<uint64_t> moveGenInsertionOrder;

Move killerMoves[MAX_PLY][2]; // Store up to 2 killer moves per ply

int historyHeuristics[2][64][64];

// Define a heat map for attacks
std::array<std::array<std::array<int, 8>, 8>, 2> attackingLayer;

// Define heat maps for piece placement for both white and black
std::array<std::array<std::array<int, 8>, 8>, 6> whitePlacementLayer = {{
    {{ // Pawns
        {{0,20,7,7,10,15,20,0}},
        {{0,20,5,5,7,15,20,0}},
        {{0,15,15,25,20,35,30,0}},
        {{0,0,20,35,50,35,30,0}},
        {{0,0,20,35,50,35,30,0}},
        {{0,15,15,25,20,35,30,0}},
        {{0,20,5,5,7,15,20,0}},
        {{0,20,7,7,10,15,20,0}}
    }},
    {{ // Knights
        {{0,0,10,15,20,20,15,10}},
        {{0,0,10,15,20,20,15,10}},
        {{0,0,15,20,25,25,20,10}},
        {{0,0,15,20,25,25,20,10}},
        {{0,0,15,20,25,25,20,10}},
        {{0,0,15,20,25,25,20,10}},
        {{0,0,10,15,20,20,15,10}},
        {{0,0,10,15,20,20,15,10}}
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
        {{10,20,40,40,35,20,15,20}},
		{{10,30,60,60,55,25,20,20}},
		{{10,40,65,65,60,20,20,20}},
		{{10,45,65,65,60,25,20,20}},
		{{10,45,65,65,60,25,20,20}},
		{{10,40,65,65,60,20,20,20}},
		{{10,30,60,60,55,20,20,20}},
		{{10,20,40,40,35,20,20,20}}
    }},
    {{ // Kings
        {{0,0,0,0,0,0,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,0,0,0,0,0,0}}
    }}
}};

std::array<std::array<std::array<int, 8>, 8>, 6> blackPlacementLayer = {{
    {{ // Pawns
        {0, 20, 15, 10, 7, 7, 20, 0},
		{0, 20, 15, 7, 5, 5, 20, 0},
		{0, 30, 35, 20, 25, 15, 15, 0},
		{0, 30, 35, 50, 35, 20, 0, 0},
		{0, 30, 35, 50, 35, 20, 0, 0},
		{0, 30, 35, 20, 25, 15, 15, 0},
		{0, 20, 15, 7, 5, 5, 20, 0},
		{0, 20, 15, 10, 7, 7, 20, 0}
    }},
    {{ // Knights
        {{10,15,20,20,15,10,0,0}},
		{{10,15,20,20,15,10,0,0}},
		{{10,20,25,25,20,15,0,0}},
		{{10,20,25,25,20,15,0,0}},
		{{10,20,25,25,20,15,0,0}},
		{{10,20,25,25,20,15,0,0}},
		{{10,15,20,20,15,10,0,0}},
		{{10,15,20,20,15,10,0,0}}
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
        {{20,15,20,35,40,40,20,10}},
		{{20,20,25,55,60,60,30,10}},
		{{20,20,20,60,65,65,40,10}},
		{{20,20,25,60,65,65,45,10}},
		{{20,20,25,60,65,65,45,10}},
		{{20,20,20,60,65,65,40,10}},
		{{20,20,20,55,60,60,30,10}},
		{{20,20,20,35,40,40,20,10}}
    }},
    {{ // Kings
        {{0,0,0,0,0,0,0,0}},
        {{0,0,2,10,10,3,0,0}},
        {{0,0,5,15,15,3,0,0}},
        {{0,0,5,25,20,3,0,0}},
        {{0,0,5,25,20,3,0,0}},
        {{0,0,5,15,15,3,0,0}},
        {{0,0,2,10,10,3,0,0}},
        {{0,0,0,0,0,0,0,0}}
    }}
}};

// Define array to hold the piece type 
std::array<uint8_t, 64> pieceTypeLookUp = {};

std::array<uint64_t, 64> attack_bitmasks = {0ULL};

std::array<int, 64> pressure_white = {0};
std::array<int, 64> support_white = {0};
std::array<int, 64> pressure_black = {0};
std::array<int, 64> support_black = {0};

std::array<int, 64> num_attackers = {0};
std::array<int, 64> num_supporters = {0};

std::array<int, 64> square_values = {0};

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
	
	// Call the function to fill up the tables for all possible queen and rook moves
	attack_table({-9, -7, 7, 9},BB_DIAG_MASKS,BB_DIAG_ATTACKS);
	attack_table({-8, 8},BB_FILE_MASKS,BB_FILE_ATTACKS);
	attack_table({-1, 1},BB_RANK_MASKS,BB_RANK_ATTACKS);
	
	rays(BB_RAYS);
}

void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<std::unordered_map<uint64_t, uint64_t>> &attack_table) {
    
	/*
		Function to initialize attack mask tables for diagonal, file and rank attacks
		
		Parameters:
		- deltas: A vector of position delta values, passed by reference
		- mask_table: An empty vector to hold the sliding attacks masks, passed by reference
		- attack_table: An empty vector to hold the attack subsets of the mask, passed by reference
	*/
	
	// Loop through all squares 
    for (int square = 0; square < 64; ++square) {
        std::unordered_map<uint64_t, uint64_t> attacks;
		
		// Acquire sliding attacks mask for the given deltas
        uint64_t mask = sliding_attacks(square, 0ULL, deltas) & ~edges(square);
        
		// Acquire subsets of attacks mask and loop through them to form the attack table
		std::vector<uint64_t> subsets;
		carry_rippler(mask,subsets);
        for (uint64_t subset : subsets) {
            attacks[subset] = sliding_attacks(square, subset, deltas);
        }

		// Push the current mask and attack tables to the full set
        mask_table.push_back(mask);
        attack_table.push_back(attacks);
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

inline int evaluate_pawns_midgame(uint8_t square){
	// Initialize the evaluation
    int total = 0;
	
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
		total -= whitePlacementLayer[PAWN - 1][x][y];
		
		// Lower white's score for more than one white pawn being on the same file
		total += 200 * (__builtin_popcountll(BB_FILES[x] & (occupied_white & pawns)) > 1);
		
		// Call the function to acquire an extra boost for passed and semi passed pawns
		ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white);
		ppIncrement = std::min(ppIncrement, 400); // cap runaway boosts

		int rank = y;
		//total -= ((rank * 15) * (ppIncrement < 200)) + (((rank * 50) + (rank * rank * 15) + (ppIncrement >> 3)) * (ppIncrement >= 200));
		total -= (default_midgame_pawn_rank_bonus[rank] * (ppIncrement < 200)) + ((passed_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement >= 200));
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_white & pawns;		

		total -= pawn_wall_file_bonus[x]     * (left  != 0); // for pawn on file x-1
		total -= pawn_wall_file_bonus[x + 2] * (right != 0); // for pawn on file x+1
		
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

			if (occupied & (square_mask) & ~kings){				
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (square_mask)));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of white's own position
			total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
			whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			total -= pawn_chain_file_bonus[x] * ((square_mask & occupied_white & pawns)  != 0);
								
			bb &= bb - 1;		
		}			

	}else{
		
		// First add the piece value
        total += values[PAWN];
		blackPieceVal += values[PAWN];
		            
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[PAWN - 1][x][y];
						
		// Lower black's score for more than one black pawn being on the same file						
		total -= 200 * (__builtin_popcountll(BB_FILES[x] & (occupied_black & pawns)) > 1);
		
		ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black);
		ppIncrement = std::min(ppIncrement, 400); // cap runaway boosts
		
		int rank = 7 - y;
		//total += ((rank * 15) * (ppIncrement < 200)) + (((rank * 50) + (rank * rank * 15) + (ppIncrement >> 3)) * (ppIncrement >= 200));
		total += (default_midgame_pawn_rank_bonus[rank] * (ppIncrement < 200)) + ((passed_midgame_pawn_rank_bonus[rank] + (ppIncrement >> 3)) * (ppIncrement >= 200));				
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
				
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_black & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_black & pawns;		

		total += pawn_wall_file_bonus[x]     * (left  != 0); // for pawn on file x-1
		total += pawn_wall_file_bonus[x + 2] * (right != 0); // for pawn on file x+1
		
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

			if (occupied & (square_mask) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y] >> 1;
			blackDefensiveScore += attackingLayer[0][x][y] >> 2;
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			
			// Increase the boost as the attacked pawn is closer to the center files			 
			total += pawn_chain_file_bonus[x] * ((square_mask & occupied_black & pawns)  != 0);

			bb &= bb - 1; 
		}   
	}	
	return total;
}

inline int pawns_simd_initializer(uint64_t bb){
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
			
			ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, square_file, square_rank, occupied_white, occupied_black);
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
			ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, square_file, square_rank, occupied_black, occupied_white);
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
		int result = evaluate_pawns_midgame(r);
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
}

inline int evaluate_knights_midgame(uint8_t square){
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
		
		// Subtract extra value for the existence of a bishop or knight in the midgame		
		total -= 250;	

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);		
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y];   
			total -= attackingLayer[1][x][y];  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 35;
			}
														
			bb &= bb - 1;		
		}
		
	}else{
		// First add the piece value
        total += values[KNIGHT];
		blackPieceVal += values[KNIGHT];
		            
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[KNIGHT - 1][x][y];
		
		// Add extra value for the existence of a bishop or knight in the midgame
		total += 250;

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb); 
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
						
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	
								
			// If each square doesn't contain a black piece, boost the score for mobility
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 35;
			}								
			
			bb &= bb - 1; 
		}            
	}
	return total;
}

inline int evaluate_bishops_midgame(uint8_t square){
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
		
		// Subtract extra value for the existence of a bishop or knight in the midgame            
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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			 			
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y];   
			total -= attackingLayer[1][x][y];  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 20;

				if (bool(~occupied_black & (BB_SQUARES[r]))){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | BB_SQUARES[r];

					uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~occupied & (BB_SQUARES[secondary]))){
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
			
		handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour);

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
			
			total += attackingLayer[0][x][y] >> 2;
							
			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){
				total -= values[xRayPieceType] >> 6;					
			}				
			bb &= bb - 1;
		}            
	}else{
		// First add the piece value
        total += values[BISHOP];
		blackPieceVal += values[BISHOP];
		            
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[BISHOP - 1][x][y];
		
		// Add extra value for the existence of a bishop or knight in the midgame            
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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
						
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y];
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility					
						
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 20;

				if (bool(~occupied_white & (BB_SQUARES[r]))){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | BB_SQUARES[r];

					uint64_t secondAttackMask = BB_DIAG_ATTACKS[r][BB_DIAG_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~occupied & (BB_SQUARES[secondary]))){
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
					
		handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour);

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
	}
	return total;
}

inline int evaluate_rooks_midgame(uint8_t square){
	// Initialize the evaluation
    int total = 0;

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

		// Boost the score if a rook is placed on the 7th Rank
		if (y >= 6){
			rookIncrement += 200;
		}

		if (((BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied]) & (occupied_white & rooks)) != 0){
			rookIncrement += 200;
		} else if (((BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]) & (occupied_white & rooks)) != 0){
			rookIncrement += 150;
		}
		
		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces
		uint64_t r = 0;
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
						
						// If the pawn is within its own (first) half, lower the rook's increment and break the loop
						if (att_square / 8 < 5){
							rookIncrement -= (50 + ((3 - (att_square / 8)) * 125));
							break;
						}
					
					// If a white knight or bishop is in the way, lower the rook increment
					} else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					}
				
				// Check if the occupied piece is black (opposite of the rook)
				}else{
					
					// Check if the piece is the opponents (black) pawn
					if (temp_piece_type == 1){
						
						// If the pawn is within the opponent's (second) half, lower the rook's increment
						if (att_square / 8 > 4){
							rookIncrement -= 50;
						}
					// If a black knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 35;
					// If a black rook is in the way, lower the rook increment
					}else if(temp_piece_type == 4){
						rookIncrement -= 75;
					}
				}
			}
		}
		// Finally use the rook increment
		total -= rookIncrement;

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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
							
			// Subtract the score based on the attack of the opposing position and defense of white's own position				
			total -= attackingLayer[0][x][y];   
			total -= attackingLayer[1][x][y] >> 2;  
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			whiteOffensiveScore += attackingLayer[0][x][y];
			whiteDefensiveScore += attackingLayer[1][x][y];
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 15;

				if (bool(~occupied_black & (BB_SQUARES[r]))){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | BB_SQUARES[r];

					uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~occupied & (BB_SQUARES[secondary]))){
							total -= 10;
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
		
		// Only consider bishop, rook and queens for x-ray attacks
					
		handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour);

		// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,ROOK);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_white;

		// Boost score for semi connected rooks			
		if ((unBlockedMask & (occupied_white & rooks)) != 0){
			total -= 150;
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
			total += attackingLayer[0][x][y] >> 4;
							
			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){					
				total -= values[xRayPieceType] >> 7;					
			}				
			bb &= bb - 1;
		}
	}else{
		// First add the piece value
        total += values[ROOK];
		blackPieceVal += values[ROOK];
		
		// Boost the score if a rook is placed on the 2nd Rank
		if (y <= 1){
			rookIncrement += 200;
		}

		// Boost the score if rooks are connected			
		if (((BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied]) & (occupied_black & rooks)) != 0){
			rookIncrement += 200;
		} else if (((BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]) & (occupied_black & rooks)) != 0){
			rookIncrement += 150;
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
						
						// If the pawn is within the opponent's (first) half, lower the rook's increment
						if (att_square / 8 < 5){
							rookIncrement -= 50;
						}
					// If a white knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 35;
					// If a white rook is in the way, lower the rook increment
					}else if(temp_piece_type == 4){
						rookIncrement -= 75;
					}
				}else{
					
					// Check if the piece is the rook's own pawn
					if (temp_piece_type == 1){
						
						// If the pawn is within its own (second) half, lower the rook's increment and break the loop
						if ((att_square / 8) > 4){
							rookIncrement -= (50 + (((att_square / 8) - 4) * 125));								
							break;
						}
						
					// If a white knight or bishop is in the way, lower the rook increment
					}else if(temp_piece_type == 2 || temp_piece_type == 3){
						rookIncrement -= 15;
					}
				}
			}
		}
		// Finally use the rook increment
		total += rookIncrement;

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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
			total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 2;
			
			// Similar to above, increment the absolute offensive and defensive scores
			// Bit shift to reduce global scores
			blackOffensiveScore += attackingLayer[1][x][y];
			blackDefensiveScore += attackingLayer[0][x][y];	

			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 15;

				if (bool(~occupied_white & (BB_SQUARES[r]))){

					uint64_t simulatedOccupied = (occupied & ~BB_SQUARES[square]) | BB_SQUARES[r];

					uint64_t secondAttackMask = BB_RANK_ATTACKS[r][BB_RANK_MASKS[r] & simulatedOccupied] | BB_FILE_ATTACKS[r][BB_FILE_MASKS[r] & simulatedOccupied];
			
					while (secondAttackMask) {
			
						// Get the position of the least significant set bit of the mask
						uint8_t secondary = __builtin_ctzll(secondAttackMask);		
						
						// If each square doesn't contain a white piece, boost the score for mobility
						if (bool(~occupied & (BB_SQUARES[secondary]))){
							total += 10;
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
				
		handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour);

		// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
		uint64_t unBlockedMask = attacks_mask(colour,occupiedCopy,square,ROOK);
		uint64_t xRayMask = (~pieceAttackMask & unBlockedMask) & ~occupied_black;

		// Boost score for semi connected rooks		
		if ((unBlockedMask & (occupied_black & rooks)) != 0){
			total += 150;
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
			total += attackingLayer[1][x][y] >> 4;			
			
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){				
				total += values[xRayPieceType] >> 7;								
			}
			bb &= bb - 1; 
		}			       
	}
	return total;	
}

inline int evaluate_queens_midgame(uint8_t square){
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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

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
			
			// Remove pieces from the copy of the occupied mask
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 5;
			}								
			bb &= bb - 1;		
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
			
		handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour);

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
			total += attackingLayer[0][x][y] >> 2;			
			
			// If a black piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){				
				total -= values[xRayPieceType] >> 7;				
			}				
			bb &= bb - 1;
		}
		
	}else{
		// First add the piece value
        total += values[QUEEN];
		blackPieceVal += values[QUEEN];
	
		// Add the placement layer for the given piece at that square
		total += blackPlacementLayer[QUEEN - 1][x][y];
		
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
			
			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
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
			
			// Remove pieces from the copy of the occupied mask
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 5;
			}					
			
			bb &= bb - 1; 
		}
		//if (piece_type == 6){std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << " rook increment: " << rookIncrement << std::endl;}
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
		
		handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour);

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
			total += attackingLayer[1][x][y] >> 2;
						
			// If a white piece exists behind the blockers, subtract a reduced piece value
			uint8_t xRayPieceType = pieceTypeLookUp[r]; 
			if (xRayPieceType != 0){				
				total += values[xRayPieceType] >> 7;								
			}
			bb &= bb - 1; 
		}					
	}
	return total;
}

inline int evaluate_kings_midgame(uint8_t square){
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

			if (occupied & (BB_SQUARES[r]) & ~kings){				
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

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
				if (isShielding || isPartialShielding) {
					if (isShielding){
						whiteDefensiveScore += (baseIncrement << 2) + 250;
						total -= (baseIncrement << 2) + 175;
					}else{
						whiteDefensiveScore += (baseIncrement << 1) + 100;
						total -= (baseIncrement << 1) + 75;
					}
									
				} else {
					whiteDefensiveScore += baseIncrement;
					total += baseIncrement >> 2;
				}
			} else {
				if (isShielding) {
					whiteDefensiveScore += baseIncrement;
					total -= baseIncrement;				
				} else {
					whiteDefensiveScore -= baseIncrement << 2;
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

			if (occupied & (BB_SQUARES[r]) & ~kings){
				//if (r == 27){std::cout << (int)piece_type << "  "<< (int)(square) <<std::endl;}
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}
			
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
				if (isShielding || isPartialShielding) {
					if (isShielding){
						blackDefensiveScore += (baseIncrement << 2) + 250;
						total += (baseIncrement << 2) + 175;
					}else{
						blackDefensiveScore += (baseIncrement << 1) + 100;
						total += (baseIncrement << 1) + 75;
					}
									
				} else {
					blackDefensiveScore += baseIncrement;
					total += baseIncrement >> 2;
				}
			} else {
				if (isShielding) {
					blackDefensiveScore += baseIncrement;
					total += baseIncrement;				
				} else {
					blackDefensiveScore -= baseIncrement << 2;
					total -= baseIncrement << 1;
				}
			}
			
			bb &= bb - 1; 
		}

		
	}
	return total;
}


inline int evaluate_pawns_endgame(uint8_t square){
	
	// Initialize the evaluation
    int total = 0;
	
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

		// Lower white's score for more than one white pawn being on the same file
		total += 300 * (__builtin_popcountll(BB_FILES[x] & (occupied_white & pawns)) > 1);
			
		// Call the function to acquire an extra pawn squared based on the position of opposing pawns
		// Only consider this if the pawn is above the 3rd rank
		
		ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white);
		ppIncrement = std::min(ppIncrement, 600); // cap runaway boosts

		int rank = y;
		// Scale with advancement and final increment value
		// int base = rank * 75;
		// int bonus = rank * rank * 15;
		int passed_bonus = ppIncrement >> 2; // dynamic influence

		total -= endgame_pawn_rank_bonus[rank] + passed_bonus;
		
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/
		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_white & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_white & pawns;
		
		total -= 150 * (left != 0);
		total -= 150 * (right != 0);
		
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/
			total -= 200 * ((BB_SQUARES[r] & occupied_white & pawns) != 0);

			bb &= bb - 1;   	
		}
	}else{
		// First subtract the piece value and increment the global white piece value
        total += values[PAWN];
		blackPieceVal += values[PAWN];
		
		// Lower white's score for more than one white pawn being on the same file
		total -= 300 * (__builtin_popcountll(BB_FILES[x] & (occupied_black & pawns)) > 1);
		
		// Call the function to acquire an extra pawn squared based on the position of opposing pawns
		// Only consider this if the pawn is below the 6th rank
		ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black);
		ppIncrement = std::min(ppIncrement, 600); // cap runaway boosts

		int rank = 7 - y;
		// Scale with advancement and final increment value
		//int base = rank * 75;
		//int bonus = rank * rank * 15;
		int passed_bonus = ppIncrement >> 2; // dynamic influence

		//total += base + bonus + passed_bonus;
		total += endgame_pawn_rank_bonus[rank] + passed_bonus;
		
		/*
			This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
		*/		
		uint64_t left = ((BB_SQUARES[square]) >> 1) & ~BB_FILE_H & occupied_black & pawns;
		uint64_t right = ((BB_SQUARES[square]) << 1) & ~BB_FILE_A & occupied_black & pawns;
		
		total += 150 * (left != 0);
		total += 150 * (right != 0);	
		
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, PAWN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			/*
				In this section, award pawn chains where pawns are supporting eachother defensively
			*/											
			total += 200 * ((BB_SQUARES[r] & occupied_black & pawns) != 0);
						
			bb &= bb - 1; 
		}
	}	
	return total;			        
}

inline int evaluate_knights_endgame(uint8_t square){
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
				
		total -= 300;

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
				
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 10;
			}				
			
			bb &= bb - 1;   	
		}
		
	}else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[KNIGHT];
		blackPieceVal += values[KNIGHT];

		total += 300;

		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KNIGHT_ATTACKS[square];
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KNIGHT, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// If each square doesn't contain a black piece, boost the score for mobility	
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 10;
			}
						
			bb &= bb - 1; 
		}
	}
	return total;
}

inline int evaluate_bishops_endgame(uint8_t square){
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
		
		// Boost the scores for the existence of a bishop or knight		
		total -= 350;

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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
							
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 10;
			}
			
			bb &= bb - 1;   	
		}		
			
		handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour);

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
		
		// Boost the scores for the existence of a bishop or knight
        total += 350;		
		
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, BISHOP, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility	
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 10;
			}
						
			bb &= bb - 1; 
		}		
			
		handle_batteries_for_pressure_and_support_tables(square, BISHOP, pieceAttackMask, colour);

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

inline int evaluate_rooks_endgame(uint8_t square){
	// Initialize the evaluation
    int total = 0;
	
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
		
		// Boost the score for the existence of a rook in the endgame
		total -= 500;
		
		// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
		rooks_mask |= BB_FILES[x] & occupied;            
		
		// Loop through the occupied pieces
		uint64_t r = 0;
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
						
						// Increment rook for supporting the white pawn
						rookIncrement += ((att_square / 8) + 1) * 100; 
					
					// Check if the pawn is black
					}else{ 
					
						// Increment rook for blockading black pawn  
						rookIncrement += (8 - (att_square / 8)) * 50; 
					}
				
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
						rookIncrement += (8 - (att_square / 8)) * 50; 
					}
				}
			}
			bb &= bb - 1;   
		}
		
		// Finally use the increment
		total -= rookIncrement;
        		
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 10;
			}
			
			bb &= bb - 1;   	
		}
							
		handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour);

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
		
	// Else the piece is black (positive values for evaluation)
    }else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[ROOK];
		blackPieceVal += values[ROOK];
				
		// Boost the score for the existence of a rook in the endgame
		total += 500;
		
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
					
						// Increment rook for blockading white pawn
						rookIncrement += ((att_square / 8) + 1) * 50;                        
					// If the pawn is black
					}else{ 
						
						// Increment rook for supporting the black pawn
						rookIncrement += (8 - (att_square / 8)) * 100; 
					}
				// Up the board from the black rook
				}else{ 
				
					// If the pawn is white
					if (temp_colour){ 
					
						// Increment rook for attacking white pawn from behind
						rookIncrement += ((att_square / 8) + 1) * 50; 
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
		total += rookIncrement;
		
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, ROOK, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility	
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 10;
			}
						
			bb &= bb - 1; 
		}
		
			
		handle_batteries_for_pressure_and_support_tables(square, ROOK, pieceAttackMask, colour);

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
    }
	//std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << std::endl;
	return total;
}

inline int evaluate_queens_endgame(uint8_t square){
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

		// Boost the score for the existence of a queen in the endgame
		total -= 1000;
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a white piece, boost the score for mobility
			if (bool(~occupied_white & (BB_SQUARES[r]))){
				total -= 10;
			}
			
			bb &= bb - 1;   	
		}
				
		handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour);

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
		// Boost the score for the existence of a queen in the endgame
		total += 1000;
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

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, QUEEN, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
				
			// Remove the piece from the occupied mask copy
			occupiedCopy &= ~(BB_SQUARES[r]);
			
			// If each square doesn't contain a black piece, boost the score for mobility	
			if (bool(~occupied_black & (BB_SQUARES[r]))){
				total += 10;
			}
						
			bb &= bb - 1; 
		}
		
		handle_batteries_for_pressure_and_support_tables(square, QUEEN, pieceAttackMask, colour);

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

inline int evaluate_kings_endgame(uint8_t square){
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

		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = BB_KING_ATTACKS[square];
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);									

			attack_bitmasks[r] |= BB_SQUARES[square];

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
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

			if (occupied & (BB_SQUARES[r]) & ~kings){
				update_pressure_and_support_tables(r, KING, 0, colour, bool(occupied_white & (BB_SQUARES[r])));
			}

			// Get the x and y coordinates for the given square
			y = r >> 3;
            x = r & 7;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
					
			bb &= bb - 1; 
		}
	}
	//std::cout << "Total: " << total << " Type: " << int(piece_type) << " Colour: " << bool(colour) << " x: " << (int)(square & 7) << " y: " << (int)(square >> 3) << std::endl;
	return total;
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

inline int advanced_endgame_eval(int total, bool turn){
	//std::cout << total <<std::endl;
	// Acquire the square positions of each king
	uint8_t whiteKingSquare = __builtin_ctzll(occupied_white&kings);
	uint8_t blackKingSquare = __builtin_ctzll(occupied_black&kings);
	
	// Acquire the separation between the kings
	uint8_t kingSeparation = square_distance(whiteKingSquare,blackKingSquare);
	
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
	//std::cout<< "Inner " << total << std::endl;
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
		
		ppIncrement = getPPIncrement(false, (occupied_white & pawns), 100, file, rank, occupied_white, occupied_black);
		
		kingDist = square_distance(whiteKingSquare, promotionSquare);
		pawnDist = rank;

		kingCanCatch = (turn) ? (kingDist <= pawnDist + 1) : (kingDist <= pawnDist);

		// Defensive penalty for not being able to catch enemy pawn
		blockModifier = 0;
		if (!kingCanCatch) {
			blockModifier = ppIncrement >> 1; // can't stop it, big problem
		} else {
			int diff = (turn) ? (pawnDist + 1 - kingDist) : (pawnDist - kingDist);
			blockModifier = -diff * (ppIncrement >> 3); // the closer we are, the better
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
		
		ppIncrement = getPPIncrement(true, (occupied_black & pawns), 100, file, rank, occupied_black, occupied_white);
		
		kingDist = square_distance(blackKingSquare, promotionSquare);

		pawnDist = 7 - rank;
		kingCanCatch = (!turn) ? (kingDist <= pawnDist + 1) : (kingDist <= pawnDist);

		// Defensive penalty for not being able to catch enemy pawn
		blockModifier = 0;
		if (!kingCanCatch) {
			blockModifier = ppIncrement >> 1; // can't stop it, big problem
		} else {
			int diff = (turn) ? (pawnDist + 1 - kingDist) : (pawnDist - kingDist);
			blockModifier = -diff * (ppIncrement >> 3); // the closer we are, the better
		}
		
		passedBonus = (rank * (ppIncrement + blockModifier)) >> 4;

		total -= blackKing_pawnSeparation * passedBonus;
		//std::cout << total << " | " << ppIncrement << " | " << passedBonus << " | "<< blockModifier << " | " << blackKing_pawnSeparation << " | " << int(r) <<std::endl;
		total -= (7 - whiteKing_pawnSeparation) * passedBonus;
		//std::cout << total <<std::endl;
		bb &= bb - 1;
	}
	//std::cout <<"Inner2 "<< total <<std::endl;
	return total;
	
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

	// Initialize piece attack arrays
	attack_bitmasks.fill(0ULL);

	pressure_white.fill(0);
	support_white.fill(0);
	pressure_black.fill(0);
	support_black.fill(0);

	num_attackers.fill(0);
	num_supporters.fill(0);

	square_values.fill(0);

	horizon_mitigation_flag = false;
	
	// Acquire the number of pieces on the board not including the kings
	int pieceNum = scan_reversed_size(occupied) - 2;
	
	// Determine if the game is at the endgame phase as well as an advanced endgame phase
	bool isEndGame = pieceNum < 16;
	bool isNearGameEnd = pieceNum < 10;
	
	// Call the function to initialize global piece values
	initializePieceValues(occupied);
		
	// If the queens are off the board, then it can be considered an endgame at a higher piece value
	if (queens == 0){
		isEndGame = pieceNum < 18;
		isNearGameEnd = pieceNum < 12;
	}
	
	// If the game is not in endgame phase
	if (!isEndGame){

		// Update the attacking layer based on the position of the king
		setAttackingLayer(5);
	
		/*
		// Loop through the occupied mask
		uint8_t r = 0;
		uint64_t bb = occupied & ~pawnsMask &~knightsMask &~bishopsMask &~rooksMask &~queensMask;
		while (bb) {
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);  // __builtin_ctzll gives the index of the least significant set bit
			//if (r == 34){std::cout << occupied << " y:" << r / 8 << "  x: " << r % 8 << " piece: " << (int)pieceTypeLookUp[r] << std::endl;}
			// Call the midgame evaluation function 
			int result = placement_and_piece_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		*/		

		uint64_t bb = pawnsMask;
		uint8_t r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame pawns evaluation function 
			int result = evaluate_pawns_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		//total += pawns_simd_initializer(bb);

		bb = knightsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_knights_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = bishopsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_bishops_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = rooksMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_rooks_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = queensMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_queens_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = kingsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_kings_midgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}
		

		adjust_pressure_and_support_tables_for_pins(occupied & ~kings);
		//std::cout << total << std::endl;
		//total += get_pressure_increment(lastMovedToSquare, occupied & ~kings, turn);
		total += approximate_capture_gains(occupied & ~kings, turn);
		//std::cout << total << std::endl;
		
		// Boost the score for the side with more piece value proportional to how many pieces are on the board
		//std::cout << total << " black:" << blackPieceVal << "  white: " << whitePieceVal  << " diff: " << (int)(((whitePieceVal - blackPieceVal)/ (1.0 * whitePieceVal)) * 10000) << std::endl;
		if (blackPieceVal > whitePieceVal){
			total += (int)(((blackPieceVal - whitePieceVal)/ (1.0 * blackPieceVal)) * 10000);		
		}else if (whitePieceVal > blackPieceVal){			
			total -= (int)(((whitePieceVal - blackPieceVal)/ (1.0 * whitePieceVal)) * 10000);
		}
	
	// Else the game is in endgame phase	
	}else{
		
		// Update the attacking layer based on the position of the king
		setAttackingLayer(5);
		/*
		// Loop through the occupied mask
		uint8_t r = 0;
		uint64_t bb = occupied & ~pawnsMask &~knightsMask &~bishopsMask &~rooksMask;
		while (bb) {
			
			// Get the position of the least significant set bit of the mask
			r = __builtin_ctzll(bb);

			// Call the endgame evaluation function 
			int result = placement_and_piece_endgame(r);
			square_values[r] = abs(result);
			total += result;
			bb &= bb - 1;			
		} 
		*/
		

		uint64_t bb = pawnsMask;
		uint8_t r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame pawns evaluation function 
			int result = evaluate_pawns_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = knightsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_knights_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = bishopsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_bishops_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = rooksMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_rooks_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = queensMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_queens_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		bb = kingsMask;
		r = 0;
		while (bb) {
			
			r = __builtin_ctzll(bb);  
			
			// Call the midgame knights evaluation function 
			int result = evaluate_kings_endgame(r);
			square_values[r] = abs(result);
			total += result;
			
			// Clear the least significant set bit
			bb &= bb - 1;  
		}

		//std::cout<< total<< std::endl;


		adjust_pressure_and_support_tables_for_pins(occupied & ~kings);
		//total += get_pressure_increment(lastMovedToSquare, occupied & ~kings, turn);
		total += approximate_capture_gains(occupied & ~kings, turn);
		//std::cout<< total<< std::endl;
		// Boost the score for the side with more piece value proportional to how many pieces are on the board
		if (blackPieceVal > whitePieceVal){
			total += (int)(((blackPieceVal - whitePieceVal)/ (1.0 * blackPieceVal)) * 15000);		
		}else if (whitePieceVal > blackPieceVal){			
			total -= (int)(((whitePieceVal - blackPieceVal)/ (1.0 * whitePieceVal)) * 15000);
		}
		//std::cout<< total<< std::endl;
		// Check if the position is an advanced endgame
		if (isNearGameEnd){			
			total = advanced_endgame_eval(total, turn);
		}
		//std::cout<< total<< std::endl;
	}
	
	/*
		In this code section, boost both white and blacks score based on the existence of bishop and knight pairs
	*/
	if (scan_reversed_size(occupied_white&bishops) == 2){
		total -= 350;
	}
	if (scan_reversed_size(occupied_white&knights) == 2){
		total -= 250;
	}
	if (scan_reversed_size(occupied_black&bishops) == 2){
		total += 350;
	}
	if (scan_reversed_size(occupied_black&knights) == 2){
		total += 250;
	}
	
	
	/*
		In this code section, boost the scores of both sides based on how poor the side's defense is relative to the opponent's offense
	*/
	//std::cout << occupied << " black:" << blackOffensiveScore << "  white: " << whiteDefensiveScore  << " diff: " << ((blackOffensiveScore - std::max(whiteDefensiveScore, 0))/18) * 100 << std::endl;
	//std::cout << occupied << " black:" << blackDefensiveScore << "  white: " << whiteOffensiveScore  << " diff: " << ((whiteOffensiveScore - std::max(blackDefensiveScore, 0))/18) * 100 << std::endl;
	if (whiteOffensiveScore > blackDefensiveScore){
		total -= ((whiteOffensiveScore - std::max(blackDefensiveScore, 0))/18) * 100;
	}
	
	if (blackOffensiveScore > whiteDefensiveScore){
		total += ((blackOffensiveScore - std::max(whiteDefensiveScore, 0))/18) * 100;
	}

	//std::cout<< total<< std::endl;
	//if (total == -288)
	//std::cout << "AAAA: " << approximate_capture_gains1(occupied & ~kings, turn) << " occupied: " << (occupied & ~kings) << " turn: " << turn <<  std::endl;
	//std::cout << total << " " << whiteOffensiveScore << " " << whiteDefensiveScore << " " <<  blackOffensiveScore << " " << blackDefensiveScore << " " <<std::endl;
	return total;
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

inline CaptureInfo* find_last_viable_capture(std::vector<CaptureInfo>& captures, uint64_t& white_pieces, uint64_t& black_pieces, bool captureColour) {
	
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

inline std::optional<CaptureInfo> find_and_pop_last_viable_capture(std::vector<CaptureInfo>& captures, uint64_t white_pieces, uint64_t black_pieces, bool captureColour) {
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
	uint64_t pieceAttackMask = attacks_mask(target_colour,occupied,target_square,pieceTypeLookUp[target_square]);

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

inline int approximate_capture_gains(uint64_t bb, bool turn) {
    int black_gains = 0;
    int white_gains = 0;

	std::vector<CaptureInfo> white_captures;
	std::vector<CaptureInfo> black_captures;

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

		std::vector<CaptureInfo>& own_captures = current_turn ? white_captures : black_captures;
		std::vector<CaptureInfo>& opp_captures = current_turn ? black_captures : white_captures;

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

inline uint8_t piece_type_at(uint8_t square){
    
	/*
		Function to get the piece type
		
		Parameters:
		- square: The square to be analyzed		
		
		Returns:
		A unsigned char representing the piece type
	*/
    
	/* 
		In this section, see if the individual piece type masks bitwise Anded with the square exists
		If so this suggests that the given square has the piece type of that mask
	*/
	uint64_t mask = (BB_SQUARES[square]);

	if (pawns & mask) {
		return 1;
	} else if (knights & mask){
		return 2;
	} else if (bishops & mask){
		return 3;
	} else if (rooks & mask){
		return 4;
	} else if (queens & mask){
		return 5;
	} else if (kings & mask){
		return 6;
	} else{
		return 0;
	}
}

inline void setAttackingLayer(int increment){
	
	/*
		Function to update the attacking layer relative to the king's positions
		
		Parameters:
		- square: The increment to be used to boost the required squares
		
		Returns:
		A unsigned char representing the piece type
	*/
	
	// Set the default attacking layer
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
	
	// Acquire the number of pieces on the board excluding the kings
	int pieceNum = scan_reversed_size(occupied) - 2;
	
	// Determine if the position is in endgame based on the number 
	bool isEndGame = pieceNum < 16;
	if (queens == 0){
		isEndGame = pieceNum < 18;
	}
	
	// Set variable for squares being open near the king
	bool squareOpen = true;

	bool pawnShield = false;
	
	// Set the multiplier for open square boosts
	int multiplier = 5;
	
	// Define the x and y coordinates for each square
	uint8_t x,y;
	
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
				if (squareOpen){
					if ((occupied_white & (BB_SQUARES[r_inner])) == 0){
						attackingLayer[1][x][y] += increment * multiplier;
					}
				} else if (pawnShield){
					attackingLayer[1][x][y] -= increment >> 1;
				}
				
			}
			bb_inner &= bb_inner - 1;
		}
		
		bb &= bb - 1;
	}
	
	// Loop through the squares around the black king
	r = 0;
	bb = attacks_mask(false,0ULL,63 - __builtin_clzll(occupied_black&kings),6);
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
				if (squareOpen){
					if ((occupied_black & (BB_SQUARES[r_inner])) == 0){
						attackingLayer[0][x][y] += increment * multiplier;
					}
				} else if (pawnShield){
					attackingLayer[0][x][y] -= increment >> 1;
				}
				
			}
			
			bb_inner &= bb_inner - 1;
		}		
		bb &= bb - 1;
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

inline int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces) {
	
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
		ppIncrement -= 125;
		bb &= bb - 1;
	}
	//if (y == 4 && x == 2){std::cout << bitmask  << " " << ppIncrement << " " << incrementCopy << std::endl;}
	// The minimum increment is 0
	if (ppIncrement < 0) {
		return 0;	
	// Otherwise check if the increment does not suffer a decrement, this suggests the pawn is a passed pawn
	} else if (ppIncrement == incrementCopy){
		
		// Check if there exists a non-pawn blocker infront of the passed pawn
		if ((infrontMask & (opposingPieces | curSidePieces)) != 0){
			
			// If the piece is the that of the opponent, decrement the score as it is blockaded
			if ((infrontMask & opposingPieces) != 0){
				ppIncrement -= 100;
			}
		// Else no blocker exists, the passed pawn is un-impeded, earning a larger boost
		} else{
			ppIncrement += 50;			
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
				ppIncrement += 75;
				if ((left & occupied_black) == 0){
					ppIncrement += 150;
				}
			}

			if (se != 0){
				ppIncrement += 75;
				if ((right & occupied_black) == 0){
					ppIncrement += 150;
				}
			}

			if ((left & occupied_white) != 0){
					ppIncrement += 225;
			}

			if ((right & occupied_white) != 0){
					ppIncrement += 225;
			}				
		} else {
			uint64_t nw = (pawnBB << 7) & ~BB_FILE_H & pawns & occupied_black;
			uint64_t ne = (pawnBB << 9) & ~BB_FILE_A & pawns & occupied_black;
			
			if (nw != 0){
				ppIncrement += 75;
				if ((left & occupied_white) == 0){
					ppIncrement += 150;
				}
			}

			if (ne != 0){
				ppIncrement += 75;
				if ((right & occupied_white) == 0){
					ppIncrement += 150;
				}
			}

			if ((left & occupied_black) != 0){
					ppIncrement += 225;
			}

			if ((right & occupied_black) != 0){
					ppIncrement += 225;
			}	
		}
	}
	//std::cout << bitmask << " | " << ppIncrement << " | " << infrontMask << " | "<< incrementCopy << " x: " << (int)x << " y: " << (int)y << std::endl;
	return ppIncrement;
}



uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co){
    
	/*
		Function to generate a mask containing attacks from the given square
		
		Parameters:
		- colour: The colour of the opposing side
		- square: The starting square
		- occupied: The mask containing all pieces
		- queens_and_rooks: The mask containing queens and rooks
		- queens_and_bishops: The mask containing queens and bishops
		- kings: The mask containing only kings
		- knights: The mask containing only knights
		- pawns: The mask containing only pawns
		- occupied_co: The mask containing the pieces of the opposing side
		
		Returns:
		A mask containing attacks from the given square
	*/
	
	// Acquire the masks of the pieces on the same rank, file and diagonal as the given square
	uint64_t rank_pieces = BB_RANK_MASKS[square] & occupied;
    uint64_t file_pieces = BB_FILE_MASKS[square] & occupied;
    uint64_t diag_pieces = BB_DIAG_MASKS[square] & occupied;

	// Acquire all attack masks for each piece type
    uint64_t attackers = (
        (BB_KING_ATTACKS[square] & kings) |
        (BB_KNIGHT_ATTACKS[square] & knights) |
        (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
        (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
        (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[!colour][square] & pawns));

	// Perform a bitwise and with the opposing pieces 
    return attackers & occupied_co;
}

uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied){    
    
	/*
		Function to generate a mask that represents check blocking pieces
		
		Parameters:
		- king: The square of the current side's king
		- queens_and_rooks: The mask containing queens and rooks
		- queens_and_bishops: The mask containing queens and bishops
		- occupied_co_opp: The mask containing only pieces from the opposition
		- occupied_co: The mask containing only pieces from the current side
		- occupied_co: The mask containing the pieces of the opposing side
		
		Returns:
		A mask containing check blockers
	*/
	
	// Define a mask for possible sliding attacks on the king square
    uint64_t snipers = ((BB_RANK_ATTACKS[king][0] & queens_and_rooks) |
                (BB_FILE_ATTACKS[king][0] & queens_and_rooks) |
                (BB_DIAG_ATTACKS[king][0] & queens_and_bishops));

	// Define a mask containing possible squares to block the attack
    uint64_t blockers = 0;
	
	
	// Loop through the mask containing sniper attackers
	uint8_t r = 0;
	uint64_t bb = snipers & occupied_co_opp;
	while (bb) {
		r = __builtin_ctzll(bb);
		
		// Update the blockers where there is exactly one blocker per attack 
		uint64_t b = betweenPieces(king, r) & occupied;        
        if (b && (1ULL << (63 - __builtin_clzll(b)) == b)){
            blockers |= b;
		}
		bb &= bb - 1;
	}
	
	// Return the blockers where the square contains a piece of the current side
    return blockers & occupied_co;
}
	
uint64_t betweenPieces(uint8_t a, uint8_t b){
		
	/*
		Function to get a mask of the squares between a and b
		
		Parameters:
		- a: The starting square
		- b: The ending square
		
		Returns:
		A mask containing the squares between a and b
	*/
	
	// Use BB_RAYS to get a mask of all square in the same ray as a and back
	// Use (~0x0ULL << a) ^ (~0x0ULL << b) to get all square between a and b in counting order
    uint64_t bb = BB_RAYS[a][b] & ((~0x0ULL << a) ^ (~0x0ULL << b));
	
	// Remove a and b themselves
    return bb & (bb - 1);
}

uint64_t ray(uint8_t a, uint8_t b){return BB_RAYS[a][b];}

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

bool is_into_check(uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, 
	               uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn){
	
	uint64_t king_mask = kingsMask & ourPieces;
	uint8_t king = 63 - __builtin_clzll(king_mask);

	uint64_t checkers = attackersMask(
				!turn,
				king,
				occupiedMask,
				queensMask | rooksMask,
				queensMask | bishopsMask,
				kingsMask,
				knightsMask,
				pawnsMask,
				opposingPieces
			);

	std::vector<uint8_t> startPos;
    std::vector<uint8_t> endPos;
    std::vector<uint8_t> promotions;

    generateEvasions(startPos, endPos, promotions,
                     0, king, checkers, BB_SQUARES[from_square], BB_SQUARES[to_square],
                     occupiedMask, occupiedWhite, opposingPieces, ourPieces,
                     pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask,
                     ep_square, turn);

    bool found = false;
		for (size_t i = 0; i < startPos.size(); ++i) {
			if (startPos[i] == from_square && endPos[i] == to_square) {
				found = true;
				break;
			}
		}

	if (!found)
		return true; // move is not a legal evasion

	uint64_t blockers = slider_blockers(king, queensMask | rooksMask, queensMask | bishopsMask, opposingPieces, ourPieces, occupiedMask);

	return !is_safe(king, blockers, from_square, to_square, occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask,
				knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, ep_square, turn);
}

bool is_safe(uint8_t king, uint64_t blockers, uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
			 uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn) {
    if (from_square == king) {
        if (is_castling(from_square, to_square, turn, ourPieces, rooksMask, kingsMask)) {
            return true;
        } else {

            return attackersMask(
				!turn,
				to_square,
				occupiedMask,
				queensMask | rooksMask,
				queensMask | bishopsMask,
				kingsMask,
				knightsMask,
				pawnsMask,
				opposingPieces
			) == 0;
        }
    }
    else if (is_en_passant(from_square, to_square, ep_square, occupiedMask, pawnsMask)) {
        return (pin_mask(turn, from_square, king, occupiedMask, opposingPieces, bishopsMask, rooksMask, queensMask) & BB_SQUARES[to_square]) &&
               !ep_skewered(king, from_square, ep_square, turn, occupiedMask, opposingPieces, bishopsMask, rooksMask, queensMask);
    }
    else {
        return (!(blockers & BB_SQUARES[from_square])) ||
               (ray(from_square, to_square) & BB_SQUARES[king]);
    }
}

bool is_castling(uint8_t from_square, uint8_t to_square, bool turn, uint64_t ourPieces, uint64_t rooksMask, uint64_t kingsMask) {

    if (kingsMask & BB_SQUARES[from_square]) {
        int diff = (from_square & 7) - (to_square & 7);
        return (std::abs(diff) > 1) ||
               ((rooksMask & ourPieces) & BB_SQUARES[to_square]);
    }
    return false;
}

bool is_en_passant(uint8_t from_square, uint8_t to_square, int ep_square, uint64_t occupiedMask, uint64_t pawnsMask) {
    // Check if the target square is the en passant square
    if (ep_square != to_square)
        return false;

    // Check if the moving piece is a pawn at the from-square
    if (!(pawnsMask & BB_SQUARES[from_square]))
        return false;

    // Check if the move is a diagonal (7 or 9 square offset)
    int diff = std::abs(to_square - from_square);
    if (diff != 7 && diff != 9)
        return false;

    // Check that the to-square is not actually occupied
    if (occupiedMask & BB_SQUARES[to_square])
        return false;

    return true;
}

uint64_t pin_mask(bool colour, int square, uint8_t king, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask) {
    uint64_t square_mask = BB_SQUARES[square];

    // File pin check (↑ ↓)
    uint64_t rays = BB_FILE_ATTACKS[king][0];
    if (rays & square_mask) {
        uint64_t snipers = rays & (rooksMask | queensMask) & opposingPieces;
        while (snipers) {
            int sniper = 63 - __builtin_clzll(snipers);
            snipers ^= BB_SQUARES[sniper];
            if ((betweenPieces(sniper, king) & (occupiedMask | square_mask)) == square_mask)
                return ray(king, sniper);
        }
    }

    // Rank pin check (← →)
    rays = BB_RANK_ATTACKS[king][0];
    if (rays & square_mask) {
        uint64_t snipers = rays & (rooksMask | queensMask) & opposingPieces;
        while (snipers) {
            int sniper = 63 - __builtin_clzll(snipers);
            snipers ^= BB_SQUARES[sniper];
            if ((betweenPieces(sniper, king) & (occupiedMask | square_mask)) == square_mask)
                return ray(king, sniper);
        }
    }

    // Diagonal pin check (↗ ↙ ↖ ↘)
    rays = BB_DIAG_ATTACKS[king][0];
    if (rays & square_mask) {
        uint64_t snipers = rays & (bishopsMask | queensMask) & opposingPieces;
        while (snipers) {
            int sniper = 63 - __builtin_clzll(snipers);
            snipers ^= BB_SQUARES[sniper];
            if ((betweenPieces(sniper, king) & (occupiedMask | square_mask)) == square_mask)
                return ray(king, sniper);
        }
    }

    return ~0ULL;  // Not pinned
}

bool ep_skewered(int king, int capturer, int ep_square, bool turn, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask) {
    // Assumes:
    // - ep_square is an int (en passant square index, or -1 if not set)
    // - turn is a bool or enum (true for white, false for black)
    // - BB_SQUARES is a constexpr array of 64 bitboards (1ULL << square)
    // - BB_RANK_ATTACKS and BB_DIAG_ATTACKS are lookup tables of attacks
    // - BB_RANK_MASKS and BB_DIAG_MASKS are masks for those directions
    // - All bitboards (occupied, rooks, queens, bishops, occupied_co, etc.) are globally defined

    // ep_square must be valid
	if (ep_square == -1)
	    return false;

    // Compute square of the pawn that moved two steps
    int last_double = ep_square + (turn? -8 : 8);

    // Reconstruct the hypothetical board: remove captured pawn and capturer,
    // add the en passant square (as if the pawn moved into it)
    uint64_t new_occupancy = (occupiedMask & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer]) | BB_SQUARES[ep_square];

    // Horizontal (rank) skewer detection
    uint64_t horizontal_attackers = opposingPieces & (rooksMask | queensMask);
    if (BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & new_occupancy] & horizontal_attackers)
        return true;

    // Diagonal skewer detection (technically impossible, but checked anyway)
    uint64_t diagonal_attackers = opposingPieces & (bishopsMask | queensMask);
    if (BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & new_occupancy] & diagonal_attackers)
        return true;

    return false;
}

bool attackedForKing(bool opponent_color,uint64_t path, uint64_t occupied, uint64_t opposingPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask){
    
	while (path) {
        // Get least significant bit index (square)
        uint8_t sq = __builtin_ctzll(path);  // GCC/Clang builtin: count trailing zeros

        // Clear the LSB from path
        path &= path - 1;

        // Call attackersMask for this square
        uint64_t attackers =  attackersMask(
				opponent_color,
				sq,
				occupied,
				queensMask | rooksMask,
				queensMask | bishopsMask,
				kingsMask,
				knightsMask,
				pawnsMask,
				opposingPieces
			);

        if (attackers != 0) {
            return true;  // square is attacked by opponent
        }
    }

    return false;  // no square in path attacked
}


/*
	Set of functions used as utilities for all above functions
*/
void scan_reversed(uint64_t bb, std::vector<uint8_t> &result){	
		
	/*
		Function to acquire a vector of set bits in a given bitmask starting from the top of the board
		
		Parameters:
		- bb: The bitmask to be scanned
		- result: An empty vector to hold the set bits, passed by reference
	*/
	
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        result.push_back(r);
        bb ^= (BB_SQUARES[r]);		
    }    
}

bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool is_en_passant){
    	
	/*
		Function to determine if a move is a capture
		
		Parameters:
		- from_square: The starting square
		- to_square: The ending square
		- occupied_co: A mask containing occupied pieces of the opposing side
		- is_en_passant: A boolean determining whether the move is an en passent
		
		Returns:
		A mask containing the squares between a and b
	*/
	
	uint64_t touched = (1ULL << from_square) ^ (1ULL << to_square);
	return bool(touched & occupied_co) || is_en_passant;
}

bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces){
		
	/*
		Function to determine if a move is a check
		
		Parameters:
		- colour: The colour of the current side
		- occupied: The mask containing all pieces
		- queens_and_rooks: The mask containing queens and rooks
		- queens_and_bishops: The mask containing queens and bishops
		- kings: The mask containing only kings
		- knights: The mask containing only knights
		- pawns: The mask containing only pawns
		- opposingPieces: The mask containing the pieces of the opposing side
		
		Returns:
		A boolean defining whether the move is a check
	*/
	
    uint8_t kingSquare = 63 - __builtin_clzll(~opposingPieces&kings);
	return (bool)(attackersMask(!colour, kingSquare, occupied, queens_and_rooks, queens_and_bishops, kings, knights, pawns, opposingPieces));
}

bool is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn){

	if (!is_check(turn, occupiedMask, (queensMask | rooksMask), (queensMask | bishopsMask), kingsMask, knightsMask, pawnsMask, opposingPieces))
		return false;
	
	std::vector<uint8_t> startPos;
	std::vector<uint8_t> endPos;
	std::vector<uint8_t> promotions;

	generateLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, ~0ULL, ~0ULL,
	 				   occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
					   rooksMask, queensMask, kingsMask, ep_square, turn);
	//std::cout << occupiedMask <<  " | " << ourPieces <<  " | " << startPos.size() << std::endl;
	if (startPos.size() == 0)
		return true;

	return false;

}

bool is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn){

	if (is_check(turn, occupiedMask, (queensMask | rooksMask), (queensMask | bishopsMask), kingsMask, knightsMask, pawnsMask, opposingPieces))
		return false;
	
	std::vector<uint8_t> startPos;
	std::vector<uint8_t> endPos;
	std::vector<uint8_t> promotions;

	generateLegalMoves(startPos, endPos, promotions, preliminary_castling_mask, ~0ULL, ~0ULL,
	 				   occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
					   rooksMask, queensMask, kingsMask, ep_square, turn);

	//std::cout << occupiedMask <<  " | " << ourPieces <<  " | " << startPos.size() << std::endl;
	if (startPos.size() == 0)
		return true;

	return false;

}

void scan_forward(uint64_t bb, std::vector<uint8_t> &result) {
		
	/*
		Function to acquire a vector of set bits in a given bitmask starting from the bottom of the board
		
		Parameters:
		- bb: The bitmask to be scanned
		- result: An empty vector to hold the set bits, passed by reference
	*/
	
	uint64_t r = 0;
    while (bb) {
        r = __builtin_ctzll(bb);
        result.push_back(r);
        bb &= bb - 1;
    }
}

uint8_t scan_reversed_size(uint64_t bb) {
		
	/*
		Function to acquires the number of set bits in a bitmask
		
		Parameters:
		- bb: The bitmask to be examined
	*/
	
    return __builtin_popcountll(bb);
}

uint8_t square_distance(uint8_t sq1, uint8_t sq2) {
		
	/*
		Function to acquire the Chebyshev distance (king's distance)
		
		Parameters:
		- sq1: The starting square
		- sq2: The ending square
		
		Returns:
		The distance that a king would have to travel to travel the squares
	*/
		
    int file_distance = abs((sq1 & 7) - (sq2 & 7));
    int rank_distance = abs((sq1 >> 3) - (sq2 >> 3));
    return std::max(file_distance, rank_distance);
}

inline uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType){	
	
	/*
		Function to acquire an attack mask for a given piece on a given square
		
		Parameters:
		- colour: The colour of the current side
		- occupied: The mask containing all pieces
		- square: The current square to be analyzed
		- pieceType: The piece type on the given square
		
		Returns:
		A boolean defining whether the move is a check
	*/
	
	switch (pieceType) {
		case 1: // Pawn
			return BB_PAWN_ATTACKS[colour][square];
		case 2: // Knight
			return BB_KNIGHT_ATTACKS[square];
		case 6: // King
			return BB_KING_ATTACKS[square];
		case 3: // Bishop
			return BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		case 4: // Rook
			return BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] |
				   BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		case 5: // Queen (bishop + rook)
			return BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied] |
				   BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] |
				   BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied];
		default:
			return 0ULL; // no attacks for unknown piece types
	}
}

int remove_piece_at(uint8_t square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted){
	
	uint8_t piece_type = 0;
	uint64_t mask = BB_SQUARES[square];

	if (pawnsMask & mask) {
		piece_type = 1;
		pawnsMask ^= mask;
	} else if (knightsMask & mask){
		piece_type = 2;
		knightsMask ^= mask;
	} else if (bishopsMask & mask){
		piece_type = 3;
		bishopsMask ^= mask;
	} else if (rooksMask & mask){
		piece_type = 4;
		rooksMask ^= mask;
	} else if (queensMask & mask){
		piece_type = 5;
		queensMask ^= mask;
	} else if (kingsMask & mask){
		piece_type = 6;
		kingsMask ^= mask;
	} else{
		return 0;
	}

	occupiedMask ^= mask;
	occupiedWhite &= ~mask;
	occupiedBlack &= ~mask;

	promoted &= ~mask;

	return piece_type;
}

void set_piece_at(uint8_t square, uint8_t piece_type, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, bool promotedFlag, bool turn){
	
	remove_piece_at(square, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted);

	uint64_t mask = BB_SQUARES[square];

	if (piece_type == 1) {
		pawnsMask |= mask;
	} else if (piece_type == 2){
		knightsMask |= mask;
	} else if (piece_type == 3){
		bishopsMask |= mask;
	} else if (piece_type == 4){
		rooksMask |= mask;
	} else if (piece_type == 5){
		queensMask |= mask;
	} else if (piece_type == 6){
		kingsMask |= mask;
	} else{
		return;
	}

	occupiedMask ^= mask;
	
	if (turn){
		occupiedWhite ^= mask;
	} else{
		occupiedBlack ^= mask;
	}

	if (!promotedFlag)
		return;
	promoted ^= mask;

}
        
void update_state(uint8_t to_square, uint8_t from_square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, uint64_t& castling_rights, int& ep_square, int promotion_type, bool turn){
	uint64_t from_bb = BB_SQUARES[from_square];
    uint64_t to_bb = BB_SQUARES[to_square];

    bool promotedFlag = bool(promoted & from_bb);

	
    int piece_type = remove_piece_at(from_square, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
									 kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted);

	if (piece_type == 0) {
		std::ostringstream oss;
		oss << "push() expects move to be pseudo-legal, but got move from: " << (int)from_square << " to: " << (int)to_square << " in " << create_fen(pawnsMask, knightsMask, bishopsMask, rooksMask,
																																					  queensMask, kingsMask, occupiedMask, occupiedWhite, occupiedBlack,
																																					  promoted, castling_rights, ep_square, turn);
		throw std::runtime_error(oss.str());
	}

	uint8_t capture_square = to_square;
	uint8_t captured_piece_type = 0;
	uint64_t mask = (BB_SQUARES[capture_square]);
		
	if (pawnsMask & mask) {
		captured_piece_type = 1;
	} else if (knightsMask & mask){
		captured_piece_type = 2;
	} else if (bishopsMask & mask){
		captured_piece_type = 3;
	} else if (rooksMask & mask){
		captured_piece_type = 4;
	} else if (queensMask & mask){
		captured_piece_type = 5;
	} else if (kingsMask & mask){
		captured_piece_type = 6;
	}

	castling_rights &= ~to_bb & ~from_bb;

	if (piece_type == 6 && !promotedFlag){
		if (turn)
			castling_rights &= ~BB_RANK_1;
		else
			castling_rights &= ~BB_RANK_8;
		
	} else if(captured_piece_type == 6 && (promoted & to_bb) == 0){
		if (turn && (to_square & 7) == 7)
			castling_rights &= ~BB_RANK_8;
		else if(!turn && (to_square & 7) == 0)
			castling_rights &= ~BB_RANK_1;
	}


    if (piece_type == 1){
		
		int ep_copy = ep_square;
		ep_square = -1;
		
		int diff = to_square - from_square;

		if(diff == 16 && (from_square >> 3) == 1)
			ep_square = from_square + 8;
		else if(diff == -16 && (from_square >> 3) == 6)
			ep_square = from_square - 8;
		else if (to_square == ep_copy && (std::abs(diff) == 7 || std::abs(diff) == 9) && captured_piece_type == 0){
			
			int down = turn ? -8 : 8;
            capture_square = to_square + down;
        
            captured_piece_type = remove_piece_at(capture_square, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
									 kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted);
		}
	}

	if (promotion_type != 1){
		promotedFlag = true;
		piece_type = promotion_type;
	}

	bool castling = false;
	if(piece_type == 6){
		if (turn)
			castling = (from_square == 4 && (to_square == 6 || to_square == 2));
		else
			castling = (from_square == 60 && (to_square == 62 || to_square == 58));
	}

	if (castling){
		bool a_side = (to_square & 7) < (from_square & 7);

		if (a_side){  
			remove_piece_at(turn ? 0 : 56, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
							kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted);
            
            set_piece_at(turn ? 2 : 58, 6, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
						kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted, promotedFlag, turn);
            
			set_piece_at(turn ? 3 : 59, 4, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
						kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted, promotedFlag, turn);
			
		}else{
			remove_piece_at(turn ? 7 : 63, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
							kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted);
            
            set_piece_at(turn ? 6 : 62, 6, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
						kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted, promotedFlag, turn);
            
			set_piece_at(turn ? 5 : 61, 4, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
						kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted, promotedFlag, turn);
		}
	} else{
		set_piece_at(to_square, piece_type, pawnsMask, knightsMask, bishopsMask, rooksMask, queensMask, 
						kingsMask, occupiedMask, occupiedWhite, occupiedBlack, promoted, promotedFlag, turn);
	}
}

std::string create_fen(uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask,
                       uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask,
                       uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack,
                       uint64_t& promoted, uint64_t& castling_rights, int& ep_square, bool turn) {
    char board[64] = {};

    for (int sq = 0; sq < 64; ++sq) {
        uint64_t mask = 1ULL << sq;
        if (!(occupiedMask & mask)) continue;

        bool is_white = (occupiedWhite & mask) != 0;
        char piece = '?';

        if (pawnsMask & mask)   piece = 'p';
        else if (knightsMask & mask) piece = 'n';
        else if (bishopsMask & mask) piece = 'b';
        else if (rooksMask & mask)   piece = 'r';
        else if (queensMask & mask)  piece = 'q';
        else if (kingsMask & mask)   piece = 'k';

        if (is_white) piece = std::toupper(piece);
        board[sq] = piece;
    }

    std::ostringstream fen;

    // Board layout
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            if (board[sq] == 0) {
                ++empty;
            } else {
                if (empty) {
                    fen << empty;
                    empty = 0;
                }
                fen << board[sq];
            }
        }
        if (empty) fen << empty;
        if (rank > 0) fen << '/';
    }

    // Active color
    fen << ' ' << (turn ? 'w' : 'b');

    // Castling rights
    std::string castling;
    if (castling_rights & (1ULL << 7))  castling += 'K'; // White kingside
    if (castling_rights & (1ULL << 0))  castling += 'Q'; // White queenside
    if (castling_rights & (1ULL << 63)) castling += 'k'; // Black kingside
    if (castling_rights & (1ULL << 56)) castling += 'q'; // Black queenside
    if (castling.empty()) castling = "-";
    fen << ' ' << castling;

    // En passant target square
    if (ep_square == -1) {
        fen << " -";
    } else {
        char file = 'a' + (ep_square % 8);
        char rank = '1' + (ep_square / 8);
        fen << ' ' << file << rank;
    }

    // Halfmove clock and fullmove number (defaults)
    fen << " 0 1";

    return fen.str();
}

