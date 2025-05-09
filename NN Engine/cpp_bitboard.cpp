/* cpp_wrapper.cpp

@author: Ranuja Pinnaduwage

This file contains c++ code to emulate the python-chess components for generating legal moves as well as functions for evaluating a position

Code augmented from python-chess: https://github.com/niklasf/python-chess/tree/5826ef5dd1c463654d2479408a7ddf56a91603d6

*/

#include "cpp_bitboard.h"
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

constexpr int NUM_SQUARES = 64;

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

// Define zobrist table, cache and insertion order for efficient hashing
uint64_t zobristTable[12][64];
std::unordered_map<uint64_t, int> moveCache;
std::deque<uint64_t> insertionOrder;

// Define caches for player move generation orders
std::unordered_map<uint64_t, std::string> OpponentMoveGenCache;
std::deque<uint64_t> OpponentMoveGenInsertionOrder;

std::unordered_map<uint64_t, std::string> curPlayerMoveGenCache;
std::deque<uint64_t> curPlayerMoveGenInsertionOrder;

// Define the file bitboards
constexpr uint64_t BB_FILE_A = 0x0101010101010101ULL << 0;
constexpr uint64_t BB_FILE_B = 0x0101010101010101ULL << 1;
constexpr uint64_t BB_FILE_C = 0x0101010101010101ULL << 2;
constexpr uint64_t BB_FILE_D = 0x0101010101010101ULL << 3;
constexpr uint64_t BB_FILE_E = 0x0101010101010101ULL << 4;
constexpr uint64_t BB_FILE_F = 0x0101010101010101ULL << 5;
constexpr uint64_t BB_FILE_G = 0x0101010101010101ULL << 6;
constexpr uint64_t BB_FILE_H = 0x0101010101010101ULL << 7;

// Array of file uint64_ts
constexpr std::array<uint64_t, 8> BB_FILES = {
    BB_FILE_A, BB_FILE_B, BB_FILE_C, BB_FILE_D, BB_FILE_E, BB_FILE_F, BB_FILE_G, BB_FILE_H
};

// Define the rank uint64_ts
constexpr uint64_t BB_RANK_1 = 0xffULL << (8 * 0);
constexpr uint64_t BB_RANK_2 = 0xffULL << (8 * 1);
constexpr uint64_t BB_RANK_3 = 0xffULL << (8 * 2);
constexpr uint64_t BB_RANK_4 = 0xffULL << (8 * 3);
constexpr uint64_t BB_RANK_5 = 0xffULL << (8 * 4);
constexpr uint64_t BB_RANK_6 = 0xffULL << (8 * 5);
constexpr uint64_t BB_RANK_7 = 0xffULL << (8 * 6);
constexpr uint64_t BB_RANK_8 = 0xffULL << (8 * 7);

// Array of rank bitboards
constexpr std::array<uint64_t, 8> BB_RANKS = {
    BB_RANK_1, BB_RANK_2, BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6, BB_RANK_7, BB_RANK_8
};

// Array of piece values
constexpr std::array<int, 7> values = {0, 1000, 3250, 3450, 5000, 10000, 12000};

// Define a heat map for attacks
std::array<std::array<std::array<int, 8>, 8>, 2> attackingLayer;

// Define heat maps for piece placement for both white and black
std::array<std::array<std::array<int, 8>, 8>, 6> whitePlacementLayer = {{
    {{ // Pawns
        {{0,0,10,15,20,20,25,0}},
        {{0,0,10,15,20,20,25,0}},
        {{0,0,15,25,30,35,30,0}},
        {{0,0,15,30,45,35,30,0}},
        {{0,0,15,30,45,35,30,0}},
        {{0,0,15,25,20,35,30,0}},
        {{0,0,10,15,20,20,25,0}},
        {{0,0,10,15,20,20,25,0}}
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
        {{0,25,20,20,15,10,0,0}},
		{{0,25,20,20,15,10,0,0}},
		{{0,30,35,30,25,15,0,0}},
		{{0,30,35,45,30,15,0,0}},
		{{0,30,35,45,30,15,0,0}},
		{{0,30,35,20,25,15,0,0}},
		{{0,25,20,20,15,10,0,0}},
		{{0,25,20,20,15,10,0,0}}
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


std::unordered_map<uint64_t, uint8_t> initialize_bit_to_square() {
    std::unordered_map<uint64_t, uint8_t> bit_to_square;
    for (uint8_t i = 0; i < 64; ++i) {
        bit_to_square[1ULL << i] = i;
    }
    return bit_to_square;
}

std::unordered_map<uint64_t, uint8_t> bit_to_square = initialize_bit_to_square();

// Define array to hold the piece type 
std::array<uint8_t, 64> pieceTypeLookUp = {};

// Define global masks for piece placement
uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

// Define global variables for offensive, defensive and piece value scores
int whiteOffensiveScore, blackOffensiveScore, whiteDefensiveScore, blackDefensiveScore;
int blackPieceVal, whitePieceVal;


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
	
    uint64_t rank_mask = (0xFFULL | 0xFF00000000000000ULL) & ~(0xFFULL << (8 * (square / 8)));
    uint64_t file_mask = (0x0101010101010101ULL | 0x8080808080808080ULL) & ~(0x0101010101010101ULL << square % 8);
    return rank_mask | file_mask;
}

/*
	Set of functions directly used to evaluate the position
*/
int placement_and_piece_midgame(uint8_t square){
    
	/*
		Function to acquire a midgame evaluation
		
		Parameters:
		- square: The starting square		
		
		Returns:
		A position evaluation
	*/
	
	// Initialize the evaluation
    int total = 0;
    
	// Define the maximum increment for rook-file positioning and attack mask
	int rookIncrement = 300;
    uint64_t rooks_mask = 0ULL;
	
	// Initialize the maximum increment for pawn placement
	int ppIncrement = 200;
    
	// Acquire the piece type and colour
	uint8_t piece_type = pieceTypeLookUp [square];
	bool colour = bool(occupied_white & (1ULL << square)); 
    
	// Acquire the x and y coordinates of the given square
    uint8_t y = square / 8;
    uint8_t x = square % 8;
    
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value
        total -= values[piece_type];
        whitePieceVal += values[piece_type];
		
		// Check if the piece is not a rook or king
        if (! (piece_type == 4 || piece_type == 6)){
            
			// Subtract the placement layer for the given piece at that square
            total -= whitePlacementLayer[piece_type - 1][x][y];
            
			// Subtract extra value for the existence of a bishop or knight in the midgame
            if (piece_type == 2 || piece_type == 3){
                total -= 375;
            }
			
			// Check if the piece is a pawn
            if (piece_type == 1){
                
				// Lower white's score for more than one white pawn being on the same file
                if (scan_reversed_size((BB_FILES[x] & (occupied_white & pawns))) > 1) {                
                    total += 300;
				}
				
				// Call the function to acquire an extra pawn squared based on the position of opposing pawns
				ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white);
				total -= ppIncrement;
				
				// If the pawn is a passed pawn, boost the score, otherwise give a basic score for how far up the board it is
				if (ppIncrement >= 200) {
					total -= (y + 1) * 50 + ((y + 1) * (y + 1)) * 10;
				} else {
					total -= (y + 1) * 75;
				}
				
				/*
					This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
				*/
				
				uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
				uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
				
				// If the left square exists (original pawn not on the A file)
				if (left != 0){
					
					// Append score if the pawn has another pawn of the same colour to its left
					// Append scores based on how close the left-side pawn is to the center files
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
					if (attackedPieceType == 1 && bool(occupied_white & left)){
						if ((left & BB_FILE_D & BB_FILE_E) != 0){
							total -= 70;
						} else if ((left & BB_FILE_C & BB_FILE_F) != 0){
							total -= 60;
						} else if ((left & BB_FILE_B & BB_FILE_G) != 0){
							total -= 50;
						} else{
							total -= 40;
						}
						
					}
				}
				
				// If the right square exists (original pawn not on the H file)
				if (right != 0){
					
					// Append score if the pawn has another pawn of the same colour to its rught
					// Append scores based on how close the right-side pawn is to the center files
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
					if (attackedPieceType == 1 && bool(occupied_white & right)){												
						if ((right & BB_FILE_D & BB_FILE_E) != 0){
							total -= 70;
						} else if ((right & BB_FILE_C & BB_FILE_F) != 0){
							total -= 60;
						} else if ((right & BB_FILE_B & BB_FILE_G) != 0){
							total -= 50;
						} else{
							total -= 40;
						}
					}
				}				                
			}
		// Check if the piece is a rook
        }else if (piece_type == 4){  
            
			// Boost the score if a rook is placed on the 7th Rank
            if (y == 6){
                rookIncrement -= 100;
			}
			
			// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
            rooks_mask |= BB_FILES[x] & occupied;            
            
			// Loop through the occupied pieces
			uint64_t r = 0;
			uint64_t bb = rooks_mask;
			while (bb) {
				
				// Get the current square as the max bit of the current mask
				r = bb &-bb;							
				uint8_t att_square = 64 - __builtin_clzll(r) - 1;
				bb ^= r;
				
				// Check if the attacked square is up the board from the rook
                if (att_square > square){
					
					// Get the piece type and colour
                    uint8_t temp_piece_type = pieceTypeLookUp [att_square];
					bool temp_colour = bool(occupied_white & (1ULL << att_square));
										
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
        }
        
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;									

			// Get the x and y coordinates for the given square
			y = r / 8;
            x = r % 8;
			
			// Check if the piece is a queen
            if (piece_type == 5){
				
				// Subtract the score based on the attack of the opposing position and defense of white's own position
				// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
                total -= attackingLayer[0][x][y] >> 2;				
				total -= attackingLayer[1][x][y] >> 3;
				
				// Similar to above, increment the absolute offensive and defensive scores
				// Bit shift to reduce global scores
				whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
				whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
				
				// Remove pieces from the copy of the occupied mask
				occupiedCopy &= ~(1ULL << r);
				
				// If each square doesn't contain a white piece, boost the score for mobility
				if (bool(~occupied_white & (1ULL << r))){
					total -= 5;
				}					
			
			// Check if the piece is a pawn
            }else if (piece_type == 1){
				
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
				
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << r))){
					
					// Increase the boost as the attacked pawn is closer to the center files
					if (((1ULL << r) & BB_FILE_D & BB_FILE_E) != 0){
						total -= 70;
					} else if (((1ULL << r) & BB_FILE_C & BB_FILE_F) != 0){
						total -= 60;
					} else if (((1ULL << r) & BB_FILE_B & BB_FILE_G) != 0){
						total -= 50;
					} else{
						total -= 40;
					}
				}
					
			// Check if the piece is a king	
			}else if (piece_type == 6){
				
				// Subtract the score based on the attack of the opposing position and absolute offensive score				
				total -= attackingLayer[0][x][y];   
				whiteOffensiveScore += attackingLayer[0][x][y];
				
				// Boost the king score for having the protection of its own pawns
				// Otherwise keep the local and global defensive score as normal
				if (pieceTypeLookUp[r] == 1 && y > (square / 8)){
					whiteDefensiveScore += (attackingLayer[1][x][y] << 1) + 50;
					total -= (attackingLayer[1][x][y] << 1) + 100;
				}else{
					whiteDefensiveScore += attackingLayer[1][x][y];
					total -= attackingLayer[1][x][y] >> 2;  
				}
			
			// Else the piece is either a knight, bishop or rook
			}else{    
			
				// Subtract the score based on the attack of the opposing position and defense of white's own position				
                total -= attackingLayer[0][x][y];   
				total -= attackingLayer[1][x][y];  
				
				// Similar to above, increment the absolute offensive and defensive scores
				// Bit shift to reduce global scores
				whiteOffensiveScore += attackingLayer[0][x][y];
				whiteDefensiveScore += attackingLayer[1][x][y];
				
				// Check if the piece is a bishop
				if (piece_type == 3){
					
					// Remove the piece from the occupied mask copy
					occupiedCopy &= ~(1ULL << r);
					
					// If each square doesn't contain a white piece, boost the score for mobility
					if (bool(~occupied_white & (1ULL << r))){
						total -= 15;
					}
				
				// Check if the piece is a rook
				} else if (piece_type == 4){
					
					// Remove the piece from the occupied mask copy
					occupiedCopy &= ~(1ULL << r);
					
					// If each square doesn't contain a white piece, boost the score for mobility
					if (bool(~occupied_white & (1ULL << r))){
						total -= 20;
					}
				
				// Else the piece is a knight
				} else{					
					
					// If each square doesn't contain a white piece, boost the score for mobility
					if (bool(~occupied_white & (1ULL << r))){
						total -= 25;
					}
				}								
			}
			bb ^= (1ULL << r);			
		}
		
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
		
		// Only consider bishop, rook and queens for x-ray attacks
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			
			// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_white;
			
			// Loop through the attacks mask
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;									

				// Get the x and y coordinates for the given square
				y = r / 8;
				x = r % 8;
				
				// Subtract a reduced score for square attacks behind a piece
				total -= attackingLayer[0][x][y] << 2;
				
				// If a black piece exists behind the blockers, subtract a reduced piece value
				uint8_t xRayPieceType = pieceTypeLookUp[r]; 
				if (xRayPieceType != 0){
					if (piece_type == 5){
						total -= values[xRayPieceType] >> 7;
					} else{
						total -= values[xRayPieceType] >> 6;
					}
				}				
				bb ^= (1ULL << r);
			}
		}
	
	// If the piece is black (add positive values for evaluation)
    }else{
		
		// First add the piece value
        total += values[piece_type];
		blackPieceVal += values[piece_type];
		
		// Check if the piece is not a rook or king
        if (! (piece_type == 4 || piece_type == 6)){
            
			// Add the placement layer for the given piece at that square
            total += blackPlacementLayer[piece_type - 1][x][y];
            
			// Add extra value for the existence of a bishop or knight in the midgame
            if (piece_type == 2 || piece_type == 3){
                total += 375;
            }
			
			// Check if the piece is a pawn
            if (piece_type == 1){
                
				// Lower black's score for more than one black pawn being on the same file				
                if (scan_reversed_size((BB_FILES[x] & (occupied_black & pawns))) > 1){                
                    total -= 300;
                }
				
				// Call the function to acquire an extra pawn squared based on the position of opposing pawns
				ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black);
				total += ppIncrement;
          
				// If the pawn is a passed pawn, boost the score, otherwise give a basic score for how far up the board it is
				if (ppIncrement >= 200){
					total += (8 - y) * 50 + ((8 - y) * (8 - y)) * 10;				
				}else{
					total += (8 - y) * 75;     
				}
				
				/*
					This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
				*/
				
				uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
				uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;				
				
				// If the left square exists (original pawn not on the A file)
				if (left != 0){
					
					// Append score if the pawn has another pawn of the same colour to its left
					// Append scores based on how close the left-side pawn is to the center files
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
					if (attackedPieceType == 1 && !bool(occupied_white & left)){
						if ((left & BB_FILE_D & BB_FILE_E) != 0){
							total += 70;
						} else if ((left & BB_FILE_C & BB_FILE_F) != 0){
							total += 60;
						} else if ((left & BB_FILE_B & BB_FILE_G) != 0){
							total += 50;
						} else{
							total += 40;
						}						
					}
				}
				
				// If the right square exists (original pawn not on the H file)
				if (right != 0){
					
					// Append score if the pawn has another pawn of the same colour to its rught
					// Append scores based on how close the right-side pawn is to the center files
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
					if (attackedPieceType == 1 && !bool(occupied_white & right)){												
						if ((right & BB_FILE_D & BB_FILE_E) != 0){
							total += 70;
						} else if ((right & BB_FILE_C & BB_FILE_F) != 0){
							total += 60;
						} else if ((right & BB_FILE_B & BB_FILE_G) != 0){
							total += 50;
						} else{
							total += 40;
						}
					}
				}                
			}
        // Check if the piece is a rook   
        }else if (piece_type == 4){
			
			// Boost the score if a rook is placed on the 2nd Rank
            if (y == 1){
                rookIncrement += 100;
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
				bb ^= (1ULL << r);
				
				// Check if the attacked square is down the board from the rook
                if (att_square < square){
					
					// Get the piece type and colour
                    uint8_t temp_piece_type = pieceTypeLookUp [att_square];
					bool temp_colour = bool(occupied_white & (1ULL << att_square));
					
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
        }
		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
        uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;		
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;
			
			// Get the x and y coordinates for the given square
			y = r / 8;
            x = r % 8;
			
			// Check if the piece is a queen
            if (piece_type == 5){
				
				// Add the score based on the attack of the opposing position and defense of black's own position
				// Adjust the score by bit shifting heavily so that the queen's ability to attack many squares isn't overrated
                total += attackingLayer[1][x][y] >> 2;
				total += attackingLayer[0][x][y] >> 3;
				
				// Similar to above, increment the absolute offensive and defensive scores
				// Bit shift to reduce global scores
				blackOffensiveScore += attackingLayer[1][x][y] >> 1;
				blackDefensiveScore += attackingLayer[0][x][y] >> 2;				
				
				// Remove pieces from the copy of the occupied mask
				occupiedCopy &= ~(1ULL << r);
				
				// If each square doesn't contain a black piece, boost the score for mobility
				if (bool(~occupied_black & (1ULL << r))){
					total += 5;
				}					
				
			// Check if the piece is a pawn
            }else if (piece_type == 1){
				
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
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << r))){
					if (((1ULL << r) & BB_FILE_D & BB_FILE_E) != 0){
						total += 70;
					} else if (((1ULL << r) & BB_FILE_C & BB_FILE_F) != 0){
						total += 60;
					} else if (((1ULL << r) & BB_FILE_B & BB_FILE_G) != 0){
						total += 50;
					} else{
						total += 40;
					}
				}
									
			// Check if the piece is a king	
			}else if (piece_type == 6){
				
				// Subtract the score based on the attack of the opposing position and absolute offensive score		
				total += attackingLayer[1][x][y];
				blackOffensiveScore += attackingLayer[1][x][y];

				// Boost the king score for having the protection of its own pawns
				// Otherwise keep the local and global defensive score as normal								
				if (pieceTypeLookUp[r] == 1 && y < (square / 8)){
					blackDefensiveScore += (attackingLayer[0][x][y] << 1) + 50;
					total += (attackingLayer[0][x][y] << 1) + 100;
				}else{
					blackDefensiveScore += attackingLayer[0][x][y];
					total += attackingLayer[0][x][y] >> 2;
				}
				
			// Else the piece is either a knight, bishop or rook
			}else{    
			
				// Subtract the score based on the attack of the opposing position and defense of black's own position
                total += attackingLayer[1][x][y];
				total += attackingLayer[0][x][y];
				
				// Similar to above, increment the absolute offensive and defensive scores
				// Bit shift to reduce global scores
				blackOffensiveScore += attackingLayer[1][x][y];
				blackDefensiveScore += attackingLayer[0][x][y];	

				// Check if the piece is a bishop
				if (piece_type == 3){
					
					// Remove the piece from the occupied mask copy
					occupiedCopy &= ~(1ULL << r);
					
					// If each square doesn't contain a black piece, boost the score for mobility					
					if (bool(~occupied_black & (1ULL << r))){
						total += 15;
					}
				// Check if the piece is a rook
				} else if (piece_type == 4){
					
					// Remove the piece from the occupied mask copy
					occupiedCopy &= ~(1ULL << r);
					
					// If each square doesn't contain a black piece, boost the score for mobility
					if (bool(~occupied_black & (1ULL << r))){
						total += 20;
					}
				// Else the piece is a knight
				} else{
					
					// If each square doesn't contain a black piece, boost the score for mobility
					if (bool(~occupied_black & (1ULL << r))){
						total += 25;
					}					
				}				
			}
			bb ^= (1ULL << r);
		}
		
		/*
			In this section, the scores for x-ray attacks are acquired
		*/
		
		// Only consider bishop, rook and queens for x-ray attacks
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			
			// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_black;
			
			// Loop through the attacks mask
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;									

				// Get the x and y coordinates for the given square
				y = r / 8;
				x = r % 8;
				
				// Subtract a reduced score for square attacks behind a piece
				total += attackingLayer[1][x][y] << 2;
				
				// If a white piece exists behind the blockers, subtract a reduced piece value
				uint8_t xRayPieceType = pieceTypeLookUp[r]; 
				if (xRayPieceType != 0){
					if (piece_type == 5){
						total += values[xRayPieceType] >> 7;
					} else{
						total += values[xRayPieceType] >> 6;
					}					
				}
				bb ^= (1ULL << r);
			}			
		}
    }
	//std::cout << total << " " << int(piece_type) << " " << bool(colour) << " " << rookIncrement  << std::endl;
	return total;
}

int placement_and_piece_endgame(uint8_t square){
    
	/*
		Function to acquire a endgame evaluation
		
		Parameters:
		- square: The starting square		
		
		Returns:
		A position evaluation
	*/
	
	// Initialize the evaluation
    int total = 0;
	
	// Define the maximum increment for rook-file positioning and attack mask
    int rookIncrement = 100;
	uint64_t rooks_mask = 0ULL;
	
	// Initialize the maximum increment for pawn placement
    int ppIncrement = 400;	
    
	// Acquire the piece type and colour
	uint8_t piece_type = pieceTypeLookUp [square];
	bool colour = bool(occupied_white & (1ULL << square));     

	// Acquire the x and y coordinates of the given square
    uint8_t y = square / 8;	
    uint8_t x = square % 8;
        
	// If the piece is white (add negative values for evaluation)
    if (colour) {
		
		// First subtract the piece value and increment the global white piece value
        total -= values[piece_type];		
        whitePieceVal += values[piece_type];
		
		// Check if the piece is a pawn
		if (piece_type == 1){
						                            
			// Lower white's score for more than one white pawn being on the same file
			if (scan_reversed_size((BB_FILES[x] & (occupied_white & pawns))) > 1) {                
				total += 300;
			}
				
			// Call the function to acquire an extra pawn squared based on the position of opposing pawns
			// Only consider this if the pawn is above the 3rd rank
			if (y > 2){
				ppIncrement = getPPIncrement(colour, (occupied_black & pawns), ppIncrement, x, y, occupied_black, occupied_white);
			} else{
				ppIncrement = 0;
			}
			
			total -= ppIncrement;
			
			// If the pawn is a passed pawn, boost the score, otherwise give a basic score for how far up the board it is and how blocked its path is			
			if (ppIncrement == 400) {
				total -= (y + 1) * 100 + ((y + 1) * (y + 1)) * 15;
			} else if (ppIncrement == 500) {
				total -= (y + 1) * 150 + ((y + 1) * (y + 1)) * 15;
			} else if (ppIncrement == 600) {
				total -= (y + 1) * 200 + ((y + 1) * (y + 1)) * 15;
			} else {
				total -= (y + 1) * 50;
			}
			
			/*
				This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
			*/
			
			uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
			uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
			
			// If the left square exists (original pawn not on the A file)
			if (left != 0){
					
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
				if (attackedPieceType == 1 && bool(occupied_white & left)){
					total -= 150;
				}
			}
			
			// If the right square exists (original pawn not on the H file)
			if (right != 0){
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
				if (attackedPieceType == 1 && bool(occupied_white & right)){
					total -= 150;
				}
			}
		
		// Check if the piece is a rook
		}else if (piece_type == 4){

			// Boost the score for the existence of a rook in the endgame
            total -= 500;
            
			// Aqcuire the rooks mask as all occupied pieces on the same file as the rook
            rooks_mask |= BB_FILES[x] & occupied;            
            
			// Loop through the occupied pieces
			uint64_t r = 0;
			uint64_t bb = rooks_mask;
			while (bb) {
				
				// Get the current square as the max bit of the current mask
				r = bb &-bb;							
				uint8_t att_square = 64 - __builtin_clzll(r) - 1;
				
				// Get the piece type and colour
				uint8_t temp_piece_type = pieceTypeLookUp [att_square];
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                
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
				bb ^= r;
			}
			
			// Finally use the increment
			total -= rookIncrement;
        
		// Boost the scores for the existence of a bishop or knight
		}else if (piece_type == 3){  
			total -= 350;
		} else if (piece_type == 3){
			total -= 300;
		}
		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;
		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;									

			// Get the x and y coordinates for the given square
			y = r / 8;
            x = r % 8;
			
			// Subtract the score based on the attack of the opposing position and defense of white's own position
            total -= attackingLayer[0][x][y];
			total -= attackingLayer[1][x][y] >> 1;			
			
			// Check if the piece is a bishop, rook or queen
			if (piece_type == 3 || piece_type == 4 || piece_type == 5){
				
				// Remove the piece from the occupied mask copy
				occupiedCopy &= ~(1ULL << r);
				
				// If each square doesn't contain a white piece, boost the score for mobility
				if (bool(~occupied_white & (1ULL << r))){
					total -= 10;
				}
			
			// Check if the piece is a knight
			}else if (piece_type == 2){
				
				// If each square doesn't contain a white piece, boost the score for mobility
				if (bool(~occupied_white & (1ULL << r))){
					total -= 10;
				}
				
			// Check if the piece is a pawn
			} else if (piece_type == 1){
				
				/*
					In this section, award pawn chains where pawns are supporting eachother defensively
				*/
				
				// Increase the boost as the attacked pawn is closer to the center files				
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << r))){
					total -= 150;
				}
			}
			bb ^= (1ULL << r);	
		}
		
		// Check if the piece is a bishop, rook or queen
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			
			// Create an attack mask that consists of the attack on non-white pieces that would occur behind the blocking piece
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_white;
			
			// Loop through the attacks mask
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;									

				// Get the x and y coordinates for the given square
				y = r / 8;
				x = r % 8;
				
				// Subtract a reduced score for square attacks behind a piece				
				total -= attackingLayer[0][x][y] << 2;
				
				// If a white piece exists behind the blockers, subtract a reduced piece value
				uint8_t xRayPieceType = pieceTypeLookUp[r]; 
				if (xRayPieceType != 0){
					total -= values[xRayPieceType] >> 6;
				}
				bb ^= (1ULL << r);	
			}
		}
	// Else the piece is black (positive values for evaluation)
    }else{
		
		// First subtract the piece value and increment the global white piece value
        total += values[piece_type];
		blackPieceVal += values[piece_type];
		
		// Check if the piece is a pawn
        if (piece_type == 1){
			
			// Lower white's score for more than one white pawn being on the same file
			if (scan_reversed_size((BB_FILES[x] & (occupied_black & pawns))) > 1){                
				total -= 300;
			}
			
			// Call the function to acquire an extra pawn squared based on the position of opposing pawns
			// Only consider this if the pawn is below the 6th rank
			if (y < 5){
                ppIncrement = getPPIncrement(colour, (occupied_white & pawns), ppIncrement, x, y, occupied_white, occupied_black);
            }else{
                ppIncrement = 0;
			}
            total += ppIncrement;
          
			// If the pawn is a passed pawn, boost the score, otherwise give a basic score for how far up the board it is and how blocked its path is
			if (ppIncrement == 400) {
				total += (8 - y) * 100 + ((8 - y) * (8 - y)) * 15;	
			} else if (ppIncrement == 500) {
				total += (8 - y) * 150 + ((8 - y) * (8 - y)) * 15;	
			} else if (ppIncrement == 600) {
				total += (8 - y) * 200 + ((8 - y) * (8 - y)) * 15;	
			} else {
				total += (8 - y) * 50;  
			}
			
			/*
				This section acquires the squares to the left and right of a given pawn, accounting for wrap arounds
			*/
			
			uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
			uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
			
			// If the left square exists (original pawn not on the A file)
			if (left != 0){
				
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << left))){
					total += 150;
				}
			}
			
			// If the right square exists (original pawn not on the H file)
			if (right != 0){
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << right))){
					total += 150;
				}
			}
		// Check if the piece is a rook
		}else if (piece_type == 4){
			
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
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                
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
				bb ^= (1ULL << r);			
			}	

			// Finally use the increment
            total += rookIncrement;
		
		// Boost the scores for the existence of a bishop or knight
        }else if (piece_type == 3){  
			total += 350;
		} else if (piece_type == 2){
			total += 300;
		}
		
		/*
			In this section, the scores for piece attacks are acquired
		*/
		
		// Acquire the attacks mask for the current piece and make a copy of the occupied mask
		uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;
        		
		// Loop through the attacks mask
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;									

			// Get the x and y coordinates for the given square
			y = r / 8;
            x = r % 8;
			
			// Subtract the score based on the attack of the opposing position and defense of black's own position
            total += attackingLayer[1][x][y];
			total += attackingLayer[0][x][y] >> 1;
			
			// Check if the piece is a bishop, rook or queen
			if (piece_type == 3 || piece_type == 4 || piece_type == 5){
				
				// Remove the piece from the occupied mask copy
				occupiedCopy &= ~(1ULL << r);
				
				// If each square doesn't contain a black piece, boost the score for mobility	
				if (bool(~occupied_black & (1ULL << r))){
					total += 10;
				}
				
			// Check if the piece is a knight
			} else if (piece_type == 2){
				
				// If each square doesn't contain a black piece, boost the score for mobility	
				if (bool(~occupied_black & (1ULL << r))){
					total += 10;
				}
				
			// Check if the piece is a pawn
			} else if (piece_type == 1){
				
				/*
					In this section, award pawn chains where pawns are supporting eachother defensively
				*/
				
				// Increase the boost as the attacked pawn is closer to the center files
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << r))){
					total += 150;
				}
			}			
			bb ^= (1ULL << r);	
		}
		
		// Check if the piece is a bishop, rook or queen
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			
			// Create an attack mask that consists of the attack on non-black pieces that would occur behind the blocking piece
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_black;
			
			// Loop through the attacks mask
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;									

				// Get the x and y coordinates for the given square
				y = r / 8;
				x = r % 8;
				
				// Subtract a reduced score for square attacks behind a piece
				total += attackingLayer[1][x][y] << 2;
				
				// If a white piece exists behind the blockers, subtract a reduced piece value
				uint8_t xRayPieceType = pieceTypeLookUp[r]; 
				if (xRayPieceType != 0){
					total += values[xRayPieceType] >> 6;
				}
				bb ^= (1ULL << r);	
			}
		}
    }
	
	return total;
}

int placement_and_piece_eval(int moveNum, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t prevKingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask){
	
	/*
		Function to acquire a positional evaluation
		
		Parameters:
		- moveNum: The current move number
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
				
		// Loop through the occupied mask
		uint8_t r = 0;
		uint64_t bb = occupied;
		while (bb) {
			
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;		
						
			// Call the midgame evaluation function 
			total += placement_and_piece_midgame(r);
			bb ^= (1ULL << r);			
		}

		// Boost the score for the side with more piece vallue proportional to how many pieces are on the board
		if (blackPieceVal > whitePieceVal){
			total += (int)(((blackPieceVal - whitePieceVal)/ blackPieceVal) * 2000);		
		}else if (whitePieceVal > blackPieceVal){			
			total -= (int)(((whitePieceVal - blackPieceVal)/ whitePieceVal) * 2000);
		}
	
	// Else the game is in endgame phase	
	}else{
		
		// Update the attacking layer based on the position of the king
		setAttackingLayer(5);
		
		// Loop through the occupied mask
		uint8_t r = 0;
		uint64_t bb = occupied;
		while (bb) {
			
			// Get the position of the most significant set bit of the mask
			r = 64 - __builtin_clzll(bb) - 1;	

			// Call the endgame evaluation function 
			total += placement_and_piece_endgame(r);
			bb ^= (1ULL << r);			
		} 
		
		// Boost the score for the side with more piece vallue proportional to how many pieces are on the board
		if (blackPieceVal > whitePieceVal){
			total += (int)(((blackPieceVal - whitePieceVal)/ blackPieceVal) * 2000);		
		}else if (whitePieceVal > blackPieceVal){			
			total -= (int)(((whitePieceVal - blackPieceVal)/ whitePieceVal) * 2000);
		}
		
		// Check if the position is an advanced endgame
		if (isNearGameEnd){
			
			// Acquire the square positions of each king
			uint8_t whiteKingSquare = 63 - __builtin_clzll(occupied_white&kings);
			uint8_t blackKingSquare = 63 - __builtin_clzll(occupied_black&kings);
			
			// Acquire the separation between the kings
			uint8_t kingSeparation = square_distance(63 - __builtin_clzll(occupied_white&kings),63 - __builtin_clzll(occupied_black&kings));
			
			// Check if the black side has a 2000 point advantage or greater
			if (total > 2000){
				
				// Increment for having the black king closer to the white king
				total += (7-kingSeparation)*200;
							
				// Get the x and y coordinates for the white king
				uint8_t x = whiteKingSquare % 8;
				uint8_t y = whiteKingSquare / 8;
				
				/*
					In this code section, lower white's score if it's king is closer to the board's edge
				*/
				if (x >= 4){
					if (y >= 4){
						total += (x + y) * 75;
					}else{
						total += (x + (7 - y)) * 75;
					}					
				} else{
					if (y >= 4){
						total += ((7 - x) + y) * 75;
					}else{
						total += ((7 - x) + (7 - y)) * 75;
					}					
				}
				
			// Check if the white side has a 2000 point advantage or greater	
			}else if (total < -2000){
				
				// Increment for having the black king closer to the white king
				total -= (7-kingSeparation)*200;
				
				// Get the x and y coordinates for the black king
				uint8_t x = blackKingSquare % 8;
				uint8_t y = blackKingSquare / 8;
				
				/*
					In this code section, lower white's score if it's king is closer to the board's edge
				*/
				if (x >= 4){
					if (y >= 4){
						total -= (x + y) * 75;
					}else{
						total -= (x + (7 - y)) * 75;
					}					
				} else{
					if (y >= 4){
						total -= ((7 - x) + y) * 75;
					}else{
						total -= ((7 - x) + (7 - y)) * 75;
					}					
				}
			}
			
			// Create bitmasks for the first and second half of the board
			uint64_t firstHalf = BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4;
			uint64_t secondHalf = BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8;
			
			// Define variables for average separation between each king and each coloured pawns
			int averageBlackKing_blackPawnSeperation = 0;
			int averageWhiteKing_whitePawnSeperation = 0;
			int averageBlackKing_whitePawnSeperation = 0;
			int averageWhiteKing_blackPawnSeperation = 0;
			
			// Loop through the mask containing black pawns in the first half
			uint8_t r = 0;
			uint64_t bb = firstHalf & occupied_black & pawns;
			uint8_t size = 0;
			while (bb) {
				
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;		

				// Add all the distances between each king and the black pawns
				averageBlackKing_blackPawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_black&kings));
				averageWhiteKing_blackPawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_white&kings));
				bb ^= (1ULL << r);
				size += 1;
			}
			
			// Divide the sums by the number of pawns to acquire the average
			if (size > 0){
				averageBlackKing_blackPawnSeperation /= size;
				averageWhiteKing_blackPawnSeperation /= size;
				
			// Otherwise set the average as the max distance
			}else{
				averageBlackKing_blackPawnSeperation = 7;
				averageWhiteKing_blackPawnSeperation = 7;
			}
			
			// Loop through the mask containing white pawns in the second half
			r = 0;
			bb = secondHalf & occupied_white & pawns;
			size = 0;
			while (bb) {
				
				// Get the position of the most significant set bit of the mask
				r = 64 - __builtin_clzll(bb) - 1;			
				
				// Add all the distances between each king and the white pawns
				averageWhiteKing_whitePawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_white&kings));
				averageBlackKing_whitePawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_black&kings));
				bb ^= (1ULL << r);
				size += 1;
			}
			
			// Divide the sums by the number of pawns to acquire the average
			if (size > 0){
				averageWhiteKing_whitePawnSeperation /= size;
				averageBlackKing_whitePawnSeperation /= size;
				
			// Otherwise set the average as the max distance
			}else{
				averageWhiteKing_whitePawnSeperation = 7;
				averageBlackKing_whitePawnSeperation = 7;
			}
			
			// Update the scores based on the average distances
			total += (7 - averageBlackKing_whitePawnSeperation) * 200 + (7 - averageBlackKing_blackPawnSeperation) * 250;
			total -= (7 - averageWhiteKing_blackPawnSeperation) * 200 + (7 - averageWhiteKing_whitePawnSeperation) * 250;
			//std::cout << averageBlackKing_blackPawnSeperation << " " << averageWhiteKing_whitePawnSeperation << " " <<  averageBlackKing_whitePawnSeperation << " " << averageWhiteKing_blackPawnSeperation << " " <<std::endl;
		}
	}
	
	/*
		In this code section, boost both white and blacks score based on the existence of bishop and knight pairs
	*/
	if (scan_reversed_size(occupied_white&bishops) == 2){
		total -= 315;
	}
	if (scan_reversed_size(occupied_white&knights) == 2){
		total -= 300;
	}
	if (scan_reversed_size(occupied_black&bishops) == 2){
		total += 315;
	}
	if (scan_reversed_size(occupied_black&knights) == 2){
		total += 300;
	}
	
	
	/*
		In this code section, boost the scores of both sides based on how poor the side's defense is relative to the opponent's offense
	*/
	if (whiteOffensiveScore > blackDefensiveScore){
		total -= ((whiteOffensiveScore - blackDefensiveScore)/12) * 100;
	}
	
	if (blackOffensiveScore > whiteDefensiveScore){
		total += ((blackOffensiveScore - whiteDefensiveScore)/12) * 100;
	}
	
	//std::cout << total << " " << whiteOffensiveScore << " " << whiteDefensiveScore << " " <<  blackOffensiveScore << " " << blackDefensiveScore << " " <<std::endl;
	return total;
}

void initializePieceValues(uint64_t bb){
	
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
		
		// Get the position of the most significant set bit of the mask
		r = 64 - __builtin_clzll(bb) - 1;			
		
		// Call the piece type function to populate the array
		pieceTypeLookUp [r] = piece_type_at (r);
		bb ^= (1ULL << r);			
	} 
}

uint8_t piece_type_at(uint8_t square){
    
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
	uint64_t mask = (1ULL << square);

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

void setAttackingLayer(int increment){
	
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
	bool isEndGame = pieceNum < 17;
	if (queens == 0){
		isEndGame = pieceNum < 20;
	}
	
	// Set variable for squares being open near the king
	bool squareOpen = true;
	
	// Set the multiplier for open square boosts
	int multiplier = 5;
	
	// Define the x and y coordinates for each square
	uint8_t x,y;
	
	// Loop through the squares around the white king
	uint8_t r = 0;
	uint64_t bb = attacks_mask(true,0ULL,63 - __builtin_clzll(occupied_white&kings),6);
	while (bb) {
		
		// Get the position of the most significant set bit of the mask
		r = 64 - __builtin_clzll(bb) - 1;									

		// Get the x and y coordinates for the given square
		y = r / 8;
		x = r % 8;
		
		// Increment the area around the king
        attackingLayer[1][x][y] += increment;
		
		// If the square is open around the king, boost the score further
        squareOpen = false;
		if (!isEndGame){
			if ((occupied_white & (1ULL << r)) == 0){
				attackingLayer[1][x][y] += increment * multiplier;
				squareOpen = true;
			}
		}
		
		// Loop through the squares around the current king move square
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			
			// Get the position of the most significant set bit of the mask
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			
			// Get the x and y coordinates for the given square
			y = r_inner / 8;
			x = r_inner % 8;
			
			// Increment the given square
			attackingLayer[1][x][y] += increment;
			
			// If the square is open around the king, boost the score further
			if (!isEndGame && squareOpen){
				if ((occupied_white & (1ULL << r_inner)) == 0){
					attackingLayer[1][x][y] += increment * multiplier;
				}
			}
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
	
	// Loop through the squares around the black king
	r = 0;
	bb = attacks_mask(false,0ULL,63 - __builtin_clzll(occupied_black&kings),6);
	while (bb) {
		
		// Get the position of the most significant set bit of the mask
		r = 64 - __builtin_clzll(bb) - 1;									

		// Get the x and y coordinates for the given square
		y = r / 8;
		x = r % 8;
		
		// Increment the area around the king
        attackingLayer[0][x][y] += increment;
		
		// If the square is open around the king, boost the score further
        squareOpen = false;
		if (!isEndGame){
			if ((occupied_black & (1ULL << r)) == 0){
				attackingLayer[0][x][y] += increment * multiplier;
				squareOpen = true;
			}
		}
		
		// Loop through the squares around the current king move square
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			
			// Get the position of the most significant set bit of the mask
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			
			// Get the x and y coordinates for the given square
			y = r_inner / 8;
			x = r_inner % 8;
			
			// Increment the given square
			attackingLayer[0][x][y] += increment;
			
			// If the square is open around the king, boost the score further
			if (!isEndGame && squareOpen){
				if ((occupied_black & (1ULL << r_inner)) == 0){
					attackingLayer[0][x][y] += increment * multiplier;
				}
			}
			bb_inner ^= (1ULL << r_inner);
		}		
		bb ^= (1ULL << r);
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

int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces) {
	
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
	
	// Define a variable to hold the new square position to be analyzed
	uint8_t pos = 0;
	
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
				bitmask |= BB_FILES [f] & ~((1 << ((rank + 1) * 8)) - 1);
				if (f == file){
					infrontMask |= BB_FILES [f] & ~((1 << ((rank + 1) * 8)) - 1);
				}				
				/*
                // Iterate over the ranks above the given square's rank
                for (int r = rank + 1; r < 8; ++r) {
					
					// Calculate the square's position and set the bit at this position
                    pos = r * 8 + f;  
                    bitmask |= (1ULL << pos);

					if (f == file){
						infrontMask |= (1ULL << pos);
					}

                }

				*/
            }
        }
	// Else the current side is black	
    } else {
        // Iterate over the three relevant files
        for (int f = file - 1; f < file + 2; ++f) {
			
			// Check if the file is within bounds
            if (f >= 0 && f <= 7) {  
				bitmask |= BB_FILES [f] & ((1 << (rank * 8)) - 1);
				if (f == file){
					infrontMask |= BB_FILES [f] & ((1 << (rank * 8)) - 1);
				}
				/*
                // Iterate over the ranks below the given square's rank
                for (int r = 0; r < rank; ++r) {
					
					// Calculate the square's position and set the bit at this position
                    pos = r * 8 + f;
                    bitmask |= (1ULL << pos);

					if (f == file){
						infrontMask |= (1ULL << pos);
					}
                }
				*/
            }
        }
    }

	// Of the squares in front of pawn, filter to only include opposing pawns
    bitmask &= opposingPawnMask;
		
	// Loop through the bitmask 
	uint8_t r = 0;
	uint64_t bb = bitmask;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		
		// If there is a blocker directly in front of the pawn, then it has no potential to be a passed pawn
		if (r % 8 == x){
			return 0;			
		}
		
		// Otherwise there is an opposing pawn defending the promotion path, thereby lowering the increment
		ppIncrement -= 125;
		bb ^= (1ULL << r);
	}
	
	// The minimum increment is 0
	if (ppIncrement < 0) {
		return 0;
	
	// Otherwise check if the increment does not suffer a decrement, this suggests the pawn is a passed pawn
	} else if (ppIncrement == incrementCopy){
		
		// Check if there exists a non-pawn blocker infront of the passed pawn
		if ((infrontMask & (opposingPieces | curSidePieces)) != 0){
			
			// If the piece is of the opponent, return the increment as it is
			if ((infrontMask & opposingPieces) != 0){
				return ppIncrement;
				
			// Else boost the passed pawn as our own piece can easily be moved to make way
			} else{
				ppIncrement += 100;
			}
		
		// Else no blocker exists, the passed pawn is un-impeded, earning a larger boost
		} else{
			ppIncrement += 200;
		}
	}
	return ppIncrement;
	
}

/*
	Set of functions used to cache data
*/
void initializeZobrist() {
	
	/*
		Function to initialize the Zobrist table
	*/	
	
	// Random number generator
    std::mt19937_64 rng;  
    for (int pieceType = 0; pieceType < 12; ++pieceType) {
        for (int square = 0; square < 64; ++square) {
			
			// Assign a random number to the table
            zobristTable[pieceType][square] = rng();
        }
    }
}

uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask) {
    
	/*
		Function to generate a Zobrist hash for the current board state
		
		Parameters:
		- pawnsMask: The mask containing only pawns
		- knightsMask: The mask containing only knights
		- bishopsMask: The mask containing only bishops
		- rooksMask: The mask containing only rooks
		- queensMask: The mask containing only queens
		- kingsMask: The mask containing only kings
		- occupied_whiteMask: The mask containing only white pieces
		- occupied_blackMask: The mask containing only black pieces
		
		Returns:
		A hash of the starting position
	*/
	
	// Define the hash
	uint64_t hash = 0;
	
	// Set the global mask variables
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	
	// Define vectors to hold the pieces of each colour
	std::vector<uint8_t> blackPieces;
	std::vector<uint8_t> whitePieces;
	
	// Call the function to fill the vector with the squares of the black pieces
	scan_reversed(occupied_black,blackPieces);
    uint8_t size = blackPieces.size();
	
	// Loop through the pieces 
    for (uint8_t square = 0; square < size; square++) {        
		// Adjust the piece type for the black pieces and use the xor operation to set the hash given the piece type and location
		uint8_t pieceType = piece_type_at(blackPieces[square]) + 5;
		hash ^= zobristTable[pieceType][blackPieces[square]];
    }
	
	// Call the function to fill the vector with the squares of the black pieces
	scan_reversed(occupied_white,whitePieces);
    size = whitePieces.size();
	
	// Loop through the pieces 
    for (uint8_t square = 0; square < size; square++) {        
	
		// Adjust the piece type for the white pieces and use the xor operation to set the hash given the piece type and location
		uint8_t pieceType = piece_type_at(whitePieces[square]) - 1;
		hash ^= zobristTable[pieceType][whitePieces[square]];
    }
	
    return hash;
}

void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bool isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion) {
    
	/*
		Function to generate a Zobrist hash for the current board state for caching position evaluations
		
		Parameters:
		- hash: The current hash before the move is made, passed by reference
		- fromSquare: The square from which the move will be made
		- toSquare: The destination square of the move
		- isCapture: A boolean describing if the move is a capture
		- pawnsMask: The mask containing only pawns
		- knightsMask: The mask containing only knights
		- bishopsMask: The mask containing only bishops
		- rooksMask: The mask containing only rooks
		- queensMask: The mask containing only queens
		- kingsMask: The mask containing only kings
		- occupied_whiteMask: The mask containing only white pieces
		- occupied_blackMask: The mask containing only black pieces
		- promotion: An integer describing the promotion piece (-1 if none)
		
		Returns:
		A hash of the starting position
	*/
	
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	
	// Acquire the piece type and colour
	bool fromSquareColour = bool(occupied_white & (1ULL << fromSquare));	
	uint8_t pieceType = piece_type_at(fromSquare) - 1;
	
	// If the piece is black, adjust the piece type
	if (!fromSquareColour){
		pieceType += 6;
	}
	
	// XOR the moving piece out of its old position
    hash ^= zobristTable[pieceType][fromSquare];
    
	/*
		This section of code checks for castling moves and adjusts the hash for the rook move
	*/
	if (pieceType == 5){
		if (fromSquare == 4){
			if (toSquare == 6){
				hash ^= zobristTable[3][5];
			}else if (toSquare == 2){
				hash ^= zobristTable[3][3];
			}
		}
	}else if (pieceType == 11){
		if (fromSquare == 60){
			if (toSquare == 62){
				hash ^= zobristTable[9][61];
			}else if (toSquare == 58){
				hash ^= zobristTable[9][59];
			}
		}
	}
	
    // If a piece was captured, XOR the captured piece out of its position
    if (isCapture) {
		
		// Acquire the captured piece
		int8_t capturedPieceType = piece_type_at(toSquare) - 1;
		
		// If the capture piece does not exist at the destination, it's because the capture was by en passent
		if (capturedPieceType == -1){
			
			// Handle removing the pawn captured through en passent
			if (fromSquareColour){
				hash ^= zobristTable[6][toSquare - 8];
			} else{
				hash ^= zobristTable[0][toSquare + 8];
			}
		// Else the capture is regular
		} else{
			
			// If the piece is black, adjust the piece type
			if (fromSquareColour){
				capturedPieceType += 6;
			}
			
			// XOR the captured piece out
			hash ^= zobristTable[capturedPieceType][toSquare];			
		}
    }
    
	// If there exists a promotion piece, then handle it
	if (promotion != 0){
		
		// Acquire the promotion piece type
		pieceType = promotion - 1;
		
		// If the piece is black, adjust the piece type
		if (!fromSquareColour){
			pieceType += 6;
		}
		
		// XOR the piece into its new position
		hash ^= zobristTable[pieceType][toSquare];
	} else{
		
		// XOR the piece into its new position
		hash ^= zobristTable[pieceType][toSquare];
	} 
}

int accessCache(uint64_t key) {
	
	/*
		Function to access the position cache
		
		Parameters:
		- key: The hash for the given position
		
		Returns:
		The stored evaluation for the position if it exists
	*/
	
    auto it = moveCache.find(key);
    if (it != moveCache.end()) {
		// Return the value if the key exists
        return it->second;  
    }
	
	// Return the default value if the key doesn't exist
    return 0;   
}

void addToCache(uint64_t key,int value) {
	
	/*
		Function to add to the position cache
		
		Parameters:
		- key: The hash for the given position
		- value: The value to be associated with the given key
	*/
	
	// Add the key-value pair to the cache as well as the key to the move order
    moveCache[key] = value;
	insertionOrder.push_back(key);
}

std::string accessOpponentMoveGenCache(uint64_t key) {
	
	/*
		Function to access the move cache of the opponent of the engine
		
		Parameters:
		- key: The hash for the given position
		
		Returns:
		The stored byte stream representing the legal moves list for the given position
	*/
	
    auto it = OpponentMoveGenCache.find(key);
	
	// Return the value if the key exists
    if (it != OpponentMoveGenCache.end()) {
        return it->second;  
    }
	
    // Return a binary representation of '0' if the key doesn't exist    
	// Allocate space for '0' and null terminator
	char* defaultValue = new char[2]; 
    
	defaultValue[0] = '0'; // Set the first byte to '0'
    defaultValue[1] = '\0'; // Null terminator for string
    return defaultValue;

}

void addToOpponentMoveGenCache(uint64_t key,char* data, int length) {
	
	/*
		Function to add to the move cache of the opponent of the engine
		
		Parameters:
		- key: The hash for the given position
		- value: The value to be associated with the given key
		- length: The length of the byte stream
	*/
	
	// Convert the byte stream to a c++ string
	std::string value(data, length);
    
	// Add the key-value pair to the cache as well as the key to the move order
	OpponentMoveGenCache[key] = value;
	OpponentMoveGenInsertionOrder.push_back(key);
	
}

std::string accessCurPlayerMoveGenCache(uint64_t key) {
	
	/*
		Function to access the move cache of the engine
		
		Parameters:
		- key: The hash for the given position
		
		Returns:
		The stored byte stream representing the legal moves list for the given position
	*/
	
    auto it = curPlayerMoveGenCache.find(key);
	
	// Return the value if the key exists
    if (it != curPlayerMoveGenCache.end()) {
        return it->second;  
    }
	
    // Return a binary representation of '0' if the key doesn't exist
	// Allocate space for '0' and null terminator
    char* defaultValue = new char[2]; 
	
    defaultValue[0] = '0'; // Set the first byte to '0'
    defaultValue[1] = '\0'; // Null terminator for string
    return defaultValue;

}

void addToCurPlayerMoveGenCache(uint64_t key,char* data, int length) {
	
	/*
		Function to add to the move cache of the engine
		
		Parameters:
		- key: The hash for the given position
		- value: The value to be associated with the given key
		- length: The length of the byte stream
	*/
	
	// Convert the byte stream to a c++ string
	std::string value(data, length);
	
	// Add the key-value pair to the cache as well as the key to the move order
    curPlayerMoveGenCache[key] = value;
	curPlayerMoveGenInsertionOrder.push_back(key);
}

int printCacheStats() {
	
	/*
		Function to print the position cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Get the number of entries in the map
    int num_entries = moveCache.size();

    // Estimate the memory usage in bytes: each entry is a pair of (key, value)
    int size_in_bytes = num_entries * (sizeof(int64_t) + sizeof(int));

    // Print the results
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}

int printOpponentMoveGenCacheStats() {
	
	/*
		Function to print the opposition move cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Get the number of entries in the map
    int num_entries = OpponentMoveGenCache.size();

    // Estimate the memory usage in bytes
    int size_in_bytes = 0;

    // Size of the key (int64_t)
    size_in_bytes += num_entries * sizeof(int64_t);

    // Iterate through the map to calculate the size of each value
    for (const auto& entry : OpponentMoveGenCache) {
        // entry.first is the key
        // entry.second is the char* value
        size_in_bytes += entry.second.length() + 1; // +1 for the null terminator
    }

    // Print the results
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}

int printCurPlayerMoveGenCacheStats() {
	
	/*
		Function to print the engine's move cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Get the number of entries in the map
    int num_entries = curPlayerMoveGenCache.size();

    // Estimate the memory usage in bytes
    int size_in_bytes = 0;

    // Size of the key (int64_t)
    size_in_bytes += num_entries * sizeof(int64_t);

    // Iterate through the map to calculate the size of each value
    for (const auto& entry : curPlayerMoveGenCache) {
        // entry.first is the key
        // entry.second is the char* value
        size_in_bytes += entry.second.length() + 1; // +1 for the null terminator
    }

    // Print the results
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}

void evictOldEntries(int numToEvict) {
	
	/*
		Function to add evict entries in the position cache in an LRU fashion
		
		Parameters:
		- numToEvict: The number of entries to be evicted from the cache
	*/
	
	// Loop through the cache until done or the cache empties
    while (numToEvict-- > 0 && !insertionOrder.empty()) {
		
		// Evict using the insertion order queue 
        uint64_t oldestKey = insertionOrder.front();
        insertionOrder.pop_front();  // Remove from deque
        moveCache.erase(oldestKey);  // Erase from map
    }
}

void evictOpponentMoveGenEntries(int numToEvict) {
	
	/*
		Function to add evict entries in the oposition move cache in an LRU fashion
		
		Parameters:
		- numToEvict: The number of entries to be evicted from the cache
	*/
	
	// Loop through the cache until done or the cache empties
    while (numToEvict-- > 0 && !OpponentMoveGenInsertionOrder.empty()) {
		
		// Evict using the insertion order queue 
        uint64_t oldestKey = OpponentMoveGenInsertionOrder.front();
        OpponentMoveGenInsertionOrder.pop_front();  // Remove from deque
        OpponentMoveGenCache.erase(oldestKey);  // Erase from map
    }
}

void evictCurPlayerMoveGenEntries(int numToEvict) {
	
	/*
		Function to add evict entries in the engine's move cache in an LRU fashion
		
		Parameters:
		- numToEvict: The number of entries to be evicted from the cache
	*/
	
	// Loop through the cache until done or the cache empties
    while (numToEvict-- > 0 && !curPlayerMoveGenInsertionOrder.empty()) {
		
		// Evict using the insertion order queue 
        uint64_t oldestKey = curPlayerMoveGenInsertionOrder.front();
        curPlayerMoveGenInsertionOrder.pop_front();  // Remove from deque
        curPlayerMoveGenCache.erase(oldestKey);  // Erase from map
    }
}


/*
	Set of functions used to generate moves
*/
void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, uint64_t our_pieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t from_mask, uint64_t to_mask){
    
	/*
		Function to generate moves for non-pawn pieces
		
		Parameters:
		- startPos: An empty vector to hold the starting positions, passed by reference 
		- endPos: An empty vector to hold the ending positions, passed by reference 
		- our_pieces: The mask containing only the pieces of the current side
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
		- from_mask: The mask of the possible starting positions for move generation
		- to_mask: The mask of the possible ending positions for move generation		
	*/
	
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
	
	// Define mask of non pawn pieces
	uint64_t non_pawns = (our_pieces & ~pawns) & from_mask;
		
	// Loop through the non pawn pieces
	uint8_t r = 0;
	uint64_t bb = non_pawns;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		
		// Define the moves as a bitwise and between the squares attacked from the starting square and the starting mask
		uint64_t moves = (attacks_mask(bool((1ULL<<r) & occupied_white),occupied,r,piece_type_at (r)) & ~our_pieces) & to_mask;		
		
		// Loop through the possible destinations
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			
			// Push the starting and ending positions to their respective vectors
			startPos.push_back(r);
			endPos.push_back(r_inner);  
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
}

void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, uint64_t occupied, bool colour, uint64_t pawnsMask, uint64_t from_mask, uint64_t to_mask){    			
	
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
		r = 64 - __builtin_clzll(bb) - 1;
		
		// Acquire the destinations that follow pawn attacks and opposing pieces
		uint64_t moves = BB_PAWN_ATTACKS[colour][r] & opposingPieces & to_mask;
		
		// Loop through the destinations 
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			
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
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
	
	/*
		In this section, define single and double pawn pushes
	*/
	uint64_t single_moves, double_moves;
	if (colour){
        single_moves = pawnsMask << 8 & ~occupied;
        double_moves = single_moves << 8 & ~occupied & (BB_RANK_3 | BB_RANK_4);
    }else{
        single_moves = pawnsMask >> 8 & ~occupied;
        double_moves = single_moves >> 8 & ~occupied & (BB_RANK_6 | BB_RANK_5);
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
		r = 64 - __builtin_clzll(bb) - 1;
		
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
		bb ^= (1ULL << r);
	}
	
	/*
		This section of code is used for double pawn pushes
	*/		
	// Loop through the pawns
	r = 0;
	bb = double_moves;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		
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
		bb ^= (1ULL << r);
	}
	
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
		r = 64 - __builtin_clzll(bb) - 1;
		
		// Update the blockers where there is exactly one blocker per attack 
		uint64_t b = betweenPieces(king, r) & occupied;        
        if (b && (1ULL << (63 - __builtin_clzll(b)) == b)){
            blockers |= b;
		}
		bb ^= (1ULL << r);
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
        bb ^= (1ULL << r);		
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

void scan_forward(uint64_t bb, std::vector<uint8_t> &result) {
		
	/*
		Function to acquire a vector of set bits in a given bitmask starting from the bottom of the board
		
		Parameters:
		- bb: The bitmask to be scanned
		- result: An empty vector to hold the set bits, passed by reference
	*/
	
	uint64_t r = 0;
    while (bb) {
        r = bb &-bb;
        result.push_back(64 - __builtin_clzll(r) - 1);
        bb ^= r;
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
		
    int file_distance = abs((sq1 % 8) - (sq2 % 8));
    int rank_distance = abs((sq1 / 8) - (sq2 / 8));
    return std::max(file_distance, rank_distance);
}

uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType){	
	
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
	
	if (pieceType == 1){		
		return BB_PAWN_ATTACKS[colour][square];
	}else if (pieceType == 2){
		return BB_KNIGHT_ATTACKS[square];
	}else if (pieceType == 6){
		
		return BB_KING_ATTACKS[square];
	}else{
		uint64_t attacks = 0;
		if (pieceType == 3 || pieceType == 5){
			attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		}
		if (pieceType == 4 || pieceType == 5){			
			attacks |= (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] |
						BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]);
		}
		return attacks;
	}
}