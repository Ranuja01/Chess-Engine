// cpp_wrapper.cpp
#include "cpp_bitboard.h"
#include <vector>
#include <cstddef>
#include <cstdint>
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

uint64_t zobristTable[12][64];
std::unordered_map<uint64_t, int> moveCache;
std::deque<uint64_t> insertionOrder;

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

constexpr std::array<int, 7> values = {0, 1000, 3150, 3250, 5000, 9000, 12000};

std::array<std::array<std::array<int, 8>, 8>, 2> attackingLayer;

std::array<std::array<std::array<int, 8>, 8>, 2> placementLayer = {{
    {{
        {{0,0,0,0,0,0,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,20,25,5,0,0}},
        {{0,0,3,15,15,5,0,0}},
        {{0,0,3,10,10,2,0,0}},
        {{0,0,0,0,0,0,0,0}}
    }},
    {{
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

std::array<std::array<std::array<int, 8>, 8>, 2> placementLayer2 = {{
    {{
        {{0,0,1,2,25,35,40,0}},
        {{0,0,2,5,25,35,40,0}},
        {{0,0,3,5,25,35,40,0}},
        {{0,0,3,5,25,35,40,0}},
        {{0,0,3,5,25,35,40,0}},
        {{0,0,3,5,25,35,40,0}},
        {{0,0,2,5,25,35,40,0}},
        {{0,0,1,5,25,35,40,0}}
    }},
    {{
        {{0,40,35,25,2,1,0,0}},
        {{0,40,35,25,5,2,0,0}},
        {{0,40,35,25,5,3,0,0}},
        {{0,40,35,25,5,3,0,0}},
        {{0,40,35,25,5,3,0,0}},
        {{0,40,35,25,5,3,0,0}},
        {{0,40,35,25,5,2,0,0}},
        {{0,40,35,25,2,1,0,0}}
    }}
}};

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

std::array<uint8_t, 64> pieceTypeLookUp = {};
uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;
int whiteOffensiveScore, blackOffensiveScore, whiteDefensiveScore, blackDefensiveScore;
int blackPieceVal, whitePieceVal;

void initialize_attack_tables() {
    std::vector<int8_t> knight_deltas = {17, 15, 10, 6, -17, -15, -10, -6};
    std::vector<int8_t> king_deltas = {9, 8, 7, 1, -9, -8, -7, -1};
    std::vector<int8_t> white_pawn_deltas = {-7, -9};
    std::vector<int8_t> black_pawn_deltas = {7, 9};

    for (int sq = 0; sq < NUM_SQUARES; ++sq) {
        BB_KNIGHT_ATTACKS[sq] = sliding_attacks(sq, ~0ULL, knight_deltas);
        BB_KING_ATTACKS[sq] = sliding_attacks(sq, ~0ULL, king_deltas);
		BB_PAWN_ATTACKS[0][sq] = sliding_attacks(sq, ~0ULL, white_pawn_deltas);
        BB_PAWN_ATTACKS[1][sq] = sliding_attacks(sq, ~0ULL, black_pawn_deltas);
    }
	
	attack_table({-9, -7, 7, 9},BB_DIAG_MASKS,BB_DIAG_ATTACKS);
	attack_table({-8, 8},BB_FILE_MASKS,BB_FILE_ATTACKS);
	attack_table({-1, 1},BB_RANK_MASKS,BB_RANK_ATTACKS);
	
	rays(BB_RAYS);
}

void setAttackingLayer(uint64_t occupied_white, uint64_t occupied_black, uint64_t kings, int increment){
	
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
	
	//bool isEndGame = (scan_reversed_size(occupied_white | occupied_black) - 2) < 16;
	int pieceNum = scan_reversed_size(occupied) - 2;
	bool isEndGame = pieceNum < 17;
	
	if (queens == 0){
		isEndGame = pieceNum < 20;
	}
	//isEndGame = true;
	bool squareOpen = true;
	int multiplier = 5;
	
	uint8_t x,y;
	
	uint8_t r = 0;
	uint64_t bb = attacks_mask(true,0ULL,63 - __builtin_clzll(occupied_white&kings),6);
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		
		y = r / 8;
        x = r % 8;
        attackingLayer[1][x][y] += increment;
		
        squareOpen = false;
		if (!isEndGame){
			if ((occupied_white & (1ULL << r)) == 0){
				attackingLayer[1][x][y] += increment * multiplier;
				squareOpen = true;
			}
		}
		
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			y = r_inner / 8;
			x = r_inner % 8;
			attackingLayer[1][x][y] += increment;
			
			if (!isEndGame && squareOpen){
				if ((occupied_white & (1ULL << r_inner)) == 0){
					attackingLayer[1][x][y] += increment * multiplier;
				}
			}
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
	
	r = 0;
	bb = attacks_mask(false,0ULL,63 - __builtin_clzll(occupied_black&kings),6);
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		
		y = r / 8;
        x = r % 8;
        attackingLayer[0][x][y] += increment;
		
        squareOpen = false;
		if (!isEndGame){
			if ((occupied_black & (1ULL << r)) == 0){
				attackingLayer[0][x][y] += increment * multiplier;
				squareOpen = true;
			}
		}
		
		uint8_t r_inner = 0;
		uint64_t bb_inner = attacks_mask(true,0ULL,r,6);
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			y = r_inner / 8;
			x = r_inner % 8;
			attackingLayer[0][x][y] += increment;
			
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

uint8_t piece_type_at(uint8_t square){
        
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

int placement_and_piece_midgame(uint8_t square){
    
    int total = 0;
    int rookIncrement = 300;
    int ppIncrement = 300;
    uint64_t rooks_mask = 0ULL;
	uint8_t piece_type = pieceTypeLookUp [square];
	bool colour = bool(occupied_white & (1ULL << square)); 
    
    uint8_t y = square / 8;
    uint8_t x = square % 8;
        
    if (colour) {
        total -= values[piece_type];
        
        if (! (piece_type == 4 || piece_type == 6)){
            
            total -= whitePlacementLayer[piece_type - 1][x][y];
            
            if (piece_type == 2 || piece_type == 3){
                total -= 350;
            }
			
            if (piece_type == 1){
                
                if (scan_reversed_size((BB_FILES[x] & (occupied_white & pawns))) > 1) {                
                    total += 200;
				}
				
				ppIncrement = getPPIncrement(square, colour, (occupied_black & pawns), ppIncrement, x);
				total -= ppIncrement;
				if (ppIncrement == 300) {
					total -= (y + 1) * 75 + ((y + 1) * (y + 1)) * 10;
				} else {
					total -= (y + 1) * 50;
				}
				
				uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
				uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
				
				if (left != 0){
					
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
					if (attackedPieceType == 1 && bool(occupied_white & (1ULL << left))){
						total -= 50;
					}
				}
				
				if (right != 0){
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
					if (attackedPieceType == 1 && bool(occupied_white & (1ULL << right))){
						total -= 50;
					}
				}
				                
			}
        }else if (piece_type == 4){  
            
            if (y == 6){
                rookIncrement += 50;
			}
			
            rooks_mask |= BB_FILES[x] & occupied;            
            
			uint64_t r = 0;
			uint64_t bb = rooks_mask;

			while (bb) {
				r = bb &-bb;							
				uint8_t att_square = 64 - __builtin_clzll(r) - 1;
				bb ^= r;
                if (att_square > square){
                    uint8_t temp_piece_type = pieceTypeLookUp [att_square];
					bool temp_colour = bool(occupied_white & (1ULL << att_square));
                    if (temp_colour){
                        if (temp_piece_type == 1){                            
                            if (att_square / 8 < 5){
                                rookIncrement -= (50 + ((3 - (att_square / 8)) * 125));
                                break;
							}
                        } else if(temp_piece_type == 2 || temp_piece_type == 3){
                            rookIncrement -= 15;
						}
                    }else{
                        if (temp_piece_type == 1){
                            if (att_square / 8 > 4){
                                rookIncrement -= 50;
							}
                        }else if(temp_piece_type == 2 || temp_piece_type == 3){
                            rookIncrement -= 35;
                        }else if(temp_piece_type == 4){
                            rookIncrement -= 75;
						}
					}
				}
			}
			
			total -= rookIncrement;
        }
        
		uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;
		
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;										
			y = r / 8;
            x = r % 8;
            if (piece_type == 5){
                total -= attackingLayer[0][x][y] >> 2;
				total -= attackingLayer[1][x][y] >> 3;
				whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
				whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
				
				occupiedCopy &= ~(1ULL << r);
				
				if (bool(~occupied_white & (1ULL << r))){
					total -= 5;
				}					
				
            }else if (piece_type == 1){
				total -= attackingLayer[0][x][y] >> 2;
				total -= attackingLayer[1][x][y] >> 3;
				whiteOffensiveScore += attackingLayer[0][x][y] >> 1;
				whiteDefensiveScore += attackingLayer[1][x][y] >> 2;
				
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << r))){
					total -= 50;
				}
					
				
			}else if (piece_type == 6){
				total -= attackingLayer[0][x][y];   
				whiteOffensiveScore += attackingLayer[0][x][y];
								
				if (pieceTypeLookUp[r] == 1 && y > (square / 8)){
					whiteDefensiveScore += (attackingLayer[1][x][y] << 1) + 50;
					total -= (attackingLayer[1][x][y] << 1) + 100;
				}else{
					whiteDefensiveScore += attackingLayer[1][x][y];
					total -= attackingLayer[1][x][y] >> 2;  
				}
				
			}else{    
                total -= attackingLayer[0][x][y];   
				total -= attackingLayer[1][x][y] >> 1;  
				whiteOffensiveScore += attackingLayer[0][x][y];
				whiteDefensiveScore += attackingLayer[1][x][y] >> 1;
				
				if (piece_type == 3 || piece_type == 4){
					occupiedCopy &= ~(1ULL << r);
					
					if (bool(~occupied_white & (1ULL << r))){
						total -= 10;
					}
					
				} else{					
					
					if (bool(~occupied_white & (1ULL << r))){
						total -= 15;
					}
				}
				
			}
			bb ^= (1ULL << r);
			
		}
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_white;
			
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;
				y = r / 8;
				x = r % 8;
				
				total -= attackingLayer[0][x][y] << 2;
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
		
    }else{
        total += values[piece_type];
        if (! (piece_type == 4 || piece_type == 6)){
            
            total += blackPlacementLayer[piece_type - 1][x][y];
            
            if (piece_type == 2 || piece_type == 3){
                total += 350;
            }
            if (piece_type == 1){
                               
                if (scan_reversed_size((BB_FILES[x] & (occupied_black & pawns))) > 1){                
                    total -= 200;
                }
				ppIncrement = getPPIncrement(square, colour, (occupied_white & pawns), ppIncrement, x);
				total += ppIncrement;
          
				if (ppIncrement == 300){
					total += (8 - y) * 75 + ((8 - y) * (8 - y)) * 10;				
				}else{
					total += (8 - y) * 50;     
				}
				
				uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
				uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
				
				if (left != 0){
					
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
					if (attackedPieceType == 1 && bool(occupied_white & (1ULL << left))){
						total += 50;
					}
				}
				
				if (right != 0){
					uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
					if (attackedPieceType == 1 && bool(occupied_white & (1ULL << right))){
						total += 50;
					}
				}
                
			}
                
        }else if (piece_type == 4){
            if (y == 1){
                rookIncrement += 50;
			}
            rooks_mask |= BB_FILES[x] & occupied;            
            
			uint8_t r = 0;
			uint64_t bb = rooks_mask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;
				uint8_t att_square = r;
				bb ^= (1ULL << r);
				
                if (att_square < square){
                    uint8_t temp_piece_type = pieceTypeLookUp [att_square];
					bool temp_colour = bool(occupied_white & (1ULL << att_square));
                    if (temp_colour){
                        if (temp_piece_type == 1){
                            if (att_square / 8 < 5){
                                rookIncrement -= 50;
							}
                        }else if(temp_piece_type == 2 || temp_piece_type == 3){
                            rookIncrement -= 35;
                        }else if (temp_piece_type == 4){
                            rookIncrement -= 75;
						}
                    }else{
                        if (temp_piece_type == 1){
                            if ((att_square / 8) > 4){
                                rookIncrement -= (50 + (((att_square / 8) - 4) * 125));								
                                break;
							}
                        }else if(temp_piece_type == 2 || temp_piece_type == 3){
                            rookIncrement -= 15;
						}
					}
				}
			}
			
            total += rookIncrement;
        }
		
        uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;		
		
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;
			y = r / 8;
            x = r % 8;
			
			
            if (piece_type == 5){
                total += attackingLayer[1][x][y] >> 2;
				total += attackingLayer[0][x][y] >> 3;
				blackOffensiveScore += attackingLayer[1][x][y] >> 1;
				blackDefensiveScore += attackingLayer[0][x][y] >> 2;				
				
				occupiedCopy &= ~(1ULL << r);
				
				if (bool(~occupied_black & (1ULL << r))){
					total += 5;
				}
					
				
			}else if (piece_type == 1){
				total += attackingLayer[1][x][y] >> 2;
				total += attackingLayer[0][x][y] >> 3;
				blackOffensiveScore += attackingLayer[1][x][y] >> 1;
				blackDefensiveScore += attackingLayer[0][x][y] >> 2;
				
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << r))){
					total += 50;
				}
					
				
			}else if (piece_type == 6){
				total += attackingLayer[1][x][y];
				blackOffensiveScore += attackingLayer[1][x][y];
								
				if (pieceTypeLookUp[r] == 1 && y < (square / 8)){
					blackDefensiveScore += (attackingLayer[0][x][y] << 1) + 50;
					total += (attackingLayer[0][x][y] << 1) + 100;
				}else{
					blackDefensiveScore += attackingLayer[0][x][y];
					total += attackingLayer[0][x][y] >> 2;
				}
				
			}else{    
                total += attackingLayer[1][x][y];
				total += attackingLayer[0][x][y] >> 1;
				blackOffensiveScore += attackingLayer[1][x][y];
				blackDefensiveScore += attackingLayer[0][x][y] >> 1;	

				if (piece_type == 3 || piece_type == 4){
					occupiedCopy &= ~(1ULL << r);
					
					if (bool(~occupied_black & (1ULL << r))){
						total += 10;
					}
					
				} else{
					
					if (bool(~occupied_black & (1ULL << r))){
						total += 15;
					}
					
				}
				
			}
			bb ^= (1ULL << r);
		}
		
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_black;
			
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;
				y = r / 8;
				x = r % 8;
				
				total += attackingLayer[1][x][y] << 2;
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
    
    int total = 0;
    int rookIncrement = 100;
    int ppIncrement = 800;
	int attackMultiplier = 1;
    uint64_t rooks_mask = 0ULL;
	uint8_t piece_type = pieceTypeLookUp [square];
	bool colour = bool(occupied_white & (1ULL << square));     

    uint8_t y = square / 8;
	//uint8_t y = square >> 3;
    uint8_t x = square % 8;
        
    if (colour) {
        total -= values[piece_type];
        whitePieceVal += values[piece_type];
		if (piece_type == 1){
						                            
			if (scan_reversed_size((BB_FILES[x] & (occupied_white & pawns))) > 1) {                
				total += 200;
			}
			
			if (y > 2){
				ppIncrement = getPPIncrement(square, colour, (occupied_black & pawns), ppIncrement, x);
			} else{
				ppIncrement = 0;
			}
			
			total -= ppIncrement;
			
			if (ppIncrement == 800) {
				total -= (y + 1) * 150 + ((y + 1) * (y + 1)) * 15;
			} else {
				total -= (y + 1) * 50;
			}
			
			uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
			uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
			
			if (left != 0){
					
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << left))){
					total -= 400;
				}
			}
			
			if (right != 0){
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << right))){
					total -= 400;
				}
			}
			
		}else if (piece_type == 4){  
            total -= 500;
            rooks_mask |= BB_FILES[x] & occupied;            
            			
			uint64_t r = 0;
			uint64_t bb = rooks_mask;

			while (bb) {
				r = bb &-bb;							
				uint8_t att_square = 64 - __builtin_clzll(r) - 1;
				uint8_t temp_piece_type = pieceTypeLookUp [att_square];
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                if (temp_piece_type == 1){
                    if (att_square > square){ // Up the board from the white rook
                        if (temp_colour){ // If the pawn is white 
                            rookIncrement += ((att_square / 8) + 1) * 100; // Increment rook for supporting the white pawn
                        }else{ // If the pawn is black
                            rookIncrement += (8 - (att_square / 8)) * 50; // Increment rook for blockading black pawn  
						}
                    }else { // Down the board from the white rook
                        if (temp_colour){ // If the pawn is white
                            if (att_square / 8 > 3){
                                rookIncrement -= 50 + ((att_square / 8) - 3) * 50; // Decrement rook for blocking own pawn
							}
                        }else{ // If the pawn is black
                            rookIncrement += (8 - (att_square / 8)) * 50; // Increment rook for attacking black pawn from behind
						}
					}
				}
				bb ^= r;
			}
			
			total -= rookIncrement;
        }else if (piece_type == 3){  
			total -= 350;
		} else if (piece_type == 3){
			total -= 250;
		}
        
    }else{
        total += values[piece_type];
		blackPieceVal += values[piece_type];
        if (piece_type == 1){
			
			if (scan_reversed_size((BB_FILES[x] & (occupied_black & pawns))) > 1){                
				total -= 200;
			}
			
			if (y < 5){
                ppIncrement = getPPIncrement(square, colour, (occupied_white & pawns), ppIncrement, x);
            }else{
                ppIncrement = 0;
			}
            total += ppIncrement;
          
            if (ppIncrement == 800){
                total += (8 - y) * 150 + ((8 - y) * (8 - y)) * 15;				
            }else{
                total += (8 - y) * 50;     
			}
			
			uint64_t left = ((1ULL << square) >> 1) & ~BB_FILE_H;
			uint64_t right = ((1ULL << square) << 1) & ~BB_FILE_A;
			
			if (left != 0){
				
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(left)]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << left))){
					total += 400;
				}
			}
			
			if (right != 0){
				uint8_t attackedPieceType = pieceTypeLookUp[63 - __builtin_clzll(right)]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << right))){
					total += 400;
				}
			}
			
		}else if (piece_type == 4){
            total += 500;
            rooks_mask |= BB_FILES[x] & occupied;            
            			
			uint8_t r = 0;
			uint64_t bb = rooks_mask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;			
				uint8_t att_square = r;
				uint8_t temp_piece_type = pieceTypeLookUp [att_square];
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                if (temp_piece_type == 1){    
                    if (att_square < square){ // Down the board from the black rook
                        if (temp_colour){ // If the pawn is white
                            rookIncrement += ((att_square / 8) + 1) * 50; // Increment rook for blockading white pawn                       
                        }else{ // If the pawn is black
                            rookIncrement += (8 - (att_square / 8)) * 100; // Increment rook for supporting the black pawn
						}
                    }else{ // Up the board from the black rook
                        if (temp_colour){ // If the pawn is white
                            rookIncrement += ((att_square / 8) + 1) * 50; // Increment rook for attacking white pawn from behind
						}else{ // If the pawn is black
                            if (att_square / 8 < 4){
                                rookIncrement -= 50 + (4 - (att_square / 8)) * 50; // Decrement rook for blocking own pawn
							}
						}
					}
				}
				bb ^= (1ULL << r);			
			}
			
            total += rookIncrement;
        }else if (piece_type == 3){  
			total += 350;
		} else if (piece_type == 2){
			total += 250;
		}
    }
	
	if (colour){
        
        if (total < -7500){
            attackMultiplier = 2;
        }else if(total < -15000){
            attackMultiplier = 3;
        }
		
		uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;        
		
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;
			y = r / 8;
            x = r % 8 ;           
            total -= attackingLayer[0][x][y] * attackMultiplier;   
			
			if (piece_type == 3 || piece_type == 4 || piece_type == 5){
				occupiedCopy &= ~(1ULL << r);
					
				if (bool(~occupied_white & (1ULL << r))){
					total -= 10;
				}
			}else if (piece_type == 2){
				if (bool(~occupied_white & (1ULL << r))){
					total -= 10;
				}
			} else if (piece_type == 1){
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && bool(occupied_white & (1ULL << r))){
					total -= 400;
				}
			}
			
			
			bb ^= (1ULL << r);	
		}
		
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_white;
						
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;
				y = r / 8;
				x = r % 8;
				
				total -= attackingLayer[0][x][y] << 2;
				uint8_t xRayPieceType = pieceTypeLookUp[r]; 
				if (xRayPieceType != 0){
					total -= values[xRayPieceType] >> 6;
				}
				bb ^= (1ULL << r);	
			}

		}
		
    }else{
        
        if (total < 7500){
            attackMultiplier = 2;
        }else if(total < 15000){
            attackMultiplier = 3;
		}
        
        uint64_t pieceAttackMask = attacks_mask(colour,occupied,square,piece_type);
		uint64_t occupiedCopy = occupied;
        
		
		uint8_t r = 0;
		uint64_t bb = pieceAttackMask;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;
			y = r / 8;
            x = r % 8;
            total += attackingLayer[1][x][y] * attackMultiplier;
			
			if (piece_type == 3 || piece_type == 4 || piece_type == 5){
				occupiedCopy &= ~(1ULL << r);
					
				if (bool(~occupied_black & (1ULL << r))){
					total += 10;
				}
			} else if (piece_type == 2){
				if (bool(~occupied_black & (1ULL << r))){
					total += 10;
				}
			} else if (piece_type == 1){
				uint8_t attackedPieceType = pieceTypeLookUp[r]; 
				if (attackedPieceType == 1 && !bool(occupied_white & (1ULL << r))){
					total += 400;
				}
			}
			
			bb ^= (1ULL << r);	
		}
		
		if (piece_type == 3 || piece_type == 4 || piece_type == 5){
			uint64_t xRayMask = (~pieceAttackMask & attacks_mask(colour,occupiedCopy,square,piece_type)) & ~occupied_black;
			
			uint8_t r = 0;
			uint64_t bb = xRayMask;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;
				y = r / 8;
				x = r % 8;
				
				total += attackingLayer[1][x][y] << 2;
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
	int total = 0;
	uint8_t size;
	
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	occupied = occupiedMask;
	
	whiteOffensiveScore = 0;
	blackOffensiveScore = 0;
	whiteDefensiveScore = 0;
	blackDefensiveScore = 0;	
	
	blackPieceVal = 0;
	whitePieceVal = 0;
	
	int pieceNum = scan_reversed_size(occupied) - 2;
	bool isEndGame = pieceNum < 16;
	bool isNearGameEnd = pieceNum < 10;
	
	initializePieceValues(occupied);
		
	if (queens == 0){
		isEndGame = pieceNum < 18;
		isNearGameEnd = pieceNum < 12;
	}
	
	if (!isEndGame){

		setAttackingLayer(occupied_white, occupied_black, kings, 5);
				
		uint8_t r = 0;
		uint64_t bb = occupied;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;			
			total += placement_and_piece_midgame(r);
			bb ^= (1ULL << r);			
		} 
		
	}else{
		
		uint8_t r = 0;
		uint64_t bb = occupied;
		while (bb) {
			r = 64 - __builtin_clzll(bb) - 1;			
			total += placement_and_piece_endgame(r);
			bb ^= (1ULL << r);			
		} 
		
		if (blackPieceVal > whitePieceVal){
			total += (int)(((blackPieceVal - whitePieceVal)/ blackPieceVal) * 1000);		
		}else if (whitePieceVal > blackPieceVal){			
			total -= (int)(((whitePieceVal - blackPieceVal)/ whitePieceVal) * 1000);
		}
		
		if (isNearGameEnd){
			uint8_t whiteKingSquare = 63 - __builtin_clzll(occupied_white&kings);
			uint8_t blackKingSquare = 63 - __builtin_clzll(occupied_black&kings);
			
			uint8_t kingSeparation = square_distance(63 - __builtin_clzll(occupied_white&kings),63 - __builtin_clzll(occupied_black&kings));
			if (total > 2000){
				total += (7-kingSeparation)*200;
								
				uint8_t x = whiteKingSquare % 8;
				uint8_t y = whiteKingSquare / 8;
				
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
				
				
			}else if (total < -2000){
				total -= (7-kingSeparation)*200;
				
				uint8_t x = blackKingSquare % 8;
				uint8_t y = blackKingSquare / 8;
				
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
			}
			
			uint64_t firstHalf = BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4;
			uint64_t secondHalf = BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8;
			
			int averageBlackKing_blackPawnSeperation = 0;
			int averageWhiteKing_whitePawnSeperation = 0;
			int averageBlackKing_whitePawnSeperation = 0;
			int averageWhiteKing_blackPawnSeperation = 0;
			
			uint8_t r = 0;
			uint64_t bb = firstHalf & occupied_black & pawns;
			size = 0;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;			
				averageBlackKing_blackPawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_black&kings));
				averageWhiteKing_blackPawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_white&kings));
				bb ^= (1ULL << r);
				size += 1;
			}
			
			if (size > 0){
				averageBlackKing_blackPawnSeperation /= size;
				averageWhiteKing_blackPawnSeperation /= size;
			}else{
				averageBlackKing_blackPawnSeperation = 7;
				averageWhiteKing_blackPawnSeperation = 7;
			}
			
			r = 0;
			bb = secondHalf & occupied_white & pawns;
			size = 0;
			while (bb) {
				r = 64 - __builtin_clzll(bb) - 1;			
				averageWhiteKing_whitePawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_white&kings));
				averageBlackKing_whitePawnSeperation += square_distance(r,63 - __builtin_clzll(occupied_black&kings));
				bb ^= (1ULL << r);
				size += 1;
			}
			
			if (size > 0){
				averageWhiteKing_whitePawnSeperation /= size;
				averageBlackKing_whitePawnSeperation /= size;
			}else{
				averageWhiteKing_whitePawnSeperation = 7;
				averageBlackKing_whitePawnSeperation = 7;
			}
			total += (7 - averageBlackKing_whitePawnSeperation) * 200 + (7 - averageBlackKing_blackPawnSeperation) * 250;
			total -= (7 - averageWhiteKing_blackPawnSeperation) * 200 + (7 - averageWhiteKing_whitePawnSeperation) * 250;
			//std::cout << averageBlackKing_blackPawnSeperation << " " << averageWhiteKing_whitePawnSeperation << " " <<  averageBlackKing_whitePawnSeperation << " " << averageWhiteKing_blackPawnSeperation << " " <<std::endl;
		}
	}
	
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
	pieceTypeLookUp = {};
	
	uint8_t r = 0;
	
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;			
		pieceTypeLookUp [r] = piece_type_at (r);
		bb ^= (1ULL << r);			
	} 
}

// Function to initialize the Zobrist table
void initializeZobrist() {
    std::mt19937_64 rng;  // Random number generator
    for (int pieceType = 0; pieceType < 12; ++pieceType) {
        for (int square = 0; square < 64; ++square) {
            zobristTable[pieceType][square] = rng();
        }
    }
}

// Function to generate Zobrist hash for the current board state
uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask) {
    uint64_t hash = 0;
	
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	
	std::vector<uint8_t> blackPieces;
	std::vector<uint8_t> whitePieces;
	
	scan_reversed(occupied_black,blackPieces);
    uint8_t size = blackPieces.size();
    for (uint8_t square = 0; square < size; square++) {        
		uint8_t pieceType = piece_type_at(blackPieces[square]) + 5;
		hash ^= zobristTable[pieceType][blackPieces[square]];
    }
	
	scan_reversed(occupied_white,whitePieces);
    size = whitePieces.size();
    for (uint8_t square = 0; square < size; square++) {        
		uint8_t pieceType = piece_type_at(whitePieces[square]) - 1;
		hash ^= zobristTable[pieceType][whitePieces[square]];
    }
	
    return hash;
}

// Function to update the Zobrist hash when a piece moves, with support for captures
void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bool isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion) {
    
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	
	bool fromSquareColour = bool(occupied_white & (1ULL << fromSquare));	
	uint8_t pieceType = piece_type_at(fromSquare) - 1;
	
	if (!fromSquareColour){
		pieceType += 6;
	}
	//pieceType = ;
	// XOR the moving piece out of its old position
    hash ^= zobristTable[pieceType][fromSquare];
    
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
		
		int8_t capturedPieceType = piece_type_at(toSquare) - 1;
		
		if (capturedPieceType == -1){
			
			if (fromSquareColour){
				hash ^= zobristTable[6][toSquare - 8];
			} else{
				hash ^= zobristTable[0][toSquare + 8];
			}
			
		} else{
			
			if (fromSquareColour){
				capturedPieceType += 6;
			}
			
			hash ^= zobristTable[capturedPieceType][toSquare];			
		}
		       
    }
    
	if (promotion != 0){
		
		pieceType = promotion - 1;
		
		if (!fromSquareColour){
			pieceType += 6;
		}
		
		hash ^= zobristTable[pieceType][toSquare];
	} else{
		hash ^= zobristTable[pieceType][toSquare];
	}
    // XOR the moving piece into its new position
    
}

int accessCache(uint64_t key) {
    auto it = moveCache.find(key);
    if (it != moveCache.end()) {
        return it->second;  // Return the value if the key exists
    }
    return 0;   // Return the default value if the key doesn't exist
}

void addToCache(uint64_t key,int value) {
    moveCache[key] = value;
	insertionOrder.push_back(key);
}

std::string accessOpponentMoveGenCache(uint64_t key) {
	
    auto it = OpponentMoveGenCache.find(key);
    if (it != OpponentMoveGenCache.end()) {
        return it->second;  // Return the value if the key exists
    }
    // Return a binary representation of '0' if the key doesn't exist
    char* defaultValue = new char[2]; // Allocate space for '0' and null terminator
    defaultValue[0] = '0'; // Set the first byte to '0'
    defaultValue[1] = '\0'; // Null terminator for string
    return defaultValue;

}

void addToOpponentMoveGenCache(uint64_t key,char* data, int length) {
	std::string value(data, length);
    OpponentMoveGenCache[key] = value;
	OpponentMoveGenInsertionOrder.push_back(key);
	
}

std::string accessCurPlayerMoveGenCache(uint64_t key) {
	
    auto it = curPlayerMoveGenCache.find(key);
    if (it != curPlayerMoveGenCache.end()) {
        return it->second;  // Return the value if the key exists
    }
    // Return a binary representation of '0' if the key doesn't exist
    char* defaultValue = new char[2]; // Allocate space for '0' and null terminator
    defaultValue[0] = '0'; // Set the first byte to '0'
    defaultValue[1] = '\0'; // Null terminator for string
    return defaultValue;

}

void addToCurPlayerMoveGenCache(uint64_t key,char* data, int length) {
	std::string value(data, length);
    curPlayerMoveGenCache[key] = value;
	curPlayerMoveGenInsertionOrder.push_back(key);
}

int printCacheStats() {
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
    while (numToEvict-- > 0 && !insertionOrder.empty()) {
        uint64_t oldestKey = insertionOrder.front();
        insertionOrder.pop_front();  // Remove from deque
        moveCache.erase(oldestKey);  // Erase from map
    }
}

void evictOpponentMoveGenEntries(int numToEvict) {
    while (numToEvict-- > 0 && !OpponentMoveGenInsertionOrder.empty()) {
        uint64_t oldestKey = OpponentMoveGenInsertionOrder.front();
        OpponentMoveGenInsertionOrder.pop_front();  // Remove from deque
        OpponentMoveGenCache.erase(oldestKey);  // Erase from map
    }
}

void evictCurPlayerMoveGenEntries(int numToEvict) {
    while (numToEvict-- > 0 && !curPlayerMoveGenInsertionOrder.empty()) {
        uint64_t oldestKey = curPlayerMoveGenInsertionOrder.front();
        curPlayerMoveGenInsertionOrder.pop_front();  // Remove from deque
        curPlayerMoveGenCache.erase(oldestKey);  // Erase from map
    }
}

uint8_t square_distance(uint8_t sq1, uint8_t sq2) {
    int file_distance = abs((sq1 % 8) - (sq2 % 8));
    int rank_distance = abs((sq1 / 8) - (sq2 / 8));
    return std::max(file_distance, rank_distance);
}

void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, uint64_t our_pieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t from_mask, uint64_t to_mask){
    
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	occupied = occupiedMask;
	
	uint64_t non_pawns = (our_pieces & ~pawns) & from_mask;
		
	uint8_t r = 0;
	uint64_t bb = non_pawns;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		uint64_t moves = (attacks_mask(bool((1ULL<<r) & occupied_white),occupied,r,piece_type_at (r)) & ~our_pieces) & to_mask;		
		
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			startPos.push_back(r);
			endPos.push_back(r_inner);  
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
}

void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, uint64_t occupied, bool colour, uint64_t pawnsMask, uint64_t from_mask, uint64_t to_mask){
    			
	std::vector<uint8_t> pawnVec;
	std::vector<uint8_t> pawnMoveVec;
		
	uint8_t r = 0;
	uint64_t bb = pawnsMask;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		uint64_t moves = BB_PAWN_ATTACKS[colour][r] & opposingPieces & to_mask;
		
		uint8_t r_inner = 0;
		uint64_t bb_inner = moves;
		while (bb_inner) {
			r_inner = 64 - __builtin_clzll(bb_inner) - 1;
			uint8_t rank = r_inner / 8;
			
			if (rank == 7 or rank == 0){
				
				for (int k = 5; k > 1; k--){
					startPos.push_back(r);
					endPos.push_back(r_inner);
					promotions.push_back(k);
				}
				
			} else{
				startPos.push_back(r);
				endPos.push_back(r_inner);
				promotions.push_back(1);
			}  
			bb_inner ^= (1ULL << r_inner);
		}
		
		bb ^= (1ULL << r);
	}
	
	pawnVec.clear();
	pawnMoveVec.clear();
	
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
	
	
	r = 0;
	bb = single_moves;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		uint8_t from_square = r;
		if (colour){
			from_square -= 8;
		} else{
			from_square += 8;
		}
		
		uint8_t rank = r / 8;
		
		if (rank == 7 or rank == 0){
			
			for (int j = 5; j > 1; j--){
				startPos.push_back(from_square);
				endPos.push_back(r);
				promotions.push_back(j);
			}
			
		} else{
			startPos.push_back(from_square);
			endPos.push_back(r);
			promotions.push_back(1);
		}
		bb ^= (1ULL << r);
	}
	
	
	r = 0;
	bb = double_moves;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		uint8_t from_square = r;
		if (colour){
			from_square -= 16;
		} else{
			from_square += 16;
		}
		startPos.push_back(from_square);
		endPos.push_back(r);
		promotions.push_back(1);
		bb ^= (1ULL << r);
	}
	
}

// Function to calculate sliding attacks
uint64_t sliding_attacks(uint8_t square, uint64_t occupied, const std::vector<int8_t>& deltas) {
    uint64_t attacks = 0ULL;

    for (int8_t delta : deltas) {
        uint8_t sq = square;

        while (true) {
            sq += delta;
            if (!(0 <= sq && sq < 64) || square_distance(sq, sq - delta) > 2) {
                break;
            }

            attacks |= (1ULL << sq);

            if (occupied & (1ULL << sq)) {
                break;
            }
        }
    }
    return attacks;
}

uint64_t edges(uint8_t square) {
    uint64_t rank_mask = (0xFFULL | 0xFF00000000000000ULL) & ~(0xFFULL << (8 * (square / 8)));
    uint64_t file_mask = (0x0101010101010101ULL | 0x8080808080808080ULL) & ~(0x0101010101010101ULL << square % 8);
    return rank_mask | file_mask;
}

void carry_rippler(uint64_t mask, std::vector<uint64_t> &subsets) {
    uint64_t subset = 0ULL;
    do {
        subsets.push_back(subset);
        subset = (subset - mask) & mask;
    } while (subset);
    
}

void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<std::unordered_map<uint64_t, uint64_t>> &attack_table) {
    
    for (int square = 0; square < 64; ++square) {
        std::unordered_map<uint64_t, uint64_t> attacks;
        uint64_t mask = sliding_attacks(square, 0ULL, deltas) & ~edges(square);
        std::vector<uint64_t> subsets;
		carry_rippler(mask,subsets);
        for (uint64_t subset : subsets) {
            attacks[subset] = sliding_attacks(square, subset, deltas);
        }

        mask_table.push_back(mask);
        attack_table.push_back(attacks);
    }
}

uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType){
	//uint64_t bb_square = 1ULL << square;

	if (pieceType == 1){		
		return BB_PAWN_ATTACKS[colour][square];
	}else if (pieceType == 2){
		return BB_KNIGHT_ATTACKS[square];
	}else if (pieceType == 6){
		
		return BB_KING_ATTACKS[square];
	}else{
		uint64_t attacks = 0;
		if (pieceType == 3 or pieceType == 5){
			attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & occupied];
		}
		if (pieceType == 4 or pieceType == 5){			
			attacks |= (BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & occupied] |
						BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & occupied]);
		}
		return attacks;
	}
}

void rays(std::vector<std::vector<uint64_t>> &rays) {
    
    for (size_t a = 0; a < 64; ++a) {
        std::vector<uint64_t> rays_row;
        uint64_t bb_a = 1ULL << a;

        for (size_t b = 0; b < 64; ++b) {
            uint64_t bb_b = 1ULL << b;
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
		
        rays.push_back(rays_row);
    }    
}

uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co){
    uint64_t rank_pieces = BB_RANK_MASKS[square] & occupied;
    uint64_t file_pieces = BB_FILE_MASKS[square] & occupied;
    uint64_t diag_pieces = BB_DIAG_MASKS[square] & occupied;

    uint64_t attackers = (
        (BB_KING_ATTACKS[square] & kings) |
        (BB_KNIGHT_ATTACKS[square] & knights) |
        (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
        (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
        (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[!colour][square] & pawns));

    return attackers & occupied_co;
}

uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied){    
    
    uint64_t snipers = ((BB_RANK_ATTACKS[king][0] & queens_and_rooks) |
                (BB_FILE_ATTACKS[king][0] & queens_and_rooks) |
                (BB_DIAG_ATTACKS[king][0] & queens_and_bishops));

    uint64_t blockers = 0;
	
	
	uint8_t r = 0;
	uint64_t bb = snipers & occupied_co_opp;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		uint64_t b = betweenPieces(king, r) & occupied;
        
        if (b and (1ULL << (63 - __builtin_clzll(b)) == b)){
            blockers |= b;
		}
		bb ^= (1ULL << r);
	}
	
    return blockers & occupied_co;
}
	
uint64_t betweenPieces(uint8_t a, uint8_t b){
	//std::cout << BB_RAYS.size() << std::endl;
    uint64_t bb = BB_RAYS[a][b] & ((~0x0ULL << a) ^ (~0x0ULL << b));
    return bb & (bb - 1);
}
// Function to scan reversed bitboard and return indices of set bits
uint8_t scan_reversed_size(uint64_t bb) {
    return __builtin_popcountll(bb);
}

uint64_t ray(uint8_t a, uint8_t b){return BB_RAYS[a][b];}

void scan_reversed(uint64_t bb, std::vector<uint8_t> &result){	
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        result.push_back(r);
        bb ^= (1ULL << r);
		//std::cout << result[0] << std::endl;
    }    
}

bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant){
        
	uint64_t touched = (1ULL << from_square) ^ (1ULL << to_square);
	return bool(touched & occupied_co) || is_en_passant;
}

bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces){
    uint8_t kingSquare = 63 - __builtin_clzll(~opposingPieces&kings);
	return (bool)(attackersMask(!colour, kingSquare, occupied, queens_and_rooks, queens_and_bishops, kings, knights, pawns, opposingPieces));
}

// Function to scan forward bitboard and return indices of set bits
void scan_forward(uint64_t bb, std::vector<uint8_t> &result) {
	uint64_t r = 0;
    while (bb) {
        r = bb &-bb;
        result.push_back(64 - __builtin_clzll(r) - 1);
        bb ^= r;
    }
}

int attackingScore(int layer[8][8], std::vector<uint8_t> scanSquares, uint8_t pieceType) {
    // Example: Print the contents of the array
	
	//std::vector<uint8_t> scanSquares = scan_reversed(bitmask);
	uint8_t size = scanSquares.size();
	uint8_t x,y;
	int total = 0;
	for (int i = 0; i < size; i++){        
	
		y = scanSquares[i] / 8;
		x = scanSquares[i] % 8;
		   
		
		if (pieceType == 1 or pieceType == 5){
                total += layer[x][y] >> 2;
		}else{    
			total += layer[x][y];   
		}
	}   
	return total;
}

int getPPIncrement(uint8_t square, bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x) {
    uint8_t file = square % 8;  // File of the given square
    uint8_t rank = square / 8;  // Rank of the given square
	uint8_t pos = 0 ;
    uint64_t bitmask = 0;
	
    if (colour) {
		
        // Iterate over the three relevant files
        for (int f = file - 1; f < file + 2; ++f) {
            if (f >= 0 && f <= 7) {  // Ensure the file is within bounds
                // Iterate over the ranks above the given square's rank
                for (int r = rank + 1; r < 8; ++r) {
                    pos = r * 8 + f;  // Calculate the square's position
                    bitmask |= (1ULL << pos);  // Set the bit at this position
                }
            }
        }
		//std::cout << "beFORE: " << bitmask<< std::endl;
    } else {
        // Iterate over the three relevant files
        for (int f = file - 1; f < file + 2; ++f) {
            if (f >= 0 && f <= 7) {  // Ensure the file is within bounds
                // Iterate over the ranks below the given square's rank
                for (int r = 0; r < rank; ++r) {
                    pos = r * 8 + f;  // Calculate the square's position
                    bitmask |= (1ULL << pos);  // Set the bit at this position
                }
            }
        }
    }

    bitmask &= opposingPawnMask;
		
	uint8_t r = 0;
	uint64_t bb = bitmask;
	while (bb) {
		r = 64 - __builtin_clzll(bb) - 1;
		if (r % 8 == x){
			return 0;			
		}
		ppIncrement -= 125;
		bb ^= (1ULL << r);
	}
	
	
	if (ppIncrement < 0) {
		return 0;
	}
	return ppIncrement;
	
}

std::vector<uint8_t> scan_forward_backward_multithreaded(uint64_t bb) {
    std::vector<uint8_t> result;
    std::mutex result_mutex;
    bool stop = false;

    // Forward scan thread
    auto forward_scan = [&]() {
        uint64_t local_bb = bb;
        while (local_bb && !stop) {
            uint64_t forward_mask = local_bb & -local_bb;
            uint8_t forward_index = 64 - __builtin_clzll(forward_mask) - 1;

            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result.push_back(forward_index);
            }
            
            local_bb ^= forward_mask;
            
            if (forward_index >= 32) {
                stop = true;
            }
        }
    };

    // Backward scan thread
    auto backward_scan = [&]() {
        uint64_t local_bb = bb;
        while (local_bb && !stop) {
            uint8_t reverse_index = 64 - __builtin_clzll(local_bb) - 1;

            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result.push_back(reverse_index);
            }
            
            local_bb ^= (1ULL << reverse_index);
            
            if (reverse_index < 32) {
                stop = true;
            }
        }
    };

    // Start both threads
    std::thread forward_thread(forward_scan);
    std::thread backward_thread(backward_scan);

    // Wait for both threads to finish
    forward_thread.join();
    backward_thread.join();

    return result;
}
