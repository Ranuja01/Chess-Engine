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

constexpr std::array<int, 7> values = {0, 1000, 3150, 3250, 5000, 9000, 0};

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

uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

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
			{{0,0,0,0,0,0,0,0}},
			{{0,0,3,3,4,5,5,0}},
			{{0,0,3,6,7,6,4,0}},
			{{0,0,3,15,20,8,5,0}},
			{{0,0,3,15,20,8,5,0}},
			{{0,0,3,6,7,6,4,0}},
			{{0,0,3,3,4,5,5,0}},
			{{0,0,0,0,0,0,0,0}}
		}},
		{{
			{{0,0,0,0,0,0,0,0}},
			{{0,5,5,4,3,3,0,0}},
			{{0,4,6,7,6,3,0,0}},
			{{0,5,8,20,15,3,0,0}},
			{{0,5,8,20,15,3,0,0}},
			{{0,4,6,7,6,3,0,0}},
			{{0,5,5,4,3,3,0,0}},
			{{0,0,0,0,0,0,0,0}}
		}}
	}};
	
	std::vector<uint8_t> outterScanSquares;
	std::vector<uint8_t> innerScanSquares;
	uint8_t x,y;
	scan_reversed(attacks_mask(true,0ULL,63 - __builtin_clzll(occupied_white&kings),6),outterScanSquares);
	uint8_t innerSize;
	uint8_t outterSize = outterScanSquares.size();
	
	for (int i = 0; i < outterSize; i++){
    
        y = outterScanSquares[i] / 8;
        x = outterScanSquares[i] % 8;
        attackingLayer[1][x][y] += increment;
        
		innerScanSquares.clear();
		scan_reversed(attacks_mask(true,0ULL,outterScanSquares[i],6),innerScanSquares);
        innerSize = innerScanSquares.size();
		
        for (int j = 0; j < innerSize; j++){
            
            y = innerScanSquares[j] / 8;
			x = innerScanSquares[j] % 8;
			attackingLayer[1][x][y] += increment;
		}
    }
	
	outterScanSquares.clear();
	scan_reversed(attacks_mask(false,0ULL,63 - __builtin_clzll(occupied_black&kings),6),outterScanSquares);
	outterSize = outterScanSquares.size();
	
	for (int i = 0; i < outterSize; i++){
    
        y = outterScanSquares[i] / 8;
        x = outterScanSquares[i] % 8;
        attackingLayer[0][x][y] += increment;
        
		innerScanSquares.clear();
		scan_reversed(attacks_mask(true,0ULL,outterScanSquares[i],6),innerScanSquares);
        innerSize = innerScanSquares.size();
		
        for (int j = 0; j < innerSize; j++){
            
            y = innerScanSquares[j] / 8;
			x = innerScanSquares[j] % 8;
			attackingLayer[0][x][y] += increment;
		}
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
	uint8_t piece_type = piece_type_at (square);
	bool colour = bool(occupied_white & (1ULL << square)); 
    uint8_t size;
    
    std::vector<uint8_t> attackVec;

    uint8_t y = square / 8;
    uint8_t x = square % 8;
        
    if (colour) {
        total -= values[piece_type];
        
        if (! (piece_type == 4 || piece_type == 6)){
            
            total -= placementLayer[0][x][y];
            
            if (piece_type == 2 || piece_type == 3){
                total -= 500;
            }
			
            if (piece_type == 1){
                total -= (y + 1) * 15;
                total -= attackingLayer[1][x][y] << 2;                                
                if (scan_reversed_size((BB_FILES[x] & (occupied_white & pawns))) > 1) {                
                    total += 200;
				}
                total -= getPPIncrement(square, colour, (occupied_black & pawns), ppIncrement, x);
			}
        }else if (piece_type == 4){  
            
            if (y == 6){
                rookIncrement += 50;
			}
			
            rooks_mask |= BB_FILES[x] & occupied;            
            
			std::vector<uint8_t> pieceVec;            
            scan_forward(rooks_mask,pieceVec);
            size = pieceVec.size();
            
            for (int i = 0; i < size; i++){             
                uint8_t att_square = pieceVec[i];
                if (att_square > square){
                    uint8_t temp_piece_type = piece_type_at (att_square);
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
        
        scan_reversed(attacks_mask(colour,occupied,square,piece_type),attackVec);
        size = attackVec.size();
        
        for (int i = 0; i < size; i++){  
            y = attackVec[i] / 8;
            x = attackVec[i] % 8;
            if (piece_type == 1 || piece_type == 5){
                total -= attackingLayer[0][x][y] >> 2;
            }else{    
                total -= attackingLayer[0][x][y];   
			}
		}
    }else{
        total += values[piece_type];
        if (! (piece_type == 4 || piece_type == 6)){
            
            total += placementLayer[1][x][y];
            
            if (piece_type == 2 || piece_type == 3){
                total += 500;
            }
            if (piece_type == 1){
                total += (8 - y) * 15;
                total += attackingLayer[0][x][y] << 2;
                if (scan_reversed_size((BB_FILES[x] & (occupied_black & pawns))) > 1){                
                    total -= 200;
                }
                total += getPPIncrement(square, colour, (occupied_white & pawns), ppIncrement, x);
			}
                
        }else if (piece_type == 4){
            if (y == 1){
                rookIncrement += 50;
			}
            rooks_mask |= BB_FILES[x] & occupied;            
            
			std::vector<uint8_t> pieceVec;            
            scan_reversed(rooks_mask,pieceVec);
            size = pieceVec.size();
            
            for (int i = 0; i < size; i++){              
                uint8_t att_square = pieceVec[i];
                if (att_square < square){
                    uint8_t temp_piece_type = piece_type_at (att_square);
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
								//std::cout << "AAA" << std::endl;
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
        scan_reversed(attacks_mask(colour,occupied,square,piece_type),attackVec);
        size = attackVec.size();
        
        for (int i = 0; i < size; i++){        
            y = attackVec[i] / 8;
            x = attackVec[i] % 8;
            if (piece_type == 1 || piece_type == 5){
                total += attackingLayer[1][x][y] >> 2;
			}else{    
                total += attackingLayer[1][x][y];       
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
	uint8_t piece_type = piece_type_at (square);
	bool colour = bool(occupied_white & (1ULL << square)); 
    uint8_t size;
    
    std::vector<uint8_t> attackVec;

    uint8_t y = square / 8;
    uint8_t x = square % 8;
        
    if (colour) {
        total -= values[piece_type];
        
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
				total -= (y + 1) * 50 + (y + 1) * (y + 1);
			} else {
				total -= (y + 1) * 50;
			}
			
			
		}else if (piece_type == 4){  
            
            rooks_mask |= BB_FILES[x] & occupied;            
            
			std::vector<uint8_t> pieceVec;            
            scan_forward(rooks_mask,pieceVec);
            size = pieceVec.size();
            
            for (int i = 0; i < size; i++){             
                uint8_t att_square = pieceVec[i];
				uint8_t temp_piece_type = piece_type_at (att_square);
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                if (temp_piece_type == 1){
                    if (att_square > square){
                        if (temp_colour){
                            rookIncrement += ((att_square / 8) + 1) * 25;
                        }else{
                            rookIncrement += (y + 1) * 15;
						}
                    }else {
                        if (temp_colour){
                            if (att_square / 8 > 3){
                                rookIncrement -= 50 + ((att_square / 8) - 3) * 50;
							}
                        }else{
                            rookIncrement += (y + 1) * 10;
						}
					}
				}
            }
			total -= rookIncrement;
        }
        
    }else{
        total += values[piece_type];
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
                total += (8 - y) * 50 + (8 - y) * (8 - y);
            }else{
                total += (8 - y) * 50;     
			}
			
		}else if (piece_type == 4){
            
            rooks_mask |= BB_FILES[x] & occupied;            
            
			std::vector<uint8_t> pieceVec;            
            scan_reversed(rooks_mask,pieceVec);
            size = pieceVec.size();
            
            for (int i = 0; i < size; i++){              
                uint8_t att_square = pieceVec[i];
				uint8_t temp_piece_type = piece_type_at (att_square);
				bool temp_colour = bool(occupied_white & (1ULL << att_square));
                if (temp_piece_type == 1){    
                    if (att_square < square){
                        if (temp_colour){
                            rookIncrement += (8 - y) * 15;                            
                        }else{
                            rookIncrement += (8 - (att_square / 8)) * 25;
						}
                    }else{
                        if (temp_colour){
                            rookIncrement += (8 - y) * 10;
						}else{
                            if (att_square / 8 < 4){
                                rookIncrement -= 50 + (4 - (att_square / 8)) * 50;
							}
						}
					}
				}
			}
            total += rookIncrement;
        }
    }
	
	if (colour){
        
        if (total < -7500){
            attackMultiplier = 2;
        }else if(total < -15000){
            attackMultiplier = 3;
        }
		scan_reversed(attacks_mask(colour,occupied,square,piece_type),attackVec);       
        size = attackVec.size();
        
        for (int i = 0; i < size; i++){         
            y = attackVec[i] / 8;
            x = attackVec[i] % 8 ;           
            total -= attackingLayer[0][x][y] * attackMultiplier;   
		}
    }else{
        
        if (total < 7500){
            attackMultiplier = 2;
        }else if(total < 15000){
            attackMultiplier = 3;
		}
        
        scan_reversed(attacks_mask(colour,occupied,square,piece_type),attackVec);
        size = attackVec.size();
        
        for (int i = 0; i < size; i++){                 
            y = attackVec[i] / 8;
            x = attackVec[i] % 8;
            total += attackingLayer[1][x][y] * attackMultiplier;
		}
	}
	return total;
}

int placement_and_piece_eval(int moveNum, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask){
	int total = 0;
	
	pawns = pawnsMask;
	knights = knightsMask;
	bishops = bishopsMask;
	rooks = rooksMask;
	queens = queensMask;
	kings = kingsMask;
	occupied_white = occupied_whiteMask;
	occupied_black = occupied_blackMask;
	occupied = occupiedMask;
	
	if (moveNum <= 50){
        std::vector<uint8_t> pieceVec;            
		scan_forward(occupied,pieceVec);
		uint8_t size = pieceVec.size();
		
		for (int i = 0; i < size; i++){ 
			total += placement_and_piece_midgame(pieceVec[i]);
		}
	}else{
		std::vector<uint8_t> pieceVec;            
		scan_forward(occupied,pieceVec);
		uint8_t size = pieceVec.size();
		//-flto=8
		for (int i = 0; i < size; i++){ 
			total += placement_and_piece_endgame(pieceVec[i]);
		}
		
		if (moveNum >= 70){
			uint8_t kingSeparation = square_distance(63 - __builtin_clzll(occupied_white&kings),63 - __builtin_clzll(occupied_black&kings&kings));
			if (total > 2500){
				total += (7-kingSeparation)*200;
			}else if (total < -2500){
				total -= (7-kingSeparation)*200;
			}
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
	return total;
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
	std::vector<uint8_t> pieceVec;
	std::vector<uint8_t> pieceMoveVec;
	
    scan_reversed(non_pawns,pieceVec);
    uint8_t outterSize = pieceVec.size();
    
    for (int i = 0; i < outterSize; i++){ 
        
        uint64_t moves = (attacks_mask(bool((1ULL<<pieceVec[i]) & occupied_white),occupied,pieceVec[i],piece_type_at (pieceVec[i])) & ~our_pieces) & to_mask;
        //std::cout << moves << " " << bool((1ULL<<pieceVec[i]) & occupied_white) << " " << occupied << " " << pieceVec[i] << " " <<   << std::endl;
        pieceMoveVec.clear();
        scan_reversed(moves,pieceMoveVec);
        uint8_t innerSize = pieceMoveVec.size();
        
        for (int j = 0; j < innerSize; j++){ 
			startPos.push_back(pieceVec[i]);
			endPos.push_back(pieceMoveVec[j]);            
		}
	}
}

void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, uint64_t occupied, bool colour, uint64_t pawnsMask, uint64_t from_mask, uint64_t to_mask){
    			
	std::vector<uint8_t> pawnVec;
	std::vector<uint8_t> pawnMoveVec;
	
    scan_reversed(pawnsMask,pawnVec);
    uint8_t outterSize = pawnVec.size();
    
    for (int i = 0; i < outterSize; i++){ 
        
        uint64_t moves = BB_PAWN_ATTACKS[colour][pawnVec[i]] & opposingPieces & to_mask;
        //std::cout << moves << " " << bool((1ULL<<pieceVec[i]) & occupied_white) << " " << occupied << " " << pieceVec[i] << " " <<   << std::endl;
        pawnMoveVec.clear();
        scan_reversed(moves,pawnMoveVec);
        uint8_t innerSize = pawnMoveVec.size();
        
        for (int j = 0; j < innerSize; j++){ 
			
			uint8_t rank = pawnMoveVec[j] / 8;
			
			if (rank == 7 or rank == 0){
				
				for (int k = 5; k > 1; k--){
					startPos.push_back(pawnVec[i]);
					endPos.push_back(pawnMoveVec[j]);
					promotions.push_back(k);
				}
				
			} else{
				startPos.push_back(pawnVec[i]);
				endPos.push_back(pawnMoveVec[j]);
				promotions.push_back(1);
			}        
		}
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
	
	scan_reversed(single_moves,pawnVec);
    outterSize = pawnVec.size();
	
	for (int i = 0; i < outterSize; i++){ 
		
		uint8_t from_square = pawnVec[i];
		if (colour){
			from_square -= 8;
		} else{
			from_square += 8;
		}
		
		uint8_t rank = pawnVec[i] / 8;
		
		if (rank == 7 or rank == 0){
			
			for (int j = 5; j > 1; j--){
				startPos.push_back(from_square);
				endPos.push_back(pawnVec[i]);
				promotions.push_back(j);
			}
			
		} else{
			startPos.push_back(from_square);
			endPos.push_back(pawnVec[i]);
			promotions.push_back(1);
		}        
	}
	
	pawnVec.clear();

	scan_reversed(double_moves,pawnVec);
    outterSize = pawnVec.size();
	
	for (int i = 0; i < outterSize; i++){ 
		
		uint8_t from_square = pawnVec[i];
		if (colour){
			from_square -= 16;
		} else{
			from_square += 16;
		}
		startPos.push_back(from_square);
		endPos.push_back(pawnVec[i]);
		promotions.push_back(1);
		        
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

uint64_t attackersMask(bool color, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co){
    uint64_t rank_pieces = BB_RANK_MASKS[square] & occupied;
    uint64_t file_pieces = BB_FILE_MASKS[square] & occupied;
    uint64_t diag_pieces = BB_DIAG_MASKS[square] & occupied;

    uint64_t attackers = (
        (BB_KING_ATTACKS[square] & kings) |
        (BB_KNIGHT_ATTACKS[square] & knights) |
        (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
        (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
        (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[!color][square] & pawns));

    return attackers & occupied_co;
}

uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied){
    std::vector<uint8_t>  vec;
    
    uint64_t snipers = ((BB_RANK_ATTACKS[king][0] & queens_and_rooks) |
                (BB_FILE_ATTACKS[king][0] & queens_and_rooks) |
                (BB_DIAG_ATTACKS[king][0] & queens_and_bishops));

    uint64_t blockers = 0;
		
    scan_reversed(snipers & occupied_co_opp,vec);
    uint8_t size = vec.size();
    for (int i = 0; i < size; i++){
		//std::cout << "AAAAA" << std::endl;
        uint64_t b = betweenPieces(king, vec[i]) & occupied;
        
        if (b and (1ULL << (63 - __builtin_clzll(b)) == b)){
            blockers |= b;
		}
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
    uint8_t count = 0;
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        bb ^= (1ULL << r);
		count += 1;
    }
    return count;
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
	
	std::vector<uint8_t> scanSquares;
	scan_reversed(bitmask,scanSquares);
	uint8_t size = scanSquares.size();
	for (int i = 0; i < size; i++){
		if (scanSquares[i] % 8 == x){
			return 0;			
		}
		ppIncrement -= 125;
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
