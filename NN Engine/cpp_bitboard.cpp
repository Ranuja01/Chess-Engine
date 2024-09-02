// cpp_wrapper.cpp
#include "cpp_bitboard.h"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_map>

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

// Function to calculate the distance between squares (example implementation)
uint8_t square_distance(uint8_t sq1, uint8_t sq2) {
    int file_distance = abs((sq1 % 8) - (sq2 % 8));
    int rank_distance = abs((sq1 / 8) - (sq2 / 8));
    return std::max(file_distance, rank_distance);
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
		//std::cout << BB_KING_ATTACKS[square] << std::endl;
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
	//std::cout << "AAAAA" << std::endl;
	//std::cout << bb << std::endl;
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        result.push_back(r);
        bb ^= (1ULL << r);
		//std::cout << result[0] << std::endl;
    }    
}

bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant){
        
	uint64_t touched = (1 << from_square) ^ (1 << to_square);
	return bool(touched & occupied_co) || is_en_passant;
}
std::vector<uint8_t> scan_reversedOld(uint64_t bb){	
	//std::cout << "AAAAA" << std::endl;
	//std::cout << bb << std::endl;
	std::vector<uint8_t> result;
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        result.push_back(r);		
        bb ^= (1ULL << r);
		//std::cout << result[0] << std::endl;
    }
	return result;
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

bool isCapture(uint64_t bb1, uint64_t bb2) {
    return __builtin_popcountll(bb2) < __builtin_popcountll(bb1); // Returns true if bb1 has fewer set bits than bb2
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
