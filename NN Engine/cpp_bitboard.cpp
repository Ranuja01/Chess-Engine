// cpp_wrapper.cpp
#include "cpp_bitboard.h"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <mutex>
// Define the function that processes the vector of bitboards
void process_bitboards(const std::vector<uint64_t>& bitboards) {
    // Example processing logic
    for (const auto& bitboard : bitboards) {
        // Process each bitboard
        std::cout << "Processing bitboard: " << bitboard << std::endl;
    }
}


void process_bitboards_wrapper(const uint64_t* bitboards, size_t size) {
        std::vector<uint64_t> vec(bitboards, bitboards + size);
        process_bitboards(vec);
}

// Function to find the most significant bit positions
std::vector<int> find_most_significant_bits(uint64_t bitmask) {
    std::vector<int> msb_positions;

    // Loop through each bit in the bitmask to find the MSBs
    for (int i = 63; i >= 0; --i) {
        // Check if the current bit is set
        if (bitmask & (1ULL << i)) {
            //std::cout << "Processing bitboard: " << i<< std::endl;
	    msb_positions.push_back(i);
        }
    }
	return msb_positions;
}

// Function to get the bit length of a Bitboard
int bit_length(uint64_t bb) {
    return (bb == 0) ? 0 : 64 - __builtin_clzll(bb);
}

// Function to create a mask for a specific square index
uint64_t make_square_mask(int index) {
    return (1ULL << index);
}

// Function to scan reversed bitboard and return indices of set bits
std::vector<uint8_t> scan_reversed(uint64_t bb) {
    std::vector<uint8_t> result;
	uint8_t r = 0;
    while (bb) {
        r = 64 - __builtin_clzll(bb) - 1;
        result.push_back(r);
        bb ^= (1ULL << r);
    }
    return result;
}

// Function to scan forward bitboard and return indices of set bits
std::vector<uint8_t> scan_forward(uint64_t bb) {
    std::vector<uint8_t> result;
	uint64_t r = 0;
    while (bb) {
        r = bb &-bb;
        result.push_back(64 - __builtin_clzll(r) - 1);
        bb ^= r;
    }
    return result;
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
	
	
	std::vector<uint8_t> scanSquares = scan_reversed(bitmask);
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
