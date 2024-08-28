#ifndef CPP_BITBOARD_H
#define CPP_BITBOARD_H

#include <vector>
#include <cstddef>
#include <cstdint>

// Function declarations
void process_bitboards(const std::vector<uint64_t>& bitboards);
void process_bitboards_wrapper(const uint64_t* bitboards, size_t size);
std::vector<int> find_most_significant_bits(uint64_t bitmask);

// Function to get the bit length (number of bits) of a Bitboard
int bit_length(uint64_t bb);

// Function to create a mask for a specific square index
uint64_t make_square_mask(int index);

// Function to scan reversed bitboard and return indices of set bits
std::vector<uint8_t> scan_reversed(uint64_t bb);
std::vector<uint8_t> scan_forward(uint64_t bb);
int getPPIncrement(uint8_t square, bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x);
int attackingScore(int layer[8][8], std::vector<uint8_t> scanSquares, uint8_t pieceType);
bool isCapture(uint64_t bb1, uint64_t bb2);
std::vector<uint8_t> scan_forward_backward_multithreaded(uint64_t bb);
#endif // CPP_BITBOARD_H
