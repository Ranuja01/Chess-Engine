#ifndef CPP_BITBOARD_H
#define CPP_BITBOARD_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

uint8_t square_distance(uint8_t sq1, uint8_t sq2);
void initialize_attack_tables();
uint8_t square_distance(uint8_t sq1, uint8_t sq2);
uint64_t sliding_attacks(uint8_t square, uint64_t occupied, const std::vector<int8_t>& deltas);
uint64_t edges(uint8_t square);
void carry_rippler(uint64_t mask, std::vector<uint64_t> &subsets);
void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<std::unordered_map<uint64_t, uint64_t>> &attack_table);
void rays(std::vector<std::vector<uint64_t>> &rays);

uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType);
uint64_t attackersMask(bool color, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co);
uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied);
uint64_t betweenPieces(uint8_t a, uint8_t b);
uint64_t ray(uint8_t a, uint8_t b);
bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant);

uint8_t scan_reversed_size(uint64_t bb);
void scan_reversed(uint64_t bb, std::vector<uint8_t> &result);
std::vector<uint8_t> scan_reversedOld(uint64_t bb);
void scan_forward(uint64_t bb, std::vector<uint8_t> &result);
int getPPIncrement(uint8_t square, bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x);
int attackingScore(int layer[8][8], std::vector<uint8_t> scanSquares, uint8_t pieceType);
bool isCapture(uint64_t bb1, uint64_t bb2);
std::vector<uint8_t> scan_forward_backward_multithreaded(uint64_t bb);
#endif // CPP_BITBOARD_H
