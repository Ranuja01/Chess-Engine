#ifndef CPP_BITBOARD_H
#define CPP_BITBOARD_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstring>

/*
	Set of functions to initialize masks for move generation
*/
void initialize_attack_tables();
void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<std::unordered_map<uint64_t, uint64_t>> &attack_table);
uint64_t sliding_attacks(uint8_t square, uint64_t occupied, const std::vector<int8_t>& deltas);
void carry_rippler(uint64_t mask, std::vector<uint64_t> &subsets);
void rays(std::vector<std::vector<uint64_t>> &rays);
uint64_t edges(uint8_t square);

/*
	Set of functions directly used to evaluate the position
*/
int placement_and_piece_midgame(uint8_t square);
int placement_and_piece_endgame(uint8_t square);
int placement_and_piece_eval(int moveNum, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t prevKings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied);
void initializePieceValues(uint64_t bb);
uint8_t piece_type_at(uint8_t squareuint8_t);
void setAttackingLayer(int increment);
void printLayers();
int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces);

/*
	Set of functions used to cache data
*/
void initializeZobrist();
uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask);
void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bool isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion);
int accessCache(uint64_t key);
void addToCache(uint64_t key,int value);
std::string accessOpponentMoveGenCache(uint64_t key);
void addToOpponentMoveGenCache(uint64_t key,char* data, int length);
std::string accessCurPlayerMoveGenCache(uint64_t key);
void addToCurPlayerMoveGenCache(uint64_t key,char* data, int length);
int printCacheStats();
int printOpponentMoveGenCacheStats();
int printCurPlayerMoveGenCacheStats();
void evictOldEntries(int numToEvict);
void evictOpponentMoveGenEntries(int numToEvict);
void evictCurPlayerMoveGenEntries(int numToEvict);

/*
	Set of functions used to generate moves
*/
void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, uint64_t our_pieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, uint64_t from_mask, uint64_t to_mask);
void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, uint64_t occupied, bool colour, uint64_t pawnsMask, uint64_t from_mask, uint64_t to_mask);
uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co);
uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied);
uint64_t betweenPieces(uint8_t a, uint8_t b);
uint64_t ray(uint8_t a, uint8_t b);

/*
	Set of functions used as utilities for all above functions
*/
bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant);
bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces);
uint8_t scan_reversed_size(uint64_t bb);
void scan_reversed(uint64_t bb, std::vector<uint8_t> &result);
std::vector<uint8_t> scan_reversedOld(uint64_t bb);
void scan_forward(uint64_t bb, std::vector<uint8_t> &result);
uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType);
uint8_t square_distance(uint8_t sq1, uint8_t sq2);

#endif // CPP_BITBOARD_H
