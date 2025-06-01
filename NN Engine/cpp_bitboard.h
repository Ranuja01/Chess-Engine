#ifndef CPP_BITBOARD_H
#define CPP_BITBOARD_H

#include "search_engine.h"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstring>
#include <optional>

constexpr int NUM_SQUARES = 64;

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

// Create a compile-time array of bitmasks
constexpr std::array<uint64_t, 64> generate_square_masks() {
    std::array<uint64_t, 64> masks{};
    for (int i = 0; i < 64; i++) {
        masks[i] = 1ULL << i;
    }
    return masks;
}

// Global constant array of square bitmasks
constexpr std::array<uint64_t, 64> BB_SQUARES = generate_square_masks();

struct CaptureInfo {
    uint8_t from;
	uint8_t to;
    int value_gained;    

	CaptureInfo(uint8_t from_square, uint8_t to_square, int value) : from(from_square), to(to_square), value_gained(value) {}
};

struct MaskPair {
    uint64_t from_mask;
	uint64_t to_mask;

	MaskPair(uint64_t from, uint64_t to) : from_mask(from), to_mask(to) {}
};

bool get_horizon_mitigation_flag();
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
void update_pressure_and_support_tables(uint8_t current_square, uint8_t attacking_piece_type, uint8_t decrement, bool attacking_piece_colour, bool current_piece_colour);
void handle_batteries_for_pressure_and_support_tables(uint8_t attacking_piece_square, uint8_t attacking_piece_type, uint64_t prev_attack_mask, bool attacking_piece_colour);
void loop_and_update(uint64_t bb, uint8_t attacking_piece_type, bool attacking_piece_colour, int decrement);
void adjust_pressure_and_support_tables_for_pins(uint64_t bb);
int advanced_endgame_eval(int total, bool turn);
int placement_and_piece_eval(int moveNum, bool turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied);
int get_pressure_increment(uint8_t last_moved_to_square, uint64_t bb, bool turn);

uint8_t lowest_value_attacker(uint64_t attackers, bool attackedColour);
void apply_basic_capture(uint8_t from, uint8_t to, uint64_t& white_pieces, uint64_t& black_pieces, bool white_to_move);
CaptureInfo* find_last_viable_capture(std::vector<CaptureInfo>& captures, uint64_t& white_pieces, uint64_t& black_pieces, bool captureColour);
std::optional<CaptureInfo> find_and_pop_last_viable_capture(std::vector<CaptureInfo>& captures, uint64_t white_pieces, uint64_t black_pieces, bool captureColour);
bool can_evade(uint8_t target_square, bool target_colour);
int approximate_capture_gains(uint64_t bb, bool turn);
int approximate_capture_gains1(uint64_t bb, bool turn);

void initializePieceValues(uint64_t bb);
uint8_t piece_type_at(uint8_t squareuint8_t);
void setAttackingLayer(int increment);
void printLayers();
int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces);

/*
	Set of functions used to cache data
*/
void initializeZobrist();
uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, bool whiteToMove);
void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bool isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion);
int accessCache(uint64_t key);
void addToCache(uint64_t key,int value);
int printCacheStats();
int getCacheStats();
void evictOldEntries(int numToEvict);

std::vector<Move> accessMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square);
void addToMoveGenCache(uint64_t key, std::vector<Move> reorderedMoves, uint64_t castling_rights, int ep_square);
void evictOldMoveGenEntries(int numToEvict);
int printMoveGenCacheStats();
uint64_t hash_castling(uint64_t castling_rights);
uint64_t make_move_cache_key(uint64_t zobrist_base, uint64_t castling_rights, int ep_square);


/*
	Set of functions used to generate moves
*/
void generateLegalMoves(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

void generatePseudoLegalMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 						  uint64_t king, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
							  uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
							  
void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t our_pieces, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask,
	 					uint64_t occupiedWhite, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask);

void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, bool colour, uint64_t pawnsMask, uint64_t occupiedMask,
					   uint64_t from_mask, uint64_t to_mask);

void generateCastlingMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint64_t to_mask, uint64_t king,
	                       uint64_t opposingPieces, uint64_t occupiedMask, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, bool turn);

void generateEnPassentMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t from_mask, uint64_t to_mask, uint64_t our_pieces, uint64_t occupiedMask, uint64_t pawnsMask, int ep_square, bool turn);

void generateEvasions(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint8_t king, uint64_t checkers, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces,
					  uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

void generateLegalCaptures(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

void generateLegalMovesReordered(std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
								 
template<std::size_t N>
void processMaskPairs(const std::array<MaskPair, N>& mask_pairs, std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask,
	                  uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
                      uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co);
uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied);
uint64_t betweenPieces(uint8_t a, uint8_t b);
uint64_t ray(uint8_t a, uint8_t b);
void update_bitmasks(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask);
bool is_into_check(uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, 
	               uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

bool is_safe(uint8_t king, uint64_t blockers, uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
			 uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
bool is_castling(uint8_t from_square, uint8_t to_square, bool turn, uint64_t ourPieces, uint64_t rooksMask, uint64_t kingsMask);
bool is_en_passant(uint8_t from_square, uint8_t to_square, int ep_square, uint64_t occupiedMask, uint64_t pawnsMask);
uint64_t pin_mask(bool colour, int square, uint8_t king, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask);
bool ep_skewered(int king, int capturer, int ep_square, bool turn, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask);
bool attackedForKing(bool opponent_color,uint64_t path, uint64_t occupied, uint64_t opposingPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask);

/*
	Set of functions used as utilities for all above functions
*/
bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant);
bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces);

bool is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn);
bool is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn);
uint8_t scan_reversed_size(uint64_t bb);
void scan_reversed(uint64_t bb, std::vector<uint8_t> &result);
std::vector<uint8_t> scan_reversedOld(uint64_t bb);
void scan_forward(uint64_t bb, std::vector<uint8_t> &result);
uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType);
uint8_t square_distance(uint8_t sq1, uint8_t sq2);

int remove_piece_at(uint8_t square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted);
void set_piece_at(uint8_t square, uint8_t piece_type, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, bool promotedFlag, bool turn);
void update_state(uint8_t to_square, uint8_t from_square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, uint64_t& castling_rights, int& ep_square, int promotion_type, bool turn);
std::string create_fen(uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask,
                       uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask,
                       uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack,
                       uint64_t& promoted, uint64_t& castling_rights, int& ep_square, bool turn);

#endif // CPP_BITBOARD_H
