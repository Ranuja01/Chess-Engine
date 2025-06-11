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
#include <sstream>

constexpr int NUM_SQUARES = 64;
constexpr int MAX_PLY = 64;

// Define masks for move generation
extern std::array<uint64_t, NUM_SQUARES> BB_KNIGHT_ATTACKS;
extern std::array<uint64_t, NUM_SQUARES> BB_KING_ATTACKS;
extern std::array<std::array<uint64_t, NUM_SQUARES>, 2> BB_PAWN_ATTACKS;
extern std::vector<uint64_t> BB_DIAG_MASKS;
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_DIAG_ATTACKS;
extern std::vector<uint64_t> BB_FILE_MASKS;
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_FILE_ATTACKS;
extern std::vector<uint64_t> BB_RANK_MASKS;
extern std::vector<std::unordered_map<uint64_t, uint64_t>> BB_RANK_ATTACKS;
extern std::vector<std::vector<uint64_t>> BB_RAYS;


// Define global masks for piece placement
extern uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

constexpr uint8_t PAWN = 1;
constexpr uint8_t KNIGHT = 2;
constexpr uint8_t BISHOP = 3;
constexpr uint8_t ROOK = 4;
constexpr uint8_t QUEEN = 5;
constexpr uint8_t KING = 6;


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

constexpr std::array<int, 8> default_midgame_pawn_rank_bonus = {0, 15, 30, 45, 60, 75, 90, 105};
//constexpr std::array<int, 8> passed_midgame_pawn_rank_bonus = {0, 65, 160, 285, 440, 625, 840, 1085};
constexpr std::array<int, 8> passed_midgame_pawn_rank_bonus = {0, 65, 160, 285, 625, 840, 1085, 1360};

//constexpr std::array<int, 8> endgame_pawn_rank_bonus = {0, 90, 210, 360, 540, 750, 990, 1260};
constexpr std::array<int, 8> endgame_pawn_rank_bonus = {0, 90, 210, 360, 750, 990, 1260, 1560};

constexpr std::array<uint8_t, 11> pawn_wall_file_bonus = {
    0,    // x - 1 invalid (x == 0)
    75,   // x == 0
    50,   // x == 1
    60,   // x == 2
    75,   // x == 3
    75,   // x == 4
    60,   // x == 5
    50,   // x == 6
    75,   // x == 7
    0,    // x + 1 invalid (x == 7)
    0     // safety pad
};

constexpr std::array<uint8_t, 8> pawn_chain_file_bonus = {
    10,   // A
    15,   // B
    100,   // C
    150,  // D
    150,  // E
    100,   // F
    15,   // G
    10    // H
};

constexpr std::array<uint64_t, 8> white_king_zones = {
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // A file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // B file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // C file
    
    (BB_FILE_B | BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // D file
    (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F | BB_FILE_G) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // E file

    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // F file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // G file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6)  // H file
};

constexpr std::array<uint64_t, 8> black_king_zones = {
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // A file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // B file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // C file
    
    (BB_FILE_B | BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // D file
    (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F | BB_FILE_G) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // E file

    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // F file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // G file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3)  // H file
};


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

constexpr uint64_t central_squares = BB_SQUARES[27] | BB_SQUARES[28] | BB_SQUARES[35] | BB_SQUARES[36];

constexpr uint64_t extended_central_squares =
    BB_SQUARES[18] | BB_SQUARES[19] | BB_SQUARES[20] | BB_SQUARES[21] |
    BB_SQUARES[26] | BB_SQUARES[29] |
    BB_SQUARES[34] | BB_SQUARES[37] |
    BB_SQUARES[42] | BB_SQUARES[43] | BB_SQUARES[44] | BB_SQUARES[45];

constexpr int rook_squares[4] = {0, 7, 56, 63};

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
void update_global_central_scores(int base_increment, uint64_t square_mask);
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
uint8_t piece_type_at(uint8_t square);
void setAttackingLayer(int increment, bool isEndGame);
void printLayers();
int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces);

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



inline uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co){
    
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

inline uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied){    
    
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
	
inline uint64_t betweenPieces(uint8_t a, uint8_t b){
		
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

inline uint64_t ray(uint8_t a, uint8_t b){return BB_RAYS[a][b];}


inline bool is_into_check(uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, 
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

inline bool is_safe(uint8_t king, uint64_t blockers, uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
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

inline bool is_castling(uint8_t from_square, uint8_t to_square, bool turn, uint64_t ourPieces, uint64_t rooksMask, uint64_t kingsMask) {

    if (kingsMask & BB_SQUARES[from_square]) {
        int diff = (from_square & 7) - (to_square & 7);
        return (std::abs(diff) > 1) ||
               ((rooksMask & ourPieces) & BB_SQUARES[to_square]);
    }
    return false;
}

inline bool is_en_passant(uint8_t from_square, uint8_t to_square, int ep_square, uint64_t occupiedMask, uint64_t pawnsMask) {
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

inline uint64_t pin_mask(bool colour, int square, uint8_t king, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask) {
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

inline bool ep_skewered(int king, int capturer, int ep_square, bool turn, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask) {
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

inline bool attackedForKing(bool opponent_color,uint64_t path, uint64_t occupied, uint64_t opposingPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask){
    
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


/*
	Set of functions used as utilities for all above functions
*/
inline void scan_reversed(uint64_t bb, std::vector<uint8_t> &result){	
		
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

inline bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool is_en_passant){
    	
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

inline bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces){
		
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

inline bool is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
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

inline bool is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
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

inline void scan_forward(uint64_t bb, std::vector<uint8_t> &result) {
		
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

inline uint8_t scan_reversed_size(uint64_t bb) {
		
	/*
		Function to acquires the number of set bits in a bitmask
		
		Parameters:
		- bb: The bitmask to be examined
	*/
	
    return __builtin_popcountll(bb);
}

inline uint8_t square_distance(uint8_t sq1, uint8_t sq2) {
		
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

inline int remove_piece_at(uint8_t square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted){
	
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

inline void set_piece_at(uint8_t square, uint8_t piece_type, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, bool promotedFlag, bool turn){
	
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
        
inline void update_state(uint8_t to_square, uint8_t from_square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, uint64_t& castling_rights, int& ep_square, int promotion_type, bool turn){
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

	int ep_copy = ep_square;
	ep_square = -1;
    if (piece_type == 1){				
		
		int diff = to_square - from_square;

		if(diff == 16 && (from_square >> 3) == 1)
			ep_square = from_square + 8;
		else if(diff == -16 && (from_square >> 3) == 6){
			ep_square = from_square - 8;
			/* if (from_square == 53 && to_square == 37)
                std::cout << occupiedMask << " | " << turn << " | " << ep_square << " | " << pawnsMask << std::endl;  */
			//std::cout << "AAAA" << std::endl;
		}
			
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

inline std::string create_fen(uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask,
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



#endif // CPP_BITBOARD_H
