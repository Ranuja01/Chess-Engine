
#ifndef CACHE_MANAGEMENT_H
#define CACHE_MANAGEMENT_H


#include "cpp_bitboard.h"
#include "search_engine.h"
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
#include <optional>
#include <sstream>

// Define zobrist table, cache and insertion order for efficient hashing
extern uint64_t zobristTable[12][64];
extern uint64_t zobristTurn;

extern uint64_t castling_hash[4];
extern uint64_t ep_hash[65];

extern std::unordered_map<uint64_t, int> evalCache;
extern std::deque<uint64_t> insertionOrder;

extern std::unordered_map<uint64_t, std::vector<Move>> moveGenCache;
extern std::deque<uint64_t> moveGenInsertionOrder;

extern uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

extern Move killerMoves[MAX_PLY][2];

extern int historyHeuristics[2][64][64];

/*
	Set of functions used to cache data
*/
inline void initializeZobrist() {
	
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

	for (int i = 0; i < 4; i++){
		castling_hash[i] = rng();
	}

	for (int i = 0; i < 65; i++){
		ep_hash[i] = rng();
	}

	zobristTurn = rng();
}

inline uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, bool whiteToMove) {
    
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
	
    if (!whiteToMove) {
        hash ^= zobristTurn;
    }
    return hash;
}

inline void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bool isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion) {
    
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
	if (pieceType == 5) { // White king
		if (fromSquare == 4) {
			if (toSquare == 6) { // White kingside castling
				hash ^= zobristTable[3][7]; // remove rook from h1
				hash ^= zobristTable[3][5]; // add rook to f1
			} else if (toSquare == 2) { // White queenside castling
				hash ^= zobristTable[3][0]; // remove rook from a1
				hash ^= zobristTable[3][3]; // add rook to d1
			}
		}
	} else if (pieceType == 11) { // Black king
		if (fromSquare == 60) {
			if (toSquare == 62) { // Black kingside castling
				hash ^= zobristTable[9][63]; // remove rook from h8
				hash ^= zobristTable[9][61]; // add rook to f8
			} else if (toSquare == 58) { // Black queenside castling
				hash ^= zobristTable[9][56]; // remove rook from a8
				hash ^= zobristTable[9][59]; // add rook to d8
			}
		}
	}    
	
    // If a piece was captured, XOR the captured piece out of its position
    if (isCapture) {
		
		// Acquire the captured piece
		int capturedPieceType = piece_type_at(toSquare) - 1;

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
	if (promotion != 1){
		
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

	// Switch the turn
	hash ^= zobristTurn;
}

inline void updateZobristHashForNullMove(uint64_t& hash){
	hash ^= zobristTurn;
}

inline int accessCache(uint64_t key) {
	
	/*
		Function to access the position cache
		
		Parameters:
		- key: The hash for the given position
		
		Returns:
		The stored evaluation for the position if it exists
	*/
	
    auto it = evalCache.find(key);
    if (it != evalCache.end()) {
		// Return the value if the key exists
        return it->second;  
    }
	
	// Return the default value if the key doesn't exist
    return 0;   
}

inline void addToCache(uint64_t key,int max_size, int value) {
	
	/*
		Function to add to the position cache
		
		Parameters:
		- key: The hash for the given position
		- value: The value to be associated with the given key
	*/
	
	// Add the key-value pair to the cache as well as the key to the move order
    evalCache[key] = value;
	insertionOrder.push_back(key);

    if (static_cast<int>(evalCache.size()) > max_size && !insertionOrder.empty()) {
        uint64_t oldestKey = insertionOrder.front();
        insertionOrder.pop_front();
        evalCache.erase(oldestKey);
    }
}

inline int printCacheStats() {
	
	/*
		Function to print the position cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Get the number of entries in the map
    int num_entries = evalCache.size();

    // Estimate the memory usage in bytes: each entry is a pair of (key, value)
    int size_in_bytes = num_entries * (sizeof(int64_t) + sizeof(int));

    // Print the results
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}


inline uint64_t hash_castling(uint64_t castling_rights) {
    uint64_t result = 0;
    for (int i = 0; i < 4; ++i) {
        if (castling_rights & (1ULL << rook_squares[i])) {
            result ^= castling_hash[i];
        }
    }
    return result;
}

inline uint64_t make_move_cache_key(uint64_t zobrist_base, uint64_t castling_rights, int ep_square) {
    uint64_t key = zobrist_base;
    key ^= hash_castling(castling_rights);

    if (ep_square != -1 && ep_square >= 0 && ep_square < 64) {
        key ^= ep_hash[ep_square];
    }


    return key;
}


inline std::vector<Move> accessMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square) {
	
	/*
		Function to access the position cache
		
		Parameters:
		- key: The hash for the given position
		
		Returns:
		The stored evaluation for the position if it exists
	*/

	uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
	
    auto it = moveGenCache.find(updatedKey);
    if (it != moveGenCache.end()) {
		// Return the value if the key exists
        return it->second;  
    }
	
	// Return the default value if the key doesn't exist
    std::vector<Move> dummy;
	return dummy;   
}

inline std::vector<Move>& accessMutableMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    return moveGenCache[updatedKey];  // If not present, creates empty vector by default
}


inline void addToMoveGenCache(uint64_t key, int max_size, std::vector<Move> reorderedMoves, uint64_t castling_rights, int ep_square){
	uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
	
	moveGenCache[updatedKey] = reorderedMoves;
	moveGenInsertionOrder.push_back(updatedKey);

    if (static_cast<int>(moveGenCache.size()) > max_size && !moveGenInsertionOrder.empty()) {
        uint64_t oldestKey = moveGenInsertionOrder.front();
        moveGenInsertionOrder.pop_front();
        moveGenCache.erase(oldestKey);
    }
}

inline void updateMoveCacheForBetaCutoff(uint64_t zobrist, uint64_t castling, uint64_t ep_square, Move move, std::vector<Move> moves, std::vector<BoardState>& state_history){
    std::vector<Move>& moveList = accessMutableMoveGenCache(zobrist, castling, ep_square);

    if (moveList.empty()){
        auto it = std::find(moves.begin(), moves.end(), move);
        if (it != moves.end() && it != moves.begin()) {
            std::iter_swap(it, moves.begin());
        }

        int num_plies = static_cast<int>(state_history.size());
        int max_cache_size;
        // Code segment to control cache size
        if(num_plies < 30){
            max_cache_size = 800000;         
        }else if(num_plies < 50){
            max_cache_size = 1600000;
        }else if(num_plies < 75){
            max_cache_size = 2400000; 
        }else{
            max_cache_size = 3000000; 
        }
        addToMoveGenCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, moves, castling, ep_square);
        return;
    }

    auto it = std::find(moveList.begin(), moveList.end(), move);
    if (it != moveList.end() && it != moveList.begin()) {
        std::iter_swap(it, moveList.begin());
    }    
}

inline void storeKillerMove(int ply, Move move) {
    if (!(killerMoves[ply][0] == move)) {
        killerMoves[ply][1] = killerMoves[ply][0];
        killerMoves[ply][0] = move;
    }
}

inline int killerBonus(int ply, Move move) {
    if (killerMoves[ply][0] == move)
		return 10000;

	if (killerMoves[ply][1] == move)
		return 9000;	
	return 0;
}

inline void decayHistoryHeuristics() {
    for (int side = 0; side < 2; ++side) {
        for (int from = 0; from < 64; ++from) {
            for (int to = 0; to < 64; ++to) {
                historyHeuristics[side][from][to] >>= DECAY_FACTOR;
            }
        }
    }
}

inline int printMoveGenCacheStats() {
    /*
        Function to print stats for the move generation cache.

        Parameters:
        

        Returns:
        - Number of entries in the cache
    */

    int num_entries = moveGenCache.size();
    //std::cout << "Move cache stats:" << std::endl;
    //std::cout << "Number of entries: " << num_entries << std::endl;
    
    size_t total_moves = 0;

    for (const auto& entry : moveGenCache) {
        total_moves += entry.second.size();
    }

    size_t size_of_keys = num_entries * sizeof(uint64_t);
    size_t size_of_vectors = num_entries * sizeof(std::vector<Move>);
    size_t size_of_moves = total_moves * sizeof(Move);

    size_t total_bytes = size_of_keys + size_of_vectors + size_of_moves;

    std::cout << "Move cache stats:\n" << std::endl;
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Total moves stored: " << total_moves << std::endl;
    std::cout << "Estimated size in bytes: " << total_bytes << std::endl;
    std::cout << "Estimated size in megabytes: " << (total_bytes >> 20) << std::endl;
    

    return num_entries;
}

#endif