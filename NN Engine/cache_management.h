
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

/* extern size_t CACHE_SIZE;  // example: 1M entries
extern uint64_t CACHE_MASK; */

struct EvalEntry {
    uint64_t key = 0;
    int value = 0;
    bool valid = false;
};

struct MoveEntry {
    uint64_t key = 0;
    std::vector<Move> moves;
    int last_cutoff_index = -1;   // index of the move that caused the last beta cutoff here (pre-promotion); -1 = unknown
    int reuse_count = 0;          // cache hits since the last full lazy re-sort / cutoff refresh (staleness gauge)
    bool valid = false;
};

extern std::vector<EvalEntry> evalCacheNew;
extern std::vector<MoveEntry> moveGenCache;

extern std::unordered_map<uint64_t, int> evalCache;
extern std::deque<uint64_t> insertionOrder;

/* extern std::unordered_map<uint64_t, int> quiesceEvalCache;
extern std::deque<uint64_t> quiesceinsertionOrder; */

/* extern std::unordered_map<uint64_t, std::vector<Move>> moveGenCache;
extern std::deque<uint64_t> moveGenInsertionOrder; */

enum class TTFlag : uint8_t {
    EXACT,        // Score is exact
    LOWERBOUND,   // Score is a lower bound (fail-high)
    UPPERBOUND    // Score is an upper bound (fail-low)
};

struct TTEntry {
    uint64_t key = 0;
    int score;             // Evaluated score
    int depth;             // Depth at which this score was obtained
    TTFlag flag;           // Type of score
    Move move;             // Node's best/cutoff move (singular verification); default {0,0,0} = no move
    int alpha = -9999999; // (dead: written, never read on the live path)
    int beta = 9999999;  // (dead)
    bool valid = false;

	TTEntry() : score(0), depth(0), flag(TTFlag::EXACT) {}

    TTEntry(int s, int d, TTFlag f, int a = -9999999, int b = 9999999)
        : score(s), depth(d), flag(f), alpha(a), beta(b) {}
};

/* struct TTEntry {
    uint64_t key = 0;
    int score;             // Evaluated score
    int depth;             // Depth at which this score was obtained
    TTFlag flag;           // Type of score
    int alpha; // NEW
    int beta;  // NEW
    bool valid = false;

	TTEntry() : score(0), depth(0), flag(TTFlag::EXACT) {}

    TTEntry(int s, int d, TTFlag f, int a = -9999999, int b = 9999999)
        : score(s), depth(d), flag(f), alpha(a), beta(b) {}
}; */

/* struct TTEntry {
    int score;
    int depth;
    TTFlag flag;
    int alpha;
    int beta;    

    std::array<Move, 8> pv;  // Short principal variation
    int pv_length;

	TTEntry() : score(0), depth(0), flag(TTFlag::EXACT) {}

	TTEntry(int s, int d, TTFlag f, int a, int b, const std::vector<Move>& pv_line)
        : score(s), depth(d), flag(f), alpha(a), beta(b), pv_length(std::min((int)pv_line.size(), 8)) {
        std::copy_n(pv_line.begin(), pv_length, pv.begin());
		}
}; */


struct QCacheEntry {
    int score;
    TTFlag flag;           // Type of score
    
	QCacheEntry() : score(0), flag(TTFlag::EXACT) {}

    QCacheEntry(int s, TTFlag f)
        : score(s), flag(f) {}
};

// Direct-mapped quiescence cache slot. Carries the bound flag so a cached
// q-search result is reused only when valid for the probing [alpha, beta]
// window (same gating the main TT uses via use_tt_entry). Replaces the prior
// raw-score EvalEntry, which handed back fail-high/fail-low bounds as if exact.
struct QTTEntry {
    uint64_t key = 0;
    int score = 0;
    TTFlag flag = TTFlag::EXACT;
    bool valid = false;
};

extern std::vector<QTTEntry> quiesceEvalCache;

extern std::vector<TTEntry> searchEvalCache;
extern bool g_no_tt_store;   // when true, addToSearchEvalCache is a no-op (ProbCut store-off diagnostic)
// Hash-move table (TT-move ordering): one remembered beta-cutoff move per TT slot, same size/index as
// searchEvalCache. Read/written only under Config::ENABLE_TT_MOVE. Default {0,0,0} = "no move".
extern std::vector<Move> g_ttMoveTable;

//extern std::vector<TTEntry> searchEvalCache;

extern uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

extern Move killerMoves[MAX_PLY][2];
extern Move counterMoves[64][64];

extern int counterMoveHeuristics[2][4096][4096];
extern int historyHeuristics[2][64][64];
extern int moveFrequency[2][64][64];

extern int contHist2[2][4096][4096];
extern int captureHistory[2][64][64];

// Per-ply move stack (single-threaded search): g_searchStack[d] = the move played to descend from
// depth d to d+1. A node at depth `ply` reads g_searchStack[ply-2] as the move 2 plies back (the
// 2-ply continuation key) without threading a previousMove2 param through the search.
extern Move g_searchStack[MAX_PLY];

// Per-ply singular-exclusion move: g_excluded_move[d] = the TT-move currently being verified at depth d
// (the singular exclusion search re-enters the node's search with this move skipped). {0,0,0}=none = not
// excluding. Default all-none, so the move-loop skip and the pruning guards are inert until singular fires.
extern Move g_excluded_move[MAX_PLY];
// Singular diagnostics (default off; printed under ENABLE_SINGULAR): eligible TT-move nodes, gate passes,
// and actual fires (extensions). The null-result confound detector — a low fire rate means TT starvation,
// not "singular doesn't help".
extern long g_sing_eligible, g_sing_gatepass, g_sing_fire;

// Per-ply static-eval stack for the improving heuristic: g_evalStack[d] = node-entry static eval at
// depth d (NO_STATIC_EVAL when in-check or outside the improving window). A node reads [d] vs [d-2].
extern int g_evalStack[MAX_PLY];

// Per-ply capture-chain density for the capture-chain LMR guard: g_captureChain[d] is a leaky counter
// along the search path -- +1 for a capture into node d, -1 (floored at 0) for a quiet move -- so it
// rises inside a capture-heavy sequence (back-to-back chains AND captures interspersed with the odd
// quiet) and decays once the tactics stop. A node reads [d] to decide whether it sits deep enough
// inside a forcing sequence to protect a quiet move from LMR.
extern int g_captureChain[MAX_PLY];

// Per-ply move-list buffers (see definitions in cpp_bitboard.cpp). Indexed by the node's ply so a
// node can hold a reference to its list across its child-search recursion without a per-node alloc.
extern std::vector<Move> g_moveBuf[MOVE_POOL_PLIES];
extern std::vector<Move> g_noisyBuf[MOVE_POOL_PLIES];
extern std::vector<Move> g_moveBufFallback;
extern std::vector<Move> g_noisyBufFallback;

// Light-eval flag (see cpp_bitboard.cpp): skip the heavy dynamic eval terms for a fast stand-pat/futility eval.
extern bool g_eval_light;

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

/*
	Pawn-only Zobrist key: a hash of just the pawn placement (both colours), reusing the same
	zobristTable randoms as the full hash (white pawn = index 0, black pawn = index 6) so it is a
	strict subset of it. Side-to-move is deliberately excluded — pawn structure is turn-independent,
	so positions differing only in whose move it is share a pawn key (the point, for structure caching
	and correction history). Cheap enough (<=16 XORs) to recompute per eval rather than maintain
	incrementally.
*/
inline uint64_t generatePawnKey(uint64_t pawnsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask) {
	uint64_t key = 0;
	uint64_t wp = pawnsMask & occupied_whiteMask;
	while (wp) {
		key ^= zobristTable[0][__builtin_ctzll(wp)];
		wp &= wp - 1;
	}
	uint64_t bp = pawnsMask & occupied_blackMask;
	while (bp) {
		key ^= zobristTable[6][__builtin_ctzll(bp)];
		bp &= bp - 1;
	}
	return key;
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

inline bool accessCacheNew(uint64_t key, int& out) {
    size_t idx = key & CACHE_MASK;
    /* assert(idx < CACHE_SIZE);
    assert(idx >= 0); */
    EvalEntry &entry = evalCacheNew[idx];

    if (entry.valid && entry.key == key) {
        out = entry.value;  // Cache hit -- value may legitimately be 0 (draws / dead-equal)
        return true;
    }
    return false;  // Cache miss
}

inline void addToCacheNew(uint64_t key, int value) {
    size_t idx = key & CACHE_MASK;
    /* assert(idx < CACHE_SIZE);
    assert(idx >= 0); */
    EvalEntry& entry = evalCacheNew[idx];
    entry.key = key;
    entry.value = value;
    entry.valid = true;
}


inline void addToCache(uint64_t key,int max_size, int value) {
	
	/*
		Function to add to the position cache
		
		Parameters:
		- key: The hash for the given position
		- value: The value to be associated with the given key
	*/
	
	// Add the key-value pair to the cache as well as the key to the move order.
	// Only record insertion order for genuinely new keys — pushing on every update
	// lets a hot key accumulate stale deque copies and be evicted from the map
	// while a fresher copy is still live
	bool isNewEntry = (evalCache.find(key) == evalCache.end());
    evalCache[key] = value;
	if (isNewEntry)
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
    std::cout << "EVAL CACHE: "<< std::endl;
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}

inline int printQCacheStats() {
	
	/*
		Function to print the position cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Get the number of entries in the map
    int num_entries = quiesceEvalCache.size();

    // Estimate the memory usage in bytes: each entry is a pair of (key, value)
    int size_in_bytes = num_entries * (sizeof(int64_t) + sizeof(int));

    // Print the 
    std::cout << "Q CACHE: "<< std::endl;
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
	size_t idx = updatedKey & CACHE_MASK;
    /* assert(idx < CACHE_SIZE);
    assert(idx >= 0); */

    MoveEntry &entry = moveGenCache[idx];

    if (entry.valid && entry.key == updatedKey) {
        return entry.moves;  // Cache hit
    }
    std::vector<Move> dummy;
	return dummy;  // Cache miss (or default value)

    /* auto it = moveGenCache.find(updatedKey);
*/}

// True iff the move-gen cache holds a non-empty move list for `key`. Lets callers that only need
// existence (is_checkmate / is_stalemate short-circuit on a cached non-empty list) skip copying the
// cached vector out of the slot, avoiding the by-value alloc+memcpy+free on the hot path.
inline bool moveGenCacheHasMoves(uint64_t key, uint64_t castling_rights, int ep_square) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    size_t idx = updatedKey & CACHE_MASK;
    MoveEntry &entry = moveGenCache[idx];
    return entry.valid && entry.key == updatedKey && !entry.moves.empty();
}

// Snapshot the cached move list for `key` into the caller-owned buffer `out` (reusing out's existing
// capacity), or clear `out` on a miss. The copy is synchronous -- no reference into the cache slot
// survives the call, so later cache churn cannot touch `out`. Lets a caller reuse a per-ply buffer
// instead of heap-allocating a fresh vector per node, while keeping the defensive snapshot semantics.
inline void fillMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square, std::vector<Move>& out) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    size_t idx = updatedKey & CACHE_MASK;
    MoveEntry &entry = moveGenCache[idx];
    if (entry.valid && entry.key == updatedKey) {
        out = entry.moves;   // snapshot copy (reuses out's capacity)
    } else {
        out.clear();
    }
}

/* dead tail of the old by-value accessMoveGenCache:
    auto it = moveGenCache.find(updatedKey);
    if (it != moveGenCache.end()) {
		// Return the value if the key exists
        return it->second;  
    }
	
	// Return the default value if the key doesn't exist
    std::vector<Move> dummy;
	return dummy;    */


/* inline std::vector<Move>& accessMutableMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    //return moveGenCache[updatedKey];  // If not present, creates empty vector by default
    size_t idx = updatedKey & CACHE_MASK;
    
    MoveEntry &entry = moveGenCache[idx];
    // If the entry is valid and belongs to the current position, return it
    if (entry.valid && entry.key == updatedKey) {
        return entry.moves;
    }

    // Otherwise, overwrite the entry
    entry.key = updatedKey;
    entry.valid = true;
    entry.moves.clear();  // Clear previous moves (for a different position!)
    return entry.moves;
} */

inline MoveEntry& accessMutableMoveGenCache(uint64_t updatedKey, uint64_t castling_rights, int ep_square) {
    
    //return moveGenCache[updatedKey];  // If not present, creates empty vector by default
    size_t idx = updatedKey & CACHE_MASK;
    /* assert(idx < CACHE_SIZE);
    assert(idx >= 0); */
    MoveEntry& entry = moveGenCache[idx];
    // If the entry is valid and belongs to the current position, return it
    if (entry.valid && entry.key == updatedKey) {
        return entry;
    }

    // Otherwise, overwrite the entry
    entry.key = updatedKey;
    entry.valid = true;
    entry.moves.clear();  // Clear previous moves (for a different position!)
    return entry;
}

inline void addToMoveGenCache(uint64_t key, std::vector<Move> reorderedMoves, uint64_t castling_rights, int ep_square) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    MoveEntry& moveEntry = accessMutableMoveGenCache(updatedKey, castling_rights, ep_square);
    moveEntry.moves = std::move(reorderedMoves);  // Move for speed    
}



/* inline bool probeQCache(uint64_t key, uint64_t castling_rights, int ep_square, int alpha, int beta, int& outScore) {
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
	auto it = quiesceEvalCache.find(updatedKey);
    if (it == quiesceEvalCache.end()) return false;

    const QCacheEntry& entry = it->second;

    switch (entry.flag) {
        case TTFlag::EXACT:
            outScore = entry.score;
            return true;

        case TTFlag::LOWERBOUND:
            if (entry.score >= beta) {
                outScore = entry.score;
                return true;
            }
            break;

        case TTFlag::UPPERBOUND:
            if (entry.score <= alpha) {
                outScore = entry.score;
                return true;
            }
            break;
    }
    return false; // Not safe to use this entry for this window
} */


inline bool probeQCache(uint64_t key, uint64_t castling_rights, int ep_square, int alpha, int beta, int& outScore) {

	/*
		Probe the quiescence cache, returning a stored score only when its bound
		flag is valid for the current window: EXACT is always usable, a
		LOWERBOUND on a beta cutoff (score >= beta), an UPPERBOUND on an alpha
		cutoff (score <= alpha). This prevents reusing a fail-high/fail-low value
		as if it were exact in an incompatible window.
	*/

    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
	size_t idx = updatedKey & CACHE_MASK;
    QTTEntry &entry = quiesceEvalCache[idx];

	if (!entry.valid || entry.key != updatedKey)
		return false;

	switch (entry.flag) {
		case TTFlag::EXACT:
			outScore = entry.score;
			return true;
		case TTFlag::LOWERBOUND:
			if (entry.score >= beta) { outScore = entry.score; return true; }
			break;
		case TTFlag::UPPERBOUND:
			if (entry.score <= alpha) { outScore = entry.score; return true; }
			break;
	}
	return false;
}

inline void addToQCache(uint64_t key, int score, TTFlag flag, uint64_t castling_rights, int ep_square) {

	// Skip mate scores — stored without ply adjustment, as the main TT does
    if (score >= 9000000 || score <= -9000000)
		return;
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
	size_t idx = updatedKey & CACHE_MASK;
    QTTEntry& entry = quiesceEvalCache[idx];
    entry.key = updatedKey;
    entry.score = score;
    entry.flag = flag;
    entry.valid = true;
}


/* inline TTEntry* accessSearchEvalCache(uint64_t key, uint64_t castling_rights, int ep_square) {
   
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    //return moveGenCache[updatedKey];  // If not present, creates empty vector by default
    size_t idx = updatedKey & TT_CACHE_MASK;
    
    TTEntry &entry = searchEvalCache[idx];
    // If the entry is valid and belongs to the current position, return it
    if (entry.valid && entry.key == updatedKey) {
        return &entry;
    }

    // Return an empty optional if not found
    return nullptr;
    
}


inline void addToSearchEvalCache(uint64_t key, int num_plies, int score, int depth_used, TTFlag flag, int alpha_orig, int beta_orig, uint64_t castling_rights, int ep_square) {	

	if (score >= 9000000 || score <= -9000000 || score == 0)
		return;
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);
    
    size_t idx = updatedKey & TT_CACHE_MASK;
    
    TTEntry& entry = searchEvalCache[idx];
    if((updatedKey != entry.key) || (entry.depth <= depth_used)){
        entry.key = updatedKey;
        entry.score = score;
        entry.depth = depth_used;
        entry.flag = flag;
        entry.alpha = alpha_orig;
        entry.beta = beta_orig;
        entry.valid = true;
    }
    
} */

inline TTEntry* accessSearchEvalCache(uint64_t key, uint64_t castling_rights, int ep_square) {
    PROF_BLOCK(PROF_TT_PROBE);
    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);

    if (Config::TT_WAYS <= 1) {
        size_t idx = updatedKey & TT_CACHE_MASK;
        TTEntry& entry = searchEvalCache[idx];
        if (entry.valid && entry.key == updatedKey)
            return &entry;
        return nullptr;
    }

    // N-way set-associative: the index selects a bucket of TT_WAYS contiguous entries.
    size_t base = (updatedKey & (TT_CACHE_SIZE / Config::TT_WAYS - 1)) * Config::TT_WAYS;
    for (int i = 0; i < Config::TT_WAYS; ++i) {
        TTEntry& entry = searchEvalCache[base + i];
        if (entry.valid && entry.key == updatedKey)
            return &entry;
    }
    return nullptr;
}


inline void addToSearchEvalCache(uint64_t key, int num_plies, int score, int depth_used, TTFlag flag, int alpha_orig, int beta_orig, uint64_t castling_rights, int ep_square, Move move = Move()) {

	if (g_no_tt_store)
		return;
	if (score >= 9000000 || score <= -9000000 || score == 0)
		return;

    uint64_t updatedKey = make_move_cache_key(key, castling_rights, ep_square);

    // Best-move field (SF rule): store a REAL move, or reset it when the slot now holds a DIFFERENT position;
    // done independently of the depth-preferred score gate below so the deepest same-position entries still
    // receive the node's move (else singular starves on exactly the best entries). A no-move store on the
    // SAME key preserves the existing move (a child-keyed score store must not erase a node-local move).
    // move stays {0,0,0}=none for every caller until the node-local accept-point stores populate it, so this
    // is inert (byte-identical) until singular reads it.
    bool real_move = (move.from_square != move.to_square);

    if (Config::TT_WAYS <= 1) {
        size_t idx = updatedKey & TT_CACHE_MASK;
        TTEntry& entry = searchEvalCache[idx];

        if (real_move || (updatedKey != entry.key))
            entry.move = move;

        // Take the slot on a new/colliding key, or replace a same-position entry only
        // when the new search is equal-or-deeper (depth-preferred). num_plies is unused now.
        if ((updatedKey != entry.key) || (entry.depth <= depth_used)) {
            entry.key = updatedKey;
            entry.score = score;
            entry.depth = depth_used;
            entry.flag = flag;
            entry.alpha = alpha_orig;
            entry.beta = beta_orig;
            entry.valid = true;
        }
        return;
    }

    // N-way set-associative: pick the target slot (empty, same-position, or shallowest victim), then apply
    // the same move + depth-preferred score rules. Non-same-position victims are always taken (cross-position
    // eviction); same-position score is kept when the existing entry is deeper.
    size_t base = (updatedKey & (TT_CACHE_SIZE / Config::TT_WAYS - 1)) * Config::TT_WAYS;
    TTEntry* victim = nullptr;
    for (int i = 0; i < Config::TT_WAYS; ++i) {
        TTEntry& e = searchEvalCache[base + i];
        if (!e.valid) { victim = &e; break; }
        if (e.key == updatedKey) { victim = &e; break; }
        if (victim == nullptr || e.depth < victim->depth)
            victim = &e;
    }
    TTEntry& entry = *victim;

    if (real_move || (updatedKey != entry.key))
        entry.move = move;

    if ((updatedKey != entry.key) || (entry.depth <= depth_used)) {
        entry.key = updatedKey;
        entry.score = score;
        entry.depth = depth_used;
        entry.flag = flag;
        entry.alpha = alpha_orig;
        entry.beta = beta_orig;
        entry.valid = true;
    }
}


inline int printSearchEvalCacheStats() {
	
	/*
		Function to print the position cache size as well as return it
		
		Returns:
		The number of entries in the cache
	*/
	
    // Direct-mapped array: number of slots (capacity), like the Q / move-gen caches
    int num_entries = searchEvalCache.size();

    // Estimate the memory usage in bytes
    size_t size_in_bytes = (size_t)num_entries * sizeof(TTEntry);

    // Print the results
    std::cout << "TT CACHE: "<< std::endl;
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Estimated size in bytes: " << size_in_bytes << std::endl;
	std::cout << "Estimated size in Megabytes: " << (size_in_bytes >> 20) << std::endl;
	
	return num_entries;
}


inline void updateMoveCacheForBetaCutoff(uint64_t zobrist, uint64_t castling, uint64_t ep_square, Move move, std::vector<Move> moves, std::vector<BoardState>& state_history){
    uint64_t updatedKey = make_move_cache_key(zobrist, castling, ep_square);
    // Read the slot directly: accessMutableMoveGenCache would stamp the key first,
    // making the miss below undetectable and silently discarding the move list
    size_t idx = updatedKey & CACHE_MASK;
    MoveEntry& moveEntry = moveGenCache[idx];

    if (!moveEntry.valid || moveEntry.key != updatedKey){
        /* auto it = std::find(moves.begin(), moves.end(), move);
        if (it != moves.end() && it != moves.begin()) {
            std::iter_swap(it, moves.begin());
        } */
        moveEntry.key = updatedKey;
        moveEntry.valid = true;
        // Record where the cutoff move sat in the freshly generated list (a cutoff deep in the
        // list flags a poorly-ordered node for the lazy re-sort). Gated: off = no extra find.
        if (Config::ENABLE_LAZY_RESORT) {
            auto it = std::find(moves.begin(), moves.end(), move);
            moveEntry.last_cutoff_index = (it != moves.end()) ? (int)std::distance(moves.begin(), it) : -1;
            moveEntry.reuse_count = 0;
        }
        promoteMoveToFront(moves, move);

        /* int num_plies = static_cast<int>(state_history.size());
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
        } */
        //addToMoveGenCache(zobrist, /* max_cache_size * Config::ACTIVE->cache_size_multiplier, */ moves, castling, ep_square);
        moveEntry.moves = std::move(moves);
        return;
    }

    /* auto it = std::find(moveList.begin(), moveList.end(), move);
    if (it != moveList.end() && it != moveList.begin()) {
        std::iter_swap(it, moveList.begin());
    } */

    if (Config::ENABLE_LAZY_RESORT) {
        auto it = std::find(moveEntry.moves.begin(), moveEntry.moves.end(), move);
        moveEntry.last_cutoff_index = (it != moveEntry.moves.end()) ? (int)std::distance(moveEntry.moves.begin(), it) : -1;
        moveEntry.reuse_count = 0;
    }

    promoteMoveToFront(moveEntry.moves, move);
}

inline void storeKillerMove(int ply, Move move) {
    if (!(killerMoves[ply][0] == move)) {
        killerMoves[ply][1] = killerMoves[ply][0];
        killerMoves[ply][0] = move;
    }
}

inline int killerBonus(int ply, Move move) {
    if (ply > 63)
        return 0;

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

inline void decayCounterMoveHeuristics() {
    for (int side = 0; side < 2; ++side) {
        for (int from = 0; from < 4096; ++from) {
            for (int to = 0; to < 4096; ++to) {
                counterMoveHeuristics[side][from][to] >>= DECAY_FACTOR;
            }
        }
    }
}

inline void decayContHist2() {
    for (int side = 0; side < 2; ++side) {
        for (int from = 0; from < 4096; ++from) {
            for (int to = 0; to < 4096; ++to) {
                contHist2[side][from][to] >>= DECAY_FACTOR;
            }
        }
    }
}

inline void decayCaptureHistory() {
    for (int side = 0; side < 2; ++side) {
        for (int from = 0; from < 64; ++from) {
            for (int to = 0; to < 64; ++to) {
                captureHistory[side][from][to] >>= DECAY_FACTOR;
            }
        }
    }
}

// Saturating "gravity" update for the cutoff-history tables: h moves toward delta but can never exceed
// +-MAX_HISTORY (the h*|delta|/MAX term cancels the gain near the bound). Used with +bonus on the move
// that caused the beta cutoff and -bonus (malus) on the moves tried-and-failed before it.
inline void hist_update(int &h, int delta) {
    if (delta > Config::MAX_HISTORY) delta = Config::MAX_HISTORY;
    else if (delta < -Config::MAX_HISTORY) delta = -Config::MAX_HISTORY;
    int ad = delta < 0 ? -delta : delta;
    h += delta - h * ad / Config::MAX_HISTORY;
}

inline void decayMoveFrequency() {
    for (int side = 0; side < 2; ++side) {
        for (int from = 0; from < 64; ++from) {
            for (int to = 0; to < 64; ++to) {
                moveFrequency[side][from][to] >>= 2;
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
        total_moves += entry.moves.size();
    }

    size_t size_of_keys = num_entries * sizeof(uint64_t);
    size_t size_of_vectors = num_entries * sizeof(std::vector<Move>);
    size_t size_of_moves = total_moves * sizeof(Move);

    size_t total_bytes = size_of_keys + size_of_vectors + size_of_moves;

    std::cout << "MOVE GEN:" << std::endl;
    std::cout << "Number of entries: " << num_entries << std::endl;
    std::cout << "Total moves stored: " << total_moves << std::endl;
    std::cout << "Estimated size in bytes: " << total_bytes << std::endl;
    std::cout << "Estimated size in megabytes: " << (total_bytes >> 20) << std::endl;
    

    return num_entries;
}

#endif