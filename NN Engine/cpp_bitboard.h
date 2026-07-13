
#ifndef CPP_BITBOARD_H
#define CPP_BITBOARD_H
#pragma once

#include "search_engine.h"

#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <cstring>
#include <optional>
#include <sstream>
#include <immintrin.h>

constexpr int NUM_SQUARES = 64;
constexpr int MAX_PLY = 64;
// Per-ply move-buffer pool depth: covers the main search (< MAX_PLY) plus qsearch (MAX_QDEPTH) with
// margin. A node's ply index into the pool is cur_depth (main) or cur_depth + qDepth (qsearch).
constexpr int MOVE_POOL_PLIES = 128;

constexpr size_t CACHE_SIZE = 1 << 23;  // example: 1M entries
constexpr uint64_t CACHE_MASK = CACHE_SIZE - 1;

constexpr size_t TT_CACHE_SIZE = 1 << 24;  // example: 1M entries
constexpr uint64_t TT_CACHE_MASK = TT_CACHE_SIZE - 1;

/*
	PEXT (Parallel Bits Extract, BMI2 _pext_u64) sliding-attack row.

	A per-square attack table for one slider direction set. The masked
	occupancy is compressed with _pext_u64 into a dense index, so a lookup is a
	single array read instead of an unordered_map hash + pointer chase. Because
	_pext_u64(mask & occupied, mask) == _pext_u64(occupied, mask) and
	_pext_u64(0, mask) == 0, every existing call site works unchanged whether it
	passes (BB_X_MASKS[sq] & occupied) or a literal 0 (empty-board rays).

	Note: _pext_u64 is fast on Intel and on AMD Zen3+, but microcoded/slow on
	AMD Zen1/Zen2 — fine on this -march=native build.
*/
struct SlidingRow {
	uint64_t mask = 0;
	std::vector<uint64_t> data;  // size 1 << popcount(mask), indexed by pext(occ, mask)
	uint64_t operator[](uint64_t occ) const { return data[_pext_u64(occ, mask)]; }
};

// Define masks for move generation
extern std::array<uint64_t, NUM_SQUARES> BB_KNIGHT_ATTACKS;
extern std::array<uint64_t, NUM_SQUARES> BB_KING_ATTACKS;
// Forward 3-file passed-pawn spans (see cpp_bitboard.cpp); read by the search's passed-pawn exemption.
extern std::array<uint64_t, NUM_SQUARES> passed_span_white;
extern std::array<uint64_t, NUM_SQUARES> passed_span_black;
extern std::array<std::array<uint64_t, NUM_SQUARES>, 2> BB_PAWN_ATTACKS;
extern std::vector<uint64_t> BB_DIAG_MASKS;
extern std::vector<SlidingRow> BB_DIAG_ATTACKS;
extern std::vector<uint64_t> BB_FILE_MASKS;
extern std::vector<SlidingRow> BB_FILE_ATTACKS;
extern std::vector<uint64_t> BB_RANK_MASKS;
extern std::vector<SlidingRow> BB_RANK_ATTACKS;
extern std::vector<std::vector<uint64_t>> BB_RAYS;

// Define global masks for piece placement
extern uint64_t pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, occupied;

extern std::array<int, 64> pressure_white;
extern std::array<int, 64> pressure_black;

extern std::array<int, 64> support_white;
extern std::array<int, 64> support_black;

extern std::array<int, 64> square_values;
extern std::array<uint64_t, 64> attack_bitmasks;

// Phase-0 light-eval gap probe accumulators (Config::LIGHT_GAP_PROBE). |gap| histogram + per-term abs sums.
extern long long g_lge_n;
extern long long g_lge_hist[7];
extern long long g_lge_abs_capture, g_lge_abs_passed, g_lge_abs_latent, g_lge_abs_adv;

// Phase-A SEE-frequency counter (Config::SEE_COUNT). Cumulative across a run; gated so the default build
// pays nothing in the hot see() path. Read alongside nodes/qnodes to get see-calls-per-node.
extern long long see_calls;

// SEE cache (Config::ENABLE_SEE_CACHE). Per-position cache keyed by [side][square], validated by a
// generation counter bumped in make_move/unmake_move (O(1) invalidation, no zobrist threading). see() is a
// pure function of (square, side, position), so a hit returns the identical value -> node counts stay
// byte-identical with the cache on; only speed changes. g_see_hits/g_see_miss are gated by SEE_COUNT.
extern uint64_t g_see_gen;
extern uint64_t see_cache_gen[2][64];
extern int see_cache_val[2][64];
extern long long g_see_hits, g_see_miss;

constexpr uint8_t PAWN = 1;
constexpr uint8_t KNIGHT = 2;
constexpr uint8_t BISHOP = 3;
constexpr uint8_t ROOK = 4;
constexpr uint8_t QUEEN = 5;
constexpr uint8_t KING = 6;

constexpr int MAX_PHASE = 24;

// Attack-unit king-safety: the danger lookup table is built up to this many units (clamp ceiling).
// king_safety_score indexes ks_safety_table[min(units, KS_MAX_UNITS)]; Config::KS_CAP (<= this) is the
// tunable runtime clamp. The phase-taper table ks_phase_taper[] is indexed by phase_score 0..128.
constexpr int KS_MAX_UNITS = 128;
extern std::array<int, KS_MAX_UNITS + 1> ks_safety_table;  // attack-units -> danger (precomputed at init)
extern std::array<int, 129>              ks_phase_taper;   // phase_score 0..128 -> /256 fade

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

// Pawn-structure base tables (literals). The eval reads the like-named working arrays (defined in
// cpp_bitboard.cpp), rebuilt = base * SCALE_* / 100 once at init by rebuild_scaled_pawn_tables() so the
// hot path is a plain array read with no per-read division. Default knobs (100) reproduce the base
// values exactly (byte-identical).
constexpr std::array<int, 8> default_midgame_pawn_rank_bonus_base = {0, 15, 30, 45, 60, 75, 90, 105};
constexpr std::array<int, 8> passed_midgame_pawn_rank_bonus_base = {0, 65, 160, 285, 625, 840, 1085, 1360};

constexpr std::array<int, 8> endgame_pawn_rank_bonus_base = {0, 90, 210, 360, 750, 990, 1260, 1560};

constexpr std::array<uint8_t, 11> pawn_wall_file_bonus_base = {
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

constexpr std::array<uint8_t, 8> pawn_chain_file_bonus_base = {
    10,   // A
    15,   // B
    100,   // C
    150,  // D
    150,  // E
    100,   // F
    15,   // G
    10    // H
};

constexpr int BAD_THRESHOLD = 100;
constexpr int NEUTRAL_THRESHOLD = 125;
constexpr int MAX_THRESHOLD = 225;
constexpr int MAX_BONUS = 275;  // in millipawns

constexpr std::array<uint64_t, 8> white_king_zones = {
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // A file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // B file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6), // C file
    
    (BB_FILE_B | BB_FILE_C | BB_FILE_D | BB_FILE_E) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // D file
    (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // E file

    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // F file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6),  // G file
    (BB_FILE_E | BB_FILE_F | BB_FILE_G | BB_FILE_H) & ~(BB_RANK_8 | BB_RANK_7 | BB_RANK_6)  // H file
};

constexpr std::array<uint64_t, 8> black_king_zones = {
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // A file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // B file
    (BB_FILE_A | BB_FILE_B | BB_FILE_C | BB_FILE_D) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3), // C file
    
    (BB_FILE_B | BB_FILE_C | BB_FILE_D | BB_FILE_E) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // D file
    (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F) & ~(BB_RANK_1 | BB_RANK_2 | BB_RANK_3),  // E file

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
alignas(64) constexpr std::array<uint64_t, 64> BB_SQUARES = generate_square_masks();

constexpr uint64_t central_squares = BB_SQUARES[27] | BB_SQUARES[28] | BB_SQUARES[35] | BB_SQUARES[36];

constexpr uint64_t extended_central_squares =
    BB_SQUARES[18] | BB_SQUARES[19] | BB_SQUARES[20] | BB_SQUARES[21] |
    BB_SQUARES[26] | BB_SQUARES[29] |
    BB_SQUARES[34] | BB_SQUARES[37] |
    BB_SQUARES[42] | BB_SQUARES[43] | BB_SQUARES[44] | BB_SQUARES[45];


constexpr uint64_t light_dark_square_mask(bool light) {
    uint64_t mask = 0ULL;
    for (int rank = 0; rank < 8; ++rank) {
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            bool is_light = (rank + file) % 2 == 1;
            if (is_light == light) {
                mask |= (1ULL << sq);
            }
        }
    }
    return mask;
}

// Base color masks
constexpr uint64_t LIGHT_SQUARES = light_dark_square_mask(true);
constexpr uint64_t DARK_SQUARES  = light_dark_square_mask(false);

// White bishop candidate zones: ranks 1–5 (0–4)
constexpr uint64_t WHITE_BASE_MASK = 
    BB_RANKS[0] | BB_RANKS[1] | BB_RANKS[2] | BB_RANKS[3] | BB_RANKS[4];

// Black bishop candidate zones: ranks 8–4 (7–3)
constexpr uint64_t BLACK_BASE_MASK = 
    BB_RANKS[7] | BB_RANKS[6] | BB_RANKS[5] | BB_RANKS[4] | BB_RANKS[3];

// Final masks
constexpr uint64_t WHITE_LIGHT_BISHOP_ZONE = WHITE_BASE_MASK & LIGHT_SQUARES;
constexpr uint64_t WHITE_DARK_BISHOP_ZONE  = WHITE_BASE_MASK & DARK_SQUARES;

constexpr uint64_t BLACK_LIGHT_BISHOP_ZONE = BLACK_BASE_MASK & LIGHT_SQUARES;
constexpr uint64_t BLACK_DARK_BISHOP_ZONE  = BLACK_BASE_MASK & DARK_SQUARES;


	constexpr int rook_squares[4] = {0, 7, 56, 63};

struct CaptureInfo {
    uint8_t from;
	uint8_t to;
    int value_gained;

	CaptureInfo() = default;
	CaptureInfo(uint8_t from_square, uint8_t to_square, int value) : from(from_square), to(to_square), value_gained(value) {}
};

/*
	Fixed-capacity LIFO stack of CaptureInfo, used per eval in place of a heap
	std::vector. Capacity 32 safely exceeds the per-side non-king piece count
	(<=15). Exposes the same member names the capture helpers call so their
	bodies are unchanged; begin()/end() return raw pointers for std::sort.
*/
struct CaptureStack {
	CaptureInfo data[32];
	int n = 0;
	void push_back(const CaptureInfo& c){ data[n++] = c; }
	bool empty() const { return n == 0; }
	int  size()  const { return n; }
	CaptureInfo& operator[](int i){ return data[i]; }
	const CaptureInfo& operator[](int i) const { return data[i]; }
	CaptureInfo& back(){ return data[n - 1]; }
	void pop_back(){ --n; }
	CaptureInfo* begin(){ return data; }
	CaptureInfo* end(){ return data + n; }
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
void rebuild_scaled_placement();
void rebuild_scaled_pawn_tables();
void rebuild_ks_tables();
void attack_table(const std::vector<int8_t>& deltas, std::vector<uint64_t> &mask_table, std::vector<SlidingRow> &attack_table);
uint64_t sliding_attacks(uint8_t square, uint64_t occupied, const std::vector<int8_t>& deltas);
void carry_rippler(uint64_t mask, std::vector<uint64_t> &subsets);
void rays(std::vector<std::vector<uint64_t>> &rays);
uint64_t edges(uint8_t square);

/*
	Set of functions directly used to evaluate the position
*/
int placement_and_piece_midgame(uint8_t square);
int placement_and_piece_endgame(uint8_t square);
inline uint64_t latent_support_mask_left(int square, bool is_white);
inline uint64_t latent_support_mask_right(int square, bool is_white); 
inline void update_pressure_and_support_tables(uint8_t current_square, uint8_t attacking_piece_type, uint8_t decrement, bool attacking_piece_colour, bool current_piece_colour);
inline void handle_batteries_for_pressure_and_support_tables(uint8_t attacking_piece_square, uint8_t attacking_piece_type, uint64_t prev_attack_mask, bool attacking_piece_colour);
inline void loop_and_update(uint64_t bb, uint8_t attacking_piece_type, bool attacking_piece_colour, int decrement);
inline void adjust_pressure_and_support_tables_for_pins(uint64_t bb);
inline int advanced_endgame_eval(int total, bool turn);
inline void update_global_central_scores(int base_increment, uint64_t square_mask);
int placement_and_piece_eval(int moveNum, bool turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied);
int cheap_eval(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black);

/*
	Diagnostic-only static-eval term attribution. When g_capture_eval_breakdown is set,
	placement_and_piece_eval records each term's contribution to `total` (captured as read-only deltas of
	`total` at term boundaries, so search behaviour is byte-identical) into g_eval_breakdown. Values are in the
	engine's absolute (Black-positive) milli-pawn units, matching `total`. `pieces` lumps the six per-piece
	placement loops; `material` is informational (blackPieceVal - whitePieceVal), NOT part of the additive sum.
	latent_threat/central/imbalance_* are midgame-only (0 in the endgame path). When advanced_endgame_fired,
	the additive terms reflect the pre-replace state and total == advanced_endgame_total + pair_bonus +
	piece_value_boost.
*/
struct EvalBreakdown {
	int total;
	int pieces;
	int material;
	int capture_gains;
	int passed_pawn_support;
	int latent_threat;
	int threats;   // SF11-style piece-on-piece static threats (0 unless ENABLE_THREATS; additive, beside latent_threat)
	int king_safety;   // attack-unit king-safety term (0 unless KING_SAFETY_MAG > 0; additive, beside latent_threat)
	int central;
	int imbalance_white;
	int imbalance_black;
	int pair_bonus;
	int piece_value_boost;
	int pawn_majority;   // wing pawn-majority bonus (0 unless PAWN_MAJORITY_MAG_* > 0)
	int pawn_struct;     // isolated + backward pawn penalties (0 unless ISOLATED/BACKWARD_PAWN_PEN > 0)
	int outpost;         // knight/bishop outpost bonus (0 unless OUTPOST_KNIGHT/BISHOP > 0)
	int space;           // pawn+knight enemy-half control, rank-weighted (0 unless SPACE_MAG > 0)
	int mobility;        // per-piece mobility (0 unless ENABLE_PIECE_MOBILITY)
	int rook_cond;       // tension-conditioned rook-file rescale (0 unless ENABLE_ROOK_TENSION_COND)
	int phase_score;
	int advanced_endgame_total;
	bool is_endgame;
	bool advanced_endgame_fired;
	// Per-piece-type contribution to `pieces` (midgame path) for color-mirror localization.
	int pt_pawns;
	int pt_knights;
	int pt_bishops;
	int pt_rooks;
	int pt_queens;
	int pt_kings;
	// advanced_endgame_eval decomposition (endgame REPLACE path). ae_input = additive pre-advanced total
	// fed in; ae_matedrive = the |total|>2000 king-to-edge mate-drive delta; ae_passer = the passed-pawn
	// king-race shepherding delta. (0 when advanced_endgame_eval did not fire.)
	int ae_input;
	int ae_matedrive;
	int ae_passer;
	// Cheap board DETECTORS (raw values) for the agentic-PACE diagnosis: compare these across position
	// classes (e.g. positions a change HELPED vs HURT) to find the discriminator to condition a knob on.
	int det_w_offense;
	int det_b_offense;
	int det_w_defense;
	int det_b_defense;
	int det_w_pieceval;
	int det_b_pieceval;
	int det_central;
	int det_pawn_count;
	int det_ks_units_w;   // raw KS attack-units per king (to MEASURE the KS_FLOOR deadzone threshold)
	int det_ks_units_b;
	int det_w_mobility;   // squares each side attacks that are not its own (cheap mobility/control proxy)
	int det_b_mobility;
};
extern EvalBreakdown g_eval_breakdown;
extern bool g_capture_eval_breakdown;
EvalBreakdown eval_breakdown_capture(int moveNum, bool turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied);

/*
	Compile-gated per-term eval profiler (PROFILE_EVAL=1 build only).

	A generic, reusable cycle-accounting primitive: a RAII `ProfScope` brackets a
	region with __rdtsc() on enter/exit and accumulates the delta into a per-term
	bucket, placed via the PROF_BLOCK(term) macro. Production builds compile
	PROF_BLOCK to a no-op and emit none of the EVAL_PROFILE symbols, so the
	shipped binary is byte-identical. We read relative %-shares across terms, not
	absolute nanoseconds. The enum is extensible — search terms can be appended
	later and instrumented the same way, with no change to the framework.

	The three API functions below are declared unconditionally (so the Cython
	bridge can always call them); their bodies are #ifdef-walled and become
	no-ops in a production build.
*/
void eval_profile_reset();
void eval_profile_dump(const char* label);
// Tight driver: evaluate one position `reps` times so the per-term accumulators
// gather signal; the rep loop lives outside every ProfScope.
void eval_profile_run(int moveNum, bool turn, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask, int reps);
// Accessors so the Python harness can read the accumulators and build its own
// per-term x per-phase table without parsing stderr. All return 0 / "" in a
// production (non-EVAL_PROFILE) build.
int eval_profile_num_terms();
unsigned long long eval_profile_cycles(int term);
unsigned long long eval_profile_calls(int term);
const char* eval_profile_name(int term);
int eval_profile_num_exclusive();

#ifdef EVAL_PROFILE
enum ProfTerm {
	PROF_PAWNS, PROF_KNIGHTS, PROF_BISHOPS, PROF_ROOKS, PROF_ROOK_ACTIVITY,
	PROF_QUEENS, PROF_KINGS, PROF_ATTACK_LAYER, PROF_CAPTURE_GAINS,
	PROF_PASSED_SUPPORT, PROF_LATENT_THREAT, PROF_KING_SAFETY, PROF_ADV_ENDGAME,
	PROF_INIT_PIECE_VALUES,
	// Drill-only terms (nested inside the evaluators above; subsets, not exclusive).
	PROF_SEE, PROF_BISHOP_ACTIVITY, PROF_BISHOP_COLOUR,
	// Search-side scopes (NOT part of the eval %-base; their absolute cycles vs the eval-term
	// sum give the whole-search breakdown when accumulated during a real get_engine_move search).
	PROF_MOVEGEN, PROF_MAKEUNMAKE, PROF_TT_PROBE,
	// Drill-only subsets of PROF_MOVEGEN: pseudo-legal generation, the scoring loop, and the
	// sort + Move rebuild. Nested, so excluded from the %-base; their sum ~= PROF_MOVEGEN.
	PROF_MG_GEN, PROF_MG_SCORE, PROF_MG_SORT,
	NUM_PROF_TERMS
};

extern uint64_t g_prof_cycles[NUM_PROF_TERMS];
extern uint64_t g_prof_calls[NUM_PROF_TERMS];

struct ProfScope {
	ProfTerm term;
	uint64_t start;
	ProfScope(ProfTerm t) : term(t), start(__rdtsc()) {}
	~ProfScope() {
		g_prof_cycles[term] += __rdtsc() - start;
		g_prof_calls[term] += 1;
	}
};

#define PROF_BLOCK(term) ProfScope _prof_guard(term)
#else
#define PROF_BLOCK(term) ((void)0)
#endif // EVAL_PROFILE

inline int get_pressure_increment(uint8_t last_moved_to_square, uint64_t bb, bool turn);

inline uint8_t lowest_value_attacker(uint64_t attackers, bool attackedColour);
inline void apply_basic_capture(uint8_t from, uint8_t to, uint64_t& white_pieces, uint64_t& black_pieces, bool white_to_move);
inline CaptureInfo* find_last_viable_capture(CaptureStack& captures, uint64_t& white_pieces, uint64_t& black_pieces, bool captureColour);
inline std::optional<CaptureInfo> find_and_pop_last_viable_capture(CaptureStack& captures, uint64_t white_pieces, uint64_t black_pieces, bool captureColour);
inline bool can_evade(uint8_t target_square, bool target_colour);
inline int approximate_capture_gains(uint64_t bb, bool turn, const BoardState& state, const std::array<int, 64>& pawn_rank_bonuses);
inline int approximate_capture_gains1(uint64_t bb, bool turn);

void initializePieceValues(uint64_t bb);
uint8_t piece_type_at(uint8_t square);
inline void setAttackingLayer(int increment, bool isEndGame);
void printLayers();
inline int getPPIncrement(bool colour, uint64_t opposingPawnMask, int ppIncrement, uint8_t x, uint8_t y, uint64_t opposingPieces, uint64_t curSidePieces, uint64_t& white_passed_pawns, uint64_t& black_passed_pawns);

/*
	Set of functions used to generate moves
*/
void generateLegalMoves(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

// Existence-only variant of generateLegalMoves: returns true on the FIRST legal move
// instead of materializing the whole list, for the checkmate/stalemate tests that only
// need "are there any legal moves". Generates the same pseudo-legal/evasion set, then
// short-circuits the is_safe filter -- so its boolean is identical to
// generateLegalMoves(...).size() != 0, but it skips the per-move legality check for the
// whole tail once one legal move is found.
bool has_any_legal_move(uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

inline void generatePseudoLegalMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions,  uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
	 						  uint64_t king, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
							  uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
							  
inline void generatePieceMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t our_pieces, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask,
	 					uint64_t occupiedWhite, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask);

inline void generatePawnMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t opposingPieces, bool colour, uint64_t pawnsMask, uint64_t occupiedMask,
					   uint64_t from_mask, uint64_t to_mask);

inline void generateCastlingMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint64_t to_mask, uint64_t king,
	                       uint64_t opposingPieces, uint64_t occupiedMask, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, bool turn);

inline void generateEnPassentMoves(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t from_mask, uint64_t to_mask, uint64_t our_pieces, uint64_t occupiedMask, uint64_t pawnsMask, int ep_square, bool turn);

void generateEvasions(std::vector<uint8_t> &startPos, std::vector<uint8_t> &endPos, std::vector<uint8_t> &promotions, uint64_t preliminary_castling_mask, uint8_t king, uint64_t checkers, uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces,
					  uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

inline void generateLegalCaptures(std::vector<uint8_t> &startPos_filtered, std::vector<uint8_t> &endPos_filtered, std::vector<uint8_t> &promotions_filtered, uint64_t from_mask, uint64_t to_mask,
	 					uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
						uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

inline void generateLegalMovesReordered(std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask, uint64_t from_mask, uint64_t to_mask,
								 uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
								 uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
								 
template<std::size_t N>
inline void processMaskPairs(const std::array<MaskPair, N>& mask_pairs, std::vector<uint8_t>& startPos, std::vector<uint8_t>& endPos, std::vector<uint8_t>& promotions, uint64_t preliminary_castling_mask,
	                  uint64_t from_mask, uint64_t to_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
                      uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn, uint8_t king, uint64_t blockers, uint64_t checkers);


inline uint64_t attackersMask(bool colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co);
inline uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied);
inline uint64_t betweenPieces(uint8_t a, uint8_t b);
inline uint64_t ray(uint8_t a, uint8_t b);
inline void update_bitmasks(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, uint64_t occupiedMask);
inline bool is_into_check(uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, 
	               uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);

inline bool is_safe(uint8_t king, uint64_t blockers, uint8_t from_square, uint8_t to_square, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask,
			 uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, int ep_square, bool turn);
inline bool is_castling(uint8_t from_square, uint8_t to_square, bool turn, uint64_t ourPieces, uint64_t rooksMask, uint64_t kingsMask);
inline bool is_en_passant(uint8_t from_square, uint8_t to_square, int ep_square, uint64_t occupiedMask, uint64_t pawnsMask);
inline uint64_t pin_mask(bool colour, int square, uint8_t king, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask);
inline bool ep_skewered(int king, int capturer, int ep_square, bool turn, uint64_t occupiedMask, uint64_t opposingPieces, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask);
inline bool attackedForKing(bool opponent_color,uint64_t path, uint64_t occupied, uint64_t opposingPieces, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask);

// Promote a move to the front of the moveList vector
inline void promoteMoveToFront(std::vector<Move>& moveList, Move move) {
    auto it = std::find(moveList.begin(), moveList.end(), move);
    if (it != moveList.end() && it != moveList.begin()) {
        Move temp = *it;
        moveList.erase(it); // O(n), but small list, so acceptable
        moveList.insert(moveList.begin(), temp); // shift others right
    }
}


inline uint8_t piece_type_at(uint8_t square){
    
	/*
		Function to get the piece type
		
		Parameters:
		- square: The square to be analyzed		
		
		Returns:
		A unsigned char representing the piece type
	*/
    
	/* 
		In this section, see if the individual piece type masks bitwise Anded with the square exists
		If so this suggests that the given square has the piece type of that mask
	*/
	uint64_t mask = (BB_SQUARES[square]);

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

	startPos.reserve(32);
	endPos.reserve(32);
	promotions.reserve(32);

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
inline bool is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bool  is_en_passant);
inline bool is_check(bool colour, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t opposingPieces);

inline bool is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn);
inline bool is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn);
inline uint8_t scan_reversed_size(uint64_t bb);
inline void scan_reversed(uint64_t bb, std::vector<uint8_t> &result);
inline std::vector<uint8_t> scan_reversedOld(uint64_t bb);
inline void scan_forward(uint64_t bb, std::vector<uint8_t> &result);
inline uint64_t attacks_mask(bool colour, uint64_t occupied, uint8_t square, uint8_t pieceType);
inline uint8_t square_distance(uint8_t sq1, uint8_t sq2);

inline int remove_piece_at(uint8_t square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted);
inline void set_piece_at(uint8_t square, uint8_t piece_type, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, bool promotedFlag, bool turn);
inline void update_state(uint8_t to_square, uint8_t from_square, uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask, uint64_t& rooksMask, uint64_t& queensMask, uint64_t& kingsMask, uint64_t& occupiedMask, uint64_t& occupiedWhite, uint64_t& occupiedBlack, uint64_t& promoted, uint64_t& castling_rights, int& ep_square, int promotion_type, bool turn);
inline std::string create_fen(uint64_t& pawnsMask, uint64_t& knightsMask, uint64_t& bishopsMask,
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
	
	uint64_t touched = /* (1ULL << from_square) ^ */ (1ULL << to_square);
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

inline std::vector<Move> accessMoveGenCache(uint64_t key, uint64_t castling_rights, int ep_square);
inline bool moveGenCacheHasMoves(uint64_t key, uint64_t castling_rights, int ep_square);

inline bool is_checkmate(uint64_t zobrist, uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn){

	if (!is_check(turn, occupiedMask, (queensMask | rooksMask), (queensMask | bishopsMask), kingsMask, knightsMask, pawnsMask, opposingPieces))
		return false;

	if (moveGenCacheHasMoves(zobrist, preliminary_castling_mask, ep_square))
		return false;

	return !has_any_legal_move(preliminary_castling_mask, ~0ULL, ~0ULL,
	 				   occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
					   rooksMask, queensMask, kingsMask, ep_square, turn);

}

inline bool is_stalemate(uint64_t zobrist, uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bool turn){

	if (is_check(turn, occupiedMask, (queensMask | rooksMask), (queensMask | bishopsMask), kingsMask, knightsMask, pawnsMask, opposingPieces))
		return false;

	if (moveGenCacheHasMoves(zobrist, preliminary_castling_mask, ep_square))
		return false;

	return !has_any_legal_move(preliminary_castling_mask, ~0ULL, ~0ULL,
	 				   occupiedMask, occupiedWhite, opposingPieces, ourPieces, pawnsMask, knightsMask, bishopsMask,
					   rooksMask, queensMask, kingsMask, ep_square, turn);

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

	if (piece_type == 0 || (bool(to_bb & occupiedWhite) && (bool(to_bb & occupiedWhite) == turn))) {
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

inline int get_pressure_at_square(bool current_colour, uint8_t r){
	int pressure = current_colour ? pressure_white[r] : pressure_black[r];
    return pressure;
}

inline int get_support_at_square(bool current_colour, uint8_t r){	
    int support = current_colour ? support_black[r] : support_white[r];
	return support;
}

inline int get_value_at(uint8_t square, const BoardState& state){
	uint64_t mask = BB_SQUARES[square];
	if (state.pawns & mask) {
		return values[PAWN];
	} else if (state.knights & mask){
		return values[KNIGHT];
	} else if (state.bishops & mask){
		return values[BISHOP];
	} else if (state.rooks & mask){
		return values[ROOK];
	} else if (state.queens & mask){
		return values[QUEEN];
	} else if (state.kings & mask){
		return values[KING];
	}
	return 0;
}

inline int get_least_valuable_attacker(uint64_t attackers, const BoardState& state){

	uint64_t bb = attackers;
	uint8_t r = 0;

	int least_value = 999999999;
	int least_value_square = -1;
	while (bb) {		
		r = __builtin_ctzll(bb);  
		
		if(square_values[r] < least_value){
			least_value = square_values[r];
			least_value_square = r;
		}
			
		bb &= bb - 1;  
	}
	return least_value_square;

	/* if ((attackers & pawns) != 0)
        return __builtin_ctzll(attackers & pawns); // pawn
    else if ((attackers & knights) != 0)
        return __builtin_ctzll(attackers & knights);  // knight
    else if ((attackers & bishops) != 0)
        return __builtin_ctzll(attackers & bishops);  // bishop
    else if ((attackers & rooks) != 0)
        return __builtin_ctzll(attackers & rooks);  // rook
    else if ((attackers & queens) != 0)
        return __builtin_ctzll(attackers & queens);  // queen

	return 0; // or INT_MAX, or some sentinel for "no attacker" */
}

inline int get_least_valuable_attacker_static(uint64_t attackers, const BoardState& state) {
    if ((attackers & state.pawns) != 0) {
        return __builtin_ctzll(attackers & state.pawns); // pawn
    } else if ((attackers & state.knights) != 0) {
        return __builtin_ctzll(attackers & state.knights);  // knight
    } else if ((attackers & state.bishops) != 0) {
        return __builtin_ctzll(attackers & state.bishops);  // bishop
    } else if ((attackers & state.rooks) != 0) {
        return __builtin_ctzll(attackers & state.rooks);  // rook
    } else if ((attackers & state.queens) != 0) {
        return __builtin_ctzll(attackers & state.queens);  // queen
    } else if ((attackers & state.kings) != 0) {
        return __builtin_ctzll(attackers & state.kings);  // king (very rare)
    }


    return 0; // or INT_MAX, or some sentinel
}


inline void update_attackers_for_piece_removal(uint8_t to_square, uint64_t& occupancy, uint64_t colour_attackers[2], bool turn, const BoardState& state){
	colour_attackers[turn] = attackersMask(turn, to_square, occupancy, (state.queens | state.rooks) & occupancy, (state.queens | state.bishops) & occupancy, state.kings & occupancy, state.knights & occupancy, state.pawns & occupancy, state.occupied_colour[turn] & occupancy);
}

inline int see_impl(uint8_t to_square, bool side_to_move, const BoardState& state){

	bool breaks_early = false;
    int gain[32];
    int depth = 0;

    //int piece_value_captured = square_values[to_square];
	int piece_value_captured = get_value_at(to_square, state);
	//std::cout << "piece_value_captured: " << piece_value_captured << std::endl;

    gain[depth++] = piece_value_captured;

    if (Config::ENABLE_SEE_INCREMENTAL) {
        // Byte-identical to the ENABLE_SEE_FIX path, but maintains the attacker set incrementally:
        // clear the used attacker each step and OR in only the x-ray sliders re-revealed through it,
        // instead of recomputing the full attackersMask() every iteration. Removal only OPENS slider
        // rays (never blocks) and never adds leaper (king/knight/pawn) attackers, so the maintained set
        // equals a fresh attackersMask(occ) each step -> same least-valuable pick -> same gain[].
        int d = 0;
        bool s = side_to_move;
        uint64_t occ = state.occupied;
        uint64_t qr = state.queens | state.rooks;
        uint64_t qb = state.queens | state.bishops;
        // All attackers of to_square (both colours), masked to the live occupancy.
        uint64_t att = (BB_KING_ATTACKS[to_square] & state.kings) |
                       (BB_KNIGHT_ATTACKS[to_square] & state.knights) |
                       (BB_RANK_ATTACKS[to_square][BB_RANK_MASKS[to_square] & occ] & qr) |
                       (BB_FILE_ATTACKS[to_square][BB_FILE_MASKS[to_square] & occ] & qr) |
                       (BB_DIAG_ATTACKS[to_square][BB_DIAG_MASKS[to_square] & occ] & qb) |
                       (BB_PAWN_ATTACKS[0][to_square] & state.pawns & state.occupied_colour[true]) |
                       (BB_PAWN_ATTACKS[1][to_square] & state.pawns & state.occupied_colour[false]);
        att &= occ;
        while (true) {
            uint64_t side_att = att & state.occupied_colour[s] & occ;
            if (!side_att)
                break;
            int from_sq = get_least_valuable_attacker_static(side_att, state);
            ++d;
            gain[d] = get_value_at(from_sq, state) - gain[d - 1];
            occ &= ~(1ULL << from_sq);
            att &= ~(1ULL << from_sq);
            // Re-reveal x-ray sliders through the vacated square (leapers never change).
            att |= ((BB_RANK_ATTACKS[to_square][BB_RANK_MASKS[to_square] & occ] |
                     BB_FILE_ATTACKS[to_square][BB_FILE_MASKS[to_square] & occ]) & qr) |
                   (BB_DIAG_ATTACKS[to_square][BB_DIAG_MASKS[to_square] & occ] & qb);
            att &= occ;
            s = !s;
        }
        while (--d > 0)
            gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        return gain[0];
    }

    if (Config::ENABLE_SEE_FIX) {
        // Corrected SEE (gated; the default-off path below is byte-identical). Recompute the
        // side-to-move's attacker set from the LIVE occupancy each iteration so x-ray reveals for
        // either side are exact (replaces the one-sided update_attackers_for_piece_removal), and pick
        // the least-valuable attacker by true piece TYPE -- consistent with the material gains, not
        // the stale eval-magnitude square_values[]. Mirrors the swap-list reference in
        // diagnostics/see_selfcheck.cpp.
        int d = 0;
        bool s = side_to_move;
        uint64_t occ = state.occupied;
        while (true) {
            uint64_t attackers = attackersMask(s, to_square, occ, (state.queens | state.rooks) & occ, (state.queens | state.bishops) & occ, state.kings & occ, state.knights & occ, state.pawns & occ, state.occupied_colour[s] & occ);
            if (!attackers)
                break;
            int from_sq = get_least_valuable_attacker_static(attackers, state);
            ++d;
            gain[d] = get_value_at(from_sq, state) - gain[d - 1];
            occ &= ~(1ULL << from_sq);
            s = !s;
        }
        while (--d > 0)
            gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        return gain[0];
    }

    bool stm = side_to_move;
    uint64_t occupancy = state.occupied;
    uint64_t color_attackers[2] = {attackersMask(false, to_square, occupancy, (state.queens | state.rooks) & occupancy, (state.queens | state.bishops) & occupancy, state.kings & occupancy, state.knights & occupancy, state.pawns & occupancy, state.occupied_colour[false] & occupancy)
		                          ,attackersMask(true, to_square, occupancy, (state.queens | state.rooks) & occupancy, (state.queens | state.bishops) & occupancy, state.kings & occupancy, state.knights & occupancy, state.pawns & occupancy, state.occupied_colour[true] & occupancy)};
		
	//std::cout << "WHITE: " << color_attackers[1] << std::endl;
	//std::cout << "BLACK: " << color_attackers[0] << std::endl;
    while (true) {
		//std::cout << "AAA: " << color_attackers[stm] << std::endl;
        uint64_t attackers_now = color_attackers[stm];
        if (!attackers_now) break;
		//std::cout << "BBB" << std::endl;
        // Get least valuable attacker
        int from_sq = get_least_valuable_attacker(attackers_now, state);
        //int piece_val = square_values[from_sq];
		int piece_val = get_value_at(from_sq, state);
		
		/* std::cout << "CCc" << std::endl;
		std::cout << "piece val former: " << gain[depth - 1] << std::endl;
		std::cout << "piece val current: " << piece_val << " | " << (int)from_sq << std::endl;
 		*/
        gain[depth] = piece_val - gain[depth - 1];
        
		/* if (std::max(-gain[depth - 1], gain[depth]) < 0){
			breaks_early = true;
			break;
		} */
            
		//std::cout << "DDD" << std::endl;
        // Remove attacker from occupancy & update attackers
        occupancy &= ~(1ULL << from_sq);
        
        update_attackers_for_piece_removal(to_square, occupancy, color_attackers, stm, state);
		//std::cout << "EEE" << std::endl;
        stm = !stm;
        depth++;
    }
	//std::cout << depth << " SEE final gain[0]: " << gain[depth - 1]  << std::endl;
    
	// Propagate max gain back
	
	if (breaks_early) {
		if (depth > 1){
			for (int i = depth - 1; i > 0; --i) {
				gain[i - 1] = -std::max(-gain[i - 1], gain[i]);
			}
		}
		
	} else {
		if (depth > 2){
			for (int i = depth - 2; i > 0; --i) {
				gain[i - 1] = -std::max(-gain[i - 1], gain[i]);
			}
		}
	}
	
	//std::cout << depth << "SEE final gain[0]: " << gain[depth] << std::endl;
    return gain[0];
}

// Cached SEE: see() is pure, so a per-position cache (keyed by [side][square], generation-validated) returns
// identical values -> byte-identical search with the cache on. Default (cache off) just forwards to see_impl,
// matching the original behavior exactly. The big win is the qsearch capture_gains<->noisy-filter overlap
// (both call see() on the same position's squares within one node).
inline int see(uint8_t to_square, bool side_to_move, const BoardState& state){
	if (Config::SEE_COUNT) ++see_calls;
	if (Config::ENABLE_SEE_CACHE){
		if (see_cache_gen[side_to_move][to_square] == g_see_gen){
			if (Config::SEE_COUNT) ++g_see_hits;
			return see_cache_val[side_to_move][to_square];
		}
		int v = see_impl(to_square, side_to_move, state);
		see_cache_gen[side_to_move][to_square] = g_see_gen;
		see_cache_val[side_to_move][to_square] = v;
		if (Config::SEE_COUNT) ++g_see_miss;
		return v;
	}
	return see_impl(to_square, side_to_move, state);
}


/* inline int get_see_score(uint8_t to_square, bool side_to_move, const BoardState& state){
	
	bool breaks_early = false;
    int gain[32];
    int depth = 0;

    //int piece_value_captured = square_values[to_square];
	int piece_value_captured = get_value_at(to_square, state);
    gain[depth++] = piece_value_captured;

    bool stm = side_to_move;
    uint64_t occupancy = state.occupied;
    uint64_t color_attackers[2] = {attackersMask(false, to_square, occupancy, (state.queens | state.rooks) & occupancy, (state.queens | state.bishops) & occupancy, state.kings & occupancy, state.knights & occupancy, state.pawns & occupancy, state.occupied_colour[false] & occupancy)
		                          ,attackersMask(true, to_square, occupancy, (state.queens | state.rooks) & occupancy, (state.queens | state.bishops) & occupancy, state.kings & occupancy, state.knights & occupancy, state.pawns & occupancy, state.occupied_colour[true] & occupancy)};

    while (true) {
        uint64_t attackers_now = color_attackers[stm];
        if (!attackers_now) break;

        // Get least valuable attacker
        int from_sq = get_least_valuable_attacker_static(attackers_now, state);
        //int piece_val = square_values[from_sq];
		int piece_val = get_value_at(from_sq, state);

        gain[depth] = piece_val - gain[depth - 1];
        if (std::max(-gain[depth - 1], gain[depth]) < 0){
			breaks_early = true;
			break;
		}
            

        // Remove attacker from occupancy & update attackers
        occupancy &= ~(1ULL << from_sq);
        //color_attackers[stm] &= ~(1ULL << from_sq);
        update_attackers_for_piece_removal(to_square, occupancy, color_attackers, stm, state);

        stm = !stm;
        depth++;
    }

    // Propagate max gain back
    if(breaks_early){
		for (depth; depth > 0; --depth){
			//std::cout <<gain[depth] <<  " : " << -gain[depth - 1] << std::endl;
			gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
		}
	}else{
		for (--depth; depth > 1; --depth){
			//std::cout <<gain[depth - 1] <<  " : " << -gain[depth - 2] << std::endl;
			gain[depth - 2] = -std::max(-gain[depth - 2], gain[depth - 1]);
		}
	}

    return gain[0];
} */

#endif // CPP_BITBOARD_H
