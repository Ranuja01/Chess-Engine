#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <atomic>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

constexpr int TIME_CHECK_INTERVAL = 200000;

// Constants for material thresholds
constexpr int MIN_MATERIAL_FOR_NULL_MOVE = 15000;

// Number of nodes before decay
constexpr int DECAY_FACTOR = 1; // Divide scores by 2

constexpr std::array<int, 4> FUTILITY_MARGINS = {200, 450, 650, 950};

constexpr int MAX_QDEPTH = 10;
constexpr int SUPPORT_MARGIN = 0;
constexpr int DELTA_MARGIN = 1500;

constexpr bool USE_Q_SEARCH = true;

struct TTEntry;
struct QCacheEntry;
struct MoveEntry;

struct ConfigData
{
    int cache_size_multiplier;
    double TIME_LIMIT;
    std::array<double, 64> MOVE_TIMES;
    std::array<int, 64> DEPTH_REDUCTION;
};

namespace Configs
{

    constexpr ConfigData LIGHTNING = {
        2,
        1.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 0.5;
            times[4] = 0.5;
            times[5] = 0.5;
            times[6] = 0.75;
            times[7] = 0.75;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 0.75;
            }
            return times;
        }(),

        []
        {
            std::array<int, 64> new_depths{};
            new_depths[1] = 0;
            new_depths[2] = 1;
            new_depths[3] = 2;
            new_depths[4] = 3;
            new_depths[5] = 4;
            new_depths[6] = 5;
            new_depths[7] = 6;
            new_depths[8] = 7;
            new_depths[9] = 8;
            new_depths[10] = 8;
            new_depths[11] = 9;
            new_depths[12] = 10;
            new_depths[13] = 11;
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData BLITZ = {
        2,
        3.5,
        []
        {
            std::array<double, 64> times{};
            times[3] = 1.0;
            times[4] = 1.0;
            times[5] = 1.5;
            times[6] = 1.5;
            times[7] = 1.5;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 2.0;
            }
            return times;
        }(),

        []
        {
            std::array<int, 64> new_depths{};
            new_depths[1] = 0;
            new_depths[2] = 1;
            new_depths[3] = 2;
            new_depths[4] = 3;
            new_depths[5] = 4;
            new_depths[6] = 5;
            new_depths[7] = 6;
            new_depths[8] = 7;
            new_depths[9] = 8;
            new_depths[10] = 8;
            new_depths[11] = 9;
            new_depths[12] = 10;
            new_depths[13] = 11;
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData STANDARD = {
        2,
        45.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            times[6] = 5.5;
            times[7] = 6.5;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 6.0;
            }
            return times;
        }(),

        []
        {
            std::array<int, 64> new_depths{};
            new_depths[1] = 0;
            new_depths[2] = 1;
            new_depths[3] = 2;
            new_depths[4] = 3;
            new_depths[5] = 4;
            new_depths[6] = 5;
            new_depths[7] = 6;
            new_depths[8] = 7;
            new_depths[9] = 8;
            new_depths[10] = 8;
            new_depths[11] = 9;
            new_depths[12] = 10;
            new_depths[13] = 11;
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData LONG_FORMAT = {
        3,
        600.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            for (int i = 6; i < 64; ++i)
            {
                times[i] = 120.0;
            }
            return times;
        }(),
        []
        {
            std::array<int, 64> new_depths{};
            new_depths[1] = 0;
            new_depths[2] = 1;
            new_depths[3] = 2;
            new_depths[4] = 3;
            new_depths[5] = 4;
            new_depths[6] = 5;
            new_depths[7] = 6;
            new_depths[8] = 7;
            new_depths[9] = 8;
            new_depths[10] = 8;
            new_depths[11] = 9;
            new_depths[12] = 10;
            new_depths[13] = 11;
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};
}

namespace Config
{
    inline const ConfigData *ACTIVE = &Configs::STANDARD; // Default to classical
    inline bool side_to_play = false;                     // Default; can be set at runtime
    inline int DECAY_INTERVAL = 35000;

    // Search-ablation toggles (default ON = current behavior). Each disables a
    // pruning/reduction mechanism at all of its live sites so the eval-vs-search
    // diagnostic can localize which one drops winning lines. Overridable at runtime
    // from the environment (see env_flag in initialize_engine); since true && X == X
    // the default build is byte-identical.
    inline bool ENABLE_LMR = true;      // late move reductions
    inline bool ENABLE_FUTILITY = true; // futility pruning (inside the LMR block)
    inline bool ENABLE_RAZORING = true; // razoring (alpha_beta root loop)
    inline bool ENABLE_NULLMOVE = true; // null-move pruning
    inline bool ENABLE_QDELTA = true;   // delta pruning in quiescence

    inline bool LMR_PROFILE = false; // env-gated LMR-miss profiler (diagnostic; off = byte-identical)

    // Sound-LMR exemptions (default OFF = current behavior). Stop reducing the
    // moves most likely to be the critical misses; env-gated so the A/B needs no
    // recompile and the default build stays byte-identical.
    inline bool PROTECT_KILLERS = false; // don't LMR-reduce killer / counter moves
    inline bool PROTECT_PV = false;      // don't LMR-reduce at PV nodes (beta - alpha > 1)

    // Margin-gated verification re-search: when > 0, a reduced move that fails low
    // by less than this margin (a near-miss) is re-searched. The one mechanism that
    // uses the "how close to alpha" signal. The default is the blitz-validated
    // keeper (margin 6000 with VERIFY_RESEARCH_REDUCTION 2); set 0 to disable it
    // and recover the old byte-identical search (e.g. for the d10 isolation control).
    inline int VERIFY_MARGIN = 6000;

    // Graduated verification: re-search the near-miss at depth_limit - this (a
    // shallow re-look) instead of full depth, to cut the cost of a wide VERIFY_MARGIN.
    // 0 = full-depth re-search (the original VERIFY behavior); larger = cheaper. The
    // default 2 is the validated config: a clean +17 at LIGHTNING (real-play) without
    // the ply-cost of the deeper re-search that made margin-6000/reduction-1 regress.
    inline int VERIFY_RESEARCH_REDUCTION = 2;

    // Exclusive upper bound on the iterative-deepening depth_limit. Default 64 is
    // the normal play cap (a time-limited preset governs the actual depth reached);
    // set the MAX_DEPTH env knob to 11 to pin the fixed-depth-10 isolation control.
    inline int MAX_ITERATIVE_DEPTH = 64;

    // Occurrence count at which an in-search repeated position is scored as a draw.
    // Default 2 treats the first repetition on the search path as a draw — the standard,
    // game-theoretically sound search behavior (if a line can repeat once, the side that
    // wants the draw can force the threefold), and the fix for the perpetual-check blind
    // spot where the winning side over-values a line that is really a forced draw. Set to
    // 3 to restore the old true-threefold search (only every third occurrence is a draw).
    inline int REPETITION_THRESHOLD = 2;

    // Check extension: a move that GIVES CHECK searches its child one ply deeper so forcing
    // lines (perpetuals, mating attacks) are reached without spending general depth. The
    // value is the hard cap on extensions per root-to-leaf path. Default 3 is the validated
    // keeper (cap-sweep plateaus ~3-4: +20 WAC@LIGHTNING / +9 @d10, and free on the clock in
    // time-limited play since mates found early hit the score cutoff). 0 = off. NOTE: only
    // checks are extended today; "forcing" (recaptures/promotions) is unimplemented.
    inline int CHECK_EXTENSION = 3;

    // SEE filter on the check extension. When SEE_EXTEND_MARGIN is set below
    // SEE_EXTEND_DISABLED, a checking move is extended only if the opponent cannot win its
    // checker (static exchange on the checker's square) by more than SEE_EXTEND_MARGIN — i.e.
    // spite checks (a hanging checker) are skipped. 0 = sound checks only; positive (in piece-
    // value units, pawn=1000) = tolerate small sacrifices. Default DISABLED = extend every
    // check (current behavior; no see() call is made, so the default build is byte-identical).
    inline constexpr int SEE_EXTEND_DISABLED = 1000000;
    inline int SEE_EXTEND_MARGIN = SEE_EXTEND_DISABLED;

    // Honest bound flags at the root / preliminary-ordering TT stores. Those four stores
    // (alpha_beta and reorder_legal_moves) hardcode TTFlag::EXACT, which is only correct under
    // the infinite root window where the returned fail-soft score always lands in-window. With a
    // narrowed root window (aspiration) the score is routinely a fail-low/high bound, so storing
    // it as EXACT poisons the TT for the re-search and later iterations. When true, the flag is
    // computed from the score's position in the window (score <= alpha -> UPPERBOUND, score >=
    // beta -> LOWERBOUND, else EXACT), the same way every deeper store already does. Required for
    // sound aspiration windows; harmless on its own. Default true ships alongside aspiration
    // (below); set 0 to recover the hardcoded-EXACT build (e.g. the byte-identical d10 control).
    inline bool HONEST_ROOT_TT = true;

    // Aspiration windows: search each iterative-deepening iteration (from ASPIRATION_MIN_DEPTH up)
    // in a narrow window centred on the previous iteration's score instead of the full window, to
    // win cutoffs and depth. ASPIRATION_DELTA is the initial half-width in piece-value units
    // (pawn=1000); a failing side is widened by ASPIRATION_WIDEN_PCT percent per resize (200 = x2)
    // up to ASPIRATION_MAX_WIDENINGS times, then the search falls back to the full window. Requires
    // HONEST_ROOT_TT (a narrow window is unsound without it). Default DELTA 500 is the sweep-picked
    // value (efficiency sweet spot, +0.32 LIGHTNING depth at equal clock, STS +2.2pp); set 0 to
    // disable (full window every iteration = the byte-identical control). An overnight LIGHTNING
    // confirm of 800/1200 may yet bump this — see dev_notes/OPTIMIZATION_LOG.md.
    inline int ASPIRATION_DELTA = 500;
    inline int ASPIRATION_MIN_DEPTH = 5;
    inline int ASPIRATION_MAX_WIDENINGS = 3;
    inline int ASPIRATION_WIDEN_PCT = 200;

    // Main TT (searchEvalCache) associativity. 1 = direct-mapped (one entry per index, today's
    // behavior, byte-identical). 2/4/8 = N-way set-associative: the index selects a bucket of N
    // entries and a conflicting store evicts the shallowest, so expensive deep entries survive
    // collisions longer (more big subtree-skips land). Must be a power of two dividing
    // TT_CACHE_SIZE; the footprint is unchanged (same array, reinterpreted as TT_CACHE_SIZE/TT_WAYS
    // buckets). Default 1 = off.
    inline int TT_WAYS = 1;
    // Debug-only (default off = byte-identical): validate the SearchData parallel-array invariant and
    // move legality in the search, logging violations to stderr instead of crashing. Enabled with
    // CHESS_DEBUG_INVARIANTS=1 to hunt the move-ordering corruption.
    inline bool DEBUG_INVARIANTS = false;
}

struct BoardState
{
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;

    uint64_t occupied_colour[2]; // occupied[0] = black, occupied[1] = white
    uint64_t occupied;

    uint64_t promoted;

    bool turn;
    uint64_t castling_rights;

    int ep_square;
    int halfmove_clock;
    int fullmove_number;

    BoardState(uint64_t pawns,
               uint64_t knights,
               uint64_t bishops,
               uint64_t rooks,
               uint64_t queens,
               uint64_t kings,
               uint64_t occupied_white,
               uint64_t occupied_black,
               uint64_t occupied,
               uint64_t promoted,
               bool turn,
               uint64_t castling_rights,
               int ep_square,
               int halfmove_clock,
               int fullmove_number)
        : pawns(pawns),
          knights(knights),
          bishops(bishops),
          rooks(rooks),
          queens(queens),
          kings(kings),
          occupied(occupied),
          promoted(promoted),
          turn(turn),
          castling_rights(castling_rights),
          ep_square(ep_square),
          halfmove_clock(halfmove_clock),
          fullmove_number(fullmove_number)
    {
        occupied_colour[0] = occupied_black;
        occupied_colour[1] = occupied_white;
    }
};

struct MoveData
{
    int a;
    int b;
    int c;
    int d;
    int promotion;
    int score;
    int num_iterations;

    // Constructor with default values
    MoveData(int a_ = 0, int b_ = 0, int c_ = 0, int d_ = 0, int promotion_ = 0, int score_ = 0, int num_iterations_ = 0)
        : a(a_), b(b_), c(c_), d(d_), promotion(promotion_), score(score_), num_iterations(num_iterations_) {}
};

struct Move
{
    uint8_t from_square;
    uint8_t to_square;
    uint8_t promotion;

    // Constructor with default values
    Move(uint8_t from_square_ = 0, uint8_t to_square_ = 0, uint8_t promotion_ = 0)
        : from_square(from_square_), to_square(to_square_), promotion(promotion_) {}

    bool operator==(const Move &other) const
    {
        return from_square == other.from_square &&
               to_square == other.to_square &&
               promotion == other.promotion;
    }
};

struct SearchData
{

    // Set of moves and preliminary scores for the top level
    std::vector<Move> moves_list;
    std::vector<int> top_level_preliminary_scores;

    // Set of moves and preliminary scores for the second recursive depth
    std::vector<std::vector<Move>> second_level_moves_list;
    std::vector<std::vector<int>> second_level_preliminary_scores;

    // Default constructor
    SearchData() = default;

    // Constructor that initializes all members (optional if you want to pass in initial values)
    SearchData(const std::vector<Move> &moves,
               const std::vector<int> &top_scores,
               const std::vector<std::vector<Move>> &second_moves,
               const std::vector<std::vector<int>> &second_scores)
        : moves_list(moves),
          top_level_preliminary_scores(top_scores),
          second_level_moves_list(second_moves),
          second_level_preliminary_scores(second_scores) {}
};

void initialize_engine(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn, bool side_to_play);
void set_current_state(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn);
inline void make_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, Move move, uint64_t zobrist, bool capture_move);
inline void unmake_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key);

inline void update_cache(int num_plies);

MoveData get_engine_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count);
int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, SearchData &previous_search_data, Move &best_move, int &num_iterations);
int minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> second_level_preliminary_scores, std::vector<Move> second_level_moves_list, SearchData &previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search);
int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search);
SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint &t0, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, int &num_iterations);
int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> &preliminary_scores, std::vector<Move> &pre_moves_list, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations);
int qSearch(int alpha, int beta, int cur_depth, int qDepth, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing);

inline void sortSearchDataByScore(SearchData &data);
inline void descending_sort_wrapper(const SearchData &preSearchData, SearchData &mainSearchData);
inline void ascending_sort(std::vector<int> &values, std::vector<Move> &moves);
inline uint8_t get_piece_type(uint8_t square, std::vector<BoardState> &state_history);
inline bool relevant_pin_exists(std::vector<BoardState> &state_history, bool probe);
inline void use_tt_entry(TTEntry &entry, int &score, bool &using_tt, int alpha, int beta, int &num_iterations, bool is_maximizing, bool using_extra_precautions);
inline void increment_node_count_with_decay(int &num_iterations);
inline bool isUnsafeForNullMovePruning(BoardState current_state);
inline bool is_repetition(const std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key, const int repetition_count);
inline int reduced_search_depth(int depth_limit, int cur_depth, bool is_in_relavent_pin, int move_number, BoardState current_state);
inline void updatePV(Move move, int cur_depth);

inline int get_q_search_eval(int alpha, int beta, int cur_depth, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing);
inline int get_board_evaluation(std::vector<BoardState> &state_history, uint64_t zobrist, int &num_iterations);
inline std::vector<Move> buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove);
inline std::vector<Move> buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, Move prevMove);

#endif // SEARCH_ENGINE_H