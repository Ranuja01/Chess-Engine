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

constexpr int TIME_CHECK_INTERVAL = 5000000;

// Constants for material thresholds
constexpr int MIN_MATERIAL_FOR_NULL_MOVE = 15000;

constexpr int DECAY_INTERVAL = 25000; // Number of nodes before decay
constexpr int DECAY_FACTOR = 1;       // Divide scores by 2

constexpr std::array<int, 4> FUTILITY_MARGINS = {200, 500, 700, 1000};

constexpr int MAX_QDEPTH = 12;
constexpr int SUPPORT_MARGIN = 0;
constexpr int DELTA_MARGIN = 100;

constexpr bool USE_Q_SEARCH = true;

struct ConfigData {
    int cache_size_multiplier;
    double TIME_LIMIT;
    std::array<double, 64> MOVE_TIMES;
    std::array<int, 64> DEPTH_REDUCTION;
};

namespace Configs {
    constexpr ConfigData STANDARD = {
        2,
        45.0,
        [] {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            times[6] = 5.5;
            times[7] = 5.0;
            for (int i = 8; i < 64; ++i) {
                times[i] = 4.5;
            }
            return times;
        }(),

        [] {
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
            for (int i = 14; i < 64; ++i) {
                new_depths[i] = 12;
            }
            return new_depths;
        }()
    };

    constexpr ConfigData LONG_FORMAT = {
        5,
        600.0,
        [] {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            for (int i = 6; i < 64; ++i) {
                times[i] = 120.0;
            }
            return times;
        }(),
        [] {
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
            new_depths[10] = 9;
            new_depths[11] = 10;
            new_depths[12] = 11;                        
            for (int i = 13; i < 64; ++i) {
                new_depths[i] = 12;
            }
            return new_depths;
        }()
    };    
}

namespace Config {
    inline const ConfigData* ACTIVE = &Configs::STANDARD; // Default to classical
    inline bool side_to_play = false; // Default; can be set at runtime
}

struct BoardState {
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;

    uint64_t occupied_colour[2];  // occupied[0] = black, occupied[1] = white
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

struct MoveData {
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


struct Move {
    uint8_t from_square;
    uint8_t to_square;
    uint8_t promotion;

    // Constructor with default values
    Move(uint8_t from_square_ = 0, uint8_t to_square_ = 0, uint8_t promotion_ = 0)
        : from_square(from_square_), to_square(to_square_), promotion(promotion_) {}

    bool operator==(const Move& other) const {
        return from_square == other.from_square &&
               to_square == other.to_square &&
               promotion == other.promotion;
    }
};

struct SearchData {

    // Set of moves and preliminary scores for the top level
    std::vector<Move> moves_list;
    std::vector<int> top_level_preliminary_scores;

    // Set of moves and preliminary scores for the second recursive depth
    std::vector<std::vector<Move>> second_level_moves_list;
    std::vector<std::vector<int>> second_level_preliminary_scores;

    // Default constructor
    SearchData() = default;

    // Constructor that initializes all members (optional if you want to pass in initial values)
    SearchData(const std::vector<Move>& moves,
               const std::vector<int>& top_scores,
               const std::vector<std::vector<Move>>& second_moves,
               const std::vector<std::vector<int>>& second_scores)
        : moves_list(moves),
          top_level_preliminary_scores(top_scores),
          second_level_moves_list(second_moves),
          second_level_preliminary_scores(second_scores) {}
};

void initialize_engine(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn, bool side_to_play);
void set_current_state(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn);
void make_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, Move move, uint64_t zobrist, bool capture_move);
void unmake_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key);

void update_cache(int num_plies);

MoveData get_engine_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count);
int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, SearchData& previous_search_data, Move& best_move, int& num_iterations);
int minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<int>second_level_preliminary_scores, std::vector<Move>second_level_moves_list, SearchData& previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move previousMove, int& num_iterations);
int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move previousMove, int& num_iterations, bool last_move_was_capture);
SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint& t0, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, int& num_iterations);
int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<int>& preliminary_scores, std::vector<Move>& pre_moves_list, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move prevMove, int& num_iterations);
int qSearch(int alpha, int beta, int cur_depth, int qDepth, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move prevMove, int& num_iterations, bool is_maximizing);

void sortSearchDataByScore(SearchData& data);
void descending_sort_wrapper(const SearchData& preSearchData, SearchData& mainSearchData);
void ascending_sort(std::vector<int>& values, std::vector<Move>& moves);
bool isUnsafeForNullMovePruning(BoardState current_state);
bool is_repetition(const std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key, const int repetition_count);
int reduced_search_depth(int depth_limit, int move_number, BoardState current_state);

int get_board_evaluation(std::vector<BoardState>& state_history, uint64_t zobrist, int& num_iterations);
std::vector<Move> buildMoveListFromReordered(std::vector<BoardState>& state_history, uint64_t zobrist, int cur_ply, Move prevMove);
std::vector<Move> buildNoisyMoveList(std::vector<BoardState>& state_history, int cur_ply, Move prevMove);

#endif // SEARCH_ENGINE_H