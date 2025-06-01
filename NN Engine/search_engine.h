#ifndef SEARCH_ENGINE_H
#define SEARCH_ENGINE_H

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <unordered_map>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace Config {
    constexpr double TIME_LIMIT = 60.0;

    constexpr std::array<double, 64> MOVE_TIMES = [] {
        std::array<double, 64> times{};
        times[3] = 5.0;
        times[4] = 5.0;
        times[5] = 5.5;
        times[6] = 5.5;
        for (int i = 7; i < 64; ++i) {
            times[i] = 2.5;
        }
        return times;
    }();

    inline bool side_to_play = false; // Default; can be set at runtime
}

struct BoardState {
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;

    uint64_t occupied_white;
    uint64_t occupied_black;
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
          occupied_white(occupied_white),
          occupied_black(occupied_black),
          occupied(occupied),
          promoted(promoted),
          turn(turn),
          castling_rights(castling_rights),
          ep_square(ep_square),
          halfmove_clock(halfmove_clock),
          fullmove_number(fullmove_number)
    {}
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
void make_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, Move move, uint64_t zobrist);
void unmake_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key);

void update_cache(int num_plies);

MoveData get_engine_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count);
int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, SearchData& previous_search_data, Move& best_move, int& num_iterations);
int minimizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<int>second_level_preliminary_scores, std::vector<Move>second_level_moves_list, SearchData& previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations);
int maximizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations);
SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, int& num_iterations);
int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<int>& preliminary_scores, std::vector<Move>& pre_moves_list, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations);

void sortSearchDataByScore(SearchData& data);
void descending_sort_wrapper(const SearchData& preSearchData, SearchData& mainSearchData);
void ascending_sort(std::vector<int>& values, std::vector<Move>& moves);
std::vector<Move> buildMoveListFromReordered(std::vector<BoardState>& state_history, uint64_t zobrist);

bool is_repetition(const std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key, const int repetition_count);
int get_board_evaluation(std::vector<BoardState>& state_history, uint64_t zobrist, int& num_iterations);

#endif // SEARCH_ENGINE_H