#include "cpp_bitboard.h"
#include "search_engine.h"
#include "move_gen.h"
#include "cache_management.h"

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <cmath>
#include <unordered_set>

std::atomic<bool> time_up;
std::atomic<bool> use_q_precautions;
std::atomic<bool> is_draw;

std::atomic<int> nodes_since_time_check;

std::atomic<int> eval_visits;
std::atomic<int> eval_cache_hits;

std::atomic<int> move_gen_visits;
std::atomic<int> move_gen_cache_hits;

std::atomic<int> tt_visits;
std::atomic<int> tt_probes;
std::atomic<int> tt_hits;

std::atomic<int> nodes;

std::atomic<int> qsearchVisits;

Move pv_table[MAX_PLY][MAX_PLY];
int pv_length[MAX_PLY];


void initialize_engine(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn, bool side_to_play){
    
    BoardState initialState(
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        occupied_white,
        occupied_black,
        occupied,
        promoted,
        turn,                  
        castling_rights,
        ep_square,                          
        halfmove_clock,                           
        fullmove_number    
    );

    state_history.push_back(initialState);

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_colour[true], initialState.occupied_colour[false], initialState.turn);
    position_count[zobrist]++;

    Config::side_to_play = side_to_play;
}

void set_current_state(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn){
    
    BoardState initialState(
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        occupied_white,
        occupied_black,
        occupied,
        promoted,
        turn,                  
        castling_rights,
        ep_square,                          
        halfmove_clock,                           
        fullmove_number    
    );

    state_history.push_back(initialState);

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_colour[true], initialState.occupied_colour[false], initialState.turn);
    position_count[zobrist]++;
}

inline void make_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, Move move, uint64_t zobrist, bool capture_move){
    
    BoardState current = state_history.back();

    uint64_t pawns = current.pawns;
    uint64_t knights = current.knights;
    uint64_t bishops = current.bishops;
    uint64_t rooks = current.rooks;
    uint64_t queens = current.queens;
    uint64_t kings = current.kings;

    uint64_t occupied_white = current.occupied_colour[true];
    uint64_t occupied_black = current.occupied_colour[false];
    uint64_t occupied = current.occupied;

    uint64_t promoted = current.promoted;

    bool turn = current.turn;
    uint64_t castling_rights = current.castling_rights;

    int ep_square = current.ep_square;
    int halfmove_clock = current.halfmove_clock;
    int fullmove_number = current.fullmove_number;
    /* std::cout << "BEFORE: "<< std::endl;
    std::cout << occupied << " | " << occupied_white<< " | " << occupied_black << std::endl; */
    update_state(
        move.to_square, 
        move.from_square,
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        occupied,
        occupied_white,
        occupied_black,
        promoted,
        castling_rights,
        ep_square,
        move.promotion,
        turn
    );
    
    /* std::cout << occupied << " | " << occupied_white<< " | " << occupied_black << std::endl;
    std::cout << "AFTER: "<< std::endl; */
    //halfmove_clock += 1
    // Reset the halfmove clock if the move is a pawn move or capture
    if (capture_move || (BB_SQUARES[move.from_square] & pawns)) {
        halfmove_clock = 0;
    } else {
        halfmove_clock += 1;
    }

    if (!turn)
        fullmove_number += 1;

    //ep_square = -1;
    turn = !turn;

    BoardState newState(
        pawns,
        knights,
        bishops,
        rooks,
        queens,
        kings,
        occupied_white,
        occupied_black,
        occupied,
        promoted,
        turn,                  
        castling_rights,
        ep_square,                          
        halfmove_clock,                           
        fullmove_number    
    );

    position_count[zobrist]++;
    state_history.push_back(newState);
}

inline void unmake_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key){
    state_history.pop_back();

    if (--position_count[zobrist_key] == 0) {
        position_count.erase(zobrist_key);  // Clean up to save space
    }

}

inline void update_cache(int num_plies){
    
    printCacheStats();
    
    /*
    // Code segment to control cache size
    if(num_plies < 30){
        if (cacheSize > 8000000)
            evictOldEntries(cacheSize - 8000000);   
    }else if(num_plies < 50){
        if (cacheSize > 16000000)
            evictOldEntries(cacheSize - 16000000); 
    }else if(num_plies < 75){
        if (cacheSize > 32000000)
            evictOldEntries(cacheSize - 32000000); 
    }else{
        if (cacheSize > 64000000)
            evictOldEntries(cacheSize - 64000000); 
    }
    */
    std::cout << std::endl;
    
    printSearchEvalCacheStats();
    
    std::cout << std::endl;

    printQCacheStats();

    std::cout << std::endl;
    
    printMoveGenCacheStats();
    
    /*
    // Code segment to control cache size
    if(num_plies < 30){
        if (cacheSize > 400000)
            evictOldMoveGenEntries(cacheSize - 400000);   
    }else if(num_plies < 50){
        if (cacheSize > 800000)
            evictOldMoveGenEntries(cacheSize - 800000); 
    }else if(num_plies < 75){
        if (cacheSize > 12000000)
            evictOldMoveGenEntries(cacheSize - 12000000); 
    }else{
        if (cacheSize > 20000000)
            evictOldMoveGenEntries(cacheSize - 20000000); 
    }
    */
    
}

MoveData get_engine_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count){

    update_cache(static_cast<int>(state_history.size()));
    std::fill(&killerMoves[0][0], &killerMoves[0][0] + 64 * 2, Move{});
    std::fill(&counterMoves[0][0], &counterMoves[0][0] + 64 * 64, Move{});
    /* std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move{});
    std::fill(pv_length, pv_length + MAX_PLY, 0); */


    time_up = false;
    use_q_precautions = false;
    is_draw = false;

    nodes_since_time_check = 0;

    eval_visits = 0;
    eval_cache_hits = 0;

    move_gen_visits = 0;
    move_gen_cache_hits = 0;

    tt_visits = 0;
    tt_probes = 0;
    tt_hits = 0;

    qsearchVisits = 0;
    
    BoardState current = state_history.back();

    int phase = 0;
	phase += 4 * __builtin_popcountll(current.queens);
	phase += 2 * __builtin_popcountll(current.rooks);
	phase += 1 * __builtin_popcountll(current.bishops | current.knights);

	int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to 
    
    if (phase_score > 117) {
    	Config::DECAY_INTERVAL = 200000;   
        use_q_precautions = true;
	}else if (phase_score >= 96) {
    	Config::DECAY_INTERVAL = 150000;
        use_q_precautions = true;                   
	} else if (phase_score >= 64) {
		Config::DECAY_INTERVAL = 125000;         
	}  
    use_q_precautions = true;
    uint64_t zobrist = generateZobristHash(current.pawns, current.knights, current.bishops, current.rooks, current.queens, current.kings, current.occupied_colour[true], current.occupied_colour[false], current.turn);

    Move move(0,0,0);

    int depth_limit = 3;
    int num_iterations = 0;
    int alpha = -9999998;
    int beta = 9999999;

    SearchData preliminary_search_data;
    TimePoint search_start_time = Clock::now();
    TimePoint t0 = Clock::now();

    decayMoveFrequency();
    int score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, search_start_time, preliminary_search_data, move, num_iterations);
    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    while (elapsed <= Config::ACTIVE->MOVE_TIMES[depth_limit] && depth_limit + 1 < 64 && score < 9000000 && !time_up){
        
        if (is_draw && score == 0)
            break;
        is_draw = false;
        int x1 = (move.from_square & 7) + 1;
        int y1 = (move.from_square >> 3) + 1;

        int x2 = (move.to_square & 7) + 1;
        int y2 = (move.to_square >> 3) + 1;

        // Resigns
        if (score <= -15000){
            MoveData defaultMove(-1,-1,-1,-1, -1, score, 0);
            return defaultMove; 
        } 

        depth_limit++;

        std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
        std::cout << std::endl;
        std::cout << "SEARCHING DEPTH: " << depth_limit << std::endl;
        
        t0 = Clock::now();
        
        /* if (depth_limit >= 8){
            int delta = std::max(100, 2000 - 50 * depth_limit);  // depth in plies
            int alpha_window = score - delta;  // prev_score is the score from the previous iteration
            int beta_window = score + delta;

            score = alpha_beta(alpha_window, beta_window, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
            
            
            if (score <= alpha_window || score >= beta_window) {
                // score is not usable, you need to re-search with wider bounds
                std::cout << "ASPIRATION WINDOW FAILED SCORE " << score << " ALPHA_WINDOW: " << alpha_window << " AND BETA_WINDOW: " << beta_window << " RE-SEARCHING DEPTH: " << depth_limit << std::endl;
                score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
            }
        }else{
            score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
        }  */
        
        if (depth_limit >= 13 && (depth_limit - 3) % 5 == 0)
            decayMoveFrequency();
        score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, search_start_time, preliminary_search_data, move, num_iterations);

        bool side = current.turn;
        for (int i = 0; i < pv_length[0]; i++) {
            
            Move move = pv_table[0][i];

            int bonus = (((depth_limit - 1) * (depth_limit - 1)) * (pv_length[0] - i) * (pv_length[0] - i));
            moveFrequency[side][move.from_square][move.to_square] += bonus;

            if(depth_limit >= 9){
                std::cout 
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") | ";
            }            
            side = !side;

        }
        if(depth_limit >= 9){
            std::cout <<std::endl;
        }    

        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "ELAPSED: " << elapsed << std::endl;
    }
         
    int x1 = (move.from_square & 7) + 1;
    int y1 = (move.from_square >> 3) + 1;

    int x2 = (move.to_square & 7) + 1;
    int y2 = (move.to_square >> 3) + 1;

    std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
    std::cout << std::endl;

    MoveData chosenMove(x1,y1,x2,y2, move.promotion, score, num_iterations);

    BoardState current_state = state_history.back();
    
    bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
    updateZobristHashForMove(
        zobrist,
        move.from_square,
        move.to_square,
        capture_move,
        current_state.pawns,
        current_state.knights,
        current_state.bishops,
        current_state.rooks,
        current_state.queens,
        current_state.kings,
        current_state.occupied_colour[true],
        current_state.occupied_colour[false],
        move.promotion
    );

    make_move(state_history, position_count, move, zobrist, capture_move);
    
    std::cout << "EVAL CACHE VISITS: "<< eval_visits << std::endl;
    std::cout << "EVAL CACHE HITS: "<< eval_cache_hits << std::endl;

    std::cout << "MOVE GEN CACHE VISITS: "<< move_gen_visits << std::endl;
    std::cout << "MOVE GEN CACHE HITS: "<< move_gen_cache_hits << std::endl;

    std::cout << "TT VISITS: "<< tt_visits << std::endl;
    std::cout << "TT PROBES: "<< tt_probes << std::endl;
    std::cout << "TT HITS: "<< tt_hits << std::endl;

    std::cout << "Q SEARCH VISITS: "<< qsearchVisits << std::endl;
    
    return chosenMove;

}

int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, SearchData& previous_search_data, Move& best_move, int& num_iterations){
    
    int best_score = -99999999;
    int score;  

    BoardState current_state = state_history.back();

    uint64_t cur_hash = zobrist;
    
    std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move{});
    std::fill(pv_length, pv_length + MAX_PLY, 0);

    SearchData current_search_data = reorder_legal_moves(alpha,beta, depth_limit, t0, zobrist, previous_search_data, state_history, position_count, num_iterations);
    
    std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move{});
    std::fill(pv_length, pv_length + MAX_PLY, 0);

    if (time_up.load(std::memory_order_relaxed)){
        std::cout << "TIME LIMIT EXCEEDED" << std::endl;
        best_move = current_search_data.moves_list[0];
        best_score = current_search_data.top_level_preliminary_scores[0];
        return best_score;
    }
    int razor_threshold;
    if (previous_search_data.moves_list.empty()) {
        razor_threshold = std::max(static_cast<int>(750 * std::pow(0.75, depth_limit - 4)), 200);
    } else {
        razor_threshold = std::max(static_cast<int>(300 * std::pow(0.75, depth_limit - 4)), 100);
    }
    //std::cout << "BBB" << std::endl;

    previous_search_data.moves_list.clear();
    previous_search_data.top_level_preliminary_scores.clear();
    previous_search_data.second_level_moves_list.clear();
    previous_search_data.second_level_preliminary_scores.clear();

    // Define the number of moves, the best move index and the current index
    int num_legal_moves = static_cast<int>(current_search_data.moves_list.size());
    int best_move_index = -1;

    /* // Define the depth that should be used
    int depth_usage = 0;

    // Define variables to hold information on repeating moves
    bool repetition_flag = false;
    Move repetition_move;
    int repetition_score = 0;
    int repetition_index = 0; */

    if (depth_limit >= 10) {
        std::cout << "Num Moves: " << num_legal_moves << std::endl;
    }

    /* if (depth_limit >= 24) {
        std::cout << "AAAA: " << num_legal_moves << std::endl;
    } */
    
    bool en_passant_move = is_en_passant(current_search_data.moves_list[0].from_square, current_search_data.moves_list[0].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(current_search_data.moves_list[0].from_square, current_search_data.moves_list[0].to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
    updateZobristHashForMove(
        zobrist,
        current_search_data.moves_list[0].from_square,
        current_search_data.moves_list[0].to_square,
        capture_move,
        current_state.pawns,
        current_state.knights,
        current_state.bishops,
        current_state.rooks,
        current_state.queens,
        current_state.kings,
        current_state.occupied_colour[true],
        current_state.occupied_colour[false],
        current_search_data.moves_list[0].promotion
    );

    //std::cout <<"BBB" << std::endl;
    make_move(state_history, position_count, current_search_data.moves_list[0], zobrist, capture_move);
    /* if (depth_limit >= 24) {
        std::cout << "BBB: " << num_legal_moves << std::endl;
    } */
    //std::cout <<"CCC"<< std::endl;
    score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, current_search_data.second_level_preliminary_scores[0], current_search_data.second_level_moves_list[0], previous_search_data, state_history, position_count, zobrist, current_search_data.moves_list[0], num_iterations, capture_move, false, false);
    
    //std::cout <<"DDDD"<< std::endl;
    /* if(is_repetition(position_count, zobrist, 2)){
        repetition_flag = true;
        repetition_move = current_search_data.moves_list[0];
        repetition_score = score;
        repetition_index = 0;
        score = -100000000;
    } */
    /* if (depth_limit >= 24) {
        std::cout << "CCCC: " << num_legal_moves << std::endl;
    } */
    BoardState updated_state = state_history.back();
    //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);  
    addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit, TTFlag::EXACT, alpha, beta/* , line */, updated_state.castling_rights, updated_state.ep_square);
    unmake_move(state_history, position_count, zobrist);
    //std::cout <<"CC" << std::endl;
    /* if (depth_limit >= 24) {
        std::cout << "DDDD: " << num_legal_moves << std::endl;
    } */
    zobrist = cur_hash;

    if (depth_limit >= 10){
        std::cout << 0 << " "
          << score << " "
          << current_search_data.top_level_preliminary_scores[0] << " "
          << "(" << ((current_search_data.moves_list[0].from_square & 7) + 1) << ","
          << ((current_search_data.moves_list[0].from_square >> 3) + 1) << ") -> ("
          << ((current_search_data.moves_list[0].to_square & 7) + 1) << ","
          << ((current_search_data.moves_list[0].to_square >> 3) + 1) << ")"
          << std::endl;
    }

    best_move = current_search_data.moves_list[0];
    best_score = score;
    best_move_index = 0;

    
    if (score > alpha)
        updatePV(current_search_data.moves_list[0], cur_depth);

    alpha = std::max(alpha, best_score);

    if (time_up.load(std::memory_order_relaxed)){
        std::cout << "TIME LIMIT EXCEEDED" << std::endl;
        best_move = current_search_data.moves_list[0];
        best_score = current_search_data.top_level_preliminary_scores[0];
        return best_score;
    }    

    if(alpha - current_search_data.top_level_preliminary_scores[0] > razor_threshold)
        razor_threshold += alpha - current_search_data.top_level_preliminary_scores[0];


    previous_search_data.moves_list = current_search_data.moves_list;
    previous_search_data.top_level_preliminary_scores.push_back(score);

    if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        return score;

    for (size_t i = 1; i < current_search_data.moves_list.size(); ++i) {
        Move& move = current_search_data.moves_list[i];

        // Razoring
        if (i < current_search_data.top_level_preliminary_scores.size()) {
            int score_diff = alpha - current_search_data.top_level_preliminary_scores[i];
            //int best_diff = current_search_data.top_level_preliminary_scores[0] - current_search_data.top_level_preliminary_scores[i];

            if ((score_diff > razor_threshold) && (alpha < 9000000)) {
                break;
            }
        }

        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
        
        updateZobristHashForMove(
            zobrist,
            move.from_square,
            move.to_square,
            capture_move,
            current_state.pawns,
            current_state.knights,
            current_state.bishops,
            current_state.rooks,
            current_state.queens,
            current_state.kings,
            current_state.occupied_colour[true],
            current_state.occupied_colour[false],
            move.promotion
        );

        make_move(state_history, position_count, move, zobrist, capture_move);
        //std::cout <<"EEE" << std::endl;
        score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], previous_search_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
        
        
        //std::cout <<"FFF" << std::endl;
        // If the score is within the window, re-search with full window
        if (alpha < score && score < beta){
            previous_search_data.second_level_preliminary_scores.pop_back();
            previous_search_data.second_level_moves_list.pop_back();
            score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], previous_search_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
        }

        /* if(is_repetition(position_count, zobrist, 2)){
            repetition_flag = true;
            repetition_move = move;
            repetition_score = score;
            repetition_index = i;
            score = -100000001;
        } */
        updated_state = state_history.back();
        //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);  
        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit, TTFlag::EXACT, alpha, beta/* , line */, updated_state.castling_rights, updated_state.ep_square);
        unmake_move(state_history, position_count, zobrist);

        if (time_up.load(std::memory_order_relaxed)){
            std::cout << "TIME LIMIT EXCEEDED" << std::endl;

            /* if (alpha < current_search_data.top_level_preliminary_scores[0]){
                best_move = current_search_data.moves_list[0];
                best_score = current_search_data.top_level_preliminary_scores[0];                
            } */
            return best_score;            
        }

        zobrist = cur_hash;
        previous_search_data.top_level_preliminary_scores.push_back(score);
        
        if (depth_limit >= 10){
            std::cout << i << " "
            << score << " "
            << current_search_data.top_level_preliminary_scores[i] << " "
            << "(" << ((move.from_square & 7) + 1) << ","
            << ((move.from_square >> 3) + 1) << ") -> ("
            << ((move.to_square & 7) + 1) << ","
            << ((move.to_square >> 3) + 1) << ")"
            << std::endl;
        }

        // Check if the current move's score is better than the existing best move
        if (score > best_score){            
            best_move = move;
            best_score = score;
            best_move_index = i;

            if (score > alpha)
                updatePV(move, cur_depth); 

        }
            
        alpha = std::max(alpha, best_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            if (depth_limit >= 10) {
                std::cout << std::endl;
                std::cout << "Best: " << best_move_index << std::endl;
            }
            
            // Fail-soft: return the actual score that exceeded beta
            return best_score;

            if(!capture_move){
                storeKillerMove(cur_depth, move);
                historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);                
            }
        }
        
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT){
            if (depth_limit >= 10) {
                std::cout << std::endl;
                std::cout << "TIME LIMIT EXCEEDED" << std::endl;
                std::cout << "Best: " << best_move_index << std::endl;
            }
            return best_score;
        }
    }

    if (cur_depth == 0){
        /* if (repetition_flag) {
            if (alpha < repetition_score) {
                if (alpha <= -500) {
                    best_move = repetition_move;
                    previous_search_data.top_level_preliminary_scores[repetition_index] = repetition_score;
                    best_score = 0;
                }
            }
        } */

        if (depth_limit >= 10) {
            std::cout << std::endl;
            std::cout << "Best: " << best_move_index << std::endl;
        }

        return best_score;
    }
    return best_score;
}

inline int get_score_for_minimizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove,
                                   std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, std::vector<BoardState>& state_history, BoardState current_state,
                                   bool& using_fp, int &num_iterations, bool is_in_null_search, bool& is_exact_hit)
{
    bool using_tt = false;
    int score = 0;
    BoardState updated_state = state_history.back();            
    if(is_repetition(position_count, zobrist, 3) || updated_state.halfmove_clock >= 100){
        score = 0;
    }else{                
        
        TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
        tt_visits++;
        if (entry != nullptr) {                
            //TTEntry entry = entry_opt.value();    
            tt_probes++;                    
            if (entry->depth >= (depth_limit - cur_depth) /* && (depth_limit - cur_depth) >= 5 */) {    
                use_tt_entry(*entry, score, using_tt, alpha, beta, num_iterations, false, true);                                                 
                /* if (using_tt && entry.flag == TTFlag::EXACT && entry.pv_length > 0 && score > alpha && !is_in_null_search) {
                    for (int j = 0; j < entry.pv_length; ++j) {
                        pv_table[cur_depth + 1][j] = entry.pv[j];
                    }
                    pv_length[cur_depth + 1] = entry.pv_length;
                }                              */
            }                
        }

        if(!using_tt){
            if (i == 0 || (depth_limit - cur_depth) == 1) {
                // Full window search for first move                                           
                if (!using_tt){
                    score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    if(cur_depth < depth_limit - 1){
                        TTFlag flag;
                        if (score <= alpha_orig) {
                            flag = TTFlag::UPPERBOUND;
                        } else if (score >= beta_orig) {
                            flag = TTFlag::LOWERBOUND;
                        } else {
                            flag = TTFlag::EXACT;
                        }
                        //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);  
                        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                    }
                } 
            } else {
                // LMR flag can still be computed here as you do
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);                
                bool do_lmr = (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */);
                                    
                // Null window search with LMR applied inside
                if (do_lmr) {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if (cur_depth > 1 && (depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin){
                        int early_score = get_board_evaluation(state_history, zobrist, num_iterations);
                        //int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);

                        if (early_score - FUTILITY_MARGINS[depth_limit - cur_depth - 1] > beta){
                            using_fp = true;
                            return early_score;
                        }
                    }
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin,  i, current_state);
                    TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);                    
                    if (entry != nullptr) {                
                        //TTEntry entry = entry_opt.value();   
                        tt_probes++;                         
                        if (entry->depth >= (reduced_depth - cur_depth)) {                            
                            /* if (entry.flag == TTFlag::EXACT){                                                             
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */    
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);            
                        }                
                    }
                    if (!using_tt){
                        score = maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < reduced_depth - 1){
                            TTFlag flag;
                            if (score <= alpha) {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            } else if (score >= alpha + 1) {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }else{
                        tt_hits++;
                    }
                } else {

                    TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);                    
                    if (entry != nullptr) {                
                        //TTEntry entry = entry_opt.value();
                        tt_probes++;                            
                        if (entry->depth >= (depth_limit - cur_depth)) {                            
                            /* if (entry.flag == TTFlag::EXACT){                                                             
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */  
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }                
                    }
                    if (!using_tt){
                        score = maximizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < depth_limit - 1){
                            TTFlag flag;
                            if (score <= alpha) {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            } else if (score >= alpha + 1) {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }else{
                        tt_hits++;
                    }                      
                }

                // If score is promising, re-search full window without reduction
                if (score > alpha && score < beta) {
                    using_tt = false;                        

                    if (!using_tt){
                        score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < depth_limit - 1){
                            TTFlag flag;
                            if (score <= alpha_orig) {
                                flag = TTFlag::UPPERBOUND;
                            } else if (score >= beta_orig) {
                                flag = TTFlag::LOWERBOUND;
                            } else {
                                flag = TTFlag::EXACT;
                            }     
                            //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);                             
                            addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }                        
                }
            }
        }else{
            tt_hits++;
        }                                
    }
    return score;
}

inline int get_score_for_maximizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove,
                                   std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, std::vector<BoardState>& state_history, BoardState current_state,
                                   bool& using_fp, int &num_iterations, bool is_in_null_search, bool& is_exact_hit)
{
    bool using_tt = false;
    int score = 0;
    std::vector<int> dummy_ints;
    std::vector<Move> dummy_moves;
    SearchData dummy_data;
    BoardState updated_state = state_history.back();            
    if(is_repetition(position_count, zobrist, 3) || updated_state.halfmove_clock >= 100){
        score = 0;
    }else{                
        
        TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
        tt_visits++;
        if (entry != nullptr) {                
            //TTEntry entry = entry_opt.value();
            tt_probes++;                        
            if (entry->depth >= (depth_limit - cur_depth) /* && (depth_limit - cur_depth) >= 5 */) {    
                use_tt_entry(*entry, score, using_tt, alpha, beta, num_iterations, true, true);    
                /* if (using_tt && entry.flag == TTFlag::EXACT && entry.pv_length > 0 && score > alpha && !is_in_null_search) {
                    for (int j = 0; j < entry.pv_length; ++j) {
                        pv_table[cur_depth + 1][j] = entry.pv[j];
                    }
                    pv_length[cur_depth + 1] = entry.pv_length;
                } */
            }                
        }

        if(!using_tt){
            if (i == 0) {
                // Full window search for first move                                           
                if (!using_tt){                    
                    score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    if(cur_depth < depth_limit - 1){
                        TTFlag flag;
                        if (score <= alpha_orig) {
                            flag = TTFlag::UPPERBOUND;
                        } else if (score >= beta_orig) {
                            flag = TTFlag::LOWERBOUND;
                        } else {
                            flag = TTFlag::EXACT;
                        }
                        //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]); 
                        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                    }
                } 
            } else {
                // LMR flag can still be computed here as you do
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);                
                bool do_lmr = (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */);
                                    
                // Null window search with LMR applied inside
                if (do_lmr) {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if ((depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin){
                        int early_score = get_board_evaluation(state_history, zobrist, num_iterations);
                        //int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                        if (early_score + FUTILITY_MARGINS[depth_limit - cur_depth - 1] < alpha){
                            using_fp = true;
                            return early_score;
                        }
                    }                    
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin,  i, current_state);

                    TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr) {                
                        //TTEntry entry = entry_opt.value();    
                        tt_probes++;                        
                        if (entry->depth >= (reduced_depth - cur_depth)) {                            
                            /* if (entry.flag == TTFlag::EXACT){                                                             
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */    
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);            
                        }                
                    }
                    if (!using_tt){
                        
                        score = minimizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < reduced_depth - 1){
                            TTFlag flag;
                            if (score <= alpha) {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            } else if (score >= alpha + 1) {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }else{
                        tt_hits++;
                    }
                } else {

                    TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr) {                
                        //TTEntry entry = entry_opt.value();
                        tt_probes++;                            
                        if (entry->depth >= (depth_limit - cur_depth)) {                            
                            /* if (entry.flag == TTFlag::EXACT){                                                             
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */  
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }                
                    }
                    if (!using_tt){                        
                        score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < depth_limit - 1){
                            TTFlag flag;
                            if (score <= alpha) {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            } else if (score >= alpha + 1) {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }else{
                        tt_hits++;
                    }                        
                }

                // If score is promising, re-search full window without reduction
                if (score > alpha && score < beta) {
                    using_tt = false;                        

                    if (!using_tt){                        
                        score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if(cur_depth < depth_limit - 1){
                            TTFlag flag;
                            if (score <= alpha_orig) {
                                flag = TTFlag::UPPERBOUND;
                            } else if (score >= beta_orig) {
                                flag = TTFlag::LOWERBOUND;
                            } else {
                                flag = TTFlag::EXACT;
                            }
                            //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);                                
                            addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }                        
                }
            }
        }else{
            tt_hits++;
        }                
        
        /* if (!using_tt){
            if(cur_depth < depth_limit - 1){
                TTFlag flag;
                if (score <= alpha_orig) {
                    flag = TTFlag::UPPERBOUND;
                } else if (score >= beta_orig) {
                    flag = TTFlag::LOWERBOUND;
                } else {
                    flag = TTFlag::EXACT;
                }                    
                addToSearchEvalCache(zobrist, state_history.size(), TTEntry(score, depth_limit - cur_depth, flag, alpha_orig, beta_orig), updated_state.castling_rights, updated_state.ep_square);
            }
        }   */
    }
    return score;
}

int minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<int>second_level_preliminary_scores, std::vector<Move>second_level_moves_list,
              SearchData& previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move previousMove,
              int& num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
{    

    if (time_up.load(std::memory_order_relaxed)) {
        return 0;
    } else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL) {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT) {
            time_up.store(true, std::memory_order_relaxed);
        }
    }
    //pv_length[cur_depth] = 0;
    BoardState current_state = state_history.back();
    
    /* if (depth_limit >= 24) {
        std::cout << "EE: " << std::endl;
    } */

    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    
    if (cur_depth >= depth_limit){        

        if (USE_Q_SEARCH /* && depth_limit >= 6 */){
            if(use_q_precautions.load(std::memory_order_relaxed)){
                if(depth_limit >= 6){                    
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);
                    return result;
                }
            } else{
                int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);
                return result;
            }

            /* int result = qSearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, previousMove, num_iterations, false);

            int num_plies = state_history.size();
            int max_cache_size;
            // Code segment to control cache size
            if(num_plies < 30){
                max_cache_size = 2000000;         
            }else if(num_plies < 50){
                max_cache_size = 4000000;
            }else if(num_plies < 75){
                max_cache_size = 8000000; 
            }else{
                max_cache_size = 16000000; 
            }

            TTFlag flag;
            if (result <= alpha)
                flag = TTFlag::UPPERBOUND;                
            else if (result >= beta)
                flag = TTFlag::LOWERBOUND;                
            else
                flag = TTFlag::EXACT;                

            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, QCacheEntry(result,flag), current_state.castling_rights, current_state.ep_square);
            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, result, current_state.castling_rights, current_state.ep_square);
            if(current_state.occupied == 3539934878248206336 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "EVAL- min: " << " "
                << result << " "                
                << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit
                << std::endl;

                std::cout << "EVAL PREV IN MIN: " << "(" << ((previousMove.from_square & 7) + 1) << ","
                << ((previousMove.from_square >> 3) + 1) << ") -> ("
                << ((previousMove.to_square & 7) + 1) << ","
                << ((previousMove.to_square >> 3) + 1) << ") " << depth_limit
                << std::endl;
            //}
            } */
            //return result;
        }
        int result = get_board_evaluation(state_history, zobrist, num_iterations);
        if(current_state.occupied == 3539934878248206336 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "EVAL- min: " << " "
                << result << " "                
                << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit
                << std::endl;

                std::cout << "EVAL PREV IN MIN: " << "(" << ((previousMove.from_square & 7) + 1) << ","
                << ((previousMove.from_square >> 3) + 1) << ") -> ("
                << ((previousMove.to_square & 7) + 1) << ","
                << ((previousMove.to_square >> 3) + 1) << ") " << depth_limit
                << std::endl;
            //}
            }
        return result;
    }

    increment_node_count_with_decay(num_iterations);
        
    uint64_t cur_hash = zobrist;

    int lowest_score = 9999999 - static_cast<int>(state_history.size()); 
    int score = lowest_score;    
    int beta_orig = beta;
    int alpha_orig = alpha;

    Move best_move;

    bool all_moves_pruned = true;
    Move best_futility_move;
    bool using_tt = false;
    bool using_fp = false;
    bool is_exact_hit = false;
    int best_early_eval = 9999999 - static_cast<int>(state_history.size());

    /*
    int razor_threshold;

    if (depth_limit  == 3) {
        razor_threshold = std::max(static_cast<int>(2000 * std::pow(0.75, depth_limit - 4)), 200);
    } else {
        razor_threshold = std::max(static_cast<int>(1500 * std::pow(0.75, depth_limit - 4)), 50);
    }
    */
   /* if (create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) == "8/8/3R3P/4P1P1/5PK1/8/2k5/2q1b3 b - - 0 1"){
                                    std::cout << "BBB" << std::endl;

    } */
    
    if (cur_depth == 1){
        if(is_repetition(position_count, zobrist, 3) || current_state.halfmove_clock >= 100){            
            is_draw = true;
            return 0;
        }
        std::vector<int>cur_second_level_preliminary_scores;
        cur_second_level_preliminary_scores.reserve(64);

        ascending_sort(second_level_preliminary_scores, second_level_moves_list);
        previous_search_data.second_level_moves_list.push_back(second_level_moves_list);
        bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        for (size_t i = 0; i < second_level_moves_list.size(); ++i) {
            Move& move = second_level_moves_list[i];
            using_tt = false;
            using_fp = false;
            is_exact_hit = false;
            /*
            // Razoring
            if (i < second_level_preliminary_scores.size()) {
                if ((second_level_preliminary_scores[i] - beta > razor_threshold)) {
                    continue;
                }
            }
            */
                        
            bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

            // Acquire the zobrist hash for the new position if the given move was made
            bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

            updateZobristHashForMove(
                zobrist,
                move.from_square,
                move.to_square,
                capture_move,
                current_state.pawns,
                current_state.knights,
                current_state.bishops,
                current_state.rooks,
                current_state.queens,
                current_state.kings,
                current_state.occupied_colour[true],
                current_state.occupied_colour[false],
                move.promotion
            );

            
            make_move(state_history, position_count, move, zobrist, capture_move);        
            score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                    position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);             
            
            unmake_move(state_history, position_count, zobrist);
            if (current_state.occupied == 10658118197365045909 && current_state.queens == 144119586122366976){
            //if (depth_limit >= 11){
                std::cout << "BBBB- min 0: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit
                << std::endl;
            //}
            }
            if (time_up.load(std::memory_order_relaxed))
                return 0;

            zobrist = cur_hash;
            cur_second_level_preliminary_scores.push_back(score);
     
            if (score < lowest_score){
                lowest_score = score;
                best_move = move;

                // Beta improved — update PV
                if (score > alpha && !is_exact_hit)
                    updatePV(move, cur_depth);

            }

            beta = std::min(beta,lowest_score);

            // Check for a beta cutoff 
            if (beta <= alpha){
                //std::cout <<score << std::endl;
                previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);                

                if(!capture_move){
                    storeKillerMove(cur_depth, move);
                    historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
                }
                return lowest_score;
            }
        }

        // Check if no moves are available, inidicating a game ending move was made previously
        if(lowest_score == 9999999 - static_cast<int>(state_history.size())){
        
            previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);
            
            if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 9999999 - static_cast<int>(state_history.size());
            }else if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }else if (all_moves_pruned){
                /* if (score > alpha)
                    updatePV(best_futility_move, cur_depth); */
                return best_early_eval;
            }
        }
        previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);

    } else{
        bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);            
        // Null Move Pruning
        if (cur_depth >= 3 && depth_limit >= 5 && !last_move_was_capture &&!last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state)){
            state_history.back().turn = !state_history.back().turn;

            int cur_ep_square = state_history.back().ep_square;
            state_history.back().ep_square = -1;
            updateZobristHashForNullMove(zobrist);

            int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

            if(depth_limit >= 10){
                reduced_depth -= 1;
            }else if (depth_limit >= 12){
                reduced_depth -= 2;
            }else if (depth_limit >= 14){
                reduced_depth -= 3;
            }
            
            /* if ((reduced_depth <= cur_depth + 1) && relevant_pin_exists(state_history, false)){
                reduced_depth = cur_depth + 2;
            }
            reduced_depth = std::min(depth_limit,reduced_depth); */
            //reduced_depth = std::max(2,reduced_depth);
            TTEntry* entry = accessSearchEvalCache(zobrist, state_history.back().castling_rights, state_history.back().ep_square);
              
            int null_move_score;
            if (entry != nullptr) {                
                //TTEntry entry = entry_opt.value();
                if (entry->depth >= (reduced_depth - cur_depth)) {
                    if (entry->flag == TTFlag::EXACT){                                                             
                        null_move_score = entry->score;
                        using_tt = true;
                        increment_node_count_with_decay(num_iterations);
                    }         
                    //use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, true);    
                }                
            }
            if (!using_tt){
                Move dummyMove;
                null_move_score = maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, state_history, position_count, zobrist, dummyMove, num_iterations, false, true, true);
            }

            /* if(current_state.occupied == 3539934878248206336 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "NULL MOVE- min: " << " "
                << null_move_score << " "                
                << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit
                << std::endl;

                std::cout << "NULL PREV IN MIN: " << "(" << ((previousMove.from_square & 7) + 1) << ","
                << ((previousMove.from_square >> 3) + 1) << ") -> ("
                << ((previousMove.to_square & 7) + 1) << ","
                << ((previousMove.to_square >> 3) + 1) << ") " << depth_limit
                << std::endl;
            //}
            } */
            
            state_history.back().turn = !state_history.back().turn;    
            state_history.back().ep_square = cur_ep_square;    
            zobrist = cur_hash;
            
            if (null_move_score <= alpha) {
                return null_move_score; // fail-high cutoff
            }
        }

        std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
        
        for (size_t i = 0; i < moves_list.size(); ++i) {
            Move& move = moves_list[i];
            using_tt = false;
            using_fp = false;
            is_exact_hit = false;
            /* if(current_state.occupied == 10497113652010869597 && current_state.queens == 1125899906842632){
            //if (depth_limit >= 11){
                std::cout << "AAAA- min: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << beta << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit
                << std::endl;
            //}
            }   */          
            bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

            // Acquire the zobrist hash for the new position if the given move was made
            bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

            // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
            updateZobristHashForMove(
                zobrist,
                move.from_square,
                move.to_square,
                capture_move,
                current_state.pawns,
                current_state.knights,
                current_state.bishops,
                current_state.rooks,
                current_state.queens,
                current_state.kings,
                current_state.occupied_colour[true],
                current_state.occupied_colour[false],
                move.promotion
            );
            
            make_move(state_history, position_count, move, zobrist, capture_move);        
            score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                    position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);

             if(current_state.occupied == 7199354783056128661 && current_state.queens == 144119586122366976){
            //if (depth_limit >= 11){
                std::cout << "BBBB- min 2: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;

                /* std::cout << "PREV IN MIN: " << "(" << ((previousMove.from_square & 7) + 1) << ","
                << ((previousMove.from_square >> 3) + 1) << ") -> ("
                << ((previousMove.to_square & 7) + 1) << ","
                << ((previousMove.to_square >> 3) + 1) << ") " << depth_limit
                << std::endl; */
            //}
            }
            /*
            if(current_state.occupied == 3539934878248206336 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "BBBB- min 2: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;

            //}
            }
            if(current_state.occupied == 3539934946665824256 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "AAAA- min 4: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;

            //}
            } */
            if(using_fp){
                best_early_eval = std::min(best_early_eval, score);
                best_futility_move = move;
                unmake_move(state_history, position_count, zobrist);
                zobrist = cur_hash;
                continue;
            }else{
                all_moves_pruned = false;
            }       
                        
            unmake_move(state_history, position_count, zobrist);
            


            
            if (time_up.load(std::memory_order_relaxed))
                return 0;
            zobrist = cur_hash;
            
            //lowest_score = std::min(score,lowest_score);
            if (score < lowest_score){

                lowest_score = score;
                best_move = move;

                // Beta improved — update PV
                /* if (score > alpha && !is_in_null_search)
                    updatePV(move, cur_depth); */

                if (score > alpha && !is_in_null_search /* && !is_exact_hit */){
                    updatePV(move, cur_depth);                    
                }
            }
            beta = std::min(beta,lowest_score);

            // Check for a beta cutoff 
            if (beta <= alpha){
                if (i != 0)
                    updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);
                
                if(!capture_move){
                    storeKillerMove(cur_depth, move);
                    historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
                }

                return lowest_score;
            }
        }       

        // Check if no moves are available, inidicating a game ending move was made previously
        if(lowest_score == 9999999 - static_cast<int>(state_history.size())){            
            /* if (current_state.occupied == 11089329065645372309){
                    std::cout << "MINIMIZER FROM: " << " SCORE: " << score << " LOWEST SCORE: " << lowest_score << std::endl; 
                } */
            if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                                
                return 9999999 - static_cast<int>(state_history.size());
            }else if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }else if (all_moves_pruned){                                
                /* if (score > alpha)
                    updatePV(best_futility_move, cur_depth); */
                return best_early_eval;
            }
            /* if (current_state.occupied == 197633){
                    std::cout << "MINIMIZER FROM: " << " SCORE: " << score << " LOWEST SCORE: " << lowest_score << std::endl; 
                } */
        }

    }
    bool en_passant_move = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
    bool capture_move = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    if(!capture_move){
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }

    return lowest_score;    
}

int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count,
              uint64_t zobrist, Move previousMove, int& num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
{
    
    BoardState current_state = state_history.back();
    

    if (time_up.load(std::memory_order_relaxed)) {
        return 0;
    } else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL) {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT) {
            time_up.store(true, std::memory_order_relaxed);
        }
    }

    //pv_length[cur_depth] = 0;
    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    if (cur_depth >= depth_limit){

        if (USE_Q_SEARCH /* && depth_limit >= 6 */){
            if(use_q_precautions.load(std::memory_order_relaxed)){
                if(depth_limit >= 6){                    
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                    return result;
                }
            } else{
                int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                return result;
            }
            /* if(current_state.occupied == 7199916633497922197 && current_state.queens == 144119586122366976){
                //if (depth_limit >= 11){
                    std::cout << "NULL- MAX 3: "<< " "
                    << result << std::endl;

                    
                //}
            } */
            /* int result = qSearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, previousMove, num_iterations, true);

            int num_plies = state_history.size();
            int max_cache_size;
            // Code segment to control cache size
            if(num_plies < 30){
                max_cache_size = 2000000;         
            }else if(num_plies < 50){
                max_cache_size = 4000000;
            }else if(num_plies < 75){
                max_cache_size = 8000000; 
            }else{
                max_cache_size = 16000000; 
            }

            TTFlag flag;
            if (result <= alpha)
                flag = TTFlag::UPPERBOUND;                
            else if (result >= beta)
                flag = TTFlag::LOWERBOUND;                
            else
                flag = TTFlag::EXACT;                

            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, QCacheEntry(result,flag), current_state.castling_rights, current_state.ep_square);
            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, result, current_state.castling_rights, current_state.ep_square); */

            //return result;
        }

        return get_board_evaluation(state_history, zobrist, num_iterations);
        
    }

    increment_node_count_with_decay(num_iterations);

    uint64_t cur_hash = zobrist;    
    
    int highest_score = -9999999 + static_cast<int>(state_history.size());
    int score = highest_score;  
    int beta_orig = beta;   
    int alpha_orig = alpha;    

    Move best_move;
    
    std::vector<int> dummy_ints;
    std::vector<Move> dummy_moves;
    SearchData dummy_data;

    bool all_moves_pruned = true;
    Move best_futility_move;
    bool using_tt = false;
    bool using_fp = false;
    bool is_exact_hit = false;
    int best_early_eval = -9999999 + static_cast<int>(state_history.size());;
    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
    
    // Null Move Pruning
    if (cur_depth >= 4 && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state)){
        state_history.back().turn = !state_history.back().turn;

        int cur_ep_square = state_history.back().ep_square;
        state_history.back().ep_square = -1;
        updateZobristHashForNullMove(zobrist);

        int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

        if(depth_limit >= 10){
            reduced_depth -= 1;
        }else if (depth_limit >= 12){
            reduced_depth -= 2;
        }else if (depth_limit >= 14){
            reduced_depth -= 3;
        }
        
        /* if ((reduced_depth <= cur_depth + 1) && relevant_pin_exists(state_history, false)){
            reduced_depth = cur_depth + 2;
        }
        reduced_depth = std::min(depth_limit,reduced_depth); */
        //reduced_depth = std::max(2,reduced_depth);
        TTEntry* entry = accessSearchEvalCache(zobrist, state_history.back().castling_rights, state_history.back().ep_square);             
        int null_move_score;
        
        if (entry != nullptr) {                    
            //TTEntry entry = entry_opt.value();
            if (entry->depth >= (reduced_depth - cur_depth)) {
                if (entry->flag == TTFlag::EXACT){                                                             
                    null_move_score = entry->score;
                    using_tt = true;
                    increment_node_count_with_decay(num_iterations);
                }         
                //use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, true);    
            }                
        }
        if (!using_tt){
            Move dummyMove;
            null_move_score = minimizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, dummyMove, num_iterations, false, true, true);
        }
        
        state_history.back().turn = !state_history.back().turn;    
        state_history.back().ep_square = cur_ep_square;    
        zobrist = cur_hash;

        /* if(current_state.occupied == 10515685296490865501 && current_state.queens == 1125899906842632){
            //if (depth_limit >= 11){
                std::cout << "NULL: " << " "
                << null_move_score << " " << reduced_depth               
                << " | " << depth_limit
                << std::endl;
            //}
        } */
        
        if (null_move_score >= beta) {
            return null_move_score; // fail-high cutoff
        }
    }

    std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
    /* if (create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) == "8/8/3R3P/4P1P1/5P2/5K2/2k5/2q1b3 w - - 0 1"){
                                    std::cout << "AAAA" << std::endl;

    } */
    
    for (size_t i = 0; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
        using_tt = false;
        using_fp = false;
        is_exact_hit = false;
        /* if(current_state.occupied == 11233439855674717077 && current_state.queens == 576460752320200704){
            //if (depth_limit >= 11){
                std::cout << "AAAA: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, true) << " | " << depth_limit
                << std::endl;
            //}
        } */
        
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

        // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
        updateZobristHashForMove(
            zobrist,
            move.from_square,
            move.to_square,
            capture_move,
            current_state.pawns,
            current_state.knights,
            current_state.bishops,
            current_state.rooks,
            current_state.queens,
            current_state.kings,
            current_state.occupied_colour[true],
            current_state.occupied_colour[false],
            move.promotion
        );

        
        make_move(state_history, position_count, move, zobrist, capture_move);   
        score = get_score_for_maximizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                    position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);

        if(using_fp){
            best_early_eval = std::max(best_early_eval, score);
            best_futility_move = move;
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;
            continue;
        }else{
            all_moves_pruned = false;
        }
        
          
        /* if(current_state.occupied == 11089329074235302805){
            std::cout << score << " | " << using_tt << " | " << !relevant_pin_exists(state_history) << " | " << std::endl;
        }  */       
        unmake_move(state_history, position_count, zobrist);
       

        if(current_state.occupied == 10658119296876669589 && current_state.queens == 144119586122366976){
            //if (depth_limit >= 11){
                std::cout << "MAX 1: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << depth_limit << " | " << cur_depth
                << std::endl;

                /* std::cout << "PREV IN MAX: " << "(" << ((previousMove.from_square & 7) + 1) << ","
                << ((previousMove.from_square >> 3) + 1) << ") -> ("
                << ((previousMove.to_square & 7) + 1) << ","
                << ((previousMove.to_square >> 3) + 1) << ") " << depth_limit
                << std::endl; */
            //}
        }

         if(current_state.occupied == 7199916633497922197 && current_state.queens == 144119586122366976){
            //if (depth_limit >= 11){
                std::cout << "BBBB- MAX 3: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;

                
            //}
        }
        /*
        if(current_state.occupied == 3611992540703752192 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "BBBB- MAX 5: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;
            //}
        }

        if(current_state.occupied == 3611992540167799296 && current_state.queens == 0){
            //if (depth_limit >= 11){
                std::cout << "BBBB- MAX 7: "<< i << " "
                << score << " "                
                << "(" << ((move.from_square & 7) + 1) << ","
                << ((move.from_square >> 3) + 1) << ") -> ("
                << ((move.to_square & 7) + 1) << ","
                << ((move.to_square >> 3) + 1) << ") " << using_tt << " | " << relevant_pin_exists(state_history, false) << " | " << using_fp << " | " << depth_limit << " | " << cur_depth
                << std::endl;
            //}
        } */
        
        if (time_up.load(std::memory_order_relaxed))
            return 0;
        
        zobrist = cur_hash;
        //highest_score = std::max(score,highest_score);

        if(score > highest_score){

            /* if (current_state.occupied == 16909313){
                std::cout << "MAXIMIZER FROM: " << (int)move.from_square << " TO: " << (int)move.to_square << std::endl; 
            } */
            highest_score = score;
            best_move = move;

            // Alpha improved — update PV
            /* if (score > alpha && !is_in_null_search)
                updatePV(move, cur_depth); */

            if (score > alpha && !is_in_null_search /* && !is_exact_hit */){
                updatePV(move, cur_depth);                
            }
        }

        alpha = std::max(alpha,highest_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            if (i != 0)
                updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);
            
            if(!capture_move){
                storeKillerMove(cur_depth, move);
                historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
                counterMoves[previousMove.from_square][previousMove.to_square] = move;
                counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
            }
            
            return highest_score;
        }
    }

    /* if (all_moves_pruned){
        return best_early_eval;
    } */

    // Check if no moves are available, inidicating a game ending move was made previously
    if(highest_score == -9999999 + static_cast<int>(state_history.size())){

        if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return highest_score;
        }else if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return 0;
        }else if (all_moves_pruned){
            /* if (score > alpha)
                updatePV(best_futility_move, cur_depth); */

            return best_early_eval;
        }


    }

    bool en_passant_move = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
    bool capture_move = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    if(!capture_move){
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }
    
    return highest_score;
}

SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint& t0, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, int& num_iterations){

    increment_node_count_with_decay(num_iterations);

    BoardState current_state = state_history.back();

    SearchData returnData;
    SearchData current_search_data;

    int score = -99999999;
    int highest_score = -99999999;
    int depth = depth_limit - 1;
    //zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);
    uint64_t cur_hash = zobrist;

    std::vector<Move> moves_list;

    Move dummyMove;

    if (previous_search_data.moves_list.empty()){
        moves_list = buildMoveListFromReordered(state_history, zobrist, 0, dummyMove);
    }else{
        moves_list = previous_search_data.moves_list;
    }
    //std::cout <<"BBB2" << std::endl;
       
    bool en_passant_move = is_en_passant(moves_list[0].from_square, moves_list[0].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(moves_list[0].from_square, moves_list[0].to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    updateZobristHashForMove(
        zobrist,
        moves_list[0].from_square,
        moves_list[0].to_square,
        capture_move,
        current_state.pawns,
        current_state.knights,
        current_state.bishops,
        current_state.rooks,
        current_state.queens,
        current_state.kings,
        current_state.occupied_colour[true],
        current_state.occupied_colour[false],
        moves_list[0].promotion
    );

    make_move(state_history, position_count, moves_list[0], zobrist, capture_move);

    std::vector<int> preliminary_scores;
    std::vector<Move> preliminary_moves;
    highest_score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, moves_list[0], num_iterations);
    //std::cout <<"BBB3" << std::endl;
    BoardState updated_state = state_history.back();
    //std::vector<Move> line(pv_table[1], pv_table[1] + pv_length[1]);  
    addToSearchEvalCache(zobrist, state_history.size(), highest_score, depth_limit, TTFlag::EXACT, alpha, beta/* , line */, updated_state.castling_rights, updated_state.ep_square);
    unmake_move(state_history, position_count, zobrist);

    if (time_up.load(std::memory_order_relaxed)){
        return previous_search_data;
    }
    zobrist = cur_hash;

    if (score > alpha)
        updatePV(moves_list[0], 0);

    alpha = std::max(alpha, highest_score);

    current_search_data.top_level_preliminary_scores.push_back(highest_score);
    current_search_data.second_level_preliminary_scores.push_back(preliminary_scores);
    current_search_data.second_level_moves_list.push_back(preliminary_moves);

    for (size_t i = 1; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
        //std::cout <<"BBB4-0" << std::endl;        
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
        //std::cout <<"BBB4-1" << std::endl;
        // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
        updateZobristHashForMove(
            zobrist,
            move.from_square,
            move.to_square,
            capture_move,
            current_state.pawns,
            current_state.knights,
            current_state.bishops,
            current_state.rooks,
            current_state.queens,
            current_state.kings,
            current_state.occupied_colour[true],
            current_state.occupied_colour[false],
            move.promotion
        );
        //std::cout <<"BBB4-2" << std::endl;

        make_move(state_history, position_count, move, zobrist, capture_move);
        //std::cout <<"BBB4-3" << std::endl;
        std::vector<int> preliminary_scores;
        std::vector<Move> preliminary_moves;
        
        //zobrist = generateZobristHash(new_state.pawns, new_state.knights, new_state.bishops, new_state.rooks, new_state.queens, new_state.kings, new_state.occupied_colour[true], new_state.occupied_colour[false], new_state.turn);
        score = pre_minimizer(1, depth, alpha, alpha + 1, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
        //std::cout <<"BBB4" << std::endl;
        // If the score is within the window, re-search with full window
        if (alpha < score && score < beta){

            preliminary_scores.clear();
            preliminary_moves.clear();
            score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
        }
        //std::cout <<"BBB5-0" << std::endl;
        current_search_data.top_level_preliminary_scores.push_back(score);
        current_search_data.second_level_preliminary_scores.push_back(preliminary_scores);
        current_search_data.second_level_moves_list.push_back(preliminary_moves);
        //std::cout <<"BBB5-1" << std::endl;
        updated_state = state_history.back();
        //std::vector<Move> line(pv_table[1], pv_table[1] + pv_length[1]);  
        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit, TTFlag::EXACT, alpha, beta/* , line */, updated_state.castling_rights, updated_state.ep_square);
        
        unmake_move(state_history, position_count, zobrist);
        
        if (time_up.load(std::memory_order_relaxed)){
            return previous_search_data;
        }
        //std::cout <<"BBB5-2" << std::endl;
        zobrist = cur_hash;
        //highest_score = std::max(score,highest_score);

        if (score> highest_score){
            highest_score = score;
            // Alpha improved — update PV
            
            if (score > alpha)
                updatePV(move, 0);          

        }

        alpha = std::max(alpha,highest_score);
        //std::cout <<"BBB5-3" << std::endl;
    }

    /* bool side = current_state.turn;
    for (int i = 0; i < pv_length[0]; i++) {
        
        Move move = pv_table[0][i];
        int bonus = (((depth - 1) * (depth - 1)) * (pv_length[0] - i) * (pv_length[0] - i)) >> 2;
        moveFrequency[side][move.from_square][move.to_square] += bonus;
        side = !side;
    } */
    //std::cout <<"BBB5" << std::endl;
    if (previous_search_data.moves_list.empty()){

        returnData = current_search_data;
        returnData.moves_list = moves_list;

        sortSearchDataByScore(returnData);
    } else{
        returnData = previous_search_data;
        returnData.moves_list = moves_list;

        descending_sort_wrapper(current_search_data, returnData);
    }
    //std::cout <<"BBB6" << std::endl;
    return returnData;
}

int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint& t0, std::vector<int>& preliminary_scores, std::vector<Move>& pre_moves_list, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move prevMove, int& num_iterations){
    
    BoardState current_state = state_history.back();

    if(is_repetition(position_count, zobrist, 3) || current_state.halfmove_clock >= 100){
        if (cur_depth == 1)
            is_draw = true;
        return 0;
    }

    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    if (cur_depth >= depth_limit){        

        if (USE_Q_SEARCH /* && depth_limit >= 6 */){
            if(use_q_precautions.load(std::memory_order_relaxed)){
                if(depth_limit >= 6){                    
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, prevMove, num_iterations, false);
                    return result;
                }
            } else{
                int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, prevMove, num_iterations, false);
                return result;
            }

            
            
            /* int result = qSearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, prevMove, num_iterations, false);

            int num_plies = state_history.size();
            int max_cache_size;
            // Code segment to control cache size
            if(num_plies < 30){
                max_cache_size = 2000000;         
            }else if(num_plies < 50){
                max_cache_size = 4000000;
            }else if(num_plies < 75){
                max_cache_size = 8000000; 
            }else{
                max_cache_size = 16000000; 
            }
            TTFlag flag;
            if (result <= alpha)
                flag = TTFlag::UPPERBOUND;                
            else if (result >= beta)
                flag = TTFlag::LOWERBOUND;                
            else
                flag = TTFlag::EXACT;                

            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, QCacheEntry(result,flag), current_state.castling_rights, current_state.ep_square);

            addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, result, current_state.castling_rights, current_state.ep_square); */
            
        }
        return get_board_evaluation(state_history, zobrist, num_iterations);
    }
        
    increment_node_count_with_decay(num_iterations);

    //zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);
    uint64_t cur_hash = zobrist;

    int lowest_score = 9999999 - static_cast<int>(state_history.size()); 
    int score;
    int beta_orig = beta;   
    int alpha_orig = alpha;    
    bool using_tt = false;

    Move best_move;
    
    std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, prevMove);
    pre_moves_list = moves_list;
    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
    for (size_t i = 0; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
        using_tt = false;
        /*
        // Razoring
        if (i < second_level_preliminary_scores.size()) {
            if ((second_level_preliminary_scores[i] - beta > razor_threshold)) {
                continue;
            }
        }
        */
       
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
         
        // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
        updateZobristHashForMove(
            zobrist,
            move.from_square,
            move.to_square,
            capture_move,
            current_state.pawns,
            current_state.knights,
            current_state.bishops,
            current_state.rooks,
            current_state.queens,
            current_state.kings,
            current_state.occupied_colour[true],
            current_state.occupied_colour[false],
            move.promotion
        );

        
        make_move(state_history, position_count, move, zobrist, capture_move); 
        
        BoardState updated_state = state_history.back();
        if(is_repetition(position_count, zobrist, 3) || updated_state.halfmove_clock >= 100){
            score = 0;
        }else{

            TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);                           
            tt_visits++;
            if (entry != nullptr) {                
                //TTEntry entry = entry_opt.value();
                tt_probes++;
                if (entry->depth >= (depth_limit - cur_depth)) {
                    use_tt_entry(*entry, score, using_tt, alpha, beta, num_iterations, false, true);                    
                    /* if(entry.flag == TTFlag::EXACT){                                                             
                        score = entry.score;
                        using_tt = true;
                        increment_node_count_with_decay(num_iterations);
                    }else if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta && entry.beta >= beta) {                
                        score = entry.score;
                        using_tt = true;
                        increment_node_count_with_decay(num_iterations);                                                                              
                    }else if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha) {                
                        score = entry.score;
                        using_tt = true;
                        increment_node_count_with_decay(num_iterations);

                        unmake_move(state_history, position_count, zobrist);
                        preliminary_scores.push_back(score);                
                        updateMoveCacheForBetaCutoff(cur_hash, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);
                        if(!capture_move){
                            storeKillerMove(cur_depth, move);
                            historyHeuristics[current_state.turn][move.from_square][move.to_square] += cur_depth * cur_depth;
                            counterMoves[prevMove.from_square][prevMove.to_square] = move;
                            counterMoveHeuristics[current_state.turn][prevMove.from_square * 64 + prevMove.to_square][move.from_square * 64 + move.to_square] += cur_depth * cur_depth * cur_depth;
                        }  

                        pv_table[cur_depth][0] = move;
                        for (int j = 0; j < pv_length[cur_depth + 1]; ++j)
                            pv_table[cur_depth][j + 1] = pv_table[cur_depth + 1][j];
                        pv_length[cur_depth] = pv_length[cur_depth + 1] + 1;
                        return score;                  
                    } */
                }                
            }
            
            if(!using_tt){
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
                bool do_lmr = (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1);

                if (do_lmr){
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    //int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

                    TTEntry* entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);                           
                    
                    if (entry != nullptr) {                
                        //TTEntry entry = entry_opt.value();
                        tt_probes++;                    
                        if (entry->depth >= (reduced_depth - cur_depth)) {                    
                            if (entry->flag == TTFlag::EXACT){                                                             
                                score = entry->score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } 
                            //use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, false);
                        }                
                    }
                    if(!using_tt){
                        score = maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
                            
                        if(cur_depth < reduced_depth - 1){
                            TTFlag flag;
                            if (score <= alpha) {
                                flag = TTFlag::UPPERBOUND;
                                //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);  
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1/* , line */, updated_state.castling_rights, updated_state.ep_square);
                            } else if (score >= alpha + 1) {
                                flag = TTFlag::LOWERBOUND;
                                //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);  
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1/* , line */, updated_state.castling_rights, updated_state.ep_square);
                            }                                            
                        }
                        
                    }else{
                        tt_hits++;
                    }
                    if (score < beta){
                        using_tt = false;
                        
                        if(!using_tt){
                            score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
                            if(cur_depth < depth_limit - 1){
                                TTFlag flag;
                                if (score <= alpha_orig) {
                                    flag = TTFlag::UPPERBOUND;
                                } else if (score >= beta_orig) {
                                    flag = TTFlag::LOWERBOUND;
                                } else {
                                    flag = TTFlag::EXACT;
                                }  
                                //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);                            
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }                        
                    }                               
                } else{
                    
                    if(!using_tt){
                        score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);                    
                        if(cur_depth < depth_limit - 1){
                            TTFlag flag;
                            if (score <= alpha_orig) {
                                flag = TTFlag::UPPERBOUND;
                            } else if (score >= beta_orig) {
                                flag = TTFlag::LOWERBOUND;
                            } else {
                                flag = TTFlag::EXACT;
                            }        
                            //std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);                  
                            addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig/* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }
                }
            }else{
                tt_hits++;
            }    
        }              

        unmake_move(state_history, position_count, zobrist);
        zobrist = cur_hash;

        preliminary_scores.push_back(score);
        
        //lowest_score = std::min(score,lowest_score);
        if(score < lowest_score){
            lowest_score = score;
            best_move = move;
            // Beta improved — update PV
            if (score > alpha)
                updatePV(move, cur_depth);
        }

        beta = std::min(beta,lowest_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            if (i != 0)
                updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

            if(!capture_move){
                storeKillerMove(cur_depth, move);
                historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
                counterMoves[prevMove.from_square][prevMove.to_square] = move;
                counterMoveHeuristics[current_state.turn][prevMove.from_square * 64 + prevMove.to_square][move.from_square * 64 + move.to_square] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
            }

            return lowest_score;
        }
    }

    // Check if no moves are available, inidicating a game ending move was made previously
    if(lowest_score == 9999999 - static_cast<int>(state_history.size())){

        if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return lowest_score;
        }else if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return 0;
        }
    }

    bool en_passant_move = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
    bool capture_move = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    if(!capture_move){
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }

    return lowest_score;
}


int qSearch(int alpha, int beta, int cur_depth, int qDepth, const TimePoint& t0, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move prevMove, int& num_iterations, bool is_maximizing){
    qsearchVisits++;
    if(is_repetition(position_count, zobrist, 3)){
        return 0;
    }
    if (time_up.load(std::memory_order_relaxed)) {
        return 0;
    } else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL) {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT) {
            time_up.store(true, std::memory_order_relaxed);
        }
    }    

    if(qDepth >= MAX_QDEPTH)
        return get_board_evaluation(state_history, zobrist, num_iterations);

    BoardState current_state = state_history.back();
    increment_node_count_with_decay(num_iterations);
    int cache_result = accessQCache(zobrist, current_state.castling_rights, current_state.ep_square);

    if (cache_result != 0){
        //eval_cache_hits++;        
        return cache_result;
    }    
    
    /* int outScore;
    if (probeQCache(zobrist, current_state.castling_rights, current_state.ep_square, alpha, beta, outScore)) {
        return outScore;
    } */
    
    int moveNum = static_cast<int>(state_history.size());
    uint64_t cur_hash = zobrist;  

    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
    
    if (currently_in_check){
        std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth + qDepth, prevMove);
        
        if (moves_list.empty()){
            int eval = 0;
            

            if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }

            if (current_state.turn){
                eval = 9999999 - moveNum;
            } else{
                eval = -9999999 + moveNum;
            }
            if(Config::side_to_play)
                eval = -eval;

            return eval;        
        }        
        int best = is_maximizing ? -9999999 + moveNum : 9999999 - moveNum;

        for (size_t i = 0; i < moves_list.size(); ++i) {
            Move& move = moves_list[i];
                    
            bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

            // Acquire the zobrist hash for the new position if the given move was made
            bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

            // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
            updateZobristHashForMove(
                zobrist,
                move.from_square,
                move.to_square,
                capture_move,
                current_state.pawns,
                current_state.knights,
                current_state.bishops,
                current_state.rooks,
                current_state.queens,
                current_state.kings,
                current_state.occupied_colour[true],
                current_state.occupied_colour[false],
                move.promotion
            );

            //bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
            make_move(state_history, position_count, move, zobrist, capture_move);   
            int score = qSearch(alpha, beta, cur_depth, qDepth + 1, t0, state_history, position_count, zobrist, move, num_iterations, !is_maximizing);
            unmake_move(state_history, position_count, zobrist);
            
            zobrist = cur_hash;
            
            if (is_maximizing) {
                if (score > best) best = score;
                if (best > alpha) alpha = best;
                if (best >= beta) return best; // beta cutoff
            } else {
                if (score < best) best = score;
                if (best < beta) beta = best;
                if (best <= alpha) return best; // alpha cutoff
            }


        }
        return best;
    }

    int static_eval = get_board_evaluation(state_history, zobrist, num_iterations);
    if (is_maximizing) {
        if (static_eval >= beta)
            return static_eval; // Fail-hard beta cutoff
        if (static_eval > alpha)
            alpha = static_eval;
        if (static_eval < alpha - DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    } else {
        if (static_eval <= alpha)
            return static_eval; // Fail-hard alpha cutoff
        if (static_eval < beta)
            beta = static_eval;
        if (static_eval > beta + DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    }


    int best = is_maximizing ? -9999999 + moveNum : 9999999 - moveNum;

    std::vector<Move> moves_list = buildNoisyMoveList(zobrist, state_history, cur_depth + qDepth, prevMove);

    for (size_t i = 0; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
                
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

        // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
        updateZobristHashForMove(
            zobrist,
            move.from_square,
            move.to_square,
            capture_move,
            current_state.pawns,
            current_state.knights,
            current_state.bishops,
            current_state.rooks,
            current_state.queens,
            current_state.kings,
            current_state.occupied_colour[true],
            current_state.occupied_colour[false],
            move.promotion
        );
        //std::cout << (int)move.from_square << " | " << (int)move.to_square << std::endl;
        //bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        make_move(state_history, position_count, move, zobrist, capture_move);
        //int test_score = get_board_evaluation(state_history, zobrist, num_iterations);   
        int score = qSearch(alpha, beta, cur_depth, qDepth + 1, t0, state_history, position_count, zobrist, move, num_iterations, !is_maximizing);
        unmake_move(state_history, position_count, zobrist);

        /* if(score < 0){
            std::cout << test_score << " | " << score << " | " << create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) << std::endl;
        } 
        std::cout << std::endl;*/
        zobrist = cur_hash;
        
        if (is_maximizing) {
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (best >= beta) return best; // beta cutoff
        } else {
            if (score < best) best = score;
            if (best < beta) beta = best;
            if (best <= alpha) return best; // alpha cutoff
        }

    }

    if (moves_list.empty()) {
        return static_eval;
    }
    return best;
}

inline void sortSearchDataByScore(SearchData& data) {
    size_t n = data.top_level_preliminary_scores.size();

    // Create index vector
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on scores (descending)
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) {
                  return data.top_level_preliminary_scores[a] > data.top_level_preliminary_scores[b];
              });

    // Helper lambda to reorder any vector by indices
    auto reorder = [&](auto& vec) {
        using T = typename std::decay<decltype(vec[0])>::type;
        std::vector<T> temp(n);
        for (size_t i = 0; i < n; ++i)
            temp[i] = std::move(vec[indices[i]]);
        vec = std::move(temp);
    };

    // Reorder all associated vectors
    reorder(data.top_level_preliminary_scores);
    reorder(data.moves_list);
    reorder(data.second_level_moves_list);
    reorder(data.second_level_preliminary_scores);
}

inline void descending_sort_wrapper(const SearchData& preSearchData, SearchData& mainSearchData){
    

    // Find max alpha index in mainSearchData.top_level_preliminary_scores
    int max_index = 0;
    int max_value = mainSearchData.top_level_preliminary_scores[0];
    for (size_t i = 1; i < mainSearchData.top_level_preliminary_scores.size(); ++i) {
        if (mainSearchData.top_level_preliminary_scores[i] > max_value) {
            max_value = mainSearchData.top_level_preliminary_scores[i];
            max_index = static_cast<int>(i);
        }
    }
    //std::cout <<"BBB7" << std::endl;
    // Swap max to front in *mainSearchData* vectors
    if (max_index != 0) {
        std::swap(mainSearchData.top_level_preliminary_scores[0], mainSearchData.top_level_preliminary_scores[max_index]);
        std::swap(mainSearchData.moves_list[0], mainSearchData.moves_list[max_index]);
        std::swap(mainSearchData.second_level_preliminary_scores[0], mainSearchData.second_level_preliminary_scores[max_index]);
        std::swap(mainSearchData.second_level_moves_list[0], mainSearchData.second_level_moves_list[max_index]);
    }
    //std::cout <<"BBB8" << std::endl;
    // Determine count of valid entries in mainSearchData.top_level_preliminary_scores
    // Since no NULL, use size directly
    size_t count = std::min({
        mainSearchData.top_level_preliminary_scores.size(),
        preSearchData.top_level_preliminary_scores.size(),
        preSearchData.second_level_preliminary_scores.size(),
        preSearchData.second_level_moves_list.size()
    });


    // Prepare sublists excluding first element
    std::vector<int> top_level_preliminary_scores_sub(mainSearchData.top_level_preliminary_scores.begin() + 1, mainSearchData.top_level_preliminary_scores.end());
    std::vector<Move> moves_list_sub(mainSearchData.moves_list.begin() + 1, mainSearchData.moves_list.end());
    std::vector<std::vector<int>> second_level_preliminary_scores_sub(mainSearchData.second_level_preliminary_scores.begin() + 1, mainSearchData.second_level_preliminary_scores.end());
    std::vector<std::vector<Move>> second_level_moves_list_sub(mainSearchData.second_level_moves_list.begin() + 1, mainSearchData.second_level_moves_list.end());
    //std::cout <<"BBB9" << std::endl;
    // Append prelim search data starting from count
    if (preSearchData.top_level_preliminary_scores.size() > count) {
        top_level_preliminary_scores_sub.insert(top_level_preliminary_scores_sub.end(), preSearchData.top_level_preliminary_scores.begin() + count, preSearchData.top_level_preliminary_scores.end());
        second_level_preliminary_scores_sub.insert(second_level_preliminary_scores_sub.end(), preSearchData.second_level_preliminary_scores.begin() + count, preSearchData.second_level_preliminary_scores.end());
        second_level_moves_list_sub.insert(second_level_moves_list_sub.end(), preSearchData.second_level_moves_list.begin() + count, preSearchData.second_level_moves_list.end());
    }

    SearchData sub_data(moves_list_sub, top_level_preliminary_scores_sub, second_level_moves_list_sub, second_level_preliminary_scores_sub);    

    // Now sort the sublists descending by top_level_preliminary_scores_sub
    sortSearchDataByScore(sub_data);

    if (mainSearchData.top_level_preliminary_scores.size() < top_level_preliminary_scores_sub.size() + 1) {
        mainSearchData.top_level_preliminary_scores.resize(top_level_preliminary_scores_sub.size() + 1);
    }

    if (mainSearchData.second_level_preliminary_scores.size() < second_level_preliminary_scores_sub.size() + 1) {
        mainSearchData.second_level_preliminary_scores.resize(second_level_preliminary_scores_sub.size() + 1);
    }

    if (mainSearchData.second_level_moves_list.size() < second_level_moves_list_sub.size() + 1) {
        mainSearchData.second_level_moves_list.resize(second_level_moves_list_sub.size() + 1);
    }

    std::copy(sub_data.top_level_preliminary_scores.begin(), sub_data.top_level_preliminary_scores.end(), mainSearchData.top_level_preliminary_scores.begin() + 1);
    std::copy(sub_data.moves_list.begin(), sub_data.moves_list.end(), mainSearchData.moves_list.begin() + 1);
    std::copy(sub_data.second_level_preliminary_scores.begin(), sub_data.second_level_preliminary_scores.end(), mainSearchData.second_level_preliminary_scores.begin() + 1);
    std::copy(sub_data.second_level_moves_list.begin(), sub_data.second_level_moves_list.end(), mainSearchData.second_level_moves_list.begin() + 1);    
}

inline void ascending_sort(std::vector<int>& values, std::vector<Move>& moves) {
    size_t count = values.size();  // number of valid entries
    if (moves.size() < count) {
        throw std::runtime_error("Mismatch: 'moves' must be at least as long as 'values'");
    }

    // Generate sorted indices for the valid prefix
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) {
                  return values[a] < values[b];
              });

    // Reorder moves[0:count] based on sorted indices
    std::vector<Move> sorted_moves(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_moves[i] = std::move(moves[indices[i]]);
    }

    // Place sorted prefix back into moves[0:count]
    std::move(sorted_moves.begin(), sorted_moves.end(), moves.begin());

    // values is unchanged in size and content, so no need to touch it
}

inline uint8_t get_piece_type(uint8_t square, std::vector<BoardState>& state_history){
    
	BoardState current_state = state_history.back();
	uint64_t mask = (BB_SQUARES[square]);

	if (current_state.pawns & mask) {
		return PAWN;
	} else if (current_state.knights & mask){
		return KNIGHT;
	} else if (current_state.bishops & mask){
		return BISHOP;
	} else if (current_state.rooks & mask){
		return ROOK;
	} else if (current_state.queens & mask){
		return QUEEN;
	} else if (current_state.kings & mask){
		return KING;
	} else{
		return 0;
	}
}

inline bool relevant_pin_exists(std::vector<BoardState>& state_history, bool probe) {
    
    BoardState current_state = state_history.back();

    //uint64_t relevant_pins = 0;
    uint64_t bb = ~current_state.kings;
    uint64_t candidates = 0;     

    uint8_t white_king = __builtin_ctzll(current_state.kings & current_state.occupied_colour[true]);
    uint64_t queen_attacks_from_square = (BB_DIAG_ATTACKS[white_king][BB_DIAG_MASKS[white_king] & current_state.occupied] | BB_RANK_ATTACKS[white_king][BB_RANK_MASKS[white_king] & current_state.occupied] | 
                                         BB_FILE_ATTACKS[white_king][BB_FILE_MASKS[white_king] & current_state.occupied]) /* & current_state.occupied_colour[true] */; 

    candidates |= queen_attacks_from_square;                                    
    /* while(queen_attacks_from_square){
        uint8_t current_square = __builtin_ctzll(queen_attacks_from_square);
        queen_attacks_from_square &= queen_attacks_from_square - 1;

        relevant_pawns |= BB_SQUARES[current_square];
    } */

    uint8_t black_king = __builtin_ctzll(current_state.kings & current_state.occupied_colour[false]);
    queen_attacks_from_square = (BB_DIAG_ATTACKS[black_king][BB_DIAG_MASKS[black_king] & current_state.occupied] | BB_RANK_ATTACKS[black_king][BB_RANK_MASKS[black_king] & current_state.occupied] | 
                                         BB_FILE_ATTACKS[black_king][BB_FILE_MASKS[black_king] & current_state.occupied])/*  & current_state.occupied_colour[false] */; 
    
    candidates |= queen_attacks_from_square;
                                         /* while(queen_attacks_from_square){
        uint8_t current_square = __builtin_ctzll(queen_attacks_from_square);
        queen_attacks_from_square &= queen_attacks_from_square - 1;

        relevant_pawns |= BB_SQUARES[current_square];
    } */

    if(current_state.occupied_colour[true] & current_state.queens){
        uint8_t white_queen = __builtin_ctzll(current_state.queens & current_state.occupied_colour[true]);
        queen_attacks_from_square = (BB_DIAG_ATTACKS[white_queen][BB_DIAG_MASKS[white_queen] & current_state.occupied] | BB_RANK_ATTACKS[white_queen][BB_RANK_MASKS[white_queen] & current_state.occupied] | 
                                         BB_FILE_ATTACKS[white_queen][BB_FILE_MASKS[white_queen] & current_state.occupied])/*  & current_state.occupied_colour[true] */;
        candidates |= queen_attacks_from_square;
        /* if(probe)
            std::cout << "CANDIDATES:" << candidates << "; " << queen_attacks_from_square << ";" << std::endl; */
    }

    

    if(current_state.occupied_colour[false] & current_state.queens){
        uint8_t black_queen = __builtin_ctzll(current_state.queens & current_state.occupied_colour[false]);
        queen_attacks_from_square = (BB_DIAG_ATTACKS[black_queen][BB_DIAG_MASKS[black_queen] & current_state.occupied] | BB_RANK_ATTACKS[black_queen][BB_RANK_MASKS[black_queen] & current_state.occupied] | 
                                         BB_FILE_ATTACKS[black_queen][BB_FILE_MASKS[black_queen] & current_state.occupied]) /* & current_state.occupied_colour[false] */;
        candidates |= queen_attacks_from_square;
        /* if(probe)
            std::cout << "CANDIDATES:" << candidates << "; " << queen_attacks_from_square << ";" << std::endl; */
    }

    

    bb &= candidates & current_state.occupied;    
    if(probe)
        std::cout << "BITMASK:" << bb << ";"<< std::endl;
    while (bb) {
        uint8_t current_square = __builtin_ctzll(bb);
        bb &= bb - 1;

        bool is_white = current_state.occupied_colour[true] & BB_SQUARES[current_square];

        uint64_t own_pieces = is_white ? current_state.occupied_colour[true] : current_state.occupied_colour[false];
        uint64_t opp_pieces = is_white ? current_state.occupied_colour[false] : current_state.occupied_colour[true];

        // Get sliding attackers
        uint64_t rank_pieces = BB_RANK_MASKS[current_square] & current_state.occupied;
        uint64_t file_pieces = BB_FILE_MASKS[current_square] & current_state.occupied;
        uint64_t diag_pieces = BB_DIAG_MASKS[current_square] & current_state.occupied;

        uint64_t attackers = (
            (BB_RANK_ATTACKS[current_square][rank_pieces] & (current_state.queens | current_state.rooks)) |
            (BB_FILE_ATTACKS[current_square][file_pieces] & (current_state.queens | current_state.rooks)) |
            (BB_DIAG_ATTACKS[current_square][diag_pieces] & (current_state.queens | current_state.bishops))
        ) & opp_pieces;

        while (attackers) {
            uint8_t attacker_square = __builtin_ctzll(attackers);
            attackers &= attackers - 1;

            uint8_t attacker_type = get_piece_type(attacker_square, state_history);
            uint64_t behind_mask = attacks_mask(!is_white, current_state.occupied ^ BB_SQUARES[current_square], attacker_square, attacker_type) &
                                   ~attacks_mask(!is_white, current_state.occupied, attacker_square, attacker_type) & own_pieces;

            while (behind_mask) {
                uint8_t pinned_to_sq = __builtin_ctzll(behind_mask);
                behind_mask &= behind_mask - 1;

                uint8_t pinned_piece_type = get_piece_type(current_square, state_history);              
                uint8_t pinned_to_piece_type = get_piece_type(pinned_to_sq, state_history);             
                               // Consider it relevant if pinned to king, queen, or rook
                if(pinned_piece_type == PAWN){
                    if(pinned_to_piece_type == KING){
                        if(probe)
                            std::cout << (int)pinned_to_sq << " | " << (int)current_square << std::endl;
                        return true;
                    }                    
                }else if (pinned_to_piece_type == KING || pinned_to_piece_type == QUEEN || pinned_to_piece_type == ROOK) {
                    if(probe)
                        std::cout << (int)pinned_to_sq << " | " << (int)current_square << std::endl;
                    return true;
                    //relevant_pins |= BB_SQUARES[current_square];
                }
            }
        }
    }
    return false;
    //return relevant_pins;
}

inline void use_tt_entry(TTEntry& entry, int& score, bool& using_tt, int alpha, int beta, int& num_iterations, bool is_maximizing, bool use_extra_precautions) {
    using_tt = false;

    if (entry.flag == TTFlag::EXACT) {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }

    if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta) {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }

    if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha) {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }
}

inline void use_tt_entry1(TTEntry& entry, int& score, bool& using_tt, int alpha, int beta, int& num_iterations, bool is_maximizing, bool use_extra_precautions){
    using_tt = false;

    /* if(!(use_extra_precautions && entry.alpha <= alpha && entry.beta >= beta)){
        return;
    } */
    if (entry.flag == TTFlag::EXACT){
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }else{
        if(is_maximizing){
            if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta /* && entry.beta >= beta */) {                
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);                    
                return;                                
            }else if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha && entry.alpha <= alpha) {                
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;                    
            }
        }else{
            if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta && entry.beta >= beta) {                
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);                    
                return;                                
            }else if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha /* && entry.alpha <= alpha */) {                
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;                    
            }   
        }
    }
    
    
}

inline void increment_node_count_with_decay(int& num_iterations){
    num_iterations++;
    if (num_iterations % Config::DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (Config::DECAY_INTERVAL * 16)) == 0)
        decayCounterMoveHeuristics();
}

inline bool is_repetition(const std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key, const int repetition_count) {
    auto it = position_count.find(zobrist_key);
    return it != position_count.end() && it->second >= repetition_count;
}

inline bool isUnsafeForNullMovePruning(BoardState current_state) {
    
    // 1. In check? Disable null move pruning    
    /* if (is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]))
        return true; */

    // 2. Low material check — define material weight function
    /* int material = 0;

    material += __builtin_popcountll(current_state.occupied_colour[current_state.turn] & current_state.pawns) * values[PAWN];
    material += __builtin_popcountll(current_state.occupied_colour[current_state.turn] & current_state.knights) * values[KNIGHT];
    material += __builtin_popcountll(current_state.occupied_colour[current_state.turn] & current_state.bishops) * values[BISHOP];
    material += __builtin_popcountll(current_state.occupied_colour[current_state.turn] & current_state.rooks) * values[ROOK];
    material += __builtin_popcountll(current_state.occupied_colour[current_state.turn] & current_state.queens) * values[QUEEN];
    

    if (material < MIN_MATERIAL_FOR_NULL_MOVE) {
        // Very low material — risky for null move pruning
        return true;
    } */


    // 3. Additional zugzwang heuristics (optional)

    // Acquire the number of pieces on the board not including the kings
	int pieceNum = __builtin_popcountll(current_state.occupied_colour[current_state.turn]) - 1;
			
	if ((current_state.occupied_colour[current_state.turn] & current_state.queens) == 0){
		if(pieceNum < 7)
            return true;
	}else{
        if(pieceNum < 4)
            return true;
    }

    // 4. (Optional) Check for fortress / locked structure heuristics

    // Passed all checks — safe to null prune
    return false;
}

inline int reduced_search_depth(int depth_limit, int cur_depth, bool is_in_relavent_pin, int move_number, BoardState current_state) {
    
    /* if(is_in_relavent_pin && depth_limit < 5)
        return depth_limit; */
    
    if (depth_limit < 4)
        return depth_limit;  // No reduction for shallow depths

    // Adjust move number so 1 and 2 map to no reduction
    int adjusted_move = std::max(move_number - 2, 1);    
    int base = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];
    
    int phase = 0;
	phase += 4 * __builtin_popcountll(current_state.queens);
	phase += 2 * __builtin_popcountll(current_state.rooks);
	phase += 1 * __builtin_popcountll(current_state.bishops | current_state.knights);

	int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to 
    double scale = 0;
    
    if (phase_score <= 24) {
    	scale = 1.5;                
	} else if (phase_score <= 64) {
    	scale = 1.75;                
	}else if (phase_score <= 96) {
    	scale = 2.0;                
	} else if (phase_score < 117) {
		scale = 2.25;              
	} else {
		return base;
	}    
    double move_factor = std::log2(adjusted_move);
    int r = static_cast<int>(base - (move_factor / scale));
    
    if (is_in_relavent_pin){
        if (r <= cur_depth + 1){
            r = cur_depth + 2;
        }
        r = std::min(depth_limit,r);
    }

    return std::max(r, 2);  // Ensure at least 4 ply is searched
}

inline void updatePV(Move move, int cur_depth){
    pv_table[cur_depth][0] = move;
    for (int j = 0; j < pv_length[cur_depth + 1]; ++j)
        pv_table[cur_depth][j + 1] = pv_table[cur_depth + 1][j];
    pv_length[cur_depth] = pv_length[cur_depth + 1] + 1;
}


// Promote a move from the list to a given index (if found beyond that index)
inline void promoteMove(std::vector<Move>& moves, const Move& move, size_t promoteToIndex, size_t& indexIncrement) {
    auto it = std::find(moves.begin(), moves.end(), move);
    if (it != moves.end()) {
        size_t foundIndex = std::distance(moves.begin(), it);
        if (foundIndex > promoteToIndex + indexIncrement) {
            Move temp = *it;
            moves.erase(it);
            moves.insert(moves.begin() + promoteToIndex + indexIncrement, temp);
            indexIncrement++;
        }
    }
}


inline std::vector<Move> buildMoveListFromReordered(std::vector<BoardState>& state_history, uint64_t zobrist, int cur_ply, Move prevMove){

    move_gen_visits++;
    BoardState current_state = state_history.back();
    //uint64_t zobrist2 = zobrist;
    //zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);
    
    //if (zobrist != zobrist2)
        //std::cout << create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights, current_state.ep_square, current_state.turn) << std::endl;
    
    std::vector<Move> cached_moves = accessMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square);
    if(cached_moves.size() != 0){
        move_gen_cache_hits++;
        if (cached_moves.size() > 1){
            
            size_t firstNonCapture = 0;      
            //int firstKillerMoveUsed = false;
            size_t indexIncrement = 0;

            for (size_t i = 0; i < cached_moves.size(); ++i) {
                if (!is_capture(cached_moves[i].from_square,cached_moves[i].to_square, current_state.occupied_colour[!current_state.turn], is_en_passant(cached_moves[i].from_square,cached_moves[i].to_square, current_state.ep_square, current_state.occupied, current_state.pawns))){
                    firstNonCapture = static_cast<int>(i);
                    break;
                }
            }

            if (firstNonCapture == 0)
                firstNonCapture = 1;

            if (cur_ply > 63){
                auto cm = std::find(cached_moves.begin(), cached_moves.end(), counterMoves[prevMove.from_square][prevMove.to_square]);

                if (cm != cached_moves.end()){
                    size_t foundIndex = std::distance(cached_moves.begin(), cm);
                    if (foundIndex > firstNonCapture) {
                        std::iter_swap(cm, cached_moves.begin() + firstNonCapture + indexIncrement);                                    
                    }
                }
            }else{
                auto km1 = std::find(cached_moves.begin(), cached_moves.end(), killerMoves[cur_ply][0]);

                if (km1 != cached_moves.end()){
                    size_t foundIndex = std::distance(cached_moves.begin(), km1);
                    if (foundIndex > firstNonCapture) {
                        std::iter_swap(km1, cached_moves.begin() + firstNonCapture);
                        indexIncrement++;
                    }
                }

                auto km2 = std::find(cached_moves.begin(), cached_moves.end(), killerMoves[cur_ply][1]);

                if (km2 != cached_moves.end()){
                    size_t foundIndex = std::distance(cached_moves.begin(), km2);
                    if (foundIndex > firstNonCapture) {
                        std::iter_swap(km2, cached_moves.begin() + firstNonCapture + indexIncrement);      
                        indexIncrement++;              
                    }
                }         

                if (!(counterMoves[prevMove.from_square][prevMove.to_square] == killerMoves[cur_ply][0]) && !(counterMoves[prevMove.from_square][prevMove.to_square] == killerMoves[cur_ply][1])){
                    auto cm = std::find(cached_moves.begin(), cached_moves.end(), counterMoves[prevMove.from_square][prevMove.to_square]);

                    if (cm != cached_moves.end()){
                        size_t foundIndex = std::distance(cached_moves.begin(), cm);
                        if (foundIndex > firstNonCapture) {
                            std::iter_swap(cm, cached_moves.begin() + firstNonCapture + indexIncrement);                                    
                        }
                    }   
                }
            }
            
            /* if (cur_ply > 63) {
                Move cmove = counterMoves[prevMove.from_square][prevMove.to_square];
                promoteMove(cached_moves, cmove, firstNonCapture, indexIncrement);
            } else {
                Move k1 = killerMoves[cur_ply][0];
                Move k2 = killerMoves[cur_ply][1];
                Move cmove = counterMoves[prevMove.from_square][prevMove.to_square];

                // Promote killer 1                
                promoteMove(cached_moves, k1, firstNonCapture, indexIncrement);
                

                // Promote killer 2 (skip if same as killer 1)
                if (!(k2 == k1)) {
                    promoteMove(cached_moves, k2, firstNonCapture, indexIncrement);
                }

                // Promote counter move if it's not equal to either killer
                if (!(cmove == k1) && !(cmove == k2)) {
                    promoteMove(cached_moves, cmove, firstNonCapture, indexIncrement);
                }
            } */

        }        
        return cached_moves;
    }
    
	std::vector<Move> moves_list;
    moves_list.reserve(64);

    generateLegalMovesReordered(moves_list, current_state.castling_rights, ~0ULL, ~0ULL,
								current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights,
								current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn, cur_ply, prevMove);
    
    
    moves_list.shrink_to_fit();

    /* int num_plies = static_cast<int>(state_history.size());
    int max_cache_size;
    // Code segment to control cache size
    if(num_plies < 30){
        max_cache_size = 1000000;         
    }else if(num_plies < 50){
        max_cache_size = 1500000;
    }else if(num_plies < 75){
        max_cache_size = 2000000; 
    }else{
        max_cache_size = 2500000; 
    } */

    addToMoveGenCache(zobrist, /* max_cache_size * Config::ACTIVE->cache_size_multiplier ,*/ moves_list, current_state.castling_rights, current_state.ep_square);
    return moves_list;
}

inline std::vector<Move> buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState>& state_history, int cur_ply, Move prevMove){

    std::vector<Move> noisy_moves;
    noisy_moves.reserve(16);
    
    BoardState current_state = state_history.back();    
    
    std::vector<Move> moves_list;    

    moves_list = accessMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square);
    if(moves_list.size() == 0){
        generateLegalMovesReordered(moves_list, current_state.castling_rights, ~0ULL, ~0ULL,
								current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights,
								current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn, cur_ply, prevMove);
        moves_list.reserve(64);
    }
      
    
    for (size_t i = 0; i < moves_list.size(); ++i) {

        bool en_passant_move = is_en_passant(moves_list[i].from_square, moves_list[i].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);        
        bool capture_move = is_capture(moves_list[i].from_square, moves_list[i].to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

        if(moves_list[i].promotion != 1){
            noisy_moves.push_back(moves_list[i]);            
        }else if (capture_move){
            /* int pressure = get_pressure_at_square(current_state.turn, endPos[i]);
            int support = get_support_at_square(current_state.turn, endPos[i]);

            if (support == 0 || pressure > (support - SUPPORT_MARGIN)) {
                noisy_moves.push_back(Move(startPos[i],endPos[i],promotions[i]));
            } */
            if(en_passant_move){
                noisy_moves.push_back(moves_list[i]);
            }else if(see (moves_list[i].to_square, current_state.turn, current_state) >= 0){
                noisy_moves.push_back(moves_list[i]);
            }

        }else{
            uint64_t pawns = current_state.pawns;
            uint64_t knights = current_state.knights;
            uint64_t bishops = current_state.bishops;
            uint64_t rooks = current_state.rooks;
            uint64_t queens = current_state.queens;
            uint64_t kings = current_state.kings;

            uint64_t occupied_white = current_state.occupied_colour[true];
            uint64_t occupied_black = current_state.occupied_colour[false];
            uint64_t occupied = current_state.occupied;

            uint64_t promoted = current_state.promoted;

            bool turn = current_state.turn;
            uint64_t castling_rights = current_state.castling_rights;

            int ep_square = current_state.ep_square;
            
            //std::cout << "AA " << current_state.occupied << " | " << (int)startPos[i] << " | " << (int)endPos[i]<< std::endl;
            update_state(
                moves_list[i].to_square, 
                moves_list[i].from_square,
                pawns,
                knights,
                bishops,
                rooks,
                queens,
                kings,
                occupied,
                occupied_white,
                occupied_black,
                promoted,
                castling_rights,
                ep_square,
                moves_list[i].promotion,
                turn
            );
            //std::cout << "BB " << current_state.occupied << " | " << (int)startPos[i] << " | " << (int)endPos[i]<< std::endl;
            uint64_t opposingPieces = 0;
            if (turn){
                opposingPieces = occupied_black;
            }else{
                opposingPieces = occupied_white;
            }                

            bool move_is_check = is_check(turn, occupied, queens | rooks, queens | bishops, kings, knights, pawns, opposingPieces);
            if (move_is_check){
                noisy_moves.push_back(moves_list[i]);
            }
        }
    }
    noisy_moves.shrink_to_fit();
    return noisy_moves;
}

inline int get_q_search_eval(int alpha, int beta, int cur_depth, const TimePoint& t0, std::vector<BoardState>& state_history, BoardState current_state, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, Move prevMove, int& num_iterations, bool is_maximizing){
    
    /* int cache_result = accessQCache(zobrist, current_state.castling_rights, current_state.ep_square);

    if (cache_result != 0){
        //eval_cache_hits++;
        num_iterations++;
        return cache_result;
    } */

    int result = qSearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, prevMove, num_iterations, is_maximizing);

    int num_plies = state_history.size();
    int max_cache_size;
    // Code segment to control cache size
    if(num_plies < 30){
        max_cache_size = 2000000;         
    }else if(num_plies < 50){
        max_cache_size = 4000000;
    }else if(num_plies < 75){
        max_cache_size = 8000000; 
    }else{
        max_cache_size = 16000000; 
    }

    /* TTFlag flag;
    if (result <= alpha)
        flag = TTFlag::UPPERBOUND;                
    else if (result >= beta)
        flag = TTFlag::LOWERBOUND;                
    else
        flag = TTFlag::EXACT;                

    addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, QCacheEntry(result,flag), current_state.castling_rights, current_state.ep_square); */
    addToQCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, result, current_state.castling_rights, current_state.ep_square);
    
    return result;
        
}


inline int get_board_evaluation(std::vector<BoardState>& state_history, uint64_t zobrist, int& num_iterations){    

    increment_node_count_with_decay(num_iterations);
    BoardState current_state = state_history.back();

    int cache_result = 0;
    
    /* if (USE_Q_SEARCH){
        cache_result = accessQCache(zobrist, current_state.castling_rights, current_state.ep_square);

        if (cache_result != 0){
            //eval_cache_hits++;
            return cache_result;
        }
    } */
    
    eval_visits++;
    //cache_result = accessCache(zobrist);
    cache_result = accessCacheNew(zobrist);
    if (cache_result != 0){
        eval_cache_hits++;
        return cache_result;
    }            

    int total = 0;
    int moveNum = static_cast<int>(state_history.size());

    if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
    {
        if (current_state.turn){
            total = 9999999 - moveNum;
        } else{
            total = -9999999 + moveNum;
        }
    }else if(is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
    {
        return 0;
    }else{
        total = placement_and_piece_eval(moveNum, current_state.turn, current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                          current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.occupied); 
    }

    if(Config::side_to_play)
        total = -total;

    /* if(total == 23939){
        std::cout << create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) << std::endl;
    } */
    
    /* if (create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) == "8/8/3R3P/4P1P1/5PK1/8/2k5/2q1b3 b - - 0 1"){
                                    std::cout << total << std::endl;
 
    } */

    

    /* int num_plies = moveNum;
    int max_cache_size;
    // Code segment to control cache size
    if(num_plies < 30){
        max_cache_size = 4000000;         
    }else if(num_plies < 50){
        max_cache_size = 8000000;
    }else if(num_plies < 75){
        max_cache_size = 16000000; 
    }else{
        max_cache_size = 32000000; 
    } */

    //addToCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, total);
    addToCacheNew(zobrist, total);       
    return total;
}

    

