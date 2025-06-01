#include "cpp_bitboard.h"
#include "search_engine.h"
#include "move_gen.h"

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <cmath>

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

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_white, initialState.occupied_black, initialState.turn);
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

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_white, initialState.occupied_black, initialState.turn);
    position_count[zobrist]++;
}

inline void make_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, Move move, uint64_t zobrist){
    
    BoardState current = state_history.back();

    uint64_t pawns = current.pawns;
    uint64_t knights = current.knights;
    uint64_t bishops = current.bishops;
    uint64_t rooks = current.rooks;
    uint64_t queens = current.queens;
    uint64_t kings = current.kings;

    uint64_t occupied_white = current.occupied_white;
    uint64_t occupied_black = current.occupied_black;
    uint64_t occupied = current.occupied;

    uint64_t promoted = current.promoted;

    bool turn = current.turn;
    uint64_t castling_rights = current.castling_rights;

    int ep_square = current.ep_square;
    int halfmove_clock = current.halfmove_clock;
    int fullmove_number = current.fullmove_number;

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

    //halfmove_clock += 1
    if (!turn)
        fullmove_number += 1;

    ep_square = -1;
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

void update_cache(int num_plies){
    
    int cacheSize = printCacheStats();
    
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
    
    std::cout << std::endl;

    cacheSize = printMoveGenCacheStats();
    
    // Code segment to control cache size
    if(num_plies < 30){
        if (cacheSize > 4000000)
            evictOldMoveGenEntries(cacheSize - 4000000);   
    }else if(num_plies < 50){
        if (cacheSize > 8000000)
            evictOldMoveGenEntries(cacheSize - 8000000); 
    }else if(num_plies < 75){
        if (cacheSize > 16000000)
            evictOldMoveGenEntries(cacheSize - 16000000); 
    }else{
        if (cacheSize > 32000000)
            evictOldMoveGenEntries(cacheSize - 32000000); 
    }
}

MoveData get_engine_move(std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count){

    update_cache(static_cast<int>(state_history.size()));
    BoardState current = state_history.back();

    uint64_t zobrist = generateZobristHash(current.pawns, current.knights, current.bishops, current.rooks, current.queens, current.kings, current.occupied_white, current.occupied_black, current.turn);

    Move move(0,0,0);

    int depth_limit = 3;
    int num_iterations = 0;
    int alpha = -9999998;
    int beta = 9999999;

    SearchData preliminary_search_data;

    TimePoint t0 = Clock::now();

    int score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    while (elapsed <= Config::MOVE_TIMES[depth_limit] && depth_limit + 1 < 64 && score < 9000000){
        
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
        //int delta = std::max(100, 1500 - 50 * depth_limit);  // depth in plies
        //int alpha_window = score - delta;  // prev_score is the score from the previous iteration
        //int beta_window = score + delta;

        score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
        
        /*
        if (score <= alpha_window || score >= beta_window) {
            // score is not usable, you need to re-search with wider bounds
            std::cout << "ASPIRATION WINDOW FAILED SCORE " << score << " ALPHA_WINDOW: " << alpha_window << " AND BETA_WINDOW: " << beta_window << " RE-SEARCHING DEPTH: " << depth_limit << std::endl;
            score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, t0, preliminary_search_data, move, num_iterations);
        }
        */
        
        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    }
         
    int x1 = (move.from_square & 7) + 1;
    int y1 = (move.from_square >> 3) + 1;

    int x2 = (move.to_square & 7) + 1;
    int y2 = (move.to_square >> 3) + 1;

    std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
    std::cout << std::endl;

    MoveData chosenMove(x1,y1,x2,y2, move.promotion, score, num_iterations);

    const BoardState& current_state = state_history.back();
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    
    bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
        current_state.occupied_white,
        current_state.occupied_black,
        move.promotion
    );

    make_move(state_history, position_count, move, zobrist);

    return chosenMove;
    
}

int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, const TimePoint& t0, SearchData& previous_search_data, Move& best_move, int& num_iterations){
    
    int best_score = -99999999;
    int score;  

    const BoardState& current_state = state_history.back();

    uint64_t cur_hash = zobrist;
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    
    SearchData current_search_data = reorder_legal_moves(alpha,beta, depth_limit, zobrist, previous_search_data, state_history, position_count, num_iterations);
    
    int razor_threshold;

    if (previous_search_data.moves_list.empty()) {
        razor_threshold = std::max(static_cast<int>(750 * std::pow(0.75, depth_limit - 4)), 200);
    } else {
        razor_threshold = std::max(static_cast<int>(300 * std::pow(0.75, depth_limit - 4)), 50);
    }

    previous_search_data.moves_list.clear();
    previous_search_data.top_level_preliminary_scores.clear();
    previous_search_data.second_level_moves_list.clear();
    previous_search_data.second_level_preliminary_scores.clear();

    // Define the number of moves, the best move index and the current index
    int num_legal_moves = static_cast<int>(current_search_data.moves_list.size());
    int best_move_index = -1;

    // Define the depth that should be used
    int depth_usage = 0;

    // Define variables to hold information on repeating moves
    bool repetition_flag = false;
    Move repetition_move;
    int repetition_score = 0;

    if (depth_limit >= 5) {
        std::cout << "Num Moves: " << num_legal_moves << std::endl;
    }
    
    bool en_passant_move = is_en_passant(current_search_data.moves_list[0].from_square, current_search_data.moves_list[0].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(current_search_data.moves_list[0].from_square, current_search_data.moves_list[0].to_square, opposingPieces, en_passant_move);

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
        current_state.occupied_white,
        current_state.occupied_black,
        current_search_data.moves_list[0].promotion
    );
    //std::cout <<"BBB" << std::endl;
    make_move(state_history, position_count, current_search_data.moves_list[0], zobrist);
    //std::cout <<"CCC"<< std::endl;
    score = minimizer(cur_depth + 1, depth_limit, alpha, beta, current_search_data.second_level_preliminary_scores[0], current_search_data.second_level_moves_list[0], previous_search_data, state_history, position_count, zobrist, num_iterations);
    //std::cout <<"DDDD"<< std::endl;
    if(is_repetition(position_count, zobrist, 2)){
        repetition_flag = true;
        repetition_move = current_search_data.moves_list[0];
        repetition_score = score;
        score = -100000000;
    }

    unmake_move(state_history, position_count, zobrist);
    zobrist = cur_hash;

    if (depth_limit >= 5){
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

    alpha = std::max(alpha, best_score);

    if(alpha - current_search_data.top_level_preliminary_scores[0] > razor_threshold)
        razor_threshold += alpha - current_search_data.top_level_preliminary_scores[0];


    previous_search_data.moves_list = current_search_data.moves_list;
    previous_search_data.top_level_preliminary_scores.push_back(score);

    if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::TIME_LIMIT)
        return score;

    for (size_t i = 1; i < current_search_data.moves_list.size(); ++i) {
        Move& move = current_search_data.moves_list[i];

        // Razoring
        if (i < current_search_data.top_level_preliminary_scores.size()) {
            if ((alpha - current_search_data.top_level_preliminary_scores[i] > razor_threshold) && (alpha < 9000000)) {
                break;
            }
        }

        // Late move reduction
        /*
        if (count >= 35)
            depth_usage = depth_limit;
        else
            depth_usage = depth_limit - 1;
        */
                
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
            current_state.occupied_white,
            current_state.occupied_black,
            move.promotion
        );

        make_move(state_history, position_count, move, zobrist);
        //std::cout <<"EEE" << std::endl;
        score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], previous_search_data, state_history, position_count, zobrist, num_iterations);
        //std::cout <<"FFF" << std::endl;
        // If the score is within the window, re-search with full window
        if (alpha < score && score < beta){
            previous_search_data.second_level_preliminary_scores.pop_back();
            previous_search_data.second_level_moves_list.pop_back();
            score = minimizer(cur_depth + 1, depth_limit, alpha, beta, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], previous_search_data, state_history, position_count, zobrist, num_iterations);
        }

        if(is_repetition(position_count, zobrist, 2)){
            repetition_flag = true;
            repetition_move = move;
            repetition_score = score;
            score = -100000001;
        }

        unmake_move(state_history, position_count, zobrist);
        zobrist = cur_hash;
        previous_search_data.top_level_preliminary_scores.push_back(score);
        
        if (depth_limit >= 5){
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
        }
            
        alpha = std::max(alpha, best_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            if (depth_limit >= 5) {
                std::cout << std::endl;
                std::cout << "Best: " << best_move_index << std::endl;
            }

            return best_score;
        }
        
        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::TIME_LIMIT){
            if (repetition_flag) {
                if (alpha < repetition_score) {
                    if (alpha <= -500) {
                        best_move = repetition_move;
                        return 0;
                    }
                }
            }
            return best_score;
        }
    }

    if (cur_depth == 0){
        if (repetition_flag) {
            if (alpha < repetition_score) {
                if (alpha <= -500) {
                    best_move = repetition_move;
                    best_score = 0;
                }
            }
        }

        if (depth_limit >= 5) {
            std::cout << std::endl;
            std::cout << "Best: " << best_move_index << std::endl;
        }

        return best_score;
    }
    return best_score;
}

int minimizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<int>second_level_preliminary_scores, std::vector<Move>second_level_moves_list, SearchData& previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations){

    if (cur_depth >= depth_limit)
        return get_board_evaluation(state_history, zobrist, num_iterations);

    const BoardState& current_state = state_history.back();
    
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    uint64_t cur_hash = zobrist;

    int lowest_score = 9999999 - static_cast<int>(state_history.size()); 
    int score;    
    

    /*
    int razor_threshold;

    if (depth_limit  == 3) {
        razor_threshold = std::max(static_cast<int>(2000 * std::pow(0.75, depth_limit - 4)), 200);
    } else {
        razor_threshold = std::max(static_cast<int>(1500 * std::pow(0.75, depth_limit - 4)), 50);
    }
    */
    
    if (cur_depth == 1){

        std::vector<int>cur_second_level_preliminary_scores;
        cur_second_level_preliminary_scores.reserve(64);

        ascending_sort(second_level_preliminary_scores, second_level_moves_list);
        previous_search_data.second_level_moves_list.push_back(second_level_moves_list);

        for (size_t i = 0; i < second_level_moves_list.size(); ++i) {
            Move& move = second_level_moves_list[i];

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
            bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
                current_state.occupied_white,
                current_state.occupied_black,
                move.promotion
            );

            make_move(state_history, position_count, move, zobrist);
            score = maximizer(cur_depth + 1, depth_limit, alpha, beta, state_history, position_count, zobrist, num_iterations);

            /*
            // If the score is within the window, re-search with full window
            if (alpha < score && score < beta){
                previous_search_data.second_level_preliminary_scores.pop_back();
                previous_search_data.second_level_moves_list.pop_back();
                score = minimizer(cur_depth + 1, depth_limit, alpha, beta, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], state_history, position_count, zobrist, num_iterations);
            }
            */
            
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;
            cur_second_level_preliminary_scores.push_back(score);
            
            lowest_score = std::min(score,lowest_score);
            beta = std::min(beta,lowest_score);

            // Check for a beta cutoff 
            if (beta <= alpha){
                previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);
                return lowest_score;
            }
        }

        // Check if no moves are available, inidicating a game ending move was made previously
        if(lowest_score == 9999999 - static_cast<int>(state_history.size())){

            num_iterations++;
            previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);


            uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;
            if (is_checkmate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return lowest_score;
            }else if(is_stalemate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return -100000000;
            }
        }
        previous_search_data.second_level_preliminary_scores.push_back(cur_second_level_preliminary_scores);

    } else{

        std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist);

        for (size_t i = 0; i < moves_list.size(); ++i) {
            Move& move = moves_list[i];

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
            bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
                current_state.occupied_white,
                current_state.occupied_black,
                move.promotion
            );

            make_move(state_history, position_count, move, zobrist);
            score = maximizer(cur_depth + 1, depth_limit, alpha, beta, state_history, position_count, zobrist, num_iterations);

            /*
            // If the score is within the window, re-search with full window
            if (alpha < score && score < beta){
                previous_search_data.second_level_preliminary_scores.pop_back();
                previous_search_data.second_level_moves_list.pop_back();
                score = minimizer(cur_depth + 1, depth_limit, alpha, beta, current_search_data.second_level_preliminary_scores[i], current_search_data.second_level_moves_list[i], state_history, position_count, zobrist, num_iterations);
            }
            */
            
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;
            
            lowest_score = std::min(score,lowest_score);
            beta = std::min(beta,lowest_score);

            // Check for a beta cutoff 
            if (beta <= alpha){
                return lowest_score;
            }
        }

        // Check if no moves are available, inidicating a game ending move was made previously
        if(lowest_score == 9999999 - static_cast<int>(state_history.size())){

            num_iterations++;

            uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;
            if (is_checkmate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return lowest_score;
            }else if(is_stalemate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return -100000000;
            }
        }

    }
    return lowest_score;    
}

int maximizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations){
    
    if (cur_depth >= depth_limit)
        return get_board_evaluation(state_history, zobrist, num_iterations);

    const BoardState& current_state = state_history.back();
    
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    uint64_t cur_hash = zobrist;

    int highest_score = -9999999;
    int score;  
    
    std::vector<int> dummy_ints;
    std::vector<Move> dummy_moves;
    SearchData dummy_data;

    std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist);

    for (size_t i = 0; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
                
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
            current_state.occupied_white,
            current_state.occupied_black,
            move.promotion
        );

        make_move(state_history, position_count, move, zobrist);

        score = minimizer(cur_depth + 1, depth_limit, alpha, beta, dummy_ints, dummy_moves, dummy_data, state_history, position_count, zobrist, num_iterations);
                
        unmake_move(state_history, position_count, zobrist);
        
        zobrist = cur_hash;
        highest_score = std::max(score,highest_score);
        alpha = std::max(alpha,highest_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            return highest_score;
        }
    }

    // Check if no moves are available, inidicating a game ending move was made previously
    if(highest_score == -9999999){

        num_iterations++;

        uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;
        if (is_checkmate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return -100000000;
        }
    }
    return highest_score;
}

SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, int& num_iterations){

    const BoardState& current_state = state_history.back();

    SearchData returnData;
    SearchData current_search_data;

    int score = -99999999;
    int highest_score = -99999999;
    int depth = depth_limit - 1;

    uint64_t cur_hash = zobrist;
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;

    std::vector<Move> moves_list;

    if (previous_search_data.moves_list.empty()){
        moves_list = buildMoveListFromReordered(state_history, zobrist);
    }else{
        moves_list = previous_search_data.moves_list;
    }
    //std::cout <<"BBB2" << std::endl;
       
    bool en_passant_move = is_en_passant(moves_list[0].from_square, moves_list[0].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

    // Acquire the zobrist hash for the new position if the given move was made
    bool capture_move = is_capture(moves_list[0].from_square, moves_list[0].to_square, opposingPieces, en_passant_move);

    // Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
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
        current_state.occupied_white,
        current_state.occupied_black,
        moves_list[0].promotion
    );

    make_move(state_history, position_count, moves_list[0], zobrist);

    std::vector<int> preliminary_scores;
    std::vector<Move> preliminary_moves;
    highest_score = pre_minimizer(1, depth, alpha, beta, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, num_iterations);
    //std::cout <<"BBB3" << std::endl;
    unmake_move(state_history, position_count, zobrist);
    zobrist = cur_hash;

    alpha = std::max(alpha, highest_score);

    current_search_data.top_level_preliminary_scores.push_back(highest_score);
    current_search_data.second_level_preliminary_scores.push_back(preliminary_scores);
    current_search_data.second_level_moves_list.push_back(preliminary_moves);

    for (size_t i = 1; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];
                
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
            current_state.occupied_white,
            current_state.occupied_black,
            move.promotion
        );

        make_move(state_history, position_count, move, zobrist);

        std::vector<int> preliminary_scores;
        std::vector<Move> preliminary_moves;

        score = pre_minimizer(1, depth, alpha, alpha + 1, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, num_iterations);
        //std::cout <<"BBB4" << std::endl;
        // If the score is within the window, re-search with full window
        if (alpha < score && score < beta){

            preliminary_scores.clear();
            preliminary_moves.clear();
            score = pre_minimizer(1, depth, alpha, beta, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, num_iterations);
        }

        current_search_data.top_level_preliminary_scores.push_back(score);
        current_search_data.second_level_preliminary_scores.push_back(preliminary_scores);
        current_search_data.second_level_moves_list.push_back(preliminary_moves);
                
        unmake_move(state_history, position_count, zobrist);
        
        zobrist = cur_hash;
        highest_score = std::max(score,highest_score);
        alpha = std::max(alpha,highest_score);
    }
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

int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, std::vector<int>& preliminary_scores, std::vector<Move>& pre_moves_list, std::vector<BoardState>& state_history, std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist, int& num_iterations){
    if (cur_depth >= depth_limit)
        return get_board_evaluation(state_history, zobrist, num_iterations);

    const BoardState& current_state = state_history.back();
    
    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    uint64_t cur_hash = zobrist;

    int lowest_score = 9999999 - static_cast<int>(state_history.size()); 
    int score;    

    std::vector<Move>moves_list = buildMoveListFromReordered(state_history, zobrist);
    pre_moves_list = moves_list;

    for (size_t i = 0; i < moves_list.size(); ++i) {
        Move& move = moves_list[i];

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
        bool capture_move = is_capture(move.from_square, move.to_square, opposingPieces, en_passant_move);

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
            current_state.occupied_white,
            current_state.occupied_black,
            move.promotion
        );

        make_move(state_history, position_count, move, zobrist);
        score = maximizer(cur_depth + 1, depth_limit, alpha, beta, state_history, position_count, zobrist, num_iterations);
        
        unmake_move(state_history, position_count, zobrist);
        zobrist = cur_hash;

        preliminary_scores.push_back(score);
        
        lowest_score = std::min(score,lowest_score);
        beta = std::min(beta,lowest_score);

        // Check for a beta cutoff 
        if (beta <= alpha){
            return lowest_score;
        }
    }

    // Check if no moves are available, inidicating a game ending move was made previously
    if(lowest_score == 9999999 - static_cast<int>(state_history.size())){

        num_iterations++;

        uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;
        if (is_checkmate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return lowest_score;
        }else if(is_stalemate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                            current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return -100000000;
        }
    }

    return lowest_score;
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
    // Precondition: moves_list is the same for both, so no need to reorder moves_list in preSearchData

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
    size_t count = mainSearchData.top_level_preliminary_scores.size();

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
    //std::cout <<"BBB10" << std::endl; 
    SearchData sub_data(moves_list_sub, top_level_preliminary_scores_sub, second_level_moves_list_sub, second_level_preliminary_scores_sub);
    //std::cout <<"BBB11" << std::endl;
    // Now sort the sublists descending by top_level_preliminary_scores_sub
    sortSearchDataByScore(sub_data);

    //std::cout <<"BBB12" << std::endl;

    if (mainSearchData.top_level_preliminary_scores.size() < top_level_preliminary_scores_sub.size() + 1) {
        mainSearchData.top_level_preliminary_scores.resize(top_level_preliminary_scores_sub.size() + 1);
    }

    if (mainSearchData.second_level_preliminary_scores.size() < second_level_preliminary_scores_sub.size() + 1) {
        mainSearchData.second_level_preliminary_scores.resize(second_level_preliminary_scores_sub.size() + 1);
    }

    if (mainSearchData.second_level_moves_list.size() < second_level_moves_list_sub.size() + 1) {
        mainSearchData.second_level_moves_list.resize(second_level_moves_list_sub.size() + 1);
    }

    std::copy(top_level_preliminary_scores_sub.begin(), top_level_preliminary_scores_sub.end(), mainSearchData.top_level_preliminary_scores.begin() + 1);
    std::copy(moves_list_sub.begin(), moves_list_sub.end(), mainSearchData.moves_list.begin() + 1);
    std::copy(second_level_preliminary_scores_sub.begin(), second_level_preliminary_scores_sub.end(), mainSearchData.second_level_preliminary_scores.begin() + 1);
    std::copy(second_level_moves_list_sub.begin(), second_level_moves_list_sub.end(), mainSearchData.second_level_moves_list.begin() + 1);
    //std::cout <<"BBB13" << std::endl;
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

inline std::vector<Move> buildMoveListFromReordered(std::vector<BoardState>& state_history, uint64_t zobrist){

    const BoardState& current_state = state_history.back();
    
    std::vector<Move> cached_moves = accessMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square);
    if(cached_moves.size() != 0){
        return cached_moves;
    }
    
	std::vector<Move> converted_moves;

    converted_moves.reserve(64);

    std::vector<uint8_t> startPos;
    std::vector<uint8_t> endPos;
    std::vector<uint8_t> promotions;

    startPos.reserve(64);
    endPos.reserve(64);
    promotions.reserve(64);

    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;

    generateLegalMovesReordered(startPos, endPos, promotions, current_state.castling_rights, ~0ULL, ~0ULL,
								current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns, current_state.knights,
								current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn);
    
    for (size_t i = 0; i < startPos.size(); ++i) {
        converted_moves.push_back(Move(startPos[i],endPos[i],promotions[i]));
    }

    addToMoveGenCache(zobrist, converted_moves, current_state.castling_rights, current_state.ep_square);
    return converted_moves;
}

inline bool is_repetition(const std::unordered_map<uint64_t, int>& position_count, uint64_t zobrist_key, const int repetition_count) {
    auto it = position_count.find(zobrist_key);
    return it != position_count.end() && it->second >= repetition_count;
}

inline int get_board_evaluation(std::vector<BoardState>& state_history, uint64_t zobrist, int& num_iterations){
    num_iterations++;

    int cache_result = accessCache(zobrist);

    if (cache_result != 0)
        return cache_result;

    BoardState current_state = state_history.back();

    int total = 0;
    int moveNum = static_cast<int>(state_history.size());

    uint64_t opposingPieces = current_state.turn ? current_state.occupied_black : current_state.occupied_white;
    uint64_t ourPieces = current_state.turn ? current_state.occupied_white : current_state.occupied_black;

    if (is_checkmate(current_state.castling_rights, current_state.occupied, current_state.occupied_white, opposingPieces, ourPieces, current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
    {
        if (current_state.turn){
            total = 9999999 - moveNum;
        } else{
            total = -9999999 + moveNum;
        }
    }else{
        total = placement_and_piece_eval(moveNum, current_state.turn, current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                          current_state.queens, current_state.kings, current_state.occupied_white, current_state.occupied_black, current_state.occupied); 
    }

    if(Config::side_to_play)
        total = -total;
            
    // Avoid the adding this evaluation to the cache if the move order matters
    addToCache(zobrist, total);
           
    return total;
}

