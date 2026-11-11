/*
 * search_v2.h — declarations for search v2 (SEARCH_ARM=1; dev_notes/active/SEARCH-V2-PLAN-2026-11-10.md).
 *
 * Search v2 is the negamax form of the search: one function per node kind instead of the maximizer / minimizer pair,
 * with the score always from the side to move. Stage S1 re-expresses today's search one for one, so at fixed depth it
 * must visit the same nodes as SEARCH_ARM=0 (bench 247 / 39,861,964 for gen 7.2).
 *
 * Score frames. Today's search scores every node in the ROOT-SIDE frame (root side positive; the eval's Black-positive
 * value is flipped once when the root side is White). v2 scores in the SIDE-TO-MOVE frame: v = s * root_frame, where
 * s = +1 when the side to move is the root side (today's maximizer) and -1 otherwise. The shared tables (TT, q-cache,
 * lazy-eval requests, corrections) keep the root-side frame in S1, so every access converts at the call.
 *
 * The implementation lives in search_v2.cpp, which search_engine.cpp #includes at its end (it uses that file's
 * file-scope helpers and tables); it is not a separate translation unit.
 *
 * @author: Ranuja Pinnaduwage
 */
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sv2
{
	// Quiescence search in the side-to-move frame. root_side = the side to move is the root side (today's is_maximizing).
	int qsearch(int alpha, int beta, int cur_depth, int q_depth, const TimePoint &t0, std::vector<BoardState> &state_history,
	            std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prev_move, int &num_iterations,
	            bool root_side);

	// An interior node below the root's children (today's maximizer, and the minimizer's non-root branch), in its own frame.
	template <bool RootSide>
	int negamax(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history,
	            std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previous_move, int &num_iterations,
	            bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search);

	// Score one already-made move of an interior node (today's get_score_for_maximizer / _minimizer), in this node's frame.
	template <bool RootSide>
	int score_move(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check,
	               Move move, Move previous_move, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0,
	               std::vector<BoardState> &state_history, BoardState current_state, bool &using_fp, int &num_iterations, bool is_in_null_search);
}
