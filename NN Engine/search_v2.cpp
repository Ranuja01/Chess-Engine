/*
 * search_v2.cpp — search v2 (SEARCH_ARM=1), the negamax form of the search. Frames and plan: search_v2.h.
 *
 * #included by search_engine.cpp at its end, NOT compiled on its own: it calls that file's file-scope helpers and reads
 * its tables. Do not add it to setupAI.py or selfplay/build_uci.sh.
 *
 * Stage S1 is a one-for-one port: every behaviour knob of today's search is read here the same way, so the two searches
 * visit the same nodes under any setting. Pure diagnostics (probes, logs, counters that change nothing) are not ported.
 *
 * @author: Ranuja Pinnaduwage
 */

namespace sv2
{
	// Mate scores keep today's form: MATE - plies from the game start, so a shorter mate scores higher.
	static constexpr int MATE = 9999999;

	// The score of the side to move being checkmated, built as today's search builds it (the mated side's colour, then the
	// root-side flip) and converted to the side-to-move frame.
	static inline int mated_score(const BoardState &st, int plies, int s)
	{
		int eval = st.turn ? MATE - plies : -MATE + plies;
		if (Config::side_to_play)
			eval = -eval;
		return s * eval;
	}

	/*
	 * Quiescence search (today's qSearch). Fail-hard stand pat, captures and promotions from buildNoisyMoveList, every
	 * evasion when in check, delta pruning (whole node, or per capture with ENABLE_QDELTA_PERMOVE), the q-cache probe at
	 * every node, and the lazy-eval request on the stand-pat eval. alpha / beta and the result are side-to-move scores.
	 */
	int qsearch(int alpha, int beta, int cur_depth, int q_depth, const TimePoint &t0, std::vector<BoardState> &state_history,
	            std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prev_move, int &num_iterations,
	            bool root_side)
	{
		const int s = root_side ? 1 : -1;
		qsearchVisits++;
		if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD, state_history.back()))
			return 0;
		if (time_up.load(std::memory_order_relaxed))
			return 0;
		else if (Config::NODE_LIMIT > 0 && num_iterations >= Config::NODE_LIMIT)
		{
			time_up.store(true, std::memory_order_relaxed);
			return 0;
		}
		else if (nodes_since_time_check++ >= Config::TIME_CHECK_NODES)
		{
			nodes_since_time_check = 0;
			if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
				time_up.store(true, std::memory_order_relaxed);
		}

		if (q_depth >= Config::MAX_QDEPTH)
		{
			// A checkmated position at the q-depth cap scores as mate, not as a static eval.
			if (Config::FIX_QS_CHECK)
			{
				const BoardState &hs = state_history.back();
				if (is_check(hs.turn, hs.occupied, hs.queens | hs.rooks, hs.queens | hs.bishops, hs.kings, hs.knights, hs.pawns, hs.occupied_colour[!hs.turn]) &&
				    buildMoveListFromReordered(state_history, zobrist, cur_depth + q_depth, prev_move).empty())
				{
					++g_fixcnt_qsmate;
					return mated_score(hs, static_cast<int>(state_history.size()), s);
				}
			}
			int horizon_eval = eval_by_mode(Config::QSTANDPAT_EVAL_MODE, state_history, zobrist, num_iterations);
			if (Config::ENABLE_CORR_HIST && Config::ENABLE_CORRHIST_QSEARCH)
				horizon_eval += corrhist_correction(state_history.back(), root_side ? 1 : 0);
			if (Config::CORR2 && Config::CORR2_QS)
				horizon_eval += corr2_root(state_history.back(), root_side ? 1 : 0);
			return s * horizon_eval;
		}

		BoardState current_state = state_history.back();
		increment_node_count_with_decay(num_iterations);
		int cache_result;
		// The q-cache holds root-frame scores: probe with the root-frame window.
		if (!Config::DISABLE_QCACHE && probeQCache(zobrist, current_state.castling_rights, current_state.ep_square,
		                                           root_side ? alpha : -beta, root_side ? beta : -alpha, cache_result))
			return s * cache_result;

		const int move_num = static_cast<int>(state_history.size());
		const uint64_t cur_hash = zobrist;
		const bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops,
		                                         current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);

		if (currently_in_check)
		{
			std::vector<Move> &moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth + q_depth, prev_move);
			if (moves_list.empty())
			{
				if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn],
				                 current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
				                 current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
					return 0;
				return mated_score(current_state, move_num, s);
			}
			int best = -MATE + move_num;
			for (size_t i = 0; i < moves_list.size(); ++i)
			{
				Move &move = moves_list[i];
				const bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
				const bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
				updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move, current_state.pawns, current_state.knights, current_state.bishops,
				                         current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false],
				                         move.promotion);
				make_move(state_history, position_count, move, zobrist, capture_move);
				const int score = -qsearch(-beta, -alpha, cur_depth, q_depth + 1, t0, state_history, position_count, zobrist, move, num_iterations, !root_side);
				unmake_move(state_history, position_count, zobrist);
				zobrist = cur_hash;
				if (score > best)
					best = score;
				if (best > alpha)
					alpha = best;
				if (best >= beta)
					return best;
			}
			return best;
		}

		// Lazy eval (LAZY_V2 bit 1): the stand-pat cutoff may take the eval's estimate. The request is in the root frame:
		// the maximizer asks about its beta, the minimizer about its alpha, both net of the corrections read first.
		const bool lazy_qs = (Config::LAZY_V2 & 1) && !((Config::LAZY_V2_GUARDS & 2) && beta - alpha > 1);
		const int qcorr_ch = (Config::ENABLE_CORR_HIST && Config::ENABLE_CORRHIST_QSEARCH) ? corrhist_correction(current_state, root_side ? 1 : 0) : 0;
		const int qcorr_c2 = (Config::CORR2 && Config::CORR2_QS) ? corr2_root(current_state, root_side ? 1 : 0) : 0;
		if (lazy_qs)
			lazy_request(root_side, s * beta - qcorr_ch - qcorr_c2, LAZY_SITE_QS);
		const int raw_eval = eval_by_mode(Config::QSTANDPAT_EVAL_MODE, state_history, zobrist, num_iterations);
		if (lazy_qs)
			lazy_request_end();
		const int static_eval = s * (raw_eval + qcorr_ch + qcorr_c2);
		if (lazy_qs && g_lazy_fired && static_eval < beta)
			++g_lazy_nocut;

		if (static_eval >= beta)
			return static_eval;   // fail-hard stand pat
		if (static_eval > alpha)
			alpha = static_eval;
		if (Config::ENABLE_QDELTA && !Config::ENABLE_QDELTA_PERMOVE && static_eval < alpha - Config::DELTA_MARGIN)
			return static_eval;   // delta pruning, whole node

		int best = Config::ENABLE_QSTANDPAT_SEED ? static_eval : -MATE + move_num;
		std::vector<Move> &moves_list = buildNoisyMoveList(zobrist, state_history, cur_depth + q_depth, q_depth, prev_move);
		for (size_t i = 0; i < moves_list.size(); ++i)
		{
			Move &move = moves_list[i];
			const bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
			const bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

			// Per-capture delta pruning: skip a capture only when winning the victim plus the margin cannot reach alpha.
			// Promotions (promotion > 1) and en passant are exempt.
			if (Config::ENABLE_QDELTA_PERMOVE && Config::ENABLE_QDELTA && capture_move && !en_passant_move && move.promotion <= 1)
			{
				const int victim = get_value_at(move.to_square, current_state);
				const int pm_margin = Config::QDELTA_PERMOVE_MARGIN > 0 ? Config::QDELTA_PERMOVE_MARGIN : Config::DELTA_MARGIN;
				++g_qdelta_permove_seen;
				if (static_eval + pm_margin + victim <= alpha)
				{
					++g_qdelta_permove_fires;
					if (Config::ENABLE_QSTANDPAT_SEED && static_eval + pm_margin + victim > best)
						best = static_eval + pm_margin + victim;
					continue;
				}
			}

			updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move, current_state.pawns, current_state.knights, current_state.bishops,
			                         current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false],
			                         move.promotion);
			make_move(state_history, position_count, move, zobrist, capture_move);
			const int score = -qsearch(-beta, -alpha, cur_depth, q_depth + 1, t0, state_history, position_count, zobrist, move, num_iterations, !root_side);
			unmake_move(state_history, position_count, zobrist);
			zobrist = cur_hash;
			if (score > best)
				best = score;
			if (best > alpha)
				alpha = best;
			if (best >= beta)
			{
				++g_q_fh_total;
				if (i == 0)
					++g_q_fh_first;
				g_q_cut_idx_sum += (long)i;
				return best;
			}
		}
		if (moves_list.empty())
			return static_eval;
		return best;
	}
}
