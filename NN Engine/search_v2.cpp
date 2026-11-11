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

	// ── Root-frame access to the shared tables ─────────────────────────────────────────────────────────────────────────
	// A window (a, b) in the frame of a node whose side to move is (or is not) the root side, as the root-frame window.
	template <bool RootSide> static inline int root_lo(int a, int b) { return RootSide ? a : -b; }
	template <bool RootSide> static inline int root_hi(int a, int b) { return RootSide ? b : -a; }

	// The TT bound flag of a score against a window, both in this node's frame, decided in the root frame as the TT stores it.
	template <bool RootSide>
	static inline TTFlag root_flag(int score, int a, int b)
	{
		const int sc = RootSide ? score : -score;
		return sc <= root_lo<RootSide>(a, b) ? TTFlag::UPPERBOUND : sc >= root_hi<RootSide>(a, b) ? TTFlag::LOWERBOUND : TTFlag::EXACT;
	}

	// Store a child's search result (this node's frame) with a stored window (this node's frame) and a flag.
	template <bool RootSide>
	static inline void tt_store(uint64_t zobrist, size_t plies, int score, int depth, TTFlag flag, int a, int b, const BoardState &child)
	{
		addToSearchEvalCache(zobrist, plies, RootSide ? score : -score, depth, flag, root_lo<RootSide>(a, b), root_hi<RootSide>(a, b),
		                     child.castling_rights, child.ep_square);
	}

	// Try a child's TT entry against a window in this node's frame; on a hit, score is set in this node's frame.
	template <bool RootSide>
	static inline void tt_use(TTEntry &entry, int &score, bool &using_tt, int a, int b, int &num_iterations, int halfmove_clock, int plies)
	{
		int sc = 0;
		use_tt_entry(entry, sc, using_tt, root_lo<RootSide>(a, b), root_hi<RootSide>(a, b), num_iterations, RootSide, false, halfmove_clock, plies);
		if (using_tt)
			score = RootSide ? sc : -sc;
	}

	template <bool RootSide>
	int score_move(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check,
	               Move move, Move previous_move, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0,
	               std::vector<BoardState> &state_history, BoardState current_state, bool &using_fp, int &num_iterations, bool is_in_null_search);

	/*
	 * A child search in this node's frame: the child (the position after `move`, already made) is searched to depth_limit with
	 * this node's window (a, b), and its result comes back in this node's frame.
	 */
	template <bool RootSide>
	static inline int child_search(int cur_depth, int depth_limit, int a, int b, const TimePoint &t0, std::vector<BoardState> &state_history,
	                               std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move move, int &num_iterations, bool capture_move,
	                               bool is_in_null_search)
	{
		// Port step 2a: today's node functions answer, in the root frame.
		if constexpr (RootSide)
		{
			std::vector<int> no_scores;
			std::vector<Move> no_moves;
			RootScore no_entry;
			return minimizer(cur_depth + 1, depth_limit, a, b, t0, no_scores, no_moves, no_entry, state_history, position_count, zobrist, move,
			                 num_iterations, capture_move, false, is_in_null_search);
		}
		else
			return -maximizer(cur_depth + 1, depth_limit, -b, -a, t0, state_history, position_count, zobrist, move, num_iterations, capture_move,
			                  false, is_in_null_search);
	}

	/*
	 * Score one move of an interior node (today's get_score_for_maximizer / get_score_for_minimizer): the move is already made.
	 * Repetition and fifty-move draws, the check and passed-pawn extensions, the TT, then either a full-window search (the
	 * first move, or remaining depth 1 per A3) or the late-move path: LMR eligibility and its guards, LMP, history / SEE
	 * pruning, futility, the reduction, the scout, and the re-search rules (A1 / VERIFY). Scores and windows in this node's
	 * frame; a pruned move returns -MATE (never the new best, never a false cutoff).
	 */
	template <bool RootSide>
	int score_move(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check,
	               Move move, Move previous_move, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0,
	               std::vector<BoardState> &state_history, BoardState current_state, bool &using_fp, int &num_iterations, bool is_in_null_search)
	{
		constexpr int s = RootSide ? 1 : -1;
		if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
			g_searchStack[cur_depth] = move;   // this node's move, for the 2-ply continuation key
		bool using_tt = false;
		int score = 0;
		const bool best_mated = Config::FIX_MATE_GATE && g_best_mated;   // read before anything below can recurse
		BoardState updated_state = state_history.back();
		if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD, updated_state) || fifty_move_draw(updated_state))
		{
			FIXCNT_FIFTY(updated_state);
			return 0;
		}

		// Check extension (capped per path by CHECK_EXTENSION, optionally dropped for a losing-SEE spite check), else the
		// passed-pawn push extension (same per-path counter).
		bool extend = false;
		if (Config::CHECK_EXTENSION > 0 && g_check_extensions < Config::CHECK_EXTENSION && depth_limit + 2 < MAX_PLY)
		{
			extend = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops,
			                  updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
			if (extend && Config::SEE_EXTEND_MARGIN < Config::SEE_EXTEND_DISABLED && see(move.to_square, updated_state.turn, updated_state) > Config::SEE_EXTEND_MARGIN)
				extend = false;
		}
		if (!extend && Config::PE_PASSER_EXT > 0 && g_check_extensions < Config::PE_PASSER_EXT && depth_limit + 2 < MAX_PLY &&
		    pe_passer_push(move, current_state, updated_state, cur_depth))
		{
			extend = true;
			++g_pe_passer_ext;
		}
		CheckExtensionGuard ceg(extend);
		if (extend)
			depth_limit++;
		const int plies = static_cast<int>(state_history.size());

		TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
		tt_visits++;
		if (entry != nullptr)
		{
			tt_probes++;
			if (entry->depth >= (depth_limit - cur_depth))
				tt_use<RootSide>(*entry, score, using_tt, alpha, beta, num_iterations, updated_state.halfmove_clock, plies);
		}
		if (using_tt)
		{
			tt_hits++;
			return score;
		}

		// A3: the minimizer also searches every move full-window at remaining depth 1 (FIX_MIN_RD1 = 1 drops that; = 2 adds it
		// to the maximizer too).
		const bool full_window_rd1 = RootSide ? (Config::FIX_MIN_RD1 == 2) : (Config::FIX_MIN_RD1 != 1);
		if (i == 0 || (full_window_rd1 && (depth_limit - cur_depth) == 1))
		{
			score = child_search<RootSide>(cur_depth, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, is_in_null_search);
			if (cur_depth < depth_limit - 1)
				tt_store<RootSide>(zobrist, state_history.size(), score, depth_limit - cur_depth, root_flag<RootSide>(score, alpha_orig, beta_orig),
				                   alpha_orig, beta_orig, updated_state);
			return score;
		}

		const bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops,
		                                    updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
		const bool lmr_eligible = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1);
		const bool guard_pv = Config::PROTECT_PV && (beta - alpha > 1) && i <= Config::PROTECT_MAX_IDX;
		const bool guard_killer = Config::PROTECT_KILLERS && i <= Config::PROTECT_MAX_IDX &&
		                          (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previous_move.from_square][previous_move.to_square] == move);
		const bool guard_capchain = Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS == 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH;
		// TT-depth exemption: the child is already resolved to at least the depth about to be spent.
		bool guard_ttdepth = false;
		if (Config::PROTECT_TT_DEPTH && lmr_eligible)
		{
			const TTEntry *tte_g = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
			if (tte_g != nullptr && tte_g->depth >= (depth_limit - cur_depth) + Config::TT_GUARD_DEPTH_MARGIN)
			{
				guard_ttdepth = true;
				g_tt_depth_guard_fires++;
			}
		}
		const bool base_lmr = lmr_eligible && !guard_pv && !guard_killer && !guard_capchain && !guard_ttdepth;
		// Passed-pawn exemption: an advanced push of a true passer (no enemy pawn in its forward span) is not reduced or pruned.
		bool passer_exempt = false;
		if (Config::ENABLE_PASSER_PRUNE_EXEMPT && base_lmr && (updated_state.pawns & (1ULL << move.to_square)))
		{
			const int from_rank = move.from_square >> 3;
			const int to_rank = move.to_square >> 3;
			const int adv = (to_rank > from_rank) ? to_rank : (7 - to_rank);
			if (adv >= Config::PASSER_EXEMPT_ADV)
			{
				const bool mover_white = (to_rank > from_rank);
				const uint64_t enemy_pawns = updated_state.pawns & updated_state.occupied_colour[!mover_white];
				const uint64_t span = mover_white ? passed_span_white[move.to_square] : passed_span_black[move.to_square];
				if ((enemy_pawns & span) == 0)
				{
					passer_exempt = true;
					g_passer_exempt_fires++;
				}
			}
		}
		bool do_lmr = base_lmr && !passer_exempt;
		do_lmr = do_lmr && !(g_verify_no_reduce_until >= 0 && cur_depth <= g_verify_no_reduce_until);
		if (Config::ENABLE_LMR_COUNTERS && do_lmr)
			g_lmr_applied++;

		// With only pawns left (PE_PRUNE_GATE), or once the best line is mated (FIX_MATE_GATE), no shallow pruning; LMR stays.
		const bool pe_gate = Config::PE_PRUNE_GATE && pe_pawns_only(current_state);
		if (pe_gate)
			++g_pe_prune_gated;
		const bool prune_gate = pe_gate || best_mated;
		const int rd = depth_limit - cur_depth;

		// Late-move pruning (forcing moves are never do_lmr, so never pruned).
		if (Config::ENABLE_LMP && do_lmr && !prune_gate && !(Config::ENABLE_LMR_CAPCHAIN && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH))
		{
			const bool lmp_reached = rd >= 1 && rd <= Config::LMP_MAX_DEPTH && (int)i >= Config::LMP_BASE + Config::LMP_SCALE * rd * rd;
			const bool lmp_exempt = Config::ENABLE_LMP_HIST_EXEMPT && historyHeuristics[current_state.turn][move.from_square][move.to_square] >= Config::LMP_HIST_EXEMPT;
			if (lmp_reached && !lmp_exempt)
				return -MATE;
		}
		// History pruning of a late quiet with very negative butterfly history.
		if (Config::ENABLE_HIST_PRUNE && do_lmr && !prune_gate && rd >= 1 && rd <= Config::HIST_PRUNE_MAX_DEPTH &&
		    historyHeuristics[current_state.turn][move.from_square][move.to_square] < -Config::HIST_PRUNE_COEF * rd)
			return -MATE;
		// SEE pruning of a quiet whose moved piece the recapture wins.
		if (Config::ENABLE_SEE_PRUNE && do_lmr && !prune_gate && rd >= 1 && rd <= Config::SEE_PRUNE_MAX_DEPTH &&
		    see(move.to_square, updated_state.turn, updated_state) > Config::SEE_PRUNE_MARGIN)
			return -MATE;
		// SEE pruning of a losing capture (a checking capture extends, so it is never pruned).
		if (Config::SEE_PRUNE_CAPTURES && capture_move && !currently_in_check && !extend && !prune_gate && rd >= 1 && rd <= Config::SEE_PRUNE_MAX_DEPTH &&
		    see(move.to_square, current_state.turn, current_state) < -Config::SEE_PRUNE_CAPTURE_MARGIN)
			return -MATE;

		// The scout window. A2: the minimizer scouts at the root-frame (alpha, alpha + 1), which in its own frame is (beta - 1,
		// beta), unless FIX_MIN_SCOUT.
		const int sa = (RootSide || Config::FIX_MIN_SCOUT) ? alpha : beta - 1;
		const int sb = sa + 1;
		bool was_reduced = false;
		if (do_lmr)
		{
			const bool is_in_relavent_pin = relevant_pin(state_history);
			if ((RootSide || cur_depth > 1) && rd <= 4 && rd > 1 && depth_limit >= 5 && !is_in_relavent_pin)
			{
				// Futility of the child, on its static eval. Lazy eval (LAZY_V2 bit 4) may answer the eval early.
				const int fut_margin = FUTILITY_MARGINS_EFF[depth_limit - cur_depth - 1];
				const bool lazy_fut = (Config::LAZY_V2 & 4) && Config::ENABLE_FUTILITY && !prune_gate && !((Config::LAZY_V2_GUARDS & 2) && beta - alpha > 1);
				if (lazy_fut)
					lazy_request(!RootSide, s * (alpha - fut_margin - 1), LAZY_SITE_FUT);
				const int early_score = s * eval_by_mode(Config::FUTILITY_EVAL_MODE, state_history, zobrist, num_iterations);
				if (lazy_fut)
					lazy_request_end();
				if (Config::ENABLE_PRUNE_LOG)
					g_futility_evals++;
				if (Config::ENABLE_FUTILITY && !prune_gate && early_score + fut_margin < alpha)
				{
					if (Config::ENABLE_PRUNE_LOG)
						g_futility_fires++;
					using_fp = true;
					return early_score;
				}
			}
			int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
			if (Config::ENABLE_HISTORY_LMR)
			{
				int hist_delta = history_lmr_delta(move, previous_move, current_state, cur_depth);
				if (hist_delta < 0 && Config::HISTORY_LMR_SCALE > 0)
					hist_delta -= std::min((depth_limit - cur_depth) / Config::HISTORY_LMR_SCALE, Config::HISTORY_LMR_SCALE_CAP);
				if (hist_delta < 0 && is_in_relavent_pin)
					hist_delta = 0;   // only reduce-less may touch a pin defender
				reduced_depth = std::clamp(reduced_depth + hist_delta, 2, depth_limit);
			}
			if (Config::ENABLE_IMPROVING)
			{
				// g_evalStack is in the root frame: a higher value improves the maximizer, a lower one the minimizer.
				bool improving = true;
				if (cur_depth >= 2 && g_evalStack[cur_depth] != NO_STATIC_EVAL && g_evalStack[cur_depth - 2] != NO_STATIC_EVAL)
					improving = RootSide ? g_evalStack[cur_depth] > g_evalStack[cur_depth - 2] - Config::IMPROVING_DELTA_MARGIN
					                     : g_evalStack[cur_depth] < g_evalStack[cur_depth - 2] + Config::IMPROVING_DELTA_MARGIN;
				if (!improving)
					reduced_depth = std::clamp(reduced_depth - Config::IMPROVING_REDUCTION, 2, depth_limit);
			}
			if (Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS > 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH)
				reduced_depth = std::min(reduced_depth + Config::CAPCHAIN_REDUCE_LESS, depth_limit);
			if (Config::LMR_MIN_REM > 0 && (depth_limit - cur_depth - 1) < Config::LMR_MIN_REM)
				reduced_depth = depth_limit;
			if (Config::LMR_REM_FLOOR_PCT > 0)
			{
				const int rem = depth_limit - cur_depth - 1;
				if (rem > 0)
				{
					const int floor_depth = cur_depth + 1 + (rem * Config::LMR_REM_FLOOR_PCT) / 100;
					if (reduced_depth < floor_depth)
						reduced_depth = std::min(floor_depth, depth_limit);
				}
			}
			was_reduced = reduced_depth < depth_limit;
			TTEntry *rentry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
			if (rentry != nullptr)
			{
				tt_probes++;
				if (rentry->depth >= (reduced_depth - cur_depth))
					tt_use<RootSide>(*rentry, score, using_tt, sa, sb, num_iterations, updated_state.halfmove_clock, plies);
			}
			if (!using_tt)
			{
				score = child_search<RootSide>(cur_depth, reduced_depth, sa, sb, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, is_in_null_search);
				if (cur_depth < reduced_depth - 1)
				{
					const TTFlag flag = root_flag<RootSide>(score, sa, sb);
					if (flag != TTFlag::EXACT)
						tt_store<RootSide>(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, sa, sb, updated_state);
				}
			}
			else
				tt_hits++;
		}
		else
		{
			TTEntry *sentry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
			if (sentry != nullptr)
			{
				tt_probes++;
				if (sentry->depth >= (depth_limit - cur_depth))
					tt_use<RootSide>(*sentry, score, using_tt, sa, sb, num_iterations, updated_state.halfmove_clock, plies);
			}
			if (!using_tt)
			{
				score = child_search<RootSide>(cur_depth, depth_limit, sa, sb, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, is_in_null_search);
				if (cur_depth < depth_limit - 1)
				{
					const TTFlag flag = root_flag<RootSide>(score, sa, sb);
					if (flag != TTFlag::EXACT)
						tt_store<RootSide>(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, sa, sb, updated_state);
				}
			}
			else
				tt_hits++;
		}

		// The re-search rules, decided in the root frame exactly as today (A1 / VERIFY differ by side).
		const int rs = s * score, ra = root_lo<RootSide>(alpha, beta), rb = root_hi<RootSide>(alpha, beta);
		bool research;
		if constexpr (RootSide)
			research = (rs > ra && rs < rb) || (Config::FIX_RESEARCH && was_reduced && rs >= rb) ||
			           (Config::VERIFY_MARGIN > 0 && do_lmr && rs <= ra && (ra - rs) < Config::VERIFY_MARGIN);
		else
			research = (rs > ra && rs < rb) ||
			           (Config::FIX_RESEARCH ? ((was_reduced && rs <= ra) || (Config::VERIFY_MARGIN > 0 && do_lmr && rs >= rb && (rs - rb) < Config::VERIFY_MARGIN))
			                                 : (Config::VERIFY_MARGIN > 0 && do_lmr && rs <= ra && (ra - rs) < Config::VERIFY_MARGIN));
		if (research)
		{
			if (Config::ENABLE_LMR_COUNTERS && do_lmr)
				g_lmr_researches++;
			// A near-miss (VERIFY) re-search runs VERIFY_RESEARCH_REDUCTION plies short; a PVS re-search at full depth.
			const bool full = RootSide ? (Config::FIX_RESEARCH ? (rs > ra) : (rs > ra && rs < rb))
			                           : (Config::FIX_RESEARCH ? (rs < rb) : (rs > ra && rs < rb));
			const int research_depth = full ? depth_limit : std::max(cur_depth + 1, depth_limit - Config::VERIFY_RESEARCH_REDUCTION);
			score = child_search<RootSide>(cur_depth, research_depth, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, is_in_null_search);
			if (cur_depth < research_depth - 1)
			{
				// ENABLE_TT_FLAG_FIX: flag against the window actually searched (the drifted one), not the node's entry window.
				const TTFlag flag = Config::ENABLE_TT_FLAG_FIX ? root_flag<RootSide>(score, alpha, beta) : root_flag<RootSide>(score, alpha_orig, beta_orig);
				tt_store<RootSide>(zobrist, state_history.size(), score, research_depth - cur_depth, flag, alpha_orig, beta_orig, updated_state);
			}
		}
		return score;
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
