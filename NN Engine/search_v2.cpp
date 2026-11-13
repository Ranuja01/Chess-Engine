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

#include "sv2_probe.h"

namespace sv2
{
#ifdef SV2_MOVE_PROBE
	// S3 step 0: remaining-depth bucket 1, 2, 3, 4+ of a pruned move.
	static inline int probe_rd(int rd) { return std::min(std::max(rd, 1), 4) - 1; }
#endif

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

	// ── Lazy generation (Config::SV2_LAZY_GEN) ───────────────────────────────────────────────────────────────────────────
	/*
	 * Phase 2: the node's full list. Today's generated list, with phase 1's moves first in the order they were searched, so
	 * every move keeps today's index (on a first visit today's list starts with exactly those captures); stored in the move
	 * cache as the node's complete list.
	 */
	static void lazy_phase2(std::vector<BoardState> &state_history, uint64_t zobrist, int ply, Move previous_move, std::vector<Move> &list, size_t phase1)
	{
		const BoardState &cs = state_history.back();
		thread_local std::vector<Move> full;
		full.clear();
		generateLegalMovesReordered(full, cs.castling_rights, ~0ULL, ~0ULL, cs.occupied, cs.occupied_colour[true], cs.occupied_colour[!cs.turn],
		                            cs.occupied_colour[cs.turn], cs.pawns, cs.knights, cs.bishops, cs.rooks, cs.queens, cs.kings, cs.ep_square, cs.turn,
		                            ply, previous_move);
		list.resize(phase1);
		for (const Move &m : full)
			if (std::find(list.begin(), list.begin() + phase1, m) == list.begin() + phase1)
				list.push_back(m);
		addToMoveGenCache(zobrist, list, cs.castling_rights, cs.ep_square);
	}

	/*
	 * Phase 1: the node's list as far as it exists. A complete cached list goes through today's builder (phase1 stays
	 * SIZE_MAX: nothing to add). Otherwise the list is the leading good-tier captures (score >= MG_CAPTURE_TIER) in today's
	 * order, from the cache when an earlier visit cut on them, or from the noisy-only generator; a newly generated phase 1
	 * is cached marked "quiets pending", so a revisit is still a cache hit. Mode 2 runs phase 2 at once.
	 */
	static std::vector<Move> &lazy_phase1(std::vector<BoardState> &state_history, uint64_t zobrist, int ply, Move previous_move, size_t &phase1)
	{
		const BoardState &cs = state_history.back();
		const int cached = moveGenCacheState(zobrist, cs.castling_rights, cs.ep_square);
		if (cached == 1)
			return buildMoveListFromReordered(state_history, zobrist, ply, previous_move);
		std::vector<Move> &list = g_moveBuf[ply];
		if (cached == 2)
		{
			fillMoveGenCache(zobrist, cs.castling_rights, cs.ep_square, list);
			phase1 = list.size();
			return list;
		}
		int lead = 0;
		list.clear();
		generateLegalMovesReordered(list, cs.castling_rights, ~0ULL, ~0ULL, cs.occupied, cs.occupied_colour[true], cs.occupied_colour[!cs.turn],
		                            cs.occupied_colour[cs.turn], cs.pawns, cs.knights, cs.bishops, cs.rooks, cs.queens, cs.kings, cs.ep_square, cs.turn,
		                            ply, previous_move, true, &lead);
		list.resize(lead);
		if (Config::SV2_LAZY_GEN == 2 || lead == 0)
		{
			lazy_phase2(state_history, zobrist, ply, previous_move, list, (size_t)lead);
			return list;
		}
		phase1 = (size_t)lead;
		addToMoveGenCache(zobrist, list, cs.castling_rights, cs.ep_square);
		markMoveGenCacheQuietsPending(zobrist, cs.castling_rights, cs.ep_square);
		return list;
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

	/*
	 * A child search in this node's frame: the child (the position after `move`, already made) is searched to depth_limit with
	 * this node's window (a, b), and its result comes back in this node's frame.
	 */
	template <bool RootSide>
	static inline int child_search(int cur_depth, int depth_limit, int a, int b, const TimePoint &t0, std::vector<BoardState> &state_history,
	                               std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move move, int &num_iterations, bool capture_move,
	                               bool is_in_null_search)
	{
		return -negamax<!RootSide>(cur_depth + 1, depth_limit, -b, -a, t0, state_history, position_count, zobrist, move, num_iterations, capture_move,
		                           false, is_in_null_search);
	}

	/*
	 * The q-search at a horizon node (today's get_q_search_eval): searches, then stores the result in the q-cache with its
	 * bound against the window, both in the root frame. An abort value or a path-dependent draw (a bare 0 on a repetition) is
	 * not stored under QCACHE_SOUND_STORE.
	 */
	template <bool RootSide>
	static inline int qsearch_entry(int alpha, int beta, int cur_depth, const TimePoint &t0, std::vector<BoardState> &state_history, const BoardState &current_state,
	                                std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prev_move, int &num_iterations)
	{
		const int result = qsearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, prev_move, num_iterations, RootSide);
		const int root_result = RootSide ? result : -result;
		const TTFlag flag = root_flag<RootSide>(result, alpha, beta);
		const bool unsound_store = Config::QCACHE_SOUND_STORE && (time_up.load(std::memory_order_relaxed) ||
		                                                          (root_result == 0 && is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD, current_state)));
		if (!Config::DISABLE_QCACHE && !unsound_store)
			addToQCache(zobrist, root_result, flag, current_state.castling_rights, current_state.ep_square);
		return result;
	}

	// History after a quiet move's cutoff: killer, counter move, butterfly / continuation (1 and 2 plies) / piece-continuation
	// bonus, the threat history, and the malus on the quiets searched before it.
	static inline void quiet_cutoff_history(const BoardState &cs, const Move &move, const Move &previous_move, int cur_depth, int depth_limit,
	                                        std::vector<Move> &searched_quiets)
	{
		storeKillerMove(cur_depth, move);
		counterMoves[previous_move.from_square][previous_move.to_square] = move;
		const int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
		const Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
		const bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
		if (Config::ENABLE_HISTORY_SATURATION)
		{
			hist_update(historyHeuristics[cs.turn][move.from_square][move.to_square], b);
			hist_update(cmh_ref(cs.turn, previous_move, move, cs), b);
			if (p2v)
				hist_update(ch2_ref(cs.turn, p2, move, cs), b / Config::CONT2_GRAVITY_DIV);
			if (Config::ENABLE_PIECE_CONTHIST)
				hist_update(pieceContHist[cs.turn][pcont_ctx_key(previous_move, cs)][pcont_ent_key(move, cs)], b);
		}
		else
		{
			historyHeuristics[cs.turn][move.from_square][move.to_square] += b;
			cmh_ref(cs.turn, previous_move, move, cs) += 4 * b;
			if (p2v)
				ch2_ref(cs.turn, p2, move, cs) += 4 * b;
			if (Config::ENABLE_PIECE_CONTHIST)
				pieceContHist[cs.turn][pcont_ctx_key(previous_move, cs)][pcont_ent_key(move, cs)] += 4 * b;
		}
		threat_hist_on_cutoff(cs, move, searched_quiets, b, Config::ENABLE_HISTORY_SATURATION, Config::MALUS_DIV, Config::ENABLE_HISTORY_MALUS);
		if (Config::ENABLE_HISTORY_MALUS)
		{
			for (const Move &q : searched_quiets)
			{
				if (q == move)
					continue;
				if (Config::ENABLE_HISTORY_SATURATION)
				{
					hist_update(historyHeuristics[cs.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
					hist_update(cmh_ref(cs.turn, previous_move, q, cs), -b / Config::MALUS_DIV);
					if (p2v)
						hist_update(ch2_ref(cs.turn, p2, q, cs), -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
					if (Config::ENABLE_PIECE_CONTHIST)
						hist_update(pieceContHist[cs.turn][pcont_ctx_key(previous_move, cs)][pcont_ent_key(q, cs)], -b / Config::MALUS_DIV);
				}
				else
				{
					historyHeuristics[cs.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
					cmh_ref(cs.turn, previous_move, q, cs) -= 4 * b / Config::MALUS_DIV;
					if (p2v)
						ch2_ref(cs.turn, p2, q, cs) -= 4 * b / Config::MALUS_DIV;
					if (Config::ENABLE_PIECE_CONTHIST)
						pieceContHist[cs.turn][pcont_ctx_key(previous_move, cs)][pcont_ent_key(q, cs)] -= 4 * b / Config::MALUS_DIV;
				}
			}
		}
	}

	// Capture history after a capture's cutoff (ENABLE_CAPTURE_HIST), with the malus on the captures searched before it.
	static inline void capture_cutoff_history(const BoardState &cs, const Move &move, int cur_depth, int depth_limit, std::vector<Move> &searched_captures)
	{
		const int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
		if (Config::ENABLE_HISTORY_SATURATION)
			hist_update(capture_hist_ref(cs.turn, move.from_square, move.to_square, cs), b);
		else
			capture_hist_ref(cs.turn, move.from_square, move.to_square, cs) += b;
		if (Config::ENABLE_HISTORY_MALUS)
		{
			for (const Move &q : searched_captures)
			{
				if (q == move)
					continue;
				if (Config::ENABLE_HISTORY_SATURATION)
					hist_update(capture_hist_ref(cs.turn, q.from_square, q.to_square, cs), -b / Config::MALUS_DIV);
				else
					capture_hist_ref(cs.turn, q.from_square, q.to_square, cs) -= b / Config::MALUS_DIV;
			}
		}
	}

	// use_tt_entry's cut test without its side effect (the node count): would this child entry decide the move?
	static inline bool tt_entry_cuts(const TTEntry &entry, int root_alpha, int root_beta, int halfmove_clock, int plies)
	{
		if (Config::FIX_TT_R50 && halfmove_clock >= 90)
			return false;
		const int stored = tt_score_from_entry(entry.score, plies);
		return entry.flag == TTFlag::EXACT || (entry.flag == TTFlag::LOWERBOUND && stored >= root_beta) ||
		       (entry.flag == TTFlag::UPPERBOUND && stored <= root_alpha);
	}

	/*
	 * Late-move pruning decided before the move is made (SV2_PRUNE_EARLY). True only when score_move, given the made move,
	 * would return its pruned sentinel through late-move pruning: every test that comes before it there gives the same
	 * answer here (the child is not a repetition or fifty-move draw, its TT entry does not decide the move, it is not the
	 * first move or a full-window move, it gives no check), and the LMR eligibility, guards and the LMP condition hold. The
	 * child's key, castling rights, en passant square and halfmove clock are built from the move as update_state and
	 * make_move build them. Any case that needs the child position itself (castling, a pawn near the fifty-move limit, the
	 * passer knobs) returns false and the caller makes the move as usual.
	 */
	template <bool RootSide>
	static inline bool lmp_before_make(int alpha, int beta, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check,
	                                   const Move &move, const Move &previous_move, std::unordered_map<uint64_t, int> &position_count,
	                                   uint64_t child_zobrist, const BoardState &cs, int plies, int best_score)
	{
		if (capture_move || move.promotion != 1 || currently_in_check || i == 0 || !Config::ENABLE_LMR || !Config::ENABLE_LMP)
			return false;
		const int rd = depth_limit - cur_depth;
		if (rd < 1 || rd > Config::LMP_MAX_DEPTH || i < Config::LMP_BASE + Config::LMP_SCALE * rd * rd)
			return false;
		const bool full_window_rd1 = RootSide ? (Config::FIX_MIN_RD1 == 2) : (Config::FIX_MIN_RD1 != 1);
		if (full_window_rd1 && rd == 1)
			return false;
		if (Config::PE_PASSER_EXT > 0 || Config::ENABLE_PASSER_PRUNE_EXEMPT || Config::PROTECT_TT_DEPTH)
			return false;
		// LMR guards, the capture-chain gates, the verification window, the pruning gates, the history exemption
		if (Config::PROTECT_PV && (beta - alpha > 1) && i <= Config::PROTECT_MAX_IDX)
			return false;
		if (Config::PROTECT_KILLERS && i <= Config::PROTECT_MAX_IDX &&
		    (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previous_move.from_square][previous_move.to_square] == move))
			return false;
		if (Config::ENABLE_LMR_CAPCHAIN && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH)
			return false;
		if (g_verify_no_reduce_until >= 0 && cur_depth <= g_verify_no_reduce_until)
			return false;
		if ((Config::PE_PRUNE_GATE && pe_pawns_only(cs)) || (Config::FIX_MATE_GATE && best_score <= -9000000))
			return false;
		if (Config::ENABLE_LMP_HIST_EXEMPT && historyHeuristics[cs.turn][move.from_square][move.to_square] >= Config::LMP_HIST_EXEMPT)
			return false;
		const uint64_t from_bb = BB_SQUARES[move.from_square];
		const bool is_pawn = (cs.pawns & from_bb) != 0;
		const bool is_king = (cs.kings & from_bb) != 0;
		if (is_king && std::abs((int)(move.to_square & 7) - (int)(move.from_square & 7)) == 2)
			return false;   // castling: the rook moves too
		if (moveGivesCheckFast(cs, move))
			return false;   // a checking move is never reduced or pruned (and may be extended)
		// The child, as update_state and make_move build it
		uint64_t child_castling = cs.castling_rights & ~BB_SQUARES[move.to_square] & ~from_bb;
		if (is_king && !(cs.promoted & from_bb))
			child_castling &= cs.turn ? ~BB_RANK_1 : ~BB_RANK_8;
		int child_ep = -1;
		if (is_pawn)
		{
			const int diff = (int)move.to_square - (int)move.from_square;
			if (diff == 16 && (move.from_square >> 3) == 1)
				child_ep = move.from_square + 8;
			else if (diff == -16 && (move.from_square >> 3) == 6)
				child_ep = move.from_square - 8;
		}
		const int child_halfmove = (Config::FIX_HALFMOVE && is_pawn) ? 0 : cs.halfmove_clock + 1;
		if (child_halfmove >= 100)
			return false;   // the fifty-move test needs the child position
		// Repetition, with the child counted on the path exactly as make_move counts it
		const uint64_t rep_key = repetition_key(child_zobrist, child_castling, child_ep);
		bool rep;
		if (Config::REP_STACK && g_rep_in_search)
		{
			g_rep_path[g_rep_len++] = RepPathEntry{rep_key, is_pawn};
			rep = is_repetition_key(position_count, rep_key, Config::REPETITION_THRESHOLD);
			--g_rep_len;
		}
		else
		{
			position_count[rep_key]++;
			rep = is_repetition_key(position_count, rep_key, Config::REPETITION_THRESHOLD);
			if (--position_count[rep_key] == 0)
				position_count.erase(rep_key);
		}
		if (rep)
			return false;
		// A child TT entry that would decide the move first
		const TTEntry *entry = accessSearchEvalCache(child_zobrist, child_castling, child_ep);
		if (entry != nullptr && entry->depth >= rd && tt_entry_cuts(*entry, root_lo<RootSide>(alpha, beta), root_hi<RootSide>(alpha, beta), child_halfmove, plies + 1))
			return false;
		return true;
	}

	/*
	 * An interior node below the root's children (today's maximizer, and the minimizer's non-root branch). In order: the time
	 * check, mate-distance bounds (FIX_MDP), the horizon (q-search or static eval), the node-entry eval (improving, RFP / null
	 * gate, lazy), reverse futility, null move, IIR, ProbCut, the singular probe, then the move loop (singular extension,
	 * score_move, OTV, the cutoff with its history updates and node TT store), and the no-legal-move and history / correction
	 * updates at the end. Scores and windows in the side-to-move frame.
	 */
	template <bool RootSide>
	int negamax(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history,
	           std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previous_move, int &num_iterations,
	           bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
	{
		constexpr int s = RootSide ? 1 : -1;
		constexpr int side_bit = RootSide ? 1 : 0;   // the corrections' maximizer flag
		// Today's two node functions update the capture chain at different points of node entry; the order is kept (it
		// matters only on a time abort or a mate-distance return).
		if constexpr (RootSide)
			g_captureChain[cur_depth] = last_move_was_capture ? g_captureChain[cur_depth - 1] + 1 : std::max(0, g_captureChain[cur_depth - 1] - 1);

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

		const int g = static_cast<int>(state_history.size());
		// FIX_MDP: when the mate-distance bounds alone settle the window, return the bound.
		if (Config::FIX_MDP && (RootSide || cur_depth > 1))
		{
			if (alpha >= MATE - (g + 1))
				return alpha;
			if (beta <= -MATE + g)
				return beta;
		}
		if constexpr (!RootSide)
			g_captureChain[cur_depth] = last_move_was_capture ? g_captureChain[cur_depth - 1] + 1 : std::max(0, g_captureChain[cur_depth - 1] - 1);

		BoardState current_state = state_history.back();
		if (cur_depth >= depth_limit)
		{
			if (USE_Q_SEARCH && (!use_q_precautions.load(std::memory_order_relaxed) || depth_limit >= 6))
				return qsearch_entry<RootSide>(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previous_move, num_iterations);
			return s * get_board_evaluation(state_history, zobrist, num_iterations);
		}

		increment_node_count_with_decay(num_iterations);
		const uint64_t cur_hash = zobrist;
		const int no_move_score = -MATE + g;   // the best score before any move improves on it
		int best_score = no_move_score;
		const int alpha_orig = alpha, beta_orig = beta;
		Move best_move;
		bool all_moves_pruned = true;
		bool using_fp = false;
		int best_early_eval = no_move_score;
		const bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops,
		                                         current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
		if (Config::ENABLE_IMPROVING && cur_depth < MAX_PLY)
			g_evalStack[cur_depth] = (!currently_in_check && (depth_limit - cur_depth) <= Config::IMPROVING_EVAL_WINDOW)
			                             ? static_eval_for_improving(state_history, zobrist)
			                             : NO_STATIC_EVAL;

		// Node-entry static eval for RFP and the null-move gate, in the ROOT frame (NO_STATIC_EVAL when not computed): the
		// corrected value, and the raw one the correction update at node exit reuses.
		int rfp_eval = NO_STATIC_EVAL;
		int rfp_raw_eval = NO_STATIC_EVAL;
		const bool rfp_want_eval = !currently_in_check &&
		                           ((Config::ENABLE_RFP && (beta - alpha) == 1 && (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH &&
		                             (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH) ||
		                            Config::ENABLE_NULL_EVAL_GATE);
		if (rfp_want_eval)
		{
			// Lazy eval (LAZY_V2 bit 2), only where the reverse-futility prune is checked. The corrections are read first, so
			// a lazy exit is decided on the corrected value.
			const bool lazy_rfp = (Config::LAZY_V2 & 2) && Config::ENABLE_RFP && (beta - alpha) == 1 && beta > -9000000 && beta < 9000000 &&
			                      (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH;
			const int corr_ch = Config::ENABLE_CORR_HIST ? corrhist_correction(current_state, side_bit) : 0;
			const int corr_c2 = Config::CORR2 ? corr2_root(current_state, side_bit) : 0;
			if (lazy_rfp)
				lazy_request(RootSide, s * (beta + Config::RFP_MARGIN * (depth_limit - cur_depth)) - corr_ch - corr_c2, LAZY_SITE_RFP);
			rfp_raw_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
			if (lazy_rfp)
				lazy_request_end();
			rfp_eval = rfp_raw_eval + corr_ch + corr_c2;
			if (lazy_rfp && g_lazy_fired && s * rfp_eval - Config::RFP_MARGIN * (depth_limit - cur_depth) < beta)
				++g_lazy_nocut;
		}

		// Reverse futility (static null move): far enough above beta, a quiet move is assumed to hold.
		if (Config::ENABLE_RFP && rfp_eval != NO_STATIC_EVAL && (beta - alpha) == 1 && beta > -9000000 && beta < 9000000 &&
		    (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH &&
		    s * rfp_eval - Config::RFP_MARGIN * (depth_limit - cur_depth) >= beta)
		{
			if (Config::RFP_RETURN_BLEND > 0)   // the returned score pulled toward the bound
				return (Config::RFP_RETURN_BLEND * beta + (100 - Config::RFP_RETURN_BLEND) * (s * rfp_eval)) / 100;
			return s * rfp_eval;
		}

		// Null move. The depth gate has one knob per side (NULLMOVE_CURDEPTH_MAXI / _MINI).
		const int null_min_depth = RootSide ? Config::NULLMOVE_CURDEPTH_MAXI : Config::NULLMOVE_CURDEPTH_MINI;
		if (Config::ENABLE_NULLMOVE && cur_depth >= null_min_depth && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move &&
		    !currently_in_check && !isUnsafeForNullMovePruning(current_state) && (!Config::ENABLE_NULL_EVAL_GATE || s * rfp_eval >= beta))
		{
			// The null search reuses this history slot: keep this node's lazy-eval record and put it back (LAZY_V2_KEEP).
			const LazyPly lazy_keep = g_lazy_ply[g & 2047];
			state_history.back().turn = !state_history.back().turn;
			const int cur_ep_square = state_history.back().ep_square;
			state_history.back().ep_square = -1;
			updateZobristHashForNullMove(zobrist);

			int reduced_depth = depth_reduction(depth_limit);
			reduced_depth -= Config::NULLMOVE_EXTRA;
			if (depth_limit >= 10)
				reduced_depth -= 1;
			if (Config::ENABLE_NULLMOVE_EVAL_R && rfp_eval != NO_STATIC_EVAL)
			{
				const int extra_R = std::min((s * rfp_eval - beta) / Config::NULLMOVE_R_DIV, Config::NULLMOVE_R_CAP);
				if (extra_R > 0)
					reduced_depth -= extra_R;
			}
			if (Config::NULLMOVE_PROGRESSIVE)
			{
				const int base_depth = depth_limit - g_check_extensions;   // the iteration depth without check extensions
				if (base_depth >= 12)
					reduced_depth -= 1;
				if (base_depth >= 14)
					reduced_depth -= 1;
			}
			TTEntry *entry = accessSearchEvalCache(zobrist, state_history.back().castling_rights, state_history.back().ep_square);
			int null_move_score = 0;
			bool null_tt = false;
			if (entry != nullptr && entry->depth >= (reduced_depth - cur_depth) && entry->flag == TTFlag::EXACT)
			{
				null_move_score = s * tt_score_from_entry(entry->score, g);
				null_tt = true;
				increment_node_count_with_decay(num_iterations);
			}
			if (!null_tt)
			{
				Move dummy_move;
				if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
					g_searchStack[cur_depth] = dummy_move;   // a null move breaks the 2-ply continuation chain
				// A4: the null search probes (beta - 1, beta), as the cut below tests beta; the maximizer probes (alpha, alpha + 1)
				// unless FIX_MAX_NULL (the same at non-PV nodes).
				const int na = (!RootSide || Config::FIX_MAX_NULL) ? beta - 1 : alpha;
				null_move_score = -negamax<!RootSide>(cur_depth + 1, reduced_depth, -(na + 1), -na, t0, state_history, position_count, zobrist, dummy_move,
				                                     num_iterations, false, true, true);
			}
			state_history.back().turn = !state_history.back().turn;
			state_history.back().ep_square = cur_ep_square;
			zobrist = cur_hash;
			if (Config::LAZY_V2_KEEP)
				g_lazy_ply[g & 2047] = lazy_keep;
			if (null_move_score >= beta)
			{
				if (Config::ENABLE_NULL_MATE_CLAMP && null_move_score >= 9000000)
					return beta;   // an unproven mate returns the bound
				return null_move_score;
			}
		}

		// IIR: one ply less on a first visit (no cached move list) with depth to spare.
		if (Config::ENABLE_IIR && !currently_in_check && (depth_limit - cur_depth) >= Config::IIR_MIN_DEPTH &&
		    !moveGenCacheHasMoves(zobrist, current_state.castling_rights, current_state.ep_square))
			depth_limit -= 1;

#ifdef SV2_MOVE_PROBE
		const bool probe_hit = moveGenCacheHasMoves(zobrist, current_state.castling_rights, current_state.ep_square);
#endif
		// Lazy generation: phase1 = the length of a list whose quiet moves are not generated yet (SIZE_MAX = complete).
		size_t phase1 = SIZE_MAX;
		const bool lazy = Config::SV2_LAZY_GEN && !currently_in_check && cur_depth < MOVE_POOL_PLIES && !Config::ENABLE_PROBCUT && !Config::ENABLE_SINGULAR &&
		                  !Config::ENABLE_TT_MOVE;
		std::vector<Move> &moves_list = lazy ? lazy_phase1(state_history, zobrist, cur_depth, previous_move, phase1)
		                                     : buildMoveListFromReordered(state_history, zobrist, cur_depth, previous_move);
#ifdef SV2_MOVE_PROBE
		// The probe's remembered move for this position (written at this node's cutoff or best-move exit on an earlier visit).
		const uint64_t probe_key = make_move_cache_key(zobrist, current_state.castling_rights, current_state.ep_square);
		Sv2ProbeSlot &probe_slot = g_sv2_probe_slots[probe_key & ((1u << SV2_PROBE_BITS) - 1)];
		const Move probe_rem_move(probe_slot.from, probe_slot.to, probe_slot.promo);
		int probe_rem_idx = -1;
		if (probe_slot.key == probe_key)
			for (size_t k = 0; k < moves_list.size(); ++k)
				if (moves_list[k] == probe_rem_move)
				{
					probe_rem_idx = (int)k;
					break;
				}
		++g_sv2_probe.lists;
		g_sv2_probe.lists_hit += probe_hit;
		g_sv2_probe.moves_main += (long long)moves_list.size();
		// Lazy generation's payoff (S3 picker option B): on a generated list (cache miss), the leading run of captures and
		// promotions, and the quiet moves behind it; a cutoff inside that run would have needed no quiet generation at all.
		int probe_noisy_lead = 0;
		long long probe_quiets = 0;
		if (!probe_hit)
		{
			const uint64_t opp = current_state.occupied_colour[!current_state.turn];
			bool lead = true;
			for (const Move &m : moves_list)
			{
				const bool noisy = m.promotion != 1 || (BB_SQUARES[m.to_square] & opp) ||
				                   (m.to_square == current_state.ep_square && (current_state.pawns & BB_SQUARES[m.from_square]));
				if (lead && noisy)
					++probe_noisy_lead;
				else
					lead = false;
				probe_quiets += !noisy;
			}
			g_sv2_probe.miss_moves += (long long)moves_list.size();
			g_sv2_probe.miss_quiets += probe_quiets;
			g_sv2_probe.miss_noisy_lead += probe_noisy_lead;
			g_sv2_probe.miss_no_noisy += (probe_noisy_lead == 0);
		}
		if (probe_rem_idx >= 0)
		{
			if (probe_hit)
			{
				++g_sv2_probe.rem_hit_have;
				g_sv2_probe.rem_hit_head += (probe_rem_idx == 0);
			}
			else
			{
				++g_sv2_probe.rem_miss_have;
				g_sv2_probe.rem_miss_at0 += (probe_rem_idx == 0);
			}
		}
#endif
		MalusList malus_quiets, malus_captures;   // history-gravity malus lists (empty when the malus is off)
		std::vector<Move> &searched_quiets = malus_quiets.v, &searched_captures = malus_captures.v;

		// OTV's reference: the node's own static eval, when RFP / the null gate did not compute it.
		if (Config::ENABLE_OTV && rfp_eval == NO_STATIC_EVAL && !g_in_verify && !currently_in_check && (!Config::OTV_PV_ONLY || (beta - alpha) > 1) &&
		    (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING)
			rfp_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);

		// ProbCut: at a non-PV node, a good capture or promotion that holds a null window PROBCUT_MARGIN above beta at reduced
		// depth cuts the node.
		if (Config::ENABLE_PROBCUT && !g_in_verify && !currently_in_check && (beta - alpha) == 1 && (depth_limit - cur_depth) >= Config::PROBCUT_MIN_DEPTH &&
		    alpha > -9000000 && alpha < 9000000 && beta > -9000000 && beta < 9000000)
		{
			const int probcut_beta = beta + Config::PROBCUT_MARGIN;
			int probcut_tried = 0;
			for (size_t i = 0; i < moves_list.size() && probcut_tried < Config::PROBCUT_CANDIDATES; ++i)
			{
				Move &move = moves_list[i];
				const bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
				const bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
				if (!(capture_move || move.promotion != 1))
					continue;
				if (see(move.to_square, current_state.turn, current_state) < 0)
					continue;
				++probcut_tried;
				updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move, current_state.pawns, current_state.knights, current_state.bishops,
				                         current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false],
				                         move.promotion);
				make_move(state_history, position_count, move, zobrist, capture_move);
				const bool prev_no_store = g_no_tt_store;
				if (Config::ENABLE_PROBCUT_NO_TT_STORE)
					g_no_tt_store = true;
				const int probcut_score = -negamax<!RootSide>(cur_depth + 1, depth_limit - Config::PROBCUT_DEPTH_REDUCTION, -probcut_beta, -probcut_beta + 1, t0,
				                                             state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
				g_no_tt_store = prev_no_store;
				unmake_move(state_history, position_count, zobrist);
				zobrist = cur_hash;
				if (time_up.load(std::memory_order_relaxed))
					return 0;
				if (probcut_score >= probcut_beta && probcut_score < 9000000)
					return probcut_score;
			}
		}

		// Singular probe (read-only): the node's TT move with its score (this node's frame), depth and bound. excluding = this
		// is the exclusion re-search of this node.
		const bool excluding = Config::ENABLE_SINGULAR && (g_excluded_move[cur_depth].from_square != g_excluded_move[cur_depth].to_square);
		Move tt_move;
		int tt_score = 0, tt_depth = -1;
		TTFlag tt_flag = TTFlag::EXACT;
		bool have_tt = false;
		if (Config::ENABLE_SINGULAR && !excluding)
		{
			TTEntry *node_tt = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
			if (node_tt != nullptr)
			{
				tt_move = node_tt->move;
				tt_score = s * tt_score_from_entry(node_tt->score, g);
				tt_depth = node_tt->depth;
				tt_flag = node_tt->flag;
				have_tt = true;
			}
			if (have_tt && tt_move.from_square != tt_move.to_square)
				++g_sing_eligible;
		}
		// A fail-high in this node's frame: a root-frame lower bound for the maximizer, an upper bound for the minimizer.
		constexpr TTFlag tt_fail_high = RootSide ? TTFlag::LOWERBOUND : TTFlag::UPPERBOUND;

		if (Config::ENABLE_KILLER_CHILD_CLEAR && cur_depth + 1 < MAX_PLY)
		{
			killerMoves[cur_depth + 1][0] = Move{};
			killerMoves[cur_depth + 1][1] = Move{};
		}
		for (size_t i = 0;; ++i)
		{
			if (i == phase1)   // phase 1's captures did not cut: the quiet moves now (lazy generation)
			{
				lazy_phase2(state_history, cur_hash, cur_depth, previous_move, moves_list, phase1);
				phase1 = SIZE_MAX;
			}
			if (i >= moves_list.size())
				break;
			Move &move = moves_list[i];
			if (excluding && move == g_excluded_move[cur_depth])
				continue;
			using_fp = false;
			const bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
			const bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

			// Singular extension: the TT move (a fail-high) is extended one ply when a half-depth exclusion search of the
			// other moves cannot reach its score less SINGULAR_MARGIN per ply.
			int singular_extra = 0;
			if (Config::ENABLE_SINGULAR && !excluding && have_tt && !currently_in_check && move == tt_move && tt_move.from_square != tt_move.to_square &&
			    (depth_limit - cur_depth) >= Config::SINGULAR_MIN_DEPTH && tt_depth >= (depth_limit - cur_depth) - 3 && tt_score > -9000000 && tt_score < 9000000 &&
			    tt_flag == tt_fail_high && g_singular_extensions < Config::SINGULAR_MAX_EXT)
			{
				++g_sing_gatepass;
				const int rem = depth_limit - cur_depth;
				const int sb = tt_score - Config::SINGULAR_MARGIN * rem;
				g_excluded_move[cur_depth] = move;
				const int v = negamax<RootSide>(cur_depth, cur_depth + rem / 2, sb - 1, sb, t0, state_history, position_count, zobrist, previous_move, num_iterations,
				                               last_move_was_capture, false, is_in_null_search);
				g_excluded_move[cur_depth] = Move();
				if (v < sb)
				{
					++g_sing_fire;
					singular_extra = 1;
				}
			}

			// Late-move pruning before the move is made: what the loop below does after a pruned return, without the move.
			if (Config::SV2_PRUNE_EARLY && singular_extra == 0)
			{
				uint64_t child_zobrist = zobrist;
				updateZobristHashForMove(child_zobrist, move.from_square, move.to_square, capture_move, current_state.pawns, current_state.knights, current_state.bishops,
				                         current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false],
				                         move.promotion);
				if (lmp_before_make<RootSide>(alpha, beta, (int)i, cur_depth, depth_limit, capture_move, currently_in_check, move, previous_move, position_count,
				                              child_zobrist, current_state, g, best_score))
				{
#ifdef SV2_PRUNE_VERIFY
					// The normal path must agree: the made move comes back as the pruned sentinel.
					{
						make_move(state_history, position_count, move, child_zobrist, capture_move);
						bool fp = false;
						g_best_mated = best_score <= -9000000;
						const int vs = score_move<RootSide>(alpha, beta, alpha_orig, beta_orig, (int)i, cur_depth, depth_limit, capture_move, currently_in_check, move,
						                                    previous_move, position_count, child_zobrist, t0, state_history, current_state, fp, num_iterations, is_in_null_search);
						unmake_move(state_history, position_count, child_zobrist);
						++g_sv2_prune_checked;
						g_sv2_prune_mismatch += (vs != -MATE || fp);
					}
#endif
					g_best_mated = best_score <= -9000000;
					if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
						g_searchStack[cur_depth] = move;
					if (Config::ENABLE_HISTORY_MALUS)
						searched_quiets.push_back(move);
					all_moves_pruned = false;
					if (time_up.load(std::memory_order_relaxed))
						return 0;
					continue;
				}
			}

			updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move, current_state.pawns, current_state.knights, current_state.bishops,
			                         current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false],
			                         move.promotion);
			make_move(state_history, position_count, move, zobrist, capture_move);
			int score;
			{
				SingularExtensionGuard seg(singular_extra > 0);   // bounds a chain of singular extensions by SINGULAR_MAX_EXT
				g_best_mated = best_score <= -9000000;
				score = score_move<RootSide>(alpha, beta, alpha_orig, beta_orig, (int)i, cur_depth, depth_limit + singular_extra, capture_move, currently_in_check, move,
				                             previous_move, position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search);
			}
			if (Config::ENABLE_HISTORY_MALUS)
				(capture_move ? searched_captures : searched_quiets).push_back(move);
			if (using_fp)
			{
				best_early_eval = std::max(best_early_eval, score);
				unmake_move(state_history, position_count, zobrist);
				zobrist = cur_hash;
				continue;
			}
			all_moves_pruned = false;

			// OTV: a child about to become the best that overshoots the node's own static eval by OTV_MARGIN is searched again
			// with reductions off for its first OTV_PLIES plies.
			if (Config::ENABLE_OTV && !g_in_verify && g_verify_count < Config::OTV_PATH_CAP && rfp_eval != NO_STATIC_EVAL && score > best_score &&
			    (!Config::OTV_PV_ONLY || (beta - alpha) > 1) && (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING && (score - s * rfp_eval) > Config::OTV_MARGIN)
			{
				VerifyGuard vg(cur_depth, Config::OTV_PLIES);
				g_best_mated = best_score <= -9000000;
				score = score_move<RootSide>(alpha, beta, alpha_orig, beta_orig, (int)i, cur_depth, depth_limit, capture_move, currently_in_check, move,
				                             previous_move, position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search);
			}
			unmake_move(state_history, position_count, zobrist);
			if (time_up.load(std::memory_order_relaxed))
				return 0;
			zobrist = cur_hash;

			if (score > best_score)
			{
				best_score = score;
				best_move = move;
				// The PV is updated on a move that does not fail low in the root frame: above the maximizer's alpha, below the
				// minimizer's beta (its constant bound here).
				if ((RootSide ? score > alpha : score < beta) && !is_in_null_search)
					updatePV(move, cur_depth);
			}
			alpha = std::max(alpha, best_score);
			if (beta <= alpha)
			{
#ifdef SV2_MOVE_PROBE
				{
					const int b = i == 0 ? 0 : i == 1 ? 1 : i == 2 ? 2 : i <= 4 ? 3 : i <= 9 ? 4 : 5;
					(probe_hit ? g_sv2_probe.cut_hit : g_sv2_probe.cut_miss)[b]++;
					if (!probe_hit && (int)i < probe_noisy_lead)
					{
						++g_sv2_probe.miss_cut_in_noisy;
						g_sv2_probe.miss_quiets_unneeded += probe_quiets;
					}
					if (!probe_hit && probe_rem_idx >= 0)
					{
						++g_sv2_probe.rem_miss_cut_any;
						if (move == probe_rem_move)
						{
							++g_sv2_probe.rem_miss_cut;
							g_sv2_probe.rem_miss_cut_at0 += (i == 0);
						}
					}
					probe_slot = Sv2ProbeSlot{probe_key, move.from_square, move.to_square, move.promotion};
				}
#endif
				++g_fh_total;
				if (i == 0)
					++g_fh_first;
				g_cutoff_histogram[i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
				// The cutoff move onto the node's own TT entry (singular's and TT-move ordering's move), not while excluding.
				if ((Config::ENABLE_SINGULAR || Config::ENABLE_TT_MOVE) && !excluding)
				{
					TTEntry *node_entry = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
					if (node_entry != nullptr)
						node_entry->move = move;
				}
				if (i != 0)
				{
					updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);
					if (phase1 != SIZE_MAX)   // a cut on phase 1's captures: the cached list is still capture-only
						markMoveGenCacheQuietsPending(zobrist, current_state.castling_rights, current_state.ep_square);
				}
				if (!capture_move)
					quiet_cutoff_history(current_state, move, previous_move, cur_depth, depth_limit, searched_quiets);
				else if (Config::ENABLE_CAPTURE_HIST)
					capture_cutoff_history(current_state, move, cur_depth, depth_limit, searched_captures);
				store_node_tt(zobrist, current_state, state_history, s * best_score, cur_depth, depth_limit, root_lo<RootSide>(alpha_orig, beta_orig),
				              root_hi<RootSide>(alpha_orig, beta_orig), best_move);
				return best_score;
			}
		}

		// No move improved on the start value: checkmate, stalemate, or every move futility-pruned.
		if (best_score == no_move_score)
		{
			if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn],
			                 current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
			                 current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
				return best_score;
			else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn],
			                      current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
			                      current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
				return 0;
			else if (all_moves_pruned)
				return best_early_eval;
		}

#ifdef SV2_MOVE_PROBE
		(probe_hit ? g_sv2_probe.nocut_hit : g_sv2_probe.nocut_miss)++;
		if (best_score > alpha_orig && best_move.from_square != best_move.to_square)
			probe_slot = Sv2ProbeSlot{probe_key, best_move.from_square, best_move.to_square, best_move.promotion};
#endif
		const bool best_en_passant = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
		const bool best_capture = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], best_en_passant);
		if (!best_capture)
		{
			historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
			if ((Config::ENABLE_CORR_HIST || Config::ENABLE_CORRHIST_LOG || Config::CORR2) && std::abs(best_score) < 9000000 &&
			    !is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings,
			              current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]))
			{
				int se;   // the raw node-entry eval when this node computed one
				if (rfp_raw_eval != NO_STATIC_EVAL)
				{
					se = rfp_raw_eval;
					g_corrhist_eval_reused++;
				}
				else
				{
					se = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
					g_corrhist_eval_computed++;
				}
				const int root_best = s * best_score;
				const int ra = root_lo<RootSide>(alpha_orig, beta_orig), rb = root_hi<RootSide>(alpha_orig, beta_orig);
				if (Config::ENABLE_CORR_HIST)
					corrhist_update(current_state, side_bit, se, root_best);
				if (Config::CORR2)
					corr2_update(current_state, side_bit, se, root_best, depth_limit - cur_depth, ra, rb);
				if (Config::ENABLE_CORRHIST_LOG)
					corrhist_log(current_state, side_bit, se, root_best, depth_limit - cur_depth, ra, rb);
			}
		}
		store_node_tt(zobrist, current_state, state_history, s * best_score, cur_depth, depth_limit, root_lo<RootSide>(alpha_orig, beta_orig),
		              root_hi<RootSide>(alpha_orig, beta_orig), best_move);
		return best_score;
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
#ifdef SV2_MOVE_PROBE
		++g_sv2_probe.scored;
#endif
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
			{
#ifdef SV2_MOVE_PROBE
				++g_sv2_probe.pruned_lmp[probe_rd(rd)];
#endif
				return -MATE;
			}
		}
		// History pruning of a late quiet with very negative butterfly history.
		if (Config::ENABLE_HIST_PRUNE && do_lmr && !prune_gate && rd >= 1 && rd <= Config::HIST_PRUNE_MAX_DEPTH &&
		    historyHeuristics[current_state.turn][move.from_square][move.to_square] < -Config::HIST_PRUNE_COEF * rd)
		{
#ifdef SV2_MOVE_PROBE
			++g_sv2_probe.pruned_hist[probe_rd(rd)];
#endif
			return -MATE;
		}
		// SEE pruning of a quiet whose moved piece the recapture wins.
		if (Config::ENABLE_SEE_PRUNE && do_lmr && !prune_gate && rd >= 1 && rd <= Config::SEE_PRUNE_MAX_DEPTH &&
		    see(move.to_square, updated_state.turn, updated_state) > Config::SEE_PRUNE_MARGIN)
		{
#ifdef SV2_MOVE_PROBE
			++g_sv2_probe.pruned_see[probe_rd(rd)];
#endif
			return -MATE;
		}
		// SEE pruning of a losing capture (a checking capture extends, so it is never pruned).
		if (Config::SEE_PRUNE_CAPTURES && capture_move && !currently_in_check && !extend && !prune_gate && rd >= 1 && rd <= Config::SEE_PRUNE_MAX_DEPTH &&
		    ((Config::SEE_PER_MOVE & 4) ? see_move(move.from_square, move.to_square, current_state.turn, current_state)
		                                : see(move.to_square, current_state.turn, current_state)) < -Config::SEE_PRUNE_CAPTURE_MARGIN)
		{
#ifdef SV2_MOVE_PROBE
			++g_sv2_probe.pruned_seec[probe_rd(rd)];
#endif
			return -MATE;
		}

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
#ifdef SV2_MOVE_PROBE
					++g_sv2_probe.pruned_fut[probe_rd(rd)];
#endif
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
#ifdef SV2_MOVE_PROBE
		++g_sv2_probe.q_lists;
		g_sv2_probe.q_moves += (long long)moves_list.size();
#endif
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
