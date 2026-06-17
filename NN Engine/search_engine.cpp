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
#include <cstdlib>
#include <cstring>

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

// Read a boolean search-ablation toggle from the environment. "0" disables;
// unset or any non-"0" value keeps the supplied default. Used to flip pruning
// mechanisms off at runtime for the eval-vs-search diagnostic without recompiling.
static bool env_flag(const char *name, bool dflt)
{
    const char *v = std::getenv(name);
    return v ? !(v[0] == '0' && v[1] == '\0') : dflt;
}

// Read an integer search-tuning parameter from the environment (e.g. a margin).
static int env_int(const char *name, int dflt)
{
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}

// --- Debug instrumentation (gated by Config::DEBUG_INVARIANTS; default off = byte-identical) --------
// These hunt the move-ordering corruption: SearchData's four parallel arrays must stay equal-length and
// index-corresponding, and every Move handed to make_move must be pseudo-legal. When the flag is off
// both are no-ops. See dev_notes/CRASH_INVESTIGATION_PLAYBOOK.md.

// Log a SearchData length invariant violation. The grouped scores vector makes the old three-way drift
// structurally impossible; the only remaining rule is that the searched-score list never outruns the
// full move list (alpha_beta reads moves_list[i] for i < scores.size()). Cheap: two size() reads.
static inline void dbg_searchdata(const char *where, const SearchData &d)
{
    if (!Config::DEBUG_INVARIANTS)
        return;
    size_t m = d.moves_list.size(), s = d.scores.size();
    if (s > m)
        std::cerr << "[INV] " << where << " scores/moves = " << s << " " << m << std::endl;
}

// Return true (and log) if `move` is NOT pseudo-legal in `st` (square index out of range, or no
// friendly piece on the from-square). Callers skip the move when this fires, so a debug run logs every
// bad move and finishes instead of throwing in update_state.
static inline bool dbg_bad_move(const char *where, int i, const Move &move, const BoardState &st)
{
    if (!Config::DEBUG_INVARIANTS)
        return false;
    bool bad = move.from_square >= 64 || move.to_square >= 64 ||
               !(BB_SQUARES[move.from_square] & st.occupied_colour[st.turn]);
    if (!bad)
        return false;
    // create_fen takes mutable references; copy into locals so we never touch the real board state.
    uint64_t pawns = st.pawns, knights = st.knights, bishops = st.bishops, rooks = st.rooks,
             queens = st.queens, kings = st.kings, occupied = st.occupied,
             ow = st.occupied_colour[true], ob = st.occupied_colour[false], promoted = st.promoted,
             castling = st.castling_rights;
    int ep = st.ep_square;
    std::cerr << "[BADMOVE] " << where << " i=" << i << " move=" << (int)move.from_square << "->"
              << (int)move.to_square << " fen="
              << create_fen(pawns, knights, bishops, rooks, queens, kings, occupied, ow, ob, promoted,
                            castling, ep, st.turn)
              << std::endl;
    return true;
}

// Bound flag for the root / preliminary-ordering TT stores. A fail-soft score is only EXACT when
// it lands strictly inside the search window; at or past a bound it is an UPPER/LOWER bound. The
// root stores historically hardcoded EXACT (sound only under the infinite window); this computes
// the honest flag when Config::HONEST_ROOT_TT is on (required for aspiration windows) and falls
// back to EXACT otherwise so the default build is byte-identical.
static inline TTFlag root_tt_flag(int score, int alpha, int beta)
{
    if (!Config::HONEST_ROOT_TT)
        return TTFlag::EXACT;
    if (score <= alpha)
        return TTFlag::UPPERBOUND;
    if (score >= beta)
        return TTFlag::LOWERBOUND;
    return TTFlag::EXACT;
}

// Number of check/forcing extensions on the current root-to-leaf search path. The search
// is single-threaded depth-first, so a file-scope counter tracks the path correctly; the
// RAII guard keeps the increment/decrement balanced across every return path.
static int g_check_extensions = 0;
struct CheckExtensionGuard
{
    bool active;
    explicit CheckExtensionGuard(bool a) : active(a)
    {
        if (active)
            ++g_check_extensions;
    }
    ~CheckExtensionGuard()
    {
        if (active)
            --g_check_extensions;
    }
};

// Aspiration-window diagnostics (cumulative across the process; printed to stderr per move when
// ASPIRATION_DELTA > 0, so the last line of a single-process suite run = suite totals). g_asp_windows
// counts aspirated iterations; g_asp_fails counts window failures that triggered a widen or the
// full-window fallback; g_asp_fallbacks counts iterations that exhausted MAX_WIDENINGS and dropped to
// the full window. Lets the DELTA sweep read the fail rate directly instead of inferring it.
static long g_asp_windows = 0;
static long g_asp_fails = 0;
static long g_asp_fallbacks = 0;
// Move-ordering quality (observability only; never affects search). Of all beta cutoffs in the main
// alternating-search loops, the fraction landing on the first-ordered move (i==0): strong engines ~90-95%.
// Cumulative across a run, like the aspiration counters above.
static long g_fh_total = 0;
static long g_fh_first = 0;
// Beta-cutoff move-index histogram (diagnostic): buckets 0, 1, 2, 3-7, 8+. Decides whether EBF headroom
// is in pruning (mass at 0-2) or secondary move ordering (mass at 8+). Cumulative across the run.
static long g_cutoff_histogram[5] = {0, 0, 0, 0, 0};
// Passed-pawn LMP/LMR exemption fires (diagnostic): how often an otherwise-reducible advanced pawn push
// was exempted from pruning/reduction. Cumulative across the run; printed beside the histogram.
static long g_passer_exempt_fires = 0;

// ===================== LMR-miss profiler (diagnostic) =====================
// Localizes WHERE late-move reductions drop winning moves. Side-effect-free:
// it only reads the existing reduced-scout result (no shadow searches, so no
// cache writes), and the whole thing is gated on Config::LMR_PROFILE — when off
// the search is byte-identical. A move is "dropped" when its reduced scout
// fails low (score <= alpha): the get_score_* helpers re-search only the open
// interval (alpha, beta), so a fail-low keeps the reduced value and is never
// verified. "margin" = how far below alpha it landed (small = a near-miss).
// Counters accumulate across the whole process (static => zero-initialized);
// the last position's stderr dump is therefore the whole-suite aggregate.

static Move g_profile_bm{255, 255, 255}; // PROFILE_BM target (from+to matched), sentinel = none

namespace
{
    constexpr int LP_LVL = 24; // tree-level (cur_depth) cap
    constexpr int LP_MV = 48;  // move-number cap
    constexpr int LP_IT = 32;  // iterative-deepening iteration (depth_limit) cap
    constexpr int LP_CLOSE_MARGIN = 1000; // a pawn: fail-lows closer than this are near-misses

    struct LmrProfile
    {
        long reductions = 0, faillow = 0, failhigh = 0, faillow_close = 0;
        long red_lvl_mv[LP_LVL][LP_MV] = {};  // denominator by (tree level, move number)
        long low_lvl_mv[LP_LVL][LP_MV] = {};  // fail-low (drop) by (tree level, move number)
        long close_lvl_mv[LP_LVL][LP_MV] = {}; // near-miss subset
        long red_it_lvl[LP_IT][LP_LVL] = {};  // denominator by (ID iteration, tree level)
        long low_it_lvl[LP_IT][LP_LVL] = {};  // fail-low by (ID iteration, tree level)
        long low_hist[4] = {};   // dropped move's history tier: 0:<=0 1:<4k 2:<32k 3:>=32k
        long low_pv = 0, low_nonpv = 0;       // dropped at PV (full-window) vs scout node
        long low_killer = 0;                  // dropped move was a killer/counter
        long low_piece[7] = {};               // dropped move's moving piece type (1=P..6=K)
        long low_phase[4] = {};               // game-phase bucket of the drop
        long bm_dropped = 0;                  // top-level (cur_depth==0) fail-low of the PROFILE_BM move
        long red_redux_sum = 0;               // sum of reduction plies over fail-lows (avg reduction when dropped)
    };
    LmrProfile g_lmr;

    inline int lmr_phase_bucket(const BoardState &s)
    {
        int phase = 4 * __builtin_popcountll(s.queens) + 2 * __builtin_popcountll(s.rooks) + __builtin_popcountll(s.bishops | s.knights);
        int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0..128 (higher = less material)
        if (phase_score <= 24)
            return 0; // opening/heavy
        if (phase_score <= 64)
            return 1; // midgame
        if (phase_score <= 96)
            return 2; // endgame
        return 3;     // deep endgame
    }

    inline int lmr_hist_tier(int h)
    {
        if (h <= 0)
            return 0;
        if (h < 4000)
            return 1;
        if (h < 32000)
            return 2;
        return 3;
    }
}

// Record one reduced move: always counts the reduction (denominator); if the
// reduced scout fails low (dropped) records where/how-close/which-kind.
static inline void lmr_profile_event(int depth_limit, int cur_depth, int move_number,
                                     int alpha, int beta, int score, int reduced_depth,
                                     const Move &move, const BoardState &cs, const Move &prevMove)
{
    int lvl = cur_depth < LP_LVL ? cur_depth : LP_LVL - 1;
    int mv = move_number < LP_MV ? move_number : LP_MV - 1;
    int it = depth_limit < LP_IT ? depth_limit : LP_IT - 1;
    if (lvl < 0)
        lvl = 0;
    if (mv < 0)
        mv = 0;
    if (it < 0)
        it = 0;

    g_lmr.reductions++;
    g_lmr.red_lvl_mv[lvl][mv]++;
    g_lmr.red_it_lvl[it][lvl]++;

    bool research = (score > alpha && score < beta);
    if (research)
        return; // verified at full depth — not a drop

    if (score >= beta)
    {
        g_lmr.failhigh++; // a cutoff, harmless
        return;
    }

    // score <= alpha: fail-low DROP — the case where a win can be lost.
    g_lmr.faillow++;
    g_lmr.low_lvl_mv[lvl][mv]++;
    g_lmr.low_it_lvl[it][lvl]++;
    g_lmr.red_redux_sum += (depth_limit - reduced_depth);

    int margin = alpha - score; // >= 0; small = near-miss
    if (margin < LP_CLOSE_MARGIN)
    {
        g_lmr.faillow_close++;
        g_lmr.close_lvl_mv[lvl][mv]++;
    }

    g_lmr.low_hist[lmr_hist_tier(historyHeuristics[cs.turn][move.from_square][move.to_square])]++;
    if (beta - alpha > 1)
        g_lmr.low_pv++;
    else
        g_lmr.low_nonpv++;
    if (killerMoves[lvl][0] == move || killerMoves[lvl][1] == move ||
        counterMoves[prevMove.from_square][prevMove.to_square] == move)
        g_lmr.low_killer++;

    uint64_t fb = 1ULL << move.from_square;
    int pt = (cs.pawns & fb) ? 1 : (cs.knights & fb) ? 2
                              : (cs.bishops & fb)     ? 3
                              : (cs.rooks & fb)       ? 4
                              : (cs.queens & fb)      ? 5
                                                      : 6;
    g_lmr.low_piece[pt]++;
    g_lmr.low_phase[lmr_phase_bucket(cs)]++;

    if (cur_depth == 0 && move.from_square == g_profile_bm.from_square && move.to_square == g_profile_bm.to_square)
        g_lmr.bm_dropped++;
}

// History-aware LMR adjustment, in plies, SIGNED:
//   > 0  "reduce-less" — a known-good late quiet (killer/counter, or a real history track record,
//        tier >= 2 == history >= 4000) is searched closer to (never beyond) full depth.
//   < 0  "reduce-more" — a never-cut quiet (tier 0 == history exactly 0 == has never produced a beta
//        cutoff) is pruned harder. This is the EBF-saving half (the Stockfish "history further
//        reduction"); gated separately by HISTORY_LMR_MORE_CAP (0 = reduce-less only).
// Coarse categorical signals only (not absolute thresholds), so it is robust to the unbounded/uneven
// history magnitudes. The caller clamps r to [2, depth_limit] and gates the whole thing on
// Config::ENABLE_HISTORY_LMR so the off-path stays byte-identical.
// Non-counting static eval for the improving heuristic: probe the eval cache, else compute
// placement_and_piece_eval directly WITHOUT incrementing the node counter (so ENABLE_IMPROVING does
// not inflate search-node counts). Only called for non-in-check nodes, so checkmate handling is skipped.
inline int static_eval_for_improving(std::vector<BoardState> &state_history, uint64_t zobrist)
{
    int cached;
    if (accessCacheNew(zobrist, cached))
        return cached;
    BoardState cs = state_history.back();
    int moveNum = static_cast<int>(state_history.size());
    int total = placement_and_piece_eval(moveNum, cs.turn, cs.pawns, cs.knights, cs.bishops, cs.rooks,
                                         cs.queens, cs.kings, cs.occupied_colour[true], cs.occupied_colour[false], cs.occupied);
    // Match get_board_evaluation's Config::side_to_play flip so this (miss) path agrees in sign with
    // the cache-hit path above, which returns the already-flipped stored value. Without this the
    // improving comparison mixes flipped and unflipped evals when side_to_play is true.
    if (Config::side_to_play)
        total = -total;
    return total;
}

inline int history_lmr_delta(const Move &move, const Move &previousMove, const BoardState &cs, int ply)
{
    int tier = lmr_hist_tier(historyHeuristics[cs.turn][move.from_square][move.to_square]);

    int reduce_less = 0;
    // A killer at this ply, or the counter to the previous move: an empirically cutoff-causing quiet.
    if (killerMoves[ply][0] == move || killerMoves[ply][1] == move ||
        counterMoves[previousMove.from_square][previousMove.to_square] == move)
        reduce_less += 1;
    // A real history track record (not a one-off).
    if (tier >= 2)
        reduce_less += 1;

    if (reduce_less > 0)
        return std::min(reduce_less, Config::HISTORY_LMR_CAP);

    // No positive signal: a never-cut quiet (history == 0) is reduced more -- UNLESS continuation
    // history says it's a contextually-good reply to previousMove (a strong 1-ply continuation score),
    // in which case we cancel the extra reduction (back to base, never deeper). Env-gated, default off.
    if (tier == 0)
    {
        if (Config::ENABLE_CONT_HIST && previousMove.from_square != previousMove.to_square &&
            counterMoveHeuristics[cs.turn][previousMove.from_square * 64 + previousMove.to_square]
                                 [move.from_square * 64 + move.to_square] >= Config::CONT_HIST_LMR_THRESH)
            return 0;
        // A strong 2-ply continuation (move 2 plies back x this move) also rescues a tier-0 quiet
        // from reduce-more. Gated; reuses the 1-ply threshold.
        if (Config::ENABLE_CONT_HIST_2PLY && ply >= 2)
        {
            Move p2 = g_searchStack[ply - 2];
            if (p2.from_square != p2.to_square &&
                contHist2[cs.turn][p2.from_square * 64 + p2.to_square]
                         [move.from_square * 64 + move.to_square] >= Config::CONT_HIST_LMR_THRESH)
                return 0;
        }
        return -Config::HISTORY_LMR_MORE_CAP;
    }

    return 0;
}

// Dump the cumulative profile to stderr (kept off the captured stdout). Compact;
// read the LAST position's dump for the whole-suite picture.
static void lmr_profile_dump()
{
    std::cerr << "\n===== LMR-miss profile (cumulative) =====\n";
    std::cerr << "reductions=" << g_lmr.reductions
              << "  fail-low(drop)=" << g_lmr.faillow
              << "  near-miss(<1p)=" << g_lmr.faillow_close
              << "  fail-high(cut)=" << g_lmr.failhigh;
    if (g_lmr.reductions)
        std::cerr << "  drop-rate=" << (100.0 * g_lmr.faillow / g_lmr.reductions) << "%";
    if (g_lmr.faillow)
        std::cerr << "  avg-reduction-when-dropped=" << (double)g_lmr.red_redux_sum / g_lmr.faillow << " ply";
    std::cerr << "\n";

    std::cerr << "drop-rate by tree-level (cur_depth): level reductions faillow near rate%\n";
    for (int l = 0; l < LP_LVL; ++l)
    {
        long r = 0, f = 0, c = 0;
        for (int m = 0; m < LP_MV; ++m)
        {
            r += g_lmr.red_lvl_mv[l][m];
            f += g_lmr.low_lvl_mv[l][m];
            c += g_lmr.close_lvl_mv[l][m];
        }
        if (r)
            std::cerr << "  L" << l << ": " << r << " " << f << " " << c
                      << " " << (100.0 * f / r) << "%\n";
    }

    std::cerr << "drop-rate by move-number: movenum reductions faillow near rate%\n";
    for (int m = 0; m < LP_MV; ++m)
    {
        long r = 0, f = 0, c = 0;
        for (int l = 0; l < LP_LVL; ++l)
        {
            r += g_lmr.red_lvl_mv[l][m];
            f += g_lmr.low_lvl_mv[l][m];
            c += g_lmr.close_lvl_mv[l][m];
        }
        if (r)
            std::cerr << "  #" << m << ": " << r << " " << f << " " << c
                      << " " << (100.0 * f / r) << "%\n";
    }

    std::cerr << "fail-low by ID-iteration (depth_limit): iter faillow / reductions\n";
    for (int i = 0; i < LP_IT; ++i)
    {
        long r = 0, f = 0;
        for (int l = 0; l < LP_LVL; ++l)
        {
            r += g_lmr.red_it_lvl[i][l];
            f += g_lmr.low_it_lvl[i][l];
        }
        if (r)
            std::cerr << "  d" << i << ": " << f << " / " << r << "\n";
    }

    std::cerr << "dropped-move history tier:  <=0:" << g_lmr.low_hist[0]
              << "  <4k:" << g_lmr.low_hist[1]
              << "  <32k:" << g_lmr.low_hist[2]
              << "  >=32k:" << g_lmr.low_hist[3] << "\n";
    std::cerr << "dropped at PV-node:" << g_lmr.low_pv << "  scout-node:" << g_lmr.low_nonpv
              << "  killer/counter:" << g_lmr.low_killer << "\n";
    std::cerr << "dropped piece (P/N/B/R/Q/K): " << g_lmr.low_piece[1] << "/" << g_lmr.low_piece[2]
              << "/" << g_lmr.low_piece[3] << "/" << g_lmr.low_piece[4] << "/"
              << g_lmr.low_piece[5] << "/" << g_lmr.low_piece[6] << "\n";
    std::cerr << "dropped phase (open/mid/end/deep): " << g_lmr.low_phase[0] << "/" << g_lmr.low_phase[1]
              << "/" << g_lmr.low_phase[2] << "/" << g_lmr.low_phase[3] << "\n";
    if (g_profile_bm.from_square != 255)
        std::cerr << "PROFILE_BM dropped at top level: " << g_lmr.bm_dropped << " time(s)\n";
    std::cerr << "=========================================\n";
}

void initialize_engine(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn, bool side_to_play)
{

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
        fullmove_number);

    state_history.push_back(initialState);

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_colour[true], initialState.occupied_colour[false], initialState.turn);
    position_count[zobrist]++;

    Config::side_to_play = side_to_play;

    // Load search-ablation toggles from the environment once per process (a fresh
    // ChessAI is built per position in the test harness, so guard against repeating
    // the parse/echo). Echo to stderr so the captured stdout stays clean for parsing.
    static bool toggles_loaded = false;
    if (!toggles_loaded)
    {
        Config::ENABLE_LMR = env_flag("ENABLE_LMR", true);
        Config::ENABLE_FUTILITY = env_flag("ENABLE_FUTILITY", true);
        Config::ENABLE_RAZORING = env_flag("ENABLE_RAZORING", true);
        Config::ENABLE_NULLMOVE = env_flag("ENABLE_NULLMOVE", true);
        Config::NULLMOVE_PROGRESSIVE = env_flag("NULLMOVE_PROGRESSIVE", Config::NULLMOVE_PROGRESSIVE);
        Config::ENABLE_QDELTA = env_flag("ENABLE_QDELTA", true);
        Config::LMR_PROFILE = env_flag("LMR_PROFILE", false);
        Config::PROTECT_KILLERS = env_flag("PROTECT_KILLERS", false);
        Config::PROTECT_PV = env_flag("PROTECT_PV", false);
        // History-aware LMR (default off = byte-identical). CAP = plies removed for good quiets;
        // MORE_CAP = plies added for never-cut quiets (0 = reduce-less only, the prior behavior).
        Config::ENABLE_HISTORY_LMR = env_flag("ENABLE_HISTORY_LMR", Config::ENABLE_HISTORY_LMR);
        Config::HISTORY_LMR_CAP = env_int("HISTORY_LMR_CAP", Config::HISTORY_LMR_CAP);
        Config::HISTORY_LMR_MORE_CAP = env_int("HISTORY_LMR_MORE_CAP", Config::HISTORY_LMR_MORE_CAP);
        Config::ENABLE_LMP = env_flag("ENABLE_LMP", Config::ENABLE_LMP);
        Config::LMP_MAX_DEPTH = env_int("LMP_MAX_DEPTH", Config::LMP_MAX_DEPTH);
        Config::LMP_BASE = env_int("LMP_BASE", Config::LMP_BASE);
        Config::LMP_SCALE = env_int("LMP_SCALE", Config::LMP_SCALE);
        Config::ENABLE_LAZY_RESORT = env_flag("ENABLE_LAZY_RESORT", Config::ENABLE_LAZY_RESORT);
        Config::PROMOTE_TOP_K = env_int("PROMOTE_TOP_K", Config::PROMOTE_TOP_K);
        Config::RESORT_AFTER_REUSES = env_int("RESORT_AFTER_REUSES", Config::RESORT_AFTER_REUSES);
        Config::LAZY_RESORT_MIN_CUTOFF_IDX = env_int("LAZY_RESORT_MIN_CUTOFF_IDX", Config::LAZY_RESORT_MIN_CUTOFF_IDX);
        Config::ENABLE_PASSER_PRUNE_EXEMPT = env_flag("ENABLE_PASSER_PRUNE_EXEMPT", Config::ENABLE_PASSER_PRUNE_EXEMPT);
        Config::PASSER_EXEMPT_ADV = env_int("PASSER_EXEMPT_ADV", Config::PASSER_EXEMPT_ADV);
        Config::ENABLE_MATE_DRIVE_SCALE = env_flag("ENABLE_MATE_DRIVE_SCALE", Config::ENABLE_MATE_DRIVE_SCALE);
        Config::ENABLE_ENDGAME_SCALE = env_flag("ENABLE_ENDGAME_SCALE", Config::ENABLE_ENDGAME_SCALE);
        Config::SCALE_PASSED_PAWN = env_int("SCALE_PASSED_PAWN", Config::SCALE_PASSED_PAWN);
        Config::SCALE_LATENT_THREAT = env_int("SCALE_LATENT_THREAT", Config::SCALE_LATENT_THREAT);
        Config::SCALE_CENTRAL = env_int("SCALE_CENTRAL", Config::SCALE_CENTRAL);
        Config::SCALE_CAPTURE_GAINS = env_int("SCALE_CAPTURE_GAINS", Config::SCALE_CAPTURE_GAINS);
        Config::PP_OPP_PAWN_PEN = env_int("PP_OPP_PAWN_PEN", Config::PP_OPP_PAWN_PEN);
        Config::PP_BLOCKADE_PEN = env_int("PP_BLOCKADE_PEN", Config::PP_BLOCKADE_PEN);
        Config::PP_UNBLOCKED = env_int("PP_UNBLOCKED", Config::PP_UNBLOCKED);
        Config::PP_DIAG_SUPPORT = env_int("PP_DIAG_SUPPORT", Config::PP_DIAG_SUPPORT);
        Config::PP_FILE_CLEAR = env_int("PP_FILE_CLEAR", Config::PP_FILE_CLEAR);
        Config::PP_HORIZ_SUPPORT = env_int("PP_HORIZ_SUPPORT", Config::PP_HORIZ_SUPPORT);
        Config::SCALE_PLACE_PAWN = env_int("SCALE_PLACE_PAWN", Config::SCALE_PLACE_PAWN);
        Config::SCALE_PLACE_KNIGHT = env_int("SCALE_PLACE_KNIGHT", Config::SCALE_PLACE_KNIGHT);
        Config::SCALE_PLACE_BISHOP = env_int("SCALE_PLACE_BISHOP", Config::SCALE_PLACE_BISHOP);
        Config::SCALE_PLACE_QUEEN = env_int("SCALE_PLACE_QUEEN", Config::SCALE_PLACE_QUEEN);
        Config::SCALE_PLACE_KING_EG = env_int("SCALE_PLACE_KING_EG", Config::SCALE_PLACE_KING_EG);
        Config::CENTER_INNER_MULT = env_int("CENTER_INNER_MULT", Config::CENTER_INNER_MULT);
        Config::CENTER_OUTER_MULT = env_int("CENTER_OUTER_MULT", Config::CENTER_OUTER_MULT);
        Config::ENABLE_CHEAP_BISHOP_COMPLEX = env_flag("ENABLE_CHEAP_BISHOP_COMPLEX", Config::ENABLE_CHEAP_BISHOP_COMPLEX);
        Config::CHEAP_BISHOP_BLOCK = env_int("CHEAP_BISHOP_BLOCK", Config::CHEAP_BISHOP_BLOCK);
        Config::CHEAP_BISHOP_MOB = env_int("CHEAP_BISHOP_MOB", Config::CHEAP_BISHOP_MOB);
        Config::CHEAP_BISHOP_FWD = env_int("CHEAP_BISHOP_FWD", Config::CHEAP_BISHOP_FWD);
        Config::CHEAP_BISHOP_KING = env_int("CHEAP_BISHOP_KING", Config::CHEAP_BISHOP_KING);
        Config::ENABLE_CHEAP_ROOK_MOBILITY = env_flag("ENABLE_CHEAP_ROOK_MOBILITY", Config::ENABLE_CHEAP_ROOK_MOBILITY);
        Config::CHEAP_ROOK_MOB = env_int("CHEAP_ROOK_MOB", Config::CHEAP_ROOK_MOB);
        Config::CHEAP_ROOK_FWD = env_int("CHEAP_ROOK_FWD", Config::CHEAP_ROOK_FWD);
        Config::ENABLE_CHEAP_QUEEN_MOBILITY = env_flag("ENABLE_CHEAP_QUEEN_MOBILITY", Config::ENABLE_CHEAP_QUEEN_MOBILITY);
        Config::CHEAP_QUEEN_MOB_MG = env_int("CHEAP_QUEEN_MOB_MG", Config::CHEAP_QUEEN_MOB_MG);
        Config::CHEAP_QUEEN_MOB_EG = env_int("CHEAP_QUEEN_MOB_EG", Config::CHEAP_QUEEN_MOB_EG);
        Config::ENABLE_CHEAP_KNIGHT_MOBILITY = env_flag("ENABLE_CHEAP_KNIGHT_MOBILITY", Config::ENABLE_CHEAP_KNIGHT_MOBILITY);
        Config::CHEAP_KNIGHT_MOB = env_int("CHEAP_KNIGHT_MOB", Config::CHEAP_KNIGHT_MOB);
        Config::ENABLE_ATTACK_LAYER_CACHE = env_flag("ENABLE_ATTACK_LAYER_CACHE", Config::ENABLE_ATTACK_LAYER_CACHE);
        Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME = env_flag("ENABLE_ATTACK_LAYER_CACHE_MIDGAME", Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME);
        Config::ENABLE_SEE_FIX = env_flag("ENABLE_SEE_FIX", Config::ENABLE_SEE_FIX);
        Config::ENABLE_QCHECK_DEPTH0 = env_flag("ENABLE_QCHECK_DEPTH0", Config::ENABLE_QCHECK_DEPTH0);
        Config::ENABLE_QCHECK_MASK = env_flag("ENABLE_QCHECK_MASK", Config::ENABLE_QCHECK_MASK);
        Config::ENABLE_CAPGAIN_PAWN_FIX = env_flag("ENABLE_CAPGAIN_PAWN_FIX", Config::ENABLE_CAPGAIN_PAWN_FIX);
        Config::ENABLE_ROOK_DBLCOUNT_FIX = env_flag("ENABLE_ROOK_DBLCOUNT_FIX", Config::ENABLE_ROOK_DBLCOUNT_FIX);
        Config::ENABLE_KNIGHT_MOB_FIX = env_flag("ENABLE_KNIGHT_MOB_FIX", Config::ENABLE_KNIGHT_MOB_FIX);
        Config::ENABLE_KNIGHT_MOB_SYM_UP = env_flag("ENABLE_KNIGHT_MOB_SYM_UP", Config::ENABLE_KNIGHT_MOB_SYM_UP);
        Config::ENABLE_ROOK_DBLCOUNT_SYM_UP = env_flag("ENABLE_ROOK_DBLCOUNT_SYM_UP", Config::ENABLE_ROOK_DBLCOUNT_SYM_UP);
        Config::ENABLE_ROOK_ENDGAME_CAP = env_flag("ENABLE_ROOK_ENDGAME_CAP", Config::ENABLE_ROOK_ENDGAME_CAP);
        Config::ROOK_ENDGAME_CAP = env_int("ROOK_ENDGAME_CAP", Config::ROOK_ENDGAME_CAP);
        Config::ENABLE_ROOK_RANKWIN_FIX = env_flag("ENABLE_ROOK_RANKWIN_FIX", Config::ENABLE_ROOK_RANKWIN_FIX);
        Config::ENABLE_QPREC_PHASE_GATE = env_flag("ENABLE_QPREC_PHASE_GATE", Config::ENABLE_QPREC_PHASE_GATE);
        Config::ENABLE_TT_DEPTH_FIX = env_flag("ENABLE_TT_DEPTH_FIX", Config::ENABLE_TT_DEPTH_FIX);
        Config::NULLMOVE_CURDEPTH_MINI = env_int("NULLMOVE_CURDEPTH_MINI", Config::NULLMOVE_CURDEPTH_MINI);
        Config::NULLMOVE_CURDEPTH_MAXI = env_int("NULLMOVE_CURDEPTH_MAXI", Config::NULLMOVE_CURDEPTH_MAXI);
        Config::ENABLE_CONT_HIST = env_flag("ENABLE_CONT_HIST", Config::ENABLE_CONT_HIST);
        Config::CONT_HIST_LMR_THRESH = env_int("CONT_HIST_LMR_THRESH", Config::CONT_HIST_LMR_THRESH);
        Config::ENABLE_CONT_HIST_2PLY = env_flag("ENABLE_CONT_HIST_2PLY", Config::ENABLE_CONT_HIST_2PLY);
        Config::ENABLE_CAPTURE_HIST = env_flag("ENABLE_CAPTURE_HIST", Config::ENABLE_CAPTURE_HIST);
        Config::ENABLE_CHECK_ORDER = env_flag("ENABLE_CHECK_ORDER", Config::ENABLE_CHECK_ORDER);
        Config::CHECK_ORDER_BONUS = env_int("CHECK_ORDER_BONUS", Config::CHECK_ORDER_BONUS);
        Config::ENABLE_HISTORY_SATURATION = env_flag("ENABLE_HISTORY_SATURATION", Config::ENABLE_HISTORY_SATURATION);
        Config::ENABLE_HISTORY_MALUS = env_flag("ENABLE_HISTORY_MALUS", Config::ENABLE_HISTORY_MALUS);
        Config::ENABLE_IMPROVING = env_flag("ENABLE_IMPROVING", Config::ENABLE_IMPROVING);
        Config::IMPROVING_EVAL_WINDOW = env_int("IMPROVING_EVAL_WINDOW", Config::IMPROVING_EVAL_WINDOW);
        Config::MAX_HISTORY = env_int("MAX_HISTORY", Config::MAX_HISTORY);
        Config::CONT2_GRAVITY_DIV = env_int("CONT2_GRAVITY_DIV", Config::CONT2_GRAVITY_DIV);
        if (Config::CONT2_GRAVITY_DIV < 1) Config::CONT2_GRAVITY_DIV = 1;
        Config::ENABLE_HISTORY_DECAY = env_flag("ENABLE_HISTORY_DECAY", Config::ENABLE_HISTORY_DECAY);
        Config::MALUS_DIV = env_int("MALUS_DIV", Config::MALUS_DIV);
        if (Config::MALUS_DIV < 1) Config::MALUS_DIV = 1;
        // Fall back to the header defaults (the blitz-validated VERIFY keeper) so the
        // built-in value is the single source of truth; an env var still overrides it
        // (e.g. VERIFY_MARGIN=0 to recover the old search for the d10 control).
        Config::VERIFY_MARGIN = env_int("VERIFY_MARGIN", Config::VERIFY_MARGIN);
        Config::VERIFY_RESEARCH_REDUCTION = env_int("VERIFY_RESEARCH_REDUCTION", Config::VERIFY_RESEARCH_REDUCTION);
        // Iterative-deepening depth cap; default 64 is normal play (a preset governs
        // the depth reached). The cap is literal: MAX_DEPTH=10 searches to depth 10.
        Config::MAX_ITERATIVE_DEPTH = env_int("MAX_DEPTH", Config::MAX_ITERATIVE_DEPTH);
        // In-search repetition-draw threshold (default 2 = first repetition on the path).
        Config::REPETITION_THRESHOLD = env_int("REPETITION_THRESHOLD", Config::REPETITION_THRESHOLD);
        // Check/forcing extension depth (per-path cap); 0 = off.
        Config::CHECK_EXTENSION = env_int("CHECK_EXTENSION", Config::CHECK_EXTENSION);
        // SEE filter on the check extension; default disabled (extend all checks).
        Config::SEE_EXTEND_MARGIN = env_int("SEE_EXTEND_MARGIN", Config::SEE_EXTEND_MARGIN);
        // Honest root/preliminary TT bound flags (default off = hardcoded EXACT); required for
        // sound aspiration windows.
        Config::HONEST_ROOT_TT = env_flag("HONEST_ROOT_TT", Config::HONEST_ROOT_TT);
        // Aspiration windows: initial half-width (0 = off), first depth, resize count and growth %.
        Config::ASPIRATION_DELTA = env_int("ASPIRATION_DELTA", Config::ASPIRATION_DELTA);
        Config::ASPIRATION_MIN_DEPTH = env_int("ASPIRATION_MIN_DEPTH", Config::ASPIRATION_MIN_DEPTH);
        Config::ASPIRATION_MAX_WIDENINGS = env_int("ASPIRATION_MAX_WIDENINGS", Config::ASPIRATION_MAX_WIDENINGS);
        Config::ASPIRATION_WIDEN_PCT = env_int("ASPIRATION_WIDEN_PCT", Config::ASPIRATION_WIDEN_PCT);
        // Main TT associativity (1 = direct-mapped = byte-identical); must be a power of two that
        // divides TT_CACHE_SIZE, so clamp anything else back to direct-mapped.
        Config::TT_WAYS = env_int("TT_WAYS", Config::TT_WAYS);
        if (Config::TT_WAYS != 1 && Config::TT_WAYS != 2 && Config::TT_WAYS != 4 && Config::TT_WAYS != 8)
            Config::TT_WAYS = 1;
        // Debug-only: validate the SearchData parallel-array invariant + move legality and LOG (not
        // crash) on violation. Default off = byte-identical; used only to hunt the move-ordering
        // corruption (see dev_notes/CRASH_INVESTIGATION_PLAYBOOK.md).
        Config::DEBUG_INVARIANTS = env_flag("CHESS_DEBUG_INVARIANTS", Config::DEBUG_INVARIANTS);
        // Select the time-control preset at runtime; absent ⇒ keep the compile-time
        // default (LONG_FORMAT), so the default build is unchanged.
        const char *preset = std::getenv("PRESET");
        if (preset)
        {
            if (std::strcmp(preset, "LIGHTNING") == 0)
                Config::ACTIVE = &Configs::LIGHTNING;
            else if (std::strcmp(preset, "BLITZ") == 0)
                Config::ACTIVE = &Configs::BLITZ;
            else if (std::strcmp(preset, "STANDARD") == 0)
                Config::ACTIVE = &Configs::STANDARD;
            else if (std::strcmp(preset, "LONG_FORMAT") == 0)
                Config::ACTIVE = &Configs::LONG_FORMAT;
        }
        // Optional: a known best move (UCI) to flag if LMR drops it at the top level.
        const char *bm = std::getenv("PROFILE_BM");
        if (bm && bm[0] >= 'a' && bm[0] <= 'h' && bm[1] >= '1' && bm[1] <= '8' &&
            bm[2] >= 'a' && bm[2] <= 'h' && bm[3] >= '1' && bm[3] <= '8')
        {
            g_profile_bm.from_square = (uint8_t)((bm[1] - '1') * 8 + (bm[0] - 'a'));
            g_profile_bm.to_square = (uint8_t)((bm[3] - '1') * 8 + (bm[2] - 'a'));
        }
        toggles_loaded = true;
        const char *active_preset =
            Config::ACTIVE == &Configs::LIGHTNING   ? "LIGHTNING"
            : Config::ACTIVE == &Configs::BLITZ     ? "BLITZ"
            : Config::ACTIVE == &Configs::STANDARD  ? "STANDARD"
            : Config::ACTIVE == &Configs::LONG_FORMAT ? "LONG_FORMAT"
                                                      : "custom";
        std::cerr << "[toggles] LMR=" << Config::ENABLE_LMR
                  << " FUTILITY=" << Config::ENABLE_FUTILITY
                  << " RAZORING=" << Config::ENABLE_RAZORING
                  << " NULLMOVE=" << Config::ENABLE_NULLMOVE
                  << " NULLMOVE_PROGRESSIVE=" << Config::NULLMOVE_PROGRESSIVE
                  << " QDELTA=" << Config::ENABLE_QDELTA
                  << " LMR_PROFILE=" << Config::LMR_PROFILE
                  << " PROTECT_KILLERS=" << Config::PROTECT_KILLERS
                  << " PROTECT_PV=" << Config::PROTECT_PV
                  << " ENABLE_HISTORY_LMR=" << Config::ENABLE_HISTORY_LMR
                  << " HISTORY_LMR_CAP=" << Config::HISTORY_LMR_CAP
                  << " HISTORY_LMR_MORE_CAP=" << Config::HISTORY_LMR_MORE_CAP
                  << " ENABLE_LMP=" << Config::ENABLE_LMP
                  << " LMP_MAX_DEPTH=" << Config::LMP_MAX_DEPTH
                  << " LMP_BASE=" << Config::LMP_BASE
                  << " LMP_SCALE=" << Config::LMP_SCALE
                  << " ENABLE_LAZY_RESORT=" << Config::ENABLE_LAZY_RESORT
                  << " PROMOTE_TOP_K=" << Config::PROMOTE_TOP_K
                  << " RESORT_AFTER_REUSES=" << Config::RESORT_AFTER_REUSES
                  << " LAZY_RESORT_MIN_CUTOFF_IDX=" << Config::LAZY_RESORT_MIN_CUTOFF_IDX
                  << " ENABLE_PASSER_PRUNE_EXEMPT=" << Config::ENABLE_PASSER_PRUNE_EXEMPT
                  << " PASSER_EXEMPT_ADV=" << Config::PASSER_EXEMPT_ADV
                  << " ENABLE_MATE_DRIVE_SCALE=" << Config::ENABLE_MATE_DRIVE_SCALE
                  << " ENABLE_ENDGAME_SCALE=" << Config::ENABLE_ENDGAME_SCALE
                  << " SCALE_PASSED_PAWN=" << Config::SCALE_PASSED_PAWN
                  << " SCALE_LATENT_THREAT=" << Config::SCALE_LATENT_THREAT
                  << " SCALE_CENTRAL=" << Config::SCALE_CENTRAL
                  << " SCALE_CAPTURE_GAINS=" << Config::SCALE_CAPTURE_GAINS
                  << " PP_OPP_PAWN_PEN=" << Config::PP_OPP_PAWN_PEN
                  << " PP_BLOCKADE_PEN=" << Config::PP_BLOCKADE_PEN
                  << " PP_UNBLOCKED=" << Config::PP_UNBLOCKED
                  << " PP_DIAG_SUPPORT=" << Config::PP_DIAG_SUPPORT
                  << " PP_FILE_CLEAR=" << Config::PP_FILE_CLEAR
                  << " PP_HORIZ_SUPPORT=" << Config::PP_HORIZ_SUPPORT
                  << " SCALE_PLACE_PAWN=" << Config::SCALE_PLACE_PAWN
                  << " SCALE_PLACE_KNIGHT=" << Config::SCALE_PLACE_KNIGHT
                  << " SCALE_PLACE_BISHOP=" << Config::SCALE_PLACE_BISHOP
                  << " SCALE_PLACE_QUEEN=" << Config::SCALE_PLACE_QUEEN
                  << " SCALE_PLACE_KING_EG=" << Config::SCALE_PLACE_KING_EG
                  << " CENTER_INNER_MULT=" << Config::CENTER_INNER_MULT
                  << " CENTER_OUTER_MULT=" << Config::CENTER_OUTER_MULT
                  << " ENABLE_CHEAP_BISHOP_COMPLEX=" << Config::ENABLE_CHEAP_BISHOP_COMPLEX
                  << " CHEAP_BISHOP_BLOCK=" << Config::CHEAP_BISHOP_BLOCK
                  << " CHEAP_BISHOP_MOB=" << Config::CHEAP_BISHOP_MOB
                  << " CHEAP_BISHOP_FWD=" << Config::CHEAP_BISHOP_FWD
                  << " CHEAP_BISHOP_KING=" << Config::CHEAP_BISHOP_KING
                  << " ENABLE_CHEAP_ROOK_MOBILITY=" << Config::ENABLE_CHEAP_ROOK_MOBILITY
                  << " CHEAP_ROOK_MOB=" << Config::CHEAP_ROOK_MOB
                  << " CHEAP_ROOK_FWD=" << Config::CHEAP_ROOK_FWD
                  << " ENABLE_CHEAP_QUEEN_MOBILITY=" << Config::ENABLE_CHEAP_QUEEN_MOBILITY
                  << " CHEAP_QUEEN_MOB_MG=" << Config::CHEAP_QUEEN_MOB_MG
                  << " CHEAP_QUEEN_MOB_EG=" << Config::CHEAP_QUEEN_MOB_EG
                  << " ENABLE_CHEAP_KNIGHT_MOBILITY=" << Config::ENABLE_CHEAP_KNIGHT_MOBILITY
                  << " CHEAP_KNIGHT_MOB=" << Config::CHEAP_KNIGHT_MOB
                  << " ENABLE_ATTACK_LAYER_CACHE=" << Config::ENABLE_ATTACK_LAYER_CACHE
                  << " ENABLE_ATTACK_LAYER_CACHE_MIDGAME=" << Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME
                  << " ENABLE_SEE_FIX=" << Config::ENABLE_SEE_FIX
                  << " ENABLE_CAPGAIN_PAWN_FIX=" << Config::ENABLE_CAPGAIN_PAWN_FIX
                  << " ENABLE_ROOK_DBLCOUNT_FIX=" << Config::ENABLE_ROOK_DBLCOUNT_FIX
                  << " ENABLE_KNIGHT_MOB_FIX=" << Config::ENABLE_KNIGHT_MOB_FIX
                  << " ENABLE_KNIGHT_MOB_SYM_UP=" << Config::ENABLE_KNIGHT_MOB_SYM_UP
                  << " ENABLE_ROOK_DBLCOUNT_SYM_UP=" << Config::ENABLE_ROOK_DBLCOUNT_SYM_UP
                  << " ENABLE_ROOK_ENDGAME_CAP=" << Config::ENABLE_ROOK_ENDGAME_CAP
                  << " ROOK_ENDGAME_CAP=" << Config::ROOK_ENDGAME_CAP
                  << " ENABLE_ROOK_RANKWIN_FIX=" << Config::ENABLE_ROOK_RANKWIN_FIX
                  << " ENABLE_QPREC_PHASE_GATE=" << Config::ENABLE_QPREC_PHASE_GATE
                  << " ENABLE_TT_DEPTH_FIX=" << Config::ENABLE_TT_DEPTH_FIX
                  << " NULLMOVE_CURDEPTH_MINI=" << Config::NULLMOVE_CURDEPTH_MINI
                  << " NULLMOVE_CURDEPTH_MAXI=" << Config::NULLMOVE_CURDEPTH_MAXI
                  << " ENABLE_QCHECK_DEPTH0=" << Config::ENABLE_QCHECK_DEPTH0
                  << " ENABLE_QCHECK_MASK=" << Config::ENABLE_QCHECK_MASK
                  << " ENABLE_CONT_HIST=" << Config::ENABLE_CONT_HIST
                  << " CONT_HIST_LMR_THRESH=" << Config::CONT_HIST_LMR_THRESH
                  << " ENABLE_CONT_HIST_2PLY=" << Config::ENABLE_CONT_HIST_2PLY
                  << " ENABLE_CAPTURE_HIST=" << Config::ENABLE_CAPTURE_HIST
                  << " ENABLE_CHECK_ORDER=" << Config::ENABLE_CHECK_ORDER
                  << " ENABLE_HISTORY_SATURATION=" << Config::ENABLE_HISTORY_SATURATION
                  << " ENABLE_HISTORY_MALUS=" << Config::ENABLE_HISTORY_MALUS
                  << " ENABLE_IMPROVING=" << Config::ENABLE_IMPROVING
                  << " IMPROVING_EVAL_WINDOW=" << Config::IMPROVING_EVAL_WINDOW
                  << " MAX_HISTORY=" << Config::MAX_HISTORY
                  << " CONT2_GRAVITY_DIV=" << Config::CONT2_GRAVITY_DIV
                  << " ENABLE_HISTORY_DECAY=" << Config::ENABLE_HISTORY_DECAY
                  << " MALUS_DIV=" << Config::MALUS_DIV
                  << " VERIFY_MARGIN=" << Config::VERIFY_MARGIN
                  << " VERIFY_RESEARCH_REDUCTION=" << Config::VERIFY_RESEARCH_REDUCTION
                  << " PRESET=" << active_preset
                  << " MAX_DEPTH=" << Config::MAX_ITERATIVE_DEPTH
                  << " REPETITION_THRESHOLD=" << Config::REPETITION_THRESHOLD
                  << " CHECK_EXTENSION=" << Config::CHECK_EXTENSION
                  << " SEE_EXTEND_MARGIN=" << Config::SEE_EXTEND_MARGIN
                  << " HONEST_ROOT_TT=" << Config::HONEST_ROOT_TT
                  << " ASPIRATION_DELTA=" << Config::ASPIRATION_DELTA
                  << " ASPIRATION_MIN_DEPTH=" << Config::ASPIRATION_MIN_DEPTH
                  << " ASPIRATION_MAX_WIDENINGS=" << Config::ASPIRATION_MAX_WIDENINGS
                  << " ASPIRATION_WIDEN_PCT=" << Config::ASPIRATION_WIDEN_PCT
                  << " TT_WAYS=" << Config::TT_WAYS
                  << " DEBUG_INVARIANTS=" << Config::DEBUG_INVARIANTS << std::endl;
    }
}

void set_current_state(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn)
{

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
        fullmove_number);

    state_history.push_back(initialState);

    uint64_t zobrist = generateZobristHash(initialState.pawns, initialState.knights, initialState.bishops, initialState.rooks, initialState.queens, initialState.kings, initialState.occupied_colour[true], initialState.occupied_colour[false], initialState.turn);
    position_count[zobrist]++;
}

inline void make_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, Move move, uint64_t zobrist, bool capture_move)
{

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
        turn);

    /* std::cout << occupied << " | " << occupied_white<< " | " << occupied_black << std::endl;
    std::cout << "AFTER: "<< std::endl; */
    // halfmove_clock += 1
    //  Reset the halfmove clock if the move is a pawn move or capture
    if (capture_move || (BB_SQUARES[move.from_square] & pawns))
    {
        halfmove_clock = 0;
    }
    else
    {
        halfmove_clock += 1;
    }

    if (!turn)
        fullmove_number += 1;

    // ep_square = -1;
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
        fullmove_number);

    position_count[zobrist]++;
    state_history.push_back(newState);
}

inline void unmake_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key)
{
    state_history.pop_back();

    if (--position_count[zobrist_key] == 0)
    {
        position_count.erase(zobrist_key); // Clean up to save space
    }
}

inline void update_cache(int num_plies)
{

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

MoveData get_engine_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count)
{

    update_cache(static_cast<int>(state_history.size()));
    std::fill(&killerMoves[0][0], &killerMoves[0][0] + 64 * 2, Move{});
    std::fill(&counterMoves[0][0], &counterMoves[0][0] + 64 * 64, Move{});
    std::fill(&g_searchStack[0], &g_searchStack[0] + MAX_PLY, Move{});
    std::fill(&g_evalStack[0], &g_evalStack[0] + MAX_PLY, NO_STATIC_EVAL);
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

    if (phase_score > 117)
    {
        Config::DECAY_INTERVAL = 200000;
        use_q_precautions = true;
    }
    else if (phase_score >= 96)
    {
        Config::DECAY_INTERVAL = 150000;
        use_q_precautions = true;
    }
    else if (phase_score >= 64)
    {
        Config::DECAY_INTERVAL = 125000;
    }
    if (!Config::ENABLE_QPREC_PHASE_GATE) use_q_precautions = true;
    uint64_t zobrist = generateZobristHash(current.pawns, current.knights, current.bishops, current.rooks, current.queens, current.kings, current.occupied_colour[true], current.occupied_colour[false], current.turn);

    Move move(0, 0, 0);

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

    // Best move/score from the last FULLY searched iteration. A deeper iteration that the
    // clock cuts short must not replace it: a timed-out iteration falls back to the head of
    // the move list, which is stale (and can be a refuted move) when a prior iteration
    // changed its mind. Refreshed only on a completed iteration below.
    Move completed_move = move;
    int completed_score = score;

    // The depth-cap guard is checked first so it short-circuits before the MOVE_TIMES[depth_limit]
    // read: the cap is now literal (search up to depth_limit == MAX_ITERATIVE_DEPTH), and at the
    // boundary depth_limit can equal the cap, where reading MOVE_TIMES[cap] would be out of bounds.
    while (depth_limit < Config::MAX_ITERATIVE_DEPTH && elapsed <= Config::ACTIVE->MOVE_TIMES[depth_limit] && score < 9000000 && !time_up)
    {

        if (is_draw && score == 0)
            break;
        is_draw = false;
        int x1 = (move.from_square & 7) + 1;
        int y1 = (move.from_square >> 3) + 1;

        int x2 = (move.to_square & 7) + 1;
        int y2 = (move.to_square >> 3) + 1;

        // Resigns
        if (score <= -15000)
        {
            MoveData defaultMove(-1, -1, -1, -1, -1, score, 0);
            return defaultMove;
        }

        depth_limit++;

        std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
        std::cout << std::endl;
        std::cout << "SEARCHING DEPTH: " << depth_limit << std::endl;

        t0 = Clock::now();

        if (depth_limit >= 13 && (depth_limit - 3) % 5 == 0)
            decayMoveFrequency();

        // Aspiration windows: from ASPIRATION_MIN_DEPTH up, search a narrow window centred on the
        // previous iteration's score and widen only the failing side (x ASPIRATION_WIDEN_PCT each
        // resize) up to ASPIRATION_MAX_WIDENINGS times before falling back to the full window. This
        // wins cutoffs/depth but is only sound with honest root TT bounds (HONEST_ROOT_TT). A near-
        // mate previous score is skipped (the next score can jump far). DELTA 0 = off: the single
        // full-window call in the else branch is byte-identical to the old behavior.
        int prev_score = score;
        if (Config::ASPIRATION_DELTA > 0 && depth_limit >= Config::ASPIRATION_MIN_DEPTH && prev_score < 9000000 && prev_score > -9000000)
        {
            ++g_asp_windows;
            int lo = Config::ASPIRATION_DELTA;
            int hi = Config::ASPIRATION_DELTA;
            int alpha_win = prev_score - lo;
            int beta_win = prev_score + hi;
            for (int attempt = 0;; ++attempt)
            {
                score = alpha_beta(alpha_win, beta_win, 0, depth_limit, state_history, position_count, zobrist, search_start_time, preliminary_search_data, move, num_iterations);

                // A timed-out search is unresolved; the completed_move guard below refuses it.
                if (time_up.load(std::memory_order_relaxed))
                    break;

                // Resolved strictly inside the window: the score is exact, accept it.
                if (alpha_win < score && score < beta_win)
                    break;

                // Reached only when the window failed (not in-window, not timed out).
                ++g_asp_fails;

                // Out of resizes: take the full window once, which always resolves.
                if (attempt >= Config::ASPIRATION_MAX_WIDENINGS)
                {
                    ++g_asp_fallbacks;
                    alpha_win = -9999998;
                    beta_win = 9999999;
                    score = alpha_beta(alpha_win, beta_win, 0, depth_limit, state_history, position_count, zobrist, search_start_time, preliminary_search_data, move, num_iterations);
                    break;
                }

                // Widen only the side that failed; the other stays tight to keep its cutoffs.
                if (score <= alpha_win)
                {
                    lo = lo * Config::ASPIRATION_WIDEN_PCT / 100;
                    alpha_win = prev_score - lo;
                }
                else
                {
                    hi = hi * Config::ASPIRATION_WIDEN_PCT / 100;
                    beta_win = prev_score + hi;
                }
            }
        }
        else
        {
            score = alpha_beta(alpha, beta, 0, depth_limit, state_history, position_count, zobrist, search_start_time, preliminary_search_data, move, num_iterations);
        }

        // Only adopt this iteration's result if it ran to completion; a timed-out
        // iteration's move/score are unreliable (see completed_move above).
        if (!time_up.load(std::memory_order_relaxed))
        {
            completed_move = move;
            completed_score = score;
        }

        bool side = current.turn;
        for (int i = 0; i < pv_length[0]; i++)
        {

            Move move = pv_table[0][i];

            int bonus = (((depth_limit - 1) * (depth_limit - 1)) * (pv_length[0] - i) * (pv_length[0] - i));
            moveFrequency[side][move.from_square][move.to_square] += bonus;

            if (depth_limit >= 9)
            {
                std::cout
                    << "(" << ((move.from_square & 7) + 1) << ","
                    << ((move.from_square >> 3) + 1) << ") -> ("
                    << ((move.to_square & 7) + 1) << ","
                    << ((move.to_square >> 3) + 1) << ") | ";
            }
            side = !side;
        }
        if (depth_limit >= 9)
        {
            std::cout << std::endl;
        }

        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "ELAPSED: " << elapsed << std::endl;
    }

    // Commit the last fully-searched iteration's choice, never a move from an iteration
    // the clock cut short mid-reorder.
    move = completed_move;
    score = completed_score;

    // Cumulative aspiration diagnostics; the last line of a single-process suite run = suite totals.
    if (Config::ASPIRATION_DELTA > 0)
        std::cerr << "[aspiration] windows=" << g_asp_windows
                  << " fails=" << g_asp_fails
                  << " fallbacks=" << g_asp_fallbacks << std::endl;

    // Move-ordering quality (cumulative first-move-cutoff rate) + this-move effective branching factor.
    if (g_fh_total > 0)
        std::cerr << "[search] first_move_cutoff=" << (100.0 * g_fh_first / g_fh_total) << "% ("
                  << g_fh_first << "/" << g_fh_total << ")  ebf="
                  << ((depth_limit > 0 && num_iterations > 0) ? std::pow((double)num_iterations, 1.0 / depth_limit) : 0.0)
                  << " (nodes=" << num_iterations << " d=" << depth_limit << ")" << std::endl;
    if (g_fh_total > 0)
        std::cerr << "[cutoff_histogram] m0=" << g_cutoff_histogram[0] << " m1=" << g_cutoff_histogram[1]
                  << " m2=" << g_cutoff_histogram[2] << " m3-7=" << g_cutoff_histogram[3]
                  << " m8+=" << g_cutoff_histogram[4] << std::endl;
        std::cerr << "[passer_exempt] fires=" << g_passer_exempt_fires << std::endl;

    int x1 = (move.from_square & 7) + 1;
    int y1 = (move.from_square >> 3) + 1;

    int x2 = (move.to_square & 7) + 1;
    int y2 = (move.to_square >> 3) + 1;

    std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
    std::cout << std::endl;

    MoveData chosenMove(x1, y1, x2, y2, move.promotion, score, num_iterations);

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
        move.promotion);

    make_move(state_history, position_count, move, zobrist, capture_move);

    std::cout << "EVAL CACHE VISITS: " << eval_visits << std::endl;
    std::cout << "EVAL CACHE HITS: " << eval_cache_hits << std::endl;

    std::cout << "MOVE GEN CACHE VISITS: " << move_gen_visits << std::endl;
    std::cout << "MOVE GEN CACHE HITS: " << move_gen_cache_hits << std::endl;

    std::cout << "TT VISITS: " << tt_visits << std::endl;
    std::cout << "TT PROBES: " << tt_probes << std::endl;
    std::cout << "TT HITS: " << tt_hits << std::endl;

    std::cout << "Q SEARCH VISITS: " << qsearchVisits << std::endl;

    if (Config::LMR_PROFILE)
        lmr_profile_dump();

    return chosenMove;
}

int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, SearchData &previous_search_data, Move &best_move, int &num_iterations)
{

    int best_score = -99999999;
    int score;

    BoardState current_state = state_history.back();

    uint64_t cur_hash = zobrist;

    std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move{});
    std::fill(pv_length, pv_length + MAX_PLY, 0);

    SearchData current_search_data = reorder_legal_moves(alpha, beta, depth_limit, t0, zobrist, previous_search_data, state_history, position_count, num_iterations);

    std::fill(&pv_table[0][0], &pv_table[0][0] + MAX_PLY * MAX_PLY, Move{});
    std::fill(pv_length, pv_length + MAX_PLY, 0);

    if (time_up.load(std::memory_order_relaxed))
    {
        std::cout << "TIME LIMIT EXCEEDED" << std::endl;
        best_move = current_search_data.moves_list[0];
        best_score = current_search_data.scores[0].top_score;
        return best_score;
    }
    int razor_threshold;
    if (previous_search_data.moves_list.empty())
    {
        razor_threshold = std::max(static_cast<int>(750 * std::pow(0.75, depth_limit - 4)), 200);
    }
    else
    {
        razor_threshold = std::max(static_cast<int>(300 * std::pow(0.75, depth_limit - 4)), 100);
    }
    // std::cout << "BBB" << std::endl;

    previous_search_data.moves_list.clear();
    previous_search_data.scores.clear();

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

    if (depth_limit >= 10)
    {
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
        current_search_data.moves_list[0].promotion);

    // std::cout <<"BBB" << std::endl;
    make_move(state_history, position_count, current_search_data.moves_list[0], zobrist, capture_move);
    /* if (depth_limit >= 24) {
        std::cout << "BBB: " << num_legal_moves << std::endl;
    } */
    // std::cout <<"CCC"<< std::endl;
    RootScore entry;
    score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, current_search_data.scores[0].second_scores, current_search_data.scores[0].second_moves, entry, state_history, position_count, zobrist, current_search_data.moves_list[0], num_iterations, capture_move, false, false);

    // std::cout <<"DDDD"<< std::endl;
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
    // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
    addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit, root_tt_flag(score, alpha, beta), alpha, beta /* , line */, updated_state.castling_rights, updated_state.ep_square);
    unmake_move(state_history, position_count, zobrist);
    // std::cout <<"CC" << std::endl;
    /* if (depth_limit >= 24) {
        std::cout << "DDDD: " << num_legal_moves << std::endl;
    } */
    zobrist = cur_hash;

    if (depth_limit >= 10)
    {
        std::cout << 0 << " "
                  << score << " "
                  << current_search_data.scores[0].top_score << " "
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

    if (time_up.load(std::memory_order_relaxed))
    {
        std::cout << "TIME LIMIT EXCEEDED" << std::endl;
        best_move = current_search_data.moves_list[0];
        best_score = current_search_data.scores[0].top_score;
        return best_score;
    }

    if (alpha - current_search_data.scores[0].top_score > razor_threshold)
        razor_threshold += alpha - current_search_data.scores[0].top_score;

    previous_search_data.moves_list = current_search_data.moves_list;
    entry.top_score = score;
    previous_search_data.scores.push_back(std::move(entry));

    if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        return score;

    for (size_t i = 1; i < current_search_data.moves_list.size(); ++i)
    {
        Move &move = current_search_data.moves_list[i];

        // Razoring
        if (i < current_search_data.scores.size())
        {
            int score_diff = alpha - current_search_data.scores[i].top_score;
            // int best_diff = current_search_data.scores[0].top_score - current_search_data.scores[i].top_score;

            if (Config::ENABLE_RAZORING && (score_diff > razor_threshold) && (alpha < 9000000))
            {
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
            move.promotion);

        // DEBUG (log-only): confirm the PARENT move is legal here. If this stays silent while
        // minimizer (searching second_level_moves_list[i] just below) logs a [BADMOVE], the corruption
        // is the second_level list losing correspondence with moves_list, not moves_list itself.
        dbg_bad_move("alpha_beta_parent", (int)i, move, current_state);
        make_move(state_history, position_count, move, zobrist, capture_move);
        // std::cout <<"EEE" << std::endl;
        RootScore entry;
        score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, current_search_data.scores[i].second_scores, current_search_data.scores[i].second_moves, entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);

        // std::cout <<"FFF" << std::endl;
        //  If the score is within the window, re-search with full window. Discard the scout's entry first
        //  (this replaces the old second_level pop_backs) so the kept entry reflects the full-window search.
        if (alpha < score && score < beta)
        {
            entry = RootScore{};
            score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, current_search_data.scores[i].second_scores, current_search_data.scores[i].second_moves, entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
        }

        /* if(is_repetition(position_count, zobrist, 2)){
            repetition_flag = true;
            repetition_move = move;
            repetition_score = score;
            repetition_index = i;
            score = -100000001;
        } */
        updated_state = state_history.back();
        // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit, root_tt_flag(score, alpha, beta), alpha, beta /* , line */, updated_state.castling_rights, updated_state.ep_square);
        unmake_move(state_history, position_count, zobrist);

        if (time_up.load(std::memory_order_relaxed))
        {
            std::cout << "TIME LIMIT EXCEEDED" << std::endl;

            /* if (alpha < current_search_data.scores[0].top_score){
                best_move = current_search_data.moves_list[0];
                best_score = current_search_data.scores[0].top_score;
            } */
            return best_score;
        }

        zobrist = cur_hash;
        entry.top_score = score;
        previous_search_data.scores.push_back(std::move(entry));

        if (depth_limit >= 10)
        {
            std::cout << i << " "
                      << score << " "
                      << current_search_data.scores[i].top_score << " "
                      << "(" << ((move.from_square & 7) + 1) << ","
                      << ((move.from_square >> 3) + 1) << ") -> ("
                      << ((move.to_square & 7) + 1) << ","
                      << ((move.to_square >> 3) + 1) << ")"
                      << std::endl;
        }

        // Check if the current move's score is better than the existing best move
        if (score > best_score)
        {
            best_move = move;
            best_score = score;
            best_move_index = i;

            if (score > alpha)
                updatePV(move, cur_depth);
        }

        alpha = std::max(alpha, best_score);

        // Check for a beta cutoff
        if (beta <= alpha)
        {
            if (depth_limit >= 10)
            {
                std::cout << std::endl;
                std::cout << "Best: " << best_move_index << std::endl;
            }

            // Fail-soft: return the actual score that exceeded beta. (A killer/history update on the
            // cutoff was originally placed after this return -- dead, and near-useless at the root
            // anyway, since root cutoffs are rare and ply-0 killers are not read in root ordering.)
            return best_score;
        }

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        {
            if (depth_limit >= 10)
            {
                std::cout << std::endl;
                std::cout << "TIME LIMIT EXCEEDED" << std::endl;
                std::cout << "Best: " << best_move_index << std::endl;
            }
            return best_score;
        }
    }

    if (cur_depth == 0)
    {
        /* if (repetition_flag) {
            if (alpha < repetition_score) {
                if (alpha <= -500) {
                    best_move = repetition_move;
                    previous_search_data.top_level_preliminary_scores[repetition_index] = repetition_score;
                    best_score = 0;
                }
            }
        } */

        if (depth_limit >= 10)
        {
            std::cout << std::endl;
            std::cout << "Best: " << best_move_index << std::endl;
        }

        return best_score;
    }
    return best_score;
}

inline int get_score_for_minimizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove,
                                   std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state,
                                   bool &using_fp, int &num_iterations, bool is_in_null_search, bool &is_exact_hit)
{
    if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
        g_searchStack[cur_depth] = move;   // record this node's move for the 2-ply continuation key
    bool using_tt = false;
    int score = 0;
    BoardState updated_state = state_history.back();
    if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || updated_state.halfmove_clock >= 100)
    {
        score = 0;
    }
    else
    {
        // Check / forcing extension: a move that gives check searches one ply deeper,
        // bounded to CHECK_EXTENSION plies per path. Bumping the local depth_limit here
        // propagates to every child search and TT store below. Gated on CHECK_EXTENSION,
        // so the default build computes no extra is_check and is byte-identical.
        bool extend = false;
        if (Config::CHECK_EXTENSION > 0 && g_check_extensions < Config::CHECK_EXTENSION && depth_limit + 2 < MAX_PLY)
        {
            extend = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
            // SEE filter (disabled by default): drop the extension for a spite check whose
            // checker the opponent can win for more than SEE_EXTEND_MARGIN. see() runs only on
            // the checking-move minority, and only when the filter is enabled.
            if (extend && Config::SEE_EXTEND_MARGIN < Config::SEE_EXTEND_DISABLED &&
                see(move.to_square, updated_state.turn, updated_state) > Config::SEE_EXTEND_MARGIN)
                extend = false;
        }
        CheckExtensionGuard ceg(extend);
        if (extend)
            depth_limit++;

        TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
        tt_visits++;
        if (entry != nullptr)
        {
            // TTEntry entry = entry_opt.value();
            tt_probes++;
            if (entry->depth >= (depth_limit - cur_depth) /* && (depth_limit - cur_depth) >= 5 */)
            {
                use_tt_entry(*entry, score, using_tt, alpha, beta, num_iterations, false, true);
                /* if (using_tt && entry.flag == TTFlag::EXACT && entry.pv_length > 0 && score > alpha && !is_in_null_search) {
                    for (int j = 0; j < entry.pv_length; ++j) {
                        pv_table[cur_depth + 1][j] = entry.pv[j];
                    }
                    pv_length[cur_depth + 1] = entry.pv_length;
                }                              */
            }
        }

        if (!using_tt)
        {
            if (i == 0 || (depth_limit - cur_depth) == 1)
            {
                // Full window search for first move
                if (!using_tt)
                {
                    score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    if (cur_depth < depth_limit - 1)
                    {
                        TTFlag flag;
                        if (score <= alpha_orig)
                        {
                            flag = TTFlag::UPPERBOUND;
                        }
                        else if (score >= beta_orig)
                        {
                            flag = TTFlag::LOWERBOUND;
                        }
                        else
                        {
                            flag = TTFlag::EXACT;
                        }
                        // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                    }
                }
            }
            else
            {
                // LMR flag can still be computed here as you do
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
                bool base_lmr = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */) && !(Config::PROTECT_PV && (beta - alpha > 1)) && !(Config::PROTECT_KILLERS && (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previousMove.from_square][previousMove.to_square] == move));
                // Passed-pawn exemption: an otherwise-reducible ADVANCED pawn push (the moved piece landed as a
                // pawn; advancement toward promotion inferred from the push direction) is kept un-pruned/un-reduced,
                // so a slow passer march stays above the LMP/LMR horizon. Folding it into do_lmr covers the LMP,
                // LMR-reduction and VERIFY re-search gates at once. Move-level; default off = byte-identical.
                bool passer_exempt = false;
                if (Config::ENABLE_PASSER_PRUNE_EXEMPT && base_lmr && (updated_state.pawns & (1ULL << move.to_square)))
                {
                    int from_rank = move.from_square >> 3;
                    int to_rank = move.to_square >> 3;
                    int adv = (to_rank > from_rank) ? to_rank : (7 - to_rank); // ranks advanced toward the mover's promotion
                    if (adv >= Config::PASSER_EXEMPT_ADV)
                    {
                        // Only exempt a TRUE passed pawn: no enemy pawn in the 3-file forward span ahead of
                        // the landing square. The rank check alone exempted every advanced push (over-fire);
                        // the span test restricts it to pawns that actually have a clear run to promotion.
                        bool mover_white = (to_rank > from_rank);
                        uint64_t enemy_pawns = updated_state.pawns & updated_state.occupied_colour[!mover_white]; // [0]=black,[1]=white
                        uint64_t span = mover_white ? passed_span_white[move.to_square] : passed_span_black[move.to_square];
                        if ((enemy_pawns & span) == 0)
                        {
                            passer_exempt = true;
                            g_passer_exempt_fires++;
                        }
                    }
                }
                bool do_lmr = base_lmr && !passer_exempt;

                // Late-move pruning: at low remaining depth, skip late quiet moves. do_lmr eligibility
                // already excludes captures, checks, promotions, killers/counter and in-check, so a forcing
                // move is never pruned. Early return is safe — the caller unmakes the move (like futility).
                if (Config::ENABLE_LMP && do_lmr)
                {
                    int rd = depth_limit - cur_depth;
                    if (rd >= 1 && rd <= Config::LMP_MAX_DEPTH && (int)i >= Config::LMP_BASE + Config::LMP_SCALE * rd * rd)
                        return 9999999;   // non-improving sentinel for the minimizer (never the new min, no false cutoff)
                }

                // Null window search with LMR applied inside
                if (do_lmr)
                {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if (cur_depth > 1 && (depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin)
                    {
                        int early_score = get_board_evaluation(state_history, zobrist, num_iterations);
                        // int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);

                        if (Config::ENABLE_FUTILITY && (early_score - FUTILITY_MARGINS[depth_limit - cur_depth - 1] > beta))
                        {
                            using_fp = true;
                            return early_score;
                        }
                    }
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    if (Config::ENABLE_HISTORY_LMR)
                    {
                        int hist_delta = history_lmr_delta(move, previousMove, current_state, cur_depth);
                        if (hist_delta < 0 && is_in_relavent_pin)
                            hist_delta = 0; // keep the pin-defender protection; only reduce-LESS may touch pins
                        reduced_depth = std::clamp(reduced_depth + hist_delta, 2, depth_limit);
                    }
                    if (Config::ENABLE_IMPROVING)
                    {
                        bool improving = true;
                        if (cur_depth >= 2 && g_evalStack[cur_depth] != NO_STATIC_EVAL && g_evalStack[cur_depth - 2] != NO_STATIC_EVAL)
                            improving = g_evalStack[cur_depth] < g_evalStack[cur_depth - 2]; // minimizer: a lower eval is improving for the side to move
                        if (!improving)
                            reduced_depth = std::clamp(reduced_depth - 1, 2, depth_limit);
                    }
                    TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr)
                    {
                        // TTEntry entry = entry_opt.value();
                        tt_probes++;
                        if (entry->depth >= (reduced_depth - cur_depth))
                        {
                            /* if (entry.flag == TTFlag::EXACT){
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }
                    }
                    if (!using_tt)
                    {
                        score = maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (Config::LMR_PROFILE)
                            lmr_profile_event(depth_limit, cur_depth, i, alpha, beta, score, reduced_depth, move, current_state, previousMove);
                        if (cur_depth < reduced_depth - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha)
                            {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                            else if (score >= alpha + 1)
                            {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                    else
                    {
                        tt_hits++;
                    }
                }
                else
                {

                    TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr)
                    {
                        // TTEntry entry = entry_opt.value();
                        tt_probes++;
                        if (entry->depth >= (depth_limit - cur_depth))
                        {
                            /* if (entry.flag == TTFlag::EXACT){
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }
                    }
                    if (!using_tt)
                    {
                        score = maximizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (cur_depth < depth_limit - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha)
                            {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                            else if (score >= alpha + 1)
                            {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                    else
                    {
                        tt_hits++;
                    }
                }

                // If score is promising, re-search full window without reduction.
                // VERIFY_MARGIN also re-searches a reduced move that fails low by only
                // a small margin (a near-miss the reduction may have wrongly buried).
                if ((score > alpha && score < beta) ||
                    (Config::VERIFY_MARGIN > 0 && do_lmr && score <= alpha && (alpha - score) < Config::VERIFY_MARGIN))
                {
                    using_tt = false;

                    if (!using_tt)
                    {
                        // Graduated verification: a near-miss (verify) re-search runs at a
                        // shallow depth_limit - VERIFY_RESEARCH_REDUCTION; the normal PVS
                        // re-search stays at full depth.
                        int research_depth = (score > alpha && score < beta) ? depth_limit : std::max(cur_depth + 1, depth_limit - Config::VERIFY_RESEARCH_REDUCTION);
                        score = maximizer(cur_depth + 1, research_depth, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (cur_depth < research_depth - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha_orig)
                            {
                                flag = TTFlag::UPPERBOUND;
                            }
                            else if (score >= beta_orig)
                            {
                                flag = TTFlag::LOWERBOUND;
                            }
                            else
                            {
                                flag = TTFlag::EXACT;
                            }
                            // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                            addToSearchEvalCache(zobrist, state_history.size(), score, research_depth - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }
                }
            }
        }
        else
        {
            tt_hits++;
        }
    }
    return score;
}

inline int get_score_for_maximizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove,
                                   std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state,
                                   bool &using_fp, int &num_iterations, bool is_in_null_search, bool &is_exact_hit)
{
    if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
        g_searchStack[cur_depth] = move;   // record this node's move for the 2-ply continuation key
    bool using_tt = false;
    int score = 0;
    std::vector<int> dummy_ints;
    std::vector<Move> dummy_moves;
    RootScore dummy_entry;
    BoardState updated_state = state_history.back();
    if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || updated_state.halfmove_clock >= 100)
    {
        score = 0;
    }
    else
    {
        // Check / forcing extension: a move that gives check searches one ply deeper,
        // bounded to CHECK_EXTENSION plies per path. Bumping the local depth_limit here
        // propagates to every child search and TT store below. Gated on CHECK_EXTENSION,
        // so the default build computes no extra is_check and is byte-identical.
        bool extend = false;
        if (Config::CHECK_EXTENSION > 0 && g_check_extensions < Config::CHECK_EXTENSION && depth_limit + 2 < MAX_PLY)
        {
            extend = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
            // SEE filter (disabled by default): drop the extension for a spite check whose
            // checker the opponent can win for more than SEE_EXTEND_MARGIN. see() runs only on
            // the checking-move minority, and only when the filter is enabled.
            if (extend && Config::SEE_EXTEND_MARGIN < Config::SEE_EXTEND_DISABLED &&
                see(move.to_square, updated_state.turn, updated_state) > Config::SEE_EXTEND_MARGIN)
                extend = false;
        }
        CheckExtensionGuard ceg(extend);
        if (extend)
            depth_limit++;

        TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
        tt_visits++;
        if (entry != nullptr)
        {
            // TTEntry entry = entry_opt.value();
            tt_probes++;
            if (entry->depth >= (depth_limit - cur_depth) /* && (depth_limit - cur_depth) >= 5 */)
            {
                use_tt_entry(*entry, score, using_tt, alpha, beta, num_iterations, true, true);
                /* if (using_tt && entry.flag == TTFlag::EXACT && entry.pv_length > 0 && score > alpha && !is_in_null_search) {
                    for (int j = 0; j < entry.pv_length; ++j) {
                        pv_table[cur_depth + 1][j] = entry.pv[j];
                    }
                    pv_length[cur_depth + 1] = entry.pv_length;
                } */
            }
        }

        if (!using_tt)
        {
            if (i == 0)
            {
                // Full window search for first move
                if (!using_tt)
                {
                    score = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    if (cur_depth < depth_limit - 1)
                    {
                        TTFlag flag;
                        if (score <= alpha_orig)
                        {
                            flag = TTFlag::UPPERBOUND;
                        }
                        else if (score >= beta_orig)
                        {
                            flag = TTFlag::LOWERBOUND;
                        }
                        else
                        {
                            flag = TTFlag::EXACT;
                        }
                        // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                        addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                    }
                }
            }
            else
            {
                // LMR flag can still be computed here as you do
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
                bool base_lmr = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */) && !(Config::PROTECT_PV && (beta - alpha > 1)) && !(Config::PROTECT_KILLERS && (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previousMove.from_square][previousMove.to_square] == move));
                // Passed-pawn exemption: an otherwise-reducible ADVANCED pawn push (the moved piece landed as a
                // pawn; advancement toward promotion inferred from the push direction) is kept un-pruned/un-reduced,
                // so a slow passer march stays above the LMP/LMR horizon. Folding it into do_lmr covers the LMP,
                // LMR-reduction and VERIFY re-search gates at once. Move-level; default off = byte-identical.
                bool passer_exempt = false;
                if (Config::ENABLE_PASSER_PRUNE_EXEMPT && base_lmr && (updated_state.pawns & (1ULL << move.to_square)))
                {
                    int from_rank = move.from_square >> 3;
                    int to_rank = move.to_square >> 3;
                    int adv = (to_rank > from_rank) ? to_rank : (7 - to_rank); // ranks advanced toward the mover's promotion
                    if (adv >= Config::PASSER_EXEMPT_ADV)
                    {
                        // Only exempt a TRUE passed pawn: no enemy pawn in the 3-file forward span ahead of
                        // the landing square. The rank check alone exempted every advanced push (over-fire);
                        // the span test restricts it to pawns that actually have a clear run to promotion.
                        bool mover_white = (to_rank > from_rank);
                        uint64_t enemy_pawns = updated_state.pawns & updated_state.occupied_colour[!mover_white]; // [0]=black,[1]=white
                        uint64_t span = mover_white ? passed_span_white[move.to_square] : passed_span_black[move.to_square];
                        if ((enemy_pawns & span) == 0)
                        {
                            passer_exempt = true;
                            g_passer_exempt_fires++;
                        }
                    }
                }
                bool do_lmr = base_lmr && !passer_exempt;

                // Late-move pruning: at low remaining depth, skip late quiet moves. do_lmr eligibility
                // already excludes captures, checks, promotions, killers/counter and in-check, so a forcing
                // move is never pruned. Early return is safe — the caller unmakes the move (like futility).
                if (Config::ENABLE_LMP && do_lmr)
                {
                    int rd = depth_limit - cur_depth;
                    if (rd >= 1 && rd <= Config::LMP_MAX_DEPTH && (int)i >= Config::LMP_BASE + Config::LMP_SCALE * rd * rd)
                        return -9999999;   // non-improving sentinel for the maximizer (never the new max, no false cutoff)
                }

                // Null window search with LMR applied inside
                if (do_lmr)
                {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if ((depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin)
                    {
                        int early_score = get_board_evaluation(state_history, zobrist, num_iterations);
                        // int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                        if (Config::ENABLE_FUTILITY && (early_score + FUTILITY_MARGINS[depth_limit - cur_depth - 1] < alpha))
                        {
                            using_fp = true;
                            return early_score;
                        }
                    }
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    if (Config::ENABLE_HISTORY_LMR)
                    {
                        int hist_delta = history_lmr_delta(move, previousMove, current_state, cur_depth);
                        if (hist_delta < 0 && is_in_relavent_pin)
                            hist_delta = 0; // keep the pin-defender protection; only reduce-LESS may touch pins
                        reduced_depth = std::clamp(reduced_depth + hist_delta, 2, depth_limit);
                    }
                    if (Config::ENABLE_IMPROVING)
                    {
                        bool improving = true;
                        if (cur_depth >= 2 && g_evalStack[cur_depth] != NO_STATIC_EVAL && g_evalStack[cur_depth - 2] != NO_STATIC_EVAL)
                            improving = g_evalStack[cur_depth] > g_evalStack[cur_depth - 2]; // maximizer: a higher eval is improving for the side to move
                        if (!improving)
                            reduced_depth = std::clamp(reduced_depth - 1, 2, depth_limit);
                    }

                    TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr)
                    {
                        // TTEntry entry = entry_opt.value();
                        tt_probes++;
                        if (entry->depth >= (reduced_depth - cur_depth))
                        {
                            /* if (entry.flag == TTFlag::EXACT){
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }
                    }
                    if (!using_tt)
                    {

                        score = minimizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (Config::LMR_PROFILE)
                            lmr_profile_event(depth_limit, cur_depth, i, alpha, beta, score, reduced_depth, move, current_state, previousMove);
                        if (cur_depth < reduced_depth - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha)
                            {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                            else if (score >= alpha + 1)
                            {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                    else
                    {
                        tt_hits++;
                    }
                }
                else
                {

                    TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
                    if (entry != nullptr)
                    {
                        // TTEntry entry = entry_opt.value();
                        tt_probes++;
                        if (entry->depth >= (depth_limit - cur_depth))
                        {
                            /* if (entry.flag == TTFlag::EXACT){
                                score = entry.score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            } */
                            use_tt_entry(*entry, score, using_tt, alpha, alpha + 1, num_iterations, false, false);
                        }
                    }
                    if (!using_tt)
                    {
                        score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (cur_depth < depth_limit - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha)
                            {
                                flag = TTFlag::UPPERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                            else if (score >= alpha + 1)
                            {
                                flag = TTFlag::LOWERBOUND;
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha, alpha + 1, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                    else
                    {
                        tt_hits++;
                    }
                }

                // If score is promising, re-search full window without reduction.
                // VERIFY_MARGIN also re-searches a reduced move that fails low by only
                // a small margin (a near-miss the reduction may have wrongly buried).
                if ((score > alpha && score < beta) ||
                    (Config::VERIFY_MARGIN > 0 && do_lmr && score <= alpha && (alpha - score) < Config::VERIFY_MARGIN))
                {
                    using_tt = false;

                    if (!using_tt)
                    {
                        // Graduated verification: a near-miss (verify) re-search runs at a
                        // shallow depth_limit - VERIFY_RESEARCH_REDUCTION; the normal PVS
                        // re-search stays at full depth.
                        int research_depth = (score > alpha && score < beta) ? depth_limit : std::max(cur_depth + 1, depth_limit - Config::VERIFY_RESEARCH_REDUCTION);
                        score = minimizer(cur_depth + 1, research_depth, alpha, beta, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                        if (cur_depth < research_depth - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha_orig)
                            {
                                flag = TTFlag::UPPERBOUND;
                            }
                            else if (score >= beta_orig)
                            {
                                flag = TTFlag::LOWERBOUND;
                            }
                            else
                            {
                                flag = TTFlag::EXACT;
                            }
                            // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                            addToSearchEvalCache(zobrist, state_history.size(), score, research_depth - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }
                }
            }
        }
        else
        {
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

int minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> second_level_preliminary_scores, std::vector<Move> second_level_moves_list,
              RootScore &out_entry, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previousMove,
              int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
{

    if (time_up.load(std::memory_order_relaxed))
    {
        return 0;
    }
    else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL)
    {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        {
            time_up.store(true, std::memory_order_relaxed);
        }
    }
    // pv_length[cur_depth] = 0;
    BoardState current_state = state_history.back();

    /* if (depth_limit >= 24) {
        std::cout << "EE: " << std::endl;
    } */

    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    if (cur_depth >= depth_limit)
    {

        if (USE_Q_SEARCH /* && depth_limit >= 6 */)
        {
            if (use_q_precautions.load(std::memory_order_relaxed))
            {
                if (depth_limit >= 6)
                {
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);
                    return result;
                }
            }
            else
            {
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
            // return result;
        }
        int result = get_board_evaluation(state_history, zobrist, num_iterations);
        if (current_state.occupied == 3539934878248206336 && current_state.queens == 0)
        {
            // if (depth_limit >= 11){
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

    if (cur_depth == 1)
    {
        if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || current_state.halfmove_clock >= 100)
        {
            is_draw = true;
            return 0;
        }
        std::vector<int> cur_second_level_preliminary_scores;
        cur_second_level_preliminary_scores.reserve(64);

        ascending_sort(second_level_preliminary_scores, second_level_moves_list);
        out_entry.second_moves = second_level_moves_list;
        bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        std::vector<Move> searched_quiets, searched_captures;   // history-gravity malus lists (empty = no cost when gravity off)
        for (size_t i = 0; i < second_level_moves_list.size(); ++i)
        {
            Move &move = second_level_moves_list[i];
            // DEBUG: this is the crash site — a corrupt second_level_moves_list (lost correspondence
            // with the parent moves_list upstream) feeds an illegal move into make_move/update_state.
            // When CHESS_DEBUG_INVARIANTS=1, log it and skip instead of throwing, so the run finishes.
            if (dbg_bad_move("minimizer", (int)i, move, current_state))
                continue;
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
                move.promotion);

            make_move(state_history, position_count, move, zobrist, capture_move);
            score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                            position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
            if (Config::ENABLE_HISTORY_MALUS)
                (capture_move ? searched_captures : searched_quiets).push_back(move);

            unmake_move(state_history, position_count, zobrist);
            if (current_state.occupied == 10658118197365045909 && current_state.queens == 144119586122366976)
            {
                // if (depth_limit >= 11){
                std::cout << "BBBB- min 0: " << i << " "
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

            if (score < lowest_score)
            {
                lowest_score = score;
                best_move = move;

                // Beta improved — update PV
                if (score > alpha && !is_exact_hit)
                    updatePV(move, cur_depth);
            }

            beta = std::min(beta, lowest_score);

            // Check for a beta cutoff
            if (beta <= alpha)
            {
                ++g_fh_total;
                if (i == 0)
                    ++g_fh_first;
                g_cutoff_histogram[i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
                // std::cout <<score << std::endl;
                out_entry.second_scores = cur_second_level_preliminary_scores;

                if (!capture_move)
                {
                    storeKillerMove(cur_depth, move);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                    bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                    if (Config::ENABLE_HISTORY_SATURATION)
                    {
                        hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                        hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square], b);
                        if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square], b / Config::CONT2_GRAVITY_DIV);
                    }
                    else
                    {
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                        counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                        if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                    }
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_quiets)
                        {
                            if (q == move) continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                            {
                                hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                                hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square], -b / Config::MALUS_DIV);
                                if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                            }
                            else
                            {
                                historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                                counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                                if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                            }
                        }
                    }
                }
                else if (Config::ENABLE_CAPTURE_HIST)
                {
                    int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    if (Config::ENABLE_HISTORY_SATURATION)
                        hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                    else
                        captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_captures)
                        {
                            if (q == move) continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                                hist_update(captureHistory[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                            else
                                captureHistory[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                        }
                    }
                }
                return lowest_score;
            }
        }

        // Check if no moves are available, inidicating a game ending move was made previously
        if (lowest_score == 9999999 - static_cast<int>(state_history.size()))
        {

            out_entry.second_scores = cur_second_level_preliminary_scores;

            if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 9999999 - static_cast<int>(state_history.size());
            }
            else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                                  current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }
            else if (all_moves_pruned)
            {
                /* if (score > alpha)
                    updatePV(best_futility_move, cur_depth); */
                return best_early_eval;
            }
        }
        out_entry.second_scores = cur_second_level_preliminary_scores;
    }
    else
    {
        bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        if (Config::ENABLE_IMPROVING && cur_depth < MAX_PLY)
            g_evalStack[cur_depth] = (!currently_in_check && (depth_limit - cur_depth) <= Config::IMPROVING_EVAL_WINDOW)
                                         ? static_eval_for_improving(state_history, zobrist) : NO_STATIC_EVAL;
        // Null Move Pruning
        if (Config::ENABLE_NULLMOVE && cur_depth >= Config::NULLMOVE_CURDEPTH_MINI && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state))
        {
            state_history.back().turn = !state_history.back().turn;

            int cur_ep_square = state_history.back().ep_square;
            state_history.back().ep_square = -1;
            updateZobristHashForNullMove(zobrist);

            int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

            if (depth_limit >= 10)
                reduced_depth -= 1;
            if (Config::NULLMOVE_PROGRESSIVE)
            {
                // Gate on the genuine iteration depth, not the check-extension-inflated depth_limit, so the
                // extra reduction only fires when the search is really at depth >=12 (not on shallow lines
                // that a check extension pushed there). g_check_extensions = active extensions on this path.
                int base_depth = depth_limit - g_check_extensions;
                if (base_depth >= 12)
                    reduced_depth -= 1;
                if (base_depth >= 14)
                    reduced_depth -= 1;
            }

            /* if ((reduced_depth <= cur_depth + 1) && relevant_pin_exists(state_history, false)){
                reduced_depth = cur_depth + 2;
            }
            reduced_depth = std::min(depth_limit,reduced_depth); */
            // reduced_depth = std::max(2,reduced_depth);
            TTEntry *entry = accessSearchEvalCache(zobrist, state_history.back().castling_rights, state_history.back().ep_square);

            int null_move_score;
            if (entry != nullptr)
            {
                // TTEntry entry = entry_opt.value();
                if (entry->depth >= (reduced_depth - cur_depth))
                {
                    if (entry->flag == TTFlag::EXACT)
                    {
                        null_move_score = entry->score;
                        using_tt = true;
                        increment_node_count_with_decay(num_iterations);
                    }
                    // use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, true);
                }
            }
            if (!using_tt)
            {
                Move dummyMove;
                if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
                    g_searchStack[cur_depth] = dummyMove;   // null move breaks the 2-ply continuation chain
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

            if (null_move_score <= alpha)
            {
                return null_move_score; // fail-high cutoff
            }
        }

        std::vector<Move> moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
        std::vector<Move> searched_quiets, searched_captures;   // history-gravity malus lists (empty = no cost when gravity off)

        for (size_t i = 0; i < moves_list.size(); ++i)
        {
            Move &move = moves_list[i];
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
                move.promotion);

            make_move(state_history, position_count, move, zobrist, capture_move);
            score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                            position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
            if (Config::ENABLE_HISTORY_MALUS)
                (capture_move ? searched_captures : searched_quiets).push_back(move);

            if (current_state.occupied == 7199354783056128661 && current_state.queens == 144119586122366976)
            {
                // if (depth_limit >= 11){
                std::cout << "BBBB- min 2: " << i << " "
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
            if (using_fp)
            {
                best_early_eval = std::min(best_early_eval, score);
                best_futility_move = move;
                unmake_move(state_history, position_count, zobrist);
                zobrist = cur_hash;
                continue;
            }
            else
            {
                all_moves_pruned = false;
            }

            unmake_move(state_history, position_count, zobrist);

            if (time_up.load(std::memory_order_relaxed))
                return 0;
            zobrist = cur_hash;

            // lowest_score = std::min(score,lowest_score);
            if (score < lowest_score)
            {

                lowest_score = score;
                best_move = move;

                // Beta improved — update PV
                /* if (score > alpha && !is_in_null_search)
                    updatePV(move, cur_depth); */

                if (score > alpha && !is_in_null_search /* && !is_exact_hit */)
                {
                    updatePV(move, cur_depth);
                }
            }
            beta = std::min(beta, lowest_score);

            // Check for a beta cutoff
            if (beta <= alpha)
            {
                ++g_fh_total;
                if (i == 0)
                    ++g_fh_first;
                g_cutoff_histogram[i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
                if (i != 0)
                    updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

                if (!capture_move)
                {
                    storeKillerMove(cur_depth, move);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                    bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                    if (Config::ENABLE_HISTORY_SATURATION)
                    {
                        hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                        hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square], b);
                        if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square], b / Config::CONT2_GRAVITY_DIV);
                    }
                    else
                    {
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                        counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                        if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                    }
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_quiets)
                        {
                            if (q == move) continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                            {
                                hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                                hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square], -b / Config::MALUS_DIV);
                                if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                            }
                            else
                            {
                                historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                                counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                                if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                            }
                        }
                    }
                }
                else if (Config::ENABLE_CAPTURE_HIST)
                {
                    int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                    if (Config::ENABLE_HISTORY_SATURATION)
                        hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                    else
                        captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_captures)
                        {
                            if (q == move) continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                                hist_update(captureHistory[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                            else
                                captureHistory[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                        }
                    }
                }

                return lowest_score;
            }
        }

        // Check if no moves are available, inidicating a game ending move was made previously
        if (lowest_score == 9999999 - static_cast<int>(state_history.size()))
        {
            /* if (current_state.occupied == 11089329065645372309){
                    std::cout << "MINIMIZER FROM: " << " SCORE: " << score << " LOWEST SCORE: " << lowest_score << std::endl;
                } */
            if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {

                return 9999999 - static_cast<int>(state_history.size());
            }
            else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                                  current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }
            else if (all_moves_pruned)
            {
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

    if (!capture_move)
    {
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }

    return lowest_score;
}

int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count,
              uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
{

    BoardState current_state = state_history.back();

    if (time_up.load(std::memory_order_relaxed))
    {
        return 0;
    }
    else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL)
    {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        {
            time_up.store(true, std::memory_order_relaxed);
        }
    }

    // pv_length[cur_depth] = 0;
    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    if (cur_depth >= depth_limit)
    {

        if (USE_Q_SEARCH /* && depth_limit >= 6 */)
        {
            if (use_q_precautions.load(std::memory_order_relaxed))
            {
                if (depth_limit >= 6)
                {
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                    return result;
                }
            }
            else
            {
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

            // return result;
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
    RootScore dummy_entry;

    bool all_moves_pruned = true;
    Move best_futility_move;
    bool using_tt = false;
    bool using_fp = false;
    bool is_exact_hit = false;
    int best_early_eval = -9999999 + static_cast<int>(state_history.size());
    ;
    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
    if (Config::ENABLE_IMPROVING && cur_depth < MAX_PLY)
        g_evalStack[cur_depth] = (!currently_in_check && (depth_limit - cur_depth) <= Config::IMPROVING_EVAL_WINDOW)
                                     ? static_eval_for_improving(state_history, zobrist) : NO_STATIC_EVAL;

    // Null Move Pruning
    if (Config::ENABLE_NULLMOVE && cur_depth >= Config::NULLMOVE_CURDEPTH_MAXI && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state))
    {
        state_history.back().turn = !state_history.back().turn;

        int cur_ep_square = state_history.back().ep_square;
        state_history.back().ep_square = -1;
        updateZobristHashForNullMove(zobrist);

        int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

        if (depth_limit >= 10)
            reduced_depth -= 1;
        if (Config::NULLMOVE_PROGRESSIVE)
        {
            // Gate on the genuine iteration depth, not the check-extension-inflated depth_limit (see the
            // minimizer block): only fire the extra reduction when the search is really at depth >=12.
            int base_depth = depth_limit - g_check_extensions;
            if (base_depth >= 12)
                reduced_depth -= 1;
            if (base_depth >= 14)
                reduced_depth -= 1;
        }

        /* if ((reduced_depth <= cur_depth + 1) && relevant_pin_exists(state_history, false)){
            reduced_depth = cur_depth + 2;
        }
        reduced_depth = std::min(depth_limit,reduced_depth); */
        // reduced_depth = std::max(2,reduced_depth);
        TTEntry *entry = accessSearchEvalCache(zobrist, state_history.back().castling_rights, state_history.back().ep_square);
        int null_move_score;

        if (entry != nullptr)
        {
            // TTEntry entry = entry_opt.value();
            if (entry->depth >= (reduced_depth - cur_depth))
            {
                if (entry->flag == TTFlag::EXACT)
                {
                    null_move_score = entry->score;
                    using_tt = true;
                    increment_node_count_with_decay(num_iterations);
                }
                // use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, true);
            }
        }
        if (!using_tt)
        {
            Move dummyMove;
            if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
                g_searchStack[cur_depth] = dummyMove;   // null move breaks the 2-ply continuation chain
            null_move_score = minimizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, dummyMove, num_iterations, false, true, true);
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

        if (null_move_score >= beta)
        {
            return null_move_score; // fail-high cutoff
        }
    }

    std::vector<Move> moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
    std::vector<Move> searched_quiets, searched_captures;   // history-gravity malus lists (empty = no cost when gravity off)
    /* if (create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) == "8/8/3R3P/4P1P1/5P2/5K2/2k5/2q1b3 w - - 0 1"){
                                    std::cout << "AAAA" << std::endl;

    } */

    for (size_t i = 0; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];
        if (dbg_bad_move("maximizer", (int)i, move, current_state))
            continue;
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
            move.promotion);

        make_move(state_history, position_count, move, zobrist, capture_move);
        score = get_score_for_maximizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove,
                                        position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
        if (Config::ENABLE_HISTORY_MALUS)
            (capture_move ? searched_captures : searched_quiets).push_back(move);

        if (using_fp)
        {
            best_early_eval = std::max(best_early_eval, score);
            best_futility_move = move;
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;
            continue;
        }
        else
        {
            all_moves_pruned = false;
        }

        /* if(current_state.occupied == 11089329074235302805){
            std::cout << score << " | " << using_tt << " | " << !relevant_pin_exists(state_history) << " | " << std::endl;
        }  */
        unmake_move(state_history, position_count, zobrist);

        if (current_state.occupied == 10658119296876669589 && current_state.queens == 144119586122366976)
        {
            // if (depth_limit >= 11){
            std::cout << "MAX 1: " << i << " "
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

        if (current_state.occupied == 7199916633497922197 && current_state.queens == 144119586122366976)
        {
            // if (depth_limit >= 11){
            std::cout << "BBBB- MAX 3: " << i << " "
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
        // highest_score = std::max(score,highest_score);

        if (score > highest_score)
        {

            /* if (current_state.occupied == 16909313){
                std::cout << "MAXIMIZER FROM: " << (int)move.from_square << " TO: " << (int)move.to_square << std::endl;
            } */
            highest_score = score;
            best_move = move;

            // Alpha improved — update PV
            /* if (score > alpha && !is_in_null_search)
                updatePV(move, cur_depth); */

            if (score > alpha && !is_in_null_search /* && !is_exact_hit */)
            {
                updatePV(move, cur_depth);
            }
        }

        alpha = std::max(alpha, highest_score);

        // Check for a beta cutoff
        if (beta <= alpha)
        {
            ++g_fh_total;
            if (i == 0)
                ++g_fh_first;
            g_cutoff_histogram[i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
            if (i != 0)
                updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

            if (!capture_move)
            {
                storeKillerMove(cur_depth, move);
                counterMoves[previousMove.from_square][previousMove.to_square] = move;
                int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                if (Config::ENABLE_HISTORY_SATURATION)
                {
                    hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                    hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square], b);
                    if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square], b / Config::CONT2_GRAVITY_DIV);
                }
                else
                {
                    historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                    counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                    if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square] += 4 * b;
                }
                if (Config::ENABLE_HISTORY_MALUS)
                {
                    for (const Move &q : searched_quiets)
                    {
                        if (q == move) continue;
                        if (Config::ENABLE_HISTORY_SATURATION)
                        {
                            hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                            hist_update(counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square], -b / Config::MALUS_DIV);
                            if (p2v) hist_update(contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                        }
                        else
                        {
                            historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                            counterMoveHeuristics[current_state.turn][previousMove.from_square * 64 + previousMove.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                            if (p2v) contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][q.from_square * 64 + q.to_square] -= 4 * b / Config::MALUS_DIV;
                        }
                    }
                }
            }
            else if (Config::ENABLE_CAPTURE_HIST)
            {
                int b = (depth_limit - cur_depth) * (depth_limit - cur_depth);
                if (Config::ENABLE_HISTORY_SATURATION)
                    hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                else
                    captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                if (Config::ENABLE_HISTORY_MALUS)
                {
                    for (const Move &q : searched_captures)
                    {
                        if (q == move) continue;
                        if (Config::ENABLE_HISTORY_SATURATION)
                            hist_update(captureHistory[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                        else
                            captureHistory[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                    }
                }
            }

            return highest_score;
        }
    }

    /* if (all_moves_pruned){
        return best_early_eval;
    } */

    // Check if no moves are available, inidicating a game ending move was made previously
    if (highest_score == -9999999 + static_cast<int>(state_history.size()))
    {

        if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                         current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return highest_score;
        }
        else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                              current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return 0;
        }
        else if (all_moves_pruned)
        {
            /* if (score > alpha)
                updatePV(best_futility_move, cur_depth); */

            return best_early_eval;
        }
    }

    bool en_passant_move = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
    bool capture_move = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    if (!capture_move)
    {
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }

    return highest_score;
}

SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint &t0, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, int &num_iterations)
{

    increment_node_count_with_decay(num_iterations);

    BoardState current_state = state_history.back();

    SearchData returnData;
    SearchData current_search_data;

    int score = -99999999;
    int highest_score = -99999999;
    int depth = depth_limit - 1;
    // zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);
    uint64_t cur_hash = zobrist;

    std::vector<Move> moves_list;

    Move dummyMove;

    if (previous_search_data.moves_list.empty())
    {
        moves_list = buildMoveListFromReordered(state_history, zobrist, 0, dummyMove);
    }
    else
    {
        moves_list = previous_search_data.moves_list;
    }
    // std::cout <<"BBB2" << std::endl;

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
        moves_list[0].promotion);

    make_move(state_history, position_count, moves_list[0], zobrist, capture_move);

    std::vector<int> preliminary_scores;
    std::vector<Move> preliminary_moves;
    highest_score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, moves_list[0], num_iterations);
    // std::cout <<"BBB3" << std::endl;
    BoardState updated_state = state_history.back();
    // std::vector<Move> line(pv_table[1], pv_table[1] + pv_length[1]);
    addToSearchEvalCache(zobrist, state_history.size(), highest_score, (Config::ENABLE_TT_DEPTH_FIX ? depth : depth_limit), root_tt_flag(highest_score, alpha, beta), alpha, beta /* , line */, updated_state.castling_rights, updated_state.ep_square);
    unmake_move(state_history, position_count, zobrist);

    if (time_up.load(std::memory_order_relaxed))
    {
        return previous_search_data;
    }
    zobrist = cur_hash;

    if (score > alpha)
        updatePV(moves_list[0], 0);

    alpha = std::max(alpha, highest_score);

    current_search_data.scores.push_back(RootScore{highest_score, std::move(preliminary_moves), std::move(preliminary_scores)});

    for (size_t i = 1; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];
        // std::cout <<"BBB4-0" << std::endl;
        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
        // std::cout <<"BBB4-1" << std::endl;
        //  Assuming `updateZobristHashForMove` is defined elsewhere and works similarly
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
            move.promotion);
        // std::cout <<"BBB4-2" << std::endl;

        make_move(state_history, position_count, move, zobrist, capture_move);
        // std::cout <<"BBB4-3" << std::endl;
        std::vector<int> preliminary_scores;
        std::vector<Move> preliminary_moves;

        // zobrist = generateZobristHash(new_state.pawns, new_state.knights, new_state.bishops, new_state.rooks, new_state.queens, new_state.kings, new_state.occupied_colour[true], new_state.occupied_colour[false], new_state.turn);
        score = pre_minimizer(1, depth, alpha, alpha + 1, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
        // std::cout <<"BBB4" << std::endl;
        //  If the score is within the window, re-search with full window
        if (alpha < score && score < beta)
        {

            preliminary_scores.clear();
            preliminary_moves.clear();
            score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
        }
        // std::cout <<"BBB5-0" << std::endl;
        current_search_data.scores.push_back(RootScore{score, std::move(preliminary_moves), std::move(preliminary_scores)});
        // std::cout <<"BBB5-1" << std::endl;
        updated_state = state_history.back();
        // std::vector<Move> line(pv_table[1], pv_table[1] + pv_length[1]);
        addToSearchEvalCache(zobrist, state_history.size(), score, (Config::ENABLE_TT_DEPTH_FIX ? depth : depth_limit), root_tt_flag(score, alpha, beta), alpha, beta /* , line */, updated_state.castling_rights, updated_state.ep_square);

        unmake_move(state_history, position_count, zobrist);

        if (time_up.load(std::memory_order_relaxed))
        {
            return previous_search_data;
        }
        // std::cout <<"BBB5-2" << std::endl;
        zobrist = cur_hash;
        // highest_score = std::max(score,highest_score);

        if (score > highest_score)
        {
            highest_score = score;
            // Alpha improved — update PV

            if (score > alpha)
                updatePV(move, 0);
        }

        alpha = std::max(alpha, highest_score);
        // std::cout <<"BBB5-3" << std::endl;
    }

    /* bool side = current_state.turn;
    for (int i = 0; i < pv_length[0]; i++) {

        Move move = pv_table[0][i];
        int bonus = (((depth - 1) * (depth - 1)) * (pv_length[0] - i) * (pv_length[0] - i)) >> 2;
        moveFrequency[side][move.from_square][move.to_square] += bonus;
        side = !side;
    } */
    // std::cout <<"BBB5" << std::endl;
    if (previous_search_data.moves_list.empty())
    {

        returnData = current_search_data;
        returnData.moves_list = moves_list;

        sortSearchDataByScore(returnData);
    }
    else
    {
        returnData = previous_search_data;
        returnData.moves_list = moves_list;

        descending_sort_wrapper(current_search_data, returnData);
    }
    // std::cout <<"BBB6" << std::endl;
    dbg_searchdata("reorder_legal_moves", returnData);
    return returnData;
}

int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> &preliminary_scores, std::vector<Move> &pre_moves_list, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations)
{

    BoardState current_state = state_history.back();

    if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || current_state.halfmove_clock >= 100)
    {
        if (cur_depth == 1)
            is_draw = true;
        return 0;
    }

    /* if (num_iterations % DECAY_INTERVAL == 0)
        decayHistoryHeuristics();

    if ((num_iterations % (DECAY_INTERVAL * 32)) == 0)
        decayCounterMoveHeuristics(); */

    if (cur_depth >= depth_limit)
    {

        if (USE_Q_SEARCH /* && depth_limit >= 6 */)
        {
            if (use_q_precautions.load(std::memory_order_relaxed))
            {
                if (depth_limit >= 6)
                {
                    int result = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, prevMove, num_iterations, false);
                    return result;
                }
            }
            else
            {
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

    // zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);
    uint64_t cur_hash = zobrist;

    int lowest_score = 9999999 - static_cast<int>(state_history.size());
    int score;
    int beta_orig = beta;
    int alpha_orig = alpha;
    bool using_tt = false;

    Move best_move;

    std::vector<Move> moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, prevMove);
    pre_moves_list = moves_list;
    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
    for (size_t i = 0; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];
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
            move.promotion);

        make_move(state_history, position_count, move, zobrist, capture_move);

        BoardState updated_state = state_history.back();
        if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || updated_state.halfmove_clock >= 100)
        {
            score = 0;
        }
        else
        {

            TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);
            tt_visits++;
            if (entry != nullptr)
            {
                // TTEntry entry = entry_opt.value();
                tt_probes++;
                if (entry->depth >= (depth_limit - cur_depth))
                {
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
                            if (Config::ENABLE_CONT_HIST_2PLY && cur_depth >= 2)
                            {
                                Move p2 = g_searchStack[cur_depth - 2];
                                if (p2.from_square != p2.to_square)
                                    contHist2[current_state.turn][p2.from_square * 64 + p2.to_square][move.from_square * 64 + move.to_square] += cur_depth * cur_depth * cur_depth;
                            }
                        }
                        else if (Config::ENABLE_CAPTURE_HIST)
                        {
                            captureHistory[current_state.turn][move.from_square][move.to_square] += cur_depth * cur_depth;
                        }

                        pv_table[cur_depth][0] = move;
                        for (int j = 0; j < pv_length[cur_depth + 1]; ++j)
                            pv_table[cur_depth][j + 1] = pv_table[cur_depth + 1][j];
                        pv_length[cur_depth] = pv_length[cur_depth + 1] + 1;
                        return score;
                    } */
                }
            }

            if (!using_tt)
            {
                bool move_is_check = is_check(updated_state.turn, updated_state.occupied, updated_state.queens | updated_state.rooks, updated_state.queens | updated_state.bishops, updated_state.kings, updated_state.knights, updated_state.pawns, updated_state.occupied_colour[!updated_state.turn]);
                bool do_lmr = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1);

                if (do_lmr)
                {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    // int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

                    TTEntry *entry = accessSearchEvalCache(zobrist, updated_state.castling_rights, updated_state.ep_square);

                    if (entry != nullptr)
                    {
                        // TTEntry entry = entry_opt.value();
                        tt_probes++;
                        if (entry->depth >= (reduced_depth - cur_depth))
                        {
                            if (entry->flag == TTFlag::EXACT)
                            {
                                score = entry->score;
                                using_tt = true;
                                increment_node_count_with_decay(num_iterations);
                            }
                            // use_tt_entry(entry, score, using_tt, alpha, alpha + 1, num_iterations, false);
                        }
                    }
                    if (!using_tt)
                    {
                        score = maximizer(cur_depth + 1, reduced_depth, alpha, alpha + 1, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);

                        if (cur_depth < reduced_depth - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha)
                            {
                                flag = TTFlag::UPPERBOUND;
                                // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1 /* , line */, updated_state.castling_rights, updated_state.ep_square);
                            }
                            else if (score >= alpha + 1)
                            {
                                flag = TTFlag::LOWERBOUND;
                                // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                                addToSearchEvalCache(zobrist, state_history.size(), score, reduced_depth - cur_depth, flag, alpha, alpha + 1 /* , line */, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                    else
                    {
                        tt_hits++;
                    }
                    if (score < beta)
                    {
                        using_tt = false;

                        if (!using_tt)
                        {
                            score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
                            if (cur_depth < depth_limit - 1)
                            {
                                TTFlag flag;
                                if (score <= alpha_orig)
                                {
                                    flag = TTFlag::UPPERBOUND;
                                }
                                else if (score >= beta_orig)
                                {
                                    flag = TTFlag::LOWERBOUND;
                                }
                                else
                                {
                                    flag = TTFlag::EXACT;
                                }
                                // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                                addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                            }
                        }
                    }
                }
                else
                {

                    if (!using_tt)
                    {
                        score = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
                        if (cur_depth < depth_limit - 1)
                        {
                            TTFlag flag;
                            if (score <= alpha_orig)
                            {
                                flag = TTFlag::UPPERBOUND;
                            }
                            else if (score >= beta_orig)
                            {
                                flag = TTFlag::LOWERBOUND;
                            }
                            else
                            {
                                flag = TTFlag::EXACT;
                            }
                            // std::vector<Move> line(pv_table[cur_depth + 1], pv_table[cur_depth + 1] + pv_length[cur_depth + 1]);
                            addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth, flag, alpha_orig, beta_orig /* , line */, updated_state.castling_rights, updated_state.ep_square);
                        }
                    }
                }
            }
            else
            {
                tt_hits++;
            }
        }

        unmake_move(state_history, position_count, zobrist);
        zobrist = cur_hash;

        preliminary_scores.push_back(score);

        // lowest_score = std::min(score,lowest_score);
        if (score < lowest_score)
        {
            lowest_score = score;
            best_move = move;
            // Beta improved — update PV
            if (score > alpha)
                updatePV(move, cur_depth);
        }

        beta = std::min(beta, lowest_score);

        // Check for a beta cutoff
        if (beta <= alpha)
        {
            if (i != 0)
                updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

            if (!capture_move)
            {
                storeKillerMove(cur_depth, move);
                historyHeuristics[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
                counterMoves[prevMove.from_square][prevMove.to_square] = move;
                counterMoveHeuristics[current_state.turn][prevMove.from_square * 64 + prevMove.to_square][move.from_square * 64 + move.to_square] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
            }
            else if (Config::ENABLE_CAPTURE_HIST)
            {
                captureHistory[current_state.turn][move.from_square][move.to_square] += (depth_limit - cur_depth) * (depth_limit - cur_depth);
            }

            return lowest_score;
        }
    }

    // Check if no moves are available, inidicating a game ending move was made previously
    if (lowest_score == 9999999 - static_cast<int>(state_history.size()))
    {

        if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                         current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return lowest_score;
        }
        else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                              current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
        {
            return 0;
        }
    }

    bool en_passant_move = is_en_passant(best_move.from_square, best_move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
    bool capture_move = is_capture(best_move.from_square, best_move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

    if (!capture_move)
    {
        historyHeuristics[current_state.turn][best_move.from_square][best_move.to_square] += (depth_limit - cur_depth);
    }

    return lowest_score;
}

int qSearch(int alpha, int beta, int cur_depth, int qDepth, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing)
{
    qsearchVisits++;
    if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD))
    {
        return 0;
    }
    if (time_up.load(std::memory_order_relaxed))
    {
        return 0;
    }
    else if (nodes_since_time_check.fetch_add(1, std::memory_order_relaxed) >= TIME_CHECK_INTERVAL)
    {
        nodes_since_time_check.store(0, std::memory_order_relaxed);

        if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        {
            time_up.store(true, std::memory_order_relaxed);
        }
    }

    if (qDepth >= MAX_QDEPTH)
        return get_board_evaluation(state_history, zobrist, num_iterations);

    BoardState current_state = state_history.back();
    increment_node_count_with_decay(num_iterations);
    int cache_result;
    if (probeQCache(zobrist, current_state.castling_rights, current_state.ep_square, alpha, beta, cache_result))
    {
        return cache_result;
    }

    int moveNum = static_cast<int>(state_history.size());
    uint64_t cur_hash = zobrist;

    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);

    if (currently_in_check)
    {
        std::vector<Move> moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth + qDepth, prevMove);

        if (moves_list.empty())
        {
            int eval = 0;

            if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                             current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
            {
                return 0;
            }

            if (current_state.turn)
            {
                eval = 9999999 - moveNum;
            }
            else
            {
                eval = -9999999 + moveNum;
            }
            if (Config::side_to_play)
                eval = -eval;

            return eval;
        }
        int best = is_maximizing ? -9999999 + moveNum : 9999999 - moveNum;

        for (size_t i = 0; i < moves_list.size(); ++i)
        {
            Move &move = moves_list[i];

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
                move.promotion);

            // bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
            make_move(state_history, position_count, move, zobrist, capture_move);
            int score = qSearch(alpha, beta, cur_depth, qDepth + 1, t0, state_history, position_count, zobrist, move, num_iterations, !is_maximizing);
            unmake_move(state_history, position_count, zobrist);

            zobrist = cur_hash;

            if (is_maximizing)
            {
                if (score > best)
                    best = score;
                if (best > alpha)
                    alpha = best;
                if (best >= beta)
                    return best; // beta cutoff
            }
            else
            {
                if (score < best)
                    best = score;
                if (best < beta)
                    beta = best;
                if (best <= alpha)
                    return best; // alpha cutoff
            }
        }
        return best;
    }

    int static_eval = get_board_evaluation(state_history, zobrist, num_iterations);
    if (is_maximizing)
    {
        if (static_eval >= beta)
            return static_eval; // Fail-hard beta cutoff
        if (static_eval > alpha)
            alpha = static_eval;
        if (Config::ENABLE_QDELTA && static_eval < alpha - DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    }
    else
    {
        if (static_eval <= alpha)
            return static_eval; // Fail-hard alpha cutoff
        if (static_eval < beta)
            beta = static_eval;
        if (Config::ENABLE_QDELTA && static_eval > beta + DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    }

    int best = is_maximizing ? -9999999 + moveNum : 9999999 - moveNum;

    std::vector<Move> moves_list = buildNoisyMoveList(zobrist, state_history, cur_depth + qDepth, qDepth, prevMove);

    for (size_t i = 0; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];

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
            move.promotion);
        // std::cout << (int)move.from_square << " | " << (int)move.to_square << std::endl;
        // bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        make_move(state_history, position_count, move, zobrist, capture_move);
        // int test_score = get_board_evaluation(state_history, zobrist, num_iterations);
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

        if (is_maximizing)
        {
            if (score > best)
                best = score;
            if (best > alpha)
                alpha = best;
            if (best >= beta)
                return best; // beta cutoff
        }
        else
        {
            if (score < best)
                best = score;
            if (best < beta)
                beta = best;
            if (best <= alpha)
                return best; // alpha cutoff
        }
    }

    if (moves_list.empty())
    {
        return static_eval;
    }
    return best;
}

inline void sortSearchDataByScore(SearchData &data)
{
    // moves_list and the grouped scores must be the same length for the index reorder below. They are
    // equal whenever this is called (the grouping makes the old three-way drift impossible), so clamping
    // to the shorter is a safety net that is byte-identical in practice.
    size_t n = std::min(data.moves_list.size(), data.scores.size());

    // Create index vector
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on scores (descending)
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b)
              {
                  return data.scores[a].top_score > data.scores[b].top_score;
              });

    // Helper lambda to reorder any vector by indices
    auto reorder = [&](auto &vec)
    {
        using T = typename std::decay<decltype(vec[0])>::type;
        std::vector<T> temp(n);
        for (size_t i = 0; i < n; ++i)
            temp[i] = std::move(vec[indices[i]]);
        vec = std::move(temp);
    };

    // Reorder the moves and their grouped scores together (one permutation each instead of four).
    reorder(data.moves_list);
    reorder(data.scores);
    dbg_searchdata("sortSearchDataByScore", data);
}

inline void descending_sort_wrapper(const SearchData &preSearchData, SearchData &mainSearchData)
{
    // mainSearchData carries the previous iteration's full move list (moves_list) with its real searched
    // scores (cutoff length <= N); preSearchData holds a fresh shallow pre-pass score for every move.
    // Keep main's first `count` real entries, fill the rest of the tail from the pre-pass, then sort the
    // tail. Grouping the scores makes the move<->score correspondence atomic, so the old length-drift and
    // out-of-bounds swap/copy (the corruption crash) are structurally impossible. Bail if nothing to do.
    size_t common = std::min(mainSearchData.scores.size(), mainSearchData.moves_list.size());
    if (common == 0)
        return;

    // Find the max-scoring entry among the real searched prefix and swap it (move + grouped scores) to
    // the front. Searching only the common prefix keeps the swap in bounds.
    int max_index = 0;
    int max_value = mainSearchData.scores[0].top_score;
    for (size_t i = 1; i < common; ++i)
    {
        if (mainSearchData.scores[i].top_score > max_value)
        {
            max_value = mainSearchData.scores[i].top_score;
            max_index = static_cast<int>(i);
        }
    }
    if (max_index != 0)
    {
        std::swap(mainSearchData.moves_list[0], mainSearchData.moves_list[max_index]);
        std::swap(mainSearchData.scores[0], mainSearchData.scores[max_index]);
    }

    // Number of real searched entries to keep from main; the tail beyond it comes from the pre-pass.
    size_t count = std::min(mainSearchData.scores.size(), preSearchData.scores.size());

    // Build the tail (everything after the front): main's remaining moves, paired with main's real
    // searched scores for the first count-1 of them and the fresh pre-pass scores beyond that.
    std::vector<Move> moves_sub(mainSearchData.moves_list.begin() + 1, mainSearchData.moves_list.end());
    std::vector<RootScore> scores_sub(mainSearchData.scores.begin() + 1, mainSearchData.scores.end());
    if (preSearchData.scores.size() > count)
        scores_sub.insert(scores_sub.end(), preSearchData.scores.begin() + count, preSearchData.scores.end());

    // Keep the scores parallel to the moves (drop any score-only overhang). No-op in the normal case.
    if (scores_sub.size() > moves_sub.size())
        scores_sub.resize(moves_sub.size());

    SearchData sub_data;
    sub_data.moves_list = std::move(moves_sub);
    sub_data.scores = std::move(scores_sub);

    // Sort the tail descending by top_score
    sortSearchDataByScore(sub_data);

    // Write the sorted tail back after the front entry, growing main to the full length if needed.
    if (mainSearchData.scores.size() < sub_data.scores.size() + 1)
        mainSearchData.scores.resize(sub_data.scores.size() + 1);

    std::copy(sub_data.moves_list.begin(), sub_data.moves_list.end(), mainSearchData.moves_list.begin() + 1);
    std::copy(sub_data.scores.begin(), sub_data.scores.end(), mainSearchData.scores.begin() + 1);
    dbg_searchdata("descending_sort_wrapper", mainSearchData);
}

inline void ascending_sort(std::vector<int> &values, std::vector<Move> &moves)
{
    size_t count = values.size(); // number of valid entries
    if (moves.size() < count)
    {
        throw std::runtime_error("Mismatch: 'moves' must be at least as long as 'values'");
    }

    // Generate sorted indices for the valid prefix
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b)
              {
                  return values[a] < values[b];
              });

    // Reorder moves[0:count] based on sorted indices
    std::vector<Move> sorted_moves(count);
    for (size_t i = 0; i < count; ++i)
    {
        sorted_moves[i] = std::move(moves[indices[i]]);
    }

    // Place sorted prefix back into moves[0:count]
    std::move(sorted_moves.begin(), sorted_moves.end(), moves.begin());

    // values is unchanged in size and content, so no need to touch it
}

inline uint8_t get_piece_type(uint8_t square, std::vector<BoardState> &state_history)
{

    BoardState current_state = state_history.back();
    uint64_t mask = (BB_SQUARES[square]);

    if (current_state.pawns & mask)
    {
        return PAWN;
    }
    else if (current_state.knights & mask)
    {
        return KNIGHT;
    }
    else if (current_state.bishops & mask)
    {
        return BISHOP;
    }
    else if (current_state.rooks & mask)
    {
        return ROOK;
    }
    else if (current_state.queens & mask)
    {
        return QUEEN;
    }
    else if (current_state.kings & mask)
    {
        return KING;
    }
    else
    {
        return 0;
    }
}

inline bool relevant_pin_exists(std::vector<BoardState> &state_history, bool probe)
{

    BoardState current_state = state_history.back();

    // uint64_t relevant_pins = 0;
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
                                 BB_FILE_ATTACKS[black_king][BB_FILE_MASKS[black_king] & current_state.occupied]) /*  & current_state.occupied_colour[false] */;

    candidates |= queen_attacks_from_square;
    /* while(queen_attacks_from_square){
uint8_t current_square = __builtin_ctzll(queen_attacks_from_square);
queen_attacks_from_square &= queen_attacks_from_square - 1;

relevant_pawns |= BB_SQUARES[current_square];
} */

    if (current_state.occupied_colour[true] & current_state.queens)
    {
        uint8_t white_queen = __builtin_ctzll(current_state.queens & current_state.occupied_colour[true]);
        queen_attacks_from_square = (BB_DIAG_ATTACKS[white_queen][BB_DIAG_MASKS[white_queen] & current_state.occupied] | BB_RANK_ATTACKS[white_queen][BB_RANK_MASKS[white_queen] & current_state.occupied] |
                                     BB_FILE_ATTACKS[white_queen][BB_FILE_MASKS[white_queen] & current_state.occupied]) /*  & current_state.occupied_colour[true] */;
        candidates |= queen_attacks_from_square;
        /* if(probe)
            std::cout << "CANDIDATES:" << candidates << "; " << queen_attacks_from_square << ";" << std::endl; */
    }

    if (current_state.occupied_colour[false] & current_state.queens)
    {
        uint8_t black_queen = __builtin_ctzll(current_state.queens & current_state.occupied_colour[false]);
        queen_attacks_from_square = (BB_DIAG_ATTACKS[black_queen][BB_DIAG_MASKS[black_queen] & current_state.occupied] | BB_RANK_ATTACKS[black_queen][BB_RANK_MASKS[black_queen] & current_state.occupied] |
                                     BB_FILE_ATTACKS[black_queen][BB_FILE_MASKS[black_queen] & current_state.occupied]) /* & current_state.occupied_colour[false] */;
        candidates |= queen_attacks_from_square;
        /* if(probe)
            std::cout << "CANDIDATES:" << candidates << "; " << queen_attacks_from_square << ";" << std::endl; */
    }

    bb &= candidates & current_state.occupied;
    if (probe)
        std::cout << "BITMASK:" << bb << ";" << std::endl;
    while (bb)
    {
        uint8_t current_square = __builtin_ctzll(bb);
        bb &= bb - 1;

        bool is_white = current_state.occupied_colour[true] & BB_SQUARES[current_square];

        uint64_t own_pieces = is_white ? current_state.occupied_colour[true] : current_state.occupied_colour[false];
        uint64_t opp_pieces = is_white ? current_state.occupied_colour[false] : current_state.occupied_colour[true];

        // Get sliding attackers
        uint64_t rank_pieces = BB_RANK_MASKS[current_square] & current_state.occupied;
        uint64_t file_pieces = BB_FILE_MASKS[current_square] & current_state.occupied;
        uint64_t diag_pieces = BB_DIAG_MASKS[current_square] & current_state.occupied;

        uint64_t attackers = ((BB_RANK_ATTACKS[current_square][rank_pieces] & (current_state.queens | current_state.rooks)) |
                              (BB_FILE_ATTACKS[current_square][file_pieces] & (current_state.queens | current_state.rooks)) |
                              (BB_DIAG_ATTACKS[current_square][diag_pieces] & (current_state.queens | current_state.bishops))) &
                             opp_pieces;

        while (attackers)
        {
            uint8_t attacker_square = __builtin_ctzll(attackers);
            attackers &= attackers - 1;

            uint8_t attacker_type = get_piece_type(attacker_square, state_history);
            uint64_t behind_mask = attacks_mask(!is_white, current_state.occupied ^ BB_SQUARES[current_square], attacker_square, attacker_type) &
                                   ~attacks_mask(!is_white, current_state.occupied, attacker_square, attacker_type) & own_pieces;

            while (behind_mask)
            {
                uint8_t pinned_to_sq = __builtin_ctzll(behind_mask);
                behind_mask &= behind_mask - 1;

                uint8_t pinned_piece_type = get_piece_type(current_square, state_history);
                uint8_t pinned_to_piece_type = get_piece_type(pinned_to_sq, state_history);
                // Consider it relevant if pinned to king, queen, or rook
                if (pinned_piece_type == PAWN)
                {
                    if (pinned_to_piece_type == KING)
                    {
                        if (probe)
                            std::cout << (int)pinned_to_sq << " | " << (int)current_square << std::endl;
                        return true;
                    }
                }
                else if (pinned_to_piece_type == KING || pinned_to_piece_type == QUEEN || pinned_to_piece_type == ROOK)
                {
                    if (probe)
                        std::cout << (int)pinned_to_sq << " | " << (int)current_square << std::endl;
                    return true;
                    // relevant_pins |= BB_SQUARES[current_square];
                }
            }
        }
    }
    return false;
    // return relevant_pins;
}

inline void use_tt_entry(TTEntry &entry, int &score, bool &using_tt, int alpha, int beta, int &num_iterations, bool is_maximizing, bool use_extra_precautions)
{
    using_tt = false;

    if (entry.flag == TTFlag::EXACT)
    {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }

    if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta)
    {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }

    if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha)
    {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }
}

inline void use_tt_entry1(TTEntry &entry, int &score, bool &using_tt, int alpha, int beta, int &num_iterations, bool is_maximizing, bool use_extra_precautions)
{
    using_tt = false;

    /* if(!(use_extra_precautions && entry.alpha <= alpha && entry.beta >= beta)){
        return;
    } */
    if (entry.flag == TTFlag::EXACT)
    {
        score = entry.score;
        using_tt = true;
        increment_node_count_with_decay(num_iterations);
        return;
    }
    else
    {
        if (is_maximizing)
        {
            if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta /* && entry.beta >= beta */)
            {
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;
            }
            else if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha && entry.alpha <= alpha)
            {
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;
            }
        }
        else
        {
            if (entry.flag == TTFlag::LOWERBOUND && entry.score >= beta && entry.beta >= beta)
            {
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;
            }
            else if (entry.flag == TTFlag::UPPERBOUND && entry.score <= alpha /* && entry.alpha <= alpha */)
            {
                score = entry.score;
                using_tt = true;
                increment_node_count_with_decay(num_iterations);
                return;
            }
        }
    }
}

inline void increment_node_count_with_decay(int &num_iterations)
{
    num_iterations++;
    if (!Config::ENABLE_HISTORY_DECAY)
        return;
    if (num_iterations % Config::DECAY_INTERVAL == 0)
    {
        decayHistoryHeuristics();
        if (Config::ENABLE_CAPTURE_HIST)
            decayCaptureHistory();
    }

    if ((num_iterations % (Config::DECAY_INTERVAL * 16)) == 0)
    {
        decayCounterMoveHeuristics();
        if (Config::ENABLE_CONT_HIST_2PLY)
            decayContHist2();
    }
}

inline bool is_repetition(const std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key, const int repetition_count)
{
    auto it = position_count.find(zobrist_key);
    return it != position_count.end() && it->second >= repetition_count;
}

inline bool isUnsafeForNullMovePruning(BoardState current_state)
{

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

    if ((current_state.occupied_colour[current_state.turn] & current_state.queens) == 0)
    {
        if (pieceNum < 7)
            return true;
    }
    else
    {
        if (pieceNum < 4)
            return true;
    }

    // 4. (Optional) Check for fortress / locked structure heuristics

    // Passed all checks — safe to null prune
    return false;
}

inline int reduced_search_depth(int depth_limit, int cur_depth, bool is_in_relavent_pin, int move_number, BoardState current_state)
{

    /* if(is_in_relavent_pin && depth_limit < 5)
        return depth_limit; */

    if (depth_limit < 4)
        return depth_limit; // No reduction for shallow depths

    // Adjust move number so 1 and 2 map to no reduction
    int adjusted_move = std::max(move_number - 2, 1);
    int base = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];

    int phase = 0;
    phase += 4 * __builtin_popcountll(current_state.queens);
    phase += 2 * __builtin_popcountll(current_state.rooks);
    phase += 1 * __builtin_popcountll(current_state.bishops | current_state.knights);

    int phase_score = 128 * (MAX_PHASE - phase) / MAX_PHASE; // 0 to
    double scale = 0;

    if (phase_score <= 24)
    {
        scale = 1.5;
    }
    else if (phase_score <= 64)
    {
        scale = 1.75;
    }
    else if (phase_score <= 96)
    {
        scale = 2.0;
    }
    else if (phase_score < 117)
    {
        scale = 2.25;
    }
    else
    {
        return base;
    }
    double move_factor = std::log2(adjusted_move);
    int r = static_cast<int>(base - (move_factor / scale));

    if (is_in_relavent_pin)
    {
        if (r <= cur_depth + 1)
        {
            r = cur_depth + 2;
        }
        r = std::min(depth_limit, r);
    }

    return std::max(r, 2); // Ensure at least 4 ply is searched
}

inline void updatePV(Move move, int cur_depth)
{
    pv_table[cur_depth][0] = move;
    for (int j = 0; j < pv_length[cur_depth + 1]; ++j)
        pv_table[cur_depth][j + 1] = pv_table[cur_depth + 1][j];
    pv_length[cur_depth] = pv_length[cur_depth + 1] + 1;
}

// Promote a move from the list to a given index (if found beyond that index)
inline void promoteMove(std::vector<Move> &moves, const Move &move, size_t promoteToIndex, size_t &indexIncrement)
{
    auto it = std::find(moves.begin(), moves.end(), move);
    if (it != moves.end())
    {
        size_t foundIndex = std::distance(moves.begin(), it);
        if (foundIndex > promoteToIndex + indexIncrement)
        {
            Move temp = *it;
            moves.erase(it);
            moves.insert(moves.begin() + promoteToIndex + indexIncrement, temp);
            indexIncrement++;
        }
    }
}

inline std::vector<Move> buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove)
{

    move_gen_visits++;
    BoardState current_state = state_history.back();
    // uint64_t zobrist2 = zobrist;
    // zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);

    // if (zobrist != zobrist2)
    // std::cout << create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights, current_state.ep_square, current_state.turn) << std::endl;

    std::vector<Move> cached_moves = accessMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square);
    if (cached_moves.size() != 0)
    {
        move_gen_cache_hits++;
        if (cached_moves.size() > 1)
        {

            size_t firstNonCapture = 0;
            // int firstKillerMoveUsed = false;
            size_t indexIncrement = 0;

            for (size_t i = 0; i < cached_moves.size(); ++i)
            {
                if (!is_capture(cached_moves[i].from_square, cached_moves[i].to_square, current_state.occupied_colour[!current_state.turn], is_en_passant(cached_moves[i].from_square, cached_moves[i].to_square, current_state.ep_square, current_state.occupied, current_state.pawns)))
                {
                    firstNonCapture = static_cast<int>(i);
                    break;
                }
            }

            if (firstNonCapture == 0)
                firstNonCapture = 1;

            if (cur_ply > 63)
            {
                auto cm = std::find(cached_moves.begin(), cached_moves.end(), counterMoves[prevMove.from_square][prevMove.to_square]);

                if (cm != cached_moves.end())
                {
                    size_t foundIndex = std::distance(cached_moves.begin(), cm);
                    if (foundIndex > firstNonCapture)
                    {
                        std::iter_swap(cm, cached_moves.begin() + firstNonCapture + indexIncrement);
                    }
                }
            }
            else
            {
                auto km1 = std::find(cached_moves.begin(), cached_moves.end(), killerMoves[cur_ply][0]);

                if (km1 != cached_moves.end())
                {
                    size_t foundIndex = std::distance(cached_moves.begin(), km1);
                    if (foundIndex > firstNonCapture)
                    {
                        std::iter_swap(km1, cached_moves.begin() + firstNonCapture);
                        indexIncrement++;
                    }
                }

                auto km2 = std::find(cached_moves.begin(), cached_moves.end(), killerMoves[cur_ply][1]);

                if (km2 != cached_moves.end())
                {
                    size_t foundIndex = std::distance(cached_moves.begin(), km2);
                    if (foundIndex > firstNonCapture)
                    {
                        std::iter_swap(km2, cached_moves.begin() + firstNonCapture + indexIncrement);
                        indexIncrement++;
                    }
                }

                if (!(counterMoves[prevMove.from_square][prevMove.to_square] == killerMoves[cur_ply][0]) && !(counterMoves[prevMove.from_square][prevMove.to_square] == killerMoves[cur_ply][1]))
                {
                    auto cm = std::find(cached_moves.begin(), cached_moves.end(), counterMoves[prevMove.from_square][prevMove.to_square]);

                    if (cm != cached_moves.end())
                    {
                        size_t foundIndex = std::distance(cached_moves.begin(), cm);
                        if (foundIndex > firstNonCapture)
                        {
                            std::iter_swap(cm, cached_moves.begin() + firstNonCapture + indexIncrement);
                        }
                    }
                }
            }

            // Lazy re-sort of the stale quiet tail (everything after the pinned captures + killers/counter).
            // The cached order was frozen when this node was first searched; when LMP prunes the late quiets,
            // re-rank them against the CURRENT history so genuinely-good quiets escape the pruned tail. The
            // cheap path bubbles the top-K to the front; a full stable_sort runs periodically. Both act on the
            // returned copy only (the cached order is left intact and refreshed by beta-cutoff promotion).
            if (Config::ENABLE_LAZY_RESORT)
            {
                uint64_t mvKey = make_move_cache_key(zobrist, current_state.castling_rights, current_state.ep_square);
                MoveEntry &entry = accessMutableMoveGenCache(mvKey, current_state.castling_rights, current_state.ep_square);
                int lci = entry.last_cutoff_index;
                size_t quiet_start = firstNonCapture + indexIncrement;
                bool do_topk = (lci > Config::LAZY_RESORT_MIN_CUTOFF_IDX);
                bool do_full = (entry.reuse_count >= Config::RESORT_AFTER_REUSES) &&
                               (lci < 0 || lci > Config::LAZY_RESORT_MIN_CUTOFF_IDX);
                if ((do_topk || do_full) && quiet_start + 1 < cached_moves.size())
                {
                    int enemy_king_sq = -1;
                    if (Config::ENABLE_CHECK_ORDER)
                    {
                        uint64_t ek = current_state.kings & current_state.occupied_colour[!current_state.turn];
                        if (ek)
                            enemy_king_sq = 63 - __builtin_clzll(ek);
                    }
                    size_t qn = cached_moves.size() - quiet_start;
                    std::vector<int> qs(qn);
                    for (size_t t = 0; t < qn; ++t)
                    {
                        const Move &m = cached_moves[quiet_start + t];
                        qs[t] = score_quiet(m.from_square, m.to_square, m.promotion, current_state.turn, cur_ply, prevMove,
                                            current_state.occupied, current_state.pawns, current_state.knights,
                                            current_state.bishops, current_state.rooks, current_state.queens, enemy_king_sq);
                    }
                    if (do_full)
                    {
                        std::vector<size_t> ord(qn);
                        std::iota(ord.begin(), ord.end(), 0);
                        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return qs[a] > qs[b]; });
                        std::vector<Move> reordered;
                        reordered.reserve(qn);
                        for (size_t t : ord)
                            reordered.push_back(cached_moves[quiet_start + t]);
                        std::copy(reordered.begin(), reordered.end(), cached_moves.begin() + quiet_start);
                        entry.reuse_count = 0;
                    }
                    else
                    {
                        int K = Config::PROMOTE_TOP_K;
                        for (int k = 0; k < K && (size_t)k < qn; ++k)
                        {
                            size_t best = (size_t)k;
                            for (size_t j = (size_t)k + 1; j < qn; ++j)
                                if (qs[j] > qs[best])
                                    best = j;
                            if (best != (size_t)k)
                            {
                                std::iter_swap(cached_moves.begin() + quiet_start + k, cached_moves.begin() + quiet_start + best);
                                std::swap(qs[k], qs[best]);
                            }
                        }
                    }
                }
                entry.reuse_count++;
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

inline std::vector<Move> buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, int qDepth, Move prevMove)
{

    std::vector<Move> noisy_moves;
    noisy_moves.reserve(16);

    BoardState current_state = state_history.back();

    std::vector<Move> moves_list;

    moves_list = accessMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square);
    if (moves_list.size() == 0)
    {
        generateLegalMovesReordered(moves_list, current_state.castling_rights, ~0ULL, ~0ULL,
                                    current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights,
                                    current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn, cur_ply, prevMove);
        moves_list.reserve(64);
    }

    for (size_t i = 0; i < moves_list.size(); ++i)
    {

        bool en_passant_move = is_en_passant(moves_list[i].from_square, moves_list[i].to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
        bool capture_move = is_capture(moves_list[i].from_square, moves_list[i].to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

        if (moves_list[i].promotion != 1)
        {
            noisy_moves.push_back(moves_list[i]);
        }
        else if (capture_move)
        {
            /* int pressure = get_pressure_at_square(current_state.turn, endPos[i]);
            int support = get_support_at_square(current_state.turn, endPos[i]);

            if (support == 0 || pressure > (support - SUPPORT_MARGIN)) {
                noisy_moves.push_back(Move(startPos[i],endPos[i],promotions[i]));
            } */
            if (en_passant_move)
            {
                noisy_moves.push_back(moves_list[i]);
            }
            else if (see(moves_list[i].to_square, current_state.turn, current_state) >= 0)
            {
                noisy_moves.push_back(moves_list[i]);
            }
        }
        else if (!(Config::ENABLE_QCHECK_DEPTH0 && qDepth > 0))
        {
            // Quiet move: include it only if it gives check. ENABLE_QCHECK_DEPTH0 (the guard above)
            // drops quiet checks past the first q-ply. ENABLE_QCHECK_MASK detects a DIRECT check with a
            // bitboard attack test from the destination square (the moving piece's attacks from `to`
            // with `from` vacated, vs the enemy king) -- no board copy; misses discovered checks (the
            // standard accepted tradeoff). Default off = the original simulate-and-test path.
            bool move_is_check;
            if (Config::ENABLE_QCHECK_MASK)
            {
                uint64_t fromBB = BB_SQUARES[moves_list[i].from_square];
                uint8_t to = moves_list[i].to_square;
                uint64_t occ = current_state.occupied & ~fromBB;
                uint8_t enemy_king = __builtin_ctzll(current_state.kings & current_state.occupied_colour[!current_state.turn]);
                uint64_t atk = 0;
                if (current_state.knights & fromBB)
                    atk = BB_KNIGHT_ATTACKS[to];
                else if (current_state.pawns & fromBB)
                    atk = BB_PAWN_ATTACKS[current_state.turn][to];
                else if (current_state.bishops & fromBB)
                    atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ];
                else if (current_state.rooks & fromBB)
                    atk = BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
                else if (current_state.queens & fromBB)
                    atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ] | BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ];
                move_is_check = (atk & BB_SQUARES[enemy_king]) != 0;
            }
            else
            {
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
                    turn);
                uint64_t opposingPieces = turn ? occupied_black : occupied_white;

                move_is_check = is_check(turn, occupied, queens | rooks, queens | bishops, kings, knights, pawns, opposingPieces);
            }

            if (move_is_check)
            {
                noisy_moves.push_back(moves_list[i]);
            }
        }
    }
    noisy_moves.shrink_to_fit();
    return noisy_moves;
}

inline int get_q_search_eval(int alpha, int beta, int cur_depth, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing)
{

    /* int cache_result = accessQCache(zobrist, current_state.castling_rights, current_state.ep_square);

    if (cache_result != 0){
        //eval_cache_hits++;
        num_iterations++;
        return cache_result;
    } */

    int result = qSearch(alpha, beta, cur_depth, 0, t0, state_history, position_count, zobrist, prevMove, num_iterations, is_maximizing);

    // Tag the q-search result with its bound type relative to the search window
    // so probeQCache can reuse it soundly.
    TTFlag flag;
    if (result <= alpha)
        flag = TTFlag::UPPERBOUND;
    else if (result >= beta)
        flag = TTFlag::LOWERBOUND;
    else
        flag = TTFlag::EXACT;

    addToQCache(zobrist, result, flag, current_state.castling_rights, current_state.ep_square);

    return result;
}

inline int get_board_evaluation(std::vector<BoardState> &state_history, uint64_t zobrist, int &num_iterations)
{

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
    // cache_result = accessCache(zobrist);
    if (accessCacheNew(zobrist, cache_result))
    {
        eval_cache_hits++;
        return cache_result;
    }

    int total = 0;
    int moveNum = static_cast<int>(state_history.size());

    if (is_checkmate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                     current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
    {
        if (current_state.turn)
        {
            total = 9999999 - moveNum;
        }
        else
        {
            total = -9999999 + moveNum;
        }
    }
    else if (is_stalemate(zobrist, current_state.castling_rights, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns,
                          current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn))
    {
        return 0;
    }
    else
    {
        total = placement_and_piece_eval(moveNum, current_state.turn, current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                         current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.occupied);
    }

    if (Config::side_to_play)
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

    // addToCache(zobrist, max_cache_size * Config::ACTIVE->cache_size_multiplier, total);
    addToCacheNew(zobrist, total);
    return total;
}
