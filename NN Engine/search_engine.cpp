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
#include <cstdio>

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

// Prune-verification diagnostic (Config::ENABLE_PRUNE_LOG, default off => byte-identical). Emits one stderr
// [PRUNEFIRE] record per sampled pruning fire for the offline verify+discriminator harness: the pruning site,
// the alpha/beta window + remaining depth (rd), the static eval the prune returned, a cheap (material+PST)
// eval for the eval-instability detector, and the FEN. Recording only; copies the board masks into locals
// (create_fen takes mutable refs) so the real board state is never touched -- same guard as dbg_bad_move.
static inline void log_prune_fire(const char *prune_id, const BoardState &st, int alpha, int beta, int rd,
                                  int static_eval, int cheap)
{
    static long seen = 0;
    if ((seen++ % std::max(1, Config::PRUNE_LOG_STRIDE)) != 0)
        return;
    uint64_t pawns = st.pawns, knights = st.knights, bishops = st.bishops, rooks = st.rooks,
             queens = st.queens, kings = st.kings, occupied = st.occupied,
             ow = st.occupied_colour[true], ob = st.occupied_colour[false], promoted = st.promoted,
             castling = st.castling_rights;
    int ep = st.ep_square;
    std::cerr << "[PRUNEFIRE] id=" << prune_id << " a=" << alpha << " b=" << beta << " rd=" << rd
              << " seval=" << static_eval << " cheap=" << cheap
              << " fen=" << create_fen(pawns, knights, bishops, rooks, queens, kings, occupied, ow, ob, promoted, castling, ep, st.turn)
              << std::endl;
}

// Correction-history SIGNAL logger (Config::ENABLE_CORRHIST_LOG, default off => byte-identical). Emits one
// stderr [CORRLOG] record per sampled update-eligible node: pawn-structure key, maxbit (1 = maximizer / root-
// side node, 0 = minimizer), the node-entry static eval (root-relative RFP frame), the node's backed-up best
// score (same frame), and remaining depth. staticEval and bestScore share the fixed root-relative frame, so
// (best - seval) is the correction-history training residual with NO sign flip. Recording only.
static inline void corrhist_log(uint64_t pawn_key, int maxbit, int static_eval, int best_score, int rd)
{
    static long seen = 0;
    if ((seen++ % std::max(1, Config::CORRHIST_LOG_STRIDE)) != 0)
        return;
    std::cerr << "[CORRLOG] " << pawn_key << ' ' << maxbit << ' ' << static_eval << ' ' << best_score
              << ' ' << rd << std::endl;
}

// Correction-history index for a position: pawn-structure key folded into the table (power-of-two mask).
static inline int corrhist_idx(const BoardState &st)
{
    return (int)(generatePawnKey(st.pawns, st.occupied_colour[true], st.occupied_colour[false]) & (CORR_SIZE - 1));
}
// The signed correction (millipawns) to ADD to the node-entry static eval, in the fixed root-relative frame.
static inline int corrhist_correction(const BoardState &st, int maxbit)
{
    return (pawnCorrHist[maxbit][corrhist_idx(st)] * Config::CORR_W) / Config::CORR_DIV;
}
// Learn: nudge the per-key EMA of the residual (bestScore - rawStaticEval) toward this node's observation.
static inline void corrhist_update(const BoardState &st, int maxbit, int static_eval, int best_score)
{
    int &e = pawnCorrHist[maxbit][corrhist_idx(st)];
    e += (best_score - static_eval - e) >> Config::CORR_SHIFT;
    if (e > Config::CORR_MAX)
        e = Config::CORR_MAX;
    else if (e < -Config::CORR_MAX)
        e = -Config::CORR_MAX;
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

// Per-path count of active singular extensions. Singular does not go through CheckExtensionGuard, so
// without its own counter a forced line of singular TT-moves could extend without bound (each singular
// move re-arms the next); this caps the depth of a pure singular chain independently of check extensions.
static int g_singular_extensions = 0;
struct SingularExtensionGuard
{
    bool active;
    explicit SingularExtensionGuard(bool a) : active(a)
    {
        if (active)
            ++g_singular_extensions;
    }
    ~SingularExtensionGuard()
    {
        if (active)
            --g_singular_extensions;
    }
};

// Optimism-triggered verification (OTV) path state. Single-threaded depth-first search, so
// file-scope counters track the current root-to-leaf path; the RAII guard keeps them balanced
// across every return path (cf. CheckExtensionGuard). g_verify_no_reduce_until = the deepest
// cur_depth at which reductions stay OFF inside an active verification window (-1 = inactive, so
// the do_lmr guards are inert and the default build is byte-identical). g_verify_count = OTV
// re-searches on the current path (the OTV_PATH_CAP budget). g_in_verify = inside a verification
// subtree (suppresses nested OTV so a confirmed tactic re-triggering at every ply cannot loop).
static int g_verify_no_reduce_until = -1;
static int g_verify_count = 0;
static bool g_in_verify = false;
// Cumulative OTV re-searches across the process (diagnostic only; never affects search). Only bumped
// when a verification actually fires, so it stays 0 in the default (ENABLE_OTV off) build.
static long g_otv_fires = 0;
struct VerifyGuard
{
    int prev_until;
    bool prev_in;
    VerifyGuard(int cur_depth, int plies)
    {
        prev_until = g_verify_no_reduce_until;
        prev_in = g_in_verify;
        g_verify_no_reduce_until = cur_depth + plies;
        g_in_verify = true;
        ++g_verify_count;
        ++g_otv_fires;
    }
    ~VerifyGuard()
    {
        g_verify_no_reduce_until = prev_until;
        g_in_verify = prev_in;
        --g_verify_count;
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

// Diagnostic (Config::ENABLE_CUTOFF_CLASS, default off): classify each beta-cutoff move so we can see WHAT the
// late-rank tail cutoffs actually are (already-LMP-exempt classes = capture/promo/killer/counter, vs plain
// quiets that tighter ordering could let a prune reach). Checks are folded into 'quiet' (a minor, conservative
// overcount — it under-states how many exempt tail cutoffs exist, so it never over-states the LMP unlock).
// Recording only; never alters the search.
enum
{
    CUTCLASS_CAPTURE = 0,
    CUTCLASS_PROMO,
    CUTCLASS_KILLER,
    CUTCLASS_COUNTER,
    CUTCLASS_QUIET,
    CUTCLASS_N
};
static long g_cutoff_class_hist[CUTCLASS_N][5] = {};
static inline int cutoff_move_class(const Move &m, const BoardState &st, int ply, const Move &prev)
{
    if (st.occupied_colour[!st.turn] & (1ULL << m.to_square))
        return CUTCLASS_CAPTURE;
    if (m.promotion > 1)
        return CUTCLASS_PROMO; // promotion==1 is the "no promotion" sentinel here
    if (ply >= 0 && ply < MAX_PLY &&
        ((killerMoves[ply][0].from_square == m.from_square && killerMoves[ply][0].to_square == m.to_square) ||
         (killerMoves[ply][1].from_square == m.from_square && killerMoves[ply][1].to_square == m.to_square)))
        return CUTCLASS_KILLER;
    const Move &cm = counterMoves[prev.from_square][prev.to_square];
    if (cm.from_square == m.from_square && cm.to_square == m.to_square)
        return CUTCLASS_COUNTER;
    return CUTCLASS_QUIET;
}
// Passed-pawn LMP/LMR exemption fires (diagnostic): how often an otherwise-reducible advanced pawn push
// was exempted from pruning/reduction. Cumulative across the run; printed beside the histogram.
static long g_passer_exempt_fires = 0;
// Per-move qsearch futility fire count. Its predecessor was believed inert with no counter to prove it;
// never claim a prune's behaviour again without one.
static long g_qdelta_permove_fires = 0;
static long g_qdelta_permove_seen = 0; // block REACHED (capture, non-ep, non-promo)
// Quiet moves dropped from the noisy list past the first q-ply under ENABLE_QCHECK_DEPTH0. The flag was
// read as inert from a byte-identical fingerprint alone; this makes the branch's reachability observable.
static long g_qcheck_d0_skipped = 0;
// TT-move ordering promotions actually performed. The predecessor read a table with no write site and was
// mistaken for a tested feature; this counter is what distinguishes "ran and did nothing" from "never ran".
static long g_tt_move_promotions = 0;
// Quiet checks actually admitted to the qsearch noisy list, at any q-ply, and those rejected by the
// safety filter. The pair gives the filter's selectivity directly.
static long g_q_quiet_checks_added = 0;
static long g_q_quiet_checks_unsafe = 0;
// Checks found by the full detector that the mask arm would have missed -- i.e. discoveries. Proves the
// capability increment is real instead of assuming the fuller test found something.
static long g_q_discovered_checks = 0;
// Direct checks the mask arm found but the full detector did NOT. Must stay at zero: the mask sees a strict
// subset of what the full test sees, so any hit here is a defect in moveGivesCheckFast.
static long g_q_checks_missed = 0;
// Node-exit TT stores actually performed (ENABLE_NODE_TT). The point of the feature is to fill
// TTEntry::move on a node's FIRST visit, so this must be large where the old cutoff-site write was 36.
static long g_node_tt_stores = 0;
// Nodes spent inside the root pre-search (reorder_legal_moves), cumulative. Against the total and the
// qsearch count this splits our per-depth node cost into main search / qsearch / pre-search, which is the
// question the SF depth race raised: we need ~62x SF's nodes for the same nominal depth, and ~18x of that
// is already present at shallow depth, so it is a fixed overhead rather than a growth-rate problem.
static long g_presearch_nodes = 0;
// Root moves whose ply-1 reply list is left EMPTY by a draw detected at cur_depth == 1. Both producers of a
// RootScore return before filling it: minimizer's repetition/50-move return (out_entry) and pre_minimizer's
// (pre_moves_list). alpha_beta then indexes scores[i].second_moves for every root move with no guard, so an
// empty list is a latent defect masked only by the draw re-firing identically on the next iteration.
static long g_draw_empty_ply1_min = 0;
static long g_draw_empty_ply1_pre = 0;
// Violations of reorder_legal_moves' documented contract that `scores` comes back full-length. Its two
// time-up bailouts return the PREVIOUS table, whose scores can be short (razor break) or empty (first
// iteration), while alpha_beta indexes scores[i] once per root move. Counted before deciding whether a
// repair is needed at all.
static long g_root_scores_short = 0;
// Tail handling (PRESEARCH_TAIL_MODE): root moves past prev_len + PRESEARCH_CHUNK. g_tail_* count how many
// were served each way, and g_razor_fires tracks the root-razor break -- the synthesised top_score used by
// mode 2 feeds `alpha - scores[i].top_score > razor_threshold` directly, so the razor rate is the first
// place a bad fill value shows up.
static long g_tail_full = 0;
static long g_tail_reduced = 0;
static long g_tail_heuristic = 0;
static long g_razor_fires = 0;
// Root moves the pre-search skipped because the previous iteration already scored them
// (ENABLE_PRESEARCH_SUBSET). Their pre-search RootScore is discarded by descending_sort_wrapper regardless.
static long g_prefix_skipped = 0;

// Root-razor SOUNDNESS audit. Within one iteration the razor can only discard low-scoring moves (the list
// is score-sorted), so a same-iteration check is vacuous. The real question is across iterations: if
// iteration k razors from index i onward, does iteration k+1 then pick a move that sat at index >= i in
// iteration k's ordering? That is the razor throwing away the eventual winner. Reset per position.
static std::vector<Move> g_prev_root_list;
static int g_prev_razor_idx = -1;
static long g_razor_audit_iters = 0;   // iterations preceded by a razoring iteration (denominator)
static long g_razor_cut_winner = 0;    // ... whose chosen move had been razored away
static long g_razor_cut_depth_sum = 0; // how far past the razor point the winner sat
// Root LMR activity: moves given a reduced scout, and how many of those beat alpha and forced the
// full-depth re-search. A high re-search rate means the reduction is too aggressive to be paying for itself.
static long g_root_lmr_reduced = 0;
static long g_root_lmr_researches = 0;
// Reductions declined because the move was the previous iteration's best (SF's best_move_count exemption).
// Counts the branch that ACTS, so g_root_lmr_reduced must fall by exactly this when the knob is enabled.
static long g_root_lmr_exempt = 0;

// NOTE: the 2026-07-30 diagnostic counters ([prune_pair], [lmr_remdepth], [lmr_guards], [corrhist_q],
// [qcache_hygiene], [qcache_hits]) are deliberately NOT in this tree. They cost ~15% NPS at byte-identical
// node counts via code layout under -Ofast -flto, not via executing the increments. Rebuild them from
// git history (see the 2026-07-30 commits) when a diagnosis needs them, and never time that build.

// Root-table coverage: how many entries the table carried, and how many of those held a score a search
// actually proved. verified/slots is the headline mechanism number -- ~34% on the push_back path (only
// searched moves get an entry), and expected near 100% once the table is pre-sized and kept.
static long g_root_table_slots = 0;
static long g_root_table_verified = 0;
static long g_root_table_has_real = 0;
// Razor decisions the recency guard declined: the entry existed but was a sentinel or too stale to prune on.
static long g_root_razor_stale_skips = 0;
// Hybrid razoring outcomes: hopeless moves still discarded vs suspect moves demoted to a reduced search.
// Counted on the branch that ACTS, so reduced+skipped must equal the razor fires under the hybrid path.
static long g_razor_hybrid_skipped = 0;
static long g_razor_hybrid_reduced = 0;
// Stale-but-suspect moves demoted from a full-depth search to a reduced one (ROOT_STALE_TO_LMR).
static long g_razor_stale_reduced = 0;

/*
    Writes one root move's result into its persistent table slot.

    SF stores a real value only when the move is the first searched or beats alpha; everything else becomes
    -VALUE_INFINITE, a marker for "unproven this iteration" that keeps its prior position under a stable
    sort and is never pruned on. The reply lists are search products either way, so they transfer even when
    the score does not -- an entry with an empty second_moves would be indexed unguarded by the next
    iteration.
*/
inline void root_table_store(RootScore &slot, int score, RootScore &&searched, bool proven, int searched_depth)
{
    if (!searched.second_moves.empty())
        slot.second_moves = std::move(searched.second_moves);
    if (!searched.second_scores.empty())
        slot.second_scores = std::move(searched.second_scores);

    // A fail-low score is MEASURED -- a real fail-soft value from a real search -- even though it proves
    // only an upper bound. The line razoring must respect is measured vs FABRICATED, not proven vs
    // unproven: the collapse cases all came from pruning on a fill value nothing ever searched. SF can
    // discard its fail-lows because it never razors at the root; we do, so the value is kept here and the
    // sort below still treats the move as unproven.
    slot.last_real = score;
    slot.last_real_depth = searched_depth;
    slot.age = 0;

    if (proven)
    {
        // SF's sort semantics: a real value only when the move is first or beat alpha, so that fail-lows
        // sink to the sentinel block and keep their prior relative order under the stable sort.
        slot.top_score = score;
        slot.verified = true;
        ++g_root_table_verified;
    }
    else
    {
        slot.top_score = ROOT_SCORE_UNPROVEN;
        slot.verified = false;
    }
}

/*
    Records the razor-soundness comparison on every alpha_beta exit. The comparison must run at EXIT (the
    chosen move is not known before then), and alpha_beta has eight returns, so a scope guard is used rather
    than instrumenting each one -- a missed return would bias the audit toward the completed case.
*/
struct RootRazorAudit
{
    const Move &best;
    const std::vector<Move> &cur_list;
    const int &razor_idx;
    ~RootRazorAudit()
    {
        if (g_prev_razor_idx >= 0 && !g_prev_root_list.empty())
        {
            ++g_razor_audit_iters;
            for (size_t k = 0; k < g_prev_root_list.size(); ++k)
            {
                if (g_prev_root_list[k].from_square == best.from_square && g_prev_root_list[k].to_square == best.to_square && g_prev_root_list[k].promotion == best.promotion)
                {
                    if (static_cast<int>(k) >= g_prev_razor_idx)
                    {
                        ++g_razor_cut_winner;
                        g_razor_cut_depth_sum += static_cast<int>(k) - g_prev_razor_idx;
                    }
                    break;
                }
            }
        }
        g_prev_root_list = cur_list;
        g_prev_razor_idx = razor_idx;
    }
};
// Phase-A qsearch ordering quality (diagnostic, cumulative across a run, like g_fh_*): cutoffs in the
// NOT-IN-CHECK qsearch loop, the fraction on the first noisy move (qfmc), and the summed cutoff move-index
// (qcut = avg index). Low qfmc / high qcut ⇒ a SEE re-sort of the noisy list should help.
static long g_q_fh_total = 0;
static long g_q_fh_first = 0;
static long g_q_cut_idx_sum = 0;

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
    constexpr int LP_LVL = 24;            // tree-level (cur_depth) cap
    constexpr int LP_MV = 48;             // move-number cap
    constexpr int LP_IT = 32;             // iterative-deepening iteration (depth_limit) cap
    constexpr int LP_CLOSE_MARGIN = 1000; // a pawn: fail-lows closer than this are near-misses

    struct LmrProfile
    {
        long reductions = 0, faillow = 0, failhigh = 0, faillow_close = 0;
        long red_lvl_mv[LP_LVL][LP_MV] = {};   // denominator by (tree level, move number)
        long low_lvl_mv[LP_LVL][LP_MV] = {};   // fail-low (drop) by (tree level, move number)
        long close_lvl_mv[LP_LVL][LP_MV] = {}; // near-miss subset
        long red_it_lvl[LP_IT][LP_LVL] = {};   // denominator by (ID iteration, tree level)
        long low_it_lvl[LP_IT][LP_LVL] = {};   // fail-low by (ID iteration, tree level)
        long low_hist[4] = {};                 // dropped move's history tier: 0:<=0 1:<4k 2:<32k 3:>=32k
        long low_pv = 0, low_nonpv = 0;        // dropped at PV (full-window) vs scout node
        long low_killer = 0;                   // dropped move was a killer/counter
        long low_piece[7] = {};                // dropped move's moving piece type (1=P..6=K)
        long low_phase[4] = {};                // game-phase bucket of the drop
        long bm_dropped = 0;                   // top-level (cur_depth==0) fail-low of the PROFILE_BM move
        long red_redux_sum = 0;                // sum of reduction plies over fail-lows (avg reduction when dropped)
    };
    LmrProfile g_lmr;

    // statScore distribution profile (diagnostic only, gated on ENABLE_STATSCORE_PROFILE): a coarse
    // histogram of the raw continuous statScore, used offline to derive our own OFFSET (median) and
    // DIVISOR (P95 spread) for the continuous statScore-LMR channel. Bucketed linearly so percentiles
    // are a cumulative walk; out-of-range samples clamp to the end buckets (counted separately).
    constexpr int SS_BUCKETS = 512;
    constexpr long SS_BUCKET_W = 1024;       // statScore units per bucket (percentile resolution)
    constexpr long SS_HALF = SS_BUCKETS / 2; // zero-centered: bucket b covers [(b-HALF)*W, (b-HALF+1)*W)
    struct StatScoreProfile
    {
        long hist[SS_BUCKETS] = {};
        long n = 0, clamped_lo = 0, clamped_hi = 0;
        long long sum = 0;
        long minv = (1L << 62), maxv = -(1L << 62);
    };
    StatScoreProfile g_ss;

    inline void statscore_profile_record(long s)
    {
        g_ss.n++;
        g_ss.sum += s;
        if (s < g_ss.minv)
            g_ss.minv = s;
        if (s > g_ss.maxv)
            g_ss.maxv = s;
        long idx = s / SS_BUCKET_W + SS_HALF;
        if (idx < 0)
        {
            g_ss.clamped_lo++;
            idx = 0;
        }
        else if (idx >= SS_BUCKETS)
        {
            g_ss.clamped_hi++;
            idx = SS_BUCKETS - 1;
        }
        g_ss.hist[idx]++;
    }

    // Lower-edge statScore value at cumulative fraction `frac` of the samples (percentile).
    inline long statscore_percentile(double frac)
    {
        if (g_ss.n == 0)
            return 0;
        long target = (long)(frac * g_ss.n);
        long cum = 0;
        for (int b = 0; b < SS_BUCKETS; ++b)
        {
            cum += g_ss.hist[b];
            if (cum >= target)
                return (b - SS_HALF) * SS_BUCKET_W;
        }
        return (SS_BUCKETS - SS_HALF) * SS_BUCKET_W;
    }

    // Prune-shadow wrong-prune tallies (diagnostic; gated on ENABLE_PRUNE_SHADOW). "enter" = the shadow
    // search of a pruned move would have entered the node's window (a wrong prune); "cut" = it would have
    // caused a cutoff (a strong wrong prune). Sampler is a deterministic 1-in-SHADOW_N counter, suppressed
    // during a shadow search (g_in_shadow) so shadows never nest.
    static constexpr int SHADOW_LVL = 16; // tree-level buckets for the LMR shadow breakdown

    struct ShadowProfile
    {
        long lmp_seen = 0, lmp_enter = 0, lmp_cut = 0;
        long fut_seen = 0, fut_enter = 0, fut_cut = 0;
        // LMR shadows: a move the search REDUCED and then DROPPED (no re-search) is re-run at full
        // depth and full window. "wrong" = the full-depth result would have improved this node's
        // bound, i.e. the reduction buried a move that mattered. Unlike the LMR-miss profiler (which
        // only reports that a reduced move failed to improve the bound -- the normal case for a late
        // move) this is an actual correctness measure.
        // [0] = maximizer node, [1] = minimizer node -- the wrong-rate spikes at some levels but not
        // their neighbours, so the two node types are tracked apart.
        long lmr_seen = 0, lmr_wrong = 0;
        long lmr_seen_lvl[2][SHADOW_LVL] = {{0}}, lmr_wrong_lvl[2][SHADOW_LVL] = {{0}};
        long sampler = 0;
    };
    ShadowProfile g_shadow;
    bool g_in_shadow = false;

    inline bool shadow_fire()
    {
        if (g_in_shadow || Config::SHADOW_N <= 0)
            return false;
        return (++g_shadow.sampler % Config::SHADOW_N) == 0;
    }

    // Record a shadow result. minimizer: entered window if shadow < beta, cutoff if shadow <= alpha.
    // maximizer (mirror): entered if shadow > alpha, cutoff if shadow >= beta.
    inline void shadow_record(bool is_lmp, bool minimizer, int shadow, int alpha, int beta)
    {
        bool entered = minimizer ? (shadow < beta) : (shadow > alpha);
        bool cut = minimizer ? (shadow <= alpha) : (shadow >= beta);
        if (is_lmp)
        {
            g_shadow.lmp_seen++;
            if (entered)
                g_shadow.lmp_enter++;
            if (cut)
                g_shadow.lmp_cut++;
        }
        else
        {
            g_shadow.fut_seen++;
            if (entered)
                g_shadow.fut_enter++;
            if (cut)
                g_shadow.fut_cut++;
        }
    }

    // Record one LMR shadow: `full` is the full-depth, full-window value of a move the search reduced
    // and then dropped. Same bound sense as shadow_record -- the reduction was WRONG when the honest
    // search would have improved the node's bound.
    inline void lmr_shadow_record(bool minimizer, int full, int alpha, int beta, int cur_depth)
    {
        bool wrong = minimizer ? (full < beta) : (full > alpha);
        int lvl = cur_depth < SHADOW_LVL ? (cur_depth < 0 ? 0 : cur_depth) : SHADOW_LVL - 1;
        int side = minimizer ? 1 : 0;
        g_shadow.lmr_seen++;
        g_shadow.lmr_seen_lvl[side][lvl]++;
        if (wrong)
        {
            g_shadow.lmr_wrong++;
            g_shadow.lmr_wrong_lvl[side][lvl]++;
        }
    }

    // Cutoff-calibration logger (diagnostic; ENABLE_CUTCAL_LOG). At each quiet beta-cutoff the labels are
    // known for free: the cutting quiet CUT, the tried-and-failed quiets (searched_quiets) FAILED. Bucket
    // both by their statScore (same sum the shipped statScore-LMR reads) to build P(cut | statScore) and the
    // 0-bucket composition -- does statScore=0 hold a large tried-and-failed population with low P(cut) that
    // malus could separate from never-tried? That is the measure-first gate for reviving gravity/malus.
    struct CutCalProfile
    {
        long cut[SS_BUCKETS] = {}, fail[SS_BUCKETS] = {};
        long tf_cut[4] = {}, tf_fail[4] = {};
    };
    CutCalProfile g_cutcal;
    long g_tf_count[2][64][64] = {}; // per-search tried-and-failed count per (side,from,to); reset each get_engine_move

    // Decoupled cut-rate table (Config::ENABLE_QCUT): gravity-bounded signed history read ONLY by
    // statScore-LMR, never by ordering. Reset per search. qcut_update = the standard gravity form,
    // parameterized bound so QCUT stays independent of the ordering tables' MAX_HISTORY.
    int g_qcut[2][64][64] = {};
    inline void qcut_update(int &h, int delta, int bound)
    {
        if (delta > bound)
            delta = bound;
        else if (delta < -bound)
            delta = -bound;
        int ad = delta < 0 ? -delta : delta;
        h += delta - h * ad / bound;
    }

    inline long cutcal_statscore(const Move &m, const Move &pm, const Move &p2, bool turn, const BoardState &st)
    {
        long s = historyHeuristics[turn][m.from_square][m.to_square];
        if (pm.from_square != pm.to_square)
            s += counterMoveHeuristics[turn][cont_ctx_key(pm, st)][cont_ent_key(m, st)];
        if (p2.from_square != p2.to_square)
            s += contHist2[turn][cont_ctx_key(p2, st)][cont_ent_key(m, st)];
        return s;
    }

    inline void cutcal_record(long statScore, bool cut)
    {
        long idx = statScore / SS_BUCKET_W + SS_HALF;
        if (idx < 0)
            idx = 0;
        else if (idx >= SS_BUCKETS)
            idx = SS_BUCKETS - 1;
        if (cut)
            g_cutcal.cut[idx]++;
        else
            g_cutcal.fail[idx]++;
    }

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
                                     const Move &move, const BoardState &cs, const Move &prevMove,
                                     bool at_minimizer)
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

    // The drop/cutoff sense is side-dependent (this engine is non-negamax, with separate
    // minimizer/maximizer). At a MIN node the bound tightened is beta, so score <= alpha
    // drives beta <= alpha = a beta CUTOFF (harmless); the move that merely fails to improve
    // the min is the one at score >= beta. At a MAX node the senses are the other way round.
    // Scoring both sides with the maximizer test counted min-node cutoffs as drops and made
    // the by-level drop-rate alternate ~95%/~8% purely as an artifact.
    bool cutoff = at_minimizer ? (score <= alpha) : (score >= beta);
    if (cutoff)
    {
        g_lmr.failhigh++; // a cutoff, harmless
        return;
    }

    // The reduced move did not improve this node's bound: the DROP case, where a win can be lost.
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
                               : (cs.bishops & fb)   ? 3
                               : (cs.rooks & fb)     ? 4
                               : (cs.queens & fb)    ? 5
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
    BoardState cs = state_history.back();
    // Cheap surrogate (material + PST): improving needs only the trend SIGN, not an accurate value, so skip
    // the full eval's expensive machinery. Does NOT touch the full eval cache -- mixing full and cheap values
    // across the two compared plies would corrupt the trend, so the cheap path is self-contained.
    if (Config::IMPROVING_CHEAP)
    {
        int total = cheap_eval(cs.pawns, cs.knights, cs.bishops, cs.rooks, cs.queens, cs.kings,
                               cs.occupied_colour[true], cs.occupied_colour[false]);
        if (Config::side_to_play)
            total = -total;
        return total;
    }
    int cached;
    if (accessCacheNew(zobrist, cached))
        return cached;
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

// Eval-mode dispatch for the quiescent decision sites (futility, qsearch stand-pat/horizon).
// mode 0 = full get_board_evaluation (byte-identical); 1 = cheap (material+PST surrogate); 2 = light
// (full eval minus the heavy dynamic terms, uncached via g_eval_light). Default mode 0 everywhere.
inline int eval_by_mode(int mode, std::vector<BoardState> &state_history, uint64_t zobrist, int &num_iterations)
{
    if (mode == 1)
    {
        BoardState cs = state_history.back();
        int t = cheap_eval(cs.pawns, cs.knights, cs.bishops, cs.rooks, cs.queens, cs.kings,
                           cs.occupied_colour[true], cs.occupied_colour[false]);
        return Config::side_to_play ? -t : t;
    }
    if (mode == 2)
    {
        g_eval_light = true;
        int t = get_board_evaluation(state_history, zobrist, num_iterations);
        g_eval_light = false;
        return t;
    }
    return get_board_evaluation(state_history, zobrist, num_iterations);
}

inline int history_lmr_delta(const Move &move, const Move &previousMove, const BoardState &cs, int ply)
{
    // Continuous statScore channel: a graded, two-sided generalization of the tiered logic below. Sum the
    // history tables into one statScore and map it smoothly to a signed reduction delta (positive =
    // reduce-less, negative = reduce-more). Env-gated; off = the tiered path runs unchanged (byte-identical).
    if (Config::ENABLE_STATSCORE_LMR)
    {
        long statScore = (long)Config::STATSCORE_MAIN_W * historyHeuristics[cs.turn][move.from_square][move.to_square];
        if (previousMove.from_square != previousMove.to_square)
            statScore += (long)Config::STATSCORE_CONT1_W *
                         counterMoveHeuristics[cs.turn][cont_ctx_key(previousMove, cs)]
                                              [cont_ent_key(move, cs)];
        if (ply >= 2)
        {
            Move p2 = g_searchStack[ply - 2];
            if (p2.from_square != p2.to_square)
                statScore += (long)Config::STATSCORE_CONT2_W *
                             contHist2[cs.turn][cont_ctx_key(p2, cs)]
                                      [cont_ent_key(move, cs)];
        }
        if (Config::ENABLE_QCUT)
            statScore += (long)Config::QCUT_LAMBDA * g_qcut[cs.turn][move.from_square][move.to_square] / 256;
        if (Config::ENABLE_STATSCORE_PROFILE)
            statscore_profile_record(statScore);
        int delta = (int)((statScore - Config::STATSCORE_OFFSET) / Config::STATSCORE_DIVISOR);
        // Structural overlay (independent knob): a killer at this ply or the counter to the previous move
        // gets extra reduce-less on top of the continuous delta. 0 = pure-continuous.
        if (Config::STATSCORE_KILLER_BONUS > 0 &&
            (killerMoves[ply][0] == move || killerMoves[ply][1] == move ||
             counterMoves[previousMove.from_square][previousMove.to_square] == move))
            delta += Config::STATSCORE_KILLER_BONUS;
        return std::clamp(delta, -Config::STATSCORE_CLAMP, Config::STATSCORE_CLAMP);
    }

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
            counterMoveHeuristics[cs.turn][cont_ctx_key(previousMove, cs)]
                                 [cont_ent_key(move, cs)] >= Config::CONT_HIST_LMR_THRESH)
            return 0;
        // A strong 2-ply continuation (move 2 plies back x this move) also rescues a tier-0 quiet
        // from reduce-more. Gated; reuses the 1-ply threshold.
        if (Config::ENABLE_CONT_HIST_2PLY && ply >= 2)
        {
            Move p2 = g_searchStack[ply - 2];
            if (p2.from_square != p2.to_square &&
                contHist2[cs.turn][cont_ctx_key(p2, cs)]
                         [cont_ent_key(move, cs)] >= Config::CONT_HIST_LMR_THRESH)
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

// Dump the statScore distribution (diagnostic; gated on ENABLE_STATSCORE_PROFILE). Read the median (P50)
// as STATSCORE_OFFSET and DIVISOR ~= round((P97.5 - P2.5)/2 / 1.5) to seed the continuous channel.
static void statscore_profile_dump()
{
    std::cerr << "\n===== statScore profile =====\n";
    std::cerr << "n=" << g_ss.n;
    if (g_ss.n)
    {
        std::cerr << "  mean=" << (double)g_ss.sum / g_ss.n
                  << "  min=" << g_ss.minv << "  max=" << g_ss.maxv
                  << "  P2.5=" << statscore_percentile(0.025)
                  << "  P50=" << statscore_percentile(0.5)
                  << "  P97.5=" << statscore_percentile(0.975)
                  << "  clamped(lo/hi)=" << g_ss.clamped_lo << "/" << g_ss.clamped_hi;
        long p50 = statscore_percentile(0.5);
        long spread = (statscore_percentile(0.975) - statscore_percentile(0.025)) / 2;
        std::cerr << "\nsuggest: STATSCORE_OFFSET=" << p50
                  << " STATSCORE_DIVISOR=" << (long)(spread / 1.5 + 0.5);
    }
    std::cerr << "\n=============================\n";
}

// Dump per-mechanism wrong-prune rates (diagnostic; gated on ENABLE_PRUNE_SHADOW). "enter%" = fraction of
// sampled pruned moves that would have entered the node window; "cut%" = fraction that would have cut.
static void shadow_profile_dump()
{
    std::cerr << "\n===== prune-shadow wrong-prune rates (1/" << Config::SHADOW_N << " sampled) =====\n";
    if (g_shadow.lmp_seen)
        std::cerr << "LMP:      sampled=" << g_shadow.lmp_seen
                  << "  enter=" << g_shadow.lmp_enter << " (" << (100.0 * g_shadow.lmp_enter / g_shadow.lmp_seen) << "%)"
                  << "  cut=" << g_shadow.lmp_cut << " (" << (100.0 * g_shadow.lmp_cut / g_shadow.lmp_seen) << "%)\n";
    if (g_shadow.fut_seen)
        std::cerr << "Futility: sampled=" << g_shadow.fut_seen
                  << "  enter=" << g_shadow.fut_enter << " (" << (100.0 * g_shadow.fut_enter / g_shadow.fut_seen) << "%)"
                  << "  cut=" << g_shadow.fut_cut << " (" << (100.0 * g_shadow.fut_cut / g_shadow.fut_seen) << "%)\n";
    if (g_shadow.lmr_seen)
    {
        std::cerr << "LMR-drop: sampled=" << g_shadow.lmr_seen
                  << "  wrong=" << g_shadow.lmr_wrong
                  << " (" << (100.0 * g_shadow.lmr_wrong / g_shadow.lmr_seen) << "%)\n";
        std::cerr << "  by tree-level (cur_depth) x node type: level  MAX(sampled wrong rate%)  MIN(sampled wrong rate%)\n";
        for (int l = 0; l < SHADOW_LVL; ++l)
        {
            long smax = g_shadow.lmr_seen_lvl[0][l], wmax = g_shadow.lmr_wrong_lvl[0][l];
            long smin = g_shadow.lmr_seen_lvl[1][l], wmin = g_shadow.lmr_wrong_lvl[1][l];
            if (!smax && !smin)
                continue;
            std::cerr << "    L" << l << ":  max " << smax << " " << wmax << " "
                      << (smax ? 100.0 * wmax / smax : 0.0) << "%   min " << smin << " " << wmin << " "
                      << (smin ? 100.0 * wmin / smin : 0.0) << "%\n";
        }
    }
    std::cerr << "=========================================================\n";
}

// Dump the reliability curve P(cut | statScore) + the 0-bucket composition (diagnostic; ENABLE_CUTCAL_LOG).
// If the [0,1024) band holds a large low-P(cut) fail population, malus (which would push those repeated
// failures negative) has calibration information the current non-negative history cannot express.
static void cutcal_profile_dump()
{
    long tc = 0, tf = 0;
    for (int b = 0; b < SS_BUCKETS; ++b)
    {
        tc += g_cutcal.cut[b];
        tf += g_cutcal.fail[b];
    }
    long n = tc + tf;
    std::cerr << "\n===== cutoff-calibration  P(cut | statScore) =====\n";
    std::cerr << "samples=" << n << "  cut=" << tc << "  fail=" << tf
              << "  overall_P(cut)=" << (n ? (double)tc / n : 0.0) << "\n";
    const long thr[6] = {0, 1024, 2048, 4096, 8192, 16384};
    long bcut[6] = {}, bfail[6] = {}, negc = 0, negf = 0;
    for (int b = 0; b < SS_BUCKETS; ++b)
    {
        long edge = (long)(b - SS_HALF) * SS_BUCKET_W;
        if (edge < 0)
        {
            negc += g_cutcal.cut[b];
            negf += g_cutcal.fail[b];
            continue;
        }
        int band = 0;
        for (int k = 0; k < 6; ++k)
            if (edge >= thr[k])
                band = k;
        bcut[band] += g_cutcal.cut[b];
        bfail[band] += g_cutcal.fail[b];
    }
    if (negc + negf)
        std::cerr << "  statScore<0      : n=" << (negc + negf) << "  P(cut)=" << (double)negc / (negc + negf) << "\n";
    for (int k = 0; k < 6; ++k)
    {
        long bn = bcut[k] + bfail[k];
        if (bn)
            std::cerr << "  statScore>=" << thr[k] << "\t: n=" << bn << "  P(cut)=" << (double)bcut[k] / bn << "\n";
    }
    long zc = g_cutcal.cut[SS_HALF], zf = g_cutcal.fail[SS_HALF];
    std::cerr << "0-bucket [0,1024): cut=" << zc << " fail=" << zf
              << "  P(cut)=" << ((zc + zf) ? (double)zc / (zc + zf) : 0.0)
              << "   (this fail count = tried-and-failed reading ~0 = malus's target; "
              << (tf ? 100.0 * zf / tf : 0.0) << "% of ALL fails)\n";
    std::cerr << "0-bucket split by tried-fail count -- the malus test (does more-failed => lower P(cut)?):\n";
    for (int t = 0; t < 4; ++t)
    {
        long bn = g_cutcal.tf_cut[t] + g_cutcal.tf_fail[t];
        if (bn)
            std::cerr << "  tf=" << t << (t == 3 ? "+" : "") << "\t: n=" << bn
                      << "  P(cut)=" << (double)g_cutcal.tf_cut[t] / bn << "\n";
    }
    std::cerr << "=================================================\n";
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
        Config::ROOT_RAZOR_CONTINUE = env_flag("ROOT_RAZOR_CONTINUE", false);
        Config::RAZOR_BASE_FIRST = env_int("RAZOR_BASE_FIRST", Config::RAZOR_BASE_FIRST);
        Config::RAZOR_FLOOR_FIRST = env_int("RAZOR_FLOOR_FIRST", Config::RAZOR_FLOOR_FIRST);
        Config::RAZOR_BASE = env_int("RAZOR_BASE", Config::RAZOR_BASE);
        Config::RAZOR_FLOOR = env_int("RAZOR_FLOOR", Config::RAZOR_FLOOR);
        Config::RAZOR_DECAY_PCT = env_int("RAZOR_DECAY_PCT", Config::RAZOR_DECAY_PCT);
        Config::RESIGN_THRESHOLD = env_int("RESIGN_THRESHOLD", Config::RESIGN_THRESHOLD);
        Config::ENABLE_TT_STORE_DRAW = env_flag("ENABLE_TT_STORE_DRAW", false);
        Config::ENABLE_NULLMOVE = env_flag("ENABLE_NULLMOVE", true);
        Config::NULLMOVE_PROGRESSIVE = env_flag("NULLMOVE_PROGRESSIVE", Config::NULLMOVE_PROGRESSIVE);
        Config::NULLMOVE_EXTRA = env_int("NULLMOVE_EXTRA", Config::NULLMOVE_EXTRA);
        Config::ENABLE_QDELTA = env_flag("ENABLE_QDELTA", true);
        Config::DELTA_MARGIN = env_int("DELTA_MARGIN", Config::DELTA_MARGIN);
        Config::MAX_QDEPTH = env_int("MAX_QDEPTH", Config::MAX_QDEPTH);
        Config::LMR_PROFILE = env_flag("LMR_PROFILE", false);
        Config::LMR_REM_FLOOR_PCT = env_int("LMR_REM_FLOOR_PCT", Config::LMR_REM_FLOOR_PCT);
        Config::LMR_MIN_REM = env_int("LMR_MIN_REM", Config::LMR_MIN_REM);
        Config::ENABLE_LMR_REMDEPTH = env_flag("ENABLE_LMR_REMDEPTH", Config::ENABLE_LMR_REMDEPTH);
        Config::LMR_REMDEPTH_SCALE = env_int("LMR_REMDEPTH_SCALE", Config::LMR_REMDEPTH_SCALE);
        Config::PROTECT_KILLERS = env_flag("PROTECT_KILLERS", false);
        Config::PROTECT_PV = env_flag("PROTECT_PV", false);
        Config::PROTECT_MAX_IDX = env_int("PROTECT_MAX_IDX", Config::PROTECT_MAX_IDX);
        // History-aware LMR (default off = byte-identical). CAP = plies removed for good quiets;
        // MORE_CAP = plies added for never-cut quiets (0 = reduce-less only, the prior behavior).
        Config::ENABLE_HISTORY_LMR = env_flag("ENABLE_HISTORY_LMR", Config::ENABLE_HISTORY_LMR);
        Config::HISTORY_LMR_CAP = env_int("HISTORY_LMR_CAP", Config::HISTORY_LMR_CAP);
        Config::HISTORY_LMR_MORE_CAP = env_int("HISTORY_LMR_MORE_CAP", Config::HISTORY_LMR_MORE_CAP);
        Config::HISTORY_LMR_SCALE = env_int("HISTORY_LMR_SCALE", Config::HISTORY_LMR_SCALE);
        Config::HISTORY_LMR_SCALE_CAP = env_int("HISTORY_LMR_SCALE_CAP", Config::HISTORY_LMR_SCALE_CAP);
        // Continuous statScore-LMR (default off = byte-identical; the tiered path above runs unchanged).
        Config::ENABLE_STATSCORE_LMR = env_flag("ENABLE_STATSCORE_LMR", Config::ENABLE_STATSCORE_LMR);
        Config::STATSCORE_OFFSET = env_int("STATSCORE_OFFSET", Config::STATSCORE_OFFSET);
        Config::STATSCORE_DIVISOR = env_int("STATSCORE_DIVISOR", Config::STATSCORE_DIVISOR);
        if (Config::STATSCORE_DIVISOR < 1)
            Config::STATSCORE_DIVISOR = 1;
        Config::STATSCORE_CLAMP = env_int("STATSCORE_CLAMP", Config::STATSCORE_CLAMP);
        Config::STATSCORE_MAIN_W = env_int("STATSCORE_MAIN_W", Config::STATSCORE_MAIN_W);
        Config::STATSCORE_CONT1_W = env_int("STATSCORE_CONT1_W", Config::STATSCORE_CONT1_W);
        Config::STATSCORE_CONT2_W = env_int("STATSCORE_CONT2_W", Config::STATSCORE_CONT2_W);
        Config::STATSCORE_KILLER_BONUS = env_int("STATSCORE_KILLER_BONUS", Config::STATSCORE_KILLER_BONUS);
        Config::ENABLE_STATSCORE_PROFILE = env_flag("ENABLE_STATSCORE_PROFILE", Config::ENABLE_STATSCORE_PROFILE);
        Config::ENABLE_PRUNE_SHADOW = env_flag("ENABLE_PRUNE_SHADOW", Config::ENABLE_PRUNE_SHADOW);
        Config::SHADOW_N = env_int("SHADOW_N", Config::SHADOW_N);
        Config::ENABLE_CUTCAL_LOG = env_flag("ENABLE_CUTCAL_LOG", Config::ENABLE_CUTCAL_LOG);
        Config::ENABLE_QCUT = env_flag("ENABLE_QCUT", Config::ENABLE_QCUT);
        Config::QCUT_LAMBDA = env_int("QCUT_LAMBDA", Config::QCUT_LAMBDA);
        Config::QCUT_MAX = env_int("QCUT_MAX", Config::QCUT_MAX);
        if (Config::QCUT_MAX < 1)
            Config::QCUT_MAX = 1;
        Config::QCUT_MALUS_DIV = env_int("QCUT_MALUS_DIV", Config::QCUT_MALUS_DIV);
        if (Config::QCUT_MALUS_DIV < 1)
            Config::QCUT_MALUS_DIV = 1;
        Config::ENABLE_LMR_CAPCHAIN = env_flag("ENABLE_LMR_CAPCHAIN", Config::ENABLE_LMR_CAPCHAIN);
        Config::CAPCHAIN_REDUCE_LESS = env_int("CAPCHAIN_REDUCE_LESS", Config::CAPCHAIN_REDUCE_LESS);
        Config::CAPCHAIN_RUN_THRESH = env_int("CAPCHAIN_RUN_THRESH", Config::CAPCHAIN_RUN_THRESH);
        Config::ENABLE_LMP = env_flag("ENABLE_LMP", Config::ENABLE_LMP);
        Config::ROOT_PRESEARCH_REDUCTION = env_int("ROOT_PRESEARCH_REDUCTION", Config::ROOT_PRESEARCH_REDUCTION);
        Config::ENABLE_ROOT_LMR = env_flag("ENABLE_ROOT_LMR", Config::ENABLE_ROOT_LMR);
        Config::ROOT_LMR_MIN_IDX = env_int("ROOT_LMR_MIN_IDX", Config::ROOT_LMR_MIN_IDX);
        Config::ROOT_LMR_BASE = env_int("ROOT_LMR_BASE", Config::ROOT_LMR_BASE);
        Config::ROOT_LMR_DIV = env_int("ROOT_LMR_DIV", Config::ROOT_LMR_DIV);
        Config::ROOT_LMR_EXEMPT_BEST = env_flag("ROOT_LMR_EXEMPT_BEST", Config::ROOT_LMR_EXEMPT_BEST);
        Config::ENABLE_ROOT_TABLE = env_flag("ENABLE_ROOT_TABLE", Config::ENABLE_ROOT_TABLE);
        Config::ROOT_RAZOR_MAX_AGE = env_int("ROOT_RAZOR_MAX_AGE", Config::ROOT_RAZOR_MAX_AGE);
        if (Config::ROOT_RAZOR_MAX_AGE < 0)
            Config::ROOT_RAZOR_MAX_AGE = 0;
        Config::ROOT_SORT_L2_LASTREAL = env_flag("ROOT_SORT_L2_LASTREAL", Config::ROOT_SORT_L2_LASTREAL);
        Config::ROOT_SORT_L1_LASTREAL = env_flag("ROOT_SORT_L1_LASTREAL", Config::ROOT_SORT_L1_LASTREAL);
        Config::ROOT_RAZOR_TO_LMR = env_flag("ROOT_RAZOR_TO_LMR", Config::ROOT_RAZOR_TO_LMR);
        Config::ROOT_RAZOR_SKIP_MARGIN = env_int("ROOT_RAZOR_SKIP_MARGIN", Config::ROOT_RAZOR_SKIP_MARGIN);
        Config::ROOT_RAZOR_LMR_BASE = env_int("ROOT_RAZOR_LMR_BASE", Config::ROOT_RAZOR_LMR_BASE);
        Config::ROOT_RAZOR_LMR_DIV = env_int("ROOT_RAZOR_LMR_DIV", Config::ROOT_RAZOR_LMR_DIV);
        if (Config::ROOT_RAZOR_LMR_DIV < 1)
            Config::ROOT_RAZOR_LMR_DIV = 1;
        Config::ROOT_RAZOR_MAX_DEPTH_DEFICIT = env_int("ROOT_RAZOR_MAX_DEPTH_DEFICIT", Config::ROOT_RAZOR_MAX_DEPTH_DEFICIT);
        Config::ROOT_STALE_TO_LMR = env_flag("ROOT_STALE_TO_LMR", Config::ROOT_STALE_TO_LMR);
        Config::ROOT_STALE_LMR_BASE = env_int("ROOT_STALE_LMR_BASE", Config::ROOT_STALE_LMR_BASE);
        Config::ROOT_STALE_LMR_DIV = env_int("ROOT_STALE_LMR_DIV", Config::ROOT_STALE_LMR_DIV);
        if (Config::ROOT_STALE_LMR_DIV < 1)
            Config::ROOT_STALE_LMR_DIV = 1;
        Config::ENABLE_ROOT_RAZOR = env_flag("ENABLE_ROOT_RAZOR", Config::ENABLE_ROOT_RAZOR);
        Config::PRESEARCH_OFF_FROM_DEPTH = env_int("PRESEARCH_OFF_FROM_DEPTH", Config::PRESEARCH_OFF_FROM_DEPTH);
        Config::ENABLE_PRESEARCH_SUBSET = env_flag("ENABLE_PRESEARCH_SUBSET", Config::ENABLE_PRESEARCH_SUBSET);
        Config::PRESEARCH_SUBSET_ALPHA_MARGIN = env_int("PRESEARCH_SUBSET_ALPHA_MARGIN", Config::PRESEARCH_SUBSET_ALPHA_MARGIN);
        Config::PRESEARCH_TAIL_MODE = env_int("PRESEARCH_TAIL_MODE", Config::PRESEARCH_TAIL_MODE);
        Config::PRESEARCH_CHUNK = env_int("PRESEARCH_CHUNK", Config::PRESEARCH_CHUNK);
        Config::PRESEARCH_TAIL_REDUCTION = env_int("PRESEARCH_TAIL_REDUCTION", Config::PRESEARCH_TAIL_REDUCTION);
        Config::ENABLE_ROOT_PRESEARCH = env_flag("ENABLE_ROOT_PRESEARCH", Config::ENABLE_ROOT_PRESEARCH);
        Config::ENABLE_SEE_PRUNE = env_flag("ENABLE_SEE_PRUNE", Config::ENABLE_SEE_PRUNE);
        Config::SEE_PRUNE_MARGIN = env_int("SEE_PRUNE_MARGIN", Config::SEE_PRUNE_MARGIN);
        Config::SEE_PRUNE_MAX_DEPTH = env_int("SEE_PRUNE_MAX_DEPTH", Config::SEE_PRUNE_MAX_DEPTH);
        Config::SEE_PRUNE_CAPTURES = env_flag("SEE_PRUNE_CAPTURES", Config::SEE_PRUNE_CAPTURES);
        Config::SEE_PRUNE_CAPTURE_MARGIN = env_int("SEE_PRUNE_CAPTURE_MARGIN", Config::SEE_PRUNE_CAPTURE_MARGIN);
        Config::LMP_MAX_DEPTH = env_int("LMP_MAX_DEPTH", Config::LMP_MAX_DEPTH);
        Config::LMP_BASE = env_int("LMP_BASE", Config::LMP_BASE);
        Config::LMP_SCALE = env_int("LMP_SCALE", Config::LMP_SCALE);
        Config::ENABLE_LMP_HIST_EXEMPT = env_flag("ENABLE_LMP_HIST_EXEMPT", Config::ENABLE_LMP_HIST_EXEMPT);
        Config::LMP_HIST_EXEMPT = env_int("LMP_HIST_EXEMPT", Config::LMP_HIST_EXEMPT);
        Config::ENABLE_HIST_PRUNE = env_flag("ENABLE_HIST_PRUNE", Config::ENABLE_HIST_PRUNE);
        Config::HIST_PRUNE_COEF = env_int("HIST_PRUNE_COEF", Config::HIST_PRUNE_COEF);
        Config::HIST_PRUNE_MAX_DEPTH = env_int("HIST_PRUNE_MAX_DEPTH", Config::HIST_PRUNE_MAX_DEPTH);
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
        Config::ENABLE_THREATS = env_flag("ENABLE_THREATS", Config::ENABLE_THREATS);
        Config::SCALE_THREATS = env_int("SCALE_THREATS", Config::SCALE_THREATS);
        Config::THREATS_STANDING_ONLY = env_flag("THREATS_STANDING_ONLY", Config::THREATS_STANDING_ONLY);
        Config::THREAT_PER_TARGET_CAP = env_int("THREAT_PER_TARGET_CAP", Config::THREAT_PER_TARGET_CAP);
        Config::THREAT_SAFE_PAWN = env_int("THREAT_SAFE_PAWN", Config::THREAT_SAFE_PAWN);
        Config::THREATS_QUIET_PCT = env_int("THREATS_QUIET_PCT", Config::THREATS_QUIET_PCT);
        Config::THREATS_TENSION_LO = env_int("THREATS_TENSION_LO", Config::THREATS_TENSION_LO);
        Config::THREATS_TENSION_HI = env_int("THREATS_TENSION_HI", Config::THREATS_TENSION_HI);
        Config::SCALE_CENTRAL = env_int("SCALE_CENTRAL", Config::SCALE_CENTRAL);
        Config::SCALE_CAPTURE_GAINS = env_int("SCALE_CAPTURE_GAINS", Config::SCALE_CAPTURE_GAINS);
        Config::ENABLE_CAPG_COND = env_flag("ENABLE_CAPG_COND", Config::ENABLE_CAPG_COND);
        Config::CAPG_TENSION_LO = env_int("CAPG_TENSION_LO", Config::CAPG_TENSION_LO);
        Config::CAPG_TENSION_HI = env_int("CAPG_TENSION_HI", Config::CAPG_TENSION_HI);
        Config::CAPG_LO_SCALE = env_int("CAPG_LO_SCALE", Config::CAPG_LO_SCALE);
        Config::CAPG_HI_SCALE = env_int("CAPG_HI_SCALE", Config::CAPG_HI_SCALE);
        Config::ENABLE_CAPG_REALIZ = env_flag("ENABLE_CAPG_REALIZ", Config::ENABLE_CAPG_REALIZ);
        Config::ENABLE_CAPG_PIN = env_flag("ENABLE_CAPG_PIN", Config::ENABLE_CAPG_PIN);
        Config::ENABLE_CAPG_TEMPO = env_flag("ENABLE_CAPG_TEMPO", Config::ENABLE_CAPG_TEMPO);
        Config::PP_OPP_PAWN_PEN = env_int("PP_OPP_PAWN_PEN", Config::PP_OPP_PAWN_PEN);
        Config::PP_BLOCKADE_PEN = env_int("PP_BLOCKADE_PEN", Config::PP_BLOCKADE_PEN);
        Config::PP_UNBLOCKED = env_int("PP_UNBLOCKED", Config::PP_UNBLOCKED);
        Config::PP_DIAG_SUPPORT = env_int("PP_DIAG_SUPPORT", Config::PP_DIAG_SUPPORT);
        Config::PP_FILE_CLEAR = env_int("PP_FILE_CLEAR", Config::PP_FILE_CLEAR);
        Config::PP_HORIZ_SUPPORT = env_int("PP_HORIZ_SUPPORT", Config::PP_HORIZ_SUPPORT);
        Config::PAWN_MAJORITY_MAG_MG = env_int("PAWN_MAJORITY_MAG_MG", Config::PAWN_MAJORITY_MAG_MG);
        Config::PAWN_MAJORITY_MAG_EG = env_int("PAWN_MAJORITY_MAG_EG", Config::PAWN_MAJORITY_MAG_EG);
        Config::PAWN_MAJORITY_ADV_K = env_int("PAWN_MAJORITY_ADV_K", Config::PAWN_MAJORITY_ADV_K);
        Config::PAWN_MAJORITY_OUTSIDE_K = env_int("PAWN_MAJORITY_OUTSIDE_K", Config::PAWN_MAJORITY_OUTSIDE_K);
        Config::PAWN_MAJORITY_BLOCKADE_K = env_int("PAWN_MAJORITY_BLOCKADE_K", Config::PAWN_MAJORITY_BLOCKADE_K);
        Config::ISOLATED_PAWN_PEN = env_int("ISOLATED_PAWN_PEN", Config::ISOLATED_PAWN_PEN);
        Config::BACKWARD_PAWN_PEN = env_int("BACKWARD_PAWN_PEN", Config::BACKWARD_PAWN_PEN);
        Config::OUTPOST_KNIGHT = env_int("OUTPOST_KNIGHT", Config::OUTPOST_KNIGHT);
        Config::OUTPOST_BISHOP = env_int("OUTPOST_BISHOP", Config::OUTPOST_BISHOP);
        Config::SPACE_MAG = env_int("SPACE_MAG", Config::SPACE_MAG);
        Config::SPACE_KNIGHT_MAG = env_int("SPACE_KNIGHT_MAG", Config::SPACE_KNIGHT_MAG);
        Config::SPACE_PHASE_MAX = env_int("SPACE_PHASE_MAX", Config::SPACE_PHASE_MAX);
        Config::SCALE_PLACE_PAWN = env_int("SCALE_PLACE_PAWN", Config::SCALE_PLACE_PAWN);
        Config::SCALE_PLACE_KNIGHT = env_int("SCALE_PLACE_KNIGHT", Config::SCALE_PLACE_KNIGHT);
        Config::SCALE_PLACE_BISHOP = env_int("SCALE_PLACE_BISHOP", Config::SCALE_PLACE_BISHOP);
        Config::SCALE_PLACE_QUEEN = env_int("SCALE_PLACE_QUEEN", Config::SCALE_PLACE_QUEEN);
        Config::SCALE_PLACE_KING_EG = env_int("SCALE_PLACE_KING_EG", Config::SCALE_PLACE_KING_EG);
        Config::CENTER_INNER_MULT = env_int("CENTER_INNER_MULT", Config::CENTER_INNER_MULT);
        Config::CENTER_OUTER_MULT = env_int("CENTER_OUTER_MULT", Config::CENTER_OUTER_MULT);
        Config::IMBALANCE_SCALE = env_int("IMBALANCE_SCALE", Config::IMBALANCE_SCALE);
        Config::BISHOP_PAIR_BONUS = env_int("BISHOP_PAIR_BONUS", Config::BISHOP_PAIR_BONUS);
        Config::KNIGHT_PAIR_BONUS = env_int("KNIGHT_PAIR_BONUS", Config::KNIGHT_PAIR_BONUS);
        Config::ROOK_OPEN_BASE = env_int("ROOK_OPEN_BASE", Config::ROOK_OPEN_BASE);
        Config::ROOK_OPEN_CAP = env_int("ROOK_OPEN_CAP", Config::ROOK_OPEN_CAP);
        Config::ROOK_7TH = env_int("ROOK_7TH", Config::ROOK_7TH);
        Config::ROOK_CONNECTED = env_int("ROOK_CONNECTED", Config::ROOK_CONNECTED);
        Config::ROOK_SEMI = env_int("ROOK_SEMI", Config::ROOK_SEMI);
        Config::ROOK_PASSER_OWN = env_int("ROOK_PASSER_OWN", Config::ROOK_PASSER_OWN);
        Config::ROOK_PASSER_ENEMY = env_int("ROOK_PASSER_ENEMY", Config::ROOK_PASSER_ENEMY);
        Config::ROOK_OWN_PAWN_BASE = env_int("ROOK_OWN_PAWN_BASE", Config::ROOK_OWN_PAWN_BASE);
        Config::ROOK_OWN_PAWN_RAMP = env_int("ROOK_OWN_PAWN_RAMP", Config::ROOK_OWN_PAWN_RAMP);
        Config::ROOK_ENEMY_PAWN_PEN = env_int("ROOK_ENEMY_PAWN_PEN", Config::ROOK_ENEMY_PAWN_PEN);
        Config::ROOK_MINOR_BLOCK = env_int("ROOK_MINOR_BLOCK", Config::ROOK_MINOR_BLOCK);
        Config::ROOK_ROOK_BLOCK = env_int("ROOK_ROOK_BLOCK", Config::ROOK_ROOK_BLOCK);
        Config::ROOK_SEMI_CONNECTED = env_int("ROOK_SEMI_CONNECTED", Config::ROOK_SEMI_CONNECTED);
        Config::ENABLE_ROOK_TENSION_COND = env_flag("ENABLE_ROOK_TENSION_COND", Config::ENABLE_ROOK_TENSION_COND);
        Config::ROOK_COND_TENSION_LO = env_int("ROOK_COND_TENSION_LO", Config::ROOK_COND_TENSION_LO);
        Config::ROOK_COND_TENSION_HI = env_int("ROOK_COND_TENSION_HI", Config::ROOK_COND_TENSION_HI);
        Config::ROOK_COND_QUIET_SCALE = env_int("ROOK_COND_QUIET_SCALE", Config::ROOK_COND_QUIET_SCALE);
        Config::SCALE_PAWN_RANK = env_int("SCALE_PAWN_RANK", Config::SCALE_PAWN_RANK);
        Config::SCALE_PASSED_RANK = env_int("SCALE_PASSED_RANK", Config::SCALE_PASSED_RANK);
        Config::SCALE_ENDGAME_RANK = env_int("SCALE_ENDGAME_RANK", Config::SCALE_ENDGAME_RANK);
        // Per-rank pawn-table scales, ranks 2..7 (indices 1..6; 0 and 7 are unreachable squares for a pawn).
        // Names are RANK_DEF_R2..R7 / RANK_PSD_R2..R7 / RANK_EG_R2..R7 so each is an ordinary KNOB=VAL the
        // fit harness can sweep individually alongside every other eval knob.
        for (int i = 1; i <= 6; ++i) {
            char nm[24];
            std::snprintf(nm, sizeof nm, "RANK_DEF_R%d", i + 1);
            Config::RANK_DEF_PCT[i] = env_int(nm, Config::RANK_DEF_PCT[i]);
            std::snprintf(nm, sizeof nm, "RANK_PSD_R%d", i + 1);
            Config::RANK_PSD_PCT[i] = env_int(nm, Config::RANK_PSD_PCT[i]);
            std::snprintf(nm, sizeof nm, "RANK_EG_R%d", i + 1);
            Config::RANK_EG_PCT[i] = env_int(nm, Config::RANK_EG_PCT[i]);
        }
        // Per-file chain/wall scales. CHAIN_F_A..H index files 0..7; WALL_F_A..H map to the wall array's
        // live entries 1..8 (it is indexed [x] and [x+2], so 0 and 9/10 are padding).
        for (int i = 0; i < 8; ++i) {
            char nm[24];
            std::snprintf(nm, sizeof nm, "CHAIN_F_%c", 'A' + i);
            Config::CHAIN_F_PCT[i] = env_int(nm, Config::CHAIN_F_PCT[i]);
            std::snprintf(nm, sizeof nm, "WALL_F_%c", 'A' + i);
            Config::WALL_F_PCT[i + 1] = env_int(nm, Config::WALL_F_PCT[i + 1]);
        }
        // Per-rank structural sensitivity, one curve per PHASE (the two evaluators are independent).
        for (int i = 1; i <= 6; ++i) {
            char nm[24];
            std::snprintf(nm, sizeof nm, "STRUCT_R_MG_R%d", i + 1);
            Config::STRUCT_R_MG_PCT[i] = env_int(nm, Config::STRUCT_R_MG_PCT[i]);
            std::snprintf(nm, sizeof nm, "STRUCT_R_EG_R%d", i + 1);
            Config::STRUCT_R_EG_PCT[i] = env_int(nm, Config::STRUCT_R_EG_PCT[i]);
        }
        Config::STRUCT_OPPOSED_MG_PCT = env_int("STRUCT_OPPOSED_MG_PCT", Config::STRUCT_OPPOSED_MG_PCT);
        Config::STRUCT_OPPOSED_EG_PCT = env_int("STRUCT_OPPOSED_EG_PCT", Config::STRUCT_OPPOSED_EG_PCT);
        Config::PAWN_CLAMP_MID = env_int("PAWN_CLAMP_MID", Config::PAWN_CLAMP_MID);
        Config::PAWN_CLAMP_EG  = env_int("PAWN_CLAMP_EG",  Config::PAWN_CLAMP_EG);
        Config::EG_PHALANX = env_int("EG_PHALANX", Config::EG_PHALANX);
        Config::EG_SUPPORT = env_int("EG_SUPPORT", Config::EG_SUPPORT);
        Config::EG_DEFEND  = env_int("EG_DEFEND",  Config::EG_DEFEND);
        Config::EG_LATENT  = env_int("EG_LATENT",  Config::EG_LATENT);
        Config::EG_EXIST_KNIGHT = env_int("EG_EXIST_KNIGHT", Config::EG_EXIST_KNIGHT);
        Config::EG_EXIST_BISHOP = env_int("EG_EXIST_BISHOP", Config::EG_EXIST_BISHOP);
        Config::EG_EXIST_ROOK   = env_int("EG_EXIST_ROOK",   Config::EG_EXIST_ROOK);
        Config::EG_EXIST_QUEEN  = env_int("EG_EXIST_QUEEN",  Config::EG_EXIST_QUEEN);
        Config::MG_CLAMP_KNIGHT   = env_int("MG_CLAMP_KNIGHT",   Config::MG_CLAMP_KNIGHT);
        Config::MG_CLAMP_BISHOP_A = env_int("MG_CLAMP_BISHOP_A", Config::MG_CLAMP_BISHOP_A);
        Config::MG_CLAMP_BISHOP_B = env_int("MG_CLAMP_BISHOP_B", Config::MG_CLAMP_BISHOP_B);
        Config::EG_CLAMP_KNIGHT = env_int("EG_CLAMP_KNIGHT", Config::EG_CLAMP_KNIGHT);
        Config::EG_CLAMP_BISHOP = env_int("EG_CLAMP_BISHOP", Config::EG_CLAMP_BISHOP);
        Config::EG_CLAMP_ROOK   = env_int("EG_CLAMP_ROOK",   Config::EG_CLAMP_ROOK);
        Config::EG_CLAMP_QUEEN  = env_int("EG_CLAMP_QUEEN",  Config::EG_CLAMP_QUEEN);
        Config::ENABLE_WINNABILITY = env_flag("ENABLE_WINNABILITY", Config::ENABLE_WINNABILITY);
        Config::WINNAB_PASSED     = env_int("WINNAB_PASSED",     Config::WINNAB_PASSED);
        Config::WINNAB_PAWNS      = env_int("WINNAB_PAWNS",      Config::WINNAB_PAWNS);
        Config::WINNAB_OUTFLANK   = env_int("WINNAB_OUTFLANK",   Config::WINNAB_OUTFLANK);
        Config::WINNAB_FLANKS     = env_int("WINNAB_FLANKS",     Config::WINNAB_FLANKS);
        Config::WINNAB_INFILT     = env_int("WINNAB_INFILT",     Config::WINNAB_INFILT);
        Config::WINNAB_NO_NPM     = env_int("WINNAB_NO_NPM",     Config::WINNAB_NO_NPM);
        Config::WINNAB_UNWINNABLE = env_int("WINNAB_UNWINNABLE", Config::WINNAB_UNWINNABLE);
        Config::WINNAB_TENSION    = env_int("WINNAB_TENSION",    Config::WINNAB_TENSION);
        Config::WINNAB_BASE       = env_int("WINNAB_BASE",       Config::WINNAB_BASE);
        Config::WINNAB_MG_OFFSET  = env_int("WINNAB_MG_OFFSET",  Config::WINNAB_MG_OFFSET);
        Config::WINNAB_SCALE      = env_int("WINNAB_SCALE",      Config::WINNAB_SCALE);
        Config::ENABLE_CLOSEDNESS = env_flag("ENABLE_CLOSEDNESS", Config::ENABLE_CLOSEDNESS);
        for (int i = 0; i < 9; i++){
            char nm[24];
            snprintf(nm, sizeof(nm), "CLOSED_N%d", i);
            Config::CLOSED_N[i] = env_int(nm, Config::CLOSED_N[i]);
            snprintf(nm, sizeof(nm), "CLOSED_R%d", i);
            Config::CLOSED_R[i] = env_int(nm, Config::CLOSED_R[i]);
            snprintf(nm, sizeof(nm), "CLOSED_B%d", i);
            Config::CLOSED_B_PCT[i] = env_int(nm, Config::CLOSED_B_PCT[i]);
        }
        Config::PHASE_BLEND_LO    = env_int("PHASE_BLEND_LO",    Config::PHASE_BLEND_LO);
        Config::PHASE_BLEND_RANGE = env_int("PHASE_BLEND_RANGE", Config::PHASE_BLEND_RANGE);
        Config::PPS_OWN_BLOCK    = env_int("PPS_OWN_BLOCK",    Config::PPS_OWN_BLOCK);
        Config::PPS_ENEMY_BLOCK  = env_int("PPS_ENEMY_BLOCK",  Config::PPS_ENEMY_BLOCK);
        Config::PPS_OWN_ATTACK   = env_int("PPS_OWN_ATTACK",   Config::PPS_OWN_ATTACK);
        Config::PPS_ENEMY_ATTACK = env_int("PPS_ENEMY_ATTACK", Config::PPS_ENEMY_ATTACK);
        Config::ENABLE_PASSER_DEFER_ON_FLAG = env_flag("ENABLE_PASSER_DEFER_ON_FLAG", Config::ENABLE_PASSER_DEFER_ON_FLAG);
        Config::PASSER_R_MAX = env_int("PASSER_R_MAX", Config::PASSER_R_MAX);
        Config::ENABLE_PAWN_OBSTRUCTION_BLEND = env_flag("ENABLE_PAWN_OBSTRUCTION_BLEND", Config::ENABLE_PAWN_OBSTRUCTION_BLEND);
        Config::PAWN_OBS_LO = env_int("PAWN_OBS_LO", Config::PAWN_OBS_LO);
        Config::PAWN_OBS_HI = env_int("PAWN_OBS_HI", Config::PAWN_OBS_HI);
        Config::PAWN_OBS_LO_EG = env_int("PAWN_OBS_LO_EG", Config::PAWN_OBS_LO_EG);
        Config::PAWN_OBS_HI_EG = env_int("PAWN_OBS_HI_EG", Config::PAWN_OBS_HI_EG);
        Config::SCALE_PAWN_WALL = env_int("SCALE_PAWN_WALL", Config::SCALE_PAWN_WALL);
        Config::SCALE_PAWN_CHAIN = env_int("SCALE_PAWN_CHAIN", Config::SCALE_PAWN_CHAIN);
        Config::BISHOP_MOB_PAWN_ATTACK = env_int("BISHOP_MOB_PAWN_ATTACK", Config::BISHOP_MOB_PAWN_ATTACK);
        Config::BISHOP_MOB_SECONDARY = env_int("BISHOP_MOB_SECONDARY", Config::BISHOP_MOB_SECONDARY);
        Config::THREAT_ATTACK_MULT = env_int("THREAT_ATTACK_MULT", Config::THREAT_ATTACK_MULT);
        Config::THREAT_PRESENCE_MULT = env_int("THREAT_PRESENCE_MULT", Config::THREAT_PRESENCE_MULT);
        Config::THREAT_PAWN = env_int("THREAT_PAWN", Config::THREAT_PAWN);
        Config::THREAT_KNIGHT = env_int("THREAT_KNIGHT", Config::THREAT_KNIGHT);
        Config::THREAT_BISHOP = env_int("THREAT_BISHOP", Config::THREAT_BISHOP);
        Config::THREAT_ROOK = env_int("THREAT_ROOK", Config::THREAT_ROOK);
        Config::THREAT_QUEEN = env_int("THREAT_QUEEN", Config::THREAT_QUEEN);
        Config::SCALE_ATTACK_LAYER = env_int("SCALE_ATTACK_LAYER", Config::SCALE_ATTACK_LAYER);
        Config::ATTACK_OPEN_MULT = env_int("ATTACK_OPEN_MULT", Config::ATTACK_OPEN_MULT);
        Config::KING_SAFETY_MAG = env_int("KING_SAFETY_MAG", Config::KING_SAFETY_MAG);
        Config::ENABLE_KS_REPLACE_LT = env_flag("ENABLE_KS_REPLACE_LT", Config::ENABLE_KS_REPLACE_LT);
        Config::KS_CONSOLIDATE = env_flag("KS_CONSOLIDATE", Config::KS_CONSOLIDATE);
        Config::ENABLE_KS_V2 = env_flag("ENABLE_KS_V2", Config::ENABLE_KS_V2);
        Config::KS_SHELTER_FULL = env_int("KS_SHELTER_FULL", Config::KS_SHELTER_FULL);
        Config::KS_SHELTER_PARTIAL = env_int("KS_SHELTER_PARTIAL", Config::KS_SHELTER_PARTIAL);
        Config::KS_SHELTER_MAG = env_int("KS_SHELTER_MAG", Config::KS_SHELTER_MAG);
        Config::KS_LIGHT_MAG = env_int("KS_LIGHT_MAG", Config::KS_LIGHT_MAG);
        Config::KS_ATT_KNIGHT = env_int("KS_ATT_KNIGHT", Config::KS_ATT_KNIGHT);
        Config::KS_ATT_BISHOP = env_int("KS_ATT_BISHOP", Config::KS_ATT_BISHOP);
        Config::KS_ATT_ROOK = env_int("KS_ATT_ROOK", Config::KS_ATT_ROOK);
        Config::KS_ATT_QUEEN = env_int("KS_ATT_QUEEN", Config::KS_ATT_QUEEN);
        Config::KS_ATTACK_COUNT = env_int("KS_ATTACK_COUNT", Config::KS_ATTACK_COUNT);
        Config::KS_MIN_ATTACKERS = env_int("KS_MIN_ATTACKERS", Config::KS_MIN_ATTACKERS);
        Config::KS_ATT_PRODUCT = env_int("KS_ATT_PRODUCT", Config::KS_ATT_PRODUCT);
        Config::KS_OVERLOAD = env_int("KS_OVERLOAD", Config::KS_OVERLOAD);
        Config::KS_WEAK = env_int("KS_WEAK", Config::KS_WEAK);
        Config::KS_SAFE_CHECK = env_int("KS_SAFE_CHECK", Config::KS_SAFE_CHECK);
        Config::KS_SAFE_CHECK_DEF = env_int("KS_SAFE_CHECK_DEF", Config::KS_SAFE_CHECK_DEF);
        Config::ENABLE_KS_CHECK_V2 = env_flag("ENABLE_KS_CHECK_V2", Config::ENABLE_KS_CHECK_V2);
        Config::KS_CHK_QUEEN = env_int("KS_CHK_QUEEN", Config::KS_CHK_QUEEN);
        Config::KS_CHK_ROOK = env_int("KS_CHK_ROOK", Config::KS_CHK_ROOK);
        Config::KS_CHK_BISHOP = env_int("KS_CHK_BISHOP", Config::KS_CHK_BISHOP);
        Config::KS_CHK_KNIGHT = env_int("KS_CHK_KNIGHT", Config::KS_CHK_KNIGHT);
        Config::KS_CHK_MULTI = env_int("KS_CHK_MULTI", Config::KS_CHK_MULTI);
        Config::ENABLE_KS_AIM = env_flag("ENABLE_KS_AIM", Config::ENABLE_KS_AIM);
        Config::KS_AIM_BISHOP = env_int("KS_AIM_BISHOP", Config::KS_AIM_BISHOP);
        Config::KS_AIM_ROOK = env_int("KS_AIM_ROOK", Config::KS_AIM_ROOK);
        Config::KS_AIM_QUEEN = env_int("KS_AIM_QUEEN", Config::KS_AIM_QUEEN);
        Config::KS_DEF_MAG = env_int("KS_DEF_MAG", Config::KS_DEF_MAG);
        Config::KS_STORM = env_int("KS_STORM", Config::KS_STORM);
        Config::KS_OPEN_FILE = env_int("KS_OPEN_FILE", Config::KS_OPEN_FILE);
        Config::KS_BATTERY = env_int("KS_BATTERY", Config::KS_BATTERY);
        Config::KS_ZONE2 = env_int("KS_ZONE2", Config::KS_ZONE2);
        Config::ENABLE_KS_ZONE_CLAMP = env_flag("ENABLE_KS_ZONE_CLAMP", Config::ENABLE_KS_ZONE_CLAMP);
        Config::KS_ZONE_NORM = env_int("KS_ZONE_NORM", Config::KS_ZONE_NORM);
        Config::KS_ZONE_ATTACK_PCT = env_int("KS_ZONE_ATTACK_PCT", Config::KS_ZONE_ATTACK_PCT);
        Config::KS_CLAMP_SHELTER = env_int("KS_CLAMP_SHELTER", Config::KS_CLAMP_SHELTER);
        Config::KS_DYN = env_int("KS_DYN", Config::KS_DYN);
        Config::KS_DYN_PIVOT = env_int("KS_DYN_PIVOT", Config::KS_DYN_PIVOT);
        Config::KS_DYN_SHIFT = env_int("KS_DYN_SHIFT", Config::KS_DYN_SHIFT);
        Config::KS_SHIELD = env_int("KS_SHIELD", Config::KS_SHIELD);
        Config::KS_DEFENDER = env_int("KS_DEFENDER", Config::KS_DEFENDER);
        Config::KS_INTERACT = env_int("KS_INTERACT", Config::KS_INTERACT);
        Config::KS_DIVISOR = env_int("KS_DIVISOR", Config::KS_DIVISOR);
        Config::KS_KNEE = env_int("KS_KNEE", Config::KS_KNEE);
        Config::KS_CAP = env_int("KS_CAP", Config::KS_CAP);
        Config::KS_FLOOR = env_int("KS_FLOOR", Config::KS_FLOOR);
        Config::KS_PHASE_FULL = env_int("KS_PHASE_FULL", Config::KS_PHASE_FULL);
        Config::KS_PHASE_ZERO = env_int("KS_PHASE_ZERO", Config::KS_PHASE_ZERO);
        Config::KS_NO_QUEEN = env_int("KS_NO_QUEEN", Config::KS_NO_QUEEN);
        Config::ENABLE_KS_SF_WEAK = env_flag("ENABLE_KS_SF_WEAK", Config::ENABLE_KS_SF_WEAK);
        Config::ENABLE_KS_SF_SAFECHECK = env_flag("ENABLE_KS_SF_SAFECHECK", Config::ENABLE_KS_SF_SAFECHECK);
        Config::REALIZ_MAT_K = env_int("REALIZ_MAT_K", Config::REALIZ_MAT_K);
        Config::REALIZ_MAT_THRESH = env_int("REALIZ_MAT_THRESH", Config::REALIZ_MAT_THRESH);
        Config::REALIZ_PHASE_K = env_int("REALIZ_PHASE_K", Config::REALIZ_PHASE_K);
        Config::REALIZ_FLOOR = env_int("REALIZ_FLOOR", Config::REALIZ_FLOOR);
        Config::PASSER_BLOCK_ADV = env_int("PASSER_BLOCK_ADV", Config::PASSER_BLOCK_ADV);
        Config::PASSER_ENEMY_CREDIT_PCT = env_int("PASSER_ENEMY_CREDIT_PCT", Config::PASSER_ENEMY_CREDIT_PCT);
        Config::ENABLE_PASSER_DANGER = env_flag("ENABLE_PASSER_DANGER", Config::ENABLE_PASSER_DANGER);
        Config::PASSER_DANGER_BASE1 = env_int("PASSER_DANGER_BASE1", Config::PASSER_DANGER_BASE1);
        Config::PASSER_DANGER_BASE2 = env_int("PASSER_DANGER_BASE2", Config::PASSER_DANGER_BASE2);
        Config::PASSER_DANGER_BASE3 = env_int("PASSER_DANGER_BASE3", Config::PASSER_DANGER_BASE3);
        Config::PASSER_DANGER_D2 = env_int("PASSER_DANGER_D2", Config::PASSER_DANGER_D2);
        Config::PASSER_DANGER_D4 = env_int("PASSER_DANGER_D4", Config::PASSER_DANGER_D4);
        Config::PASSER_CONTEST_STOP = env_int("PASSER_CONTEST_STOP", Config::PASSER_CONTEST_STOP);
        Config::PASSER_CONTEST_PATH = env_int("PASSER_CONTEST_PATH", Config::PASSER_CONTEST_PATH);
        Config::PASSER_REAR_ENEMY = env_int("PASSER_REAR_ENEMY", Config::PASSER_REAR_ENEMY);
        Config::PASSER_REAR_OWN = env_int("PASSER_REAR_OWN", Config::PASSER_REAR_OWN);
        Config::PASSER_KING_FAR = env_int("PASSER_KING_FAR", Config::PASSER_KING_FAR);
        Config::PASSER_KING_HELP = env_int("PASSER_KING_HELP", Config::PASSER_KING_HELP);
        Config::PASSER_MAG_SCALE = env_int("PASSER_MAG_SCALE", Config::PASSER_MAG_SCALE);
        Config::PASSER_R_CAP = env_int("PASSER_R_CAP", Config::PASSER_R_CAP);
        Config::PASSER_R_FLOOR = env_int("PASSER_R_FLOOR", Config::PASSER_R_FLOOR);
        Config::PASSER_RFLOOR_R5 = env_int("PASSER_RFLOOR_R5", Config::PASSER_RFLOOR_R5);
        Config::PASSER_RFLOOR_R6 = env_int("PASSER_RFLOOR_R6", Config::PASSER_RFLOOR_R6);
        Config::PASSER_RESID_PCT = env_int("PASSER_RESID_PCT", Config::PASSER_RESID_PCT);
        Config::ENABLE_PASSER_ORD_FLOOR = env_flag("ENABLE_PASSER_ORD_FLOOR", Config::ENABLE_PASSER_ORD_FLOOR);
        Config::ENABLE_KS_DEBUG = env_flag("ENABLE_KS_DEBUG", Config::ENABLE_KS_DEBUG);
        Config::ENABLE_PASSER_KRACE_MG = env_flag("ENABLE_PASSER_KRACE_MG", Config::ENABLE_PASSER_KRACE_MG);
        Config::PASSER_KRACE_MG_PCT = env_int("PASSER_KRACE_MG_PCT", Config::PASSER_KRACE_MG_PCT);
        Config::ENABLE_PASSER_V2 = env_flag("ENABLE_PASSER_V2", Config::ENABLE_PASSER_V2);
        Config::ENABLE_PASSER_V3 = env_flag("ENABLE_PASSER_V3", Config::ENABLE_PASSER_V3);
        Config::CAPG_PAWN_RANK_CLAMP = env_int("CAPG_PAWN_RANK_CLAMP", Config::CAPG_PAWN_RANK_CLAMP);
        Config::ENABLE_KAUFMAN_IMBALANCE = env_flag("ENABLE_KAUFMAN_IMBALANCE", Config::ENABLE_KAUFMAN_IMBALANCE);
        Config::KAUFMAN_SCALE = env_int("KAUFMAN_SCALE", Config::KAUFMAN_SCALE);
        Config::ENABLE_PASSER_BLOCKADE_QUALITY = env_flag("ENABLE_PASSER_BLOCKADE_QUALITY", Config::ENABLE_PASSER_BLOCKADE_QUALITY);
        Config::PASSER_CONTEST_PCT = env_int("PASSER_CONTEST_PCT", Config::PASSER_CONTEST_PCT);
        Config::PASSER_KRACE_MAG = env_int("PASSER_KRACE_MAG", Config::PASSER_KRACE_MAG);
        Config::PV_BOOST_MAG = env_int("PV_BOOST_MAG", Config::PV_BOOST_MAG);
        Config::PV_BOOST_TRIGGER = env_int("PV_BOOST_TRIGGER", Config::PV_BOOST_TRIGGER);
        Config::PV_BOOST_PHASE_K = env_int("PV_BOOST_PHASE_K", Config::PV_BOOST_PHASE_K);
        Config::MOD_FLOOR = env_int("MOD_FLOOR", Config::MOD_FLOOR);
        Config::MOD_CEIL = env_int("MOD_CEIL", Config::MOD_CEIL);
        Config::MOD_MAT_PAWNS = env_int("MOD_MAT_PAWNS", Config::MOD_MAT_PAWNS);
        Config::MOD_MAT_OPPB = env_int("MOD_MAT_OPPB", Config::MOD_MAT_OPPB);
        Config::MOD_LT_BACKING = env_int("MOD_LT_BACKING", Config::MOD_LT_BACKING);
        Config::MOD_PAIR_OPEN = env_int("MOD_PAIR_OPEN", Config::MOD_PAIR_OPEN);
        Config::MOD_KS_BACKING = env_int("MOD_KS_BACKING", Config::MOD_KS_BACKING);
        Config::MOD_KS_CONTROL = env_int("MOD_KS_CONTROL", Config::MOD_KS_CONTROL);
        Config::MOD_KS_REALIZ = env_int("MOD_KS_REALIZ", Config::MOD_KS_REALIZ);
        Config::KS_REALIZ_FLOOR = env_int("KS_REALIZ_FLOOR", Config::KS_REALIZ_FLOOR);
        Config::MOD_PVBOOST_COMP = env_int("MOD_PVBOOST_COMP", Config::MOD_PVBOOST_COMP);
        Config::MOD_PVBOOST_MOB = env_int("MOD_PVBOOST_MOB", Config::MOD_PVBOOST_MOB);
        Config::MOD_PIECES_LEVEL = env_int("MOD_PIECES_LEVEL", Config::MOD_PIECES_LEVEL);
        Config::MOD_PIECES_MAT_THRESH = env_int("MOD_PIECES_MAT_THRESH", Config::MOD_PIECES_MAT_THRESH);
        Config::MOD_PIECES_FLOOR = env_int("MOD_PIECES_FLOOR", Config::MOD_PIECES_FLOOR);
        Config::MOD_PIECES_CONTROL = env_int("MOD_PIECES_CONTROL", Config::MOD_PIECES_CONTROL);
        Config::MOD_PIECES_DEFEND = env_int("MOD_PIECES_DEFEND", Config::MOD_PIECES_DEFEND);
        Config::MOD_PIECES_DEFEND_THRESH = env_int("MOD_PIECES_DEFEND_THRESH", Config::MOD_PIECES_DEFEND_THRESH);
        Config::ENABLE_NPEDGE_DAMP = env_flag("ENABLE_NPEDGE_DAMP", Config::ENABLE_NPEDGE_DAMP);
        Config::NPEDGE_DAMP_LO = env_int("NPEDGE_DAMP_LO", Config::NPEDGE_DAMP_LO);
        Config::NPEDGE_DAMP_HI = env_int("NPEDGE_DAMP_HI", Config::NPEDGE_DAMP_HI);
        Config::NPEDGE_DAMP_MAX = env_int("NPEDGE_DAMP_MAX", Config::NPEDGE_DAMP_MAX);
        Config::NPEDGE_DAMP_TQUIET = env_int("NPEDGE_DAMP_TQUIET", Config::NPEDGE_DAMP_TQUIET);
        Config::ENABLE_NPEDGE_DAMP_EG = env_flag("ENABLE_NPEDGE_DAMP_EG", Config::ENABLE_NPEDGE_DAMP_EG);
        Config::NPEDGE_EG_PIECE_FLOOR = env_int("NPEDGE_EG_PIECE_FLOOR", Config::NPEDGE_EG_PIECE_FLOOR);
        Config::ENABLE_MOBILITY = env_flag("ENABLE_MOBILITY", Config::ENABLE_MOBILITY);
        Config::MOBILITY_SCALE = env_int("MOBILITY_SCALE", Config::MOBILITY_SCALE);
        rebuild_scaled_placement();   // rebuild scaled placement working arrays once from the loaded SCALE_PLACE_* knobs (no per-read division in eval)
        rebuild_scaled_pawn_tables(); // rebuild scaled pawn-structure working arrays from the loaded SCALE_PAWN_* knobs (no per-read division in eval)
        rebuild_ks_tables();          // rebuild the king-safety non-linear danger table + phase-taper from the loaded KS_* knobs (no per-eval division)
        Config::ENABLE_CHEAP_BISHOP_COMPLEX = env_flag("ENABLE_CHEAP_BISHOP_COMPLEX", Config::ENABLE_CHEAP_BISHOP_COMPLEX);
        Config::CHEAP_BISHOP_BLOCK = env_int("CHEAP_BISHOP_BLOCK", Config::CHEAP_BISHOP_BLOCK);
        Config::CHEAP_BISHOP_MOB = env_int("CHEAP_BISHOP_MOB", Config::CHEAP_BISHOP_MOB);
        Config::CHEAP_BISHOP_FWD = env_int("CHEAP_BISHOP_FWD", Config::CHEAP_BISHOP_FWD);
        Config::CHEAP_BISHOP_KING = env_int("CHEAP_BISHOP_KING", Config::CHEAP_BISHOP_KING);
        Config::ENABLE_PIECE_MOBILITY = env_flag("ENABLE_PIECE_MOBILITY", Config::ENABLE_PIECE_MOBILITY);
        Config::SCALE_MOBILITY = env_int("SCALE_MOBILITY", Config::SCALE_MOBILITY);
        Config::ENABLE_CHEAP_ROOK_MOBILITY = env_flag("ENABLE_CHEAP_ROOK_MOBILITY", Config::ENABLE_CHEAP_ROOK_MOBILITY);
        Config::CHEAP_ROOK_MOB = env_int("CHEAP_ROOK_MOB", Config::CHEAP_ROOK_MOB);
        Config::CHEAP_ROOK_FWD = env_int("CHEAP_ROOK_FWD", Config::CHEAP_ROOK_FWD);
        Config::ENABLE_CHEAP_QUEEN_MOBILITY = env_flag("ENABLE_CHEAP_QUEEN_MOBILITY", Config::ENABLE_CHEAP_QUEEN_MOBILITY);
        Config::CHEAP_QUEEN_MOB_MG = env_int("CHEAP_QUEEN_MOB_MG", Config::CHEAP_QUEEN_MOB_MG);
        Config::CHEAP_QUEEN_MOB_EG = env_int("CHEAP_QUEEN_MOB_EG", Config::CHEAP_QUEEN_MOB_EG);
        Config::ENABLE_CHEAP_KNIGHT_MOBILITY = env_flag("ENABLE_CHEAP_KNIGHT_MOBILITY", Config::ENABLE_CHEAP_KNIGHT_MOBILITY);
        Config::CHEAP_KNIGHT_MOB = env_int("CHEAP_KNIGHT_MOB", Config::CHEAP_KNIGHT_MOB);
        // Full per-piece mobility REPLACES the cheap rook/knight/queen surrogates (avoid double-counting mobility).
        if (Config::ENABLE_PIECE_MOBILITY)
        {
            Config::ENABLE_CHEAP_ROOK_MOBILITY = false;
            Config::ENABLE_CHEAP_QUEEN_MOBILITY = false;
            Config::ENABLE_CHEAP_KNIGHT_MOBILITY = false;
        }
        Config::ENABLE_ATTACK_LAYER_CACHE = env_flag("ENABLE_ATTACK_LAYER_CACHE", Config::ENABLE_ATTACK_LAYER_CACHE);
        Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME = env_flag("ENABLE_ATTACK_LAYER_CACHE_MIDGAME", Config::ENABLE_ATTACK_LAYER_CACHE_MIDGAME);
        Config::ENABLE_SEE_FIX = env_flag("ENABLE_SEE_FIX", Config::ENABLE_SEE_FIX);
        Config::ENABLE_SEE_INCREMENTAL = env_flag("ENABLE_SEE_INCREMENTAL", Config::ENABLE_SEE_INCREMENTAL);
        Config::FUTILITY_EVAL_MODE = env_int("FUTILITY_EVAL_MODE", Config::FUTILITY_EVAL_MODE);
        Config::QSTANDPAT_EVAL_MODE = env_int("QSTANDPAT_EVAL_MODE", Config::QSTANDPAT_EVAL_MODE);
        Config::QDELTA_PERMOVE_MARGIN = env_int("QDELTA_PERMOVE_MARGIN", Config::QDELTA_PERMOVE_MARGIN);
        Config::ENABLE_QDELTA_PERMOVE = env_flag("ENABLE_QDELTA_PERMOVE", Config::ENABLE_QDELTA_PERMOVE);
        Config::ENABLE_RFP = env_flag("ENABLE_RFP", Config::ENABLE_RFP);
        Config::RFP_MARGIN = env_int("RFP_MARGIN", Config::RFP_MARGIN);
        Config::RFP_MIN_DEPTH = env_int("RFP_MIN_DEPTH", Config::RFP_MIN_DEPTH);
        Config::RFP_MAX_DEPTH = env_int("RFP_MAX_DEPTH", Config::RFP_MAX_DEPTH);
        Config::RFP_EVAL_MODE = env_int("RFP_EVAL_MODE", Config::RFP_EVAL_MODE);
        Config::RFP_RETURN_BLEND = env_int("RFP_RETURN_BLEND", Config::RFP_RETURN_BLEND);
        Config::ENABLE_NULL_EVAL_GATE = env_flag("ENABLE_NULL_EVAL_GATE", Config::ENABLE_NULL_EVAL_GATE);
        Config::ENABLE_PROBCUT = env_flag("ENABLE_PROBCUT", Config::ENABLE_PROBCUT);
        Config::PROBCUT_MARGIN = env_int("PROBCUT_MARGIN", Config::PROBCUT_MARGIN);
        Config::PROBCUT_MIN_DEPTH = env_int("PROBCUT_MIN_DEPTH", Config::PROBCUT_MIN_DEPTH);
        Config::PROBCUT_DEPTH_REDUCTION = env_int("PROBCUT_DEPTH_REDUCTION", Config::PROBCUT_DEPTH_REDUCTION);
        Config::PROBCUT_CANDIDATES = env_int("PROBCUT_CANDIDATES", Config::PROBCUT_CANDIDATES);
        Config::ENABLE_PROBCUT_NO_TT_STORE = env_flag("ENABLE_PROBCUT_NO_TT_STORE", Config::ENABLE_PROBCUT_NO_TT_STORE);
        Config::ENABLE_SINGULAR = env_flag("ENABLE_SINGULAR", Config::ENABLE_SINGULAR);
        Config::SINGULAR_MARGIN = env_int("SINGULAR_MARGIN", Config::SINGULAR_MARGIN);
        Config::SINGULAR_MIN_DEPTH = env_int("SINGULAR_MIN_DEPTH", Config::SINGULAR_MIN_DEPTH);
        Config::SINGULAR_MAX_EXT = env_int("SINGULAR_MAX_EXT", Config::SINGULAR_MAX_EXT);
        Config::ENABLE_IIR = env_flag("ENABLE_IIR", Config::ENABLE_IIR);
        Config::IIR_MIN_DEPTH = env_int("IIR_MIN_DEPTH", Config::IIR_MIN_DEPTH);
        Config::ENABLE_SIMPL_BIAS = env_flag("ENABLE_SIMPL_BIAS", Config::ENABLE_SIMPL_BIAS);
        Config::SIMPL_AHEAD_THRESH = env_int("SIMPL_AHEAD_THRESH", Config::SIMPL_AHEAD_THRESH);
        Config::SIMPL_MARGIN = env_int("SIMPL_MARGIN", Config::SIMPL_MARGIN);
        Config::ENABLE_PRUNE_LOG = env_flag("ENABLE_PRUNE_LOG", Config::ENABLE_PRUNE_LOG);
        Config::PRUNE_LOG_STRIDE = env_int("PRUNE_LOG_STRIDE", Config::PRUNE_LOG_STRIDE);
        Config::ENABLE_CORRHIST_LOG = env_flag("ENABLE_CORRHIST_LOG", Config::ENABLE_CORRHIST_LOG);
        Config::CORRHIST_LOG_STRIDE = env_int("CORRHIST_LOG_STRIDE", Config::CORRHIST_LOG_STRIDE);
        Config::ENABLE_CORR_HIST = env_flag("ENABLE_CORR_HIST", Config::ENABLE_CORR_HIST);
        Config::ENABLE_CORRHIST_QSEARCH = env_flag("ENABLE_CORRHIST_QSEARCH", Config::ENABLE_CORRHIST_QSEARCH);
        Config::DISABLE_QCACHE = env_flag("DISABLE_QCACHE", Config::DISABLE_QCACHE);
        Config::QCACHE_SOUND_STORE = env_flag("QCACHE_SOUND_STORE", Config::QCACHE_SOUND_STORE);
        Config::QCACHE_EXACT_ONLY = env_flag("QCACHE_EXACT_ONLY", Config::QCACHE_EXACT_ONLY);
        Config::ENABLE_MATERIAL_COUNT_FIX = env_flag("ENABLE_MATERIAL_COUNT_FIX", Config::ENABLE_MATERIAL_COUNT_FIX);
        Config::PIECEVAL_RECOMPUTE_LATE = env_flag("PIECEVAL_RECOMPUTE_LATE", Config::PIECEVAL_RECOMPUTE_LATE);
        Config::CORR_SHIFT = env_int("CORR_SHIFT", Config::CORR_SHIFT);
        Config::CORR_MAX = env_int("CORR_MAX", Config::CORR_MAX);
        Config::CORR_W = env_int("CORR_W", Config::CORR_W);
        Config::CORR_DIV = env_int("CORR_DIV", Config::CORR_DIV);
        if (Config::CORR_DIV < 1)
            Config::CORR_DIV = 1;
        Config::ENABLE_CUTOFF_CLASS = env_flag("ENABLE_CUTOFF_CLASS", Config::ENABLE_CUTOFF_CLASS);
        Config::ENABLE_PIECE_CONTHIST = env_flag("ENABLE_PIECE_CONTHIST", Config::ENABLE_PIECE_CONTHIST);
        Config::PIECE_CONTHIST_SHIFT = env_int("PIECE_CONTHIST_SHIFT", Config::PIECE_CONTHIST_SHIFT);
        Config::ENABLE_THREAT_HIST = env_flag("ENABLE_THREAT_HIST", Config::ENABLE_THREAT_HIST);
        Config::THREAT_HIST_SHIFT = env_int("THREAT_HIST_SHIFT", Config::THREAT_HIST_SHIFT);
        Config::ENABLE_NULLMOVE_EVAL_R = env_flag("ENABLE_NULLMOVE_EVAL_R", Config::ENABLE_NULLMOVE_EVAL_R);
        Config::NULLMOVE_R_DIV = env_int("NULLMOVE_R_DIV", Config::NULLMOVE_R_DIV);
        if (Config::NULLMOVE_R_DIV < 1)
            Config::NULLMOVE_R_DIV = 1;
        Config::NULLMOVE_R_CAP = env_int("NULLMOVE_R_CAP", Config::NULLMOVE_R_CAP);
        Config::ENABLE_QCHECK_DEPTH0 = env_flag("ENABLE_QCHECK_DEPTH0", Config::ENABLE_QCHECK_DEPTH0);
        Config::ENABLE_QCHECK_SAFE = env_flag("ENABLE_QCHECK_SAFE", Config::ENABLE_QCHECK_SAFE);
        Config::QCHECK_SAFE_LEVEL = env_int("QCHECK_SAFE_LEVEL", Config::QCHECK_SAFE_LEVEL);
        Config::ENABLE_QCHECK_FULL = env_flag("ENABLE_QCHECK_FULL", Config::ENABLE_QCHECK_FULL);
        Config::ENABLE_QCHECK_MASK_COMPARE = env_flag("ENABLE_QCHECK_MASK_COMPARE", Config::ENABLE_QCHECK_MASK_COMPARE);
        Config::ENABLE_NODE_TT = env_flag("ENABLE_NODE_TT", Config::ENABLE_NODE_TT);
        Config::ENABLE_ROOT_SORT_SPLIT = env_flag("ENABLE_ROOT_SORT_SPLIT", Config::ENABLE_ROOT_SORT_SPLIT);
        Config::ENABLE_ITER_LOG = env_flag("ENABLE_ITER_LOG", Config::ENABLE_ITER_LOG);
        Config::ENABLE_STATIC_ORDER = env_flag("ENABLE_STATIC_ORDER", Config::ENABLE_STATIC_ORDER);
        Config::STATIC_ORDER_MODE = env_int("STATIC_ORDER_MODE", Config::STATIC_ORDER_MODE);
        Config::STATIC_ORDER_WEIGHT = env_int("STATIC_ORDER_WEIGHT", Config::STATIC_ORDER_WEIGHT);
        Config::STATIC_ORDER_HIST_MAX = env_int("STATIC_ORDER_HIST_MAX", Config::STATIC_ORDER_HIST_MAX);
        Config::STATIC_ORDER_PIECES = env_int("STATIC_ORDER_PIECES", Config::STATIC_ORDER_PIECES);
        Config::STATIC_ORDER_KING_EG_ONLY = env_flag("STATIC_ORDER_KING_EG_ONLY", Config::STATIC_ORDER_KING_EG_ONLY);
        Config::ENABLE_QCHECK_MASK = env_flag("ENABLE_QCHECK_MASK", Config::ENABLE_QCHECK_MASK);
        Config::ENABLE_RP_KPK_DRAW = env_flag("ENABLE_RP_KPK_DRAW", Config::ENABLE_RP_KPK_DRAW);
        Config::ENABLE_CAPGAIN_PAWN_FIX = env_flag("ENABLE_CAPGAIN_PAWN_FIX", Config::ENABLE_CAPGAIN_PAWN_FIX);
        Config::ENABLE_ROOK_DBLCOUNT_FIX = env_flag("ENABLE_ROOK_DBLCOUNT_FIX", Config::ENABLE_ROOK_DBLCOUNT_FIX);
        Config::ENABLE_KNIGHT_MOB_FIX = env_flag("ENABLE_KNIGHT_MOB_FIX", Config::ENABLE_KNIGHT_MOB_FIX);
        Config::ENABLE_KNIGHT_MOB_SYM_UP = env_flag("ENABLE_KNIGHT_MOB_SYM_UP", Config::ENABLE_KNIGHT_MOB_SYM_UP);
        Config::ENABLE_PAWN_SUPPORT_WRAP_FIX = env_flag("ENABLE_PAWN_SUPPORT_WRAP_FIX", Config::ENABLE_PAWN_SUPPORT_WRAP_FIX);
        Config::ENABLE_CAPG_INVARIANT_ORDER = env_flag("ENABLE_CAPG_INVARIANT_ORDER", Config::ENABLE_CAPG_INVARIANT_ORDER);
        Config::ENABLE_CAPG_EVADE_POLARITY_FIX = env_flag("ENABLE_CAPG_EVADE_POLARITY_FIX", Config::ENABLE_CAPG_EVADE_POLARITY_FIX);
        Config::ENABLE_KS_ROUND_FIX = env_flag("ENABLE_KS_ROUND_FIX", Config::ENABLE_KS_ROUND_FIX);
        Config::ROOK_ENEMY_RANKWIN_MODE = env_int("ROOK_ENEMY_RANKWIN_MODE", Config::ROOK_ENEMY_RANKWIN_MODE);
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
        Config::ENABLE_TT_MOVE = env_flag("ENABLE_TT_MOVE", Config::ENABLE_TT_MOVE);
        Config::TT_MOVE_POLICY = env_int("TT_MOVE_POLICY", Config::TT_MOVE_POLICY);
        Config::CHECK_ORDER_BONUS = env_int("CHECK_ORDER_BONUS", Config::CHECK_ORDER_BONUS);
        Config::ENABLE_HISTORY_SATURATION = env_flag("ENABLE_HISTORY_SATURATION", Config::ENABLE_HISTORY_SATURATION);
        Config::ENABLE_HISTORY_MALUS = env_flag("ENABLE_HISTORY_MALUS", Config::ENABLE_HISTORY_MALUS);
        Config::ENABLE_IMPROVING = env_flag("ENABLE_IMPROVING", Config::ENABLE_IMPROVING);
        Config::IMPROVING_EVAL_WINDOW = env_int("IMPROVING_EVAL_WINDOW", Config::IMPROVING_EVAL_WINDOW);
        Config::IMPROVING_CHEAP = env_flag("IMPROVING_CHEAP", Config::IMPROVING_CHEAP);
        Config::IMPROVING_REDUCTION = env_int("IMPROVING_REDUCTION", Config::IMPROVING_REDUCTION);
        Config::IMPROVING_DELTA_MARGIN = env_int("IMPROVING_DELTA_MARGIN", Config::IMPROVING_DELTA_MARGIN);
        Config::MAX_HISTORY = env_int("MAX_HISTORY", Config::MAX_HISTORY);
        Config::CONT2_GRAVITY_DIV = env_int("CONT2_GRAVITY_DIV", Config::CONT2_GRAVITY_DIV);
        if (Config::CONT2_GRAVITY_DIV < 1)
            Config::CONT2_GRAVITY_DIV = 1;
        Config::ENABLE_HISTORY_DECAY = env_flag("ENABLE_HISTORY_DECAY", Config::ENABLE_HISTORY_DECAY);
        Config::MALUS_DIV = env_int("MALUS_DIV", Config::MALUS_DIV);
        if (Config::MALUS_DIV < 1)
            Config::MALUS_DIV = 1;
        // Fall back to the header defaults (the blitz-validated VERIFY keeper) so the
        // built-in value is the single source of truth; an env var still overrides it
        // (e.g. VERIFY_MARGIN=0 to recover the old search for the d10 control).
        Config::VERIFY_MARGIN = env_int("VERIFY_MARGIN", Config::VERIFY_MARGIN);
        Config::VERIFY_RESEARCH_REDUCTION = env_int("VERIFY_RESEARCH_REDUCTION", Config::VERIFY_RESEARCH_REDUCTION);
        // Optimism-triggered verification (default off = byte-identical).
        Config::ENABLE_OTV = env_flag("ENABLE_OTV", Config::ENABLE_OTV);
        Config::OTV_MARGIN = env_int("OTV_MARGIN", Config::OTV_MARGIN);
        Config::OTV_PLIES = env_int("OTV_PLIES", Config::OTV_PLIES);
        Config::OTV_PATH_CAP = env_int("OTV_PATH_CAP", Config::OTV_PATH_CAP);
        Config::OTV_PV_ONLY = env_flag("OTV_PV_ONLY", Config::OTV_PV_ONLY);
        Config::OTV_MIN_REMAINING = env_int("OTV_MIN_REMAINING", Config::OTV_MIN_REMAINING);
        // Iterative-deepening depth cap; default 64 is normal play (a preset governs
        // the depth reached). The cap is literal: MAX_DEPTH=10 searches to depth 10.
        Config::MAX_ITERATIVE_DEPTH = env_int("MAX_DEPTH", Config::MAX_ITERATIVE_DEPTH);
        // Fixed-node search cap for low-variance mid-funnel self-play (0 = off, clock-bound).
        Config::NODE_LIMIT = env_int("NODE_LIMIT", Config::NODE_LIMIT);
        // In-search repetition-draw threshold (default 2 = first repetition on the path).
        Config::REPETITION_THRESHOLD = env_int("REPETITION_THRESHOLD", Config::REPETITION_THRESHOLD);
        // Check/forcing extension depth (per-path cap); 0 = off.
        Config::CHECK_EXTENSION = env_int("CHECK_EXTENSION", Config::CHECK_EXTENSION);
        // SEE filter on the check extension; default disabled (extend all checks).
        Config::SEE_EXTEND_MARGIN = env_int("SEE_EXTEND_MARGIN", Config::SEE_EXTEND_MARGIN);
        Config::LIGHT_GAP_PROBE = env_flag("LIGHT_GAP_PROBE", Config::LIGHT_GAP_PROBE);
        Config::SEE_COUNT = env_flag("SEE_COUNT", Config::SEE_COUNT);
        Config::ENABLE_SEE_CACHE = env_flag("ENABLE_SEE_CACHE", Config::ENABLE_SEE_CACHE);
        Config::ENABLE_QSEE_RESORT = env_flag("ENABLE_QSEE_RESORT", Config::ENABLE_QSEE_RESORT);
        Config::LMR_EXTRA = env_int("LMR_EXTRA", Config::LMR_EXTRA);
        Config::HISTORY_BONUS_SCALE = env_int("HISTORY_BONUS_SCALE", Config::HISTORY_BONUS_SCALE);
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
        Config::ENABLE_TT_PREFETCH = env_flag("ENABLE_TT_PREFETCH", Config::ENABLE_TT_PREFETCH);
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
            Config::ACTIVE == &Configs::LIGHTNING     ? "LIGHTNING"
            : Config::ACTIVE == &Configs::BLITZ       ? "BLITZ"
            : Config::ACTIVE == &Configs::STANDARD    ? "STANDARD"
            : Config::ACTIVE == &Configs::LONG_FORMAT ? "LONG_FORMAT"
                                                      : "custom";
        std::cerr << "[toggles] LMR=" << Config::ENABLE_LMR
                  << " FUTILITY=" << Config::ENABLE_FUTILITY
                  << " RAZORING=" << Config::ENABLE_RAZORING
                  << " ROOT_RAZOR_CONTINUE=" << Config::ROOT_RAZOR_CONTINUE
                  << " RAZOR_BASE=" << Config::RAZOR_BASE
                  << " RAZOR_FLOOR=" << Config::RAZOR_FLOOR
                  << " RAZOR_DECAY_PCT=" << Config::RAZOR_DECAY_PCT
                  << " NULLMOVE=" << Config::ENABLE_NULLMOVE
                  << " NULLMOVE_PROGRESSIVE=" << Config::NULLMOVE_PROGRESSIVE
                  << " NULLMOVE_EXTRA=" << Config::NULLMOVE_EXTRA
                  << " ENABLE_PROBCUT=" << Config::ENABLE_PROBCUT
                  << " PROBCUT_MARGIN=" << Config::PROBCUT_MARGIN
                  << " PROBCUT_MIN_DEPTH=" << Config::PROBCUT_MIN_DEPTH
                  << " PROBCUT_DEPTH_REDUCTION=" << Config::PROBCUT_DEPTH_REDUCTION
                  << " PROBCUT_CANDIDATES=" << Config::PROBCUT_CANDIDATES
                  << " ENABLE_PROBCUT_NO_TT_STORE=" << Config::ENABLE_PROBCUT_NO_TT_STORE
                  << " ENABLE_IIR=" << Config::ENABLE_IIR
                  << " IIR_MIN_DEPTH=" << Config::IIR_MIN_DEPTH
                  << " ENABLE_SIMPL_BIAS=" << Config::ENABLE_SIMPL_BIAS
                  << " SIMPL_AHEAD_THRESH=" << Config::SIMPL_AHEAD_THRESH
                  << " SIMPL_MARGIN=" << Config::SIMPL_MARGIN
                  << " ENABLE_PRUNE_LOG=" << Config::ENABLE_PRUNE_LOG
                  << " PRUNE_LOG_STRIDE=" << Config::PRUNE_LOG_STRIDE
                  << " ENABLE_CORRHIST_LOG=" << Config::ENABLE_CORRHIST_LOG
                  << " CORRHIST_LOG_STRIDE=" << Config::CORRHIST_LOG_STRIDE
                  << " ENABLE_CORR_HIST=" << Config::ENABLE_CORR_HIST
                  << " CORR_SHIFT=" << Config::CORR_SHIFT << " CORR_MAX=" << Config::CORR_MAX
                  << " CORR_W=" << Config::CORR_W << " CORR_DIV=" << Config::CORR_DIV
                  << " ENABLE_CUTOFF_CLASS=" << Config::ENABLE_CUTOFF_CLASS
                  << " ENABLE_PIECE_CONTHIST=" << Config::ENABLE_PIECE_CONTHIST
                  << " PIECE_CONTHIST_SHIFT=" << Config::PIECE_CONTHIST_SHIFT
                  << " ENABLE_THREAT_HIST=" << Config::ENABLE_THREAT_HIST
                  << " THREAT_HIST_SHIFT=" << Config::THREAT_HIST_SHIFT
                  << " ENABLE_SINGULAR=" << Config::ENABLE_SINGULAR
                  << " SINGULAR_MARGIN=" << Config::SINGULAR_MARGIN
                  << " SINGULAR_MIN_DEPTH=" << Config::SINGULAR_MIN_DEPTH
                  << " SINGULAR_MAX_EXT=" << Config::SINGULAR_MAX_EXT
                  << " ENABLE_NULLMOVE_EVAL_R=" << Config::ENABLE_NULLMOVE_EVAL_R
                  << " NULLMOVE_R_DIV=" << Config::NULLMOVE_R_DIV
                  << " NULLMOVE_R_CAP=" << Config::NULLMOVE_R_CAP
                  << " QDELTA=" << Config::ENABLE_QDELTA
                  << " DELTA_MARGIN=" << Config::DELTA_MARGIN
                  << " MAX_QDEPTH=" << Config::MAX_QDEPTH
                  << " LMR_PROFILE=" << Config::LMR_PROFILE
                  << " PROTECT_KILLERS=" << Config::PROTECT_KILLERS
                  << " PROTECT_PV=" << Config::PROTECT_PV
                  << " PROTECT_MAX_IDX=" << Config::PROTECT_MAX_IDX
                  << " ENABLE_HISTORY_LMR=" << Config::ENABLE_HISTORY_LMR
                  << " HISTORY_LMR_CAP=" << Config::HISTORY_LMR_CAP
                  << " HISTORY_LMR_MORE_CAP=" << Config::HISTORY_LMR_MORE_CAP
                  << " HISTORY_LMR_SCALE=" << Config::HISTORY_LMR_SCALE
                  << " HISTORY_LMR_SCALE_CAP=" << Config::HISTORY_LMR_SCALE_CAP
                  << " ENABLE_STATSCORE_LMR=" << Config::ENABLE_STATSCORE_LMR
                  << " STATSCORE_OFFSET=" << Config::STATSCORE_OFFSET
                  << " STATSCORE_DIVISOR=" << Config::STATSCORE_DIVISOR
                  << " STATSCORE_CLAMP=" << Config::STATSCORE_CLAMP
                  << " STATSCORE_MAIN_W=" << Config::STATSCORE_MAIN_W
                  << " STATSCORE_CONT1_W=" << Config::STATSCORE_CONT1_W
                  << " STATSCORE_CONT2_W=" << Config::STATSCORE_CONT2_W
                  << " STATSCORE_KILLER_BONUS=" << Config::STATSCORE_KILLER_BONUS
                  << " ENABLE_STATSCORE_PROFILE=" << Config::ENABLE_STATSCORE_PROFILE
                  << " ENABLE_PRUNE_SHADOW=" << Config::ENABLE_PRUNE_SHADOW
                  << " SHADOW_N=" << Config::SHADOW_N
                  << " ENABLE_CUTCAL_LOG=" << Config::ENABLE_CUTCAL_LOG
                  << " ENABLE_QCUT=" << Config::ENABLE_QCUT
                  << " QCUT_LAMBDA=" << Config::QCUT_LAMBDA
                  << " QCUT_MAX=" << Config::QCUT_MAX
                  << " QCUT_MALUS_DIV=" << Config::QCUT_MALUS_DIV
                  << " ENABLE_LMR_CAPCHAIN=" << Config::ENABLE_LMR_CAPCHAIN
                  << " CAPCHAIN_REDUCE_LESS=" << Config::CAPCHAIN_REDUCE_LESS
                  << " CAPCHAIN_RUN_THRESH=" << Config::CAPCHAIN_RUN_THRESH
                  << " ENABLE_LMP=" << Config::ENABLE_LMP
                  << " ENABLE_SEE_PRUNE=" << Config::ENABLE_SEE_PRUNE
                  << " SEE_PRUNE_MARGIN=" << Config::SEE_PRUNE_MARGIN
                  << " SEE_PRUNE_MAX_DEPTH=" << Config::SEE_PRUNE_MAX_DEPTH
                  << " LMP_MAX_DEPTH=" << Config::LMP_MAX_DEPTH
                  << " LMP_BASE=" << Config::LMP_BASE
                  << " LMP_SCALE=" << Config::LMP_SCALE
                  << " ENABLE_LMP_HIST_EXEMPT=" << Config::ENABLE_LMP_HIST_EXEMPT
                  << " LMP_HIST_EXEMPT=" << Config::LMP_HIST_EXEMPT
                  << " ENABLE_HIST_PRUNE=" << Config::ENABLE_HIST_PRUNE
                  << " HIST_PRUNE_COEF=" << Config::HIST_PRUNE_COEF
                  << " HIST_PRUNE_MAX_DEPTH=" << Config::HIST_PRUNE_MAX_DEPTH
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
                  << " ENABLE_THREATS=" << Config::ENABLE_THREATS
                  << " SCALE_THREATS=" << Config::SCALE_THREATS
                  << " THREATS_STANDING_ONLY=" << Config::THREATS_STANDING_ONLY
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
                  << " IMBALANCE_SCALE=" << Config::IMBALANCE_SCALE
                  << " BISHOP_PAIR_BONUS=" << Config::BISHOP_PAIR_BONUS
                  << " KNIGHT_PAIR_BONUS=" << Config::KNIGHT_PAIR_BONUS
                  << " ROOK_OPEN_BASE=" << Config::ROOK_OPEN_BASE
                  << " ROOK_OPEN_CAP=" << Config::ROOK_OPEN_CAP
                  << " ROOK_7TH=" << Config::ROOK_7TH
                  << " ROOK_CONNECTED=" << Config::ROOK_CONNECTED
                  << " ROOK_SEMI=" << Config::ROOK_SEMI
                  << " ROOK_PASSER_OWN=" << Config::ROOK_PASSER_OWN
                  << " ROOK_PASSER_ENEMY=" << Config::ROOK_PASSER_ENEMY
                  << " ROOK_OWN_PAWN_BASE=" << Config::ROOK_OWN_PAWN_BASE
                  << " ROOK_OWN_PAWN_RAMP=" << Config::ROOK_OWN_PAWN_RAMP
                  << " ROOK_ENEMY_PAWN_PEN=" << Config::ROOK_ENEMY_PAWN_PEN
                  << " ROOK_MINOR_BLOCK=" << Config::ROOK_MINOR_BLOCK
                  << " ROOK_ROOK_BLOCK=" << Config::ROOK_ROOK_BLOCK
                  << " ROOK_SEMI_CONNECTED=" << Config::ROOK_SEMI_CONNECTED
                  << " SCALE_PAWN_RANK=" << Config::SCALE_PAWN_RANK
                  << " SCALE_PASSED_RANK=" << Config::SCALE_PASSED_RANK
                  << " SCALE_ENDGAME_RANK=" << Config::SCALE_ENDGAME_RANK
                  << " ENABLE_PAWN_OBSTRUCTION_BLEND=" << Config::ENABLE_PAWN_OBSTRUCTION_BLEND
                  << " PAWN_OBS_LO=" << Config::PAWN_OBS_LO
                  << " PAWN_OBS_HI=" << Config::PAWN_OBS_HI
                  << " PAWN_OBS_LO_EG=" << Config::PAWN_OBS_LO_EG
                  << " PAWN_OBS_HI_EG=" << Config::PAWN_OBS_HI_EG
                  << " PAWN_CLAMP_MID=" << Config::PAWN_CLAMP_MID
                  << " PAWN_CLAMP_EG=" << Config::PAWN_CLAMP_EG
                  << " EG_PHALANX=" << Config::EG_PHALANX
                  << " EG_SUPPORT=" << Config::EG_SUPPORT
                  << " EG_DEFEND=" << Config::EG_DEFEND
                  << " EG_LATENT=" << Config::EG_LATENT
                  << " EG_EXIST_KNIGHT=" << Config::EG_EXIST_KNIGHT
                  << " EG_EXIST_BISHOP=" << Config::EG_EXIST_BISHOP
                  << " EG_EXIST_ROOK=" << Config::EG_EXIST_ROOK
                  << " EG_EXIST_QUEEN=" << Config::EG_EXIST_QUEEN
                  << " MG_CLAMP_KNIGHT=" << Config::MG_CLAMP_KNIGHT
                  << " MG_CLAMP_BISHOP_A=" << Config::MG_CLAMP_BISHOP_A
                  << " MG_CLAMP_BISHOP_B=" << Config::MG_CLAMP_BISHOP_B
                  << " EG_CLAMP_KNIGHT=" << Config::EG_CLAMP_KNIGHT
                  << " EG_CLAMP_BISHOP=" << Config::EG_CLAMP_BISHOP
                  << " EG_CLAMP_ROOK=" << Config::EG_CLAMP_ROOK
                  << " EG_CLAMP_QUEEN=" << Config::EG_CLAMP_QUEEN
                  << " ENABLE_WINNABILITY=" << Config::ENABLE_WINNABILITY
                  << " WINNAB_PASSED=" << Config::WINNAB_PASSED
                  << " WINNAB_PAWNS=" << Config::WINNAB_PAWNS
                  << " WINNAB_OUTFLANK=" << Config::WINNAB_OUTFLANK
                  << " WINNAB_FLANKS=" << Config::WINNAB_FLANKS
                  << " WINNAB_INFILT=" << Config::WINNAB_INFILT
                  << " WINNAB_NO_NPM=" << Config::WINNAB_NO_NPM
                  << " WINNAB_UNWINNABLE=" << Config::WINNAB_UNWINNABLE
                  << " WINNAB_TENSION=" << Config::WINNAB_TENSION
                  << " WINNAB_BASE=" << Config::WINNAB_BASE
                  << " WINNAB_MG_OFFSET=" << Config::WINNAB_MG_OFFSET
                  << " WINNAB_SCALE=" << Config::WINNAB_SCALE
                  << " ENABLE_CLOSEDNESS=" << Config::ENABLE_CLOSEDNESS
                  << " CLOSED_N4=" << Config::CLOSED_N[4]
                  << " CLOSED_R4=" << Config::CLOSED_R[4]
                  << " CLOSED_B4=" << Config::CLOSED_B_PCT[4]
                  << " PHASE_BLEND_LO=" << Config::PHASE_BLEND_LO
                  << " PHASE_BLEND_RANGE=" << Config::PHASE_BLEND_RANGE
                  << " STRUCT_OPPOSED_MG_PCT=" << Config::STRUCT_OPPOSED_MG_PCT
                  << " STRUCT_OPPOSED_EG_PCT=" << Config::STRUCT_OPPOSED_EG_PCT
                  << " ENABLE_PASSER_DEFER_ON_FLAG=" << Config::ENABLE_PASSER_DEFER_ON_FLAG
                  << " PASSER_R_MAX=" << Config::PASSER_R_MAX
                  << " PPS_OWN_BLOCK=" << Config::PPS_OWN_BLOCK
                  << " PPS_ENEMY_BLOCK=" << Config::PPS_ENEMY_BLOCK
                  << " PPS_OWN_ATTACK=" << Config::PPS_OWN_ATTACK
                  << " PPS_ENEMY_ATTACK=" << Config::PPS_ENEMY_ATTACK
                  << " SCALE_PAWN_WALL=" << Config::SCALE_PAWN_WALL
                  << " SCALE_PAWN_CHAIN=" << Config::SCALE_PAWN_CHAIN
                  << " BISHOP_MOB_PAWN_ATTACK=" << Config::BISHOP_MOB_PAWN_ATTACK
                  << " BISHOP_MOB_SECONDARY=" << Config::BISHOP_MOB_SECONDARY
                  << " THREAT_ATTACK_MULT=" << Config::THREAT_ATTACK_MULT
                  << " THREAT_PRESENCE_MULT=" << Config::THREAT_PRESENCE_MULT
                  << " THREAT_PAWN=" << Config::THREAT_PAWN
                  << " THREAT_KNIGHT=" << Config::THREAT_KNIGHT
                  << " THREAT_BISHOP=" << Config::THREAT_BISHOP
                  << " THREAT_ROOK=" << Config::THREAT_ROOK
                  << " THREAT_QUEEN=" << Config::THREAT_QUEEN
                  << " SCALE_ATTACK_LAYER=" << Config::SCALE_ATTACK_LAYER
                  << " REALIZ_MAT_K=" << Config::REALIZ_MAT_K
                  << " REALIZ_MAT_THRESH=" << Config::REALIZ_MAT_THRESH
                  << " REALIZ_PHASE_K=" << Config::REALIZ_PHASE_K
                  << " REALIZ_FLOOR=" << Config::REALIZ_FLOOR
                  << " PASSER_BLOCK_ADV=" << Config::PASSER_BLOCK_ADV
                  << " PASSER_ENEMY_CREDIT_PCT=" << Config::PASSER_ENEMY_CREDIT_PCT
                  << " ENABLE_PASSER_DANGER=" << Config::ENABLE_PASSER_DANGER
                  << " PASSER_DANGER_BASE1=" << Config::PASSER_DANGER_BASE1
                  << " PASSER_DANGER_D2=" << Config::PASSER_DANGER_D2
                  << " PASSER_DANGER_D4=" << Config::PASSER_DANGER_D4
                  << " PASSER_CONTEST_STOP=" << Config::PASSER_CONTEST_STOP
                  << " PASSER_CONTEST_PATH=" << Config::PASSER_CONTEST_PATH
                  << " PASSER_REAR_ENEMY=" << Config::PASSER_REAR_ENEMY
                  << " PASSER_REAR_OWN=" << Config::PASSER_REAR_OWN
                  << " PASSER_KING_FAR=" << Config::PASSER_KING_FAR
                  << " PASSER_KING_HELP=" << Config::PASSER_KING_HELP
                  << " PASSER_MAG_SCALE=" << Config::PASSER_MAG_SCALE
                  << " PASSER_R_CAP=" << Config::PASSER_R_CAP
                  << " PASSER_R_FLOOR=" << Config::PASSER_R_FLOOR
                  << " ENABLE_PASSER_KRACE_MG=" << Config::ENABLE_PASSER_KRACE_MG
                  << " PASSER_KRACE_MG_PCT=" << Config::PASSER_KRACE_MG_PCT
                  << " ENABLE_PASSER_V2=" << Config::ENABLE_PASSER_V2
                  << " ENABLE_PASSER_V3=" << Config::ENABLE_PASSER_V3
                  << " ENABLE_KAUFMAN_IMBALANCE=" << Config::ENABLE_KAUFMAN_IMBALANCE
                  << " KAUFMAN_SCALE=" << Config::KAUFMAN_SCALE
                  << " ENABLE_KS_AIM=" << Config::ENABLE_KS_AIM
                  << " KS_AIM_BISHOP=" << Config::KS_AIM_BISHOP
                  << " KS_AIM_ROOK=" << Config::KS_AIM_ROOK
                  << " KS_AIM_QUEEN=" << Config::KS_AIM_QUEEN
                  << " KS_MIN_ATTACKERS=" << Config::KS_MIN_ATTACKERS
                  << " KS_ATT_PRODUCT=" << Config::KS_ATT_PRODUCT
                  << " KS_OVERLOAD=" << Config::KS_OVERLOAD
                  << " ENABLE_PASSER_BLOCKADE_QUALITY=" << Config::ENABLE_PASSER_BLOCKADE_QUALITY
                  << " PASSER_CONTEST_PCT=" << Config::PASSER_CONTEST_PCT
                  << " PASSER_KRACE_MAG=" << Config::PASSER_KRACE_MAG
                  << " PV_BOOST_MAG=" << Config::PV_BOOST_MAG
                  << " PV_BOOST_TRIGGER=" << Config::PV_BOOST_TRIGGER
                  << " PV_BOOST_PHASE_K=" << Config::PV_BOOST_PHASE_K
                  << " MOD_FLOOR=" << Config::MOD_FLOOR
                  << " MOD_CEIL=" << Config::MOD_CEIL
                  << " MOD_MAT_PAWNS=" << Config::MOD_MAT_PAWNS
                  << " MOD_MAT_OPPB=" << Config::MOD_MAT_OPPB
                  << " MOD_LT_BACKING=" << Config::MOD_LT_BACKING
                  << " MOD_PAIR_OPEN=" << Config::MOD_PAIR_OPEN
                  << " MOD_KS_BACKING=" << Config::MOD_KS_BACKING
                  << " MOD_KS_CONTROL=" << Config::MOD_KS_CONTROL
                  << " MOD_KS_REALIZ=" << Config::MOD_KS_REALIZ
                  << " KS_REALIZ_FLOOR=" << Config::KS_REALIZ_FLOOR
                  << " KS_CONSOLIDATE=" << Config::KS_CONSOLIDATE
                  << " ENABLE_KS_V2=" << Config::ENABLE_KS_V2
                  << " KS_SHELTER_FULL=" << Config::KS_SHELTER_FULL
                  << " KS_SHELTER_PARTIAL=" << Config::KS_SHELTER_PARTIAL
                  << " KS_SHELTER_MAG=" << Config::KS_SHELTER_MAG
                  << " MOD_PIECES_LEVEL=" << Config::MOD_PIECES_LEVEL
                  << " MOD_PIECES_MAT_THRESH=" << Config::MOD_PIECES_MAT_THRESH
                  << " MOD_PIECES_FLOOR=" << Config::MOD_PIECES_FLOOR
                  << " MOD_PIECES_CONTROL=" << Config::MOD_PIECES_CONTROL
                  << " ENABLE_NPEDGE_DAMP=" << Config::ENABLE_NPEDGE_DAMP
                  << " NPEDGE_DAMP_LO=" << Config::NPEDGE_DAMP_LO
                  << " NPEDGE_DAMP_HI=" << Config::NPEDGE_DAMP_HI
                  << " NPEDGE_DAMP_MAX=" << Config::NPEDGE_DAMP_MAX
                  << " NPEDGE_DAMP_TQUIET=" << Config::NPEDGE_DAMP_TQUIET
                  << " ENABLE_NPEDGE_DAMP_EG=" << Config::ENABLE_NPEDGE_DAMP_EG
                  << " NPEDGE_EG_PIECE_FLOOR=" << Config::NPEDGE_EG_PIECE_FLOOR
                  << " ENABLE_MOBILITY=" << Config::ENABLE_MOBILITY
                  << " MOBILITY_SCALE=" << Config::MOBILITY_SCALE
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
                  << " ENABLE_SEE_INCREMENTAL=" << Config::ENABLE_SEE_INCREMENTAL
                  << " FUTILITY_EVAL_MODE=" << Config::FUTILITY_EVAL_MODE
                  << " QSTANDPAT_EVAL_MODE=" << Config::QSTANDPAT_EVAL_MODE
                  << " ENABLE_RP_KPK_DRAW=" << Config::ENABLE_RP_KPK_DRAW
                  << " ENABLE_CAPGAIN_PAWN_FIX=" << Config::ENABLE_CAPGAIN_PAWN_FIX
                  << " ENABLE_ROOK_DBLCOUNT_FIX=" << Config::ENABLE_ROOK_DBLCOUNT_FIX
                  << " ENABLE_KNIGHT_MOB_FIX=" << Config::ENABLE_KNIGHT_MOB_FIX
                  << " ENABLE_KNIGHT_MOB_SYM_UP=" << Config::ENABLE_KNIGHT_MOB_SYM_UP
                  << " ENABLE_PAWN_SUPPORT_WRAP_FIX=" << Config::ENABLE_PAWN_SUPPORT_WRAP_FIX
                  << " ENABLE_CAPG_INVARIANT_ORDER=" << Config::ENABLE_CAPG_INVARIANT_ORDER
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
                  << " ENABLE_QCHECK_SAFE=" << Config::ENABLE_QCHECK_SAFE
                  << " QCHECK_SAFE_LEVEL=" << Config::QCHECK_SAFE_LEVEL
                  << " ENABLE_QCHECK_FULL=" << Config::ENABLE_QCHECK_FULL
                  << " ENABLE_QCHECK_MASK_COMPARE=" << Config::ENABLE_QCHECK_MASK_COMPARE
                  << " ENABLE_NODE_TT=" << Config::ENABLE_NODE_TT
                  << " ENABLE_ROOT_SORT_SPLIT=" << Config::ENABLE_ROOT_SORT_SPLIT
                  << " ENABLE_ROOT_LMR=" << Config::ENABLE_ROOT_LMR
                  << " ROOT_LMR_MIN_IDX=" << Config::ROOT_LMR_MIN_IDX
                  << " ROOT_LMR_BASE=" << Config::ROOT_LMR_BASE
                  << " ROOT_LMR_DIV=" << Config::ROOT_LMR_DIV
                  << " ROOT_LMR_EXEMPT_BEST=" << Config::ROOT_LMR_EXEMPT_BEST
                  << " ENABLE_ROOT_TABLE=" << Config::ENABLE_ROOT_TABLE
                  << " ROOT_RAZOR_MAX_AGE=" << Config::ROOT_RAZOR_MAX_AGE
                  << " ROOT_SORT_L2_LASTREAL=" << Config::ROOT_SORT_L2_LASTREAL
                  << " ROOT_SORT_L1_LASTREAL=" << Config::ROOT_SORT_L1_LASTREAL
                  << " ROOT_RAZOR_TO_LMR=" << Config::ROOT_RAZOR_TO_LMR
                  << " ROOT_RAZOR_SKIP_MARGIN=" << Config::ROOT_RAZOR_SKIP_MARGIN
                  << " ROOT_RAZOR_LMR_BASE=" << Config::ROOT_RAZOR_LMR_BASE
                  << " ROOT_RAZOR_LMR_DIV=" << Config::ROOT_RAZOR_LMR_DIV
                  << " ROOT_RAZOR_MAX_DEPTH_DEFICIT=" << Config::ROOT_RAZOR_MAX_DEPTH_DEFICIT
                  << " ENABLE_ROOT_RAZOR=" << Config::ENABLE_ROOT_RAZOR
                  << " PRESEARCH_OFF_FROM_DEPTH=" << Config::PRESEARCH_OFF_FROM_DEPTH
                  << " ENABLE_PRESEARCH_SUBSET=" << Config::ENABLE_PRESEARCH_SUBSET
                  << " PRESEARCH_TAIL_MODE=" << Config::PRESEARCH_TAIL_MODE
                  << " PRESEARCH_CHUNK=" << Config::PRESEARCH_CHUNK
                  << " PRESEARCH_TAIL_REDUCTION=" << Config::PRESEARCH_TAIL_REDUCTION
                  << " ENABLE_STATIC_ORDER=" << Config::ENABLE_STATIC_ORDER
                  << " STATIC_ORDER_MODE=" << Config::STATIC_ORDER_MODE
                  << " STATIC_ORDER_WEIGHT=" << Config::STATIC_ORDER_WEIGHT
                  << " STATIC_ORDER_HIST_MAX=" << Config::STATIC_ORDER_HIST_MAX
                  << " STATIC_ORDER_PIECES=" << Config::STATIC_ORDER_PIECES
                  << " STATIC_ORDER_KING_EG_ONLY=" << Config::STATIC_ORDER_KING_EG_ONLY
                  << " ENABLE_TT_MOVE=" << Config::ENABLE_TT_MOVE
                  << " TT_MOVE_POLICY=" << Config::TT_MOVE_POLICY
                  << " ENABLE_CONT_HIST=" << Config::ENABLE_CONT_HIST
                  << " CONT_HIST_LMR_THRESH=" << Config::CONT_HIST_LMR_THRESH
                  << " ENABLE_CONT_HIST_2PLY=" << Config::ENABLE_CONT_HIST_2PLY
                  << " ENABLE_CAPTURE_HIST=" << Config::ENABLE_CAPTURE_HIST
                  << " ENABLE_CHECK_ORDER=" << Config::ENABLE_CHECK_ORDER
                  << " ENABLE_HISTORY_SATURATION=" << Config::ENABLE_HISTORY_SATURATION
                  << " ENABLE_HISTORY_MALUS=" << Config::ENABLE_HISTORY_MALUS
                  << " ENABLE_IMPROVING=" << Config::ENABLE_IMPROVING
                  << " IMPROVING_CHEAP=" << Config::IMPROVING_CHEAP
                  << " IMPROVING_REDUCTION=" << Config::IMPROVING_REDUCTION
                  << " IMPROVING_DELTA_MARGIN=" << Config::IMPROVING_DELTA_MARGIN
                  << " IMPROVING_EVAL_WINDOW=" << Config::IMPROVING_EVAL_WINDOW
                  << " MAX_HISTORY=" << Config::MAX_HISTORY
                  << " CONT2_GRAVITY_DIV=" << Config::CONT2_GRAVITY_DIV
                  << " ENABLE_HISTORY_DECAY=" << Config::ENABLE_HISTORY_DECAY
                  << " MALUS_DIV=" << Config::MALUS_DIV
                  << " VERIFY_MARGIN=" << Config::VERIFY_MARGIN
                  << " VERIFY_RESEARCH_REDUCTION=" << Config::VERIFY_RESEARCH_REDUCTION
                  << " ENABLE_OTV=" << Config::ENABLE_OTV
                  << " OTV_MARGIN=" << Config::OTV_MARGIN
                  << " OTV_PLIES=" << Config::OTV_PLIES
                  << " OTV_PATH_CAP=" << Config::OTV_PATH_CAP
                  << " OTV_PV_ONLY=" << Config::OTV_PV_ONLY
                  << " OTV_MIN_REMAINING=" << Config::OTV_MIN_REMAINING
                  << " PRESET=" << active_preset
                  << " MAX_DEPTH=" << Config::MAX_ITERATIVE_DEPTH
                  << " NODE_LIMIT=" << Config::NODE_LIMIT
                  << " REPETITION_THRESHOLD=" << Config::REPETITION_THRESHOLD
                  << " CHECK_EXTENSION=" << Config::CHECK_EXTENSION
                  << " SEE_EXTEND_MARGIN=" << Config::SEE_EXTEND_MARGIN
                  << " LIGHT_GAP_PROBE=" << Config::LIGHT_GAP_PROBE
                  << " SEE_COUNT=" << Config::SEE_COUNT
                  << " ENABLE_SEE_CACHE=" << Config::ENABLE_SEE_CACHE
                  << " ENABLE_QSEE_RESORT=" << Config::ENABLE_QSEE_RESORT
                  << " LMR_EXTRA=" << Config::LMR_EXTRA
                  << " HISTORY_BONUS_SCALE=" << Config::HISTORY_BONUS_SCALE
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
    PROF_BLOCK(PROF_MAKEUNMAKE);
    ++g_see_gen; // new position -> invalidate the per-position SEE cache (O(1))

    const BoardState &current = state_history.back();

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

    position_count[zobrist]++;
    // Construct the new state directly in the history vector; a named temporary + push_back would
    // copy the ~100-byte BoardState an extra time (lvalue push_back cannot elide the copy).
    state_history.emplace_back(
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

    // Prefetch the child node's TT slot now that its key (zobrist) and cache-key inputs (child
    // castling_rights / ep_square, already updated above) are known, hiding the probe's DRAM latency
    // behind the make/return work before get_score_*() reads it. Pure hint -> byte-identical.
    if (Config::ENABLE_TT_PREFETCH)
        prefetchSearchEvalCache(zobrist, castling_rights, ep_square);
}

inline void unmake_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key)
{
    PROF_BLOCK(PROF_MAKEUNMAKE);
    ++g_see_gen; // restored position -> invalidate the per-position SEE cache (O(1))

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
#ifdef EVAL_PROFILE
    eval_profile_reset(); // accumulate PROF scopes across this one search, dump at the end
#endif

    update_cache(static_cast<int>(state_history.size()));
    std::fill(&killerMoves[0][0], &killerMoves[0][0] + 64 * 2, Move{});
    std::fill(&counterMoves[0][0], &counterMoves[0][0] + 64 * 64, Move{});
    std::fill(&g_searchStack[0], &g_searchStack[0] + MAX_PLY, Move{});
    if (Config::ENABLE_CUTCAL_LOG)
        std::fill(&g_tf_count[0][0][0], &g_tf_count[0][0][0] + 2 * 64 * 64, 0L);
    if (Config::ENABLE_QCUT)
        std::fill(&g_qcut[0][0][0], &g_qcut[0][0][0] + 2 * 64 * 64, 0);
    std::fill(&g_evalStack[0], &g_evalStack[0] + MAX_PLY, NO_STATIC_EVAL);
    std::fill(&g_captureChain[0], &g_captureChain[0] + MAX_PLY, 0);
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
    if (!Config::ENABLE_QPREC_PHASE_GATE)
        use_q_precautions = true;
    uint64_t zobrist = generateZobristHash(current.pawns, current.knights, current.bishops, current.rooks, current.queens, current.kings, current.occupied_colour[true], current.occupied_colour[false], current.turn);

    Move move(0, 0, 0);

    int depth_limit = 3;
    int num_iterations = 0;
    int alpha = -9999998;
    int beta = 9999999;

    SearchData preliminary_search_data;
    // The razor audit compares consecutive iterations of the SAME search; carrying state across positions
    // would compare a move list against an unrelated one.
    g_prev_root_list.clear();
    g_prev_razor_idx = -1;
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
        if (score <= Config::RESIGN_THRESHOLD)
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

        // Per-iteration cumulative node count. The headline "ebf" figure is
        // pow(cumulative_nodes, 1/depth_limit) over a counter that also absorbs the pre-search, aspiration
        // re-searches, qsearch and TT-hit bookkeeping, with depth_limit taken at loop exit -- so it is not
        // nodes(d)/nodes(d-1) and is not comparable to the figures other engines publish. Differencing this
        // line across iterations recovers the real per-iteration growth.
        if (Config::ENABLE_ITER_LOG)
            std::cerr << "[iter] d=" << depth_limit << " cum_nodes=" << num_iterations << std::endl;
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
                  << " (nodes=" << num_iterations << " qnodes=" << qsearchVisits << " d=" << depth_limit << ")"
                  << " see=" << see_calls
                  << " seehit=" << ((g_see_hits + g_see_miss) > 0 ? (100.0 * g_see_hits / (g_see_hits + g_see_miss)) : 0.0) << "%"
                  << " qfmc=" << (g_q_fh_total > 0 ? (100.0 * g_q_fh_first / g_q_fh_total) : 0.0)
                  << "% (" << g_q_fh_first << "/" << g_q_fh_total << ")"
                  << " qcut=" << (g_q_fh_total > 0 ? ((double)g_q_cut_idx_sum / g_q_fh_total) : 0.0) << std::endl;
    if (g_fh_total > 0)
        std::cerr << "[cutoff_histogram] m0=" << g_cutoff_histogram[0] << " m1=" << g_cutoff_histogram[1]
                  << " m2=" << g_cutoff_histogram[2] << " m3-7=" << g_cutoff_histogram[3]
                  << " m8+=" << g_cutoff_histogram[4] << std::endl;
    if (Config::ENABLE_CUTOFF_CLASS)
    {
        static const char *cc[] = {"cap", "promo", "killer", "counter", "quiet"};
        std::cerr << "[cutoff_class] (m0/m1/m2/m3-7/m8+ per class)";
        for (int c = 0; c < CUTCLASS_N; ++c)
            std::cerr << "  " << cc[c] << "=" << g_cutoff_class_hist[c][0] << "/" << g_cutoff_class_hist[c][1]
                      << "/" << g_cutoff_class_hist[c][2] << "/" << g_cutoff_class_hist[c][3] << "/"
                      << g_cutoff_class_hist[c][4];
        std::cerr << std::endl;
    }
    std::cerr << "[passer_exempt] fires=" << g_passer_exempt_fires
              << "  [qdelta_permove] seen=" << g_qdelta_permove_seen << " fires=" << g_qdelta_permove_fires << std::endl;
    std::cerr << "[qcheck] quiet_checks_added=" << g_q_quiet_checks_added
              << " unsafe_rejected=" << g_q_quiet_checks_unsafe
              << " discovered=" << g_q_discovered_checks
              << " missed=" << g_q_checks_missed
              << " d0_quiets_skipped=" << g_qcheck_d0_skipped << std::endl;
    std::cerr << "[node_split] total=" << num_iterations
              << " qnodes=" << qsearchVisits
              << " presearch=" << g_presearch_nodes << std::endl;
    std::cerr << "[draw_empty_ply1] minimizer=" << g_draw_empty_ply1_min
              << " pre_minimizer=" << g_draw_empty_ply1_pre
              << " root_scores_short=" << g_root_scores_short << std::endl;
    std::cerr << "[tail_mode] mode=" << Config::PRESEARCH_TAIL_MODE
              << " chunk=" << Config::PRESEARCH_CHUNK
              << " full=" << g_tail_full
              << " reduced=" << g_tail_reduced
              << " heuristic=" << g_tail_heuristic
              << " razor_fires=" << g_razor_fires
              << " prefix_skipped=" << g_prefix_skipped << std::endl;
    std::cerr << "[razor_audit] iters_after_razor=" << g_razor_audit_iters
              << " winner_was_razored=" << g_razor_cut_winner
              << " pct=" << (g_razor_audit_iters > 0 ? (100.0 * g_razor_cut_winner / g_razor_audit_iters) : 0.0) << "%"
              << " avg_depth_past_razor=" << (g_razor_cut_winner > 0 ? (1.0 * g_razor_cut_depth_sum / g_razor_cut_winner) : 0.0)
              << std::endl;
    std::cerr << "[root_table] slots=" << g_root_table_slots
              << " has_real=" << g_root_table_has_real
              << " evidence_pct=" << (g_root_table_slots > 0 ? (100.0 * g_root_table_has_real / g_root_table_slots) : 0.0)
              << " proven_this_iter=" << g_root_table_verified
              << " razor_stale_skips=" << g_root_razor_stale_skips
              << " hybrid_reduced=" << g_razor_hybrid_reduced
              << " hybrid_skipped=" << g_razor_hybrid_skipped
              << " stale_reduced=" << g_razor_stale_reduced << std::endl;
    std::cerr << "[root_lmr] reduced=" << g_root_lmr_reduced
              << " exempt_best=" << g_root_lmr_exempt
              << " researches=" << g_root_lmr_researches
              << " research_pct=" << (g_root_lmr_reduced > 0 ? (100.0 * g_root_lmr_researches / g_root_lmr_reduced) : 0.0)
              << "%" << std::endl;
    if (Config::ENABLE_NODE_TT)
        std::cerr << "[node_tt] stores=" << g_node_tt_stores << std::endl;
    if (Config::ENABLE_STATIC_ORDER)
        std::cerr << "[static_order] eligible=" << g_static_order_eligible
                  << " fires=" << g_static_order_fires
                  << " fire_pct=" << (g_static_order_eligible > 0 ? (100.0 * g_static_order_fires / g_static_order_eligible) : 0.0)
                  << "%" << std::endl;
    if (Config::ENABLE_TT_MOVE)
        std::cerr << "[tt_move] policy=" << Config::TT_MOVE_POLICY
                  << " promotions=" << g_tt_move_promotions << std::endl;
    if (Config::ENABLE_OTV)
        std::cerr << "[otv] fires=" << g_otv_fires
                  << " per_cutoff=" << (g_fh_total > 0 ? (100.0 * g_otv_fires / g_fh_total) : 0.0) << "%" << std::endl;
    if (Config::ENABLE_SINGULAR)
        std::cerr << "[singular] eligible=" << g_sing_eligible << " gatepass=" << g_sing_gatepass
                  << " fire=" << g_sing_fire
                  << " fire_per_eligible=" << (g_sing_eligible > 0 ? (100.0 * g_sing_fire / g_sing_eligible) : 0.0) << "%" << std::endl;

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

    if (Config::LIGHT_GAP_PROBE)
        std::cout << "LIGHT_GAP n=" << g_lge_n
                  << " h=" << g_lge_hist[0] << "," << g_lge_hist[1] << "," << g_lge_hist[2] << ","
                  << g_lge_hist[3] << "," << g_lge_hist[4] << "," << g_lge_hist[5] << "," << g_lge_hist[6]
                  << " cap=" << g_lge_abs_capture << " pas=" << g_lge_abs_passed
                  << " lat=" << g_lge_abs_latent << " adv=" << g_lge_abs_adv << std::endl;

    if (Config::LMR_PROFILE)
        lmr_profile_dump();

    if (Config::ENABLE_STATSCORE_PROFILE)
        statscore_profile_dump();

    if (Config::ENABLE_PRUNE_SHADOW)
        shadow_profile_dump();

    if (Config::ENABLE_CUTCAL_LOG)
        cutcal_profile_dump();

#ifdef EVAL_PROFILE
    eval_profile_dump("search"); // whole-search cycle breakdown (eval terms + MOVEGEN/MAKEUNMAKE/TT_PROBE)
#endif
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

    // Cost of the root pre-search, measured rather than estimated: the node counter is shared, so the
    // difference across the call is exactly what reorder_legal_moves spent. Charged on every aspiration
    // widening too, since alpha_beta is re-entered for each.
    int presearch_nodes_before = num_iterations;
    SearchData current_search_data = reorder_legal_moves(alpha, beta, depth_limit, t0, zobrist, previous_search_data, state_history, position_count, num_iterations);
    g_presearch_nodes += (long)(num_iterations - presearch_nodes_before);

    if (current_search_data.scores.size() < current_search_data.moves_list.size())
        ++g_root_scores_short;

    int root_razor_idx = -1;
    RootRazorAudit root_razor_audit{best_move, current_search_data.moves_list, root_razor_idx};
    // Alpha as this call entered, before move 0 raises it. Root LMR delays reductions by one move while
    // nothing has beaten it, which is SF's (rootNode && bestValue < alpha) term.
    const int root_alpha_entry = alpha;
    // The previous iteration's chosen root move. best_move is an in/out parameter carried across every
    // iterative-deepening pass and every aspiration retry, so on entry it still holds the last winner --
    // but the loop below overwrites it, hence the snapshot here. Approximates SF's
    // `best_move_count(move) == 0` term, which exempts a recently-best root move from reduction.
    const Move root_prev_best = best_move;

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
        razor_threshold = std::max(static_cast<int>(Config::RAZOR_BASE_FIRST * std::pow(Config::RAZOR_DECAY_PCT / 100.0, depth_limit - 4)), Config::RAZOR_FLOOR_FIRST);
    }
    else
    {
        razor_threshold = std::max(static_cast<int>(Config::RAZOR_BASE * std::pow(Config::RAZOR_DECAY_PCT / 100.0, depth_limit - 4)), Config::RAZOR_FLOOR);
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

    if (depth_limit >= 13)
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

    if (depth_limit >= 13)
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

    // Simplification-bias bookkeeping: does the current best root move trade a non-pawn piece?
    uint64_t simpl_best_to = 1ULL << current_search_data.moves_list[0].to_square;
    bool best_is_simpl = capture_move && (simpl_best_to & current_state.occupied_colour[!current_state.turn]) && !(simpl_best_to & current_state.pawns);

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
    if (Config::ENABLE_ROOT_TABLE)
    {
        // Size the table to the FULL root list before the loop runs. Every move gets an entry up front,
        // so each of alpha_beta's eight exits leaves a complete, index-consistent table instead of a
        // truncated stump -- the invariant becomes structural rather than a convention six sites uphold.
        // Entries start unproven and inherit their incoming score as the second-level sort key.
        size_t n_root = previous_search_data.moves_list.size();
        previous_search_data.scores.assign(n_root, RootScore{});
        for (size_t k = 0; k < n_root; ++k)
        {
            RootScore &slot = previous_search_data.scores[k];
            bool had = k < current_search_data.scores.size();
            slot.prev_score = had ? current_search_data.scores[k].top_score : ROOT_SCORE_UNPROVEN;
            slot.top_score = ROOT_SCORE_UNPROVEN;
            slot.verified = false;
            // The last proven value survives an iteration that only failed low; only its age advances.
            slot.last_real = had ? current_search_data.scores[k].last_real : ROOT_SCORE_UNPROVEN;
            slot.last_real_depth = had ? current_search_data.scores[k].last_real_depth : 0;
            slot.age = (had && current_search_data.scores[k].age < ROOT_AGE_NEVER)
                           ? current_search_data.scores[k].age + 1
                           : ROOT_AGE_NEVER;
            // Keep a legal reply list on every entry: alpha_beta indexes second_moves unguarded.
            if (had)
                slot.second_moves = current_search_data.scores[k].second_moves;

            // Evidence coverage: does this move have ANY proven score behind it, at any age? This is the
            // number the lane is about -- the push_back table could only answer yes for moves searched in
            // the immediately preceding iteration.
            ++g_root_table_slots;
            if (slot.last_real != ROOT_SCORE_UNPROVEN)
                ++g_root_table_has_real;
        }
        root_table_store(previous_search_data.scores[0], score, std::move(entry), true, depth_limit);
    }
    else
    {
        entry.top_score = score;
        previous_search_data.scores.push_back(std::move(entry));
    }

    if (std::chrono::duration<double>(Clock::now() - t0).count() >= Config::ACTIVE->TIME_LIMIT)
        return score;

    for (size_t i = 1; i < current_search_data.moves_list.size(); ++i)
    {
        Move &move = current_search_data.moves_list[i];

        // Plies removed because the razor judged this move suspect rather than hopeless (hybrid path).
        // Combines with the index-keyed root LMR reduction below; the deeper of the two wins.
        int razor_r = 0;

        // Razoring
        if (i < current_search_data.scores.size())
        {
            // Under the table a fail-low entry carries a sentinel in top_score, so the razor must read the
            // last PROVEN value instead -- differencing against the sentinel would clear any threshold and
            // prune the whole tail on a number nothing measured.
            int razor_ref = Config::ENABLE_ROOT_TABLE ? current_search_data.scores[i].last_real
                                                      : current_search_data.scores[i].top_score;
            int score_diff = alpha - razor_ref;
            // int best_diff = current_search_data.scores[0].top_score - current_search_data.scores[i].top_score;

            // Never razor on a synthetic score: an unsearched move has NO score, which is not the same as a
            // score of zero. Pruning on a value nothing measured is what collapsed the pre-search-off path
            // to 57/300 -- with a winning alpha, `alpha - 0` clears the razor margin on move 1 and the root
            // loop abandons everything after it.
            // Under the table, "may I prune this?" is answered by the entry's own provenance rather than by
            // a positional cutoff: a sentinel or a stale score is skipped, never razored. This is what makes
            // ROOT_RAZOR_CONTINUE sound -- previously it wrote a sentinel that the NEXT iteration's razor
            // read back as a real score, which is why it collapsed to 92/300.
            bool razorable = i < current_search_data.synthetic_from;
            if (Config::ENABLE_ROOT_TABLE)
            {
                // Evidence test: a real score, recent enough, AND searched deep enough. The depth clause
                // matters once reductions can write the table -- a score from a heavily reduced search is
                // weak evidence, and treating it as full-depth is the "shallower overrides deeper" fault.
                razorable = current_search_data.scores[i].last_real != ROOT_SCORE_UNPROVEN && current_search_data.scores[i].age <= Config::ROOT_RAZOR_MAX_AGE && current_search_data.scores[i].last_real_depth >= depth_limit - Config::ROOT_RAZOR_MAX_DEPTH_DEFICIT;
                if (!razorable && Config::ENABLE_RAZORING && Config::ENABLE_ROOT_RAZOR && (score_diff > razor_threshold) && (alpha < 9000000))
                {
                    ++g_root_razor_stale_skips;
                    // This move looks bad but the evidence is too stale or too shallow to prune on. Paying
                    // full depth for it is what makes the table expensive; reduce instead, and let the
                    // re-search promote it if the reduction was wrong. The only branch here that removes
                    // work rather than adding it.
                    if (Config::ROOT_STALE_TO_LMR)
                    {
                        razor_r = Config::ROOT_STALE_LMR_BASE + (score_diff - razor_threshold) / std::max(1, Config::ROOT_STALE_LMR_DIV);
                        razor_r = std::min(razor_r, depth_limit - 2);
                        if (razor_r < 0)
                            razor_r = 0;
                        if (razor_r > 0)
                            ++g_razor_stale_reduced;
                    }
                }
            }
            if (razorable && Config::ENABLE_RAZORING && Config::ENABLE_ROOT_RAZOR && (score_diff > razor_threshold) && (alpha < 9000000))
            {
                ++g_razor_fires;
                if (root_razor_idx < 0)
                    root_razor_idx = static_cast<int>(i);
                if (Config::ENABLE_ROOT_TABLE && Config::ROOT_RAZOR_TO_LMR)
                {
                    // Hybrid: skip only the hopeless, reduce the merely suspect. The razor's own deficit is
                    // the reduction signal -- a MEASURED quantity, unlike the list index root LMR keys on,
                    // which carries no score information inside the sentinel block.
                    int over = score_diff - razor_threshold;
                    if (Config::ROOT_RAZOR_SKIP_MARGIN > 0 && over > Config::ROOT_RAZOR_SKIP_MARGIN)
                    {
                        ++g_razor_hybrid_skipped;
                        continue;
                    }
                    razor_r = Config::ROOT_RAZOR_LMR_BASE + over / std::max(1, Config::ROOT_RAZOR_LMR_DIV);
                    razor_r = std::min(razor_r, depth_limit - 2);
                    if (razor_r < 0)
                        razor_r = 0;
                    if (razor_r > 0)
                        ++g_razor_hybrid_reduced;
                }
                else if (Config::ENABLE_ROOT_TABLE)
                {
                    // The slot already holds ROOT_SCORE_UNPROVEN with an inherited reply list and an aged
                    // counter, so razoring needs no write at all -- skipping the move simply leaves it
                    // unproven for this iteration. Nothing desyncs because the table is pre-sized.
                    continue;
                }
                else if (Config::ROOT_RAZOR_CONTINUE)
                {
                    // Push a FRESH entry: scores must stay parallel with moves_list (skipping the push
                    // desyncs them and aborts), and every entry must carry a non-empty legal reply list
                    // because alpha_beta indexes second_moves unguarded -- reusing the loop's `entry` here
                    // would push a moved-from husk with an empty list. SF's analogue: a root move that does
                    // not beat alpha is stored as -VALUE_INFINITE, a sentinel for "unproven this iteration",
                    // never a measured value, and keeps its prior position under a stable sort.
                    RootScore razored;
                    razored.top_score = -9999998;
                    razored.second_moves = current_search_data.scores[i].second_moves;
                    previous_search_data.scores.push_back(std::move(razored));
                    continue; // skip only this stale-low move; keep searching later moves
                }
                else
                {
                    break; // default: abandon all remaining root moves (byte-identical)
                }
                // Hybrid path alone reaches here: razor_r is set and the move is searched reduced below.
            }
        }

        // Alpha as this move's search begins. A scout that fails low proves only "not better than alpha",
        // which is a bound rather than this move's value, so SF refuses to store it -- the table records a
        // real score only when the search beat this threshold.
        const int alpha_before_move = alpha;

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
        // Root LMR: reduce the null-window scout for late root moves instead of abandoning them. Eligibility
        // starts one move later while nothing has beaten the entry alpha, mirroring SF's
        // `moveCount > 1 + rootNode + (rootNode && bestValue < alpha)`. A reduced scout that beats alpha is
        // re-searched at FULL depth below, so a reduction can never by itself decide a root move.
        int root_r = 0;
        // Never reduce a move we have not measured. SF reduces by index in a list where EVERY root move
        // carries a real score, so every reduction rests on evidence. When our table is incomplete the
        // unscored moves are ordered by move_gen, which measures WORSE than arbitrary at the root -- so
        // reducing them is reducing at random, and a good move that fails low at reduced depth is lost.
        // Same rule as the razor guard: act on measurement, never on its absence.
        // A move that was best last iteration has the strongest evidence in the table of being best again,
        // so reducing it risks losing the very move the search is most likely to want. SF exempts it
        // outright; we approximate with the immediately preceding iteration's winner.
        bool root_lmr_exempt = Config::ROOT_LMR_EXEMPT_BEST && move.from_square == root_prev_best.from_square && move.to_square == root_prev_best.to_square && move.promotion == root_prev_best.promotion;
        if (Config::ENABLE_ROOT_LMR && depth_limit >= 3 && i < current_search_data.synthetic_from && !root_lmr_exempt)
        {
            int first_reduced = Config::ROOT_LMR_MIN_IDX + (best_score <= root_alpha_entry ? 1 : 0);
            if (static_cast<int>(i) >= first_reduced)
            {
                root_r = Config::ROOT_LMR_BASE + (static_cast<int>(i) - first_reduced) / std::max(1, Config::ROOT_LMR_DIV);
                root_r = std::min(root_r, depth_limit - 2);
                if (root_r < 0)
                    root_r = 0;
            }
        }
        else if (root_lmr_exempt && Config::ENABLE_ROOT_LMR && depth_limit >= 3 && i < current_search_data.synthetic_from && static_cast<int>(i) >= Config::ROOT_LMR_MIN_IDX + (best_score <= root_alpha_entry ? 1 : 0))
        {
            // Count only reductions the exemption actually declined. Below first_reduced the move would
            // not have been reduced anyway, so counting those would overstate the exemption's reach.
            ++g_root_lmr_exempt;
        }
        // The razor's score-deficit reduction and the index-keyed one measure different things; take the
        // deeper rather than summing, so the two cannot compound into an unsound reduction.
        if (razor_r > root_r)
            root_r = razor_r;
        if (root_r > 0)
            ++g_root_lmr_reduced;

        // Depth this move's kept score actually came from. Starts at the reduced depth and is promoted to
        // full whenever a re-search supersedes the scout, so the table never records a shallow value as
        // though it were deep.
        int searched_depth = depth_limit - root_r;

        score = minimizer(cur_depth + 1, depth_limit - root_r, alpha, alpha + 1, t0, current_search_data.scores[i].second_scores, current_search_data.scores[i].second_moves, entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);

        // A reduced scout that beats alpha is unreliable -- redo it at full depth before deciding anything.
        if (root_r > 0 && score > alpha)
        {
            ++g_root_lmr_researches;
            searched_depth = depth_limit;
            entry = RootScore{};
            score = minimizer(cur_depth + 1, depth_limit, alpha, alpha + 1, t0, current_search_data.scores[i].second_scores, current_search_data.scores[i].second_moves, entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, false);
        }

        // std::cout <<"FFF" << std::endl;
        //  If the score is within the window, re-search with full window. Discard the scout's entry first
        //  (this replaces the old second_level pop_backs) so the kept entry reflects the full-window search.
        if (alpha < score && score < beta)
        {
            searched_depth = depth_limit;
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
        if (Config::ENABLE_ROOT_TABLE)
            root_table_store(previous_search_data.scores[i], score, std::move(entry), score > alpha_before_move, searched_depth);
        else
        {
            entry.top_score = score;
            previous_search_data.scores.push_back(std::move(entry));
        }

        if (depth_limit >= 13)
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
        uint64_t simpl_move_to = 1ULL << move.to_square;
        bool move_is_simpl = capture_move && (simpl_move_to & current_state.occupied_colour[!current_state.turn]) && !(simpl_move_to & current_state.pawns);
        if (score > best_score)
        {
            best_move = move;
            best_score = score;
            best_move_index = i;
            best_is_simpl = move_is_simpl;

            if (score > alpha)
                updatePV(move, cur_depth);
        }
        else if (Config::ENABLE_SIMPL_BIAS && score >= best_score - Config::SIMPL_MARGIN && best_score > Config::SIMPL_AHEAD_THRESH && best_score < 9000000 // clearly ahead, not a mate score
                 && move_is_simpl && !best_is_simpl)
        {
            // Clearly ahead and this move trades a non-pawn piece at no score cost: prefer simplifying.
            best_move = move;
            best_move_index = i;
            best_is_simpl = true;
        }

        alpha = std::max(alpha, best_score);

        // Check for a beta cutoff
        if (beta <= alpha)
        {
            if (depth_limit >= 13)
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
            if (depth_limit >= 13)
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

        if (depth_limit >= 13)
        {
            std::cout << std::endl;
            std::cout << "Best: " << best_move_index << std::endl;
        }

        return best_score;
    }
    return best_score;
}

/*
    Stores THIS node's own search result, including its best move, under its own key.

    Every other store in the engine runs in the PARENT's frame -- keyed on the child's zobrist, with depth
    `depth_limit - cur_depth` measured from the parent. That is why TTEntry::move cannot be filled on a first
    visit: at the parent's store site the child's best move is out of scope. Singular extension probes for
    exactly that move at node entry, so it is starved by construction rather than by tuning.

    ⚠️ Depth deliberately reproduces the parent-frame convention (the node's own remaining PLUS ONE) instead
    of the node's own remaining. Probes compare against that same parent-frame expression, so a store using
    the node's own frame would sit one short and would SILENTLY never produce a hit.

    Skipped while a singular exclusion search is running at this ply, so the re-search cannot overwrite the
    node's real entry -- the same rule the existing cutoff-site move write already applies.

    Parameters:
        zobrist, current_state - this node's position
        score                  - the value being returned
        cur_depth, depth_limit - this node's own frame
        alpha_orig, beta_orig  - the node's entry window, for the bound flag
        best_move              - the move being credited; a null move is not stored
*/
inline void store_node_tt(uint64_t zobrist, const BoardState &current_state, std::vector<BoardState> &state_history,
                          int score, int cur_depth, int depth_limit, int alpha_orig, int beta_orig, const Move &best_move)
{
    if (!Config::ENABLE_NODE_TT)
        return;
    if (best_move.from_square == best_move.to_square)
        return;
    if (cur_depth >= 0 && cur_depth < MAX_PLY &&
        g_excluded_move[cur_depth].from_square != g_excluded_move[cur_depth].to_square)
        return;

    TTFlag flag = (score <= alpha_orig)  ? TTFlag::UPPERBOUND
                  : (score >= beta_orig) ? TTFlag::LOWERBOUND
                                         : TTFlag::EXACT;
    ++g_node_tt_stores;
    addToSearchEvalCache(zobrist, state_history.size(), score, depth_limit - cur_depth + 1, flag,
                         alpha_orig, beta_orig, current_state.castling_rights, current_state.ep_square, best_move);
}

inline int get_score_for_minimizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove, [[maybe_unused]] bool last_move_was_capture,
                                   std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state,
                                   bool &using_fp, int &num_iterations, bool is_in_null_search, bool &is_exact_hit)
{
    if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
        g_searchStack[cur_depth] = move; // record this node's move for the 2-ply continuation key
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
                bool lmr_eligible = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */);
                bool guard_pv = Config::PROTECT_PV && (beta - alpha > 1) && i <= Config::PROTECT_MAX_IDX;
                bool guard_killer = Config::PROTECT_KILLERS && i <= Config::PROTECT_MAX_IDX && (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previousMove.from_square][previousMove.to_square] == move);
                bool guard_capchain = Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS == 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH;
                bool base_lmr = lmr_eligible && !guard_pv && !guard_killer && !guard_capchain;
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
                // Inside an OTV verification window, keep reductions off for the first OTV_PLIES plies of the
                // re-searched subtree so the buried refutation is examined at honest depth. Inert (no-op) while
                // g_verify_no_reduce_until < 0 -- always so in the default build -> byte-identical.
                do_lmr = do_lmr && !(g_verify_no_reduce_until >= 0 && cur_depth <= g_verify_no_reduce_until);

                // Late-move pruning: at low remaining depth, skip late quiet moves. do_lmr eligibility
                // already excludes captures, checks, promotions, killers/counter and in-check, so a forcing
                // move is never pruned. Early return is safe — the caller unmakes the move (like futility).
                if (Config::ENABLE_LMP && do_lmr && !(Config::ENABLE_LMR_CAPCHAIN && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH))
                {
                    int rd = depth_limit - cur_depth;
                    bool lmp_reached = rd >= 1 && rd <= Config::LMP_MAX_DEPTH && (int)i >= Config::LMP_BASE + Config::LMP_SCALE * rd * rd;
                    bool lmp_exempt = Config::ENABLE_LMP_HIST_EXEMPT && historyHeuristics[current_state.turn][move.from_square][move.to_square] >= Config::LMP_HIST_EXEMPT;
                    if (lmp_reached && !lmp_exempt)
                    {
                        if (Config::ENABLE_PRUNE_SHADOW && shadow_fire())
                        {
                            g_in_shadow = true;
                            int shadow = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                            g_in_shadow = false;
                            shadow_record(true, true, shadow, alpha, beta);
                        }
                        return 9999999; // non-improving sentinel for the minimizer (never the new min, no false cutoff)
                    }
                }

                // History pruning: skip a late quiet the ordering strongly condemns (very negative butterfly
                // history) at low remaining depth. Reads the existing from×to table; default off = byte-identical.
                if (Config::ENABLE_HIST_PRUNE && do_lmr)
                {
                    int rd_hp = depth_limit - cur_depth;
                    if (rd_hp >= 1 && rd_hp <= Config::HIST_PRUNE_MAX_DEPTH &&
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] < -Config::HIST_PRUNE_COEF * rd_hp)
                    {
                        return 9999999;
                    }
                }

                // SEE pruning: at low remaining depth, skip a quiet whose moved piece can be profitably
                // captured by the immediate recapture (post-move see() from the opponent's side). do_lmr
                // already excludes captures/checks/promotions/killers/in-check. Default off = byte-identical.
                if (Config::ENABLE_SEE_PRUNE && do_lmr)
                {
                    int rd_see = depth_limit - cur_depth;
                    if (rd_see >= 1 && rd_see <= Config::SEE_PRUNE_MAX_DEPTH &&
                        see(move.to_square, updated_state.turn, updated_state) > Config::SEE_PRUNE_MARGIN)
                        return 9999999;
                }

                // SEE pruning of losing captures: a capture whose static exchange loses material (pre-move
                // see() from the mover's side) is skipped at low remaining depth. Captures bypass do_lmr; a
                // checking capture (extend) is never pruned. Default off = byte-identical.
                if (Config::SEE_PRUNE_CAPTURES && capture_move && !currently_in_check && !extend)
                {
                    int rd_cap = depth_limit - cur_depth;
                    if (rd_cap >= 1 && rd_cap <= Config::SEE_PRUNE_MAX_DEPTH &&
                        see(move.to_square, current_state.turn, current_state) < -Config::SEE_PRUNE_CAPTURE_MARGIN)
                        return 9999999;
                }

                // Null window search with LMR applied inside
                if (do_lmr)
                {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if (cur_depth > 1 && (depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin)
                    {
                        int early_score = eval_by_mode(Config::FUTILITY_EVAL_MODE, state_history, zobrist, num_iterations);
                        // int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, false);

                        if (Config::ENABLE_FUTILITY && (early_score - FUTILITY_MARGINS[depth_limit - cur_depth - 1] > beta))
                        {
                            if (Config::ENABLE_PRUNE_SHADOW && shadow_fire())
                            {
                                g_in_shadow = true;
                                int shadow = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                                g_in_shadow = false;
                                shadow_record(false, true, shadow, alpha, beta);
                            }
                            using_fp = true;
                            return early_score;
                        }
                    }
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    if (Config::ENABLE_HISTORY_LMR)
                    {
                        int hist_delta = history_lmr_delta(move, previousMove, current_state, cur_depth);
                        // History-scaled reduce-more: a quiet history_lmr_delta already flagged for extra
                        // reduction (tier-0, not killer/counter/cont-hist-rescued) is reduced FURTHER at
                        // deeper nodes (more remaining depth = the cheap/safe place to prune harder),
                        // scaled by HISTORY_LMR_SCALE (0 = off = byte-identical) and capped.
                        if (hist_delta < 0 && Config::HISTORY_LMR_SCALE > 0)
                            hist_delta -= std::min((depth_limit - cur_depth) / Config::HISTORY_LMR_SCALE, Config::HISTORY_LMR_SCALE_CAP);
                        if (hist_delta < 0 && is_in_relavent_pin)
                            hist_delta = 0; // keep the pin-defender protection; only reduce-LESS may touch pins
                        reduced_depth = std::clamp(reduced_depth + hist_delta, 2, depth_limit);
                    }
                    if (Config::ENABLE_IMPROVING)
                    {
                        bool improving = true;
                        if (cur_depth >= 2 && g_evalStack[cur_depth] != NO_STATIC_EVAL && g_evalStack[cur_depth - 2] != NO_STATIC_EVAL)
                            improving = g_evalStack[cur_depth] < g_evalStack[cur_depth - 2] + Config::IMPROVING_DELTA_MARGIN; // minimizer: a lower eval is improving for the side to move
                        if (!improving)
                            reduced_depth = std::clamp(reduced_depth - Config::IMPROVING_REDUCTION, 2, depth_limit);
                    }
                    // Capture-chain reduce-LESS: resolving a capture sequence -> search the quiet move
                    // closer to full depth so a forcing line is not buried by the reduction.
                    if (Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS > 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH)
                        reduced_depth = std::min(reduced_depth + Config::CAPCHAIN_REDUCE_LESS, depth_limit);
                    // Where only a ply or two remains, a constant ply-reduction truncates the child
                    // straight into qsearch; search it honestly instead (see LMR_MIN_REM).
                    if (Config::LMR_MIN_REM > 0 && (depth_limit - cur_depth - 1) < Config::LMR_MIN_REM)
                        reduced_depth = depth_limit;
                    // Keep a share of the child's remaining depth so a constant ply-reduction cannot
                    // truncate a deep node straight into qsearch (see LMR_REM_FLOOR_PCT).
                    if (Config::LMR_REM_FLOOR_PCT > 0)
                    {
                        int rem = depth_limit - cur_depth - 1;
                        if (rem > 0)
                        {
                            int floor_depth = cur_depth + 1 + (rem * Config::LMR_REM_FLOOR_PCT) / 100;
                            if (reduced_depth < floor_depth)
                                reduced_depth = std::min(floor_depth, depth_limit);
                        }
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
                            lmr_profile_event(depth_limit, cur_depth, i, alpha, beta, score, reduced_depth, move, current_state, previousMove, true);
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
                else if (Config::ENABLE_PRUNE_SHADOW && do_lmr && shadow_fire())
                {
                    // This move was REDUCED and then accepted without any verification re-search.
                    // Re-run it honestly (full depth, full window) and record whether the reduction
                    // buried a move that would have improved this node's bound.
                    g_in_shadow = true;
                    int full = maximizer(cur_depth + 1, depth_limit, alpha, beta, t0, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    g_in_shadow = false;
                    lmr_shadow_record(true, full, alpha, beta, cur_depth);
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

inline int get_score_for_maximizer(int alpha, int beta, int alpha_orig, int beta_orig, int i, int cur_depth, int depth_limit, bool capture_move, bool currently_in_check, Move move, Move previousMove, [[maybe_unused]] bool last_move_was_capture,
                                   std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state,
                                   bool &using_fp, int &num_iterations, bool is_in_null_search, bool &is_exact_hit)
{
    if (Config::ENABLE_CONT_HIST_2PLY && cur_depth < MAX_PLY)
        g_searchStack[cur_depth] = move; // record this node's move for the 2-ply continuation key
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
                bool lmr_eligible = Config::ENABLE_LMR && (i != 0 && !capture_move && !move_is_check && !currently_in_check && move.promotion == 1 /* && !relevant_pin_exists(state_history, false) */);
                bool guard_pv = Config::PROTECT_PV && (beta - alpha > 1) && i <= Config::PROTECT_MAX_IDX;
                bool guard_killer = Config::PROTECT_KILLERS && i <= Config::PROTECT_MAX_IDX && (killerMoves[cur_depth][0] == move || killerMoves[cur_depth][1] == move || counterMoves[previousMove.from_square][previousMove.to_square] == move);
                bool guard_capchain = Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS == 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH;
                bool base_lmr = lmr_eligible && !guard_pv && !guard_killer && !guard_capchain;
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
                // Inside an OTV verification window, keep reductions off for the first OTV_PLIES plies of the
                // re-searched subtree so the buried refutation is examined at honest depth. Inert (no-op) while
                // g_verify_no_reduce_until < 0 -- always so in the default build -> byte-identical.
                do_lmr = do_lmr && !(g_verify_no_reduce_until >= 0 && cur_depth <= g_verify_no_reduce_until);

                // Late-move pruning: at low remaining depth, skip late quiet moves. do_lmr eligibility
                // already excludes captures, checks, promotions, killers/counter and in-check, so a forcing
                // move is never pruned. Early return is safe — the caller unmakes the move (like futility).
                if (Config::ENABLE_LMP && do_lmr && !(Config::ENABLE_LMR_CAPCHAIN && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH))
                {
                    int rd = depth_limit - cur_depth;
                    bool lmp_reached = rd >= 1 && rd <= Config::LMP_MAX_DEPTH && (int)i >= Config::LMP_BASE + Config::LMP_SCALE * rd * rd;
                    bool lmp_exempt = Config::ENABLE_LMP_HIST_EXEMPT && historyHeuristics[current_state.turn][move.from_square][move.to_square] >= Config::LMP_HIST_EXEMPT;
                    if (lmp_reached && !lmp_exempt)
                    {
                        if (Config::ENABLE_PRUNE_SHADOW && shadow_fire())
                        {
                            g_in_shadow = true;
                            int shadow = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                            g_in_shadow = false;
                            shadow_record(true, false, shadow, alpha, beta);
                        }
                        return -9999999; // non-improving sentinel for the maximizer (never the new max, no false cutoff)
                    }
                }

                // History pruning: skip a late quiet the ordering strongly condemns (very negative butterfly
                // history) at low remaining depth. Reads the existing from×to table; default off = byte-identical.
                if (Config::ENABLE_HIST_PRUNE && do_lmr)
                {
                    int rd_hp = depth_limit - cur_depth;
                    if (rd_hp >= 1 && rd_hp <= Config::HIST_PRUNE_MAX_DEPTH &&
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] < -Config::HIST_PRUNE_COEF * rd_hp)
                    {
                        return -9999999;
                    }
                }

                // SEE pruning: at low remaining depth, skip a quiet whose moved piece can be profitably
                // captured by the immediate recapture (post-move see() from the opponent's side). do_lmr
                // already excludes captures/checks/promotions/killers/in-check. Default off = byte-identical.
                if (Config::ENABLE_SEE_PRUNE && do_lmr)
                {
                    int rd_see = depth_limit - cur_depth;
                    if (rd_see >= 1 && rd_see <= Config::SEE_PRUNE_MAX_DEPTH &&
                        see(move.to_square, updated_state.turn, updated_state) > Config::SEE_PRUNE_MARGIN)
                        return -9999999;
                }

                // SEE pruning of losing captures: a capture whose static exchange loses material (pre-move
                // see() from the mover's side) is skipped at low remaining depth. Captures bypass do_lmr; a
                // checking capture (extend) is never pruned. Default off = byte-identical.
                if (Config::SEE_PRUNE_CAPTURES && capture_move && !currently_in_check && !extend)
                {
                    int rd_cap = depth_limit - cur_depth;
                    if (rd_cap >= 1 && rd_cap <= Config::SEE_PRUNE_MAX_DEPTH &&
                        see(move.to_square, current_state.turn, current_state) < -Config::SEE_PRUNE_CAPTURE_MARGIN)
                        return -9999999;
                }

                // Null window search with LMR applied inside
                if (do_lmr)
                {
                    bool is_in_relavent_pin = relevant_pin_exists(state_history, false);
                    if ((depth_limit - cur_depth) <= 4 && (depth_limit - cur_depth) > 1 && depth_limit >= 5 && !is_in_relavent_pin)
                    {
                        int early_score = eval_by_mode(Config::FUTILITY_EVAL_MODE, state_history, zobrist, num_iterations);
                        // int early_score = get_q_search_eval(alpha, beta, cur_depth, t0, state_history, current_state, position_count, zobrist, previousMove, num_iterations, true);
                        if (Config::ENABLE_FUTILITY && (early_score + FUTILITY_MARGINS[depth_limit - cur_depth - 1] < alpha))
                        {
                            if (Config::ENABLE_PRUNE_SHADOW && shadow_fire())
                            {
                                g_in_shadow = true;
                                int shadow = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                                g_in_shadow = false;
                                shadow_record(false, false, shadow, alpha, beta);
                            }
                            using_fp = true;
                            return early_score;
                        }
                    }
                    int reduced_depth = reduced_search_depth(depth_limit, cur_depth, is_in_relavent_pin, i, current_state);
                    if (Config::ENABLE_HISTORY_LMR)
                    {
                        int hist_delta = history_lmr_delta(move, previousMove, current_state, cur_depth);
                        // History-scaled reduce-more: a quiet history_lmr_delta already flagged for extra
                        // reduction (tier-0, not killer/counter/cont-hist-rescued) is reduced FURTHER at
                        // deeper nodes (more remaining depth = the cheap/safe place to prune harder),
                        // scaled by HISTORY_LMR_SCALE (0 = off = byte-identical) and capped.
                        if (hist_delta < 0 && Config::HISTORY_LMR_SCALE > 0)
                            hist_delta -= std::min((depth_limit - cur_depth) / Config::HISTORY_LMR_SCALE, Config::HISTORY_LMR_SCALE_CAP);
                        if (hist_delta < 0 && is_in_relavent_pin)
                            hist_delta = 0; // keep the pin-defender protection; only reduce-LESS may touch pins
                        reduced_depth = std::clamp(reduced_depth + hist_delta, 2, depth_limit);
                    }
                    if (Config::ENABLE_IMPROVING)
                    {
                        bool improving = true;
                        if (cur_depth >= 2 && g_evalStack[cur_depth] != NO_STATIC_EVAL && g_evalStack[cur_depth - 2] != NO_STATIC_EVAL)
                            improving = g_evalStack[cur_depth] > g_evalStack[cur_depth - 2] - Config::IMPROVING_DELTA_MARGIN; // maximizer: a higher eval is improving for the side to move
                        if (!improving)
                            reduced_depth = std::clamp(reduced_depth - Config::IMPROVING_REDUCTION, 2, depth_limit);
                    }
                    // Capture-chain reduce-LESS: resolving a capture sequence -> search the quiet move
                    // closer to full depth so a forcing line is not buried by the reduction.
                    if (Config::ENABLE_LMR_CAPCHAIN && Config::CAPCHAIN_REDUCE_LESS > 0 && g_captureChain[cur_depth] >= Config::CAPCHAIN_RUN_THRESH)
                        reduced_depth = std::min(reduced_depth + Config::CAPCHAIN_REDUCE_LESS, depth_limit);
                    // Where only a ply or two remains, a constant ply-reduction truncates the child
                    // straight into qsearch; search it honestly instead (see LMR_MIN_REM).
                    if (Config::LMR_MIN_REM > 0 && (depth_limit - cur_depth - 1) < Config::LMR_MIN_REM)
                        reduced_depth = depth_limit;
                    // Keep a share of the child's remaining depth so a constant ply-reduction cannot
                    // truncate a deep node straight into qsearch (see LMR_REM_FLOOR_PCT).
                    if (Config::LMR_REM_FLOOR_PCT > 0)
                    {
                        int rem = depth_limit - cur_depth - 1;
                        if (rem > 0)
                        {
                            int floor_depth = cur_depth + 1 + (rem * Config::LMR_REM_FLOOR_PCT) / 100;
                            if (reduced_depth < floor_depth)
                                reduced_depth = std::min(floor_depth, depth_limit);
                        }
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
                            lmr_profile_event(depth_limit, cur_depth, i, alpha, beta, score, reduced_depth, move, current_state, previousMove, false);
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
                else if (Config::ENABLE_PRUNE_SHADOW && do_lmr && shadow_fire())
                {
                    // Mirror of the minimizer's LMR shadow: a reduced move accepted without a
                    // verification re-search is re-run at full depth/full window to see whether the
                    // reduction buried a move that would have improved this node's bound.
                    g_in_shadow = true;
                    int full = minimizer(cur_depth + 1, depth_limit, alpha, beta, t0, dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move, num_iterations, capture_move, false, is_in_null_search);
                    g_in_shadow = false;
                    lmr_shadow_record(false, full, alpha, beta, cur_depth);
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
    else if (Config::NODE_LIMIT > 0 && num_iterations >= Config::NODE_LIMIT)
    {
        time_up.store(true, std::memory_order_relaxed);
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

    // Throwaway containers for the singular exclusion re-search — minimizer's signature takes second-level
    // score/move lists + a RootScore by reference, which carry root bookkeeping; the exclusion search must
    // not touch them, so it gets its own dummies.
    std::vector<int> sing_dummy_ints;
    std::vector<Move> sing_dummy_moves;
    RootScore sing_dummy_entry;

    // Leaky capture-chain density (for the capture-chain LMR guard): rise on a capture into this node,
    // decay on a quiet move so a forcing sequence keeps its score across the odd quiet interruption.
    g_captureChain[cur_depth] = last_move_was_capture ? g_captureChain[cur_depth - 1] + 1 : std::max(0, g_captureChain[cur_depth - 1] - 1);

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
            ++g_draw_empty_ply1_min;
            // alpha_beta indexes scores[i].second_moves for every root move with no bounds or emptiness
            // guard, so this early return must not leave the ply-1 list empty. Carry the incoming list
            // through; regenerate only when it is itself empty, which pre_minimizer's own draw return can
            // produce. Masked today because the draw re-fires identically next iteration, but a persisted
            // root table would carry the empty entry into a position where the draw no longer holds.
            if (!second_level_moves_list.empty())
                out_entry.second_moves = second_level_moves_list;
            else
                out_entry.second_moves = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
            is_draw = true;
            return 0;
        }
        std::vector<int> cur_second_level_preliminary_scores;
        cur_second_level_preliminary_scores.reserve(64);

        ascending_sort(second_level_preliminary_scores, second_level_moves_list);
        out_entry.second_moves = second_level_moves_list;
        bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);
        std::vector<Move> searched_quiets, searched_captures; // history-gravity malus lists (empty = no cost when gravity off)
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
            score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove, last_move_was_capture,
                                            position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
            if (Config::ENABLE_HISTORY_MALUS || Config::ENABLE_CUTCAL_LOG || Config::ENABLE_QCUT)
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
                if (Config::ENABLE_CUTOFF_CLASS)
                    g_cutoff_class_hist[cutoff_move_class(move, current_state, cur_depth, previousMove)][i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
                // std::cout <<score << std::endl;
                out_entry.second_scores = cur_second_level_preliminary_scores;

                if (!capture_move)
                {
                    if (Config::ENABLE_CUTCAL_LOG && (depth_limit - cur_depth) >= 2)
                    {
                        Move p2c = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                        bool tn = current_state.turn;
                        long ssc = cutcal_statscore(move, previousMove, p2c, tn, current_state);
                        cutcal_record(ssc, true);
                        if (ssc >= 0 && ssc < SS_BUCKET_W)
                            g_cutcal.tf_cut[std::min(g_tf_count[tn][move.from_square][move.to_square], 3L)]++;
                        for (const Move &q : searched_quiets)
                        {
                            if (q == move)
                                continue;
                            long ssq = cutcal_statscore(q, previousMove, p2c, tn, current_state);
                            cutcal_record(ssq, false);
                            if (ssq >= 0 && ssq < SS_BUCKET_W)
                                g_cutcal.tf_fail[std::min(g_tf_count[tn][q.from_square][q.to_square], 3L)]++;
                            g_tf_count[tn][q.from_square][q.to_square]++;
                        }
                    }
                    storeKillerMove(cur_depth, move);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                    if (Config::ENABLE_QCUT)
                    {
                        qcut_update(g_qcut[current_state.turn][move.from_square][move.to_square], b, Config::QCUT_MAX);
                        for (const Move &q : searched_quiets)
                            if (!(q == move))
                                qcut_update(g_qcut[current_state.turn][q.from_square][q.to_square], -b / Config::QCUT_MALUS_DIV, Config::QCUT_MAX);
                    }
                    Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                    bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                    if (Config::ENABLE_HISTORY_SATURATION)
                    {
                        hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                        hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)], b);
                        if (p2v)
                            hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)], b / Config::CONT2_GRAVITY_DIV);
                        if (Config::ENABLE_PIECE_CONTHIST)
                            hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)], b);
                    }
                    else
                    {
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                        counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                        if (p2v)
                            contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                        if (Config::ENABLE_PIECE_CONTHIST)
                            pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)] += 4 * b;
                    }
                    threat_hist_on_cutoff(current_state, move, searched_quiets, b, Config::ENABLE_HISTORY_SATURATION, Config::MALUS_DIV, Config::ENABLE_HISTORY_MALUS);
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_quiets)
                        {
                            if (q == move)
                                continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                            {
                                hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                                hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                                if (p2v)
                                    hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                                if (Config::ENABLE_PIECE_CONTHIST)
                                    hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                            }
                            else
                            {
                                historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                                counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                                if (p2v)
                                    contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                                if (Config::ENABLE_PIECE_CONTHIST)
                                    pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                            }
                        }
                    }
                }
                else if (Config::ENABLE_CAPTURE_HIST)
                {
                    int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                    if (Config::ENABLE_HISTORY_SATURATION)
                        hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                    else
                        captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_captures)
                        {
                            if (q == move)
                                continue;
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
                                         ? static_eval_for_improving(state_history, zobrist)
                                         : NO_STATIC_EVAL;
        // Node-entry static eval, shared by RFP and the null-move eval gate (computed only when needed).
        int rfp_static_eval = NO_STATIC_EVAL;
        bool rfp_want_eval = !currently_in_check &&
                             ((Config::ENABLE_RFP && (beta - alpha) == 1 &&
                               (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH) ||
                              Config::ENABLE_NULL_EVAL_GATE);
        if (rfp_want_eval)
        {
            rfp_static_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
            if (Config::ENABLE_CORR_HIST)
                rfp_static_eval += corrhist_correction(current_state, 0);
        }

        // Reverse futility (static null): so far below alpha a quiet move is assumed to hold.
        if (Config::ENABLE_RFP && rfp_static_eval != NO_STATIC_EVAL &&
            (beta - alpha) == 1 && alpha > -9000000 && alpha < 9000000 &&
            (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH &&
            rfp_static_eval + Config::RFP_MARGIN * (depth_limit - cur_depth) <= alpha)
        {
            if (Config::ENABLE_PRUNE_LOG)
                log_prune_fire("RFP_MIN", current_state, alpha, beta, depth_limit - cur_depth, rfp_static_eval,
                               cheap_eval(current_state.pawns, current_state.knights, current_state.bishops,
                                          current_state.rooks, current_state.queens, current_state.kings,
                                          current_state.occupied_colour[true], current_state.occupied_colour[false]));
            // Pull the returned score toward the bound rather than trusting the unverified static eval (SF16+).
            if (Config::RFP_RETURN_BLEND > 0)
                return (Config::RFP_RETURN_BLEND * alpha + (100 - Config::RFP_RETURN_BLEND) * rfp_static_eval) / 100;
            return rfp_static_eval;
        }

        // Null Move Pruning
        if (Config::ENABLE_NULLMOVE && cur_depth >= Config::NULLMOVE_CURDEPTH_MINI && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state) && (!Config::ENABLE_NULL_EVAL_GATE || rfp_static_eval <= alpha))
        {
            state_history.back().turn = !state_history.back().turn;

            int cur_ep_square = state_history.back().ep_square;
            state_history.back().ep_square = -1;
            updateZobristHashForNullMove(zobrist);

            int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];
            reduced_depth -= Config::NULLMOVE_EXTRA;

            if (depth_limit >= 10)
                reduced_depth -= 1;
            // Eval-scaled null reduction (LOW side): the further the pre-null static eval sits BELOW alpha,
            // the more confidently the null holds -> reduce more. Only when the static eval is already valid.
            if (Config::ENABLE_NULLMOVE_EVAL_R && rfp_static_eval != NO_STATIC_EVAL)
            {
                int extra_R = std::min((alpha - rfp_static_eval) / Config::NULLMOVE_R_DIV, Config::NULLMOVE_R_CAP);
                if (extra_R > 0)
                    reduced_depth -= extra_R;
            }
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
                    g_searchStack[cur_depth] = dummyMove; // null move breaks the 2-ply continuation chain
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

        // IIR: reduce this node's search depth by 1 when it has no cached ordering evidence (movegen-cache
        // miss = first visit) and depth to spare, so the shallow pass seeds a good first move for the
        // re-search. Off => byte-identical.
        if (Config::ENABLE_IIR && !currently_in_check && (depth_limit - cur_depth) >= Config::IIR_MIN_DEPTH && !moveGenCacheHasMoves(zobrist, current_state.castling_rights, current_state.ep_square))
            depth_limit -= 1;

        std::vector<Move> &moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
        std::vector<Move> searched_quiets, searched_captures; // history-gravity malus lists (empty = no cost when gravity off)

        // OTV needs the node's OWN static eval as the phantom reference. Compute it once here, at the node
        // position before any child move is made, when OTV is eligible and RFP/null-gate did not already.
        // Gated on ENABLE_OTV, so the default build computes no extra eval and stays byte-identical.
        if (Config::ENABLE_OTV && rfp_static_eval == NO_STATIC_EVAL && !g_in_verify && !currently_in_check &&
            (!Config::OTV_PV_ONLY || (beta - alpha) > 1) &&
            (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING)
            rfp_static_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);

        // ProbCut: at a non-PV node with depth to spare, a shallow null-window search a margin BELOW alpha
        // on the strong captures/promos confirms the position is losing enough to skip the full-depth search.
        // Self-verifying (the child re-searches), so it holds despite an imperfect static eval. Minimizing
        // side wants the score LOW -> push the window DOWN: a confirmed fail-low past alpha-PROBCUT_MARGIN
        // means the true score is at most alpha, so we can cut here.
        if (Config::ENABLE_PROBCUT && !g_in_verify && !currently_in_check && (beta - alpha) == 1 &&
            (depth_limit - cur_depth) >= Config::PROBCUT_MIN_DEPTH &&
            alpha > -9000000 && alpha < 9000000 && beta > -9000000 && beta < 9000000)
        {
            int probcut_alpha = alpha - Config::PROBCUT_MARGIN;
            int probcut_tried = 0;
            for (size_t i = 0; i < moves_list.size() && probcut_tried < Config::PROBCUT_CANDIDATES; ++i)
            {
                Move &move = moves_list[i];
                bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
                bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
                if (!(capture_move || move.promotion != 1))
                    continue;
                if (see(move.to_square, current_state.turn, current_state) < 0)
                    continue;
                ++probcut_tried;

                updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move,
                                         current_state.pawns, current_state.knights, current_state.bishops,
                                         current_state.rooks, current_state.queens, current_state.kings,
                                         current_state.occupied_colour[true], current_state.occupied_colour[false], move.promotion);
                make_move(state_history, position_count, move, zobrist, capture_move);
                bool prev_no_store = g_no_tt_store;
                if (Config::ENABLE_PROBCUT_NO_TT_STORE)
                    g_no_tt_store = true;
                int probcut_score = maximizer(cur_depth + 1, depth_limit - Config::PROBCUT_DEPTH_REDUCTION, probcut_alpha, probcut_alpha + 1, t0,
                                              state_history, position_count, zobrist, move,
                                              num_iterations, capture_move, false, is_in_null_search);
                g_no_tt_store = prev_no_store;
                unmake_move(state_history, position_count, zobrist);
                zobrist = cur_hash;

                if (time_up.load(std::memory_order_relaxed))
                    return 0;

                if (probcut_score <= probcut_alpha && probcut_score > -9000000)
                    return probcut_score;
            }
        }

        // Singular node-entry probe (read-only): the TT-move to verify + its value/depth/bound. excluding =
        // we are inside an exclusion re-search of THIS node (skip the tested move + don't re-fire singular).
        bool excluding = Config::ENABLE_SINGULAR && (g_excluded_move[cur_depth].from_square != g_excluded_move[cur_depth].to_square);
        Move ttMove;
        int ttScore = 0, ttDepth = -1;
        TTFlag ttFlag = TTFlag::EXACT;
        bool haveTT = false;
        if (Config::ENABLE_SINGULAR && !excluding)
        {
            TTEntry *nodeTT = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
            if (nodeTT != nullptr)
            {
                ttMove = nodeTT->move;
                ttScore = nodeTT->score;
                ttDepth = nodeTT->depth;
                ttFlag = nodeTT->flag;
                haveTT = true;
            }
            if (haveTT && ttMove.from_square != ttMove.to_square)
                ++g_sing_eligible;
        }

        for (size_t i = 0; i < moves_list.size(); ++i)
        {
            Move &move = moves_list[i];
            if (excluding && move == g_excluded_move[cur_depth])
                continue;
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

            // Singular extension (min side): if this is the TT-move (a fail-low UPPERBOUND) and a shallow
            // exclusion search of the OTHER moves cannot reach ttScore + margin (stay as low), the move is
            // uniquely good -> extend it one ply. The exclusion re-searches THIS node (same cur_depth, half
            // depth, null window at sb) with the move skipped via g_excluded_move; child-store => no TT poison.
            int singular_extra = 0;
            if (Config::ENABLE_SINGULAR && !excluding && haveTT && !currently_in_check && move == ttMove &&
                ttMove.from_square != ttMove.to_square && (depth_limit - cur_depth) >= Config::SINGULAR_MIN_DEPTH &&
                ttDepth >= (depth_limit - cur_depth) - 3 && ttScore > -9000000 && ttScore < 9000000 &&
                ttFlag == TTFlag::UPPERBOUND && g_singular_extensions < Config::SINGULAR_MAX_EXT)
            {
                ++g_sing_gatepass;
                int rem = depth_limit - cur_depth;
                int sb = ttScore + Config::SINGULAR_MARGIN * rem;
                g_excluded_move[cur_depth] = move;
                int v = minimizer(cur_depth, cur_depth + rem / 2, sb, sb + 1, t0, sing_dummy_ints, sing_dummy_moves, sing_dummy_entry, state_history, position_count, zobrist, previousMove, num_iterations, last_move_was_capture, false, is_in_null_search);
                g_excluded_move[cur_depth] = Move();
                if (v > sb)
                {
                    ++g_sing_fire;
                    singular_extra = 1;
                }
            }

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
            {
                // Count this singular extension on the path for the duration of its child search so a chain
                // of singular moves is bounded by SINGULAR_MAX_EXT.
                SingularExtensionGuard seg(singular_extra > 0);
                score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit + singular_extra, capture_move, currently_in_check, move, previousMove, last_move_was_capture,
                                                position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
            }
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

            // Optimism-triggered verification (minimizing node): this child's score is about to be trusted
            // (become the new best / cause the cutoff) and it UNDERSHOOTS the node's own static eval by more
            // than OTV_MARGIN -- the phantom signature here, a score too LOW = the minimizing side is
            // over-optimistic. Re-search the child (still on the board) with reductions off for the first
            // OTV_PLIES plies of its subtree; the existing best-score logic below then runs on the corrected
            // score. Same callee/args as the node's normal full child search, so a fail-high hits the same
            // full-depth re-search chain.
            bool verified_this_move = false;
            if (Config::ENABLE_OTV && !g_in_verify && !verified_this_move &&
                g_verify_count < Config::OTV_PATH_CAP &&
                rfp_static_eval != NO_STATIC_EVAL && score < lowest_score &&
                (!Config::OTV_PV_ONLY || (beta - alpha) > 1) &&
                (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING &&
                (rfp_static_eval - score) > Config::OTV_MARGIN)
            {
                VerifyGuard vg(cur_depth, Config::OTV_PLIES);
                score = get_score_for_minimizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove, last_move_was_capture,
                                                position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
                verified_this_move = true;
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
                if (Config::ENABLE_CUTOFF_CLASS)
                    g_cutoff_class_hist[cutoff_move_class(move, current_state, cur_depth, previousMove)][i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
                // Node-local best-move populate (singular's TT-move): stamp this node's cutoff move onto its
                // OWN TT entry (key-verified), so singular can verify/exclude it. Move-only write — touches
                // nothing the search reads until singular or TT-move ordering does, and the SF move-rule in
                // tt_store preserves it against the parent's later child-keyed score store. ENABLE_TT_MOVE
                // shares the store because it is the only other consumer of this field; with both flags off
                // the write never happens = byte-identical.
                // Skip while excluding so a min exclusion re-search doesn't overwrite the node's real TT-move.
                if ((Config::ENABLE_SINGULAR || Config::ENABLE_TT_MOVE) && !excluding)
                {
                    TTEntry *nodeEntry = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
                    if (nodeEntry != nullptr)
                        nodeEntry->move = move;
                }
                if (i != 0)
                    updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

                if (!capture_move)
                {
                    storeKillerMove(cur_depth, move);
                    counterMoves[previousMove.from_square][previousMove.to_square] = move;
                    int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                    Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                    bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                    if (Config::ENABLE_HISTORY_SATURATION)
                    {
                        hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                        hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)], b);
                        if (p2v)
                            hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)], b / Config::CONT2_GRAVITY_DIV);
                        if (Config::ENABLE_PIECE_CONTHIST)
                            hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)], b);
                    }
                    else
                    {
                        historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                        counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                        if (p2v)
                            contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                        if (Config::ENABLE_PIECE_CONTHIST)
                            pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)] += 4 * b;
                    }
                    threat_hist_on_cutoff(current_state, move, searched_quiets, b, Config::ENABLE_HISTORY_SATURATION, Config::MALUS_DIV, Config::ENABLE_HISTORY_MALUS);
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_quiets)
                        {
                            if (q == move)
                                continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                            {
                                hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                                hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                                if (p2v)
                                    hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                                if (Config::ENABLE_PIECE_CONTHIST)
                                    hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                            }
                            else
                            {
                                historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                                counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                                if (p2v)
                                    contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                                if (Config::ENABLE_PIECE_CONTHIST)
                                    pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                            }
                        }
                    }
                }
                else if (Config::ENABLE_CAPTURE_HIST)
                {
                    int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                    if (Config::ENABLE_HISTORY_SATURATION)
                        hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                    else
                        captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                    if (Config::ENABLE_HISTORY_MALUS)
                    {
                        for (const Move &q : searched_captures)
                        {
                            if (q == move)
                                continue;
                            if (Config::ENABLE_HISTORY_SATURATION)
                                hist_update(captureHistory[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                            else
                                captureHistory[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                        }
                    }
                }

                store_node_tt(zobrist, current_state, state_history, lowest_score, cur_depth, depth_limit, alpha_orig, beta_orig, best_move);
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
        if ((Config::ENABLE_CORR_HIST || Config::ENABLE_CORRHIST_LOG) && std::abs(lowest_score) < 9000000 &&
            !is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]))
        {
            int se = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
            if (Config::ENABLE_CORR_HIST)
                corrhist_update(current_state, 0, se, lowest_score);
            if (Config::ENABLE_CORRHIST_LOG)
                corrhist_log(generatePawnKey(current_state.pawns, current_state.occupied_colour[true], current_state.occupied_colour[false]),
                             0, se, lowest_score, depth_limit - cur_depth);
        }
    }

    store_node_tt(zobrist, current_state, state_history, lowest_score, cur_depth, depth_limit, alpha_orig, beta_orig, best_move);
    return lowest_score;
}

int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count,
              uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search)
{

    BoardState current_state = state_history.back();

    // Leaky capture-chain density (for the capture-chain LMR guard): rise on a capture into this node,
    // decay on a quiet move so a forcing sequence keeps its score across the odd quiet interruption.
    g_captureChain[cur_depth] = last_move_was_capture ? g_captureChain[cur_depth - 1] + 1 : std::max(0, g_captureChain[cur_depth - 1] - 1);

    if (time_up.load(std::memory_order_relaxed))
    {
        return 0;
    }
    else if (Config::NODE_LIMIT > 0 && num_iterations >= Config::NODE_LIMIT)
    {
        time_up.store(true, std::memory_order_relaxed);
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
                                     ? static_eval_for_improving(state_history, zobrist)
                                     : NO_STATIC_EVAL;

    // Node-entry static eval, shared by RFP and the null-move eval gate (computed only when needed).
    int rfp_static_eval = NO_STATIC_EVAL;
    bool rfp_want_eval = !currently_in_check &&
                         ((Config::ENABLE_RFP && (beta - alpha) == 1 &&
                           (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH) ||
                          Config::ENABLE_NULL_EVAL_GATE);
    if (rfp_want_eval)
    {
        rfp_static_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
        if (Config::ENABLE_CORR_HIST)
            rfp_static_eval += corrhist_correction(current_state, 1);
    }

    // Reverse futility (static null): so far above beta a quiet move is assumed to hold.
    if (Config::ENABLE_RFP && rfp_static_eval != NO_STATIC_EVAL &&
        (beta - alpha) == 1 && beta > -9000000 && beta < 9000000 &&
        (depth_limit - cur_depth) >= Config::RFP_MIN_DEPTH && (depth_limit - cur_depth) <= Config::RFP_MAX_DEPTH &&
        rfp_static_eval - Config::RFP_MARGIN * (depth_limit - cur_depth) >= beta)
    {
        if (Config::ENABLE_PRUNE_LOG)
            log_prune_fire("RFP_MAX", current_state, alpha, beta, depth_limit - cur_depth, rfp_static_eval,
                           cheap_eval(current_state.pawns, current_state.knights, current_state.bishops,
                                      current_state.rooks, current_state.queens, current_state.kings,
                                      current_state.occupied_colour[true], current_state.occupied_colour[false]));
        // Pull the returned score toward the bound rather than trusting the unverified static eval (SF16+).
        if (Config::RFP_RETURN_BLEND > 0)
            return (Config::RFP_RETURN_BLEND * beta + (100 - Config::RFP_RETURN_BLEND) * rfp_static_eval) / 100;
        return rfp_static_eval;
    }

    // Null Move Pruning
    if (Config::ENABLE_NULLMOVE && cur_depth >= Config::NULLMOVE_CURDEPTH_MAXI && depth_limit >= 5 && !last_move_was_capture && !last_move_was_null_move && !currently_in_check && !isUnsafeForNullMovePruning(current_state) && (!Config::ENABLE_NULL_EVAL_GATE || rfp_static_eval >= beta))
    {
        state_history.back().turn = !state_history.back().turn;

        int cur_ep_square = state_history.back().ep_square;
        state_history.back().ep_square = -1;
        updateZobristHashForNullMove(zobrist);

        int reduced_depth = Config::ACTIVE->DEPTH_REDUCTION[depth_limit];
        reduced_depth -= Config::NULLMOVE_EXTRA;

        if (depth_limit >= 10)
            reduced_depth -= 1;
        // Eval-scaled null reduction (HIGH side): the further the pre-null static eval sits ABOVE beta,
        // the more confidently the null holds -> reduce more. Only when the static eval is already valid.
        if (Config::ENABLE_NULLMOVE_EVAL_R && rfp_static_eval != NO_STATIC_EVAL)
        {
            int extra_R = std::min((rfp_static_eval - beta) / Config::NULLMOVE_R_DIV, Config::NULLMOVE_R_CAP);
            if (extra_R > 0)
                reduced_depth -= extra_R;
        }
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
                g_searchStack[cur_depth] = dummyMove; // null move breaks the 2-ply continuation chain
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

    // IIR: reduce this node's search depth by 1 when it has no cached ordering evidence (movegen-cache
    // miss = first visit) and depth to spare, so the shallow pass seeds a good first move for the
    // re-search. Off => byte-identical.
    if (Config::ENABLE_IIR && !currently_in_check && (depth_limit - cur_depth) >= Config::IIR_MIN_DEPTH && !moveGenCacheHasMoves(zobrist, current_state.castling_rights, current_state.ep_square))
        depth_limit -= 1;

    std::vector<Move> &moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, previousMove);
    std::vector<Move> searched_quiets, searched_captures; // history-gravity malus lists (empty = no cost when gravity off)

    // OTV needs the node's OWN static eval as the phantom reference. Compute it once here, at the node
    // position before any child move is made, when OTV is eligible and RFP/null-gate did not already.
    // Gated on ENABLE_OTV, so the default build computes no extra eval and stays byte-identical.
    if (Config::ENABLE_OTV && rfp_static_eval == NO_STATIC_EVAL && !g_in_verify && !currently_in_check &&
        (!Config::OTV_PV_ONLY || (beta - alpha) > 1) &&
        (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING)
        rfp_static_eval = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
    /* if (create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks,
                                current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true],
                                current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights,
                                current_state.ep_square, current_state.turn) == "8/8/3R3P/4P1P1/5P2/5K2/2k5/2q1b3 w - - 0 1"){
                                    std::cout << "AAAA" << std::endl;

    } */

    // ProbCut: at a non-PV node with depth to spare, a shallow null-window search a margin ABOVE beta on
    // the strong captures/promos confirms the position is winning enough to skip the full-depth search.
    // Self-verifying (the child re-searches), so it holds despite an imperfect static eval. Maximizing side
    // wants the score HIGH -> push the window UP: a confirmed fail-high past beta+PROBCUT_MARGIN means the
    // true score is at least beta, so we can cut here.
    if (Config::ENABLE_PROBCUT && !g_in_verify && !currently_in_check && (beta - alpha) == 1 &&
        (depth_limit - cur_depth) >= Config::PROBCUT_MIN_DEPTH &&
        alpha > -9000000 && alpha < 9000000 && beta > -9000000 && beta < 9000000)
    {
        int probcut_beta = beta + Config::PROBCUT_MARGIN;
        int probcut_tried = 0;
        for (size_t i = 0; i < moves_list.size() && probcut_tried < Config::PROBCUT_CANDIDATES; ++i)
        {
            Move &move = moves_list[i];
            bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
            bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);
            if (!(capture_move || move.promotion != 1))
                continue;
            if (see(move.to_square, current_state.turn, current_state) < 0)
                continue;
            ++probcut_tried;

            updateZobristHashForMove(zobrist, move.from_square, move.to_square, capture_move,
                                     current_state.pawns, current_state.knights, current_state.bishops,
                                     current_state.rooks, current_state.queens, current_state.kings,
                                     current_state.occupied_colour[true], current_state.occupied_colour[false], move.promotion);
            make_move(state_history, position_count, move, zobrist, capture_move);
            bool prev_no_store = g_no_tt_store;
            if (Config::ENABLE_PROBCUT_NO_TT_STORE)
                g_no_tt_store = true;
            int probcut_score = minimizer(cur_depth + 1, depth_limit - Config::PROBCUT_DEPTH_REDUCTION, probcut_beta - 1, probcut_beta, t0,
                                          dummy_ints, dummy_moves, dummy_entry, state_history, position_count, zobrist, move,
                                          num_iterations, capture_move, false, is_in_null_search);
            g_no_tt_store = prev_no_store;
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;

            if (time_up.load(std::memory_order_relaxed))
                return 0;

            if (probcut_score >= probcut_beta && probcut_score < 9000000)
                return probcut_score;
        }
    }

    // Singular node-entry probe (read-only): the TT-move to verify + its value/depth/bound. excluding = we
    // are inside an exclusion re-search of THIS node (skip the tested move + don't re-fire singular).
    bool excluding = Config::ENABLE_SINGULAR && (g_excluded_move[cur_depth].from_square != g_excluded_move[cur_depth].to_square);
    Move ttMove;
    int ttScore = 0, ttDepth = -1;
    TTFlag ttFlag = TTFlag::EXACT;
    bool haveTT = false;
    if (Config::ENABLE_SINGULAR && !excluding)
    {
        TTEntry *nodeTT = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
        if (nodeTT != nullptr)
        {
            ttMove = nodeTT->move;
            ttScore = nodeTT->score;
            ttDepth = nodeTT->depth;
            ttFlag = nodeTT->flag;
            haveTT = true;
        }
        if (haveTT && ttMove.from_square != ttMove.to_square)
            ++g_sing_eligible;
    }

    for (size_t i = 0; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];
        if (excluding && move == g_excluded_move[cur_depth])
            continue;
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

        // Singular extension (max side): if this is the TT-move (a fail-high LOWERBOUND) and a shallow
        // exclusion search of the OTHER moves cannot reach ttScore - margin, the move is uniquely good ->
        // extend it one ply. The exclusion re-searches THIS node (same cur_depth, half depth, null window at
        // sb) with the move skipped via g_excluded_move; our child-store architecture => no TT poison.
        int singular_extra = 0;
        if (Config::ENABLE_SINGULAR && !excluding && haveTT && !currently_in_check && move == ttMove &&
            ttMove.from_square != ttMove.to_square && (depth_limit - cur_depth) >= Config::SINGULAR_MIN_DEPTH &&
            ttDepth >= (depth_limit - cur_depth) - 3 && ttScore > -9000000 && ttScore < 9000000 &&
            ttFlag == TTFlag::LOWERBOUND && g_singular_extensions < Config::SINGULAR_MAX_EXT)
        {
            ++g_sing_gatepass;
            int rem = depth_limit - cur_depth;
            int sb = ttScore - Config::SINGULAR_MARGIN * rem;
            g_excluded_move[cur_depth] = move;
            int v = maximizer(cur_depth, cur_depth + rem / 2, sb - 1, sb, t0, state_history, position_count, zobrist, previousMove, num_iterations, last_move_was_capture, false, is_in_null_search);
            g_excluded_move[cur_depth] = Move();
            if (v < sb)
            {
                ++g_sing_fire;
                singular_extra = 1;
            }
        }

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
        {
            // Count this singular extension on the path for the duration of its child search so a chain
            // of singular moves is bounded by SINGULAR_MAX_EXT.
            SingularExtensionGuard seg(singular_extra > 0);
            score = get_score_for_maximizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit + singular_extra, capture_move, currently_in_check, move, previousMove, last_move_was_capture,
                                            position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
        }
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

        // Optimism-triggered verification (maximizing node): this child's score is about to be trusted
        // (become the new best / cause the cutoff) and it OVERSHOOTS the node's own static eval by more than
        // OTV_MARGIN -- the phantom signature here, a score too HIGH = the maximizing side is over-optimistic.
        // Re-search the child (still on the board) with reductions off for the first OTV_PLIES plies of its
        // subtree; the existing best-score logic below then runs on the corrected score. Same callee/args as
        // the node's normal full child search, so a fail-high hits the same full-depth re-search chain.
        bool verified_this_move = false;
        if (Config::ENABLE_OTV && !g_in_verify && !verified_this_move &&
            g_verify_count < Config::OTV_PATH_CAP &&
            rfp_static_eval != NO_STATIC_EVAL && score > highest_score &&
            (!Config::OTV_PV_ONLY || (beta - alpha) > 1) &&
            (depth_limit - cur_depth) >= Config::OTV_MIN_REMAINING &&
            (score - rfp_static_eval) > Config::OTV_MARGIN)
        {
            VerifyGuard vg(cur_depth, Config::OTV_PLIES);
            score = get_score_for_maximizer(alpha, beta, alpha_orig, beta_orig, i, cur_depth, depth_limit, capture_move, currently_in_check, move, previousMove, last_move_was_capture,
                                            position_count, zobrist, t0, state_history, current_state, using_fp, num_iterations, is_in_null_search, is_exact_hit);
            verified_this_move = true;
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
            if (Config::ENABLE_CUTOFF_CLASS)
                g_cutoff_class_hist[cutoff_move_class(move, current_state, cur_depth, previousMove)][i < 3 ? (int)i : (i < 8 ? 3 : 4)]++;
            // Node-local best-move populate (singular's TT-move) — see the minimizer cutoff for rationale.
            // Skip while excluding so a max exclusion re-search doesn't overwrite the node's real TT-move.
            if ((Config::ENABLE_SINGULAR || Config::ENABLE_TT_MOVE) && !excluding)
            {
                TTEntry *nodeEntry = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
                if (nodeEntry != nullptr)
                    nodeEntry->move = move;
            }
            if (i != 0)
                updateMoveCacheForBetaCutoff(zobrist, current_state.castling_rights, current_state.ep_square, move, moves_list, state_history);

            if (!capture_move)
            {
                storeKillerMove(cur_depth, move);
                counterMoves[previousMove.from_square][previousMove.to_square] = move;
                int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                Move p2 = (cur_depth >= 2) ? g_searchStack[cur_depth - 2] : Move{};
                bool p2v = Config::ENABLE_CONT_HIST_2PLY && p2.from_square != p2.to_square;
                if (Config::ENABLE_HISTORY_SATURATION)
                {
                    hist_update(historyHeuristics[current_state.turn][move.from_square][move.to_square], b);
                    hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)], b);
                    if (p2v)
                        hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)], b / Config::CONT2_GRAVITY_DIV);
                    if (Config::ENABLE_PIECE_CONTHIST)
                        hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)], b);
                }
                else
                {
                    historyHeuristics[current_state.turn][move.from_square][move.to_square] += b;
                    counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                    if (p2v)
                        contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)] += 4 * b;
                    if (Config::ENABLE_PIECE_CONTHIST)
                        pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(move, current_state)] += 4 * b;
                }
                threat_hist_on_cutoff(current_state, move, searched_quiets, b, Config::ENABLE_HISTORY_SATURATION, Config::MALUS_DIV, Config::ENABLE_HISTORY_MALUS);
                if (Config::ENABLE_HISTORY_MALUS)
                {
                    for (const Move &q : searched_quiets)
                    {
                        if (q == move)
                            continue;
                        if (Config::ENABLE_HISTORY_SATURATION)
                        {
                            hist_update(historyHeuristics[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                            hist_update(counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                            if (p2v)
                                hist_update(contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)], -b / Config::CONT2_GRAVITY_DIV / Config::MALUS_DIV);
                            if (Config::ENABLE_PIECE_CONTHIST)
                                hist_update(pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)], -b / Config::MALUS_DIV);
                        }
                        else
                        {
                            historyHeuristics[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                            counterMoveHeuristics[current_state.turn][cont_ctx_key(previousMove, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                            if (p2v)
                                contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                            if (Config::ENABLE_PIECE_CONTHIST)
                                pieceContHist[current_state.turn][pcont_ctx_key(previousMove, current_state)][pcont_ent_key(q, current_state)] -= 4 * b / Config::MALUS_DIV;
                        }
                    }
                }
            }
            else if (Config::ENABLE_CAPTURE_HIST)
            {
                int b = ((depth_limit - cur_depth) * (depth_limit - cur_depth) * Config::HISTORY_BONUS_SCALE) / 100;
                if (Config::ENABLE_HISTORY_SATURATION)
                    hist_update(captureHistory[current_state.turn][move.from_square][move.to_square], b);
                else
                    captureHistory[current_state.turn][move.from_square][move.to_square] += b;
                if (Config::ENABLE_HISTORY_MALUS)
                {
                    for (const Move &q : searched_captures)
                    {
                        if (q == move)
                            continue;
                        if (Config::ENABLE_HISTORY_SATURATION)
                            hist_update(captureHistory[current_state.turn][q.from_square][q.to_square], -b / Config::MALUS_DIV);
                        else
                            captureHistory[current_state.turn][q.from_square][q.to_square] -= b / Config::MALUS_DIV;
                    }
                }
            }

            store_node_tt(zobrist, current_state, state_history, highest_score, cur_depth, depth_limit, alpha_orig, beta_orig, best_move);
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
        if ((Config::ENABLE_CORR_HIST || Config::ENABLE_CORRHIST_LOG) && std::abs(highest_score) < 9000000 &&
            !is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]))
        {
            int se = eval_by_mode(Config::RFP_EVAL_MODE, state_history, zobrist, num_iterations);
            if (Config::ENABLE_CORR_HIST)
                corrhist_update(current_state, 1, se, highest_score);
            if (Config::ENABLE_CORRHIST_LOG)
                corrhist_log(generatePawnKey(current_state.pawns, current_state.occupied_colour[true], current_state.occupied_colour[false]),
                             1, se, highest_score, depth_limit - cur_depth);
        }
    }

    store_node_tt(zobrist, current_state, state_history, highest_score, cur_depth, depth_limit, alpha_orig, beta_orig, best_move);
    return highest_score;
}

SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint &t0, uint64_t zobrist, const SearchData &previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, int &num_iterations)
{

    increment_node_count_with_decay(num_iterations);

    BoardState current_state = state_history.back();

    SearchData returnData;
    SearchData current_search_data;

    int score = -99999999;
    int highest_score = -99999999;
    // Pre-search depth. Reducing it shrinks the shallow root pass, but it must stay >= 2 so pre_minimizer
    // recurses past its leaf branch and actually generates the second-level move lists the real search consumes
    // (a depth <= 1 pass returns an EMPTY list -> sentinel leak). Low ID iterations (depth_limit <= 2) keep the
    // original depth_limit-1; reduction=1 is byte-identical everywhere.
    int depth = depth_limit - 1;
    if (depth >= 2)
        depth = std::max(2, depth_limit - Config::ROOT_PRESEARCH_REDUCTION);
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

    // Hard-off path: skip the pre_minimizer tree and reuse the previous iteration's REAL second-level data as
    // the ordering hint. Every root move needs a RootScore with a NON-EMPTY legal second_moves list, and the
    // scores vector must stay full-length (alpha_beta indexes second_moves per root move) — so heuristic-fill
    // any move the previous iteration razored away (its scores vector is shorter than moves_list), and all
    // moves on the first iteration. second_scores may be empty (ascending_sort only needs moves >= scores).
    const bool presearch_active = Config::ENABLE_ROOT_PRESEARCH && (Config::PRESEARCH_OFF_FROM_DEPTH <= 0 || depth_limit < Config::PRESEARCH_OFF_FROM_DEPTH);
    if (!presearch_active)
    {
        SearchData rd;
        rd.moves_list = moves_list;
        size_t reuse = previous_search_data.moves_list.empty()
                           ? 0
                           : std::min(previous_search_data.scores.size(), moves_list.size());
        rd.scores.reserve(moves_list.size());
        int off_floor = 0;
        bool have_off_floor = false;
        for (size_t i = 0; i < reuse; ++i)
        {
            rd.scores.push_back(previous_search_data.scores[i]);
            int s = previous_search_data.scores[i].top_score;
            if (!have_off_floor || s < off_floor)
            {
                off_floor = s;
                have_off_floor = true;
            }
        }
        for (size_t i = reuse; i < moves_list.size(); ++i)
        {
            Move &move = moves_list[i];
            bool ep = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);
            bool cap = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], ep);
            updateZobristHashForMove(zobrist, move.from_square, move.to_square, cap,
                                     current_state.pawns, current_state.knights, current_state.bishops,
                                     current_state.rooks, current_state.queens, current_state.kings,
                                     current_state.occupied_colour[true], current_state.occupied_colour[false],
                                     move.promotion);
            make_move(state_history, position_count, move, zobrist, cap);
            RootScore rs;
            // Rank an unscored move at the WORST real score we hold, never at 0. Zero means "equal
            // material", so in any position we are behind it sorts every unknown move ABOVE every known one
            // -- we would try the moves we know least about first, exactly when losing. It also makes
            // alpha - top_score enormous, which is what razors the whole root loop away (the historical
            // "57/300 collapse" attributed to this path). Falls back to 0 only when nothing is known yet.
            rs.top_score = have_off_floor ? off_floor : 0;
            rs.second_moves = buildMoveListFromReordered(state_history, zobrist, 1, move); // copies out of g_moveBuf[1]
            rd.scores.push_back(std::move(rs));
            unmake_move(state_history, position_count, zobrist);
            zobrist = cur_hash;
        }
        // Everything from `reuse` on carries a synthetic score. This path does NOT sort, so the boundary
        // survives to alpha_beta unchanged and root razoring can skip exactly those entries.
        rd.synthetic_from = reuse;
        dbg_searchdata("reorder_legal_moves(no-presearch)", rd);
        return rd;
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

    // Boundary between list 1 (root moves the previous iteration left a real score for) and list 2 (the
    // tail). Move 0 is always searched in full: it seeds alpha for every scout below.
    const size_t presearch_prev_len = previous_search_data.scores.size();
    // Worst real score the previous iteration retained -- the floor an unknown tail move is ranked at under
    // PRESEARCH_TAIL_MODE 2. Filling with 0 instead is the historical "57/300 collapse": it makes
    // alpha - top_score enormous and razors the whole root loop away.
    int trusted_floor = 0;
    bool have_trusted_floor = false;
    for (size_t k = 0; k < presearch_prev_len; ++k)
    {
        int s = previous_search_data.scores[k].top_score;
        if (!have_trusted_floor || s < trusted_floor)
        {
            trusted_floor = s;
            have_trusted_floor = true;
        }
    }
    // Tail handling only applies once we actually have previous data; the first iteration (depth_limit 3,
    // pre-search depth 2) has none and stays on the full pass, which costs almost nothing.
    const bool tail_active = have_trusted_floor && Config::PRESEARCH_TAIL_MODE > 0;
    const size_t tail_start = presearch_prev_len + static_cast<size_t>(std::max(0, Config::PRESEARCH_CHUNK));

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

        // Prefix: the previous iteration already scored this move at full depth, and the merge below throws
        // the pre-search's version away. Reuse the real entry instead of re-deriving it -- and seed alpha
        // from it, so the tail scouts below keep the tight window the skipped pass would have produced.
        if (Config::ENABLE_PRESEARCH_SUBSET && i < presearch_prev_len)
        {
            ++g_prefix_skipped;
            current_search_data.scores.push_back(previous_search_data.scores[i]);
            highest_score = std::max(highest_score, previous_search_data.scores[i].top_score);
            // Stored root scores are mostly PVS bounds, not values, so raising alpha to one directly can
            // mis-set the window for every scout below. The margin backs off from it; a very large margin
            // disables seeding from skipped moves entirely.
            alpha = std::max(alpha, highest_score - Config::PRESEARCH_SUBSET_ALPHA_MARGIN);
            continue;
        }

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

        // Tail moves (no previous-iteration score, past the chunk the root loop actually reaches) can be
        // served three ways. Mode 2 skips the search entirely and takes move_gen's own ordering, which is
        // what produces the main search's 86% first-move-cutoff rate at interior nodes.
        if (tail_active && i >= tail_start)
        {
            if (Config::PRESEARCH_TAIL_MODE == 2)
            {
                ++g_tail_heuristic;
                preliminary_moves = buildMoveListFromReordered(state_history, zobrist, 1, move);
                // A CONSTANT, deliberately. Stepping the score down by index preserves move_gen's ordering
                // through the (unstable) sort, and that was MEASURED WORSE: -12 WAC solves and -44 STS at
                // CHUNK 4. move_gen ranks by killer/history/countermove signal tuned for interior nodes; one
                // ply from a fresh root it is apparently worse than no signal, so letting the tied entries
                // fall arbitrarily beats imposing that order.
                score = trusted_floor;
            }
            else if (Config::PRESEARCH_TAIL_MODE == 3)
            {
                // Same search as mode 0 -- identical warming and ply-1 list -- but the root score is
                // replaced. Any node delta versus base is therefore attributable to the SCORE alone.
                ++g_tail_heuristic;
                score = pre_minimizer(1, depth, alpha, alpha + 1, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
                if (alpha < score && score < beta)
                {
                    preliminary_scores.clear();
                    preliminary_moves.clear();
                    score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
                }
                score = trusted_floor;
            }
            else
            {
                ++g_tail_reduced;
                int tail_depth = std::max(2, depth - Config::PRESEARCH_TAIL_REDUCTION);
                score = pre_minimizer(1, tail_depth, alpha, alpha + 1, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
                if (alpha < score && score < beta)
                {
                    preliminary_scores.clear();
                    preliminary_moves.clear();
                    score = pre_minimizer(1, tail_depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
                }
            }
        }
        else
        {
            ++g_tail_full;
            score = pre_minimizer(1, depth, alpha, alpha + 1, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
            //  If the score is within the window, re-search with full window
            if (alpha < score && score < beta)
            {

                preliminary_scores.clear();
                preliminary_moves.clear();
                score = pre_minimizer(1, depth, alpha, beta, t0, preliminary_scores, preliminary_moves, state_history, position_count, zobrist, move, num_iterations);
            }
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

    // The preliminary pass has no incoming capture context; start its capture-chain density fresh.
    g_captureChain[cur_depth] = 0;

    if (is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD) || current_state.halfmove_clock >= 100)
    {
        if (cur_depth == 1)
        {
            ++g_draw_empty_ply1_pre;
            // Same contract as minimizer's draw return: reorder_legal_moves stores this list in a RootScore
            // that alpha_beta later indexes unguarded, so it must not be left empty.
            if (pre_moves_list.empty())
                pre_moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth, prevMove);
            is_draw = true;
        }
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
                            if (Config::ENABLE_THREAT_HIST) threat_hist_update(current_state.turn, node_threats_of(current_state), move.from_square, move.to_square, cur_depth * cur_depth, false);
                            counterMoves[prevMove.from_square][prevMove.to_square] = move;
                            counterMoveHeuristics[current_state.turn][cont_ctx_key(prevMove, current_state)][cont_ent_key(move, current_state)] += cur_depth * cur_depth * cur_depth;
                            if (Config::ENABLE_PIECE_CONTHIST) pieceContHist[current_state.turn][pcont_ctx_key(prevMove, current_state)][pcont_ent_key(move, current_state)] += cur_depth * cur_depth * cur_depth;
                            if (Config::ENABLE_CONT_HIST_2PLY && cur_depth >= 2)
                            {
                                Move p2 = g_searchStack[cur_depth - 2];
                                if (p2.from_square != p2.to_square)
                                    contHist2[current_state.turn][cont_ctx_key(p2, current_state)][cont_ent_key(move, current_state)] += cur_depth * cur_depth * cur_depth;
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
                // Keep reductions off inside an OTV verification window (inert while g_verify_no_reduce_until < 0).
                do_lmr = do_lmr && !(g_verify_no_reduce_until >= 0 && cur_depth <= g_verify_no_reduce_until);

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
                if (Config::ENABLE_THREAT_HIST)
                    threat_hist_update(current_state.turn, node_threats_of(current_state), move.from_square, move.to_square, (depth_limit - cur_depth) * (depth_limit - cur_depth), false);
                counterMoves[prevMove.from_square][prevMove.to_square] = move;
                counterMoveHeuristics[current_state.turn][cont_ctx_key(prevMove, current_state)][cont_ent_key(move, current_state)] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
                if (Config::ENABLE_PIECE_CONTHIST)
                    pieceContHist[current_state.turn][pcont_ctx_key(prevMove, current_state)][pcont_ent_key(move, current_state)] += 4 * (depth_limit - cur_depth) * (depth_limit - cur_depth);
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
    else if (Config::NODE_LIMIT > 0 && num_iterations >= Config::NODE_LIMIT)
    {
        time_up.store(true, std::memory_order_relaxed);
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

    if (qDepth >= Config::MAX_QDEPTH)
    {
        int horizon_eval = eval_by_mode(Config::QSTANDPAT_EVAL_MODE, state_history, zobrist, num_iterations);
        if (Config::ENABLE_CORR_HIST && Config::ENABLE_CORRHIST_QSEARCH)
            horizon_eval += corrhist_correction(state_history.back(), is_maximizing ? 1 : 0);
        return horizon_eval;
    }

    BoardState current_state = state_history.back();
    increment_node_count_with_decay(num_iterations);
    int cache_result;
    if (!Config::DISABLE_QCACHE && probeQCache(zobrist, current_state.castling_rights, current_state.ep_square, alpha, beta, cache_result))
    {
        return cache_result;
    }

    int moveNum = static_cast<int>(state_history.size());
    uint64_t cur_hash = zobrist;

    bool currently_in_check = is_check(current_state.turn, current_state.occupied, current_state.queens | current_state.rooks, current_state.queens | current_state.bishops, current_state.kings, current_state.knights, current_state.pawns, current_state.occupied_colour[!current_state.turn]);

    if (currently_in_check)
    {
        std::vector<Move> &moves_list = buildMoveListFromReordered(state_history, zobrist, cur_depth + qDepth, prevMove);

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

    int static_eval = eval_by_mode(Config::QSTANDPAT_EVAL_MODE, state_history, zobrist, num_iterations);
    if (Config::ENABLE_CORR_HIST && Config::ENABLE_CORRHIST_QSEARCH)
        static_eval += corrhist_correction(current_state, is_maximizing ? 1 : 0);
    if (is_maximizing)
    {
        if (static_eval >= beta)
            return static_eval; // Fail-hard beta cutoff
        if (static_eval > alpha)
            alpha = static_eval;
        if (Config::ENABLE_QDELTA && !Config::ENABLE_QDELTA_PERMOVE && static_eval < alpha - Config::DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    }
    else
    {
        if (static_eval <= alpha)
            return static_eval; // Fail-hard alpha cutoff
        if (static_eval < beta)
            beta = static_eval;
        if (Config::ENABLE_QDELTA && !Config::ENABLE_QDELTA_PERMOVE && static_eval > beta + Config::DELTA_MARGIN)
            return static_eval; // Optional delta pruning
    }

    int best = is_maximizing ? -9999999 + moveNum : 9999999 - moveNum;

    std::vector<Move> &moves_list = buildNoisyMoveList(zobrist, state_history, cur_depth + qDepth, qDepth, prevMove);

    for (size_t i = 0; i < moves_list.size(); ++i)
    {
        Move &move = moves_list[i];

        bool en_passant_move = is_en_passant(move.from_square, move.to_square, current_state.ep_square, current_state.occupied, current_state.pawns);

        // Acquire the zobrist hash for the new position if the given move was made
        bool capture_move = is_capture(move.from_square, move.to_square, current_state.occupied_colour[!current_state.turn], en_passant_move);

        // Per-capture futility: skip a capture only when even WINNING THE VICTIM cannot reach the bound, so
        // the margin sits ON TOP of the captured value instead of replacing it. Placed before the zobrist
        // update so a skip leaves the hash untouched. Promotions and en-passant are exempt (their swing is not
        // the value standing on the destination square); in-check nodes never reach here (handled above).
        // move.promotion == 1 means "not a promotion" (move_gen.h pushes 1 for every non-promotion and
        // 2..5 for the real thing) -- testing against 0 here made this block unreachable.
        if (Config::ENABLE_QDELTA_PERMOVE && Config::ENABLE_QDELTA && capture_move && !en_passant_move && move.promotion <= 1)
        {
            int victim = get_value_at(move.to_square, current_state);
            int pm_margin = Config::QDELTA_PERMOVE_MARGIN > 0 ? Config::QDELTA_PERMOVE_MARGIN : Config::DELTA_MARGIN;
            ++g_qdelta_permove_seen;
            if (is_maximizing)
            {
                if (static_eval + pm_margin + victim <= alpha)
                {
                    ++g_qdelta_permove_fires;
                    continue;
                }
            }
            else
            {
                if (static_eval - pm_margin - victim >= beta)
                {
                    ++g_qdelta_permove_fires;
                    continue;
                }
            }
        }

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
            {
                ++g_q_fh_total;
                if (i == 0)
                    ++g_q_fh_first;
                g_q_cut_idx_sum += (long)i;
                return best; // beta cutoff
            }
        }
        else
        {
            if (score < best)
                best = score;
            if (best < beta)
                beta = best;
            if (best <= alpha)
            {
                ++g_q_fh_total;
                if (i == 0)
                    ++g_q_fh_first;
                g_q_cut_idx_sum += (long)i;
                return best; // alpha cutoff
            }
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

    // Sort indices based on scores (descending). Under the persistent table this becomes SF's two-level
    // key: this iteration's score first, the previous iteration's as the tiebreak. Level 1 is entirely
    // this-iteration and level 2 entirely previous-iteration, so depths are never compared against each
    // other and a shallow score cannot outrank a deep one. stable_sort so equal keys -- in particular the
    // block of unproven sentinels -- keep the order the previous iteration left them in.
    if (Config::ENABLE_ROOT_TABLE)
    {
        // Level-2 choice matters more than it looks. SF's `previousScore` is last iteration's top_score,
        // which for a chronic fail-low move is ITSELF the sentinel -- so the whole fail-low block ties on
        // both levels and stable_sort merely freezes the previous order, discarding the fine-grained
        // ordering those moves used to get from their real fail-soft scores. Keying level 2 on last_real
        // instead orders the block by the most recent MEASURED value, which still never compares against a
        // this-iteration score because level 1 has already separated the two groups.
        const bool l2_last_real = Config::ROOT_SORT_L2_LASTREAL;
        const bool l1_last_real = Config::ROOT_SORT_L1_LASTREAL;
        std::stable_sort(indices.begin(), indices.end(),
                         [&](size_t a, size_t b)
                         {
                             // Level 1 by most recent measured score: keeps the table's bookkeeping but
                             // restores value ordering among the fail-low majority, which the sentinel key
                             // flattens into a single block.
                             if (l1_last_real)
                             {
                                 if (data.scores[a].last_real != data.scores[b].last_real)
                                     return data.scores[a].last_real > data.scores[b].last_real;
                                 return data.scores[a].top_score > data.scores[b].top_score;
                             }
                             if (data.scores[a].top_score != data.scores[b].top_score)
                                 return data.scores[a].top_score > data.scores[b].top_score;
                             if (l2_last_real)
                                 return data.scores[a].last_real > data.scores[b].last_real;
                             return data.scores[a].prev_score > data.scores[b].prev_score;
                         });
    }
    else
    {
        std::sort(indices.begin(), indices.end(),
                  [&](size_t a, size_t b)
                  {
                      return data.scores[a].top_score > data.scores[b].top_score;
                  });
    }

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

/*
    Sorts [lo, hi) of a SearchData descending by top_score, keeping moves and grouped scores parallel.
    Same index-permutation form as sortSearchDataByScore, restricted to a sub-range so the root tail's
    two provenance groups can be ordered without ever being compared against each other.
*/
inline void sortSearchDataRange(SearchData &data, size_t lo, size_t hi)
{
    size_t n = std::min(data.moves_list.size(), data.scores.size());
    if (hi > n)
        hi = n;
    if (lo >= hi)
        return;

    std::vector<size_t> indices(hi - lo);
    std::iota(indices.begin(), indices.end(), lo);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b)
              {
                  return data.scores[a].top_score > data.scores[b].top_score;
              });

    std::vector<Move> moves_tmp(indices.size());
    std::vector<RootScore> scores_tmp(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        moves_tmp[i] = std::move(data.moves_list[indices[i]]);
        scores_tmp[i] = std::move(data.scores[indices[i]]);
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        data.moves_list[lo + i] = std::move(moves_tmp[i]);
        data.scores[lo + i] = std::move(scores_tmp[i]);
    }
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

    // Sort the tail descending by top_score. Split mode orders the two provenance groups separately and
    // concatenates trusted-first: entries carrying the previous iteration's real searched scores occupy
    // sub_data[0, real_len), pre-pass entries the remainder, and the two are never compared. Note this
    // also stops a previously-searched move that failed low (upper bound near alpha, i.e. known bad) from
    // sinking below an unscored move, which the single sort does today.
    if (Config::ENABLE_ROOT_SORT_SPLIT)
    {
        size_t real_len = mainSearchData.scores.empty() ? 0 : mainSearchData.scores.size() - 1;
        if (real_len > sub_data.moves_list.size())
            real_len = sub_data.moves_list.size();
        sortSearchDataRange(sub_data, 0, real_len);
        sortSearchDataRange(sub_data, real_len, sub_data.moves_list.size());
    }
    else
    {
        sortSearchDataByScore(sub_data);
    }

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
        if (Config::ENABLE_PIECE_CONTHIST)
            decayPieceContHist();
        if (Config::ENABLE_THREAT_HIST)
            decayThreatHist();
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

    // Re-derive the base from the node's own remaining depth (see ENABLE_LMR_REMDEPTH). The same tuned
    // curve supplies the reduction in plies, scaled, and is never allowed to consume the child's last ply.
    if (Config::ENABLE_LMR_REMDEPTH)
    {
        int rem = std::clamp(depth_limit - cur_depth, 0, 63);
        int red = (rem - Config::ACTIVE->DEPTH_REDUCTION[rem]) * Config::LMR_REMDEPTH_SCALE / 100;
        red = std::clamp(red, 0, std::max(rem - 2, 0));
        base = depth_limit - red;
    }

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
    int r = static_cast<int>(base - (move_factor / scale)) - Config::LMR_EXTRA;

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

/*
    Promotes a transposition-table cutoff move within the QUIET region of an already-ordered move list.

    The list arrives captures-first (SEE/MVV-ordered) with quiets after. A TT move that is itself a capture
    is deliberately left where it is: hoisting it ahead of better-scoring captures is the shape the
    2026-07-03 ordering audit identified as the reason the original TT-move experiment failed. A quiet TT
    move moves to the front of the quiet region -- ahead of the killer/counter promotions, behind every
    capture.

    Parameters:
        moves         - ordered move list, modified in place
        move          - the TT cutoff move (callers skip the default {0,0,0} "no move")
        current_state - the board state the list was generated from, for capture detection
    Returns: true when the move was found and repositioned.
*/
inline bool promoteMoveWithinQuiets(std::vector<Move> &moves, const Move &move, const BoardState &current_state)
{
    auto it = std::find(moves.begin(), moves.end(), move);
    if (it == moves.end())
        return false;

    size_t quiet_start = moves.size();
    for (size_t i = 0; i < moves.size(); ++i)
    {
        if (!is_capture(moves[i].from_square, moves[i].to_square, current_state.occupied_colour[!current_state.turn], is_en_passant(moves[i].from_square, moves[i].to_square, current_state.ep_square, current_state.occupied, current_state.pawns)))
        {
            quiet_start = i;
            break;
        }
    }

    // Below quiet_start the move is a capture (leave it); at quiet_start it already leads the quiets.
    size_t foundIndex = std::distance(moves.begin(), it);
    if (foundIndex <= quiet_start)
        return false;

    Move temp = *it;
    moves.erase(it);
    moves.insert(moves.begin() + quiet_start, temp);
    return true;
}

inline std::vector<Move> &buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove)
{

    move_gen_visits++;
    BoardState current_state = state_history.back();
    // uint64_t zobrist2 = zobrist;
    // zobrist = generateZobristHash(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.turn);

    // if (zobrist != zobrist2)
    // std::cout << create_fen(current_state.pawns, current_state.knights, current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[false], current_state.promoted, current_state.castling_rights, current_state.ep_square, current_state.turn) << std::endl;

    // Per-ply working buffer: this node's move list lives here for its whole lifetime; deeper nodes
    // use deeper buffers, so it is never overwritten while we iterate it. fillMoveGenCache takes a
    // synchronous snapshot of the cache into it (no surviving reference into the cache slot). A ply
    // beyond the pool (cannot happen at real depth) falls back to a shared buffer.
    std::vector<Move> &cached_moves =
        (cur_ply >= 0 && cur_ply < MOVE_POOL_PLIES) ? g_moveBuf[cur_ply] : g_moveBufFallback;
    fillMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square, cached_moves);
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
                    uint64_t node_threats = 0;
                    if (Config::ENABLE_THREAT_HIST)
                    {
                        uint64_t opp = current_state.occupied_colour[!current_state.turn];
                        node_threats = opponent_threats(!current_state.turn, current_state.pawns & opp, current_state.knights & opp,
                                                        current_state.bishops & opp, current_state.rooks & opp, current_state.queens & opp,
                                                        current_state.kings & opp, current_state.occupied);
                    }
                    size_t qn = cached_moves.size() - quiet_start;
                    std::vector<int> qs(qn);
                    for (size_t t = 0; t < qn; ++t)
                    {
                        const Move &m = cached_moves[quiet_start + t];
                        qs[t] = score_quiet(m.from_square, m.to_square, m.promotion, current_state.turn, cur_ply, prevMove,
                                            current_state.occupied, current_state.pawns, current_state.knights,
                                            current_state.bishops, current_state.rooks, current_state.queens, enemy_king_sq, node_threats);
                    }
                    if (do_full)
                    {
                        std::vector<size_t> ord(qn);
                        std::iota(ord.begin(), ord.end(), 0);
                        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b)
                                         { return qs[a] > qs[b]; });
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
        if (Config::ENABLE_TT_MOVE)
        {
            // Reads TTEntry::move, where the cutoff move has actually lived since the 07-08 store rework.
            // The former source, g_ttMoveTable, has no write site anywhere in the engine -- every lookup
            // returned the default {0,0,0}, which is why enabling this flag was byte-identical.
            TTEntry *tte = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
            if (tte != nullptr && tte->move.from_square != tte->move.to_square)
            {
                if (Config::TT_MOVE_POLICY == Config::TT_MOVE_POLICY_FRONT)
                {
                    promoteMoveToFront(cached_moves, tte->move);
                    ++g_tt_move_promotions;
                }
                else if (promoteMoveWithinQuiets(cached_moves, tte->move, current_state))
                    ++g_tt_move_promotions;
            }
        }
        return cached_moves;
    }

    // Cache miss: fillMoveGenCache cleared cached_moves; generate into the per-ply buffer (no alloc).
    cached_moves.reserve(64);

    generateLegalMovesReordered(cached_moves, current_state.castling_rights, ~0ULL, ~0ULL,
                                current_state.occupied, current_state.occupied_colour[true], current_state.occupied_colour[!current_state.turn], current_state.occupied_colour[current_state.turn], current_state.pawns, current_state.knights,
                                current_state.bishops, current_state.rooks, current_state.queens, current_state.kings, current_state.ep_square, current_state.turn, cur_ply, prevMove);

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

    addToMoveGenCache(zobrist, /* max_cache_size * Config::ACTIVE->cache_size_multiplier ,*/ cached_moves, current_state.castling_rights, current_state.ep_square);
    if (Config::ENABLE_TT_MOVE)
    {
        TTEntry *tte = accessSearchEvalCache(zobrist, current_state.castling_rights, current_state.ep_square);
        if (tte != nullptr && tte->move.from_square != tte->move.to_square)
        {
            if (Config::TT_MOVE_POLICY == Config::TT_MOVE_POLICY_FRONT)
            {
                promoteMoveToFront(cached_moves, tte->move);
                ++g_tt_move_promotions;
            }
            else if (promoteMoveWithinQuiets(cached_moves, tte->move, current_state))
                ++g_tt_move_promotions;
        }
    }
    return cached_moves;
}

/*
    Decides whether a quiet checking move is worth searching in qsearch.

    Qsearch already admits captures only when see() >= 0; quiet checks are the one noisy category taken
    unconditionally, and most of the junk ones simply hang the checking piece -- the opponent captures and the
    line dies after we have paid to search it. see() cannot be reused here: it is keyed on the SQUARE and
    starts from get_value_at(to_square), which is zero for a quiet destination, so it does not know which piece
    is being moved. This applies the same intent with bitboard tests and no board copy, keeping the
    ENABLE_QCHECK_MASK path's advantage over the simulate path.

    A check is rejected when the destination is attacked by an enemy pawn (any non-pawn mover loses material
    outright), or when it is attacked at all and we do not defend it.

    Parameters:
        current_state - board state the move is generated from
        m             - the quiet checking move under consideration
    Returns: true when the check looks materially survivable and should be searched.
*/
inline bool quietCheckIsSafe(const BoardState &current_state, const Move &m)
{
    uint64_t from_bb = BB_SQUARES[m.from_square];
    uint8_t to = m.to_square;
    // Vacate the origin so a slider behind the mover is seen correctly through the square it leaves.
    uint64_t occ = current_state.occupied & ~from_bb;
    uint64_t enemy = current_state.occupied_colour[!current_state.turn] & occ;
    uint64_t own = current_state.occupied_colour[current_state.turn] & occ;

    // A pawn of the side to move standing on `to` would attack the same squares an enemy pawn attacks it
    // from, so the reverse mask locates the enemy pawns bearing on the destination.
    uint64_t enemy_pawn_attackers = BB_PAWN_ATTACKS[current_state.turn][to] & current_state.pawns & enemy;
    if (enemy_pawn_attackers != 0 && (current_state.pawns & from_bb) == 0)
        return false;

    // Level 1 stops here. The attacked-and-undefended rule below rejects checks the opponent can simply
    // take, but a check that hangs its piece is frequently a sacrifice and therefore the whole point of the
    // line -- at level 2 it costs more in solved tactics than it saves in nodes.
    if (Config::QCHECK_SAFE_LEVEL < Config::QCHECK_SAFE_UNDEFENDED)
        return true;

    uint64_t queens_and_rooks = (current_state.queens | current_state.rooks) & occ;
    uint64_t queens_and_bishops = (current_state.queens | current_state.bishops) & occ;
    uint64_t enemy_attackers = attackersMask(!current_state.turn, to, occ, queens_and_rooks, queens_and_bishops,
                                             current_state.kings & occ, current_state.knights & occ,
                                             current_state.pawns & occ, enemy);
    if (enemy_attackers == 0)
        return true;

    uint64_t own_defenders = attackersMask(current_state.turn, to, occ, queens_and_rooks, queens_and_bishops,
                                           current_state.kings & occ, current_state.knights & occ,
                                           current_state.pawns & occ, own);
    return own_defenders != 0;
}

/*
    Reports whether a quiet move gives check, including DISCOVERED checks, without copying the board.

    ENABLE_QCHECK_MASK tests only the moving piece's attacks from its destination, so it is structurally
    blind to a discovery -- moving a piece off a ray and letting a slider behind it give check. The simulate
    path would catch those, but only once its own defect is fixed (update_state takes `turn` BY VALUE, so the
    caller's copy never flips and is_check ends up asking whether the MOVER is in check), and it pays a full
    board update per quiet move.

    This rebuilds only the moving piece's type mask with the bit relocated from->to and asks who attacks the
    enemy king under that occupancy, which answers both cases in one query at close to mask cost.

    ⚠️ Castling is NOT covered: a quiet king move that castles delivers check with the ROOK, and only the
    king's from/to are modelled here. The mask path shares this gap; it is a known exclusion, not a silent
    one, and it is rare enough not to justify the extra branch in the hot loop.

    Parameters:
        current_state - board state the move is generated from
        m             - the quiet move under consideration
    Returns: true when the move leaves the enemy king attacked.
*/
inline bool moveGivesCheckFast(const BoardState &current_state, const Move &m)
{
    uint64_t from_bb = BB_SQUARES[m.from_square];
    uint64_t to_bb = BB_SQUARES[m.to_square];
    uint64_t moved = from_bb | to_bb;
    uint64_t occ = (current_state.occupied & ~from_bb) | to_bb;

    uint8_t enemy_king = __builtin_ctzll(current_state.kings & current_state.occupied_colour[!current_state.turn]);

    // Relocate the mover inside whichever type mask holds it; the others pass through untouched.
    uint64_t pawns = current_state.pawns;
    uint64_t knights = current_state.knights;
    uint64_t bishops = current_state.bishops;
    uint64_t rooks = current_state.rooks;
    uint64_t queens = current_state.queens;
    uint64_t kings = current_state.kings;
    if (pawns & from_bb)
        pawns ^= moved;
    else if (knights & from_bb)
        knights ^= moved;
    else if (bishops & from_bb)
        bishops ^= moved;
    else if (rooks & from_bb)
        rooks ^= moved;
    else if (queens & from_bb)
        queens ^= moved;
    else
        kings ^= moved;

    uint64_t movers = (current_state.occupied_colour[current_state.turn] & ~from_bb) | to_bb;

    return attackersMask(current_state.turn, enemy_king, occ, (queens | rooks) & occ, (queens | bishops) & occ,
                         kings & occ, knights & occ, pawns & occ, movers & occ) != 0;
}

inline std::vector<Move> &buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, int qDepth, Move prevMove)
{

    // Per-ply buffers (see g_moveBuf/g_noisyBuf): the returned noisy list lives in g_noisyBuf[ply]
    // for the caller's qsearch loop; the full list is snapshotted into g_moveBuf[ply] transiently to
    // filter from. Deeper qsearch nodes use deeper buffers, so neither is overwritten mid-iteration.
    bool in_pool = (cur_ply >= 0 && cur_ply < MOVE_POOL_PLIES);
    std::vector<Move> &noisy_moves = in_pool ? g_noisyBuf[cur_ply] : g_noisyBufFallback;
    noisy_moves.clear();

    // Parallel sort keys for the optional SEE re-sort (ENABLE_QSEE_RESORT). Only populated when enabled.
    std::vector<int> noisy_scores;
    if (Config::ENABLE_QSEE_RESORT)
        noisy_scores.reserve(16);

    BoardState current_state = state_history.back();

    std::vector<Move> &moves_list = in_pool ? g_moveBuf[cur_ply] : g_moveBufFallback;
    fillMoveGenCache(zobrist, current_state.castling_rights, current_state.ep_square, moves_list);
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
            if (Config::ENABLE_QSEE_RESORT)
                noisy_scores.push_back(900000);
        }
        else if (capture_move)
        {
            if (en_passant_move)
            {
                noisy_moves.push_back(moves_list[i]);
                if (Config::ENABLE_QSEE_RESORT)
                    noisy_scores.push_back(500000);
            }
            else
            {
                int sv = see(moves_list[i].to_square, current_state.turn, current_state);
                if (sv >= 0)
                {
                    noisy_moves.push_back(moves_list[i]);
                    if (Config::ENABLE_QSEE_RESORT)
                        noisy_scores.push_back(500000 + sv);
                }
            }
        }
        else if (Config::ENABLE_QCHECK_DEPTH0 && qDepth > 0)
        {
            // Quiet move dropped past the first q-ply. Counted so the guard's reachability is provable
            // rather than inferred from an output fingerprint: a zero here means the branch never runs.
            ++g_qcheck_d0_skipped;
        }
        else
        {
            // Quiet move: include it only if it gives check. ENABLE_QCHECK_DEPTH0 (the guard above)
            // drops quiet checks past the first q-ply. ENABLE_QCHECK_MASK detects a DIRECT check with a
            // bitboard attack test from the destination square (the moving piece's attacks from `to`
            // with `from` vacated, vs the enemy king) -- no board copy; misses discovered checks (the
            // standard accepted tradeoff). Default off = the original simulate-and-test path.
            bool move_is_check;
            if (Config::ENABLE_QCHECK_FULL)
            {
                // Direct AND discovered checks, no board copy. Counted against what the mask arm would have
                // found so the capability increment is observable rather than assumed.
                move_is_check = moveGivesCheckFast(current_state, moves_list[i]);
                if (Config::ENABLE_QCHECK_MASK_COMPARE)
                {
                    uint64_t from_bb = BB_SQUARES[moves_list[i].from_square];
                    uint8_t to = moves_list[i].to_square;
                    uint64_t occ_nomove = current_state.occupied & ~from_bb;
                    uint8_t ek = __builtin_ctzll(current_state.kings & current_state.occupied_colour[!current_state.turn]);
                    uint64_t atk = 0;
                    if (current_state.knights & from_bb)
                        atk = BB_KNIGHT_ATTACKS[to];
                    else if (current_state.pawns & from_bb)
                        atk = BB_PAWN_ATTACKS[current_state.turn][to];
                    else if (current_state.bishops & from_bb)
                        atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ_nomove];
                    else if (current_state.rooks & from_bb)
                        atk = BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ_nomove] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ_nomove];
                    else if (current_state.queens & from_bb)
                        atk = BB_DIAG_ATTACKS[to][BB_DIAG_MASKS[to] & occ_nomove] | BB_RANK_ATTACKS[to][BB_RANK_MASKS[to] & occ_nomove] | BB_FILE_ATTACKS[to][BB_FILE_MASKS[to] & occ_nomove];
                    // Both directions. A non-zero missed count means the full detector is WRONG, not merely
                    // different -- the mask arm is a strict subset (direct checks) and must never win.
                    bool mask_says_check = (atk & BB_SQUARES[ek]) != 0;
                    if (move_is_check && !mask_says_check)
                        ++g_q_discovered_checks;
                    else if (!move_is_check && mask_says_check)
                        ++g_q_checks_missed;
                }
            }
            else if (Config::ENABLE_QCHECK_MASK)
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

            if (move_is_check && Config::ENABLE_QCHECK_SAFE && !quietCheckIsSafe(current_state, moves_list[i]))
            {
                ++g_q_quiet_checks_unsafe;
            }
            else if (move_is_check)
            {
                // Quiet checks entering qsearch. Counted on the rare (taken) branch only: if this stays at
                // zero the noisy list is captures-only and ENABLE_QCHECK_DEPTH0 can have nothing to drop.
                ++g_q_quiet_checks_added;
                noisy_moves.push_back(moves_list[i]);
                if (Config::ENABLE_QSEE_RESORT)
                    noisy_scores.push_back(0);
            }
        }
    }

    // Optional SEE re-sort: promotions, then captures by SEE-descending, then quiet checks, for earlier
    // qsearch stand-pat cutoffs. Stable to preserve the main sort's tie-breaks within equal-score groups.
    if (Config::ENABLE_QSEE_RESORT && noisy_moves.size() > 1)
    {
        std::vector<size_t> idx(noisy_moves.size());
        for (size_t k = 0; k < idx.size(); ++k)
            idx[k] = k;
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b)
                         { return noisy_scores[a] > noisy_scores[b]; });
        std::vector<Move> sorted;
        sorted.reserve(noisy_moves.size());
        for (size_t k : idx)
            sorted.push_back(noisy_moves[k]);
        noisy_moves.swap(sorted);
    }

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

    // A fabricated abort value or a path-dependent draw must not be cached as an evaluation of the
    // position (see QCACHE_SOUND_STORE). qSearch returns a bare 0 on every path this guards, so a
    // non-zero result cannot be one of them -- testing that first keeps is_repetition, a hash lookup,
    // off the per-qsearch-entry hot path.
    bool unsound_store = Config::QCACHE_SOUND_STORE && (time_up.load(std::memory_order_relaxed) || (result == 0 && is_repetition(position_count, zobrist, Config::REPETITION_THRESHOLD)));

    if (!Config::DISABLE_QCACHE && !unsound_store)
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
    if (!g_eval_light && accessCacheNew(zobrist, cache_result))
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
    if (!g_eval_light)
        addToCacheNew(zobrist, total);
    return total;
}
