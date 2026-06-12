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

constexpr int TIME_CHECK_INTERVAL = 200000;

// Constants for material thresholds
constexpr int MIN_MATERIAL_FOR_NULL_MOVE = 15000;

// Number of nodes before decay
constexpr int DECAY_FACTOR = 1; // Divide scores by 2
constexpr int NO_STATIC_EVAL = -1000000000; // g_evalStack sentinel: no valid static eval (in-check / outside the improving window)

// History-gravity bounds (MAX_HISTORY, CONT2_GRAVITY_DIV) are env-tunable Config members below.

constexpr std::array<int, 4> FUTILITY_MARGINS = {200, 450, 650, 950};

constexpr int MAX_QDEPTH = 10;
constexpr int SUPPORT_MARGIN = 0;
constexpr int DELTA_MARGIN = 1500;

constexpr bool USE_Q_SEARCH = true;

struct TTEntry;
struct QCacheEntry;
struct MoveEntry;

struct ConfigData
{
    int cache_size_multiplier;
    double TIME_LIMIT;
    std::array<double, 64> MOVE_TIMES;
    std::array<int, 64> DEPTH_REDUCTION;
};

namespace Configs
{

    constexpr ConfigData LIGHTNING = {
        2,
        1.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 0.5;
            times[4] = 0.5;
            times[5] = 0.5;
            times[6] = 0.75;
            times[7] = 0.75;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 0.75;
            }
            return times;
        }(),

        []
        {
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
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData BLITZ = {
        2,
        3.5,
        []
        {
            std::array<double, 64> times{};
            times[3] = 1.0;
            times[4] = 1.0;
            times[5] = 1.5;
            times[6] = 1.5;
            times[7] = 1.5;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 2.0;
            }
            return times;
        }(),

        []
        {
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
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData STANDARD = {
        2,
        45.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            times[6] = 5.5;
            times[7] = 6.5;
            for (int i = 8; i < 64; ++i)
            {
                times[i] = 6.0;
            }
            return times;
        }(),

        []
        {
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
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};

    constexpr ConfigData LONG_FORMAT = {
        3,
        600.0,
        []
        {
            std::array<double, 64> times{};
            times[3] = 5.0;
            times[4] = 5.0;
            times[5] = 5.5;
            for (int i = 6; i < 64; ++i)
            {
                times[i] = 120.0;
            }
            return times;
        }(),
        []
        {
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
            for (int i = 14; i < 64; ++i)
            {
                new_depths[i] = 12;
            }
            return new_depths;
        }()};
}

namespace Config
{
    inline const ConfigData *ACTIVE = &Configs::STANDARD; // Default to classical
    inline bool side_to_play = false;                     // Default; can be set at runtime
    inline int DECAY_INTERVAL = 35000;

    // Search-ablation toggles (default ON = current behavior). Each disables a
    // pruning/reduction mechanism at all of its live sites so the eval-vs-search
    // diagnostic can localize which one drops winning lines. Overridable at runtime
    // from the environment (see env_flag in initialize_engine); since true && X == X
    // the default build is byte-identical.
    inline bool ENABLE_LMR = true;      // late move reductions
    inline bool ENABLE_FUTILITY = true; // futility pruning (inside the LMR block)
    inline bool ENABLE_RAZORING = true; // razoring (alpha_beta root loop)
    inline bool ENABLE_NULLMOVE = true; // null-move pruning
    inline bool NULLMOVE_PROGRESSIVE = false; // depth-scaled null-move reduction (-2 at d>=12, -3 at d>=14); off = flat -1
    inline bool ENABLE_QDELTA = true;   // delta pruning in quiescence

    inline bool LMR_PROFILE = false; // env-gated LMR-miss profiler (diagnostic; off = byte-identical)

    // Sound-LMR exemptions (default OFF = current behavior). Stop reducing the
    // moves most likely to be the critical misses; env-gated so the A/B needs no
    // recompile and the default build stays byte-identical.
    inline bool PROTECT_KILLERS = false; // don't LMR-reduce killer / counter moves
    inline bool PROTECT_PV = false;      // don't LMR-reduce at PV nodes (beta - alpha > 1)

    // History-aware LMR ("reduce-less"): search known-good late quiets a little less reduced (toward,
    // never beyond, full depth). Categorical signal — killer/counter membership + a coarse history
    // tier — not an absolute score threshold, so it is robust to the unbounded/uneven history values
    // and to a later continuation-history upgrade. SHIPPED default = the reduce-MORE arm (CAP=0,
    // MORE_CAP=1): STS300 50.1->51.7% and -1.3% nodes @d10; self-play +25.5 +/-65 (positive, not sig).
    inline bool ENABLE_HISTORY_LMR = true;  // master gate for the history-aware LMR adjustment
    inline int HISTORY_LMR_CAP = 0;         // plies to REMOVE for good quiets (reduce-less; 0 = off, the shipped arm)
    inline int HISTORY_LMR_MORE_CAP = 1;    // plies to ADD for never-cut (tier-0) quiets (reduce-more; the shipped lever)

    // Improving heuristic: reduce one extra ply when the side-to-move's static eval is NOT rising vs
    // 2 ply back (stagnant -> prune harder). Modulates LMR only (composes with history_lmr_delta); gated.
    inline bool ENABLE_IMPROVING = false;
    inline int IMPROVING_EVAL_WINDOW = 6;   // populate g_evalStack only within this many plies of the leaf (cost control)

    // Continuation-aware LMR (default on): a tier-0 (never-cut) quiet with a strong 1-ply continuation
    // score (counterMoveHeuristics) is NOT reduced-more -- a known-good reply to the previous move, so
    // we cancel the extra reduction (never deeper than base). THRESH=2000 is the d10 node-efficiency
    // optimum (WAC nodes -6.2% vs off, WAC 262/300, STS neutral); 4000 and 1000 both save less.
    inline bool ENABLE_CONT_HIST = true;
    inline int CONT_HIST_LMR_THRESH = 2000; // min continuation score to cancel the reduce-more

    // Move-ordering experiments, each benched independently.
    inline bool ENABLE_CONT_HIST_2PLY = false; // 2-ply continuation history -- d10 LOSS at equal weight (WAC -3, +7% nodes); needs down-weight (b/4) + the bonus/malus rework before it's worth anything
    inline bool ENABLE_CAPTURE_HIST = false;   // capture-history refinement -- marginal (+0.7 STS/+1 WAC but +2.6% nodes); knob, revisit after bonus/malus
    inline bool ENABLE_CHECK_ORDER = false;    // direct-check bonus -- on the SCALE-OFF baseline it's -6 WAC for -9.8% nodes (accuracy traded for speed; bad at fixed depth). BONUS=6000 too hot -> recalibrate lower before re-enabling
    inline int CHECK_ORDER_BONUS = 6000;       // the flat quiet-check ordering bonus

    // History gravity: replace the bonus-only `+= depth²` cutoff update with a saturating bonus/MALUS --
    // reward the move that cut off, penalize the quiets/captures tried-and-failed before it. Applies to
    // HH + 1-ply counter + cont2 + capture uniformly (bounds: MAX_HISTORY/CONT2_GRAVITY_DIV above).
    // Default off = byte-identical (the existing += depth² path runs untouched).
    // Decomposed into two orthogonal knobs: SATURATION (bounded hist_update vs simple +=) and
    // MALUS (penalize searched-and-failed quiets/captures). gravity == SATURATION && MALUS.
    inline bool ENABLE_HISTORY_SATURATION = false;
    inline bool ENABLE_HISTORY_MALUS = false;
    inline int MAX_HISTORY = 16384;     // gravity saturation bound (env-tunable for the sweep; bake the winner to constexpr for ship)
    inline int CONT2_GRAVITY_DIV = 4;   // 2-ply gravity down-weight divisor
    inline bool ENABLE_HISTORY_DECAY = true; // periodic >>=1 aging of history tables; off = saturation-only bounding (gravity tuning knob)
    inline int MALUS_DIV = 1;           // gravity malus softening: malus = bonus / MALUS_DIV (1 = symmetric, current)

    // Eval: scale the advanced-endgame mate-drive by the winner's material margin (default off =
    // byte-identical). Unproven (no definitive self-play result); the R+N-vs-R case it targeted is now
    // handled by is_practically_drawn. Kept as a knob for the graded endgame-scaling rework.
    inline bool ENABLE_MATE_DRIVE_SCALE = false;

    // Eval: continuous endgame "convertibility" scale (default OFF -- reverted). Damps an unconvertible
    // material/placement lead toward draw (bare minor, opposite-coloured bishops). The 5 FEN spot-checks
    // looked surgical, but a scale-ON STS bench showed it changes far more leaf evals than they implied:
    // -3.9 STS / -3 WAC vs scale-off for only -3% nodes -- a net suite regression. Kept as a knob; redo
    // with tighter targeting (or after the bonus/malus history rework) before re-enabling.
    inline bool ENABLE_ENDGAME_SCALE = false;

    // Eval: replace the per-bishop colour-complex flood-fill (get_bishop_colour_complex_score, profiled
    // at ~33% of the entire midgame eval) with a cheap popcount approximation of the same good/bad-bishop
    // + activity signal: own pawns on the bishop's colour (bad bishop) traded against the bishop's current
    // diagonal scope (mobility / forward reach into the enemy half / enemy-king-zone pressure). Default
    // off = byte-identical (the flood-fill runs untouched). K_* are tunable weights for the sweep; the
    // output is clamped to the same [-200, +275] mp range as the flood-fill term.
    inline bool ENABLE_CHEAP_BISHOP_COMPLEX = false;
    inline int CHEAP_BISHOP_BLOCK = 30;   // penalty per own pawn on the bishop's colour
    inline int CHEAP_BISHOP_MOB   = 6;    // bonus per diagonally-attacked square (current scope)
    inline int CHEAP_BISHOP_FWD   = 8;    // extra bonus per attacked square in the enemy half
    inline int CHEAP_BISHOP_KING  = 12;   // extra bonus per attacked square in the enemy king zone

    // Corrected static-exchange evaluation (see() in cpp_bitboard.h). Default off = the existing
    // (buggy) path. ON recomputes the side-to-move's attacker set from live occupancy each iteration
    // (exact x-ray reveals) and picks the least-valuable attacker by true piece type instead of the
    // stale eval-magnitude square_values[]. The original see() is wrong on ~2.16% of capture targets
    // (diagnostics/see_selfcheck.cpp). Behavioral (ordering + qsearch SEE filter + capture_gains) -> gated.
    inline bool ENABLE_SEE_FIX = false;

    // Margin-gated verification re-search: when > 0, a reduced move that fails low
    // by less than this margin (a near-miss) is re-searched. The one mechanism that
    // uses the "how close to alpha" signal. The default is the blitz-validated
    // keeper (margin 6000 with VERIFY_RESEARCH_REDUCTION 2); set 0 to disable it
    // and recover the old byte-identical search (e.g. for the d10 isolation control).
    inline int VERIFY_MARGIN = 6000;

    // Graduated verification: re-search the near-miss at depth_limit - this (a
    // shallow re-look) instead of full depth, to cut the cost of a wide VERIFY_MARGIN.
    // 0 = full-depth re-search (the original VERIFY behavior); larger = cheaper. The
    // default 2 is the validated config: a clean +17 at LIGHTNING (real-play) without
    // the ply-cost of the deeper re-search that made margin-6000/reduction-1 regress.
    inline int VERIFY_RESEARCH_REDUCTION = 2;

    // Exclusive upper bound on the iterative-deepening depth_limit. Default 64 is
    // the normal play cap (a time-limited preset governs the actual depth reached);
    // set the MAX_DEPTH env knob to 11 to pin the fixed-depth-10 isolation control.
    inline int MAX_ITERATIVE_DEPTH = 64;

    // Occurrence count at which an in-search repeated position is scored as a draw.
    // Default 2 treats the first repetition on the search path as a draw — the standard,
    // game-theoretically sound search behavior (if a line can repeat once, the side that
    // wants the draw can force the threefold), and the fix for the perpetual-check blind
    // spot where the winning side over-values a line that is really a forced draw. Set to
    // 3 to restore the old true-threefold search (only every third occurrence is a draw).
    inline int REPETITION_THRESHOLD = 2;

    // Check extension: a move that GIVES CHECK searches its child one ply deeper so forcing
    // lines (perpetuals, mating attacks) are reached without spending general depth. The
    // value is the hard cap on extensions per root-to-leaf path. Default 3 is the validated
    // keeper (cap-sweep plateaus ~3-4: +20 WAC@LIGHTNING / +9 @d10, and free on the clock in
    // time-limited play since mates found early hit the score cutoff). 0 = off. NOTE: only
    // checks are extended today; "forcing" (recaptures/promotions) is unimplemented.
    inline int CHECK_EXTENSION = 3;

    // SEE filter on the check extension. When SEE_EXTEND_MARGIN is set below
    // SEE_EXTEND_DISABLED, a checking move is extended only if the opponent cannot win its
    // checker (static exchange on the checker's square) by more than SEE_EXTEND_MARGIN — i.e.
    // spite checks (a hanging checker) are skipped. 0 = sound checks only; positive (in piece-
    // value units, pawn=1000) = tolerate small sacrifices. Default DISABLED = extend every
    // check (current behavior; no see() call is made, so the default build is byte-identical).
    inline constexpr int SEE_EXTEND_DISABLED = 1000000;
    inline int SEE_EXTEND_MARGIN = SEE_EXTEND_DISABLED;

    // Honest bound flags at the root / preliminary-ordering TT stores. Those four stores
    // (alpha_beta and reorder_legal_moves) hardcode TTFlag::EXACT, which is only correct under
    // the infinite root window where the returned fail-soft score always lands in-window. With a
    // narrowed root window (aspiration) the score is routinely a fail-low/high bound, so storing
    // it as EXACT poisons the TT for the re-search and later iterations. When true, the flag is
    // computed from the score's position in the window (score <= alpha -> UPPERBOUND, score >=
    // beta -> LOWERBOUND, else EXACT), the same way every deeper store already does. Required for
    // sound aspiration windows; harmless on its own. Default true ships alongside aspiration
    // (below); set 0 to recover the hardcoded-EXACT build (e.g. the byte-identical d10 control).
    inline bool HONEST_ROOT_TT = true;

    // Aspiration windows: search each iterative-deepening iteration (from ASPIRATION_MIN_DEPTH up)
    // in a narrow window centred on the previous iteration's score instead of the full window, to
    // win cutoffs and depth. ASPIRATION_DELTA is the initial half-width in piece-value units
    // (pawn=1000); a failing side is widened by ASPIRATION_WIDEN_PCT percent per resize (200 = x2)
    // up to ASPIRATION_MAX_WIDENINGS times, then the search falls back to the full window. Requires
    // HONEST_ROOT_TT (a narrow window is unsound without it). Default DELTA 500 is the sweep-picked
    // value (efficiency sweet spot, +0.32 LIGHTNING depth at equal clock, STS +2.2pp); set 0 to
    // disable (full window every iteration = the byte-identical control). An overnight LIGHTNING
    // confirm of 800/1200 may yet bump this — see dev_notes/OPTIMIZATION_LOG.md.
    inline int ASPIRATION_DELTA = 500;
    inline int ASPIRATION_MIN_DEPTH = 5;
    inline int ASPIRATION_MAX_WIDENINGS = 3;
    inline int ASPIRATION_WIDEN_PCT = 200;

    // Main TT (searchEvalCache) associativity. 1 = direct-mapped (one entry per index, today's
    // behavior, byte-identical). 2/4/8 = N-way set-associative: the index selects a bucket of N
    // entries and a conflicting store evicts the shallowest, so expensive deep entries survive
    // collisions longer (more big subtree-skips land). Must be a power of two dividing
    // TT_CACHE_SIZE; the footprint is unchanged (same array, reinterpreted as TT_CACHE_SIZE/TT_WAYS
    // buckets). Default 1 = off.
    inline int TT_WAYS = 1;
    // Debug-only (default off = byte-identical): validate the SearchData parallel-array invariant and
    // move legality in the search, logging violations to stderr instead of crashing. Enabled with
    // CHESS_DEBUG_INVARIANTS=1 to hunt the move-ordering corruption.
    inline bool DEBUG_INVARIANTS = false;
}

struct BoardState
{
    uint64_t pawns;
    uint64_t knights;
    uint64_t bishops;
    uint64_t rooks;
    uint64_t queens;
    uint64_t kings;

    uint64_t occupied_colour[2]; // occupied[0] = black, occupied[1] = white
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

struct MoveData
{
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

struct Move
{
    uint8_t from_square;
    uint8_t to_square;
    uint8_t promotion;

    // Constructor with default values
    Move(uint8_t from_square_ = 0, uint8_t to_square_ = 0, uint8_t promotion_ = 0)
        : from_square(from_square_), to_square(to_square_), promotion(promotion_) {}

    bool operator==(const Move &other) const
    {
        return from_square == other.from_square &&
               to_square == other.to_square &&
               promotion == other.promotion;
    }
};

struct RootScore
{
    // The top-level score alpha_beta found for one root move, paired with the second-level reply
    // ordering and scores minimizer produced for it (carried into the next iteration as a move-ordering
    // hint). Grouping these three formerly-parallel fields makes a per-move push/pop atomic, so they can
    // never drift out of index-correspondence -- the SearchData desync that caused the warm-cache
    // corruption crashes.
    int top_score;
    std::vector<Move> second_moves;
    std::vector<int> second_scores;

    RootScore() : top_score(0) {}

    RootScore(int top_score_, std::vector<Move> second_moves_, std::vector<int> second_scores_)
        : top_score(top_score_),
          second_moves(std::move(second_moves_)),
          second_scores(std::move(second_scores_)) {}
};

struct SearchData
{
    // Full ordered root-move list (length N). Set once per node; never push/pop'd element-wise.
    std::vector<Move> moves_list;

    // One entry per *searched* root move (cutoff length <= N): the top-level score plus the second-level
    // reply ordering/scores. moves_list[i] corresponds to scores[i] for every i < scores.size().
    std::vector<RootScore> scores;

    SearchData() = default;
};

void initialize_engine(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn, bool side_to_play);
void set_current_state(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bool turn);
inline void make_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, Move move, uint64_t zobrist, bool capture_move);
inline void unmake_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key);

inline void update_cache(int num_plies);

MoveData get_engine_move(std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count);
int alpha_beta(int alpha, int beta, int cur_depth, int depth_limit, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, const TimePoint &t0, SearchData &previous_search_data, Move &best_move, int &num_iterations);
int minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> second_level_preliminary_scores, std::vector<Move> second_level_moves_list, RootScore &out_entry, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search);
int maximizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move previousMove, int &num_iterations, bool last_move_was_capture, bool last_move_was_null_move, bool is_in_null_search);
SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint &t0, uint64_t zobrist, SearchData previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, int &num_iterations);
int pre_minimizer(int cur_depth, int depth_limit, int alpha, int beta, const TimePoint &t0, std::vector<int> &preliminary_scores, std::vector<Move> &pre_moves_list, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations);
int qSearch(int alpha, int beta, int cur_depth, int qDepth, const TimePoint &t0, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing);

inline void sortSearchDataByScore(SearchData &data);
inline void descending_sort_wrapper(const SearchData &preSearchData, SearchData &mainSearchData);
inline void ascending_sort(std::vector<int> &values, std::vector<Move> &moves);
inline uint8_t get_piece_type(uint8_t square, std::vector<BoardState> &state_history);
inline bool relevant_pin_exists(std::vector<BoardState> &state_history, bool probe);
inline void use_tt_entry(TTEntry &entry, int &score, bool &using_tt, int alpha, int beta, int &num_iterations, bool is_maximizing, bool using_extra_precautions);
inline void increment_node_count_with_decay(int &num_iterations);
inline bool isUnsafeForNullMovePruning(BoardState current_state);
inline bool is_repetition(const std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist_key, const int repetition_count);
inline int reduced_search_depth(int depth_limit, int cur_depth, bool is_in_relavent_pin, int move_number, BoardState current_state);
inline void updatePV(Move move, int cur_depth);

inline int get_q_search_eval(int alpha, int beta, int cur_depth, const TimePoint &t0, std::vector<BoardState> &state_history, BoardState current_state, std::unordered_map<uint64_t, int> &position_count, uint64_t zobrist, Move prevMove, int &num_iterations, bool is_maximizing);
inline int get_board_evaluation(std::vector<BoardState> &state_history, uint64_t zobrist, int &num_iterations);
inline std::vector<Move> buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove);
inline std::vector<Move> buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, Move prevMove);

#endif // SEARCH_ENGINE_H