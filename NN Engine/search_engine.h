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
// RootScore sentinel for "this move has no score we proved" -- SF's -VALUE_INFINITE in the same role.
// It is a MARKER, not a magnitude: nothing may prune, reduce, or seed alpha on it. Matches the value the
// root razor's CONTINUE path already writes, so the two paths agree on one representation of "unproven".
constexpr int ROOT_SCORE_UNPROVEN = -9999998;
// Age for a root entry that has never carried a verified score. Large enough that no ROOT_RAZOR_MAX_AGE
// setting can ever treat it as recent evidence.
constexpr int ROOT_AGE_NEVER = 1000000;

// History-gravity bounds (MAX_HISTORY, CONT2_GRAVITY_DIV) are env-tunable Config members below.

constexpr std::array<int, 4> FUTILITY_MARGINS = {200, 450, 650, 950};

constexpr int SUPPORT_MARGIN = 0;
// MAX_QDEPTH / DELTA_MARGIN promoted to env-tunable Config knobs below (qsearch EBF levers).

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
    inline bool ROOT_RAZOR_CONTINUE = false; // razor individual low root moves (continue) vs abandon the rest (break).
                                             // PROVABLY INERT: the root list is sorted score-descending before the loop,
                                             // so the razor condition is monotone in the move index -- once it trips it
                                             // holds for every later move, and continue skips the same set as break.
                                             // Verified byte-identical (WAC d10 248/44,038,704). The real lever is the
                                             // threshold below, not the skip mode.

    // Root razoring threshold: a root move whose PREVIOUS-iteration score is more than the threshold below
    // alpha is abandoned. The threshold decays with depth (RAZOR_DECAY_PCT per ply beyond 4) down to a floor,
    // and widens at runtime whenever alpha jumps past the front move's stale score. These are absolute
    // millipawn margins, so they are coupled to eval magnitude -- they were calibrated before the de-king
    // change shrank attack-position evals. Defaults reproduce the previous hardcoded values exactly.
    inline int RAZOR_BASE_FIRST = 750;  // base on the first iteration (no previous root list)
    inline int RAZOR_FLOOR_FIRST = 200; // floor on the first iteration
    inline int RAZOR_BASE = 300;        // base once a previous root list exists
    inline int RAZOR_FLOOR = 100;       // floor once a previous root list exists
    inline int RAZOR_DECAY_PCT = 75;    // per-ply decay beyond depth 4, in percent
    inline int RESIGN_THRESHOLD = -15000;   // engine resigns (returns no move) at score <= this. Default -15000 = current.
                                             // Set very negative to measure the resign leak (games that would draw if played on).
    inline bool ENABLE_TT_STORE_DRAW = false; // allow caching EXACT-draw (score==0) subtrees in the TT. Default false =
                                             // current (refused -> every draw subtree re-searched). Mates still refused
                                             // (they need mate-distance adjustment). Diagnostic: does storing draws cut nodes?
                                             // WARNING: the score==0 refusal is also what keeps the FABRICATED value out
                                             // of the TT -- minimizer/maximizer return a bare 0 on a timeout or node-limit
                                             // abort, and a parent stores that score before reaching its own abort check.
                                             // Enabling this therefore also starts caching abort values as real bounds.
    inline bool ENABLE_NULLMOVE = true; // null-move pruning
    inline bool NULLMOVE_PROGRESSIVE = false; // depth-scaled null-move reduction (-2 at d>=12, -3 at d>=14); off = flat -1
    inline int NULLMOVE_EXTRA = 2;      // extra plies off the null-move search depth (more aggressive null pruning); 0 = byte-id baseline, 2 = combo1
    inline bool ENABLE_QDELTA = true;   // delta pruning in quiescence
    inline int DELTA_MARGIN = 1500;     // qsearch delta-pruning margin (lower = prune more captures); EBF lever
    inline int MAX_QDEPTH = 10;         // qsearch depth cap (lower = shallower qsearch); EBF lever

    inline bool LMR_PROFILE = false; // env-gated LMR-miss profiler (diagnostic; off = byte-identical)

    // Remaining-depth floor for LMR. reduced_search_depth returns an ABSOLUTE target depth derived from
    // the ITERATION depth (DEPTH_REDUCTION[depth_limit]) and the move number -- it never reads cur_depth,
    // so the reduction is a constant number of plies regardless of how deep the node sits. Near the root
    // that leaves a real search; deep in the tree, where only 1-2 plies remain, the same constant wipes
    // the child out entirely and it drops straight to qsearch. The prune-shadow LMR measurement shows the
    // damage exactly there: wrong-reduction is ~1% at L1-L5 but 4.1% at L6 and 6.2% at L8.
    // This keeps at least LMR_REM_FLOOR_PCT percent of the child's remaining depth. 0 = off = byte-identical.
    inline int LMR_REM_FLOOR_PCT = 0;

    // Minimum remaining depth for LMR to fire at all. A percentage floor cannot help where the damage
    // actually is: at remaining depth 1 the floor rounds to zero and the child still drops to qsearch.
    // SF gates LMR on a minimum depth and its log(remaining) reduction collapses toward zero down there.
    // Below this many remaining plies the move is searched unreduced. 0 = off = byte-identical.
    inline int LMR_MIN_REM = 0;

    // Key the LMR reduction on the node's OWN remaining depth instead of the iteration depth.
    // DEPTH_REDUCTION is a table of absolute target depths, so DEPTH_REDUCTION[D] implies a reduction of
    // (D - DEPTH_REDUCTION[D]) plies; indexing it with depth_limit applies that one constant at every node
    // of the iteration. Indexing it with the remaining depth instead reuses the same tuned curve where it
    // was meant to apply, and collapses the reduction toward zero near the horizon the way SF's
    // reductions[remaining] does. The clamp to rem-2 is part of the form, not a tuning choice: a reduction
    // may never consume the child's last ply, which is the truncate-straight-into-qsearch case the
    // prune-shadow measurement blamed for 4.1% (L6) / 6.2% (L8) wrong reductions.
    // Default off = byte-identical.
    inline bool ENABLE_LMR_REMDEPTH = false;

    // Percent applied to the remaining-depth reduction. Re-indexing alone reduces LESS almost everywhere
    // (at a depth-10 iteration the constant is 2 plies while the curve gives 1 below the root), and pure
    // de-aggression is already known to pay nothing here -- LMR_MIN_REM=4 removed ~35% of all wrong
    // reductions for WAC +1 / STS -8. This dial restores or exceeds the old aggression on top of the
    // sounder shape, so the pair can be tested at the corner rather than only on the de-aggression side.
    inline int LMR_REMDEPTH_SCALE = 100;

    // Margin for the PER-MOVE qsearch futility test (ENABLE_QDELTA_PERMOVE). It must be its own knob:
    // the node-level prune asks "is static_eval below alpha by more than DELTA_MARGIN", while the
    // per-move test asks "is static_eval below alpha by more than the margin PLUS the victim's value".
    // Reusing DELTA_MARGIN (1500) therefore demands a >=2500 gap to skip even a pawn capture and >=11500
    // for a queen -- which is why the per-move prune never fired. SF's equivalent margin is ~0.6 pawn
    // (SF11 futilityBase = bestValue + 128 with its own scale), i.e. far smaller than ours.
    // 0 = fall back to DELTA_MARGIN (previous behaviour).
    inline int QDELTA_PERMOVE_MARGIN = 1500;

    // Sound-LMR exemptions (default OFF = current behavior). Stop reducing the
    // moves most likely to be the critical misses; env-gated so the A/B needs no
    // recompile and the default build stays byte-identical.
    inline bool PROTECT_KILLERS = false; // don't LMR-reduce killer / counter moves
    inline bool PROTECT_PV = false;      // don't LMR-reduce at PV nodes (beta - alpha > 1)
    // Only apply the PV/killer protection to EARLY moves (index i <= this): move 0 is already never
    // reduced (base_lmr requires i != 0), so the value is in protecting the 2nd/3rd, where a strong
    // move is not yet guaranteed. A killer/PV appearing deep in the list is likely stale and can be
    // reduced. Bounds the (otherwise large) node cost of blanket protection. Default 64 = protect
    // everywhere (the original behavior when PROTECT_* is on); PROTECT_* default off keeps this byte-id.
    inline int PROTECT_MAX_IDX = 64;

    // History-aware LMR ("reduce-less"): search known-good late quiets a little less reduced (toward,
    // never beyond, full depth). Categorical signal — killer/counter membership + a coarse history
    // tier — not an absolute score threshold, so it is robust to the unbounded/uneven history values
    // and to a later continuation-history upgrade. SHIPPED default = the reduce-MORE arm (CAP=0,
    // MORE_CAP=1): STS300 50.1->51.7% and -1.3% nodes @d10; self-play +25.5 +/-65 (positive, not sig).
    inline bool ENABLE_HISTORY_LMR = true;  // master gate for the history-aware LMR adjustment
    inline int HISTORY_LMR_CAP = 0;         // plies to REMOVE for good quiets (reduce-less; 0 = off, the shipped arm)
    inline int HISTORY_LMR_MORE_CAP = 1;    // plies to ADD for never-cut (tier-0) quiets (reduce-more; the shipped lever)
    // Depth-scaled reduce-more: for a quiet already flagged reduce-more, add (remaining_depth / SCALE) extra
    // reduction plies (capped) -- prune low-history quiets HARDER at deeper nodes. 0 = off = byte-identical.
    inline int HISTORY_LMR_SCALE = 2;       // divisor on remaining depth (smaller = more aggressive); 0 = off (byte-id baseline), 2 = combo1
    inline int HISTORY_LMR_SCALE_CAP = 2;   // max extra reduction plies from the depth scaling

    // Continuous statScore-LMR: the graded, two-sided generalization of the tiered HISTORY_LMR_CAP above.
    // Instead of bucketing the plain history score, sum the history tables into one statScore and map it
    // smoothly to a signed LMR delta: delta = clamp((statScore - OFFSET) / DIVISOR, -CLAMP, +CLAMP), where
    // positive = reduce-less (search a proven quiet closer to full depth) and negative = reduce-more (prune a
    // never-good quiet harder). One formula reallocates search budget both directions (vs the flat cap that
    // only adds nodes). OFFSET is our OWN median (measured, NOT Stockfish's -4926); DIVISOR is tuned to our
    // units (P95(|statScore-OFFSET|)/DIVISOR ~= 1.5 plies). Default off (ENABLE_STATSCORE_LMR=false) leaves
    // the tiered history_lmr_delta path untouched = byte-identical. Master gate + all knobs env-tunable so
    // the continuous channel and the structural killer/counter overlay can be swept independently on node_ab.
    // COUPLING CAVEAT: OFFSET/DIVISOR are fit to the CURRENT history-table distribution. Any change to the
    // history scoring (bonus formula, the 4x continuation multiplier, decay cadence, ENABLE_HISTORY_SATURATION,
    // ENABLE_HISTORY_MALUS) shifts that distribution and INVALIDATES these constants -- re-derive them via the
    // ENABLE_STATSCORE_PROFILE sweep + node_ab. See dev_notes/history-scoring-calibration-2026-07-07.md.
    // Measured node_ab: peak at OFFSET=512, DIVISOR=768-1024 (~+30 Elo); KILLER_BONUS added nothing.
    inline bool ENABLE_STATSCORE_LMR = true;  // SHIPPED 2026-07-08: +23 Elo lightning SPRT (906g) / +33 node_ab;
                                              // replaces the tiered HISTORY_LMR_CAP path (off = tiered, byte-id 245).
    // SHIPPED 2026-07-30 with gravity (see ENABLE_HISTORY_SATURATION/MALUS): re-derived from the
    // POST-GRAVITY statScore distribution via the ENABLE_STATSCORE_PROFILE sweep. The pre-gravity values
    // were 512/1024; keeping them under gravity costs -90 STS, because the constants grade QUIET moves and
    // gravity changes the scale of every table statScore reads.
    // ⚠️ These are one atomic unit WITH gravity. Reverting gravity while leaving these (or vice versa)
    // leaves the shipped statScore-LMR channel reading a distribution it was never fitted to.
    inline int STATSCORE_OFFSET = 0;          // re-centering offset (empirical; zero-history quiet -> delta 0)
    inline int STATSCORE_DIVISOR = 683;       // statScore units per ply
    inline int STATSCORE_CLAMP = 2;           // max |delta| plies applied to the reduction
    inline int STATSCORE_MAIN_W = 1;          // weight on main history in the statScore sum
    inline int STATSCORE_CONT1_W = 1;         // weight on 1-ply continuation history (counterMoveHeuristics)
    inline int STATSCORE_CONT2_W = 1;         // weight on 2-ply continuation history (contHist2)
    inline int STATSCORE_KILLER_BONUS = 0;    // extra reduce-less plies for a killer/counter move, layered ON
                                              // TOP of the continuous delta (SF-style structural overlay).
                                              // 0 = pure-continuous (clean v1); >0 = overlay
    inline bool ENABLE_STATSCORE_PROFILE = false; // diagnostic only: accumulate the raw statScore distribution
                                                  // (for OFFSET/DIVISOR derivation); no effect on the search

    // Prune-shadow verification (diagnostic only): at every SHADOW_N-th LMP / futility skip, search the
    // pruned move anyway (full window, full remaining depth) and record whether it WOULD have entered the
    // node's window (a wrong prune) or even cut -- the per-mechanism wrong-prune rate we otherwise have zero
    // visibility into (LMP/futility skip moves with no measurement, unlike LMR's fail-low profile). The
    // shadow result is discarded (the real prune still fires), so this only ADDS observational nodes; it does
    // not change the search decisions. Off (ENABLE_PRUNE_SHADOW=false) = byte-identical. Run it alone; the
    // extra nodes make node counts non-comparable, so read only the wrong-prune RATES it prints.
    inline bool ENABLE_PRUNE_SHADOW = false;
    inline int SHADOW_N = 256; // sample 1 in N prune sites (deterministic; larger = cheaper, coarser)

    // Cutoff-calibration logger (diagnostic; measure-first gate for reviving gravity/malus): at each quiet
    // beta-cutoff, log the cutting move as CUT and the tried-and-failed quiets as FAILED, bucketed by their
    // statScore -> reliability curve P(cut|statScore) + 0-bucket composition. Populates the searched_quiets
    // list (like malus) so it costs a little when on; no effect on search decisions. Off = default. See
    // dev_notes/history-scoring-calibration-2026-07-07.md.
    inline bool ENABLE_CUTCAL_LOG = false;

    // Decoupled cut-rate table (Qcut): a SEPARATE gravity-bounded signed history (+bonus on the cutting quiet,
    // -malus on tried-and-failed quiets) read ONLY by statScore-LMR (statScore += QCUT_LAMBDA*Qcut/256), NEVER
    // by move ordering. This is how malus escapes its 4x ORDERING-pollution failure: it feeds the scale-
    // sensitive LMR consumer (which the count-refinement proved wants it: 0-bucket P(cut) 0.48->0.25 by
    // tried-fail count) while the ordering history table stays clean. QCUT_LAMBDA=0 => byte-identical to the
    // shipped statScore (a continuous dial). Reset per search. Off = default = byte-identical. ⚠️ non-zero
    // LAMBDA shifts the statScore distribution -> re-derive OFFSET/DIVISOR (coupling caveat above).
    inline bool ENABLE_QCUT = false;
    inline int QCUT_LAMBDA = 256;     // read weight, fixed-point /256 (256 = weight 1); 0 = off even when enabled
    inline int QCUT_MAX = 16384;      // gravity saturation bound (signed)
    inline int QCUT_MALUS_DIV = 1;    // malus softening: tried-and-failed penalty = bonus / QCUT_MALUS_DIV

    // Capture-chain LMR guard: protect a quiet move when the move that led to this node was a capture
    // (we're resolving a capture sequence -- a forcing line where reductions bury tactics, e.g. the
    // Gap-T axb5 mis-reduction). Lets aggressive LMR (LMR_EXTRA / DEPTH_REDUCTION) be pushed harder
    // safely. Default off = byte-identical.
    //   CAPCHAIN_REDUCE_LESS == 0 -> hard skip (no reduction at all on the resolving quiet) -- strong
    //     but node-expensive (a single capture parent disables LMR for the whole quiet subtree).
    //   CAPCHAIN_REDUCE_LESS  > 0 -> reduce-LESS by that many plies (toward, never beyond, full depth):
    //     keeps most of the node savings while still de-pruning the forcing line.
    inline bool ENABLE_LMR_CAPCHAIN = false;
    inline int CAPCHAIN_REDUCE_LESS = 0; // plies to shave off the reduction in a capture sequence (0 = hard skip)
    inline int CAPCHAIN_RUN_THRESH = 2;  // min capture-chain density (g_captureChain) to fire the guard
                                         // (1 = any capture in the recent line; 2+ = a multi-capture sequence)

    // Improving heuristic: reduce one extra ply when the side-to-move's static eval is NOT rising vs
    // 2 ply back (stagnant -> prune harder). Modulates LMR only (composes with history_lmr_delta); gated.
    inline bool ENABLE_IMPROVING = false;
    inline int IMPROVING_EVAL_WINDOW = 6;   // populate g_evalStack only within this many plies of the leaf (cost control)
    // Improving's per-node eval was shelved as too costly (a full eval at interior nodes -> Delta-depth -0.18).
    // The heuristic needs only the SIGN of the eval trend, so use the cheap material+PST surrogate (cheap_eval)
    // instead of the full eval -- far cheaper, and accuracy is irrelevant to the trend. Only consulted when
    // ENABLE_IMPROVING is on, so default has no effect on the byte-identical (improving-off) build.
    inline bool IMPROVING_CHEAP = true;
    // The two tunables behind the improving LMR nudge (defaults = the original strict-sign / -1-ply behavior,
    // so an improving-on build is byte-identical at defaults). PACE sweep targets. DELTA_MARGIN widens the
    // "still improving" band so a small (noisy, esp. with the cheap eval) eval wobble is not treated as
    // not-improving -> fewer false extra-reductions. REDUCTION is how many extra LMR plies when not improving.
    inline int IMPROVING_REDUCTION = 1;
    inline int IMPROVING_DELTA_MARGIN = 0;

    // Late-move pruning (LMP / move-count pruning): at low remaining depth, SKIP late quiet moves entirely
    // (not just reduce them like LMR). Reuses the LMR eligibility (do_lmr) so captures, checks, promotions,
    // killers/counter and in-check are never pruned. Prune when the move index i >= LMP_BASE + LMP_SCALE*rd*rd
    // (rd = remaining depth = depth_limit - cur_depth) and rd <= LMP_MAX_DEPTH. Default off = byte-identical.
    // Behavioral (changes the tree). SHIPPED default-on: paired with the gentle lazy-resort below, base-vs-this
    // self-play = +59.2 +-19.2 Elo (1729 games, LIGHTNING; +0.50 ply at equal time). env-off recovers the old tree.
    inline bool ENABLE_LMP = true;
    inline int LMP_MAX_DEPTH = 5;   // only LMP when remaining depth (depth_limit - cur_depth) <= this; 3 = byte-id baseline, 5 = combo1
    inline int LMP_BASE = 1;        // base late-move count; 3 = byte-id baseline, 2 = combo1, 1 = shipped w/ capg-cond 2026-07-02 (+39 STS on reduced capg)
    inline int LMP_SCALE = 1;       // quadratic depth term in the threshold

    // History-gated pruning (Ethereal-style): read the EXISTING from×to butterfly historyHeuristics to make the
    // move-count prune ORDERING-AWARE — no new table, no fragmentation (the threat-hist/piece-key under-fill
    // trap). Both default off = byte-identical.
    //  - LMP history EXEMPTION: a late quiet whose butterfly history >= LMP_HIST_EXEMPT is NOT LMP-pruned
    //    (protects proven-good quiets from the move-count prune = the antidote to prunes eating strategic quiets).
    //  - History PRUNING: a late quiet whose butterfly history < -HIST_PRUNE_COEF*rd is skipped at low remaining
    //    depth (prunes ordering-condemned quiets harder = the cash-in; rd = remaining depth).
    inline bool ENABLE_LMP_HIST_EXEMPT = false;
    inline int LMP_HIST_EXEMPT = 4000;
    inline bool ENABLE_HIST_PRUNE = false;
    inline int HIST_PRUNE_COEF = 512;
    inline int HIST_PRUNE_MAX_DEPTH = 4;

    // Root pre-search (reorder_legal_moves) depth = depth_limit - this. The pre-search is a full-width shallow
    // search of the root run every iteration to order moves (and generate the 2nd-level lists). 1 = byte-id
    // (depth_limit-1, original). Higher shrinks the pre-search toward a 1-ply pass (cheaper, weaker ordering) —
    // tests whether the deep pre-pass earns its ~1/EBF node overhead. Clamped to depth >= 1.
    inline int ROOT_PRESEARCH_REDUCTION = 1;
    // How the root pre-search treats the TAIL -- root moves the previous iteration left no score for, past
    // the first PRESEARCH_CHUNK of them. 0 = full pre_minimizer (default, byte-identical), 1 = pre_minimizer
    // at a reduced depth, 2 = no search at all, just move_gen's own ordering. Tail nodes are ~61% of
    // pre-search nodes (~25% of ALL nodes), and 47.8% of deep iterations never read them, so this is where
    // the saving is. CHUNK keeps the moves the root loop actually reaches on a real search.
    // Skip pre-searching root moves the previous iteration already scored. descending_sort_wrapper discards
    // those pre-search entries anyway (it keeps the previous iteration's real RootScores for the prefix and
    // uses the pre-pass only beyond `count`), so the work is re-derived and thrown away. Unlike the TAIL,
    // these moves were searched to FULL DEPTH last iteration, so the TT already holds deep entries for them
    // and the shallow re-search adds little warming that is not already present.
    // Root late-move reductions. Stockfish (invariant SF11->SF18) and Ethereal both REDUCE late root moves
    // and never abandon one; we do neither -- we razor and break, so ~23 of ~35 root moves are never touched
    // by the main search at any iteration. Reductions start later at root than at interior nodes, and later
    // still while nothing has beaten alpha (SF: moveCount > 1 + rootNode + (rootNode && bestValue < alpha)).
    // A reduced scout that beats alpha is re-searched at full depth, so a reduction can never decide a move.
    inline bool ENABLE_ROOT_LMR = false;
    inline int ROOT_LMR_MIN_IDX = 3;    // first move index eligible for reduction
    inline int ROOT_LMR_BASE = 1;       // plies removed at the first eligible move
    inline int ROOT_LMR_DIV = 4;        // +1 ply per this many moves further down the list
    // Exempt the previous iteration's best move from root reduction. SF spells this
    // `(!rootNode || best_move_count(move) == 0)` -- a move that has recently been best is never reduced,
    // because losing it to a reduced fail-low costs the search the move it most likely wants. SF counts
    // best-ness across several iterations; we approximate with the immediately preceding winner only.
    inline bool ROOT_LMR_EXEMPT_BEST = false;
    // Persistent root table, SF's RootMoves semantics. Today alpha_beta CLEARS the caller's scores and
    // re-pushes only the moves it searched, so a move scored at iteration 5 and razored at 6 loses its
    // score permanently and our table covers ~34% of root moves against SF's 100%. On: the table is sized
    // to the full root list BEFORE the loop, every entry starts as ROOT_SCORE_UNPROVEN, and searched moves
    // are written IN PLACE -- which also makes the full-length invariant structural instead of conventional,
    // so all eight of alpha_beta's exits leave a complete table rather than a truncated stump.
    inline bool ENABLE_ROOT_TABLE = false;
    // How stale a verified score may be and still be razorable, in iterations (0 = proven this iteration
    // only). Entries older than this -- and every unproven sentinel -- are SKIPPED, never razored, which is
    // the generalisation of the synthetic_from guard. Only consulted when ENABLE_ROOT_TABLE is on.
    inline int ROOT_RAZOR_MAX_AGE = 1;
    // Second level of the root sort key when ENABLE_ROOT_TABLE is on. Off = SF's literal `previousScore`.
    // On = the most recent MEASURED score, which breaks ties inside the fail-low block that previousScore
    // leaves degenerate (a chronic fail-low move's previousScore is itself the sentinel, so the block ties
    // on both levels and the stable sort just freezes the old order).
    inline bool ROOT_SORT_L2_LASTREAL = false;
    // Sort the root list primarily by the most recent MEASURED score rather than by SF's sentinel key.
    // Store-only-verified collapses ~97% of moves into one undifferentiated sentinel block -- a fail-low is
    // not a bad move, it merely failed to beat alpha this iteration, and before the table those moves
    // sorted among each other by their real fail-soft values. SF can afford to discard that ordering
    // because its root neither razors nor reduces by index; ours does both. Keeps the table's full-length
    // bookkeeping while restoring the pre-table ordering policy.
    inline bool ROOT_SORT_L1_LASTREAL = false;
    // Hybrid razoring: three outcomes instead of two. Today a root move is either razored away entirely or
    // searched at full depth, and the recency guard turned much of the first bucket into the second -- which
    // is where the table's node cost comes from. A move whose deficit only just clears the razor threshold
    // is "suspect", not "hopeless": reduce it rather than paying full depth or discarding it. Reductions are
    // re-searched at full depth whenever they beat alpha, so a reduction can never by itself decide a root
    // move -- which is what the razor's outright skip cannot promise (it discards the eventual winner
    // 16.7-17.9% of the time).
    inline bool ROOT_RAZOR_TO_LMR = false;
    // Deficit beyond the razor threshold at which a move is hopeless enough to skip outright. Below this it
    // gets a reduced search instead. 0 = never skip (reduce everything the razor catches).
    inline int ROOT_RAZOR_SKIP_MARGIN = 0;
    inline int ROOT_RAZOR_LMR_BASE = 1;   // plies removed as soon as the razor condition trips
    inline int ROOT_RAZOR_LMR_DIV = 400;  // +1 further ply per this much extra score deficit
    // Reject a stored score searched shallower than (current depth - this) as evidence for pruning. Guards
    // the reduced-score-reuse hazard the depth field exists for.
    inline int ROOT_RAZOR_MAX_DEPTH_DEFICIT = 2;
    // The other half of the hybrid, and the only branch that can REMOVE work. A move whose score clears the
    // razor threshold but whose evidence is too stale/shallow to prune on currently falls through to a FULL
    // depth search -- that bucket (tens of thousands of moves per bench) is where the table's node cost
    // lives. We cannot skip it (we do not trust the score) but we need not pay full depth either: reduce it,
    // and let the standard re-search promote it if it beats alpha.
    inline bool ROOT_STALE_TO_LMR = false;
    inline int ROOT_STALE_LMR_BASE = 1;   // plies removed for a stale-but-suspect move
    inline int ROOT_STALE_LMR_DIV = 600;  // +1 further ply per this much score deficit
    // Root razoring independently of interior razoring, so the razor x LMR 2x2 can be measured. SF has no
    // root razoring at all; ours prunes on PREVIOUS-ITERATION scores, which is different information.
    inline bool ENABLE_ROOT_RAZOR = true;
    // Run the root pre-search only while depth_limit < this, then rely purely on previous-iteration data.
    // The pre-search earns its keep when the previous table is absent or unstable (the early iterations);
    // by the time ordering has settled it is mostly re-deriving what the main search already knows.
    // 0 = never switch off (default, byte-identical).
    inline int PRESEARCH_OFF_FROM_DEPTH = 0;
    inline bool ENABLE_PRESEARCH_SUBSET = false;
    // Safety margin (millipawns) subtracted before a SKIPPED prefix move is allowed to raise alpha. Under
    // PVS most stored root scores are fail-low UPPER bounds -- "at most X", where the true value may be far
    // lower -- so seeding alpha from one directly can set it too high or too low and widen every scout
    // window below. 0 = seed exactly (current behaviour); a very large value = never seed from skipped moves.
    inline int PRESEARCH_SUBSET_ALPHA_MARGIN = 0;
    // 3 = DIAGNOSTIC: search the tail exactly as mode 0 (full warming, full ply-1 list) but overwrite only
    // the root SCORE with the fill value. Isolates the score channel (root ordering + razoring) from the
    // warming channel (TT/killer/history), which mode 2 conflates.
    inline int PRESEARCH_TAIL_MODE = 0;
    inline int PRESEARCH_CHUNK = 4;
    inline int PRESEARCH_TAIL_REDUCTION = 2;   // mode 1 only; pre-search depth floor is 2 (below it
                                               // pre_minimizer returns an EMPTY ply-1 list -- sentinel leak)
    // Hard-off for the root pre-search: when false, skip pre_minimizer entirely and reuse the PREVIOUS
    // iteration's real second-level move lists/scores as the ordering hint (heuristic-fill only the razored
    // tail + first iteration). The clean "is the pre-search worth its nodes?" test. true = byte-identical.
    inline bool ENABLE_ROOT_PRESEARCH = true;

    // SEE pruning (main search): at low remaining depth, skip a do_lmr-eligible quiet whose moved piece
    // can be profitably captured by the immediate recapture (post-move see() from the opponent's side >
    // SEE_PRUNE_MARGIN). The standard "don't search quiets that hang material" lever. Default off = byte-id.
    inline bool ENABLE_SEE_PRUNE = false;
    inline int SEE_PRUNE_MARGIN = 0;    // opponent recapture gain (centipawns; piece=1000) above which to prune
    inline int SEE_PRUNE_MAX_DEPTH = 3; // only prune when remaining depth (depth_limit - cur_depth) <= this

    // SEE pruning of LOSING captures in the main search: at low remaining depth, skip a capture whose static
    // exchange loses more than SEE_PRUNE_CAPTURE_MARGIN (pre-move see() from the mover's side). Captures bypass
    // do_lmr, so this is a separate branch; a checking capture is never pruned. Shares SEE_PRUNE_MAX_DEPTH;
    // independent of ENABLE_SEE_PRUNE (the quiet lever). Default off = byte-identical.
    inline bool SEE_PRUNE_CAPTURES = false;
    inline int SEE_PRUNE_CAPTURE_MARGIN = 0;  // prune a capture whose see() < -this (centipawns; piece=1000)

    // Lazy cached-quiet re-sort: a move-gen cache hit replays an order frozen when the node was first
    // searched, so the late quiets LMP prunes may be stale. On a hit, re-rank the quiet tail against the
    // CURRENT history with score_quiet (move_gen.h), keeping captures + killers/counter pinned at the front.
    // Gated on the node's last cutoff index so it only fires where ordering looks suspect (a deep cutoff).
    // Default off = byte-identical (the cutoff index is not even recorded when off).
    // SHIPPED default-on (the gentle nudge that makes aggressive LMP safe): top-K=2 promote at poorly-ordered
    // nodes only; the periodic FULL re-sort regressed STS so it is OFF (RESORT_AFTER_REUSES set unreachably high).
    inline bool ENABLE_LAZY_RESORT = true;
    inline int PROMOTE_TOP_K = 2;            // cheap path: bubble this many best-by-live-score quiets to the tail front
    inline int RESORT_AFTER_REUSES = 1000000; // full stable_sort of the quiet tail -- effectively OFF (regressed STS)
    inline int LAZY_RESORT_MIN_CUTOFF_IDX = 2; // only re-sort when last_cutoff_index > this (cutoff NOT in the first 3)

    // Passed-pawn pruning exemption: keep an ADVANCED pawn push (landing >= PASSER_EXEMPT_ADV ranks toward
    // promotion, direction inferred from the push) OUT of LMP pruning and LMR reduction, so a slow passer
    // march stays above the horizon -- helps BOTH seeing the opponent's passer threat (defensive) and
    // converting our own (offensive). Move-level (only the push escapes; the rest of the node prunes
    // normally). Default off = byte-identical.
    inline bool ENABLE_PASSER_PRUNE_EXEMPT = false;
    inline int PASSER_EXEMPT_ADV = 5;   // advancement of the landing square toward promotion (0-7); 5 = 6th rank (W) / 3rd (B), catches a4->a3

    // Continuation-aware LMR (default on): a tier-0 (never-cut) quiet with a strong 1-ply continuation
    // score (counterMoveHeuristics) is NOT reduced-more -- a known-good reply to the previous move, so
    // we cancel the extra reduction (never deeper than base). THRESH=2000 is the d10 node-efficiency
    // optimum (WAC nodes -6.2% vs off, WAC 262/300, STS neutral); 4000 and 1000 both save less.
    inline bool ENABLE_CONT_HIST = true;
    inline int CONT_HIST_LMR_THRESH = 2000; // min continuation score to cancel the reduce-more

    // Move-ordering experiments, each benched independently.
    inline bool ENABLE_CONT_HIST_2PLY = false; // 2-ply continuation history -- d10 LOSS at equal weight (WAC -3, +7% nodes); needs down-weight (b/4) + the bonus/malus rework before it's worth anything
    inline bool ENABLE_CAPTURE_HIST = true;    // SHIPPED 2026-07-30 as part of gravcap (+33.0 Elo, 1203 games).
                                               // Looked "marginal" pre-gravity (+0.7 STS/+1 WAC, +2.6% nodes) because
                                               // capture history, like every history consumer, only discriminates once
                                               // malus exists. Best single component on bench under gravity (254 WAC).
    inline bool ENABLE_CHECK_ORDER = false;    // direct-check bonus -- on the SCALE-OFF baseline it's -6 WAC for -9.8% nodes (accuracy traded for speed; bad at fixed depth). BONUS=6000 too hot -> recalibrate lower before re-enabling
    inline int CHECK_ORDER_BONUS = 6000;       // the flat quiet-check ordering bonus
    // TT best-move ordering: promote the transposition table's remembered cutoff move (TTEntry::move) when a
    // position is revisited. It formerly read g_ttMoveTable, which has no write site anywhere in the engine --
    // every lookup returned the default {0,0,0}, so the flag was byte-identical rather than tested.
    // Default off = byte-identical (the TT is never consulted for ordering).
    inline bool ENABLE_TT_MOVE = false;
    // Promotion policy. QUIETS moves the TT move to the head of the quiet region and leaves the SEE/MVV
    // capture order untouched; FRONT hoists it over everything, which is the shape the 2026-07-03 ordering
    // audit blamed for the original experiment's failure (kept so the two can be measured against each other).
    constexpr int TT_MOVE_POLICY_QUIETS = 1;
    constexpr int TT_MOVE_POLICY_FRONT = 2;
    inline int TT_MOVE_POLICY = TT_MOVE_POLICY_QUIETS;

    // History gravity: replace the bonus-only `+= depth²` cutoff update with a saturating bonus/MALUS --
    // reward the move that cut off, penalize the quiets/captures tried-and-failed before it. Applies to
    // HH + 1-ply counter + cont2 + capture uniformly (bounds: MAX_HISTORY/CONT2_GRAVITY_DIV above).
    // Decomposed into two orthogonal knobs: SATURATION (bounded hist_update vs simple +=) and
    // MALUS (penalize searched-and-failed quiets/captures). gravity == SATURATION && MALUS.
    // SHIPPED 2026-07-30 as "gravcap" (+ ENABLE_CAPTURE_HIST + the recalibrated statScore constants):
    // +33.0 Elo over 1203 games (+506 -392 =305, 95% CI [+16.1, +50.1]).
    // ⚠️ Malus is a PRECONDITION, not a feature: without it history accumulates only bonuses, saturates and
    // stops discriminating -- which is why every history CONSUMER (capture hist, hist-prune, statScore-LMR)
    // measured null or inert before this landed. Set both off to recover the old bonus-only tree.
    inline bool ENABLE_HISTORY_SATURATION = true;
    inline bool ENABLE_HISTORY_MALUS = true;   // SHIPPED with SATURATION above (gravcap, +33.0 Elo)
    inline int MAX_HISTORY = 16384;     // gravity saturation bound (env-tunable for the sweep; bake the winner to constexpr for ship)
    inline int CONT2_GRAVITY_DIV = 4;   // 2-ply gravity down-weight divisor
    inline bool ENABLE_HISTORY_DECAY = true; // periodic >>=1 aging of history tables; off = saturation-only bounding (gravity tuning knob)
    inline int MALUS_DIV = 1;           // gravity malus softening: malus = bonus / MALUS_DIV (1 = symmetric, current)

    // Eval: scale the advanced-endgame mate-drive by the winner's material margin (default off =
    // byte-identical). Unproven (no definitive self-play result); the R+N-vs-R case it targeted is now
    // handled by is_practically_drawn. Kept as a knob for the graded endgame-scaling rework.
    inline bool ENABLE_MATE_DRIVE_SCALE = false;

    // Eval: recognize the lone Rook-Pawn KPvK draw inside is_practically_drawn (return 0 before the
    // material/piece_value_boost runs). The basic rook-pawn draw was absent from the existing KBP/KN
    // rook-pawn cases, so a drawn K+rook-pawn-vs-K read ~+4.9 (the chesscom-2200 conversion loss: the
    // engine traded rooks INTO this dead draw). Drawing rule = defender king reaches the promotion
    // corner no later than the pawn/attacker (chebyshev opposition), validated against a full KPvK
    // retrograde oracle (diagnostics/_kpk_oracle.py): zero won positions flagged drawn. SHIPPED
    // default-on (2026-06-27): position-fix verified (KPvK +4870->0, the chesscom-2200 rook trade gone),
    // byte-identical on WAC (252/70,150,573) and STS (1568) since it only touches rook-pawn KPvK. A
    // self-play tournament is uninformative here (self-play-invisible). ENABLE_RP_KPK_DRAW=0 reverts.
    inline bool ENABLE_RP_KPK_DRAW = true;

    // Eval: continuous endgame "convertibility" scale (default OFF -- reverted). Damps an unconvertible
    // material/placement lead toward draw (bare minor, opposite-coloured bishops). The 5 FEN spot-checks
    // looked surgical, but a scale-ON STS bench showed it changes far more leaf evals than they implied:
    // -3.9 STS / -3 WAC vs scale-off for only -3% nodes -- a net suite regression. Kept as a knob; redo
    // with tighter targeting (or after the bonus/malus history rework) before re-enabling.
    inline bool ENABLE_ENDGAME_SCALE = false;

    // Eval: per-term linear SCALE knobs (percent; 100 = byte-identical). Damp or boost an individual
    // positional eval term without touching material/PST, applied as total += SCALE_X * term / 100 at the
    // term's add site. Tuned via selfplay/tune_fit.py (against SF static / game result) and gated by
    // self-play SPRT. LATENT_THREAT and CENTRAL contribute only in the midgame path.
    inline int SCALE_PASSED_PAWN   = 100;
    inline int SCALE_LATENT_THREAT = 100;

    // SF11-style piece-on-piece STATIC threats (weak enemies attacked by minor/rook/pawn, hanging pieces) —
    // representation our king-directed latent_threat lacks. SHIPPED 2026-08-04 with STANDING_ONLY + the
    // per-target cap: +45.0 Elo (+171 -121 =96 of 388, LLR +3.035, SPRT H1 accepted vs the previous default).
    // ⚠️ The bench DISAGREED: STS 1746 -> 1685 (d10) and 1751 -> 1674 (d12), i.e. reproducibly negative in
    // BOTH regimes, while WAC rose 246 -> 250. The STS loss is concentrated in the sts_guard tier; the term
    // repairs our worst-scored positions and taxes our best. Games decide -- do not "fix" the STS drop.
    inline bool ENABLE_THREATS = true;
    inline int  SCALE_THREATS = 75;
    // Drop the currently-hanging bonus: MEASURED to be ~87% a subset of capture_gains (14 of 16 firings
    // co-occur, corr 0.60), which resolves the same en-prise facts properly via SEE -- and qsearch resolves
    // them a third time. Keep only the standing underdefended-piece pressure.
    inline bool THREATS_STANDING_ONLY = true;
    // Per-target contribution cap (millipawns, 0 = uncapped/byte-identical). threats_by sums an unbounded
    // stack of per-target bonuses (minor + rook + king + safe-pawn), so one misjudged target can dominate
    // the whole term; SF's threat terms each contribute within a bounded band instead. This bounds the
    // BLAST RADIUS of a single wrong target without touching which targets are detected.
    // SHIPPED at 800. Verified to be genuine STACK-bounding, not just a SAFE_PAWN cut: with the cap on,
    // THREAT_SAFE_PAWN=800 is byte-identical to 1600 (the cap already clamps it), yet NO uncapped
    // SAFE_PAWN reduction reaches the capped accuracy -- even SAFE_PAWN=400 uncapped reads 264.068 vs the
    // capped 263.920. So it also bounds minor+rook+king combinations. 0 = uncapped (pre-ship behaviour).
    inline int THREAT_PER_TARGET_CAP = 800;
    // Largest single per-target contribution in threats_by (next is 850). Every cap that helped is BELOW
    // it, so the cap's benefit may just be a SAFE_PAWN reduction in disguise. Knob to separate the two.
    inline int THREAT_SAFE_PAWN = 1600;
    // Tension gate on the threat term. The per-position profile of the capped term is a REDISTRIBUTION, not
    // a rescale: it repairs our worst decile (mean -28 win%^2) and taxes our best (+2.9), and the only tier
    // it worsens is sts_guard -- which is why the corpus improves while STS falls. Damp threats in QUIET
    // positions (g_capg_tension = pending SEE>=0 captures) instead of applying them everywhere.
    // Ramp: tension<=LO -> THREATS_QUIET_PCT, tension>=HI -> 100%. QUIET_PCT=100 = ungated/byte-identical.
    inline int THREATS_QUIET_PCT   = 100;
    inline int THREATS_TENSION_LO  = 0;
    inline int THREATS_TENSION_HI  = 3;
    inline int SCALE_CENTRAL       = 100;
    inline int SCALE_CAPTURE_GAINS = 100;

    // Position-conditioned capture_gains: slide the capg weight by tactical tension (g_capg_tension =
    // # of viable SEE>=0 captures pending). Default OFF = flat SCALE_CAPTURE_GAINS (byte-identical).
    // When on, capg_scale = ramp from CAPG_LO_SCALE (tension<=CAPG_TENSION_LO, quiet) to CAPG_HI_SCALE
    // (tension>=CAPG_TENSION_HI, tactical). Conservative: keep HI small so any real tension keeps capg high.
    // SHIPPED 2026-07-02 (condE): lightning SPRT ~+15 Elo (555-495-302, no-regression) + WAC +8 / STS +1.1% /
    // nodes -11.5% vs base. Set ENABLE_CAPG_COND=false to revert to flat SCALE_CAPTURE_GAINS (byte-id).
    inline bool ENABLE_CAPG_COND = true;
    inline int CAPG_TENSION_LO = 0;    // tension at/below which capg = CAPG_LO_SCALE (quiet floor)
    inline int CAPG_TENSION_HI = 3;    // tension at/above which capg = CAPG_HI_SCALE (tactical ceiling)
    inline int CAPG_LO_SCALE   = 10;   // capg weight (%) in confidently-quiet positions
    inline int CAPG_HI_SCALE   = 100;  // capg weight (%) in tactical positions
    inline bool ENABLE_CAPG_REALIZ = false; // discount the pre-booked capgains material when the gaining side can't
                                       // convert it, via the imbalance term's realizability_factor(material_edge,phase).
                                       // Off => byte-id; on with REALIZ_* still 0 => identity (needs REALIZ_MAT_K etc).

    // Eval: capture-gains legality/tempo awareness. approximate_capture_gains folds a full-magnitude,
    // pin-blind, tempo-blind SEE exchange into the static material/capture_gains terms, so it over-credits
    // illegal or unrealizable captures (the material-collapse over-read). Both default off => byte-identical.
    //  - ENABLE_CAPG_PIN: drop a gathered capture whose chosen attacker is ABSOLUTELY PINNED and the target
    //    is OFF the pin ray (an illegal capture SEE would otherwise count, e.g. a pinned bishop "winning" the
    //    enemy queen). Uses slider_blockers per king + BB_RAYS; targets the initial-capturer phantom.
    //  - ENABLE_CAPG_TEMPO: drop a credited capture whose attacker is itself attacked and cannot both survive
    //    and keep the threat (the opponent captures/forces it first). Targets the attacker-hanging tempo case.
    // CAPG_INVARIANT_ORDER: give the capture-stack sort a deterministic, COLOUR-RELATIVE tie-break.
    // Sorting on value_gained alone is mirror-invariant, but std::sort is not stable and there is no
    // tie-break, so equal-valued captures keep insertion (square) order -- which reverses under a
    // mirror. Since the consumer takes back(), the two orientations then simulate DIFFERENT capture
    // sequences: the base took a rook-takes-pawn (which earns the pawn-rank bonus) where the mirror
    // took a pawn-takes-pawn (which does not). Accounted for ~87% of ALL remaining colour-asymmetry
    // mass and every large violation, plus every large file-mirror violation.
    // Tie-break: least valuable attacker (MVV-LVA), then own-perspective square (sq for white,
    // sq^56 for black). Behavioral -> gated default-off.
    inline bool ENABLE_CAPG_INVARIANT_ORDER = true;   // SHIPPED 2026-08-08 in the 7-fix colour bundle
    // CAPG_LVA_STATIC: the capture-gains GATHER picks its attacker with get_least_valuable_attacker,
    // which ranks by square_values[] -- the EVAL MAGNITUDE on the square, not the piece's material
    // value. That is not colour-blind, so a target attacked by both a rook and a pawn resolves to
    // DIFFERENT attackers in mirrored positions, and the divergence cascades through the evasion logic.
    // _static ranks by true piece TYPE. Same correction ENABLE_SEE_FIX already made inside see();
    // the gather was never updated. Behavioral -> gated.
    inline bool ENABLE_CAPG_LVA_STATIC = false;
    // BISHOP_FWD_RANK: get_bishop_colour_complex_score's "forward" staging mask is built from the
    // SQUARE INDEX, not the rank, so for a staging square on e4 White's `~0ULL << 29` also includes
    // f4/g4/h4 (same rank, to the right) and excludes a4-d4. Wrong on its own terms, and not
    // file-mirror invariant.
    // ☠️ BUT IT IS CURRENTLY INERT -- MEASURED: 0 of 1200 positions change. The expensive
    // colour-complex path it lives in is unreachable at defaults, because ENABLE_CHEAP_BISHOP_COMPLEX
    // (default ON) returns earlier in the same function. So this is a real defect in DEAD code, and it
    // is NOT the residual file-mirror class -- that class (42/651, max 24 mp, pt_bishops) lives in the
    // CHEAP path and is still unlocated.
    // ⚠️ Keep the knob: the defect becomes live the moment the expensive path is re-enabled.
    inline bool ENABLE_BISHOP_FWD_RANK_FIX = false;
    // KING_ZONE_SYM: white/black_king_zones are indexed by the king's FILE, and the two MIDDLE entries
    // are not file mirrors of each other -- D leans queenside (BCDE) while E is centred (CDEF), where
    // flip(BCDE) is DEFG. So a king on D and its mirror-image king on E get differently shaped zones.
    // Worth 24 mp via CHEAP_BISHOP_KING (2 zone squares x 12): zeroing that knob drops file-mirror
    // violations 42 -> 13 and the worst case 24 -> 5 mp. Both repairs are symmetric, so balanced STS
    // decides. 0 = legacy (asymmetric) · 1 = LEAN (E -> DEFG, each middle file leans to its own side,
    // continuing the table's A/B/C-queenside F/G/H-kingside pattern) · 2 = CENTRED (D -> CDEF).
    inline int KING_ZONE_SYM_MODE = 0;
    // CAPG_EVADE_POLARITY: the evasion branches pop from opp_captures with `current_turn`, but that
    // stack belongs to the OTHER side (the sibling find_last_viable_capture right above uses
    // !current_turn). With the wrong polarity isValid fails for every entry and the helper -- which
    // pops unconditionally -- DRAINS THE WHOLE OPPONENT STACK. Turn-order dependent, hence asymmetric:
    // on 3q1rk1/8/5n1b/2n5/2b5/3P4/8/2R3K1 w, capture_gains reads -2175 base and EXACTLY 0 mirrored.
    // ⚠️ The in-code note says switching to !current_turn measured WORSE (asymmetry 82,264 -> 92,450).
    // RE-TEST rather than inherit that: it predates the ENABLE_CAPG_INVARIANT_ORDER tie-break defect in
    // this same function, which was corrupting the measurement it was based on. Behavioral -> gated.
    inline bool ENABLE_CAPG_EVADE_POLARITY_FIX = false;
    // KS_ROUND: the king-safety modulators scale a SIGNED, Black-positive `ks` with `>> 8`. An arithmetic
    // right shift rounds toward -inf, so (v)>>8 = floor(v/256) but (-v)>>8 = -ceil(v/256) -- the two
    // differ by ONE unit unless v divides 256 exactly. Since `ks` flips sign under a colour mirror, that
    // is a direct antisymmetry break. Integer DIVISION truncates toward zero and is antisymmetric.
    // ★ KING_SAFETY_MAG=3000 makes the final `MAG * ks / 100` equal 30*ks, so one unit of rounding is
    // EXACTLY 30 mp -- which is why every measured king_safety violation was exactly 30 mp (68 of 800
    // positions, mean 30.0, max 30). Behavioral -> gated default-off.
    inline bool ENABLE_KS_ROUND_FIX = true;           // SHIPPED 2026-08-08 in the 7-fix colour bundle
    // ROOK_ENEMY_RANKWIN: the midgame rook's ENEMY-pawn penalty uses non-mirrored rank windows -- White
    // fires on `> 4` (ranks 5-7, three ranks) but Black on `< 5` (ranks 0-4, FIVE ranks). White's window
    // mirrors to black {0,1,2} = `< 3`, so Black fires on two extra ranks, worth exactly
    // ROOK_ENEMY_PAWN_PEN = 50 mp -- precisely the constant 50 that every pt_rooks violation measured
    // (36 of 800 positions, mean 50.2, max 50). Sibling of ENABLE_ROOK_RANKWIN_FIX, which is the same
    // defect on the OWN-pawn window. Both directions are symmetric, so balanced STS picks the value:
    //   0 = legacy (asymmetric)   1 = widen White to match Black   2 = narrow Black to match White
    // SHIPPED 2026-08-08 at 2: balanced STS 3407 (mode 2, narrow Black) vs 3275 (mode 1, widen White).
    // Both are perfectly symmetric, so the 132-point spread is purely the MAGNITUDE choice.
    inline int ROOK_ENEMY_RANKWIN_MODE = 2;
    inline bool ENABLE_CAPG_PIN   = true;
    inline bool ENABLE_CAPG_TEMPO = false;

    // Eval: passed-pawn scoring magnitudes inside getPPIncrement (absolute increments, defaults = the
    // original literals = byte-identical). Finer than SCALE_PASSED_PAWN — these tune the SHAPE of the
    // passer bonus (penalty for a defended path, blockade, un-impeded run, diagonal/file/horizontal
    // support) rather than a single multiplier. Move-match tuning targets, gated by SPRT.
    inline int PP_OPP_PAWN_PEN  = 125;  // per enemy pawn defending the promotion path
    inline int PP_BLOCKADE_PEN  = 100;  // enemy non-pawn blocker directly in front
    inline int PP_UNBLOCKED     = 50;   // no blocker in front (clear run)
    inline int PP_DIAG_SUPPORT  = 75;   // friendly pawn supporting diagonally from behind
    inline int PP_FILE_CLEAR    = 150;  // supported AND the neighbouring file is clear of enemy pawns
    inline int PP_HORIZ_SUPPORT = 225;  // friendly pawn alongside on the same rank

    // Pawn-majority / candidate-passer bonus: a wing pawn majority (more pawns on the queenside or
    // kingside than the opponent) can force a passed pawn before one exists -- the engine otherwise
    // values pawns only once ACTUALLY passed, under-reading won pawn-up positions. Per surplus pawn.
    // Phase-aware: _MG weights the majority in the midgame/early-endgame (a structural asset that
    // foreshadows the endgame -- shapes trade-down decisions), _EG its value in the late endgame
    // (active conversion); blended linearly by phase_score so the two regimes tune independently.
    // Both 0 = off = byte-identical; PACE-tuned (/eval-tune).
    inline int PAWN_MAJORITY_MAG_MG = 0;
    inline int PAWN_MAJORITY_MAG_EG = 0;
    // Modulators on the per-wing majority bonus (all 0 = flat base = neutral). Only active when the
    // base MAG is on, so all-default stays byte-identical. PACE finds the constants jointly.
    inline int PAWN_MAJORITY_ADV_K      = 0;  // + per (most-advanced own pawn rank x surplus): a rolling majority is worth more
    inline int PAWN_MAJORITY_OUTSIDE_K  = 0;  // + per surplus when the majority wing is opposite the enemy king (outside passer)
    inline int PAWN_MAJORITY_BLOCKADE_K = 0;  // - per (enemy knight/bishop x surplus): a minor can blockade the would-be passer

    // Pawn-structure weaknesses (Black-positive units; 0 = off = byte-identical). ISOLATED = no friendly pawn
    // on either adjacent file. BACKWARD = adjacent friendly pawns exist but are all more advanced (none at/
    // behind this rank) AND the stop square is attacked by an enemy pawn (can't advance safely). Texel/PACE.
    inline int ISOLATED_PAWN_PEN = 0;
    inline int BACKWARD_PAWN_PEN = 0;

    // Minor-piece OUTPOSTS (Black-positive units; 0 = off = byte-identical). A knight/bishop in the enemy half,
    // defended by an own pawn, that can never be attacked by an enemy pawn (no enemy pawn on an adjacent file
    // able to advance onto it). Knight outposts usually worth more than bishop. Texel/PACE-tuned.
    inline int OUTPOST_KNIGHT = 0;
    inline int OUTPOST_BISHOP = 0;

    // SPACE: rank-progressive control of the enemy half by PAWNS and KNIGHTS (the piece types whose advanced
    // control is under-scored — rook/bishop forward reach is already in the cheap-mobility surrogates, centre is
    // in central/PSQT). Credits the improving quiet advance (pawn push, knight-to-outpost) a flat PSQT misses.
    // Milli-pawns per rank-weighted controlled square (rank5/6/7 = weight 1/2/3 for White, mirror for Black).
    // 0 = off = byte-identical. SPACE is a MIDGAME concept — apply only while phase_score < SPACE_PHASE_MAX
    // (low phase = midgame; see phase_score convention). Firing space in the endgame poisons endgame themes
    // (King Activity, Recapturing) where controlled advanced squares are meaningless.
    inline int SPACE_MAG = 0;          // PAWN enemy-half control magnitude (the clean signal)
    inline int SPACE_KNIGHT_MAG = 0;   // KNIGHT enemy-half control magnitude (noisier; separate so it can be 0)
    inline int SPACE_PHASE_MAX = 40;

    // Eval: placement (piece-square) MAGNITUDE scales (percent; 100 = byte-identical). Whole-map per-piece
    // multiplier applied at every read of that piece's placement table — white read, black read, AND the
    // central-score feed — so colour symmetry and the central term stay consistent. ROOK PST is dead code
    // (never read) so has no knob; KING placement is endgame-only. Gap-tracking's #1 finding (midgame
    // placement over-optimism) tunes these DOWNWARD. PACE move-match targets, SPRT-gated.
    inline int SCALE_PLACE_PAWN    = 100;
    inline int SCALE_PLACE_KNIGHT  = 100;
    inline int SCALE_PLACE_BISHOP  = 100;
    inline int SCALE_PLACE_QUEEN   = 100;
    inline int SCALE_PLACE_KING_EG = 100;  // king placement is endgame-only
    // Eval: central-occupation bonus multipliers in update_global_central_scores (percent; defaults
    // reproduce the original inner ×2 / outer ×1.5 exactly — base * MULT / 100 == trunc(base * MULT/100)).
    inline int CENTER_INNER_MULT = 200;  // inner center (d4/e4/d5/e5)
    inline int CENTER_OUTER_MULT = 150;  // extended center

    // Eval: scalar positional-term magnitudes (absolute, defaults = the original literals = byte-identical).
    // The knob IS the value (no mul/div in the hot path; sign applied at the site).
    inline int IMBALANCE_SCALE   = 3;    // offense-vs-defense imbalance multiplier (was ×3)
    inline int BISHOP_PAIR_BONUS = 300;  // magnitude of the bishop-pair bonus
    inline int KNIGHT_PAIR_BONUS = 200;  // magnitude of the knight-pair bonus

    // Kaufman quadratic material-imbalance term (public Kaufman 1999 MODEL; coefficients FIT from OUR data by
    // ridge regression of the SF11-eval residual onto piece-count products -- diagnostics/kaufman_fit.py, NOT
    // copied from SF). A once-per-eval scalar that re-prices material by the whole piece census (bishop pair,
    // knight-loves-pawns, rook redundancy, pawns-vs-minor). When on, the flat BISHOP/KNIGHT_PAIR_BONUS block is
    // SKIPPED (Kaufman owns material-combo -- no double-count). Default off = byte-identical. KAUFMAN_SCALE =
    // post-fit global trim (%). Our offense/defense "imbalance" term is a different axis, left untouched.
    inline bool ENABLE_KAUFMAN_IMBALANCE = true;
    inline int  KAUFMAN_SCALE = 100;

    // Eval: rook open-file / placement magnitudes inside evaluate_rooks_midgame (absolute, defaults =
    // the original literals = byte-identical). The knob IS the value (sign applied at the site; no hot-path
    // division — the per-passer-rank multipliers stay constant multiplies). Each knob drives BOTH the white
    // and the mirrored black site so colour symmetry holds. Tunes the SHAPE of the rook open-file term.
    inline int ROOK_OPEN_BASE        = 250;  // base rookIncrement before file scan
    inline int ROOK_OPEN_CAP         = 300;  // std::min clamp on rookIncrement
    inline int ROOK_7TH              = 150;  // rook on the 7th (white) / 2nd (black) rank
    inline int ROOK_CONNECTED        = 150;  // rooks connected on the rank
    inline int ROOK_SEMI             = 125;  // rooks doubled on the file
    inline int ROOK_PASSER_OWN       = 50;   // per-rank bonus for own passed pawn on the file
    inline int ROOK_PASSER_ENEMY     = 25;   // per-rank bonus for enemy passed pawn on the file
    inline int ROOK_OWN_PAWN_BASE    = 50;   // base penalty for own (non-passed) pawn blocking the file in own half
    inline int ROOK_OWN_PAWN_RAMP    = 125;  // per-rank ramp on that own-pawn penalty
    inline int ROOK_ENEMY_PAWN_PEN   = 50;   // penalty for enemy (non-passed) pawn blocking the file in enemy half
    inline int ROOK_MINOR_BLOCK      = 15;   // penalty for a knight/bishop blocking the file
    inline int ROOK_ROOK_BLOCK       = 35;   // penalty for an enemy rook blocking the file
    inline int ROOK_SEMI_CONNECTED   = 125;  // x-ray (semi-connected) rook bonus

    // Tension-conditioned rook-file boost (position-conditional; capg template). The midgame rook open-file/7th/
    // connected edge is worth MORE in quiet positions and should stay at default in tactical ones (a flat boost
    // regresses Recapturing — the Pareto trade-off found 2026-07-02). Reuses g_capg_tension: rescale the net
    // rook-file bonus by ROOK_COND_QUIET_SCALE% when tension<=LO (quiet), sliding to 100% at tension>=HI. Off =
    // byte-identical.
    inline bool ENABLE_ROOK_TENSION_COND = false;
    inline int ROOK_COND_TENSION_LO    = 0;
    inline int ROOK_COND_TENSION_HI    = 3;
    inline int ROOK_COND_QUIET_SCALE   = 150;  // rook-file weight (%) in fully-quiet positions

    // Eval: pawn-structure table MAGNITUDE scales (percent; 100 = byte-identical). Whole-table multipliers
    // baked into the working arrays once at init by rebuild_scaled_pawn_tables() (no per-read division). The
    // rank tables drive advancement value; wall/chain drive structural cohesion. PACE move-match targets.
    inline int SCALE_PAWN_RANK    = 100;  // default_midgame_pawn_rank_bonus (non-passed advancement)
    inline int SCALE_PASSED_RANK  = 100;  // passed_midgame_pawn_rank_bonus (passed-pawn advancement, midgame)
    inline int SCALE_ENDGAME_RANK = 100;  // endgame_pawn_rank_bonus (advancement, endgame)
    inline int SCALE_PAWN_WALL    = 100;  // pawn_wall_file_bonus (pawn-shield / wall cohesion)
    inline int SCALE_PAWN_CHAIN   = 100;  // pawn_chain_file_bonus (diagonal chain support)

    // Eval: GRADED obstruction selector for the pawn rank tables. Both pawn evaluators pick between the
    // ordinary and passed rank tables with a single threshold on ppIncrement, discarding the fact that
    // ppIncrement is already a continuous 0..cap obstruction score. Measured against SF18, that boolean is
    // misplaced: a pawn whose only stopper sits on an adjacent file ahead scores ppIncrement 75 -- just
    // under the midgame threshold -- yet is worth as much as a true passer (+258..+329 vs +240..+273 at the
    // 6th rank), and SF15.1 flags exactly that shape as passed. Blending interpolates between the two
    // tables over [LO, HI] instead. Off => the original threshold, byte-identical.
    // Eval: PER-RANK percentage scales on the three pawn rank tables, applied on top of the whole-table
    // SCALE_* knobs in rebuild_scaled_pawn_tables(). The tables are hand-picked; a single whole-table
    // multiplier can only rescale that hand-picked SHAPE, never derive a different one. These let the
    // win%-error tuner choose the entries themselves, jointly with every other eval term, which is the
    // only way the curve's shape becomes data-driven rather than assumed.
    // Only indices 1..6 (ranks 2..7) are reachable: index 0 is the pawn's own back rank and index 7 the
    // promotion square, neither of which a pawn can occupy. All default 100 => byte-identical.
    inline std::array<int, 8> RANK_DEF_PCT = {100, 100, 100, 100, 100, 100, 100, 100};  // default_midgame_pawn_rank_bonus
    inline std::array<int, 8> RANK_PSD_PCT = {100, 100, 100, 100, 100, 100, 100, 100};  // passed_midgame_pawn_rank_bonus
    inline std::array<int, 8> RANK_EG_PCT  = {100, 100, 100, 100, 100, 100, 100, 100};  // endgame_pawn_rank_bonus

    // PER-FILE percentage scales on the chain and wall tables, same rationale as the per-rank knobs above:
    // the base values (chain A 10 / B 15 / C 100 / D 150 / E 150 / F 100 / G 15 / H 10) are hand-picked and
    // only a whole-table multiplier existed, so the file SHAPE has never been fitted to anything. Chain is
    // indexed by file 0..7; wall is indexed [x] and [x+2] so its live entries are 1..8. Default 100 => same.
    inline std::array<int, 8>  CHAIN_F_PCT = {100, 100, 100, 100, 100, 100, 100, 100};
    inline std::array<int, 11> WALL_F_PCT  = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};

    // PER-PAWN BONUS CAPS, made tunable. `min(225, structural + positional)` midgame and
    // `min(175, structural)` endgame were fixed literals chosen to stop central pawns growing until two of
    // them equalled a minor piece. Measured: the midgame cap binds for ~36% of pawns (corpus-invariant) and
    // the endgame cap barely binds at all. Whether those are the right heights is a question for the
    // win%-descent, not for judgement -- so they become knobs rather than constants.
    inline int PAWN_CLAMP_MID = 225;
    inline int PAWN_CLAMP_EG  = 175;

    // ENDGAME structural magnitudes. These were HARDCODED literals in evaluate_pawns_endgame and no knob
    // reached them -- SCALE_PAWN_WALL/_CHAIN only rebuild the file tables that the MIDGAME path reads. So
    // endgame pawn structure was flat in file, flat in rank, and had never been fitted even once, while
    // SF's endgame connected term is its most rank-sensitive component. Defaults reproduce the literals.
    inline int EG_PHALANX = 100;   // same-rank neighbour (SF's phalanx)
    inline int EG_SUPPORT = 135;   // diagonally-behind supporter (SF's support)
    inline int EG_DEFEND  = 115;   // this pawn defends another pawn
    inline int EG_LATENT  = 50;    // potential (not yet realised) support

    // ENDGAME piece-EXISTENCE bonuses. Each endgame piece evaluator adds a flat amount on top of
    // values[] simply for the piece being on the board, restating what a piece is worth once the
    // position simplifies. They were hardcoded literals reachable by no knob, so the endgame piece
    // scale has never been fitted. Measured motivation: on real positions our minor/pawn exchange
    // rate is ~3.4-3.6 where SF11, SF15.1-classical and SF18 all sit at 3.9-4.1, and the per-removal
    // error is fat-tailed rather than a uniform offset. Defaults reproduce the literals exactly.
    inline int EG_EXIST_KNIGHT = 200;
    inline int EG_EXIST_BISHOP = 250;
    inline int EG_EXIST_ROOK   = 350;
    inline int EG_EXIST_QUEEN  = 900;

    // MIDGAME per-piece accumulator clamps, previously fixed literals. The bishop path clamps TWICE,
    // 150 apart, straddling the colour-complex term -- a stacked clamp, which makes any knob feeding
    // it step-shaped (plateau then discontinuity) rather than smooth. Exposed so the descent can see
    // the steps instead of fighting them.
    inline int MG_CLAMP_KNIGHT   = 3750;
    inline int MG_CLAMP_BISHOP_A = 3850;   // before get_bishop_colour_complex_score
    inline int MG_CLAMP_BISHOP_B = 4000;   // after it

    // ENDGAME per-piece accumulator clamps. These DO NOT EXIST today: only the knight and bishop
    // MIDGAME paths are clamped, so every endgame piece score is unbounded while its midgame twin is
    // capped. That asymmetry is a live suspect for endgame divergence -- the identical asymmetry was
    // found and closed for pawns this session (PAWN_CLAMP_MID existed, PAWN_CLAMP_EG did not).
    // 0 = disabled, which is the default, so the engine stays byte-identical until a value is set.
    inline int EG_CLAMP_KNIGHT = 0;
    inline int EG_CLAMP_BISHOP = 0;
    inline int EG_CLAMP_ROOK   = 0;
    inline int EG_CLAMP_QUEEN  = 0;

    // ---------------------------------------------------------------------------------------------
    // WINNABILITY. A second-order correction asking "can this nominal advantage actually be converted?"
    // Verified ABSENT from our eval and our entire commit history; both reference lineages have one.
    // SF1.1 had nothing -> SF11 `initiative()` -> SF15.1 renamed it `winnable()`, which is the semantic
    // point: it never measured initiative in the chess sense, and not one of its inputs is an attacking
    // signal. Ethereal carries the same clamp shape with a simpler, endgame-only feature set.
    //
    // Form ported, constants ours. `complexity` is a linear combination of convertibility features; the
    // result is applied ONCE to the summed total, sign-preserving and bounded by |total| so it can drive
    // a score to zero but never flip who is winning. SF damps midgame-only (its upper bound is literally
    // 0) and allows a two-way endgame adjustment; we hold one blended total rather than an (mg,eg) pair,
    // so that split becomes a phase blend of the two forms -- the boost half vanishes toward the opening,
    // preserving the restraint that NEITHER reference engine lets this term boost in the middlegame.
    inline bool ENABLE_WINNABILITY = false;
    inline int WINNAB_PASSED      = 9;    // passed pawns: a concrete conversion mechanism
    inline int WINNAB_PAWNS       = 12;   // total pawns: material to make a passer from
    inline int WINNAB_OUTFLANK    = 9;    // king file distance + SIGNED rank difference (SF15.1's form)
    inline int WINNAB_FLANKS      = 21;   // pawns on both wings: two fronts to attack
    inline int WINNAB_INFILT      = 24;   // a king already advanced into enemy territory
    inline int WINNAB_NO_NPM      = 51;   // pure pawn endgame: sharply more convertible
    inline int WINNAB_UNWINNABLE  = 43;   // subtracted when outflanking < 0 and pawns on one wing only
    inline int WINNAB_TENSION     = 0;    // OUR signal, no SF/Ethereal analogue: g_capg_tension. Opt-in.
    inline int WINNAB_BASE        = 110;  // subtracted; makes complexity negative in simplified positions
    inline int WINNAB_MG_OFFSET   = 50;   // SF's `complexity + 50` on the damp-only branch
    inline int WINNAB_SCALE       = 10;   // complexity units -> millipawns (our pawn = 1000, SF's ~126)

    // ---------------------------------------------------------------------------------------------
    // CLOSEDNESS. Ethereal's `evaluateClosedness`, which Stockfish has no analogue for: a pawn-structure
    // index that shifts knight and rook value relative to each other. This is a Kaufman-family imbalance
    // term (which piece is better here?), NOT a winnability term (is the advantage convertible?) -- a
    // damp on the total moves pawns and minors together and so cannot correct a piece/pawn RATIO.
    // Motivation from our own data: minors sit ~7% below the classical consensus on the piece/pawn
    // exchange rate while rooks are exact, and the knight's closed-vs-open win% error spread is 3.9
    // against the bishop's 1.2 -- a knight keeps its eight target squares in a closed position, so
    // nothing in our eval notices it is BETTER there.
    inline bool ENABLE_CLOSEDNESS = false;
    // Indexed by closedness 0 (wide open) .. 8 (fully closed). Millipawns per net piece. All-zero =
    // inert even with the gate on, so the descent supplies the shape rather than inheriting Ethereal's.
    inline int CLOSED_N[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    inline int CLOSED_R[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // 🚨 Bishops deliberately get NO additive table -- they would double-count. Bishops are ALREADY
    // closedness-sensitive through get_bishop_colour_complex_score, which subtracts `block` (own pawns
    // on the bishop's colour) and pays `mob` (real diagonal scope, which collapses when the position
    // closes). This modulates that existing term instead, keeping it the single payer for the signal.
    // Percent, 100 = unchanged.
    inline int CLOSED_B_PCT[9] = {100, 100, 100, 100, 100, 100, 100, 100, 100};

    // ---------------------------------------------------------------------------------------------
    // PHASE BLEND POINTS. 🐛 The blend does not complete. `PHASE_BLEND_RANGE` of 30 implies a 40->70
    // ramp, but the blend is only reached inside `!isEndGame` (phase_score <= 64), so the endgame weight
    // tops out at 24/30 = 80% and then JUMPS to 100% when isEndGame flips -- a ~20% step of
    // (result_end - result_mid) at the phase boundary, on every blended piece type. Setting RANGE to 24
    // closes the step. Defaults reproduce the current behaviour exactly.
    inline int PHASE_BLEND_LO    = 40;
    inline int PHASE_BLEND_RANGE = 30;

    // RANK sensitivity for the structural bonuses, per phase. Ours are file-indexed with NO rank term at
    // all; SF's `Connected[r]` is rank-indexed and its endgame component scales `v*(r-2)/4`, a 4x swing.
    // Because we have two separate evaluators the two phases can carry entirely different curves -- SF must
    // encode an (mg,eg) pair from one formula, we do not. Indices are ranks 1..6 (2..7); 100 => unchanged.
    inline std::array<int, 8> STRUCT_R_MG_PCT = {100, 100, 100, 100, 100, 100, 100, 100};
    inline std::array<int, 8> STRUCT_R_EG_PCT = {100, 100, 100, 100, 100, 100, 100, 100};

    // OPPOSED modulation. SF scales the connected bonus by `(2 + phalanx - opposed)`; we had no `opposed`
    // signal exposed at all, even though getPPIncrement computes it and throws it away via an early return.
    // `opposed` = an enemy pawn anywhere ahead on OUR OWN file. Note opposed is a STRICT SUBSET of
    // "not passed": a pawn contested only on an adjacent file is not passed, but is unopposed.
    inline int STRUCT_OPPOSED_MG_PCT = 100;
    inline int STRUCT_OPPOSED_EG_PCT = 100;

    // Defer the inline rank bonus for every FLAGGED passer, instead of for every pawn above the ppIncrement
    // threshold. Deferring discards the value (g_passer_*_deferred are written and never read) because
    // evaluate_passers independently pays each pawn in the passed bitboard, so the two populations must
    // coincide -- and in the endgame they do not. There the threshold is 300, but a flagged passer's
    // ppIncrement is 200 - blockade + 50(clear) + 75*diag + 150(file clear) + 225*horiz, so an UNSUPPORTED
    // passer caps at 250 and always keeps the inline base while a SUPPORTED one reaches 475 and loses it:
    // the base is granted to weak passers and withheld from strong ones. Measured on a lone blockaded
    // endgame passer: ~12 cp inline PLUS 43.5 cp from evaluate_passers, for the same pawn.
    // ⚠️ Also the precondition for tuning the PP_* constants: `ppInc >= 100 <=> flagged` holds in the
    // midgame only because base 200 - PP_BLOCKADE_PEN 100 = exactly 100. Move that constant and midgame
    // passers start double-paying too. Behaviour change, so gated; default off = today's engine.
    // PASSED-PAWN SUPPORT magnitudes (boost_pieces_for_supporting_passed_pawns). Every one of these was a
    // hardcoded literal scaled only by the whole-function SCALE_PASSED_PAWN, so the SHAPE of the
    // path-occupancy/path-control trade has never been fitted. Per promotion-path square ahead of a passer:
    // an own piece standing there, an enemy piece standing there, and — for empty squares — each own or
    // enemy attacker of it. All multiplied by rank and halved unless the passer is advanced in an endgame.
    // Defaults reproduce the literals exactly.
    inline int PPS_OWN_BLOCK    = 75;   // own piece occupying a path square (was y * 75)
    inline int PPS_ENEMY_BLOCK  = 100;  // enemy piece occupying a path square (was y * 100)
    inline int PPS_OWN_ATTACK   = 60;   // own attacker of an empty path square (was y * 60)
    inline int PPS_ENEMY_ATTACK = 50;   // enemy attacker of an empty path square (was y * 50)

    inline bool ENABLE_PASSER_DEFER_ON_FLAG = false;

    // Upper bound of the realizability clamp inside passer_realizability_R. PASSER_R_CAP could never exceed
    // this, which is why raising the cap alone was found inert -- and the joint descent pinned the cap at
    // exactly 384, i.e. it wanted more upside and the architecture refused. Raise both together to give it
    // room. Default 384 => byte-identical.
    inline int PASSER_R_MAX = 384;

    inline bool ENABLE_PAWN_OBSTRUCTION_BLEND = false;
    inline int PAWN_OBS_LO        = 50;   // ppIncrement at/below which the ORDINARY table is used outright (midgame)
    inline int PAWN_OBS_HI        = 250;  // ppIncrement at/above which the PASSED table is used outright (midgame)
    inline int PAWN_OBS_LO_EG     = 150;  // endgame twin; that path runs on a wider ppIncrement scale (cap 600,
    inline int PAWN_OBS_HI_EG     = 450;  // legacy threshold 300) so it carries its own span rather than sharing.

    // Eval: latent bishop-activity increments in get_latent_bishop_activity_score (absolute, defaults =
    // the original literals = byte-identical; the knob IS the value, no division). Reward for a bishop's
    // blocked-diagonal reach onto an enemy pawn and its second-order diagonal scope.
    inline int BISHOP_MOB_PAWN_ATTACK = 15;  // reachable square that attacks an enemy pawn
    inline int BISHOP_MOB_SECONDARY   = 5;   // per second-order diagonal square reachable after simulation

    // Eval: latent king-zone threat magnitudes in get_latent_threat_score (absolute, defaults = the
    // original literals = byte-identical; the knob IS the value). The two MULT bases are the per-excess
    // attacked-square / attacker weights; the per-piece values are the presence-attacker increments
    // (an enemy piece standing in our king zone). Each knob drives BOTH king zones (colour-symmetric).
    inline int THREAT_ATTACK_MULT   = 50;  // base white/black_zone_attack_increment (per excess attacked square)
    inline int THREAT_PRESENCE_MULT = 80;  // base white/black_zone_presence_increment (per excess attacker)
    inline int THREAT_PAWN   = 10;  // enemy pawn in our king zone
    inline int THREAT_KNIGHT = 7;   // enemy knight in our king zone
    inline int THREAT_BISHOP = 7;   // enemy bishop in our king zone
    inline int THREAT_ROOK   = 3;   // enemy rook in our king zone
    inline int THREAT_QUEEN  = 7;   // enemy queen in our king zone

    // Eval: uniform king-zone attack-layer MAGNITUDE scale (percent; 100 = byte-identical). Applied to BOTH
    // king-zone maps equally at the source inside setAttackingLayer (after the raw layer is assembled, before
    // the eval reads it), gated on default so the ~96 attackingLayer reads and the raw caches are untouched
    // and zero per-read division is added. A single shared scale (not per-layer) preserves eval colour
    // symmetry: the two layers swap under a colour mirror, so they must scale together. Tunes overall
    // king-safety / attacking weight vs material+position.
    inline int SCALE_ATTACK_LAYER = 100;

    // Attack fan-out SHAPE: the open-square boost multiplier inside setAttackingLayer (how much an OPEN
    // square in the king 2-ring is amplified vs a closed one). Default 5 = byte-identical. Tunes the
    // DISTRIBUTION of the attack fan-out (which SCALE_ATTACK_LAYER's uniform scale cannot reshape).
    inline int ATTACK_OPEN_MULT = 5;

    // Eval: ATTACK-UNIT KING SAFETY (king_safety_score) — the canonical pre-NNUE model that replaces the
    // crude flat-increment get_latent_threat_score. Per king, enemy pressure is accumulated as ATTACK
    // UNITS and mapped through a PRECOMPUTED non-linear safety table (danger = clamp(units)^2 / KS_DIVISOR),
    // so two attackers are worth far more than twice one (additive pressure). king_safety = danger(white
    // king) - danger(black king) (Black-positive), PHASE-TAPERED to ~0 by the endgame. The master knob
    // KING_SAFETY_MAG defaults 0 => the term is gated off at the call site => byte-identical; the component
    // knobs only take effect once MAG > 0. Each component is additive into `units`, so any sub-knob at 0
    // disables just that component (lets us build/tune one at a time; lets PACE/Texel tune them jointly).
    inline int KING_SAFETY_MAG = 3000;  // master percent scale (0 = off = byte-identical)
    // REPLACE the flat latent_threat with king_safety_score (the structural swap, not an additive run-beside).
    // Off (default) = byte-identical: latent_threat adds as today, king_safety only if MAG>0. On = skip the
    // latent_threat add entirely and route king danger through king_safety_score (no double-count); needs
    // KING_SAFETY_MAG>0 to do anything. Phase A finds the neutral MAG where the swap is ~0 regression.
    inline bool ENABLE_KS_REPLACE_LT = true;
    // CONSOLIDATE the flat, un-realizability-conditioned pawn-shelter bonus (the +185/+75 constants in
    // evaluate_kings_midgame) out of the `pieces` term so shelter is scored ONCE, through the realizability-
    // conditioned KS_SHIELD inside king_safety_danger, instead of a flat placement constant. Off (default) =
    // byte-identical: the +185/+75 add as today. On (paired with the KS term) = the flat constants are gated to 0,
    // moving shelter credit to the conditioned KS site (no double-count). The attack-layer-derived (baseIncrement)
    // shield credit + the O/D accumulators are untouched (the placement/pressure lens stays intact).
    inline bool KS_CONSOLIDATE = false;
    // Consolidated king-safety HOME master gate (supersedes KS_CONSOLIDATE). Off (default) = byte-identical:
    // the flat 185/75 shelter constants add as today and the realizability gate is inert. On = the shelter
    // constants become the tunable KS_SHELTER_* terms (identity at 185/75/100) and MOD_KS_REALIZ may damp the
    // whole unit-KS danger budget. The unit-KS budget already lives in ONE term (king_safety_score, wrapped by
    // evaluate_king_safety); this gate makes the surrounding shelter credit tunable + adds the whole-budget gate.
    // The attackingLayer placement lens and the long-term OvD lens are untouched (scope: gate the storm lens only).
    inline bool ENABLE_KS_V2 = false;
    inline int  KS_SHELTER_FULL    = 185;  // tunable full-shield shelter bonus (replaces the flat 185 under KS_V2)
    inline int  KS_SHELTER_PARTIAL = 75;   // tunable partial-shield shelter bonus (replaces the flat 75 under KS_V2)
    inline int  KS_SHELTER_MAG     = 100;  // percent scale on the re-homed shelter terms (100 = identity)
    inline bool ENABLE_KS_DEBUG = false; // DIAGNOSTIC: dump king_safety_score zone sub-parts + MOD signals per eval. Off = byte-id.
    inline int KS_LIGHT_MAG    = 0;     // LIGHT-eval king-pressure surrogate scale (g_eval_light path; 0 = off = byte-id)
    inline int KS_ATT_KNIGHT   = 2;     // attack units per enemy knight bearing on the king zone
    inline int KS_ATT_BISHOP   = 2;     // per enemy bishop
    inline int KS_ATT_ROOK     = 3;     // per enemy rook
    inline int KS_ATT_QUEEN    = 5;     // per enemy queen
    // Latent king-AIM detection (Front A): our line-of-sight attacker scan reads ZERO for an enemy slider
    // whose only obstruction sits OUTSIDE the narrow king zone (occupancy-limited attack_bitmasks never x-ray).
    // This catches an enemy slider ALIGNED with the king through EXACTLY ONE blocker (the discovered/latent
    // line) -- reuses the empty-board king rays (as slider_blockers does) + betweenPieces, iterating only the
    // 0-3 sliders on the king's lines (no new per-piece loop). Our own obstruction-graded concept (heir to
    // latent_threat), not SF's queen-only x-ray. Default off (ENABLE_KS_AIM=false) = byte-identical. Weights
    // are per aiming-piece type, tuned DOWN vs a direct attacker (aim is latent, not yet delivered).
    inline bool ENABLE_KS_AIM  = false;
    inline int KS_AIM_BISHOP   = 1;     // aim units per enemy bishop aligned through one blocker
    inline int KS_AIM_ROOK     = 2;     // per enemy rook
    inline int KS_AIM_QUEEN    = 3;     // per enemy queen
    inline int KS_ATTACK_COUNT = 1;     // per zone square the enemy attacks (additive zone pressure)
    // Ethereal-style DISCRIMINATION gate: king danger is scored only when at least this many ENEMY PIECES
    // (N/B/R/Q) attack the king zone (a single piece near the king is not danger; a coordinated group is). When
    // the enemy has a queen the threshold drops by one (a lone queen still threatens). 0 = gate OFF = byte-
    // identical. This is what lets KS_FLOOR come down without waking calm positions (0-1 attackers -> 0 danger).
    inline int KS_MIN_ATTACKERS = 0;
    // Optional SF-style attacker COORDINATION product added to units: KS_ATT_PRODUCT * attackerCount *
    // attackerWeightSum >> 4 (super-linear in the number of coordinating attackers). 0 = OFF = byte-identical.
    // Default off; the gate above is the LEADING (gentler, N^2) discriminator. Kept as a fit-testable variant.
    inline int KS_ATT_PRODUCT  = 0;
    // Per-zone-square OVERLOAD (the DISCRIMINATIVE coordination signal the blanket product lacked): sum over zone
    // squares of max(0, #enemy-attackers - #own-defenders). A breakthrough square (e.g. attacked by N+R+Q,
    // defended by 1 pawn = +2) fires; a properly-defended king (attackers <= defenders everywhere) stays ~0.
    // Subtracts defenders per-square, so it does NOT over-fire on safe-but-crowded kings. 0 = OFF = byte-identical.
    inline int KS_OVERLOAD     = 0;
    inline int KS_WEAK         = 2;     // per weak zone square. Baseline: enemy-attacked AND no own defender. With
                                        // ENABLE_KS_SF_WEAK: enemy-attacked AND under-defended (<=1 defender, K/Q only).
    inline int KS_SAFE_CHECK   = 3;     // per safe-check square vs the ENEMY (offensive) king. Default 3.
    inline bool ENABLE_KS_CHECK_V2 = false;  // per-type safe-check weighting (replaces the flat KS_SAFE_CHECK*count).
                                        // Off => byte-identical flat behavior. On: units += per-type weights below.
    inline int KS_CHK_QUEEN    = 14;    // per queen safe-check (used only when ENABLE_KS_CHECK_V2): sized so a LONE
    inline int KS_CHK_ROOK     = 14;    // queen/rook safe-check clears KS_FLOOR (13) on its own merit (real danger,
    inline int KS_CHK_BISHOP   = 7;     // not calm-king noise) -- the floor stays; targeting is emergent, branchless.
    inline int KS_CHK_KNIGHT   = 9;
    inline int KS_CHK_MULTI    = 0;     // graded bump when a check TYPE has a SECOND safe square (SF15's
                                        // more_than_one level); saturates there. Per type, not per square. 0 = off.
    inline int KS_SAFE_CHECK_DEF = 5;   // per safe-check square vs the SIDE-TO-MOVE's OWN (defensive) king. Boosts
                                        // DEFENSIVE safe-check sensitivity (lifts a real counter-attack on our king
                                        // over the KS_FLOOR deadzone) WITHOUT over-crediting our own attacks (which
                                        // hurt WAC). Categorical game gate: KS-attack collapse class -23% at flat
                                        // total (600g SF@2400). Set to KS_SAFE_CHECK (3) to restore prior symmetric.
    inline int KS_DEF_MAG      = 100;   // percent multiplier on the SIDE-TO-MOVE king's FINAL danger ("carry
                                        // our-king danger harder"): our king-danger magnitude is ~10x under SF11's
                                        // (SF weights a safe check 635-1080 units, quadratic), so a real counter-
                                        // attack doesn't offset our capgains-hot material. 100 = byte-identical;
                                        // >100 scales up the defensive side only (offense stays put -- attackingLayer
                                        // already covers it). Games-tuned; conditional (0-danger kings unaffected).
    inline int KS_STORM        = 1;     // per rank of enemy pawn-storm advance on the king's three files
    inline int KS_OPEN_FILE    = 2;     // per open/semi-open file on/adjacent to the king file
    inline int KS_BATTERY      = 3;     // per rook/queen battery (doubled on a file / Q+B diagonal) aimed at the zone
    inline int KS_ZONE2        = 0;     // widen the king-danger zone from ring1+one-rank to the full king_ring2
                                        // (2-ring), so attackers staging one square further out are detected.
                                        // 0 = narrow zone (byte-identical baseline); 1 = wide 2-ring.
    inline bool ENABLE_KS_ZONE_CLAMP = false;  // build the king-danger ring around a clamped center (file B..G,
                                        // rank 2..7) so a corner/edge king gets a full 9-square ring and sees the
                                        // attackers a raw corner ring misses. Off => raw ring = byte-identical.
    inline int KS_ZONE_NORM    = 0;     // Ethereal-style density normalization: scale the attacked-square COUNT to
                                        // a KS_ZONE_NORM-square reference ring (count * KS_ZONE_NORM / popcount(zone))
                                        // so a larger zone isn't charged more for size alone. 0 = off = byte-id; 9 typ.
    inline int KS_ZONE_ATTACK_PCT = 50; // scale the KING-DIRECTED boost in setAttackingLayer (king-ring + open-hole
                                        // credit) independently of the base central heatmap. That boost was counted
                                        // THREE times -- into `pieces` via positional_bonus, into OvD via the
                                        // offensive/defensive scores, and again by the dedicated king-safety term.
                                        // Halving it removes the duplication without losing the long-term
                                        // aimed-piece signal (0 = full de-king REGRESSES: STS 1538 vs 1647 at 50).
                                        // 3-seed 200g SF@2400: score 40.1%->47.5% (up all 3 seeds), POSITIONAL
                                        // collapses -22% (down all 3 seeds), STS 1555->1647, WAC 247->248.
                                        // 100 restores the pre-ship identity path (byte-id 247/39,971,153).
    inline int KS_CLAMP_SHELTER = 8;    // ENABLE_KS_ZONE_CLAMP fires only when own-pawn count in the king's ring-1
                                        // is <= this. 8 = ring is never that full => always clamp (default). Lower
                                        // it to shelter-gate the clamp (a hand-gate on shelter DID NOT help the
                                        // wrongsign on the corpus -- kept tunable for the fit, not defaulted on).
    // PER-KING DYNAMIC magnitude: scale EACH king's danger by how REAL its attack is = the CO-OCCURRENCE of
    // its own signature detectors (attackers acting THROUGH open lines / undefended holes), not their additive
    // sum. Computed per king, so the genuinely-attacked king scales UP (toward the crusher regime) while the
    // safe king scales DOWN -> the netted term reflects the true asymmetry instead of cancelling. Realness =
    // att_cnt*(open_files+weak_squares) - KS_DYN_PIVOT, fed through mod_gain (clamped 0.5x..2.0x). 0 = off (byte-id).
    inline int KS_DYN          = 0;     // dynamic-magnitude coefficient (mod_gain k); 0 = no per-king scaling
    inline int KS_DYN_PIVOT    = 4;     // realness level treated as neutral (1.0x); below damps, above boosts
    inline int KS_DYN_SHIFT    = 4;     // sensitivity of the factor to realness (mod_gain right-shift)
    inline int KS_SHIELD       = 2;     // units subtracted per friendly pawn shielding the king on its three files
    inline int KS_DEFENDER     = 0;     // units subtracted per friendly PIECE (N/B/R/Q) defending the king zone
                                        // (the attacker-vs-defender balance detector: an attack only "blows up"
                                        // when attackers outweigh defenders, like the old latent_threat gates)
    inline int KS_INTERACT     = 0;     // SUPER-LINEAR "coffin" interaction the additive units sum can't express:
                                        // units += (KS_INTERACT * undefended_pressure * (open_files+1) * attackers) >> 4.
                                        // Fires only when undefended pressure AND open lines AND real attackers all
                                        // co-occur (a king that can't be defended through open lines despite shelter).
                                        // Default 0 = byte-identical. Magnitude tuned from KS-relevant failures.
    inline int KS_DIVISOR      = 4;     // non-linear table denominator: danger = units^2 / KS_DIVISOR (below the knee)
    inline int KS_KNEE         = 12;    // units up to here grow QUADRATICALLY; above, growth is LINEAR (continuous
                                        // slope) so a crowded king zone ramps gently instead of exploding. Set >= KS_CAP
                                        // for pure quadratic-then-clamp (the old behaviour).
    inline int KS_CAP          = 80;    // units clamp (table is built up to KS_MAX_UNITS; KS_CAP <= that)
    inline int KS_FLOOR        = 13;    // DEADZONE: attack-units below this -> ZERO danger, so TRIVIAL king-danger
                                        // can't perturb non-king positions (the def1 passer bleed). Default 0 = byte-id.
    inline int KS_PHASE_FULL   = 48;    // phase_score AT/BELOW which king safety is full weight (0=full material/opening)
    inline int KS_PHASE_ZERO   = 104;   // phase_score AT/ABOVE which king safety is ~0 (128=bare kings/deep endgame)
    // King-danger definition-alignment toward classical SF11 (all default = byte-identical). SF encodes defense
    // IMPLICITLY (a defended square just isn't weak / a covered check isn't safe) rather than a blanket defender
    // subtraction, and heavily discounts attacks when the attacker has no queen.
    inline int KS_NO_QUEEN     = 6;     // units subtracted when the ENEMY of this king has NO queen (SF -873, but
                                        // on our 0..KS_CAP unit scale = single digits). 0 = off = byte-identical.
    inline bool ENABLE_KS_SF_WEAK = true;       // weak square = under-defended (<=1 defender, only K/Q), not "zero
                                                // defenders". Superset of the baseline -> re-tune KS_WEAK when on.
    inline bool ENABLE_KS_SF_SAFECHECK = true;  // safe check also counts an overwhelmed square: weak AND attacked
                                                // twice by the enemy (not only totally-uncovered squares).

    // Eval: REALIZABILITY modulation of the midgame offense-vs-defense IMBALANCE term (the king-zone
    // pressure differential, a proven ~400-Elo term that is crude: a flat reward for unmatched offense
    // with no convertibility check). A factor R (over 256, default 256 = full) scales the imbalance bonus
    // down when the attack is *unrealizable*, composed from cheap signals the eval already has. All knobs
    // default 0/256 -> R=256 -> byte-identical (the call site is gated so an all-default build skips it).
    // Position-conditional (unlike the washed scalar magnitude knobs); PACE tunes the blend. No division
    // (shift-based); colour-symmetric (same knobs drive both branches, each using its own side's material).
    inline int REALIZ_MAT_K     = 0;    // strength of the material-backing discount (0 = off; a master gate)
    inline int REALIZ_MAT_THRESH = 0;   // material edge (piece-value units, pawn=1000) below which under-backed
    inline int REALIZ_PHASE_K   = 0;    // strength of the phase/density discount (0 = off; the other master gate)
    inline int REALIZ_FLOOR     = 256;  // minimum R in /256 units (caps how far the combined discount can go)

    // Eval: discount the defender's blockade over-credit for an ADVANCED enemy passed pawn in
    // boost_pieces_for_supporting_passed_pawns. Bug (Tal loss): a single piece parked in front of a
    // far-advanced enemy passer is credited as (near-)fully stopping it, and that credit GROWS as the pawn
    // nears promotion -> our eval reads the passer as contained when it's decisive. This knob (percent,
    // default 0 = off = byte-identical) reduces the "piece directly in front" stopping credit when the
    // enemy passer is within 2 squares of promotion. Colour-symmetric. Gated on default (no hot-path div).
    inline int PASSER_BLOCK_ADV = 0;

    // Gap-P fix: scale ONLY the enemy-defender term in boost_pieces_for_supporting_passed_pawns
    // (the std::max(-rank_bonus, white_adjustment) for a black passer / std::min(...) for a white
    // passer = the enemy's blockade + path-control credit). Bug: that term over-credits the defender
    // and can flip an advancing enemy passer NET pro-defender. This knob leaves the OWN-side term
    // (own support minus self-block malus) untouched and bounds only the enemy credit. Percent;
    // 100 = byte-identical (full credit, current behavior); lower = trim the wrong-signed duplicate
    // (blockade still counts via getPPIncrement PP_BLOCKADE_PEN + rooks). Colour-symmetric.
    inline int PASSER_ENEMY_CREDIT_PCT = 0;   // SHIPPED (collapse bundle): trim the wrong-signed enemy passer credit

    // Midgame passer-DANGER: a passer within 3 steps of promotion is worth far more than the linear rank
    // bonus prices it (a supported passer 2 steps out with a passive defender ~ a rook). Separate additive
    // term (NOT spliced into boost_pieces_for_supporting_passed_pawns) = value BASE[s] x realizability(R)/256,
    // R from cheap detectors on the precomputed attack_bitmasks (blockade quality, path control, defender-king
    // distance). Midgame-only (endgame path has its own race logic). Black-positive: black passer -> +danger,
    // white passer -> -danger. Default off (ENABLE_PASSER_DANGER=false) = byte-identical. Free params = the
    // three BASE magnitudes + the D2 path-control and D4 king-distance scales (BLOCK[] fixed in v1).
    inline bool ENABLE_PASSER_DANGER = false;
    inline int PASSER_DANGER_BASE1 = 3500;  // steps-to-promote = 1 (millipawns, pre-realizability)
    inline int PASSER_DANGER_BASE2 = 2200;  // steps-to-promote = 2
    inline int PASSER_DANGER_BASE3 = 1200;  // steps-to-promote = 3
    inline int PASSER_DANGER_D2 = 70;       // realizability penalty per defender-controlled, uncontested path square
    inline int PASSER_DANGER_D4 = 24;       // realizability bonus per rank the defender king is too far from the promo square

    // ENABLE_PASSER_V3 composite-realizability terms folded into passer_realizability_R (R units, 256=neutral):
    // graded path contest (net attacker-defender counts, SF's binary k made continuous via our attack_bitmasks
    // counts), rear-file rook/queen control (SF's unsafeSquares rear logic), and king-proximity to the stop
    // square (SF's kingProximity, enemy-king-far dominates). Default values are SF-referenced starting points.
    inline int PASSER_CONTEST_STOP = 90;    // R dock per net (enemy att - own def) attacker on the STOP square
    inline int PASSER_CONTEST_PATH = 40;    // R dock per net attacker on a deeper promotion-path square
    inline int PASSER_REAR_ENEMY  = 128;    // R dock when an enemy rook/queen is behind the passer (first on the clear file)
    inline int PASSER_REAR_OWN    = 48;     // R credit when an own rook/queen is behind the passer (Tarrasch)
    inline int PASSER_KING_FAR    = 16;     // R credit per step the enemy king is from the stop square (dist capped 5)
    inline int PASSER_KING_HELP   = 6;      // R dock per step the own king is from the stop square (dist capped 5)
    inline int PASSER_MAG_SCALE   = 100;    // passer-specific scale on the base rank magnitude inside evaluate_passers()
                                            // (percent). Independent of the global SCALE_ENDGAME_RANK so tuning the passer
                                            // magnitude does not disturb non-passer endgame pawns.
    inline int PASSER_R_CAP       = 320;    // upside cap on R (256=neutral). Raising it only lifts the MOST realizable
                                            // passers (R>256) = a selective under-fire lever that never touches stopped guards.
    inline int PASSER_R_FLOOR     = 64;     // soft floor for R when scaling the additive king-race term (so a fully
                                            // stopped passer, R~0, still can't leak unbounded king-race, but a mostly-
                                            // stopped one is heavily damped): king_race * max(R, R_FLOOR) / 256
    inline int PASSER_RFLOOR_R5   = 0;      // ☠️ DEAD (2026-08-03): rank-keyed floors on R degrade held-out passer
    inline int PASSER_RFLOOR_R6   = 0;      // accuracy MONOTONICALLY and cost 107-149 STS. A rank floor fires on EVERY
                                            // advanced passer, so it cannot tell a stopped passer from an unstoppable
                                            // one, and it distorts the ASSESSMENT (R) rather than bounding the OUTPUT.
                                            // Kept at 0 as the record; use PASSER_RESID_PCT instead. Do not re-propose.

    // Residual passer value that survives R collapsing, as a percent of the rank magnitude, SCALED DOWN BY
    // BLOCKADE PERMANENCE. Rationale: our multiplicative `mag * R/256` sends a contested passer to ~0, so a
    // single wrong realizability call costs the whole pawn (measured: we read -0.01 where SF reads -1.26).
    // Bounding the OUTPUT fixes that without touching the assessment. Unlike a rank floor this DISCRIMINATES:
    // the residual is keyed on the stop-square blockade quality we already compute, so a knight blockade
    // (permanent, BLOCK 140) still collapses to ~0 while a queen "blockade" (BLOCK 50, must move) keeps most
    // of its residual. 0 = byte-identical.
    inline int PASSER_RESID_PCT   = 0;

    // Floor a passer at what the SAME pawn would earn if it were NOT passed
    // (default_midgame_pawn_rank_bonus[rank]). Under V3 the pawn loop defers the rank bonus for a flagged
    // passer, so `mag * R/256` is the sole payer and a collapsed R leaves the passer worth LESS than an
    // ordinary pawn on that square (w1: 6 mp vs 90). Being detected as passed must never be a penalty.
    // With this on, R interpolates ordinary-pawn -> full-passer instead of zero -> full-passer. The ceiling
    // is unchanged, so unlike every unconditional-base arm this adds nothing to healthy passers.
    // Default false = byte-identical.
    inline bool ENABLE_PASSER_ORD_FLOOR = false;


    // Gap-P P2: run the per-passer king-race realizability (advanced_endgame_eval's passer block,
    // extracted to passer_realizability_delta) in ALL phases, not just deep endgame, so an advancing
    // passer's danger is seen in the midgame (the Tal-bot a-pawn march). When on, the in-AE copy is
    // skipped (de-dup). PASSER_KRACE_MG_PCT = the midgame weight in percent; it ramps to 100 (full,
    // = old AE behavior) at the deep-endgame phase. Default off = byte-identical.
    inline bool ENABLE_PASSER_KRACE_MG = false;
    inline int PASSER_KRACE_MG_PCT = 100;

    // Passed-pawn realizability redesign (v2) master gate. Consolidates the scattered passer conditioning
    // into two complementary post-loop pillars and removes the double-mentions, so a passer is priced once by
    // its realizability instead of credited flat across ~9 channels. When true it: (1) fires passer_danger in
    // BOTH phases (blockade + path-attack pillar), (2) zeroes passer_danger's D4 king term (the king dimension
    // is owned solely by pillar 2), (3) fires passer_realizability_delta as the sole king-race authority (the
    // in-AE deep-eg copy auto-skips), (4) drops evaluate_kings_endgame's king-attacks-passer credit (folded
    // into pillar 2), (5) drops the inline own/enemy-symmetric +-100 piece-proximity re-credits, (6) feeds the
    // CLAMPED rank bonus to approximate_capture_gains. Default false = byte-identical. The pillars' own
    // magnitudes stay tunable via the existing PASSER_DANGER_* / PASSER_KRACE_* knobs (sweep with this on).
    inline bool ENABLE_PASSER_V2 = false;
    // Bound applied to a captured pawn's rank bonus when it is folded into approximate_capture_gains under v2
    // (the inline midgame pt_pawns contribution is clamped to +-275; capgains reads the UNCLAMPED array, so a
    // deep passer inflates the captured-pawn value far past what the board credits). Matches the inline clamp.
    inline int CAPG_PAWN_RANK_CLAMP = 275;

    // Passer redesign (V3): a FRESH, COMPLETE consolidation intended to supersede V2 (a strict superset).
    // Landed incrementally under this one gate. Round 1a: a REAR doubled pawn (a friendly pawn ahead on its
    // own file) can never promote, so it is no longer mis-flagged as a passed pawn (getPPIncrement). Later
    // rounds add the board-driven R-gate on the rank bonus, disable the smeared per-piece credits, and hook the
    // midgame king-race. Shipped default 2026-08-01 as part of the material-fix bundle (+36.7 Elo, 503 games).
    inline bool ENABLE_PASSER_V3 = true;

    // Gap-P C1: blockade-QUALITY in getPPIncrement. When on, only a secure blockade (enemy minor on the
    // stop square) gets the full PP_BLOCKADE_PEN; a rook/queen merely contesting the file ahead gets only
    // PASSER_CONTEST_PCT of it (the pawn still advances; the contester is tied down). Un-zeroes a
    // rook-contested advancing passer. Default off = byte-identical.
    inline bool ENABLE_PASSER_BLOCKADE_QUALITY = true;   // SHIPPED (collapse bundle): rook-contest vs secure-blockade
    inline int PASSER_CONTEST_PCT = 30;   // % of PP_BLOCKADE_PEN applied to a file-contest (vs a secure blockade)
    // Gap-P C2: magnitude of the per-passer king-race realizability (passer_realizability_delta + the AE
    // endgame block). Percent; 100 = byte-identical. The king-race is calibrated too weak for an
    // unstoppable passer (worth ~a queen but scored ~0.4); raise to make irresistible passers decisive.
    inline int PASSER_KRACE_MAG = 100;

    // Eval: calibrate the piece_value_boost ("dominate when material ahead") term in
    // placement_and_piece_eval. The boost = (matDiff / leaderMat) * MAG, which mechanically ESCALATES
    // as material thins (the same rook is a larger fraction of a smaller army) -> it is the measured
    // material-edge over-valuation (+231cp mid -> +462cp end up-a-rook). PV_BOOST_MAG replaces the 10000
    // magnitude literal (default 10000 = byte-identical). PV_BOOST_TRIGGER replaces the +-1500 lead at
    // which the boost engages (default 1500 = byte-identical). PV_BOOST_PHASE_K (default 0 = off, gated)
    // damps the boost harder as material thins, targeting the mid->end escalation directly; cold tail
    // (once per eval), so the gated path may use float, but the default path is the exact current
    // expression. Colour-symmetric (same knobs drive both the boost_white and boost_black branches).
    inline int PV_BOOST_MAG     = 10000; // magnitude of the material-domination boost (the (matDiff/leaderMat)*MAG term)
    inline int PV_BOOST_TRIGGER = 1500;  // |total| lead (Black-positive units) at which the boost engages
    inline int PV_BOOST_PHASE_K = 0;     // strength of the endgame-escalation damp (0 = off = byte-identical)

    // Dynamic conditional-eval layer: detector-gated modulation of the UNIVERSAL (non-evaluate_*_*) eval
    // terms. Each MOD_* strength knob defaults to 0 -> the term's master gate (if (k1|k2|...)) is false ->
    // the gain multiply is skipped -> byte-identical. mod_gain() blends cheap detector signals (already
    // computed in placement_and_piece_eval) onto a 256 base, clamped to [MOD_FLOOR, MOD_CEIL]; the term is
    // then scaled (term*gain)>>8. Integer/bitwise. Generalizes realizability_factor to all universal terms.
    // Active (first build): contextual material (pawn count + opposite bishops), latent-threat backing,
    // bishop-pair openness. Phase enters as a signal where used. See dev_notes/dynamic-conditional-eval.md.
    inline int MOD_FLOOR    = 128;   // min gain over 256 (0.5x); caps how far a term can be damped
    inline int MOD_CEIL     = 512;   // max gain over 256 (2.0x); caps how far a term can be boosted
    inline int MOD_MAT_PAWNS = 0;    // material boost x pawn-count (convertibility: fewer pawns -> ...)
    inline int MOD_MAT_OPPB  = 0;    // material boost x opposite-coloured-bishops (drawishness)
    inline int MOD_LT_BACKING = 0;   // latent-threat x material backing of the threatening side (unbacked = fantasy)
    inline int MOD_PAIR_OPEN = 0;    // bishop-pair bonus x openness (pawn count): open positions favour the pair
    // King-safety conditioning (the king_safety_score swap term): a flat danger magnitude over/under-fires
    // uniformly (symmetric static scatter), so make the danger a FUNCTION of whether the attacking side
    // actually backs the attack. MOD_KS_BACKING damps an under-backed king attack (material, MOD_LT_BACKING
    // template, damp-only). MOD_KS_CONTROL scales by the attacker's board-control edge (offensive-vs-defensive,
    // the same signal the imbalance term reads): a space-backed attack boosts, a space-less one damps.
    inline int MOD_KS_BACKING = 0;
    inline int MOD_KS_CONTROL = 0;
    // Whole-budget realizability gate for the consolidated KS home (evaluate_king_safety). Same material-backing
    // signal as MOD_KS_BACKING (damp-only when the attacking side is under-backed) but with its OWN floor so it
    // can damp the phantom attack BELOW the shared MOD_FLOOR=128 (0.5x) that MOD_KS_BACKING saturates at -- the
    // step-0 probe showed MOD_KS_BACKING bottoms out at ~halving the budget and cannot fully fix the fantasy
    // over-read. MOD_KS_REALIZ=0 (default) => not applied => byte-identical. Acts on the netted ks, so it damps
    // the whole unit-KS danger budget the home now owns.
    // Shipped default 128 on 2026-08-01 (peak of a smooth unimodal sweep: 0/64/128/256/512 = 1658/1669/1746/
    // 1705/1575). Requires ENABLE_MATERIAL_COUNT_FIX -- without it this same value is -196 STS. Note that with
    // KS_REALIZ_FLOOR == MOD_FLOOR (both 128) this is the SAME function as MOD_KS_BACKING; do not set both.
    inline int MOD_KS_REALIZ = 128;    // strength of the whole-budget material-backing damp (0 = off = byte-id)
    inline int KS_REALIZ_FLOOR = 128;  // min gain over 256 for MOD_KS_REALIZ (tunable below 128 for harder damp)
    inline int MOD_PVBOOST_COMP = 0;  // damp the material-domination boost x opponent offense-vs-our-defense COMPENSATION
                                      // (a material lead is worth less under an unmatched attack; the collapse over-read)
    inline int MOD_PVBOOST_MOB = 0;   // damp the material-domination boost x our MOBILITY edge deficit (a cramped material
                                      // lead is illusory: collapses are +4p material but -0.8 mobility). sh=0 (small edges)

    // S4: placement-confidence shrinkage (the level-material plurality lever). level_sep.py showed LOST
    // level-material positions over-fire the placement terms ~2x vs healthy ones -> large placement claims in
    // a LEVEL position are over-confident / collapse-prone. In a level-material position, shrink the aggregate
    // placement contribution (br_pieces, the cold-tail snapshot) toward 0 in proportion to how far |placement|
    // exceeds a floor. Cold tail (once per eval), integer/bitwise, gated default-off = byte-identical.
    inline int MOD_PIECES_LEVEL = 0;      // shrink strength (0 = off); cut = (K * (|pieces| - FLOOR)) >> 8
    inline int MOD_PIECES_MAT_THRESH = 1000; // |material edge| (pawn=1000) at/below which the position is "level"
    inline int MOD_PIECES_FLOOR = 500;    // |placement| below which no shrink (don't touch modest placement)
    inline int MOD_PIECES_CONTROL = 0;    // damp placement x the favoured side's offensive-vs-defensive control edge (unbacked activity over-credited; shares the KS_CONTROL detector)
    inline int MOD_PIECES_DEFEND = 0;     // damp placement x the OPPONENT's offensive pressure on the favoured side (a side under attack can't cash a static placement edge; greed-under-attack collapse cluster)
    inline int MOD_PIECES_DEFEND_THRESH = 0; // deadzone on net-attack: only damp when (oppOffense - favOffense) exceeds this (suppress in marginal/sharp positions where placement is load-bearing; cuts collateral)

    // Realizability damp on the pawn-placement credit (br_pt_pawns) when a pawn/positional lead is NOT
    // backed by a non-pawn (piece) material edge -- the fantasy-vs-real discriminator (real wins are up
    // ~a piece: npedge ~+4400 engine units; fantasy wins are pawn-only: npedge ~0). Midgame-only (a pure
    // pawn endgame has npedge~0 yet converts -> the endgame draw-scale lane owns it). Clamped linear ramp
    // in engine units (pawn=1000), NOT a hard gate (npedge jumps on every trade -> a cliff lets the search
    // game the boundary). Damp-only, cold tail (once per eval), integer/bitwise, gated default-off = byte-identical.
    inline bool ENABLE_NPEDGE_DAMP = false;
    inline int NPEDGE_DAMP_LO = 800;      // non-pawn material edge (engine units) at/below which fully unbacked (full damp)
    inline int NPEDGE_DAMP_HI = 2500;     // non-pawn material edge at/above which fully backed (no damp; real piece-up wins spared)
    inline int NPEDGE_DAMP_MAX = 90;      // max damp depth in /256 (90 ~= 0.35 -> floor retains ~65% of the placement claim)
    inline int NPEDGE_DAMP_TQUIET = 0;    // tactical-tension gate: 0=off (damp fires regardless of tension). When >0,
                                          // ramp the damp to zero as g_capg_tension rises to this value, so the damp
                                          // only fires in QUIET positions and never disturbs tactical move-choice
                                          // (moves_dump showed the damp helps positional strata but scatters the sts/
                                          // tactical stratum -- gating it off under tension keeps the gain, drops the cost).
    // Endgame extension of the npedge damp. The over-read class is ENDGAME slight-imbalance positions where we
    // are "up pawns" but the opponent's piece(s) actually compensate (advanced pawns over-valued vs, e.g., a
    // bishop). The base damp is midgame-only because a PURE pawn endgame (KPK) has npedge~0 yet converts, so
    // damping there would wrongly deflate winning pawn endgames. Guard: in the endgame only damp when the
    // DEFENDER (side without the pawn-placement claim) still has real piece material (>= NPEDGE_EG_PIECE_FLOOR),
    // i.e. there is a piece to over-value our pawns against -- pure pawn endgames are spared. Reuses the same
    // LO/HI/MAX/TQUIET ramp. Default off = byte-identical.
    inline bool ENABLE_NPEDGE_DAMP_EG = false;
    inline int NPEDGE_EG_PIECE_FLOOR = 3250;  // defender non-pawn material (engine units) required to damp in the
                                              // endgame (~one minor). Below this it is a pawn endgame -> no damp.

    // Whole-board mobility imbalance (piece activity) -- the SF-style term our eval lacks. Our PST placement
    // is context-blind: it credits our pieces' squares while the opponent out-activates us. Validated: our
    // post-refutation leaf over-read correlates -0.5 with our attacked-square edge (we over-read +320cp when
    // the opponent is more mobile vs +74cp when we are). Additive (shifts the feature ratio, not a damp) and
    // SLOW (attacked-square count moves WITH the material resolution) => volatility-safe. Default-off = byte-identical.
    inline bool ENABLE_MOBILITY = false;
    inline int  MOBILITY_SCALE = 40;      // millipawns per net attacked-square edge (Black-positive); tune via the leaf screen

    // Eval: replace the per-bishop colour-complex flood-fill (get_bishop_colour_complex_score, profiled
    // at ~33% of the entire midgame eval) with a cheap popcount approximation of the same good/bad-bishop
    // + activity signal: own pawns on the bishop's colour (bad bishop) traded against the bishop's current
    // diagonal scope (mobility / forward reach into the enemy half / enemy-king-zone pressure). Default
    // off = byte-identical (the flood-fill runs untouched). K_* are tunable weights for the sweep; the
    // output is clamped to the same [-200, +275] mp range as the flood-fill term.
    inline bool ENABLE_CHEAP_BISHOP_COMPLEX = true;   // default-on: ~25% cheaper midgame eval, STS +33, lightning +0.090 ply (midgame), self-play +2.7 Elo (no regression). Knob retained.
    inline int CHEAP_BISHOP_BLOCK = 30;   // penalty per own pawn on the bishop's colour
    inline int CHEAP_BISHOP_MOB   = 6;    // bonus per diagonally-attacked square (current scope)
    inline int CHEAP_BISHOP_FWD   = 8;    // extra bonus per attacked square in the enemy half
    inline int CHEAP_BISHOP_KING  = 12;   // extra bonus per attacked square in the enemy king zone

    // Eval: replace the per-rook mobility sub-block in evaluate_rooks_{midgame,endgame} (profiled as the
    // #1 eval hotspot, ~22% early-mid -> ~34% mid/endgame) with a cheap popcount approximation of the
    // same mobility signal. The original, per attacked non-own-occupied square, runs a five-way
    // lower-value-attacker test and a nested second-order scan; the surrogate popcounts the rook's
    // attacked-but-not-own-occupied squares (weighting those in the forward zone as a proxy for the
    // dropped second-order term). Default off = byte-identical (the per-square filter + nested loop run
    // untouched). The output reuses the term's existing std::min envelope (225 midgame / 350 endgame).
    // Proper per-piece MOBILITY: popcount(pieceAttackMask & safe mobilityArea) -> nonlinear MobilityBonus table,
    // reusing each evaluator's already-computed attack bitboard (cheap). When ON it REPLACES the cheap rook/knight/
    // queen surrogates (those gate on !ENABLE_PIECE_MOBILITY) to avoid double-count. Default off = byte-identical.
    inline bool ENABLE_PIECE_MOBILITY = false;
    inline int SCALE_MOBILITY = 100;   // percent on the MobilityBonus tables (env-tunable magnitude; 100 = table as-is)
    inline bool ENABLE_CHEAP_ROOK_MOBILITY = true;   // default-ON: +7.8% nps, WAC +2, STS +18, self-play +11 Elo (no regression)
    inline int CHEAP_ROOK_MOB = 15;   // bonus per non-own-occupied attacked square (matches original +15 midgame seed)
    inline int CHEAP_ROOK_FWD = 10;   // extra bonus per attacked square in the forward zone (proxy for the dropped second-order term)

    // Eval: same cheap-mobility surrogate applied to the queen evaluators. The queen folds mobility
    // directly into total (no accumulator / no clamp) via a per-square five-way lower-value-attacker
    // test (midgame) plus a nested second-order scan (endgame). The surrogate skips that and credits
    // a popcount of the queen's attacked, non-own-occupied squares. Per-phase weights because the
    // endgame original (with its nested scan) credits substantially more than the flat-+5 midgame.
    // Default off = byte-identical. Magnitudes are seeds for the sweep.
    inline bool ENABLE_CHEAP_QUEEN_MOBILITY = false;   // default-off = byte-identical
    inline int CHEAP_QUEEN_MOB_MG = 5;   // per non-own-occupied attacked square, midgame (matches the original flat +5)
    inline int CHEAP_QUEEN_MOB_EG = 8;   // per non-own-occupied attacked square, endgame (covers base + dropped nested scan)

    // Eval: same cheap-mobility surrogate for the knight evaluators. The knight folds mobility into
    // total via a per-square three-way lower-value-attacker test plus a nested second-order knight-hop
    // scan; the surrogate skips both and credits a popcount of the knight's reachable, non-own-occupied
    // squares (the knight attack set is small, so one weight covers both phases). Default off = byte-identical.
    inline bool ENABLE_CHEAP_KNIGHT_MOBILITY = false;   // default-off = byte-identical
    inline int CHEAP_KNIGHT_MOB = 12;   // per non-own-occupied attacked square (covers base + dropped nested scan)

    // Eval: lossless one-entry cache for setAttackingLayer in the ENDGAME. In the endgame the per-square
    // open-square / pawn-shield branches are skipped, so the whole king-danger layer is a pure function
    // of the two king squares; cache the last (white_king_sq, black_king_sq) -> layer and skip the rebuild
    // on a match. Content-keyed, so it can never go stale -> byte-identical output (the WAC node count is
    // unchanged); the win is wall-time on the endgame leaves where this term is the hottest. Default off.
    inline bool ENABLE_ATTACK_LAYER_CACHE = true;   // default-ON: lossless (byte-identical) endgame king-layer reuse

    // Eval: lossless midgame counterpart of the attack-layer cache. The midgame king-danger layer also
    // depends on own pieces + pawns in each king's 2-ring (the open-square / pawn-shield branches), so it
    // is cached as two INDEPENDENT half-layers: attackingLayer[1] keyed by (white_king_sq, white pieces &
    // ring, white pawns & ring), attackingLayer[0] by the black equivalent. Each half stays valid while
    // that side plays away from its king -> high reuse. Content-keyed -> byte-identical; the only question
    // is whether the per-node key cost beats the saved loop (measure nps). Default off.
    inline bool ENABLE_ATTACK_LAYER_CACHE_MIDGAME = true;   // default-ON: lossless (byte-identical) midgame king-layer reuse, +0.8% nps

    // Corrected static-exchange evaluation (see() in cpp_bitboard.h). Default off = the existing
    // (buggy) path. ON recomputes the side-to-move's attacker set from live occupancy each iteration
    // (exact x-ray reveals) and picks the least-valuable attacker by true piece type instead of the
    // stale eval-magnitude square_values[]. The original see() is wrong on ~2.16% of capture targets
    // (diagnostics/see_selfcheck.cpp). Behavioral (ordering + qsearch SEE filter + capture_gains) -> gated.
    inline bool ENABLE_SEE_FIX = true;   // default-on: corrected see() (harness-proven, fuzz see==ref 0.00%), self-play +18.2 Elo / no regression, −0.02 ply (the per-iteration attacker recompute). Knob retained.
    // Incremental-x-ray SEE: same result as ENABLE_SEE_FIX but maintains the attacker set across exchange
    // iterations (clear the used attacker, OR in only x-ray sliders re-revealed through it) instead of a
    // full attackersMask() recompute each step. BYTE-IDENTICAL (same SEE value -> same WAC node count);
    // ~1.3% wall-time faster (measured); byte-id proven (on -> exact 258/97,507,126). Default-ON: free
    // speed, identical moves, zero strength risk. env ENABLE_SEE_INCREMENTAL=0 recovers the SEE_FIX path.
    inline bool ENABLE_SEE_INCREMENTAL = true;
    // Per-site eval mode for the quiescent decision sites. 0=full (byte-identical), 1=cheap (material+PST
    // surrogate `cheap_eval`), 2=light (full eval MINUS the heavy dynamic terms capture_gains/passed-
    // support/latent_threat/adv-endgame, uncached via g_eval_light). Light bets those terms ~=0 at
    // quiescent leaves (where qsearch stand-pat fires) -> big NPS for ~no accuracy at the leaf.
    inline int FUTILITY_EVAL_MODE = 0;
    inline int QSTANDPAT_EVAL_MODE = 0;
    // Per-CAPTURE qsearch futility (SF-style) replacing the node-level delta prune. The node-level form
    // ("if stand-pat is DELTA_MARGIN below alpha, return without searching ANY capture") is unsound by up to
    // a queen -- it discards a hanging piece. SF instead credits the VICTIM first:
    //     futilityValue = static_eval + margin + value(captured);  if (futilityValue <= alpha) skip THIS move
    // so the margin sits ON TOP of what the capture wins. Ours only survived because stand-pat pre-banks the
    // pending capture material via approximate_capture_gains -- i.e. capture-gains masks this hole.
    // On => node-level delta is bypassed and the per-move test is used instead. Off => byte-identical.
    inline bool ENABLE_QDELTA_PERMOVE = true;

    // Node-entry reverse futility pruning (static null) + a null-move eval gate. Both read ONE node-entry
    // static eval (eval_by_mode); default-off = byte-identical (see minimizer/maximizer node entry). RFP
    // fires only at non-PV (beta-alpha==1), not-in-check, non-mate, shallow-remaining nodes.
    inline bool ENABLE_RFP = true;      // node-entry reverse futility (static null) -- SHIPPED (+73 Elo SPRT)
    inline int RFP_MARGIN = 1500;       // milli-pawn margin PER remaining ply (pawn=1000); shipped value
    inline int RFP_MIN_DEPTH = 1;       // fire only when (depth_limit - cur_depth) >= RFP_MIN_DEPTH (skip leaf-adjacent rd)
    inline int RFP_MAX_DEPTH = 6;       // fire only when (depth_limit - cur_depth) in [RFP_MIN_DEPTH, RFP_MAX_DEPTH]
    inline int RFP_EVAL_MODE = 0;       // eval_by_mode arg for the RFP/gate eval: 0=full, 1=cheap
    // What RFP RETURNS when it fires. We return the raw static eval; SF stopped doing that after SF15 and
    // now returns a value pulled toward the bound it cut against -- SF16 `(eval+beta)/2`, SF17
    // `beta + (eval-beta)/3`, SF18 `(2*beta+eval)/3` (doc: dev_notes/sf-pruning-schedules-comparison.md §1.5).
    // The rationale is to not fully trust a static eval that was never verified by search, which applies at
    // least as strongly to our slower/less accurate eval as to SF's.
    // Percent of the BOUND (alpha on the min side, beta on the max side) mixed into the returned score:
    //   0  = return the raw eval (identity, byte-identical)
    //   50 = SF16's (eval+bound)/2      67 = SF18's (2*bound+eval)/3
    inline int RFP_RETURN_BLEND = 0;
    inline bool ENABLE_NULL_EVAL_GATE = false; // only attempt null move when static eval is past beta/alpha

    // ProbCut: node-level, non-PV, not-in-check. At depth to spare, a strong capture whose reduced
    // null-window search already clears an inflated bound is taken as proof the node cuts -> prune early.
    // Self-verifying (a shallow SEARCH confirms, not the eval) -> works despite a weak eval. Default off =
    // byte-identical. Non-negamax: max side pushes beta UP, min side pushes alpha DOWN. Margin in OUR
    // millipawn units (pawn=1000) -- NOT SF's 189 (different scale).
    inline bool ENABLE_PROBCUT          = false;
    inline int  PROBCUT_MARGIN          = 2200;  // sweep ~2000-2500
    inline int  PROBCUT_MIN_DEPTH       = 5;     // fire only when (depth_limit - cur_depth) >= this
    inline int  PROBCUT_DEPTH_REDUCTION = 3;     // child target = depth_limit - this => child remaining = rem - 4
    inline int  PROBCUT_CANDIDATES      = 3;     // max strong captures/promos verified per node
    inline bool ENABLE_PROBCUT_NO_TT_STORE = false;  // diagnostic: run ProbCut verification WITHOUT writing the TT

    // Singular extensions (anti-phantom): if the TT-move's value is implausibly better than every
    // alternative (a reduced-depth exclusion search can't reach ttValue - margin), extend it. Default off =
    // byte-identical (the node-local move-populate + the whole singular block are gated on this flag).
    inline bool ENABLE_SINGULAR    = false;
    inline int  SINGULAR_MARGIN    = 2;    // singularBeta = ttValue - Sign * SINGULAR_MARGIN * depth (SF11 v1)
    inline int  SINGULAR_MIN_DEPTH = 6;    // fire only when (depth_limit - cur_depth) >= this
    inline int  SINGULAR_MAX_EXT   = 8;    // cap on active singular extensions per root-to-leaf path

    // Internal Iterative Reduction (IIR): at a node with no cached ordering evidence (movegen-cache miss =
    // first visit -> no generated/promoted move) and depth to spare, search one ply shallower; the shallow
    // pass populates the movegen/eval caches so the real re-search enters with a good move first. Our native
    // analog of Stockfish's "!ttMove". Off = byte-identical (the reduction is fully gated on this flag).
    inline bool ENABLE_IIR   = false;
    inline int  IIR_MIN_DEPTH = 6;    // fire only when (depth_limit - cur_depth) >= this

    // Simplification bias (root-only; off = byte-identical). When the side to move is clearly ahead
    // (root maximizer SIMPL_AHEAD_THRESH < best_score < mate), break EXACT score ties at the root argmax toward a
    // move that trades a non-pawn piece (simplifies toward the win) over an equal-scoring non-simplifying
    // move. Acts ONLY on eval-equal forks, so it never trades away a real edge (a worse simplifying move
    // still loses the tie on score); targets the measured under-simplification (STS Offer-of-Simplification
    // 42%) and the messy-position over-read. Not a search-tree change -- only the root argmax tie is affected.
    inline bool ENABLE_SIMPL_BIAS  = false;
    inline int  SIMPL_AHEAD_THRESH = 1500;  // root maximizer score (millipawns, pawn=1000) above which the bias fires
    inline int  SIMPL_MARGIN       = 0;     // accept a simplifying move scoring within this (millipawns) BELOW the
                                            // incumbent, not only exact ties. 0 = exact-tie only. Bounded by the
                                            // clearly-ahead gate: we deliberately trade <=MARGIN of eval for a
                                            // safer, simpler winning position (the eval over-reads messy lines).

    // Prune-verification diagnostic (default off => byte-identical). Logs one stderr [PRUNEFIRE] record per
    // sampled pruning fire (FEN + window + static/cheap eval) for the offline verify+discriminator harness.
    inline bool ENABLE_PRUNE_LOG = false;
    inline int  PRUNE_LOG_STRIDE = 20;   // log 1 in N fires to bound volume (clamped >= 1)

    // Correction-history SIGNAL diagnostic (default off => byte-identical). Logs one stderr [CORRLOG] record per
    // sampled update-eligible node (pawn key, maxbit, node-entry static eval, backed-up best score, remaining
    // depth) for diagnostics/corrhist_signal.py, which tests offline whether a per-pawn-key correction shrinks
    // held-out static-eval error BEFORE we build the corrhist table (Phase 0 signal gate).
    inline bool ENABLE_CORRHIST_LOG = false;
    inline int  CORRHIST_LOG_STRIDE = 32;   // log 1 in N eligible nodes to bound volume (clamped >= 1)

    // Correction history (Phase 1). pawnCorrHist[maxbit][pawnKey] holds an integer EMA of the residual
    // (bestScore - staticEval); the node-entry static eval is corrected by CORR_W/CORR_DIV of the entry before
    // the RFP/null gates. Default off => byte-identical. Signal-verified in Phase 0 (pawn x maxbit).
    inline bool ENABLE_CORR_HIST = false;

    // Extend the correction history to the qsearch static eval (stand-pat + the qDepth horizon return).
    // SF applies the correction once, at the staticEval assignment, so every consumer inherits it -- in both
    // the main search and qsearch (to_corrected_static_eval is called at four sites there). Ours reaches only
    // rfp_static_eval, so RFP and the null-move eval gate see a de-biased eval while qsearch stand-pat, the
    // node-level qdelta and the per-move qdelta futility all still compare a RAW eval against a bound.
    // Requires ENABLE_CORR_HIST; separate so RFP-only can be A/B'd against RFP+qsearch. Default off.
    // Note the correction is learned against RFP_EVAL_MODE and applied here against QSTANDPAT_EVAL_MODE:
    // the frames agree only while both modes are 0 (the default).
    inline bool ENABLE_CORRHIST_QSEARCH = false;

    // DIAGNOSTIC ONLY: bypass the quiescence cache on both the probe and the store. The q-cache is keyed by
    // zobrist alone, so any eval adjustment that drifts over time (correction history) gets frozen into it
    // and served stale to later probes -- which confounds ENABLE_CORRHIST_QSEARCH with a cache-coherence
    // artifact. Turning this on isolates the correction's own value at a large cost in nodes; it is not
    // shippable. Default off = byte-identical.
    inline bool DISABLE_QCACHE = false;

    // Refuse to store a quiescence result the search never actually produced. qSearch returns a bare 0 on
    // three paths that are not evaluations: a timeout abort, a node-limit abort, and a repetition draw --
    // and the last is a property of the PATH, not of the position. get_q_search_eval caches whatever comes
    // back, tags it by comparison against the window, and the q-cache has no generation or age field and is
    // never cleared, so a 0 stored during one move's timeout unwind is served as a real evaluation for the
    // rest of the game. The main TT already refuses draws for this reason (ENABLE_TT_STORE_DRAW); the
    // q-cache does not. Fixed-depth benches never reach the timeout paths (measured: 0 fires at fixed depth
    // vs 144 timed), so this is byte-identical on every fixed-depth bench and can only change timed play --
    // which is why it defaults ON despite being unmeasurable by the benches.
    inline bool QCACHE_SOUND_STORE = true;

    // Serve only EXACT entries from the quiescence cache, refusing LOWER/UPPER reuse. Diagnostic: a bound
    // entry was produced under a DIFFERENT window, and qsearch is window-dependent (per-move delta pruning
    // keys on alpha), so a value computed under a WIDE window explored more captures than a fresh search in
    // a narrow window would. If most of the cache's benefit disappears here, the cache is not acting as a
    // transparent lookup but as a carrier of better-window values. Default off = byte-identical.
    inline bool QCACHE_EXACT_ONLY = false;

    // Recompute whitePieceVal/blackPieceVal from the bitboards after the piece loops, instead of trusting the
    // side-effect accumulation inside the per-piece evaluators. The phase-blend path (phase_score > 40) calls
    // BOTH the midgame and endgame variant for the same square: the returned scores are blended, but the
    // `pieceVal +=` side effect is not, so pawns/rooks/queens/kings are counted TWICE. Measured 2026-07-31:
    // 277 of 600 banked positions carry a wrong `material`, per-side accumulators inflated ~3.8 pawns (max 30).
    // Its value is almost entirely in what it UNBLOCKS: alone it is worth ~+5 STS, but it turns MOD_KS_REALIZ
    // from -196 STS into +88 (the consumer reads the material edge, so a doubled edge damped the wrong side).
    // Shipped default 2026-08-01 with ENABLE_PASSER_V3 and MOD_KS_REALIZ=128: +36.7 Elo over 503 games.
    inline bool ENABLE_MATERIAL_COUNT_FIX = true;

    // approximate_capture_gains mutates whitePieceVal/blackPieceVal as a side effect: pieces it simulates off
    // the board come out of the accumulators. ENABLE_MATERIAL_COUNT_FIX recomputes them BEFORE that call, so
    // capgains immediately undoes it and every later consumer of the material edge (MOD_KS_REALIZ, imbalance
    // realizability, piece_value_boost, the `material` diagnostic) reads exchange-adjusted material rather
    // than raw material. Measured 2026-08-02: moving the White queen d1->d4 shifts `material` by 1.045 pawns
    // with no capture on the board. This re-runs the recompute AFTER capgains so those consumers see raw
    // material. Default false = byte-identical; it changes the input to four tuned terms, so it needs a joint
    // retune (sweep MOD_KS_REALIZ on top) rather than a straight A/B.
    inline bool PIECEVAL_RECOMPUTE_LATE = false;
    inline int  CORR_SHIFT = 6;     // EMA learning rate = 1 / 2^CORR_SHIFT (higher = slower/steadier)
    inline int  CORR_MAX   = 2000;  // clamp on the stored EMA residual (millipawns)
    inline int  CORR_W     = 192;   // applied correction = entry * CORR_W / CORR_DIV (192/256 = 0.75x)
    inline int  CORR_DIV   = 256;
    inline bool ENABLE_CUTOFF_CLASS = false; // diagnostic: tally beta-cutoff moves by class(cap/promo/killer/counter/quiet)×rank

    // Piece-type×to continuation history as a SEPARATE additive ordering term (SF-style coexistence), NOT a
    // re-keying of counterMoveHeuristics/contHist2 (those stay from×to). Default off ⇒ term never read/written
    // (byte-identical). On ⇒ the dense piece×to signal is summed into the quiet ordering + statScore alongside
    // the from×to terms, keeping origin-square specificity for strategy while adding tactical generalization.
    inline bool ENABLE_PIECE_CONTHIST = false;
    // Right-shift applied to the pieceContHist value before it is summed into the ordering score / statScore
    // (a cheap weight: 0 = full, 1 = half, …). Tune so the dense term refines without drowning the specific ones.
    inline int PIECE_CONTHIST_SHIFT = 0;

    // Threat-conditioned butterfly quiet history (Ethereal/Caissa style): the from×to butterfly split by whether
    // the from/to squares are attacked by the opponent, summed into quiet ordering as an additive term. Default
    // off = never read/written = byte-identical. SHIFT is a cheap weight (right-shift before summing).
    inline bool ENABLE_THREAT_HIST = false;
    inline int THREAT_HIST_SHIFT = 0;

    // Eval-scaled null-move reduction: reduce MORE when the static eval is far past the bound (max: eval>>beta,
    // min: eval<<alpha). We currently have only the FLAT NULLMOVE_EXTRA. Default off = byte-identical.
    inline bool ENABLE_NULLMOVE_EVAL_R  = false;
    inline int  NULLMOVE_R_DIV          = 1920;  // millipawns of margin per extra ply (larger = gentler)
    inline int  NULLMOVE_R_CAP          = 3;     // max extra plies removed

    // qsearch quiet-check cost (buildNoisyMoveList). Default off = byte-identical (full board-copy +
    // is_check per quiet move at every q-ply). QCHECK_DEPTH0: include quiet checks only at the first
    // q-ply (qDepth==0), mainstream practice -- shrinks the q-tree. QCHECK_MASK: detect direct checks
    // with a bitboard attack test from the destination square (reusing the ENABLE_CHECK_ORDER logic)
    // instead of simulating the move; misses discovered checks (standard accepted tradeoff). Behavioral.
    inline bool ENABLE_QCHECK_DEPTH0 = false;
    // Reject quiet checks that hang the checking piece (destination attacked by an enemy pawn with a
    // non-pawn mover, or attacked and undefended). Captures in qsearch are already gated on see() >= 0;
    // this applies the same standard to the one noisy category that was admitted unconditionally.
    // Default off = byte-identical (every check that passes detection is still searched).
    inline bool ENABLE_QCHECK_SAFE = false;
    // Strictness of that filter. PAWN rejects only checks an enemy pawn attacks (a pawn-takes-piece
    // refutation is rarely a real sacrifice); UNDEFENDED additionally rejects checks that are attacked and
    // undefended -- measured at 248 solves / 40.4M nodes vs 253 / 49.3M unfiltered, i.e. it discards the
    // sacrificial checks the feature exists to find.
    constexpr int QCHECK_SAFE_PAWN = 1;
    constexpr int QCHECK_SAFE_UNDEFENDED = 2;
    inline int QCHECK_SAFE_LEVEL = QCHECK_SAFE_UNDEFENDED;
    // Full quiet-check detection: direct AND discovered, via relocated piece masks and one attackersMask
    // query, with no board copy. Takes precedence over ENABLE_QCHECK_MASK (which sees only direct checks)
    // and over the simulate path (whose is_check tests the wrong side). Castling checks by the rook are a
    // known exclusion. Default off = byte-identical.
    inline bool ENABLE_QCHECK_FULL = false;
    // Diagnostic: while ENABLE_QCHECK_FULL runs, count the checks the mask arm would have missed.
    inline bool ENABLE_QCHECK_MASK_COMPARE = false;

    // Node-exit TT store. Every existing store happens in the PARENT's frame (keyed on the child's zobrist),
    // so a node's best move is out of scope at its store site and TTEntry::move can only ever be filled on a
    // REVISIT that happens to reach a beta cutoff -- 36 times per 300-position bench. Singular extension
    // needs that move at node entry, so it is structurally starved rather than merely ineffective.
    // When on, minimizer/maximizer additionally store their OWN result with their own best move.
    // Default off = byte-identical (no extra store, no extra probe).
    inline bool ENABLE_NODE_TT = false;
    // Sort the root tail's two provenance groups independently instead of in one comparison. The tail
    // holds the previous iteration's REAL searched scores concatenated with the shallow pre-pass scores
    // for moves that have no previous entry; sorting them together compares a deep value against a
    // shallow one. Split keeps the trusted group ahead of the pre-pass group.
    inline bool ENABLE_ROOT_SORT_SPLIT = false;
    // Log cumulative nodes at the end of each iterative-deepening iteration, so the REAL per-iteration
    // branching factor can be recovered by differencing. Diagnostic only (a stderr line; search untouched).
    inline bool ENABLE_ITER_LOG = false;

    // Static placement ordering (L0): break ties among quiets the history tables have never seen, using the
    // eval's own placement layer. Quiets with zero history currently score identically, so their order is
    // whatever move generation produced -- and LMP/LMR prune and reduce by that arbitrary index, which is
    // also where wrong reductions cluster. Applied ONLY where |history| <= STATIC_ORDER_HIST_MAX so every
    // move that already has a real score is untouched and the change stays measurable.
    // Default off = byte-identical.
    inline bool ENABLE_STATIC_ORDER = false;
    // DELTA = destination minus origin (the eval's own view of the move's positional change);
    // DEST = destination only (rewards reaching good squares, ignores what was given up).
    constexpr int STATIC_ORDER_DELTA = 0;
    constexpr int STATIC_ORDER_DEST = 1;
    inline int STATIC_ORDER_MODE = STATIC_ORDER_DELTA;
    inline int STATIC_ORDER_WEIGHT = 100;      // percent scaling of the placement term
    inline int STATIC_ORDER_HIST_MAX = 0;      // apply only when |history| is at or below this
    // Bitmask of piece types to score, bit = pieceType-1 (0=pawn,1=knight,2=bishop,3=rook,4=queen,5=king).
    // Rook is EXCLUDED by default: its PST is dead code in the eval, so ordering by it ranks moves on numbers
    // the evaluation never reads. 55 = all except rook.
    inline int STATIC_ORDER_PIECES = 55;
    // King placement is endgame-only in the eval; scoring king quiets by that table in the midgame inverts
    // the safety/centralisation tradeoff. Gate it to phase_score > 62, matching move generation's threshold.
    inline bool STATIC_ORDER_KING_EG_ONLY = true;
    inline bool ENABLE_QCHECK_MASK = false;

    // Phase B endgame colour-asymmetry fixes + dead-pin revival (adversarial rescan). All behavioral
    // -> gated default-off. CAPGAIN: capture-gains pawn_rank sign in the black branch. ROOK_DBLCOUNT:
    // endgame black-rook extra unconditional rookIncrement add (no white mirror). KNIGHT_MOB: endgame
    // knight mobility bonus is 15 for black vs 10 for white.
    inline bool ENABLE_CAPGAIN_PAWN_FIX = true;
    // ☠️ TURNED OFF 2026-08-08. This and ENABLE_ROOK_DBLCOUNT_SYM_UP are MUTUALLY EXCLUSIVE designs for
    // the same defect -- FIX symmetrizes DOWN (delete Black's extra "rook behind enemy pawn" term),
    // SYM_UP symmetrizes UP (give White the matching one). BOTH were shipped in the collapse bundle, so
    // White gained the term and Black lost it and the pair REPRODUCED the very asymmetry each was
    // written to remove, sign-flipped: 105 mp on 8/1R6/5k2/1p6/8/6K1/8/8 b. Live in every game since.
    // Balanced STS picks UP: symmetrize-up 3264 (+5 vs the broken pair, i.e. free) against
    // symmetrize-down 3146 (-113). Chess agrees -- a rook behind an enemy passer is worth real material,
    // so deleting it from both sides discards signal. ⚠️ Do not re-enable without disabling SYM_UP.
    inline bool ENABLE_ROOK_DBLCOUNT_FIX = false;  // SHIPPED OFF 2026-08-08: exclusive with SYM_UP, which stays ON
    // SHIPPED 2026-08-08 (correctness). Colour-balanced measurement: positional 3406 -> 3405 (neutral),
    // tactical 485 -> 495 (+10 solves), colour-swap violations 57.4% -> 47.1%, nodes +0.06%. Free.
    // ⚠️ It was in the 2026-06 three-fix bundle that measured -32.2 Elo, but that bundle was never
    // isolated and its villain was pinned on CAPGAIN_PAWN_FIX (later shipped alone at ~neutral), so the
    // -32 is UNATTRIBUTED, not evidence against this knob.
    inline bool ENABLE_KNIGHT_MOB_FIX = true;    // SHIPPED 2026-08-08 in the 7-fix colour bundle

    // Symmetrize-UP counterparts to the KNIGHT_MOB / ROOK_DBLCOUNT colour asymmetries: instead of
    // collapsing black DOWN to white's value (the _FIX knobs), raise WHITE up to black's higher
    // magnitude so the term is colour-symmetric at the larger value (tests whether the magnitude, not
    // the asymmetry, carried the eval signal). KNIGHT_MOB_SYM_UP: endgame white knight base mobility
    // 10 -> 15. ROOK_DBLCOUNT_SYM_UP: add white's missing extra (7-rank)*35 "rook behind enemy pawn"
    // term. Behavioral -> gated default-off.
    inline bool ENABLE_KNIGHT_MOB_SYM_UP = false;
    inline bool ENABLE_ROOK_DBLCOUNT_SYM_UP = true;   // SHIPPED (collapse bundle): rook double-count symmetric-up

    // PAWN_SUPPORT_WRAP: evaluate_pawns_endgame's BLACK branch has its two file-wrap guards SWAPPED --
    // `<<9` (up-right, wraps onto file A) is masked with ~BB_FILE_H and `<<7` (up-left, wraps onto file
    // H) with ~BB_FILE_A. So it both misses the real wrap and DELETES a legitimate diagonal supporter on
    // the guarded file: a black pawn on b5 supported by a6 reads as unsupported, losing EG_SUPPORT and
    // then wrongly collecting EG_LATENT (which is gated on the support being absent). Worth −85 mp on
    // `8/8/p4k2/1p6/8/8/8/5K2 w`. NOT a judgement call -- the midgame twin and the sibling near the
    // pawn-shield code both use the correct pairing; this one site is the outlier. Behavioral -> gated.
    // SHIPPED 2026-08-08 as CORRECTNESS, with its cost recorded and UNEXPLAINED. Colour-swap violations
    // 47.1% -> 27.5% and file-mirror 18.4% -> 7.2% (two independent invariants, the corroboration);
    // tactical FREE (+10 balanced solves, same as KNIGHT_MOB_FIX); positional -105 balanced STS.
    // ☠️ The "stale EG_SUPPORT/EG_LATENT" explanation for that -105 was MEASURED AND REFUTED: with the
    // fix on, Black's endgame clamp binds LESS (13.1% -> 12.3%) and its mean structural bonus goes DOWN
    // (88.2 -> 86.5), so neither clamp saturation nor a magnitude shift accounts for it. (White is
    // byte-identical across that probe, which validates it.) The cost is diffuse -- every constant in
    // the eval was fitted against the buggy function -- so it is NOT repairable by a targeted sweep.
    // Shipped anyway: a correct eval is the foundation a retune has to sit on, and STS has been wrong
    // in this exact direction before (capped threats read -61/-77 STS and won +45 Elo).
    inline bool ENABLE_PAWN_SUPPORT_WRAP_FIX = true;    // SHIPPED 2026-08-08 in the 7-fix colour bundle

    // Batch 1 endgame-asymmetry tail. ROOK_ENDGAME_CAP: evaluate_rooks_endgame applies rookIncrement
    // UNCAPPED in both colour branches (reaching ~725 on a behind-passer file), unlike
    // evaluate_rooks_midgame which clamps std::min(rookIncrement, 300) -- suspected endgame rook/material
    // over-valuation; the fix clamps the endgame increment to ROOK_ENDGAME_CAP (magnitude fix, colour-
    // symmetric). ROOK_RANKWIN_FIX: in evaluate_rooks_midgame the white own-pawn rank window (< 5, with
    // the term flipping to a +75 bonus at rank 4) and the black mirror (> 4) are not true mirrors at the
    // boundary; the fix aligns them. Both behavioral -> gated default-off.
    inline bool ENABLE_ROOK_ENDGAME_CAP = false;
    inline int ROOK_ENDGAME_CAP = 300;
    // SHIPPED 2026-08-08. Midgame rook OWN-pawn rank window: White fires `< 5` (ranks 0-4), whose mirror
    // is Black `> 2`, but the code said `> 4` so Black skipped ranks 3-4. Parked in 2026-06 because the
    // instrument of the day could not resolve it; re-tested with the mirror suites it was the biggest
    // single symmetry win of the sweep (violations 23.8% -> 14.6%, pt_rooks 122 positions -> 36).
    inline bool ENABLE_ROOK_RANKWIN_FIX = true;

    // Search-side latent-bug knobs (adversarial rescan #8; bench-decided on WAC nodes/solves + STS d10).
    // TT_DEPTH_FIX (DEFAULT-ON, the keeper): reorder_legal_moves' pre-pass searches depth_limit-1 but
    // stored depth_limit in the TT (+1 over-trust); storing the honest depth WON the bench -- WAC +1 solve
    // (259->260) AND -2.1% nodes (255,372,592->249,966,786), STS +33 (1517->1550). New d10 control baseline.
    // QPREC_PHASE_GATE (default-off, kept as a documented negative): an unconditional use_q_precautions=true
    // overrides the phase branches (meant true only for phase_score>=96), so midgame shallow leaves skip
    // qsearch; restoring the phase logic tested WORSE (-3 WAC solves, +9.1% nodes) -> the accidental
    // always-on is better, leave off. NULLMOVE_CURDEPTH_* (defaults 3/4 optimal): the minimizer null-moves
    // at cur_depth>=3 and the maximizer at >=4 -- NOT a colour bug but a PARITY artifact (minimizer sits at
    // odd cur_depths, maximizer at even), confirmed: MAXI=3 is byte-identical (no maximizer node at
    // cur_depth 3) and MINI=4 is worse (-1 solve, +13% nodes). Knobs retained for future MAXI=2 tuning.
    inline bool ENABLE_QPREC_PHASE_GATE = true;   // SHIPPED (collapse bundle): restores midgame qsearch phase logic
    inline bool ENABLE_TT_DEPTH_FIX = true;
    inline int NULLMOVE_CURDEPTH_MINI = 3;
    inline int NULLMOVE_CURDEPTH_MAXI = 4;

    // Margin-gated verification re-search: when > 0, a reduced move that fails low
    // by less than this margin (a near-miss) is re-searched. The one mechanism that
    // uses the "how close to alpha" signal. The wider 16000 margin catches the buried
    // winning-capture line (Gap-T: the benoni-29 axb5 collapse vs tal-BOT) that
    // history-LMR over-reduces past the old 6000 margin; it recovers the positional
    // credit combo1's pruning traded away (STS +62) at ACPL-neutral / self-play
    // +3.7 +/-31 (no regression). Set 6000 for the old blitz default, or 0 to disable
    // and recover the old byte-identical search (e.g. for a d10 isolation control).
    inline int VERIFY_MARGIN = 16000;

    // Graduated verification: re-search the near-miss at depth_limit - this (a
    // shallow re-look) instead of full depth, to cut the cost of a wide VERIFY_MARGIN.
    // 0 = full-depth re-search (the original VERIFY behavior); larger = cheaper. The
    // default 2 is the validated config: a clean +17 at LIGHTNING (real-play) without
    // the ply-cost of the deeper re-search that made margin-6000/reduction-1 regress.
    inline int VERIFY_RESEARCH_REDUCTION = 2;

    // Optimism-triggered verification (OTV). When a child's backed-up score is about to be
    // trusted (become the node's new best / cause the cutoff) AND it overshoots the node's OWN
    // static eval by more than OTV_MARGIN (the "phantom" signature -- a buried LMR-reduced
    // refutation the search never re-surfaced), re-search that child with reductions turned OFF
    // for the first OTV_PLIES plies of its subtree and accept the corrected score. Default off =
    // byte-identical (the whole mechanism is gated on ENABLE_OTV, and the reductions-off window is
    // inert while g_verify_no_reduce_until < 0). Games-gated (node_ab / STS), never the static
    // compass. OTV_MARGIN is in engine units (pawn = 1000). OTV_PATH_CAP bounds re-searches per
    // root-to-leaf path (via g_verify_count). OTV_PV_ONLY restricts the trigger to full-window
    // (PV) nodes (beta - alpha > 1). OTV_MIN_REMAINING fires only with at least this much depth
    // left (depth_limit - cur_depth), where a reduced refutation can actually hide.
    inline bool ENABLE_OTV = false;
    inline int OTV_MARGIN = 1750;
    inline int OTV_PLIES = 2;
    inline int OTV_PATH_CAP = 3;
    inline bool OTV_PV_ONLY = true;
    inline int OTV_MIN_REMAINING = 4;

    // Exclusive upper bound on the iterative-deepening depth_limit. Default 64 is
    // the normal play cap (a time-limited preset governs the actual depth reached);
    // set the MAX_DEPTH env knob to 11 to pin the fixed-depth-10 isolation control.
    inline int MAX_ITERATIVE_DEPTH = 64;

    // Fixed-node search cap for the low-variance mid-funnel self-play A/B (deterministic
    // node budget instead of the clock -> only eval/search decisions differ between arms).
    // 0 = off (clock-bound, the shipped behavior). When >0, the search stops after this many
    // total nodes (main + qsearch) via the same time_up fallback as a timeout, returning the
    // move from the deepest fully-completed iteration. Checked per node (num_iterations is a
    // plain int), so it is exact at any budget, unlike the 200k-node TIME_CHECK_INTERVAL.
    inline int NODE_LIMIT = 0;

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
    // value units, pawn=1000) = tolerate small sacrifices. Default 300 is the validated keeper:
    // extending every check is a large node adder; SEE-filtering it costs ~3 WAC / ~4.7% STS at
    // fixed depth while cutting search ~27.5%, which converts to depth and net Elo in time-limited
    // play. Set to SEE_EXTEND_DISABLED to restore the old extend-every-check behavior.
    inline constexpr int SEE_EXTEND_DISABLED = 1000000;
    inline int SEE_EXTEND_MARGIN = 300;

    // Phase-0 diagnostic (default off = byte-identical): measure the "light eval" gap = the signed sum of
    // the skippable tail terms (capture_gains + passed_support + latent_threat + advanced_endgame delta) per
    // eval, so the lazy-eval margin/skip-rate can be sized before building the fast path. Recording only;
    // never alters the returned eval. Accumulates into g_lge_* (cpp_bitboard), dumped at search end.
    inline bool LIGHT_GAP_PROBE = false;

    // Phase-A diagnostic (default off = byte-identical, and zero hot-path cost when off): count see() calls
    // (g_see_calls), surfaced on the [search] line, to measure SEE frequency per node.
    inline bool SEE_COUNT = false;

    // Per-position SEE cache (default off = byte-identical). see() is pure, so cached values are identical and
    // node counts are unchanged with it on -- it is a pure speed lever (fewer recomputed exchanges).
    inline bool ENABLE_SEE_CACHE = false;

    // qsearch SEE re-sort (default off = byte-identical): order the noisy move list as promotions, then
    // captures by SEE-descending (reusing the value already computed by the >=0 filter), then quiet checks --
    // for earlier stand-pat cutoffs. Currently the noisy list inherits the main sort (clearly-good captures
    // are MVV-LVA, not SEE), so this is the change. Behavior change -> validated by qnodes/qfmc, not byte-id.
    inline bool ENABLE_QSEE_RESORT = false;

    // EBF / node-lowering knobs exposing previously-hardcoded search formulas (defaults = the original
    // literals = byte-identical). PACE / joint-tune targets on the reliable node/STS/timed-depth proxies.
    // LMR_EXTRA: extra plies of late-move reduction subtracted from the computed reduced depth in
    //   reduced_search_depth (applied BEFORE the pin clamp so pin protection still holds). 0 = original.
    //   Higher = reduce more = fewer nodes (gated by tactics holding on the proxies).
    // HISTORY_BONUS_SCALE: percent scale on the depth^2 history/counter/cont-hist bonus `b` (move-ordering
    //   strength -> first-move-cutoff -> EBF). 100 = original `(depth_limit-cur_depth)^2`.
    inline int LMR_EXTRA = 0;
    inline int HISTORY_BONUS_SCALE = 100;

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
    // Software-prefetch the child node's TT slot at the end of make_move, hiding the DRAM latency of
    // the transposition-table probe behind the remaining make/return work (the classic do_move prefetch).
    // A pure hardware hint with no architectural effect -> byte-identical either way; the knob exists only
    // to A/B the NPS gain on one binary (ENABLE_TT_PREFETCH=0 vs 1). Default OFF: banked neutral -- the d12
    // NPS A/B was +0.26% (within noise) on the cold-TT depth bench; revisit under a hot-TT / timed-game
    // instrument where the end-of-make_move prefetch has more latency to hide.
    inline bool ENABLE_TT_PREFETCH = false;
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

    // ---- Persistent-table fields (ENABLE_ROOT_TABLE; inert while the knob is off) ----
    // The score this move carried into the current iteration -- SF's RootMove::previousScore, used only as
    // the SECOND level of the sort key. Because level 1 is all-this-iteration and level 2 is
    // all-previous-iteration, depths are never compared against each other and a shallow score can never
    // outrank a deep one.
    int prev_score;
    // The most recent score a search actually PROVED for this move, surviving iterations in which the move
    // only failed low. top_score is the sort key and carries a sentinel when unproven, so it cannot answer
    // "what is this move worth?" -- razoring reads this instead, paired with `age` for how stale it is.
    // Separating the two is what lets the razor fire on real evidence while the sort still ranks a fail-low
    // move below everything proven this iteration.
    int last_real;
    // The depth `last_real` was actually searched at. Once reductions can produce stored scores, a value is
    // meaningless without it: a shallow score standing in for a deep one is exactly the "shallower overrides
    // deeper" fault that invalidated ROOT_PRESEARCH_REDUCTION, and it is why SF refuses to store reduced
    // scores at all. We do store them, so the depth travels with the value and pruning checks both.
    int last_real_depth;
    // Was top_score produced by a search that actually proved something this iteration? SF stores a real
    // value only when `moveCount == 1 || value > alpha` and writes -VALUE_INFINITE otherwise. An unproven
    // entry is a sentinel, never a measured value, and must never be pruned on.
    bool verified;
    // Iterations since this move last carried a verified score. 0 = proven this iteration. Root razoring
    // consults this so it can fire on recent evidence and skip (not abandon) everything staler.
    int age;

    RootScore() : top_score(0), prev_score(ROOT_SCORE_UNPROVEN), last_real(ROOT_SCORE_UNPROVEN), last_real_depth(0), verified(false), age(ROOT_AGE_NEVER) {}

    RootScore(int top_score_, std::vector<Move> second_moves_, std::vector<int> second_scores_)
        : top_score(top_score_),
          second_moves(std::move(second_moves_)),
          second_scores(std::move(second_scores_)),
          prev_score(ROOT_SCORE_UNPROVEN),
          last_real(ROOT_SCORE_UNPROVEN),
          last_real_depth(0),
          verified(false),
          age(ROOT_AGE_NEVER) {}
};

struct SearchData
{
    // Full ordered root-move list (length N). Set once per node; never push/pop'd element-wise.
    std::vector<Move> moves_list;

    // One entry per *searched* root move (cutoff length <= N): the top-level score plus the second-level
    // reply ordering/scores. moves_list[i] corresponds to scores[i] for every i < scores.size().
    std::vector<RootScore> scores;

    // Index from which top_score is SYNTHETIC rather than searched. scores must stay full-length because
    // alpha_beta indexes scores[i].second_moves for every root move, so a move with no score still needs an
    // entry -- but "no score" and "a score of zero" are different things, and root razoring must not prune on
    // a value nothing measured. SIZE_MAX = every entry is a real searched score (the normal path).
    size_t synthetic_from = SIZE_MAX;

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
// previous_search_data is read-only here (four read sites; the two time-up bailouts copy it out
// explicitly). Taking it by const reference avoids deep-copying moves_list plus every RootScore's
// second_moves/second_scores on every alpha_beta entry AND on every aspiration re-search.
SearchData reorder_legal_moves(int alpha, int beta, int depth_limit, const TimePoint &t0, uint64_t zobrist, const SearchData &previous_search_data, std::vector<BoardState> &state_history, std::unordered_map<uint64_t, int> &position_count, int &num_iterations);
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
inline std::vector<Move>& buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove);
inline std::vector<Move>& buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, int qDepth, Move prevMove);

#endif // SEARCH_ENGINE_H