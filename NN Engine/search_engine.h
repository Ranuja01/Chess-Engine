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
    inline bool ENABLE_NULLMOVE = true; // null-move pruning
    inline bool NULLMOVE_PROGRESSIVE = false; // depth-scaled null-move reduction (-2 at d>=12, -3 at d>=14); off = flat -1
    inline int NULLMOVE_EXTRA = 2;      // extra plies off the null-move search depth (more aggressive null pruning); 0 = byte-id baseline, 2 = combo1
    inline bool ENABLE_QDELTA = true;   // delta pruning in quiescence
    inline int DELTA_MARGIN = 1500;     // qsearch delta-pruning margin (lower = prune more captures); EBF lever
    inline int MAX_QDEPTH = 10;         // qsearch depth cap (lower = shallower qsearch); EBF lever

    inline bool LMR_PROFILE = false; // env-gated LMR-miss profiler (diagnostic; off = byte-identical)

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

    // Root pre-search (reorder_legal_moves) depth = depth_limit - this. The pre-search is a full-width shallow
    // search of the root run every iteration to order moves (and generate the 2nd-level lists). 1 = byte-id
    // (depth_limit-1, original). Higher shrinks the pre-search toward a 1-ply pass (cheaper, weaker ordering) —
    // tests whether the deep pre-pass earns its ~1/EBF node overhead. Clamped to depth >= 1.
    inline int ROOT_PRESEARCH_REDUCTION = 1;
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
    inline bool ENABLE_CAPTURE_HIST = false;   // capture-history refinement -- marginal (+0.7 STS/+1 WAC but +2.6% nodes); knob, revisit after bonus/malus
    inline bool ENABLE_CHECK_ORDER = false;    // direct-check bonus -- on the SCALE-OFF baseline it's -6 WAC for -9.8% nodes (accuracy traded for speed; bad at fixed depth). BONUS=6000 too hot -> recalibrate lower before re-enabling
    inline int CHECK_ORDER_BONUS = 6000;       // the flat quiet-check ordering bonus
    // TT best-move ordering: remember each node's beta-cutoff move in a hash-move table (g_ttMoveTable) and
    // promote it to the front of the move list on the next visit. The engine has no hash move in TTEntry; the
    // move-gen cache promotes cutoff moves only while its own entry survives, so this is a longer-lived backup.
    // Default off = byte-identical (no writes, no reads).
    inline bool ENABLE_TT_MOVE = false;

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
    inline int KING_SAFETY_MAG = 0;     // master percent scale (0 = off = byte-identical)
    // REPLACE the flat latent_threat with king_safety_score (the structural swap, not an additive run-beside).
    // Off (default) = byte-identical: latent_threat adds as today, king_safety only if MAG>0. On = skip the
    // latent_threat add entirely and route king danger through king_safety_score (no double-count); needs
    // KING_SAFETY_MAG>0 to do anything. Phase A finds the neutral MAG where the swap is ~0 regression.
    inline bool ENABLE_KS_REPLACE_LT = false;
    inline int KS_LIGHT_MAG    = 0;     // LIGHT-eval king-pressure surrogate scale (g_eval_light path; 0 = off = byte-id)
    inline int KS_ATT_KNIGHT   = 2;     // attack units per enemy knight bearing on the king zone
    inline int KS_ATT_BISHOP   = 2;     // per enemy bishop
    inline int KS_ATT_ROOK     = 3;     // per enemy rook
    inline int KS_ATT_QUEEN    = 5;     // per enemy queen
    inline int KS_ATTACK_COUNT = 1;     // per zone square the enemy attacks (additive zone pressure)
    inline int KS_WEAK         = 2;     // per weak zone square (enemy-attacked, not defended by a friendly pawn)
    inline int KS_SAFE_CHECK   = 3;     // per square from which the enemy can deliver a safe check
    inline int KS_STORM        = 1;     // per rank of enemy pawn-storm advance on the king's three files
    inline int KS_OPEN_FILE    = 2;     // per open/semi-open file on/adjacent to the king file
    inline int KS_BATTERY      = 3;     // per rook/queen battery (doubled on a file / Q+B diagonal) aimed at the zone
    inline int KS_ZONE2        = 0;     // widen the king-danger zone from ring1+one-rank to the full king_ring2
                                        // (2-ring), so attackers staging one square further out are detected.
                                        // 0 = narrow zone (byte-identical baseline); 1 = wide 2-ring.
    // PER-KING DYNAMIC magnitude: scale EACH king's danger by how REAL its attack is = the CO-OCCURRENCE of
    // its own signature detectors (attackers acting THROUGH open lines / undefended holes), not their additive
    // sum. Computed per king, so the genuinely-attacked king scales UP (toward the crusher regime) while the
    // safe king scales DOWN -> the netted term reflects the true asymmetry instead of cancelling. Realness =
    // att_cnt*(open_files+weak_squares) - KS_DYN_PIVOT, fed through mod_gain (clamped 0.5x..2.0x). 0 = off (byte-id).
    inline int KS_DYN          = 0;     // dynamic-magnitude coefficient (mod_gain k); 0 = no per-king scaling
    inline int KS_DYN_PIVOT    = 4;     // realness level treated as neutral (1.0x); below damps, above boosts
    inline int KS_DYN_SHIFT    = 4;     // sensitivity of the factor to realness (mod_gain right-shift)
    inline int KS_SHIELD       = 2;     // units subtracted per friendly pawn shielding the king on its three files
    inline int KS_DEFENDER     = 2;     // units subtracted per friendly PIECE (N/B/R/Q) defending the king zone
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
    inline int KS_FLOOR        = 0;     // DEADZONE: attack-units below this -> ZERO danger, so TRIVIAL king-danger
                                        // can't perturb non-king positions (the def1 passer bleed). Default 0 = byte-id.
    inline int KS_PHASE_FULL   = 48;    // phase_score AT/BELOW which king safety is full weight (0=full material/opening)
    inline int KS_PHASE_ZERO   = 104;   // phase_score AT/ABOVE which king safety is ~0 (128=bare kings/deep endgame)

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

    // Gap-P P2: run the per-passer king-race realizability (advanced_endgame_eval's passer block,
    // extracted to passer_realizability_delta) in ALL phases, not just deep endgame, so an advancing
    // passer's danger is seen in the midgame (the Tal-bot a-pawn march). When on, the in-AE copy is
    // skipped (de-dup). PASSER_KRACE_MG_PCT = the midgame weight in percent; it ramps to 100 (full,
    // = old AE behavior) at the deep-endgame phase. Default off = byte-identical.
    inline bool ENABLE_PASSER_KRACE_MG = false;
    inline int PASSER_KRACE_MG_PCT = 100;

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

    // Node-entry reverse futility pruning (static null) + a null-move eval gate. Both read ONE node-entry
    // static eval (eval_by_mode); default-off = byte-identical (see minimizer/maximizer node entry). RFP
    // fires only at non-PV (beta-alpha==1), not-in-check, non-mate, shallow-remaining nodes.
    inline bool ENABLE_RFP = true;      // node-entry reverse futility (static null) -- SHIPPED (+73 Elo SPRT)
    inline int RFP_MARGIN = 1500;       // milli-pawn margin PER remaining ply (pawn=1000); shipped value
    inline int RFP_MIN_DEPTH = 1;       // fire only when (depth_limit - cur_depth) >= RFP_MIN_DEPTH (skip leaf-adjacent rd)
    inline int RFP_MAX_DEPTH = 6;       // fire only when (depth_limit - cur_depth) in [RFP_MIN_DEPTH, RFP_MAX_DEPTH]
    inline int RFP_EVAL_MODE = 0;       // eval_by_mode arg for the RFP/gate eval: 0=full, 1=cheap
    inline bool ENABLE_NULL_EVAL_GATE = false; // only attempt null move when static eval is past beta/alpha

    // qsearch quiet-check cost (buildNoisyMoveList). Default off = byte-identical (full board-copy +
    // is_check per quiet move at every q-ply). QCHECK_DEPTH0: include quiet checks only at the first
    // q-ply (qDepth==0), mainstream practice -- shrinks the q-tree. QCHECK_MASK: detect direct checks
    // with a bitboard attack test from the destination square (reusing the ENABLE_CHECK_ORDER logic)
    // instead of simulating the move; misses discovered checks (standard accepted tradeoff). Behavioral.
    inline bool ENABLE_QCHECK_DEPTH0 = false;
    inline bool ENABLE_QCHECK_MASK = false;

    // Phase B endgame colour-asymmetry fixes + dead-pin revival (adversarial rescan). All behavioral
    // -> gated default-off. CAPGAIN: capture-gains pawn_rank sign in the black branch. ROOK_DBLCOUNT:
    // endgame black-rook extra unconditional rookIncrement add (no white mirror). KNIGHT_MOB: endgame
    // knight mobility bonus is 15 for black vs 10 for white.
    inline bool ENABLE_CAPGAIN_PAWN_FIX = true;
    inline bool ENABLE_ROOK_DBLCOUNT_FIX = true;   // SHIPPED (collapse bundle, with SYM_UP)
    inline bool ENABLE_KNIGHT_MOB_FIX = false;

    // Symmetrize-UP counterparts to the KNIGHT_MOB / ROOK_DBLCOUNT colour asymmetries: instead of
    // collapsing black DOWN to white's value (the _FIX knobs), raise WHITE up to black's higher
    // magnitude so the term is colour-symmetric at the larger value (tests whether the magnitude, not
    // the asymmetry, carried the eval signal). KNIGHT_MOB_SYM_UP: endgame white knight base mobility
    // 10 -> 15. ROOK_DBLCOUNT_SYM_UP: add white's missing extra (7-rank)*35 "rook behind enemy pawn"
    // term. Behavioral -> gated default-off.
    inline bool ENABLE_KNIGHT_MOB_SYM_UP = false;
    inline bool ENABLE_ROOK_DBLCOUNT_SYM_UP = true;   // SHIPPED (collapse bundle): rook double-count symmetric-up

    // Batch 1 endgame-asymmetry tail. ROOK_ENDGAME_CAP: evaluate_rooks_endgame applies rookIncrement
    // UNCAPPED in both colour branches (reaching ~725 on a behind-passer file), unlike
    // evaluate_rooks_midgame which clamps std::min(rookIncrement, 300) -- suspected endgame rook/material
    // over-valuation; the fix clamps the endgame increment to ROOK_ENDGAME_CAP (magnitude fix, colour-
    // symmetric). ROOK_RANKWIN_FIX: in evaluate_rooks_midgame the white own-pawn rank window (< 5, with
    // the term flipping to a +75 bonus at rank 4) and the black mirror (> 4) are not true mirrors at the
    // boundary; the fix aligns them. Both behavioral -> gated default-off.
    inline bool ENABLE_ROOK_ENDGAME_CAP = false;
    inline int ROOK_ENDGAME_CAP = 300;
    inline bool ENABLE_ROOK_RANKWIN_FIX = false;

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
inline std::vector<Move>& buildMoveListFromReordered(std::vector<BoardState> &state_history, uint64_t zobrist, int cur_ply, Move prevMove);
inline std::vector<Move>& buildNoisyMoveList(uint64_t zobrist, std::vector<BoardState> &state_history, int cur_ply, int qDepth, Move prevMove);

#endif // SEARCH_ENGINE_H