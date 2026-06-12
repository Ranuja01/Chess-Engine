/* see_selfcheck.cpp

@author: Ranuja Pinnaduwage

Standalone correctness check for the engine's static-exchange evaluator see()
(cpp_bitboard.h). It compares see() against two independent references over a set
of named cases plus a randomized fuzz:

  - see_truth():  brute-force exchange search. Tries EVERY attacker at each step
                  with declining always allowed, and recomputes attackers from
                  scratch after each removal so x-ray reveals are exact. This is
                  ground truth for "best value of optionally initiating the
                  capture", matching see()'s own model (pins/promotions ignored).
  - ref_see():    the canonical CPW swap-list SEE -- least-valuable-attacker by
                  true piece value, BOTH sides' attacker sets recomputed against
                  the live occupancy after every removal, canonical backward fold.

The gate to assert against is `see() == ref_see()`, NOT `see() == see_truth()`:
brute force with free attacker choice legitimately differs from a swap-list on a
small fraction of LVA-tie / reveal-order cases (~0.1% in fuzz), which is a model
edge, not a see() defect. ref_see() is what a corrected see() must match exactly.

Two known see() defects this exercises (see dev_notes / the adversarial-rescan
notes): (A) update_attackers_for_piece_removal refreshes only the side that just
captured, so an x-ray reveal for the recapturing side is missed; (B) the LVA pick
uses the global square_values[] array (stale outside the eval / on eval-cache
hits) instead of true piece values. Fix shape: recompute the side-to-move's
attacker set from live occupancy at the top of each loop iteration, and pick via
the already-present get_least_valuable_attacker_static().

Build & run (WSL, from NN Engine/):
  g++ -std=c++20 -O1 -march=native -fopenmp diagnostics/see_selfcheck.cpp \
      cpp_bitboard.cpp -I. -o /tmp/see_selfcheck && /tmp/see_selfcheck
*/

#include "cpp_bitboard.h"
#include "search_engine.h"
#include <cstdio>
#include <random>

extern std::array<int, 64> square_values;

static uint64_t B(int sq) { return 1ULL << sq; }

static uint64_t atk_of(bool side, uint8_t target, uint64_t occ, const BoardState& s) {
    return attackersMask(side, target, occ, (s.queens | s.rooks) & occ,
                         (s.queens | s.bishops) & occ, s.kings & occ,
                         s.knights & occ, s.pawns & occ, s.occupied_colour[side] & occ);
}

// Value for `stm` of OPTIONALLY capturing the piece (value tv) sitting on `target`.
static int exchange_after(uint8_t target, bool stm, uint64_t occ, int tv, const BoardState& s) {
    uint64_t atk = atk_of(stm, target, occ, s);
    int best = 0; // declining is always available
    while (atk) {
        uint8_t a = __builtin_ctzll(atk); atk &= atk - 1;
        int net = tv - exchange_after(target, !stm, occ & ~B(a), get_value_at(a, s), s);
        if (net > best) best = net;
    }
    return best;
}

// Ground truth matching see()'s semantics: `stm` initiates the capture on `target`
// (best choice of first attacker), then both sides play on optimally (may stop).
static int see_truth(uint8_t target, bool stm, const BoardState& s) {
    uint64_t occ = s.occupied;
    int tv = get_value_at(target, s);
    uint64_t atk = atk_of(stm, target, occ, s);
    if (!atk) return 0;
    int best = -1000000000;
    while (atk) {
        uint8_t a = __builtin_ctzll(atk); atk &= atk - 1;
        int net = tv - exchange_after(target, !stm, occ & ~B(a), get_value_at(a, s), s);
        if (net > best) best = net;
    }
    return best;
}

// CPW-reference swap SEE: LVA by true piece value (tie -> lowest square), both sides'
// attacker sets recomputed against the live occupancy every step, canonical fold.
static int ref_see(uint8_t target, bool stm, const BoardState& s) {
    int gain[40];
    int d = 0;
    uint64_t occ = s.occupied;
    gain[0] = get_value_at(target, s);
    bool side = stm;
    while (true) {
        uint64_t atk = atk_of(side, target, occ, s);
        if (!atk) break;
        int bestSq = -1, bestV = 1 << 30;
        uint64_t t = atk;
        while (t) {
            int q = __builtin_ctzll(t); t &= t - 1;
            int v = get_value_at(q, s);
            if (v < bestV) { bestV = v; bestSq = q; }
        }
        ++d;
        gain[d] = bestV - gain[d - 1];
        occ &= ~B(bestSq);
        side = !side;
    }
    while (--d > 0) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

static void fill_square_values_fresh(const BoardState& s) {
    for (int sq = 0; sq < 64; ++sq) square_values[sq] = get_value_at(sq, s);
}

static BoardState mk(uint64_t p, uint64_t n, uint64_t b, uint64_t r, uint64_t q,
                     uint64_t k, uint64_t ow, uint64_t ob) {
    return BoardState(p, n, b, r, q, k, ow, ob, ow | ob, 0, true, 0, -1, 0, 1);
}

int main() {
    initialize_attack_tables();

    // Exercise the corrected see() (set false to reproduce the original bugs: CASE A +1000, fuzz 2.16%).
    Config::ENABLE_SEE_FIX = true;
    printf("ENABLE_SEE_FIX = %d\n\n", (int)Config::ENABLE_SEE_FIX);

    // CASE A: one-sided refresh. 4k3/8/8/3p4/4B3/8/6b1/4K3 w, target d5.
    // White Be4 x d5(pawn); removing e4 reveals BLACK Bg2 -> recapture. Black's
    // attacker set is never refreshed after White's capture, so see() misses it.
    {
        uint64_t pawns = B(35), bish = B(28) | B(14), kings = B(4) | B(60);
        uint64_t ow = B(28) | B(4), ob = B(35) | B(14) | B(60);
        BoardState s = mk(pawns, 0, bish, 0, 0, kings, ow, ob);
        fill_square_values_fresh(s); // fresh values: isolates the refresh bug from the stale-pick bug
        printf("CASE A  (reveal)        see=%6d   ref_see=%6d   truth=%6d\n",
               see(35, true, s), ref_see(35, true, s), see_truth(35, true, s));
    }

    // CASE B: stale square_values pick. 7k/8/4p3/3p4/4P3/8/8/3Q3K w, target d5.
    // Attackers Qd1 + Pe4; defender pe6. Correct: PxP pxP QxP = +1000.
    // With square_values zeroed (program start / eval-cache-hit staleness), the
    // "least valuable" pick degenerates to lowest square index = the QUEEN.
    {
        uint64_t pawns = B(28) | B(35) | B(44), queens = B(3), kings = B(7) | B(63);
        uint64_t ow = B(28) | B(3) | B(7), ob = B(35) | B(44) | B(63);
        BoardState s = mk(pawns, 0, 0, 0, queens, kings, ow, ob);

        square_values.fill(0); // stale (as at startup, or after eval-cache hit elsewhere)
        int stale = see(35, true, s);
        fill_square_values_fresh(s); // values as the eval would have left them for THIS position
        int fresh = see(35, true, s);
        printf("CASE B  (stale pick)    see[stale sq_vals]=%6d   see[fresh]=%6d   ref_see=%6d   truth=%6d\n",
               stale, fresh, ref_see(35, true, s), see_truth(35, true, s));
    }

    // CASE C: fold sanity, 3-capture sequence, no reveals. Target d5 pawn;
    // white Pc4+Pe4 vs black pe6 defender. PxP pxP PxP = +1000.
    {
        uint64_t pawns = B(26) | B(28) | B(35) | B(44), kings = B(7) | B(63);
        uint64_t ow = B(26) | B(28) | B(7), ob = B(35) | B(44) | B(63);
        BoardState s = mk(pawns, 0, 0, 0, 0, kings, ow, ob);
        fill_square_values_fresh(s);
        printf("CASE C  (fold, 3 caps)  see=%6d   ref_see=%6d   truth=%6d\n",
               see(35, true, s), ref_see(35, true, s), see_truth(35, true, s));
    }

    // FUZZ: random sparse boards (kings parked in far corners so illegal king-
    // recapture tails don't muddy the model). square_values kept FRESH, so every
    // see()-vs-ref_see mismatch is attributable to the one-sided refresh, and
    // ref_see-vs-truth gauges the swap-list model itself.
    {
        std::mt19937 rng(42);
        int trials = 0, see_vs_ref = 0, see_vs_truth = 0, ref_vs_truth = 0;
        for (int it = 0; it < 20000; ++it) {
            uint64_t pc[2][5] = {{0}}; // [side][P,N,B,R,Q]
            uint64_t used = B(0) | B(63);
            int npieces = 4 + (int)(rng() % 7);
            for (int i = 0; i < npieces; ++i) {
                int sq = 8 + (int)(rng() % 48); // ranks 2..7 region
                if (used & B(sq)) continue;
                used |= B(sq);
                pc[rng() & 1][rng() % 5] |= B(sq);
            }
            uint64_t ow = pc[1][0] | pc[1][1] | pc[1][2] | pc[1][3] | pc[1][4] | B(0);
            uint64_t ob = pc[0][0] | pc[0][1] | pc[0][2] | pc[0][3] | pc[0][4] | B(63);
            BoardState s = mk(pc[0][0] | pc[1][0], pc[0][1] | pc[1][1], pc[0][2] | pc[1][2],
                              pc[0][3] | pc[1][3], pc[0][4] | pc[1][4], B(0) | B(63), ow, ob);
            uint64_t targets = ob & ~s.kings;
            while (targets) {
                uint8_t tsq = __builtin_ctzll(targets); targets &= targets - 1;
                if (!atk_of(true, tsq, s.occupied, s)) continue;
                fill_square_values_fresh(s);
                int v1 = see(tsq, true, s), v2 = ref_see(tsq, true, s), v3 = see_truth(tsq, true, s);
                ++trials;
                see_vs_ref += (v1 != v2);
                see_vs_truth += (v1 != v3);
                ref_vs_truth += (v2 != v3);
            }
        }
        printf("FUZZ    trials=%d   see!=ref_see: %d (%.2f%%)   see!=truth: %d (%.2f%%)   ref_see!=truth: %d (%.2f%%)\n",
               trials, see_vs_ref, 100.0 * see_vs_ref / trials,
               see_vs_truth, 100.0 * see_vs_truth / trials,
               ref_vs_truth, 100.0 * ref_vs_truth / trials);
    }
    return 0;
}
