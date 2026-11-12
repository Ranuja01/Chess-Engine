/*
 * sv2_probe.h — S3 step 0 counters (dev_notes/active/S3-MOVE-SOURCE-DESIGN-2026-11-11.md §6, the move-generator audit's §7):
 * what today's move source does on the search v2 path, measured before any arm is built. Compiled only with
 * -DSV2_MOVE_PROBE; the probe reads the search's state and keeps one side table of its own, so a probe build must still
 * bench 247 / 39,861,964. `bench` prints the summary.
 *
 * @author: Ranuja Pinnaduwage
 */
#pragma once

#ifdef SV2_PRUNE_VERIFY
// SV2_PRUNE_EARLY's check (-DSV2_PRUNE_VERIFY): every early prune is also made and scored the normal way; a mismatch is a
// prune the normal path would not have made. Must stay 0, and the build must still bench the production fingerprint.
inline long long g_sv2_prune_checked = 0, g_sv2_prune_mismatch = 0;
#endif

#ifdef SV2_SEE_VERIFY
// see_move's check (-DSV2_SEE_VERIFY): where the moving piece is the side's least valuable attacker, see_move must equal see.
inline long long g_sv2_see_checked = 0, g_sv2_see_mismatch = 0;
#endif

#ifdef MOVEGEN_FAST_VERIFY
// MOVEGEN_FAST's check (-DMOVEGEN_FAST_VERIFY): every fast piece-type pass is also run the normal way; a mismatch is a list
// (or an order) that differs from today's. Must stay 0, and the build must still bench the production fingerprint.
inline long long g_mgf_checked = 0, g_mgf_mismatch = 0;
#endif

#ifdef SV2_MOVE_PROBE
#include <cstdint>

struct Sv2MoveProbe
{
	// interior nodes (v2 negamax) that build a move list, by move-cache state at the build
	long long lists = 0, lists_hit = 0;
	long long moves_main = 0;              // moves in those lists
	// cutoff index histogram, by cache state: buckets 0, 1, 2, 3-4, 5-9, 10+; plus nodes with no cutoff
	long long cut_hit[6] = {}, cut_miss[6] = {}, nocut_hit = 0, nocut_miss = 0;
	// a remembered move (the probe's own table: the cutoff move, or the best move of a node that raised alpha, by position)
	long long rem_miss_have = 0;           // cache-miss nodes where a remembered move exists and is in the list
	long long rem_miss_at0 = 0;            // ... and today's sort already put it first
	long long rem_miss_cut = 0;            // ... and the node cut on it
	long long rem_miss_cut_at0 = 0;        // ... cut on it and it was first
	long long rem_miss_cut_any = 0;        // cache-miss nodes with a remembered move that cut at all
	long long rem_hit_have = 0, rem_hit_head = 0;   // cache-hit nodes with a remembered move; ... equal to the cached head
	// moves made only to be pruned (score_move returns after make_move): calls, and prunes by remaining depth 1, 2, 3, 4+
	long long scored = 0, pruned_lmp[4] = {}, pruned_hist[4] = {}, pruned_see[4] = {}, pruned_seec[4] = {}, pruned_fut[4] = {};
	// qsearch lists (buildNoisyMoveList) and their moves
	long long q_lists = 0, q_moves = 0;
	// the scorer's exchange branch (victim cheaper than attacker): captures, and those where a cheaper piece of ours also
	// attacks the square, so the per-square exchange value describes that piece's capture, not this one (audit C3)
	long long see_branch = 0, see_cheaper_attacker = 0;
	// full lists sorted by the generator, and those where a non-capture sorts above a good-tier capture (option B)
	long long sorted_lists = 0, quiet_above_good_capture = 0;
};
inline Sv2MoveProbe g_sv2_probe{};

// The probe's remembered-move table: 2^22 slots of (key, move), last writer wins.
struct Sv2ProbeSlot { uint64_t key; uint8_t from, to, promo; };
inline constexpr int SV2_PROBE_BITS = 22;   // the TT's capacity (2^22), so a miss here is not an artefact of a small table
inline Sv2ProbeSlot g_sv2_probe_slots[1u << SV2_PROBE_BITS] = {};
#endif
