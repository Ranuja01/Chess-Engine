/* eval_v2.h

@author: Ranuja Pinnaduwage

Public surface of the ground-up second static evaluation (eval_v2.cpp). Selected at runtime by
Config::EVAL_ARM; the shipped eval is untouched and remains the byte-identical control arm in the same
binary. See eval_v2.cpp's header for the design contract and the build-up ladder.

*/

#ifndef EVAL_V2_H
#define EVAL_V2_H

#include <cstdint>

/*
	The v2 static evaluation. Same 12 arguments as the shipped eval and the SAME output contract:
	ABSOLUTE Black-positive milli-pawns (pawn = 1000) for a NON-TERMINAL position, before the single
	side-to-move flip that the caller applies once at search_engine.cpp:9045-9046.
	⚠️ That flip keys on Config::side_to_play -- the ROOT colour, latched once per search at
	search_engine.cpp:1524 -- so it is a search-wide CONSTANT, not a per-node side-relative conversion
	(this is not negamax). Anything here that depends on whose move it is must therefore key on the
	`turn` argument directly, in absolute Black-positive space.

	Which rung of the build-up ladder is evaluated is selected by Config::EVAL_V2_RUNG.
*/
int placement_and_piece_eval_v2(int moveNum, bool turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied, uint64_t castling_rights);

/*
	Initialise v2's tapered piece-square tables (Config::PST_V2_TAPERED) once at engine init, after the shared
	placement layer has been rebuilt. Defaults reproduce the shipped eval; PST_V2_FILE=<path> loads fitted
	tables and PST_V2_DUMP=<path> writes the active ones. File format: 768 whitespace-separated integers in
	millipawns, White's point of view -- for each piece type (pawn, knight, bishop, rook, queen, king) the 64 mg
	values then the 64 eg values, square order a1..h1, a2..h2, ..., a8..h8. '#' starts a comment line.
*/
void v2_pst_init();

/*
	TEXEL FIT C1 feature extractor (diagnostic; never called from search). For one position, writes each side's
	raw feature COUNTS for the linear C1 blocks -- mobility, pawn structure, passers, placement -- so a fitter can
	express those blocks as sum(count x parameter). The PST is not included (the fitter counts occupancy itself).
	Layout: out[s * V2F_PER_SIDE + k], s = 0 White / 1 Black, then out[2 * V2F_PER_SIDE] = flags
	(1 = draw classifier / KPK row, 2 = tier-2b row, 4 = a live knob this extractor does not model).
	Parameter k order: mobility 0..65 (N 0-8, B 9-22, R 23-37, Q 38-65 by move count) · doubled 66 · isolated per
	file 67-74 · backward 75 · weak-unopposed 76 · passed by relative rank 77-84 · candidate by rank 85-92 ·
	passer king terms 93-96 (passed: sum kdist_them*w, sum kdist_us*w; candidate: same) · outpost N 97 ·
	outpost B 98 · behind 99 · bad-bishop file classes 100-103 · trapped-rook units 104 · weak queen 105.
	v2_features_theta writes each parameter's STARTING value (mg, eg) in millipawns from the live Config, so every
	unit conversion stays here in C++. Blocks: see diagnostics/_texel_feature_pass.py.
*/
constexpr int V2F_PER_SIDE = 106;
void v2_features(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                 uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t castling_rights,
                 long long *out);
void v2_features_theta(double *mg, double *eg);

/* Load Texel C1 fitted values (Config::C1_V2_FIT, env C1_V2_FILE) once at engine init, after every v2 knob. */
void v2_c1_init();

/*
	SHADOW-arm instrumentation (Config::EVAL_ARM == 2 only). v1's value is what search uses; these record
	the v2-v1 disagreement distribution so a full arm comparison can be taken over the real search
	distribution at zero risk. No-ops on every other arm.
*/
/*
	DETECTOR ORACLE PROBE (diagnostic; never called from search). Exports rung 2a's Layer A masks so they
	can be compared mask-for-mask against the independent Python implementation in
	diagnostics/_pawn_term_overlap.py. A detector bug and a scoring bug are indistinguishable from outside
	-- both read as "the eval moved" -- and this is the only correctness check we have that does not
	depend on any constant being right.

	`out` must have room for 27 entries; layout is documented at the definition in eval_v2.cpp.
*/
void pawn_entry_probe(uint64_t pawns, uint64_t occupied_white, uint64_t occupied_black, uint64_t *out);

/*
	SLICE 2 MOBILITY DETECTOR PROBE (diagnostic). Per-side, per-type area-filtered counts, raw SF11 table sums
	and the area masks, for comparison against diagnostics/_mobility_detector_oracle.py. `out` needs 14
	entries. ⚠️ Reads KS_V2_XRAY and MOB_V2_EXCL_*, so build an engine under the arm's environment first.
*/
void mobility_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                    uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, long long *out);

/*
	EXACT KPK BITBASE PROBE (diagnostic). Strong side normalised to White, pawn on files a-d, ranks 2-7.
	Returns 1 = win, 0 = draw, -1 = outside the normalised domain. Compared against every state of
	diagnostics/_kpk_oracle.py --all-files --engine.
*/
int kpk_probe(int wksq, int wpsq, int bksq, int strong_to_move);

/*
	SLICE 2 PLACEMENT DETECTOR PROBE (diagnostic). Per side: outpost knights, outpost bishops, reachable-outpost
	knights, minors behind a pawn, bad-bishop units, long-diagonal bishops, trapped-rook units, weak queens, latent
	bishop hits, latent rook hits -- White 0-9, Black 10-19. `out` needs 20. Form-dependent packing is documented at the
	definition. ⚠️ Reads KS_V2_XRAY, MOB_V2_EXCL_* and the *_V2_FORM knobs; build an engine first.
*/
/*
	DETECTOR ORACLE PROBE for slice-3 space (diagnostic; never called from search). `out` needs 6 entries:
	0-1 per-side safe-square COUNTS (White, Black, after the BEHIND double count), 2-3 per-side piece counts used by
	the weight, 4 the Black-positive millipawn score, 5 phase256. ⚠️ Reads every SPACE_V2_* knob plus KS_V2_XRAY
	(the attack occupancy its safe mask depends on), so build an engine under the arm's environment first.
	Compared against diagnostics/_space_detector_oracle.py.
*/
void space_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                 uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, long long *out);

/*
	DETECTOR ORACLE PROBE for slice-3 threats (diagnostic; never called from search). `out` needs 16 entries, per side
	W,B: 0-1 minor-threat victims · 2-3 rook-threat victims · 4-5 king-threat victims · 6-7 hanging · 8-9 restricted
	squares · 10-11 safe-pawn threats · 12-13 pawn-push threats · 14 the Black-positive millipawn score · 15 phase256.
	★ Comparing counts as well as the score means a compensating pair of errors cannot hide.
	☠️ CONTRACT (corrected 2026-09-17): the LEG COUNTS ARE UNCONDITIONAL -- every leg's detector count is filled whether or
	not that leg's knob is on, so a disabled leg still reports what it WOULD contribute. Only the SCORE (out[14]) respects
	the knobs. The one filter applied to a count is PAWN_TARGETS, which is part of the minor/rook victim definition.
	⚠️ Reads every THREAT_V2_* knob plus KS_V2_XRAY (the attack occupancy its gates depend on), so build an engine under
	the arm's environment first. Compared against diagnostics/_threats_detector_oracle.py.
*/
void threats_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                   uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, long long *out);

/*
	KING-SAFETY COUNT PROBE (diagnostic; never called from search). `out` needs 12 entries, indexed by the KING under
	examination (0 = White's king, 1 = Black's king): 0-1 attacker COUNT (n_att) · 2-3 weighted attacker sum (w_att) ·
	4-5 weak zone squares · 6-7 attacks on king-adjacent squares · 8-9 safe-check squares (all four piece channels
	summed, per-square regardless of KS_V2_CHK_COUNT) · 10-11 the raw unit total from ks_units.
	★ WHY THIS EXISTS (2026-09-17): king safety is the ONE subsystem the collinearity gate cannot see -- flagged as a
	coverage hole since slice 2, and now load-bearing, because threats taxes KS-critical accuracy in proportion to its
	magnitude and "threats double-counts KS" is otherwise an untested story.
	⚠️ Reads every KS_V2_* knob plus KS_V2_XRAY. ☠️ Counts are UNCONDITIONAL (each channel is reported whether or not its
	knob is non-zero) -- the same contract as threats_probe, corrected there on 2026-09-17; only out[10-11] respects the
	knobs, because it IS the scored total.
*/
void ks_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
              uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, long long *out);

void placement_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                     uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t castling_rights,
                     long long *out);

void eval_v2_shadow_record(int v1, int v2);
void eval_v2_shadow_report();

#endif
