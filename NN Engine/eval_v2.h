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
	outpost B 98 · behind 99 · bad-bishop file classes 100-103 · trapped-rook units 104 · weak queen 105 ·
	C3-a king shelter cells 106-129 (file class F x 6 states) and pawn storm cells 130-161 (F x 8 states), counted at
	the king's actual square · C3-b 162-171 (nearest own pawn d 2/3/4/5+, nearest enemy pawn d 2/3/4/5+, flank empty,
	flank only enemy) · C3-c KingProtector 172-183 (knight d 1..6+, bishop d 1..6+) · PX passer system 184-234
	(layout at eval_v2.cpp px_counts; 2026-10-03) · CONN connected pawns 235-336 (2026-10-09): 96 main cells
	[rank 1..6][file class a-h/b-g/c-f/d-e][phalanx][opposed] (count of pawns) then 6 supporter cells by rank (count of
	supporters; layout at eval_v2.cpp conn_cell). All C3 cells are reported whether
	or not their knob is on (layouts: ksb_cells / kfl_cells / kprot_counts in eval_v2.cpp).
	v2_features_theta writes each parameter's STARTING value (mg, eg) in millipawns from the live Config, so every
	unit conversion stays here in C++. Blocks: see diagnostics/_texel_feature_pass.py.
	☠️ ChessAI.pyx hard-codes V2F_PER_SIDE and the 2*V2F_PER_SIDE+1 buffer size; change them together.
*/
constexpr int V2F_PER_SIDE = 337;   // 184 + 51 PX passer cells (2026-10-03) + 102 CONN cells (2026-10-09)
void v2_features(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                 uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t castling_rights,
                 long long *out);
void v2_features_theta(double *mg, double *eg);

/* Load Texel C1 fitted values (Config::C1_V2_FIT, env C1_V2_FILE) once at engine init, after every v2 knob. */
void v2_c1_init();

/* Load the C3 detector cell tables once at engine init: KSB_V2 (king shelter + pawn storm, env KSB_V2_FILE),
   KFL_V2 (pawnless flank + king-pawn distance, KFL_V2_FILE), KPROT_V2 (KingProtector, KPROT_V2_FILE). */
void v2_c3_init();

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
	KING-SAFETY COUNT PROBE (diagnostic; never called from search). `out` needs 70 entries (the layout below is the
	original 12; ChessAI.pyx ks_counts names all 35 pairs, incl. per-type safe checks at 36-43 and per-type attackers /
	x-ray attackers / contested share / x-ray weight at 44-69), indexed by the KING under
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

/* OvD eg WINNABILITY input probe (diagnostic): out[0..6] = passed, pawns, outflanking, infiltration, both_flanks,
   pawn_ending, almost_unwinnable (eval_v2.cpp win_inputs); out[7] = phase256. `out` needs 8. Knob-free. */
void win_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
               uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, long long *out);

void placement_probe(uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
                     uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t castling_rights,
                     long long *out);

void eval_v2_shadow_record(int v1, int v2);
void eval_v2_shadow_report();

/*
	LAZY EVAL (Config::LAZY_V2; dev_notes/active/LAZY-EVAL-RESEARCH-2026-10-30.md §6). A search site that would cut or prune once
	the eval clears a bound sets g_lazy_req before asking for the eval and clears it after. After its cheap part the v2 eval forms
	an estimate (the cheap part + parent_pos when known) and, when that clears the bound by the margin and the guards allow it,
	returns the estimate with g_lazy_fired set; the caller must not cache it. Every full v2 eval leaves its positional part
	(total - cheap part) in g_v2_last_pos, LAZY_POS_NA after an early return. All values Black-positive millipawns.
*/
constexpr int LAZY_POS_NA = -1000000000;
struct LazyReq {
	bool active;      // the search site asks for a lazy eval
	bool armed;       // set by get_board_evaluation around the one eval call whose result it will not cache; the eval reads only this
	bool above;       // true: lazy when estimate >= bound + margin; false: lazy when estimate <= bound - margin
	int  bound;
	int  parent_pos;  // the parent position's positional part, or LAZY_POS_NA
	int  scale;       // the site's margin scale, percent
};
extern LazyReq g_lazy_req;
extern bool g_lazy_fired;
extern int  g_v2_last_pos;

#ifdef LAZY_PROBE
/*
	LAZY EVAL PROBE (diagnostic build only: compiled with -DLAZY_PROBE, absent from every other build; research and plan in
	dev_notes/active/LAZY-EVAL-RESEARCH-2026-10-30.md). The v2 eval records each stage's contribution and some board features
	a lazy-exit guard could read before the attack build; the search logs a sample of them with the window of the site that
	asked for the eval. Values are Black-positive millipawns, like the eval's return. Assumes EVAL_V2_PAIR = 0 (the shipped
	path, where every stage adds to the running total).
*/
struct LazyProbe {
	bool fresh;       // an eval body ran since the search last read the probe
	int  special;     // 0 = normal, 1 = draw classifier, 2 = tier-2b replacement, 3 = MOPUP_V2 replacement (returned before any stage)
	int  cheap;       // material + PST + bishop pair + Kaufman + narrow material classes
	int  mob, space, threats, ks;               // the consumers of the shared attack build
	int  pawns, passers, rookfile, placement;   // the pawn-entry block
	int  c3, px, tempo, win;                    // king shelter / flank / protector, passer system, tempo, endgame scales
	int  total;                                 // the eval's return
	int  npm_w, npm_b, mat_w, mat_b, phase256;
	int  ksu_w, ksu_b;                          // king-safety units against White's king, against Black's king
	int  q_w, q_b, minors_w, minors_b, rooks_w, rooks_b, pawns_w, pawns_b;
	int  advp_w, advp_b;                        // passed pawns on the relative 6th / 7th rank
	int  near_wk, near_bk;                      // enemy knights, bishops, rooks, queens within 2 squares of each king
	bool turn;                                  // true = White to move
};
extern LazyProbe g_lazy_probe;
#endif

#endif
