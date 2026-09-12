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
	side-to-move flip that the caller applies once at search_engine.cpp:8910.

	Which rung of the build-up ladder is evaluated is selected by Config::EVAL_V2_RUNG.
*/
int placement_and_piece_eval_v2(int moveNum, bool turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied, uint64_t castling_rights);

/*
	SHADOW-arm instrumentation (Config::EVAL_ARM == 2 only). v1's value is what search uses; these record
	the v2-v1 disagreement distribution so a full arm comparison can be taken over the real search
	distribution at zero risk. No-ops on every other arm.
*/
void eval_v2_shadow_record(int v1, int v2);
void eval_v2_shadow_report();

#endif
