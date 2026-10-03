/* === SHIPPED FITTED TABLES (2026-10-03) -- compiled in so V2_PRESET=shipped needs no file at runtime ===================
 * GENERATED from the fit outputs; do not hand-edit (regenerate from the files below).
 *  KAUF_SHIP_* : Kaufman census cells, Texel-fitted on SF18 labels (`diagnostics/_texel_kauf_fit.py`, arm CELLS lambda 0.01,
 *               E:/chess_data/texel/kauf_full.txt). Millipawns per unit, White-POV; index 0=pair 1=P 2=N 3=B 4=R 5=Q.
 *               Calibrated SF18@800 +13.1 over 2,000 paired, self-play +11.8 +- 17.9 (C3 doc 18m/18p).
 *  KSB_SHIP_*  : KS-B shelter/storm cells, JOINT KS + KS-B depth fit (`diagnostics/_ks_depth_fit.py` KS_LAMBDA=1e-2,
 *               E:/chess_data/texel/ks_depth_L1e-2_ksb.txt). v2_features index k (106-161), leg (0 = mg), millipawns.
 *               Ships WITH the KS knobs of the same fit (search_engine.cpp preset). Calibrated +6.0 / self-play +17.7 (18o/18q).
 */
#pragma once
static constexpr int KAUF_SHIP_OURS[6][6] = {
	{     4,     0,     0,     0,     0,     0 },
	{    28,     8,     0,     0,     0,     0 },
	{     4,    32,    -3,     0,     0,     0 },
	{     8,    12,     3,    -6,     0,     0 },
	{     8,    21,     1,     2,    -1,     0 },
	{    -4,    -7,    -3,    -6,    -6,    -2 }
};
static constexpr int KAUF_SHIP_THEIRS[6][6] = {
	{     0,     0,     0,     0,     0,     0 },
	{   -26,     0,     0,     0,     0,     0 },
	{    -3,    24,     0,     0,     0,     0 },
	{    -2,    25,    -4,     0,     0,     0 },
	{   -11,    28,    -4,     0,     0,     0 },
	{     2,    -2,    -1,     2,    -1,     0 }
};
static constexpr int KSB_SHIP_N = 56;
static constexpr int KSB_SHIP_K[KSB_SHIP_N] = { 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161 };
static constexpr int KSB_SHIP_LEG[KSB_SHIP_N] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static constexpr int KSB_SHIP_V[KSB_SHIP_N] = { -11, 52, 161, 111, -108, 36, 69, 248, 222, -61, 36, -75, 166, 169, 22, 85, -42, -70, 12, 74, 9, 57, 77, -9, 52, 17, 14, 7, -26, 57, 17, 96, 6, -64, -87, -41, -92, 25, -13, 30, 20, -20, -98, 57, 50, -74, -35, 35, -28, -117, -83, 16, 42, -10, -2, -20 };
