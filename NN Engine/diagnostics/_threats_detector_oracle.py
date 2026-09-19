# -*- coding: utf-8 -*-
"""Slice 3 THREATS DETECTOR ORACLE: eval v2's per-leg counts AND score vs an independent python-chess rebuild.

WHY: threats is the widest form surface in the slice -- TWO defence-gate definitions x SEVEN switchable legs -- and a
detector bug is indistinguishable from a scoring bug from outside (both read as "the eval moved"). This rebuilds every
leg from python-chess primitives and compares position by position.
★ It compares COUNTS as well as the final score. A score-only check passes on a compensating pair of errors (one leg
over-counting while a constant is too small) -- the hole the first version of the SPACE oracle had.
★ INDEPENDENT BY CONSTRUCTION: every attack set is rebuilt here from python-chess's own tables. Nothing is read from the
engine except `threats_counts`'s output.

☠️ The one shared input is the engine's PHASE (`phase256`), read back from the probe: v2 owns the phase and this oracle is
not re-deriving it. So a phase bug would pass here -- it is covered by the pawn/mobility oracles instead.

GATE FORMS mirrored (read from the same env the engine latches):
  THREAT_V2_GATE 0 = SF `stronglyProtected = their pawn attacks | (their attackedBy2 & ~our attackedBy2)`; minors are paid
                     on `defended | weak`, every other leg on `weak` only (SF11 evaluate.cpp:494-519).
                 1 = Ethereal `poorlyDefended` (victim's view; PAWN support overrides):
                     `(attacked[THEM] & ~attacked[US]) | (attackedBy2[THEM] & ~attackedBy2[US] & ~attackedBy[US][PAWN])`.
LEGS: minor / rook victims (always) · KING · HANGING · RESTRICT · PAWN_TARGETS (filters pawn victims out of minor+rook) ·
      PUSH. Safe-pawn threats are always on (4/4 references, the largest constant in every one).

USAGE (knobs latch at engine init; KEY=VAL args exported before load):
  pyrun diagnostics/_threats_detector_oracle.py EVAL_ARM=1 KS_V2_XRAY=1 THREAT_V2_PCT=25 [THREAT_V2_GATE=0|1]
      [THREAT_V2_HANGING=1] [THREAT_V2_RESTRICT=1] [THREAT_V2_KING=1] [THREAT_V2_PAWN_TARGETS=1] [THREAT_V2_PUSH=1] [N=2000]
Run BOTH gate forms and every leg at least once. Exit 0 = zero mismatches AND non-vacuous (the term fired somewhere).
"""
import os, sys, csv, random

N = 2000
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        if k == "N":
            N = int(v)
        else:
            os.environ[k] = v

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)
import chess, ChessAI


def env_int(name, default):
    return int(os.environ.get(name, str(default)) or default)


def env_on(name):
    return os.environ.get(name, "0") not in ("0", "", "false")


PCT = env_int("THREAT_V2_PCT", 0)
GATE = env_int("THREAT_V2_GATE", 0)
HANG = env_on("THREAT_V2_HANGING")
RESTRICT = env_on("THREAT_V2_RESTRICT")
KING = env_on("THREAT_V2_KING")
PAWN_T = env_on("THREAT_V2_PAWN_TARGETS")
PUSH = env_on("THREAT_V2_PUSH")
XRAY = env_on("KS_V2_XRAY")
print("[knobs] " + " ".join(a for a in sys.argv[1:] if "=" in a))
print("[oracle reads] PCT=%d GATE=%d HANGING=%d RESTRICT=%d KING=%d PAWN_TARGETS=%d PUSH=%d XRAY=%d"
      % (PCT, GATE, HANG, RESTRICT, KING, PAWN_T, PUSH, XRAY))
if PCT <= 0:
    print("☠️ THREAT_V2_PCT=0 -- the term is OFF, so this run would compare 0 against 0 and PASS VACUOUSLY. Set PCT.")
    sys.exit(1)

# Mirrors eval_v2.cpp TH_* (SF11 evaluate.cpp:116-121,:133-147, each leg pawn-converted by its OWN phase's pawn).
MINOR_MG = [47, 461, 617, 703, 617]
MINOR_EG = [150, 192, 263, 559, 756]
ROOK_MG = [23, 297, 297, 0, 398]
ROOK_EG = [207, 333, 286, 178, 178]
KING_MG, KING_EG = 187, 418
HANG_MG, HANG_EG = 539, 169
SAFEPAWN_MG, SAFEPAWN_EG = 1352, 441
PUSH_MG, PUSH_EG = 375, 183
RESTRICT_MG, RESTRICT_EG = 55, 33
VICTIM = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def pawn_attacks(pawns, color):
    m = 0
    for sq in chess.scan_forward(pawns):
        m |= chess.BB_PAWN_ATTACKS[color][sq]
    return m


def maps(b, color):
    """Per-type attack maps, their union, and the DOUBLY-attacked set -- the engine's SideAttacks.by[] / .all / .dbl.
    Occupancy follows KS_V2_XRAY exactly: bishops see through ALL queens, rooks through all queens + own rooks."""
    occ, own = b.occupied, b.occupied_co[color]
    by = {}
    allm = 0
    dbl = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        m = 0
        for sq in chess.scan_forward(b.pieces_mask(pt, color)):
            if pt == chess.PAWN:
                a = chess.BB_PAWN_ATTACKS[color][sq]
            elif pt == chess.KNIGHT:
                a = chess.BB_KNIGHT_ATTACKS[sq]
            elif pt == chess.KING:
                a = chess.BB_KING_ATTACKS[sq]
            elif pt == chess.BISHOP:
                o = (occ ^ b.queens) if XRAY else occ
                a = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & o]
            elif pt == chess.ROOK:
                o = (occ ^ b.queens ^ (b.rooks & own)) if XRAY else occ
                a = (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & o]
                     | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & o])
            else:
                a = (chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
                     | chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
                     | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])
            # ⚠️ dbl must accumulate as the engine does: each piece's attacks intersected with the union SO FAR.
            dbl |= allm & a
            allm |= a
            m |= a
        by[pt] = m
    return by, allm, dbl


def victim_of(b, bit):
    for i, pt in enumerate(VICTIM):
        if bit & b.pieces_mask(pt, chess.WHITE) or bit & b.pieces_mask(pt, chess.BLACK):
            return i
    return 5 if (bit & b.kings) else -1


def reference(b, phase256):
    side = {}
    counts = {k: [0, 0] for k in ("minor", "rook", "king", "hanging", "restricted", "safepawn", "push")}
    M = {c: maps(b, c) for c in (chess.WHITE, chess.BLACK)}
    for color in (chess.WHITE, chess.BLACK):
        idx = 0 if color == chess.WHITE else 1
        them_c = not color
        own, them = b.occupied_co[color], b.occupied_co[them_c]
        their_np = them & ~b.pawns
        us_by, us_all, us_dbl = M[color]
        th_by, th_all, th_dbl = M[them_c]
        if GATE == 1:
            poorly = (th_all & ~us_all) | (th_dbl & ~us_dbl & ~us_by[chess.PAWN])
            weak = them & poorly & us_all
            protected_set = them & ~poorly
        else:
            protected_set = th_by[chess.PAWN] | (th_dbl & ~us_dbl)
            weak = them & ~protected_set & us_all
        defended = their_np & protected_set
        mg = eg = 0
        for bit in chess.scan_forward((defended | weak) & (us_by[chess.KNIGHT] | us_by[chess.BISHOP])):
            bit = 1 << bit
            v = victim_of(b, bit)
            if v < 0 or v > 4 or (v == 0 and not PAWN_T):
                continue
            counts["minor"][idx] += 1
            mg += MINOR_MG[v]; eg += MINOR_EG[v]
        for bit in chess.scan_forward(weak & us_by[chess.ROOK]):
            bit = 1 << bit
            v = victim_of(b, bit)
            if v < 0 or v > 4 or (v == 0 and not PAWN_T):
                continue
            counts["rook"][idx] += 1
            mg += ROOK_MG[v]; eg += ROOK_EG[v]
        # ☠️ COUNTS ARE UNCONDITIONAL, SCORING IS GATED -- this is the probe's actual contract (`threats_probe` fills
        # every leg's detector count before scoring, so you can see what a disabled leg WOULD contribute). The first
        # version of this oracle filled a count only when its knob was on and produced a false mismatch on every
        # position where a disabled leg had a non-zero detector. ★ SECOND INSTANCE of this exact bug class in one week
        # (the space oracle had it for the material gate) -- when a probe and an oracle disagree only in COUNT slots
        # while the SCORES match exactly, suspect the gating convention, not the detector.
        n = bin(weak & us_by[chess.KING] & ~b.pawns).count("1")
        counts["king"][idx] = n
        if KING:
            mg += n * KING_MG; eg += n * KING_EG
        n = bin(weak & ((~th_all & chess.BB_ALL) | (their_np & us_dbl))).count("1")
        counts["hanging"][idx] = n
        if HANG:
            mg += n * HANG_MG; eg += n * HANG_EG
        n = bin(th_all & ~protected_set & us_all).count("1")
        counts["restricted"][idx] = n
        if RESTRICT:
            mg += n * RESTRICT_MG; eg += n * RESTRICT_EG
        safe = (~th_all & chess.BB_ALL) | us_all
        our_p = b.pawns & own
        patt = pawn_attacks(our_p & safe, color)
        n = bin(patt & their_np).count("1")
        counts["safepawn"][idx] = n
        mg += n * SAFEPAWN_MG; eg += n * SAFEPAWN_EG
        empty = ~b.occupied & chess.BB_ALL
        p_att = pawn_attacks(b.pawns & them, them_c)
        push = ((our_p << 8) if color == chess.WHITE else (our_p >> 8)) & empty & chess.BB_ALL
        rank3 = chess.BB_RANK_3 if color == chess.WHITE else chess.BB_RANK_6
        push |= (((push & rank3) << 8) if color == chess.WHITE else ((push & rank3) >> 8)) & empty & chess.BB_ALL
        push &= ~p_att & safe
        n = bin(pawn_attacks(push, color) & their_np).count("1")
        counts["push"][idx] = n
        if PUSH:
            mg += n * PUSH_MG; eg += n * PUSH_EG
        m, g = mg * PCT // 100, eg * PCT // 100
        side[color] = (m * phase256 + g * (256 - phase256)) >> 8
    return side[chess.BLACK] - side[chess.WHITE], counts


seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)   # latches knobs + builds the attack tables

fens = []
with open(os.path.join(ENGINE, "diagnostics", "ks_sets", "game_regret_set.csv"), newline="") as f:
    fens = [r["fen"] for r in csv.DictReader(f) if r.get("fen")]
random.Random(20260917).shuffle(fens)
fens = fens[:N]
# Hand cases: a hanging piece, a pawn forking two pieces, a push threat, a double-attacked defender, a bare endgame.
fens += [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w - - 4 4",
    "4k3/8/8/3nn3/4P3/8/8/4K3 w - - 0 1",
    "4k3/8/4nn2/8/4P3/8/8/4K3 w - - 0 1",
    "r3k2r/8/8/3q4/3Q4/8/8/R3K2R w - - 0 1",
    "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
]

bad = fired = 0
for fen in fens:
    b = chess.Board(fen)
    got = ChessAI.threats_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                                 b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])
    ref, rc = reference(b, got["phase256"])
    if got["score"]:
        fired += 1
    mismatch = ref != got["score"]
    for k in rc:
        if tuple(rc[k]) != got[k]:
            mismatch = True
    if mismatch:
        bad += 1
        if bad <= 10:
            print("MISMATCH %s" % fen)
            print("  engine score %d  %s" % (got["score"], {k: got[k] for k in rc}))
            print("  oracle score %d  %s" % (ref, {k: tuple(rc[k]) for k in rc}))

n = len(fens)
print("\npositions=%d  mismatches=%d  fired=%d (%.1f%%)" % (n, bad, fired, 100.0 * fired / n))
vacuous = fired < 0.25 * n
if vacuous:
    print("☠️ VACUOUS: the term scored on under a quarter of positions -- check the knobs actually reached the engine")
print("RESULT: %s" % ("PASS" if bad == 0 and not vacuous else "FAIL"))
sys.exit(0 if bad == 0 and not vacuous else 1)
