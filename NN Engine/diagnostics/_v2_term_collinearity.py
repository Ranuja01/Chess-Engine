# -*- coding: utf-8 -*-
"""COLLINEARITY GATE for eval v2 terms: do two terms re-express the same signal?

WHY (owner charter, 2026-09-14): v2 exists to avoid v1's "~30 terms / ~2 signals". §I additivity is NOT
non-collinearity — two terms can add on accuracy and still largely say the same thing. This measures overlap
directly: per position, each term's DETECTOR value as White − Black, then the correlation matrix and the
variance inflation factor (VIF) of every term against all the others.

ONE-OWNER RULE: a flagged pair keeps the concept in the subsystem that measures better and REDEFINES the other
to be disjoint. Slice-2 suspects: trapped rook vs rook mobility's negative floor · bad bishop vs bishop mobility ·
outpost vs knight mobility.

WHAT IT COVERS (as of 2026-09-17, ALL FIVE scoring subsystems): mobility (per-type area counts + the raw SF11
table sums) · the eight placement detectors · threats' seven legs · king safety's six channels · pawn
structure's thirteen predicates. ★ Pawn structure was the last coverage hole; it closed WITHOUT a build,
because `pawn_entry_probe` / `ChessAI.pawn_masks` already existed for the rung-2 detector oracle and exports
every Layer A mask. Only the column set here was missing — check DIAGNOSTICS-TOOLKIT before writing a probe.
⚠️ STILL NOT COVERED: `PawnEntry.attacks2` (DOUBLE pawn attacks) is built at eval_v2.cpp:657 but is not among
the probe's 27 exported slots, so "does threats' stronglyProtected re-express the double-attack map?" remains
unmeasurable here. Exporting it is a probe change plus a rebuild; do it only if a run leaves that pair open.
Counts, not millipawns: correlation is scale-free, and counts avoid re-implementing the phase blend in Python.
⚠️ Mobility's raw table sum is concave in the counts, so the table-sum column is the one to compare against the
placement terms (it carries the negative floor that trapped rook might duplicate).

USAGE (knobs latch at engine init; KEY=VAL args exported before load):
  pyrun diagnostics/_v2_term_collinearity.py EVAL_ARM=1 KS_V2_XRAY=1 [N=2500] [R_FLAG=0.7] [VIF_FLAG=5]
Exit 0 = nothing flagged.
"""
import os, sys, csv, random

N, R_FLAG, VIF_FLAG = 2500, 0.7, 5.0
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1)
        if k == "N":
            N = int(v)
        elif k == "R_FLAG":
            R_FLAG = float(v)
        elif k == "VIF_FLAG":
            VIF_FLAG = float(v)
        else:
            os.environ[k] = v

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)
import chess, ChessAI
import numpy as np

print("[knobs] " + " ".join(a for a in sys.argv[1:] if "=" in a))
seed = chess.Board()
ai = ChessAI.ChessAI(None, None, seed, seed.turn)   # latches knobs + builds attack tables for the probes

# SETS= overrides the corpora (added 2026-09-17). ★ WHY: the gate's first KS-aware run sampled the four GENERAL corpora and
# found no threats×KS overlap — but the accuracy tax threats pays is concentrated on `lichess_ks_labelled`. Overlap is a
# property of a POPULATION, so "no overlap in general play" does not answer "no overlap where the tax appears".
CORPORA = os.environ.get("SETS", ",".join([
    "ks_sets/game_regret_set.csv", "ks_sets/game_regret_set_v2.csv",
    "ks_sets/game_regret_set_x4.csv", "ks_sets/game_regret_set_uho.csv"])).split(",")
PLACE = ["outpost_n", "outpost_b", "reach_n", "behind", "badb_units", "longdiag", "traprook_units", "weakq"]
# Slice 3 (added 2026-09-17): threats' per-leg detector counts, from `threats_counts`. ★ `th_restricted` is the REASON this
# extension exists -- SF's own comment says RestrictedPiece reads the same attack maps as the mobility area, so slice 3's §0
# requires this gate to run BEFORE any threats magnitude ladder. `th_safepawn` / `th_push` are pawn-driven and should be
# largely independent of mobility; if they are NOT, that is a finding about our own area definition.
# ⚠️ The probe reports leg counts UNCONDITIONALLY (see eval_v2.h), so these columns are populated even with the legs' knobs
# off -- which is what makes a single run able to gate every leg at once. Only THREAT_V2_PCT>0 is needed for the score.
THREAT = ["th_minor", "th_rook", "th_king", "th_hanging", "th_restricted", "th_safepawn", "th_push"]
# KING SAFETY (added 2026-09-17): the LAST coverage hole in this gate, flagged since slice 2 and made load-bearing by
# slice 3 -- threats taxes KS-critical accuracy in proportion to its magnitude, and "threats double-counts KS" cannot be
# tested without these columns. ★ Sign convention: ks_* are indexed by the king EXAMINED, so ks_*[0] is White's king
# (attacked by Black). The White−Black difference below therefore reads "how much more exposed White's king is", which is
# the correct orientation to correlate against a White−Black threat difference.
KS = ["ks_natt", "ks_watt", "ks_weak", "ks_adj", "ks_checks", "ks_units"]
# PAWN STRUCTURE (added 2026-09-17): the last subsystem the gate could not see. ★ Sourced from the EXISTING rung-2 detector
# oracle probe (`pawn_masks`), which is knob-free and needs no engine -- so these columns cost nothing and cannot move a
# fingerprint. The masks are reduced by POPCOUNT: `ps_*` are pawn-set sizes, `ps_pattacks` is the number of SQUARES the side's
# pawns attack, `ps_halfopen` the number of files with no pawn of that side (a FILE bitmask at eval_v2.cpp:747, hence popcount).
# ☠️ `openFiles` is deliberately absent: it is side-neutral, so its White−Black difference is identically 0 and the
# zero-variance filter would report it as dead every run -- clutter that trains you to ignore the dead-column line.
# ★ The pairs this exists to test: ps_pattacks × mob_* (our mobility AREA subtracts enemy pawn attacks, so the two are wired
# to the same map) · ps_pattacks × th_safepawn / th_push (both pawn-driven by construction) · ps_blocked / ps_opposed ×
# space (space's safe mask is pawn-defined) · ps_halfopen × traprook_units (trap_rook_units TAKES halfOpen at :1630).
# ☠️ `blocked` is DELIBERATELY ABSENT, and not because it is uninteresting. eval_v2.cpp:682 defines blocked[White] as the
# white pawns of every white/black RAM pair and blocked[Black] as the black pawns of the SAME pairs, so the two popcounts are
# ALWAYS EQUAL and White−Black is identically zero -- a structural blind spot of this gate's differencing convention, not a
# small-sample accident. Measuring `blocked` needs a different reduction (W+B total, or a side-to-move signing); until one
# exists, "does space/mobility re-express the rammed centre?" is UNMEASURABLE here. Recorded in INSTRUMENT-MAP §F.
# ⚠️ `lever` is kept even though it read zero-variance at N=25: it is NOT an identity (one pawn attacked by TWO enemy pawns
# gives 1 vs 2), so a zero at full N is a finding about how rare that geometry is, not a definition.
PAWN = ["ps_isolated", "ps_doubled", "ps_backward", "ps_phalanx", "ps_supported", "ps_opposed", "ps_lever",
        "ps_stopheld", "ps_pattacks", "ps_halfopen", "ps_passed", "ps_candidate", "ps_npawns"]
PAWN_KEYS = ["isolated", "doubled", "backward", "phalanx", "supported", "opposed", "lever",
             "stop_held", "attacks", "halfOpen", "passed", "candidate"]
COLS = ["mob_N", "mob_B", "mob_R", "mob_Q", "mob_table_mg", "mob_table_eg"] + PLACE + THREAT + KS + PAWN

def popcnt(x):
    return bin(int(x)).count("1")

fens = []
for c in CORPORA:
    path = os.path.join(ENGINE, "diagnostics", c)
    if not os.path.exists(path):
        print("  (missing corpus, skipped: %s)" % c)
        continue
    with open(path, newline="") as f:
        rows = [r["fen"] for r in csv.DictReader(f)]
    random.Random(20260914).shuffle(rows)
    fens += rows[:N]

X = []
for fen in fens:
    b = chess.Board(fen)
    args = (b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])
    m = ChessAI.mobility_counts(*args)
    p = ChessAI.placement_counts(*args, b.castling_rights)
    th = ChessAI.threats_counts(*args)
    # ☠️ Form-dependent PACKING (placement_probe): OUTPOST_V2_FORM 1 packs Ethereal's four cells 8 bits each into outpost_n/_b,
    # BADB_V2_FORM 1 packs SF15.1's four file classes 12 bits each into badb_units. Correlating the packed integer would be
    # meaningless — unpack to the TOTAL detector count first.
    if int(os.environ.get("OUTPOST_V2_FORM", "0") or 0) == 1:
        p = dict(p)
        for key in ("outpost_n", "outpost_b"):
            p[key] = tuple(sum((v >> (8 * k)) & 0xFF for k in range(4)) for v in p[key])
    if int(os.environ.get("BADB_V2_FORM", "0") or 0) == 1:
        p = dict(p)
        p["badb_units"] = tuple(sum((v >> (12 * k)) & 0xFFF for k in range(4)) for v in p["badb_units"])
    row = [m["count"][0][i] - m["count"][1][i] for i in range(4)]
    row += [m["raw_mg"][0] - m["raw_mg"][1], m["raw_eg"][0] - m["raw_eg"][1]]
    row += [p[t][0] - p[t][1] for t in PLACE]
    row += [th[k][0] - th[k][1] for k in ("minor", "rook", "king", "hanging", "restricted", "safepawn", "push")]
    ks = ChessAI.ks_counts(*args)
    row += [ks[k][0] - ks[k][1] for k in ("n_att", "w_att", "weak", "adj", "checks", "units")]
    # `pawn_masks` takes only (pawns, occ_white, occ_black) -- it is a pure function of the two pawn bitboards.
    pm = ChessAI.pawn_masks(args[0], args[6], args[7])
    row += [popcnt(pm[k][0]) - popcnt(pm[k][1]) for k in PAWN_KEYS]
    # ★ PAWN COUNT, as a CONTROL rather than a term. `ps_pattacks`, `ps_halfopen` and `ps_passed` all scale with simply
    # HAVING more pawns, so without this column a high overlap among them is ambiguous between "shared structure signal"
    # and "both are restating the material difference rung 0 already owns". With it, the share is readable (block below).
    row.append(popcnt(args[0] & args[6]) - popcnt(args[0] & args[7]))
    X.append(row)

X = np.array(X, dtype=float)
live = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
dead = [COLS[i] for i in range(X.shape[1]) if i not in live]
X = X[:, live]
names = [COLS[i] for i in live]
print("positions=%d  terms=%d%s" % (X.shape[0], len(names), ("  (zero-variance, excluded: %s)" % ", ".join(dead)) if dead else ""))

R = np.corrcoef(X, rowvar=False)
print("\n== correlation matrix (White − Black detector values) ==")
print("  %-15s" % "" + "".join("%9s" % n[:8] for n in names))
for i, n in enumerate(names):
    print("  %-15s" % n + "".join("%9.2f" % R[i, j] for j in range(len(names))))

# VIF_i = diag(inv(R))_i  (equivalent to 1 / (1 − R²) from regressing term i on all the others)
try:
    vif = np.diag(np.linalg.inv(R))
except np.linalg.LinAlgError:
    # ⚠️ A SINGULAR correlation matrix means some column is an EXACT linear combination of the others, which makes every
    # VIF infinite and prints a wall of flags that looks like a catastrophic finding. It is a property of the COLUMN SET.
    # Say so out loud rather than letting inf flags be read as evidence -- drop the offending column and re-run.
    print("  ☠️ correlation matrix is SINGULAR -- one column is an exact combination of others; VIF is UNREADABLE.")
    print("     Treat the VIF flags below as a column-set defect, not a result. Bisect by removing one column set at a time.")
    vif = np.full(len(names), np.inf)

print("\n== variance inflation factor (term explained by ALL the others) ==")
for n, v in zip(names, vif):
    print("  %-15s VIF %7.2f%s" % (n, v, "   <<< FLAG" if v >= VIF_FLAG else ""))

# Mobility's own per-type counts are EXPECTED to correlate with its table sums (same detector) — never flag those.
MOB = set(["mob_N", "mob_B", "mob_R", "mob_Q", "mob_table_mg", "mob_table_eg"])
# ☠️ Same exemption for KING SAFETY's internals, added 2026-09-17 after the first KS-aware run flagged
# `ks_natt x ks_watt  r=+0.93`: those are the attacker COUNT and the WEIGHTED SUM over the IDENTICAL piece set, i.e. one
# detector reported twice, and `ks_units` is the scored total of all the channels below it. Flagging them is a property of
# the column choice, not a finding — exactly the case the MOB set already covers. ★ The gate's question is CROSS-subsystem
# overlap (does threats re-express KS?), so intra-KS correlation must not drown it.
KS_SET = set(["ks_natt", "ks_watt", "ks_weak", "ks_adj", "ks_checks", "ks_units"])
# ☠️ Third instance of the same exemption, for PAWN STRUCTURE (2026-09-17). These thirteen predicates are intercorrelated
# BY CONSTRUCTION -- `supported` and `phalanx` are the two halves of "connected", `blocked` and `opposed` are both
# same-file relations, `passed` and `candidate` are mutually exclusive by definition, and `stop_held` is a conjunct of
# `backward`. Their mutual correlation is a restatement of the definitions, not a finding.
PAWN_SET = set(PAWN)
EXEMPT = [MOB, KS_SET, PAWN_SET]                      # intra-subsystem pairs are exempt; CROSS-subsystem pairs are the question
flags = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if any(names[i] in s and names[j] in s for s in EXEMPT):
            continue
        if abs(R[i, j]) >= R_FLAG:
            flags.append((names[i], names[j], R[i, j]))
# ⚠️ VIF is computed against ALL other columns and therefore CANNOT separate intra- from cross-subsystem explanation. Adding
# thirteen pawn columns raises every VIF mechanically, so a subsystem whose internals are exempt from the PAIR test must be
# exempt here too, or the flag list fills with column-set artefacts. The CROSS-subsystem question is carried by the pairs.
EXEMPT_VIF = MOB | KS_SET | PAWN_SET
place_vif = [(n, v) for n, v in zip(names, vif) if n not in EXEMPT_VIF and v >= VIF_FLAG]

# ★ Ranked cross-subsystem pairs. At 40 columns the matrix above is too wide to read, and a clean flag list only says
# "nothing crossed the bar" -- it does not say how CLOSE anything came. A near-miss at 0.65 is the thing worth reading
# before a term is laddered, so print the top of the distribution regardless of the threshold.
SUBSYS = [(MOB, "mob"), (KS_SET, "ks"), (PAWN_SET, "pawn"), (set(THREAT), "threat"), (set(PLACE), "place")]
def subsys_of(n):
    for s, tag in SUBSYS:
        if n in s:
            return tag
    return "?"
cross = sorted(((abs(R[i, j]), names[i], names[j], R[i, j])
                for i in range(len(names)) for j in range(i + 1, len(names))
                if subsys_of(names[i]) != subsys_of(names[j])), reverse=True)
print("\n== strongest CROSS-subsystem pairs (top 15; the bar is |r| >= %.2f) ==" % R_FLAG)
for _, a, b2, r in cross[:15]:
    print("  %-14s (%-6s) x %-14s (%-6s)  r=%+.2f%s"
          % (a, subsys_of(a), b2, subsys_of(b2), r, "   <<< FLAG" if abs(r) >= R_FLAG else ""))

# ★ MATERIAL SHARE. Every detector count grows with the number of pieces or pawns present, so a term can look like it
# overlaps another when both are really tracking the census -- which rung 0 owns outright. Correlating each column against
# the raw pawn-count difference separates "this is a structure signal" from "this is material wearing a structure name".
# ⚠️ This is a LABEL, not a verdict: a term may legitimately scale with material (mobility does) and still carry its own
# signal. It exists so the pawn block's internal VIF is interpretable instead of merely exempt.
if "ps_npawns" in names:
    k = names.index("ps_npawns")
    mat = sorted(((abs(R[i, k]), names[i], R[i, k]) for i in range(len(names)) if i != k), reverse=True)
    print("\n== material share: every term vs the raw PAWN-COUNT difference (top 10) ==")
    for _, n, r in mat[:10]:
        tag = "  <<< mostly MATERIAL" if abs(r) >= 0.70 else ("  (substantial)" if abs(r) >= 0.45 else "")
        print("  %-14s (%-6s) r=%+.2f%s" % (n, subsys_of(n), r, tag))

print("\n== FLAGS (|r| >= %.2f across subsystems, or a NON-EXEMPT term with VIF >= %.1f) ==" % (R_FLAG, VIF_FLAG))
for a, b2, r in flags:
    print("  PAIR  %-15s x %-15s r=%+.2f" % (a, b2, r))
for n, v in place_vif:
    print("  VIF   %-15s %.2f" % (n, v))
if not flags and not place_vif:
    print("  none")
print("\nSuspects to read first: ★ th_restricted x mob_* (SF's own comment: RestrictedPiece reads the SAME attack maps as")
print("  the mobility area -- this is the pair slice 3 §0 requires gated BEFORE any threats magnitude) · th_minor x")
print("  th_rook (both victim-indexed off the same weak set) · traprook_units x mob_table_mg / mob_R · badb_units x mob_B")
print("  · outpost_n x mob_N · th_safepawn / th_push are pawn-driven and SHOULD be independent of mobility -- if they are")
print("  not, that is a finding about our own area definition, not about threats.")
print("PAWN pairs to read first (added 2026-09-17): ★ ps_pattacks x mob_* -- mob_area SUBTRACTS enemy pawn attacks, so this")
print("  is the one pair we KNOW is wired to a shared map; a high |r| here is expected and is NOT a defect, it is the")
print("  measurement of how much of mobility is really pawn structure · ps_pattacks x th_safepawn / th_push · ps_halfopen x")
print("  traprook_units (trap_rook_units takes halfOpen as a parameter) · ps_blocked / ps_opposed x mob_* (a blocked centre")
print("  is the regime where space read class-local). ⚠️ A CLEAN result here means 'not the same SIGNAL' -- it does NOT mean")
print("  'safe to add together': VIF sees co-movement of COUNTS, never the height of the SCORED curves (threats x KS, 09-17).")
sys.exit(1 if (flags or place_vif) else 0)
