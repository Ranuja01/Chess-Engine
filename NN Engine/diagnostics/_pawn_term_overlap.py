# -*- coding: utf-8 -*-
"""PAWN STRUCTURAL TERM OVERLAP — the anti-collinearity instrument, run BEFORE any constant exists.

WHY THIS EXISTS (2026-09-12, eval-v2 rung 2a). `bundling-is-refuted-components-cancel-26-percent` showed
components overlapping 50-64%% cancel ~26%% of each other's move changes. v1 could never see that coming,
because its terms had no separable detector -- you could only measure overlap by ABLATING built, tuned
terms. Every pawn structural predicate, however, is a PURE FUNCTION OF THE TWO PAWN BITBOARDS, so their
firing sets can be computed straight from corpus FENs with NO engine, NO build and NO constants.

⇒ This answers "are these two terms the same signal?" BEFORE we write them, let alone tune them.
★ Protocol (EVAL-V2-RUNG2-PAWN-DESIGN.md §3): anything over ~50%% conditional overlap is two names for one
signal and gets merged or dropped on the spot.

  pyrun diagnostics/_pawn_term_overlap.py [N=4000] [CORPUS=ks_sets/game_regret_set.csv]

⚠️ Firing-set overlap is NOT value collinearity: two terms can fire on the same pawns and still price them
differently. This REJECTS duplicates cheaply; it does not certify the survivors.
"""
import os, sys, csv

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
N = int(os.environ.get("N", "4000"))
CORPUS = os.environ.get("CORPUS", "ks_sets/game_regret_set.csv")

FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
FULL   = (1 << 64) - 1


def nfill(b):
    """Smear every bit northward (exclusive of the bit itself is handled by callers)."""
    b |= (b << 8) & FULL
    b |= (b << 16) & FULL
    b |= (b << 32) & FULL
    return b & FULL


def sfill(b):
    b |= b >> 8
    b |= b >> 16
    b |= b >> 32
    return b


def north(b):
    return (b << 8) & FULL


def south(b):
    return b >> 8


def watt(p):
    """White pawn attack span."""
    return (((p & ~FILE_A) << 7) | ((p & ~FILE_H) << 9)) & FULL


def batt(p):
    """Black pawn attack span."""
    return ((p & ~FILE_A) >> 9) | ((p & ~FILE_H) >> 7)


def east(b):
    return (b & ~FILE_H) << 1 & FULL


def west(b):
    return (b & ~FILE_A) >> 1


def _file_of(b):
    """Full-board mask of every file that b occupies."""
    return nfill(sfill(b))


def _span3(b, white):
    """The forward three-file span (own + both neighbours, strictly ahead) of every bit in b."""
    f = (nfill((b << 8) & FULL) if white else sfill(b >> 8))
    return f | east(f) | west(f)


def terms(own, enemy, white):
    """Every structural predicate as a bitboard of OWN pawns that it fires on.

    All SF-faithful (evaluate.cpp / pawns.cpp), all pure functions of the two pawn bitboards.
    """
    up   = north if white else south
    down = south if white else north
    oatt = watt(own)   if white else batt(own)      # squares OUR pawns attack
    eatt = batt(enemy) if white else watt(enemy)    # squares THEIR pawns attack

    adj = east(own) | west(own)                     # our pawns shifted onto adjacent files

    t = {}
    # isolated: no own pawn anywhere on an adjacent file
    t["isolated"]  = own & ~nfill(sfill(adj))
    # doubled (SF): a friendly pawn directly behind on the same file
    t["doubled"]   = own & down(own)
    # phalanx: a friendly pawn beside it on the same rank
    t["phalanx"]   = own & adj
    # supported: defended by one of our own pawns
    t["supported"] = own & oatt
    # opposed: an enemy pawn anywhere ahead on our OWN file
    t["opposed"]   = own & (sfill(enemy) if white else nfill(enemy))
    # ★ lever: WE attack an enemy pawn. A white pawn on s attacks s+7/s+9, so the squares attacking an
    #   enemy pawn e are exactly {e-9, e-7} = batt(e). Mirrored for black.
    t["lever"]     = own & (batt(enemy) if white else watt(enemy))
    # stop square occupied by an enemy pawn / controlled by one (SF's leverPush condition)
    stop           = up(own)
    t["blocked"]   = own & down(stop & enemy)
    t["stop_held"] = own & down(stop & eatt)
    # backward (SF): NO friendly neighbour on an adjacent file at or BEHIND our rank, and the push is
    # contested or blocked. ⚠️ nfill smears north, so bit (f,r) is set iff a source sits at r' <= r —
    # which is "a neighbour at or behind us" for White. Mirrored for Black.
    has_rear_nb    = own & (nfill(adj) if white else sfill(adj))
    t["backward"]  = own & ~has_rear_nb & (t["blocked"] | t["stop_held"])
    # passed: no enemy pawn anywhere in the forward three-file span (exact, per-pawn)
    t["passed"]    = own & ~_stopped(own, enemy, white)
    # ★ SF candidate passers (getPPIncrement's ENABLE_PASSER_DETECT_SF block, which is OFF in v1 -- this
    # is the ~14% of SF's passers we do not currently detect). A pawn whose stoppers are ALL pawns it
    # attacks (lever), or all pawns its pushed self would attack with a phalanx at least as large
    # (leverPush), or a single same-file blocker it out-supports from the 5th rank up (blocked).
    # ⚠️ A REAR-DOUBLED pawn is excluded: a friendly pawn ahead on its own file can never promote.
    cand = 0
    b = own
    while b:
        sq = (b & -b).bit_length() - 1
        b &= b - 1
        m = 1 << sq
        st = _span3(m, white) & enemy
        if st == 0:
            continue                      # already passed by the stock rule
        phal = ((east(m) | west(m))) & own
        if white:
            lev = (((m & ~FILE_A) << 7) | ((m & ~FILE_H) << 9)) & enemy
            sup = (((m & ~FILE_H) >> 7) | ((m & ~FILE_A) >> 9)) & own
            front = (m << 8) & FULL
            lpush = (((front & ~FILE_A) << 7) | ((front & ~FILE_H) << 9)) & enemy
            sps = ((sup << 8) & FULL) & ~enemy
            rank_owner = (sq >> 3) + 1
            rear = (nfill(front) & _file_of(m)) & own
        else:
            lev = (((m & ~FILE_A) >> 9) | ((m & ~FILE_H) >> 7)) & enemy
            sup = (((m & ~FILE_H) << 9) | ((m & ~FILE_A) << 7)) & own
            front = m >> 8
            lpush = (((front & ~FILE_A) >> 9) | ((front & ~FILE_H) >> 7)) & enemy
            sps = (sup >> 8) & ~enemy
            rank_owner = 8 - (sq >> 3)
            rear = (sfill(front) & _file_of(m)) & own
        blocked_m = front & enemy
        ok = ((st ^ lev) == 0)              or (((st ^ lpush) == 0) and bin(phal).count("1") >= bin(lpush).count("1"))              or (st == blocked_m and st != 0 and rank_owner >= 5 and sps != 0)
        if ok and not rear:
            cand |= m
    t["candidate"] = cand

    t["weak"]       = t["isolated"] | t["backward"]
    t["weak_unopp"] = t["weak"] & ~t["opposed"]
    return t


def _stopped(own, enemy, white):
    """Own pawns whose forward 3-file span contains an enemy pawn (i.e. NOT passed)."""
    out = 0
    b = own
    while b:
        sq = (b & -b).bit_length() - 1
        b &= b - 1
        f = sq & 7
        r = sq >> 3
        if white:
            ahead = 0
            for rr in range(r + 1, 8):
                for ff in (f - 1, f, f + 1):
                    if 0 <= ff <= 7:
                        ahead |= 1 << (rr * 8 + ff)
        else:
            ahead = 0
            for rr in range(0, r):
                for ff in (f - 1, f, f + 1):
                    if 0 <= ff <= 7:
                        ahead |= 1 << (rr * 8 + ff)
        if ahead & enemy:
            out |= 1 << sq
    return out


def main():
    try:
        import chess
    except ImportError:
        print("python-chess required"); return 2

    path = os.path.join(ENGINE, CORPUS)
    if not os.path.exists(path):
        path = os.path.join(THIS, CORPUS)
    if not os.path.exists(path):
        print("corpus not found: %s" % CORPUS); return 2

    names = ["isolated", "doubled", "backward", "phalanx", "supported", "opposed",
             "lever", "blocked", "stop_held", "passed", "candidate", "weak", "weak_unopp"]
    counts = {n: 0 for n in names}
    pair = {(a, b): 0 for a in names for b in names}
    total_pawns = 0
    rows = 0

    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for rec in csv.DictReader(fh):
            if rows >= N:
                break
            fen = (rec.get("fen") or "").strip()
            if not fen:
                continue
            try:
                bd = chess.Board(fen)
            except Exception:
                continue
            rows += 1
            wp = int(bd.pieces(chess.PAWN, chess.WHITE))
            bp = int(bd.pieces(chess.PAWN, chess.BLACK))
            for own, enemy, white in ((wp, bp, True), (bp, wp, False)):
                if not own:
                    continue
                total_pawns += bin(own).count("1")
                t = terms(own, enemy, white)
                for n in names:
                    counts[n] += bin(t[n]).count("1")
                for a in names:
                    for b in names:
                        pair[(a, b)] += bin(t[a] & t[b]).count("1")

    if not total_pawns:
        print("no pawns scanned"); return 2

    print("PAWN STRUCTURAL TERM OVERLAP — %d positions, %d pawns, corpus=%s"
          % (rows, total_pawns, os.path.basename(path)))
    print("Firing RATE = %% of all pawns the predicate flags.\n")
    print("  %-12s %8s" % ("term", "rate%"))
    for n in names:
        print("  %-12s %7.2f%%" % (n, 100.0 * counts[n] / total_pawns))

    print("")
    print(u"☠️ LIFT = P(col | row) / base_rate(col). THIS is the collinearity number, not raw")
    print(u"overlap: a term firing on 72% of pawns co-occurs with everything at ~72% for free.")
    print(u"1.0 = INDEPENDENT.  ★ >2.0 = structurally coupled, must be priced aware of each other.")
    print("")
    hdr = "  %-12s" % ""
    for b in names:
        hdr += "%7s" % b[:6]
    print(hdr)
    for a in names:
        line = "  %-12s" % a
        for b in names:
            if counts[a] and counts[b]:
                cond = float(pair[(a, b)]) / counts[a]
                base = float(counts[b]) / total_pawns
                line += "%7.2f" % (cond / base)
            else:
                line += "%7s" % "-"
        print(line)
    print("")
    print("\nCONDITIONAL OVERLAP  P(col | row) — of the pawns ROW fires on, what %% does COL also fire on?")
    print("☠️ >50%% = two names for one signal. Asymmetry matters: a rare term inside a common one is SUBSUMED.\n")
    hdr = "  %-12s" % ""
    for b in names:
        hdr += "%7s" % b[:6]
    print(hdr)
    for a in names:
        line = "  %-12s" % a
        for b in names:
            line += "%6.0f%%" % (100.0 * pair[(a, b)] / counts[a]) if counts[a] else "%7s" % "-"
        print(line)

    print("\n⚠️ Firing-set overlap is not value collinearity: it REJECTS duplicates, it does not certify survivors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
