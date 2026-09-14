# -*- coding: utf-8 -*-
"""DRAW CLASSIFIER ORACLE -- measure the FALSE-POSITIVE rate of endgame draw rules against 7-piece
tablebase ground truth, WEIGHTED BY HOW FAR AWAY THE WIN IS.

WHY THIS EXISTS (2026-09-13). A draw rule returns a hard 0 for the WHOLE eval, so a case that fires on a
position that is actually WON throws the win away. The owner's standing rule is asymmetric:
    ★ a won position flagged drawn is catastrophic; a missed draw only forfeits an opportunity.
The June 2026 fix that added R+N-vs-R / KRKN / KRKB was validated on MEAN BIAS against an NNUE trust gate,
which cannot see false positives at all. This closes that gap.

★★ WHY FALSE POSITIVES ARE WEIGHTED BY DTM (revised 2026-09-13).
"Zero false positives" is UNATTAINABLE for insufficient-material-style rules: KBvKB, KNvKN, KBvKN and KNNvK
each contain a handful of genuine tablebase wins -- forced mates where a king is boxed into a corner, often
by its own piece. Every reference (SF's generic rule, Weiss) draws these anyway. And they are nearly
harmless: SEARCH detects checkmate itself, so a forced mate within the horizon is found whatever the leaf
eval says. The false positives that actually cost games are LONG-TECHNIQUE wins beyond the horizon -- the
K+R+B vs K+R win in 21, K+R+N vs K+R in 25.
    ⇒ SHORT false positive (|dtm| <= SHORT plies): search finds it; reported, tolerated.
    ⇒ LONG  false positive (|dtm| >  SHORT, or dtm unknown): the rule discards a win search cannot see.
       This is the gate. Exit 1 iff any LONG false positive exists.

★ WHY THERE IS A CORNER-BIASED MODE. Uniform random placement essentially never generates the rare boxed-king
mates, so "0 in 382" on a uniform sample was true and proved nothing (the same trap as the rook-pawn rule
at 0-for-62, which a third seed then broke at 6.2%). EDGE=1 puts one king on a corner or corner-adjacent
square, where those mates live.

Ground truth: the Lichess tablebase HTTP API (authoritative to 7 pieces, no download). We have NO Syzygy
data on disk. Every result is cached to disk, so a re-run costs no network.
⚠️ Be a good citizen: modest samples, a small delay, cache everything.

  pyrun diagnostics/_draw_oracle.py [ARM=v1|v2] [N=60] [CASES=a,b] [SEED=] [EDGE=0|1] [SHORT=12] [DELAY=0.25]

Exit 0 = no LONG false positive in any case tested. Exit 1 = a rule discards a win beyond the horizon.
"""
import os, sys, json, time, random, urllib.request, urllib.error

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS  = os.path.dirname(os.path.abspath(__file__))
N     = int(os.environ.get("N", "60"))
DELAY = float(os.environ.get("DELAY", "0.25"))
CASES = os.environ.get("CASES", "all")
# Tracked FIXTURE, not scratch: today's published false-positive numbers were measured through it, tablebase answers
# never change so it cannot go stale, and it spares ~1,900 queries to a free public API on any re-run.
CACHE = os.environ.get("CACHE", os.path.join(THIS, "ks_sets", "tablebase_labels_draw.json"))
SEED  = int(os.environ.get("SEED", "20260913"))
EDGE  = os.environ.get("EDGE", "0") == "1"
SHORT = int(os.environ.get("SHORT", "12"))   # plies; our search reaches d12 in ~1s
API   = "https://tablebase.lichess.ovh/standard?fen="

try:
    import chess
except ImportError:
    print("python-chess required"); sys.exit(2)


# ---------------------------------------------------------------- v1 helpers, mirrored exactly
def is_white_square(sq):
    # cpp_bitboard.cpp:1665 -- (file + rank) % 2 == 1
    return ((sq % 8) + (sq // 8)) % 2 == 1

def cheb(a, b):
    # cpp_bitboard.cpp:6530
    return max(abs(a % 8 - b % 8), abs(a // 8 - b // 8))

FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080


def v1_case(bd):
    """Return the NAME of the is_practically_drawn case that fires, or None.

    Mirrors cpp_bitboard.cpp:6769-6995 in order. Reads only what the C++ reads.
    """
    occ   = int(bd.occupied)
    kings = int(bd.kings)
    pawns = int(bd.pawns)
    kn    = int(bd.knights)
    bi    = int(bd.bishops)
    ro    = int(bd.rooks)
    w     = int(bd.occupied_co[chess.WHITE])
    b     = int(bd.occupied_co[chess.BLACK])
    nk    = occ & ~kings
    pc    = lambda x: bin(x).count("1")

    # 1. equal piece count, only bishops or only knights
    if pc(w) == pc(b):
        if nk == bi or nk == kn:
            return "eq_only_minor"
    # 2. lone bishop / lone knight vs bare king
    if pc(bi) == 1 and nk == bi:
        return "KBvK"
    if pc(kn) == 1 and nk == kn:
        return "KNvK"
    # 3. bare kings
    if nk == 0:
        return "KvK"
    # 4. K + lone rook pawn vs K (ENABLE_RP_KPK_DRAW, default true)
    if nk == pawns and pc(pawns) == 1 and ((pawns & FILE_A) or (pawns & FILE_H)):
        pw   = bool(pawns & w)
        psq  = (pawns & -pawns).bit_length() - 1
        promo = (56 if pw else 0) if (pawns & FILE_A) else (63 if pw else 7)
        wk = (kings & w); bk = (kings & b)
        wks = (wk & -wk).bit_length() - 1
        bks = (bk & -bk).bit_length() - 1
        atk, dfd = (wks, bks) if pw else (bks, wks)
        if cheb(dfd, promo) <= min(cheb(psq, promo), cheb(atk, promo)):
            return "rookpawn_KPvK"
    # 5. bishop + rook pawn, bishop on the wrong colour
    if nk == (bi | pawns) and pc(bi) == 1 and pc(pawns) == 1:
        if (pawns & FILE_A) or (pawns & FILE_H):
            pw = bool(pawns & w)
            bsq = (bi & -bi).bit_length() - 1
            psq = (pawns & -pawns).bit_length() - 1
            promo = (56 if pw else 0) if (pawns & FILE_A) else (63 if pw else 7)
            if is_white_square(bsq) != is_white_square(promo):
                wk = (kings & w); bk = (kings & b)
                wks = (wk & -wk).bit_length() - 1
                bks = (bk & -bk).bit_length() - 1
                atk, dfd = (wks, bks) if pw else (bks, wks)
                if cheb(dfd, promo) <= min(cheb(psq, promo), cheb(atk, promo)):
                    return "wrongB_rookpawn"
    # 6. R+B vs R   ☠️ REFUTED 2026-09-13
    if pc(nk) == 3 and pc(ro) == 2 and pc(bi) == 1:
        if (pc(ro & w) == 1 and pc(bi & w) == 1 and pc(ro & b) == 1 and pc(bi & b) == 0) or \
           (pc(ro & b) == 1 and pc(bi & b) == 1 and pc(ro & w) == 1 and pc(bi & w) == 0):
            return "RB_vs_R"
    # 7. R+N vs R   ☠️ REFUTED 2026-09-13
    if pc(nk) == 3 and pc(ro) == 2 and pc(kn) == 1:
        if (pc(ro & w) == 1 and pc(kn & w) == 1 and pc(ro & b) == 1 and pc(kn & b) == 0) or \
           (pc(ro & b) == 1 and pc(kn & b) == 1 and pc(ro & w) == 1 and pc(kn & w) == 0):
            return "RN_vs_R"
    # 8. bare rook vs bare minor
    if pc(nk) == 2 and pc(ro) == 1 and (pc(kn) == 1 or pc(bi) == 1):
        minor = kn | bi
        if bool(ro & w) != bool(minor & w):
            return "R_vs_minor"
    # 9. bishop vs lone pawn (rook file, wrong-colour bishop, opposition)
    if pc(nk) == 2 and pc(bi) == 1 and pc(pawns) == 1:
        if bool(bi & w) != bool(pawns & w):
            if (pawns & FILE_A) or (pawns & FILE_H):
                pw = bool(pawns & w)
                bsq = (bi & -bi).bit_length() - 1
                psq = (pawns & -pawns).bit_length() - 1
                promo = (56 if pw else 0) if (pawns & FILE_A) else (63 if pw else 7)
                if is_white_square(bsq) != is_white_square(promo):
                    wk = (kings & w); bk = (kings & b)
                    wks = (wk & -wk).bit_length() - 1
                    bks = (bk & -bk).bit_length() - 1
                    atk, dfd = (wks, bks) if pw else (bks, wks)
                    if cheb(dfd, promo) <= min(cheb(psq, promo), cheb(atk, promo)):
                        return "B_vs_P"
    # 10. knight vs lone pawn -- same shape (the C++ tail continues past :6978)
    if pc(nk) == 2 and pc(kn) == 1 and pc(pawns) == 1:
        if bool(kn & w) != bool(pawns & w):
            return "N_vs_P_maybe"   # ⚠️ approximate: the C++ adds distance guards past :6978
    return None


# ---------------------------------------------------------------- position generation
# (pieces for WHITE, pieces for BLACK) -- kings implied
SIGS = {
    "KBvK":            ([chess.BISHOP], []),
    "KNvK":            ([chess.KNIGHT], []),
    "eq_only_minor":   ([chess.BISHOP], [chess.BISHOP]),
    "eq_only_minor_N": ([chess.KNIGHT], [chess.KNIGHT]),
    "rookpawn_KPvK":   ([chess.PAWN], []),
    "wrongB_rookpawn": ([chess.BISHOP, chess.PAWN], []),
    "RB_vs_R":         ([chess.ROOK, chess.BISHOP], [chess.ROOK]),
    "RN_vs_R":         ([chess.ROOK, chess.KNIGHT], [chess.ROOK]),
    "R_vs_minor":      ([chess.ROOK], [chess.BISHOP]),
    "R_vs_minor_N":    ([chess.ROOK], [chess.KNIGHT]),
    "B_vs_P":          ([chess.BISHOP], [chess.PAWN]),
    "N_vs_P_maybe":    ([chess.KNIGHT], [chess.PAWN]),
    # 2026-09-13 candidates for v2, adopted from the references and checked here before any C++:
    "KBvKN":           ([chess.BISHOP], [chess.KNIGHT]),          # Weiss draws it; v2 misses it
    "KNNvK":           ([chess.KNIGHT, chess.KNIGHT], []),        # Weiss draws, SF scales 4/64 -- they DISAGREE
    "sf_wrongB":       ([chess.BISHOP, chess.PAWN], []),          # SF KBPsK fortress form, not v1's race
}


def v2_case(bd):
    """Return the case name from eval_v2.cpp's `draw_class`, or None.

    ★ This is the NARROWED set: only the cases that measured clean. It must stay in lockstep with
    eval_v2.cpp -- if they diverge, this oracle is validating something we do not ship.
    ⚠️ Deliberately narrower than v1 in one more way: the `n_nk > 2` early-out drops KBBvKBB / KNNvKNN,
    which the sweep never generated. An unmeasured case fails toward forfeiting a draw, never toward
    discarding a win.
    """
    occ   = int(bd.occupied); kings = int(bd.kings); pawns = int(bd.pawns)
    kn    = int(bd.knights);  bi    = int(bd.bishops)
    w     = int(bd.occupied_co[chess.WHITE]); b = int(bd.occupied_co[chess.BLACK])
    nk    = occ & ~kings
    pc    = lambda x: bin(x).count("1")

    if nk == 0:
        return "KvK"
    n_nk = pc(nk)
    if n_nk > 2:
        return None
    n_b, n_n, n_p = pc(bi), pc(kn), pc(pawns)
    if n_b == 1 and nk == bi:
        return "KBvK"
    if n_n == 1 and nk == kn:
        return "KNvK"
    if pc(w) == pc(b) and (nk == bi or nk == kn):
        return "eq_only_minor"
    # --- 2026-09-13 reference-derived cases; NOW IN eval_v2.cpp draw_class (keep these in lockstep) ---
    # Added after passing this oracle: KBvKN 0/400, KNNvK 0/400, sf_wrongB 0/286 (uniform + EDGE=1, 2 seeds).
    # KB vs KN (Weiss draws it): one bishop, one knight, opposite sides, nothing else.
    if n_b == 1 and n_n == 1 and nk == (bi | kn) and bool(bi & w) != bool(kn & w):
        return "KBvKN"
    # KNN vs K: Weiss draws it, SF scales it to 4/64 -- the references DISAGREE, so the TB decides.
    if n_n == 2 and nk == kn and (kn & w) in (0, kn):
        return "KNNvK"
    # SF KBPsK fortress form (endgame.cpp:356): bishop + rook pawn on the same side, wrong-coloured
    # bishop, and the defending king ALREADY within one square of the queening corner. A static
    # fortress, not v1's race -- v1's race version measured 10% false positives.
    if n_b == 1 and n_p == 1 and nk == (bi | pawns) and bool(bi & w) == bool(pawns & w) \
            and ((pawns & FILE_A) or (pawns & FILE_H)):
        sw   = bool(pawns & w)
        psq_ = (pawns & -pawns).bit_length() - 1
        bsq_ = (bi & -bi).bit_length() - 1
        qsq  = (56 if sw else 0) + (psq_ % 8)
        dk   = kings & (b if sw else w)
        dks  = (dk & -dk).bit_length() - 1
        if is_white_square(bsq_) != is_white_square(qsq) and cheb(dks, qsq) <= 1:
            return "sf_wrongB"
    if n_p != 1:
        return None
    psq   = (pawns & -pawns).bit_length() - 1
    p_w   = bool(pawns & w)
    file_a = bool(pawns & FILE_A)
    if not file_a and not (pawns & FILE_H):
        return None
    promo = (56 if p_w else 0) if file_a else (63 if p_w else 7)
    wk = (kings & w); bk = (kings & b)
    wks = (wk & -wk).bit_length() - 1
    bks = (bk & -bk).bit_length() - 1
    atk, dfd = (wks, bks) if p_w else (bks, wks)
    # ☠️ TEMPO CORRECTION. v1 compares raw chebyshev distances and IGNORES whose move it is, which is a
    # measured 6.2% false-positive source on KPvK alone (e.g. 8/8/8/8/8/2k1K2P/8/8 w -- all three
    # distances are 5, so `defender <= min(...)` fires, and White WINS because White moves first).
    # If the attacking side is on move it effectively gains a tempo, so the defender needs one more.
    attacker_to_move = (p_w == (bd.turn == chess.WHITE))
    d_def = cheb(dfd, promo) + (1 if attacker_to_move else 0)
    holds = d_def <= min(cheb(psq, promo), cheb(atk, promo))
    if nk == pawns:
        return "rookpawn_KPvK" if holds else None
    if n_nk == 2 and n_b == 1 and bool(bi & w) != p_w:
        bsq = (bi & -bi).bit_length() - 1
        if is_white_square(bsq) != is_white_square(promo) and holds:
            return "B_vs_P"
    return None


# Corners and their edge-adjacent squares -- where boxed-king forced mates live.
EDGE_SQ = [0, 1, 8, 9, 7, 6, 15, 14, 56, 57, 48, 49, 63, 62, 55, 54]


def random_position(wp, bp, rng):
    """Place kings + the given pieces; return a legal Board or None.

    With EDGE=1 one king (either colour, at random) is forced onto a corner or corner-adjacent square.
    """
    bd = chess.Board.empty()
    n_extra = len(wp) + len(bp)
    if EDGE:
        boxed_white = rng.random() < 0.5
        ksq = rng.choice(EDGE_SQ)
        rest = [s for s in range(64) if s != ksq]
        others = rng.sample(rest, 1 + n_extra)
        wk, bk = (ksq, others[0]) if boxed_white else (others[0], ksq)
        squares = [wk, bk] + others[1:]
    else:
        squares = rng.sample(range(64), 2 + n_extra)
        wk, bk = squares[0], squares[1]
    if cheb(wk, bk) <= 1:
        return None
    bd.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    i = 2
    for pt in wp:
        sq = squares[i]; i += 1
        if pt == chess.PAWN and not (8 <= sq < 56):
            return None
        bd.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
    for pt in bp:
        sq = squares[i]; i += 1
        if pt == chess.PAWN and not (8 <= sq < 56):
            return None
        bd.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
    bd.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not bd.is_valid():
        return None
    if bd.is_game_over():
        return None
    return bd


# ---------------------------------------------------------------- tablebase, cached
_cache = {}
if os.path.exists(CACHE):
    try:
        _cache = json.load(open(CACHE))
    except Exception:
        _cache = {}
_net = [0]


def tb_lookup(fen):
    """Return (category, |dtm| or None). Legacy cache entries stored only the category; a decisive one
    is re-queried once so its DTM is known, because DTM is what the gate is decided on."""
    hit = _cache.get(fen)
    if isinstance(hit, dict):
        return hit.get("c"), hit.get("m")
    if isinstance(hit, str) and hit not in ("win", "loss"):
        return hit, None
    url = API + fen.replace(" ", "_")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.load(r)
            cat = data.get("category")
            dtm = data.get("dtm")
            m = abs(dtm) if isinstance(dtm, int) else None
            _cache[fen] = {"c": cat, "m": m}
            _net[0] += 1
            time.sleep(DELAY)
            return cat, m
        except urllib.error.HTTPError as e:
            if e.code == 400:
                _cache[fen] = {"c": "bad-fen", "m": None}
                return "bad-fen", None
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.0 + attempt)
    return None, None


def save_cache():
    try:
        json.dump(_cache, open(CACHE, "w"))
    except Exception as e:
        sys.stderr.write("cache save failed: %s\n" % e)


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    wanted = None if CASES == "all" else set(CASES.split(","))
    which = os.environ.get("ARM", "v1")
    classifier = v2_case if which == "v2" else v1_case
    print("ARM = %s  (%s)   EDGE = %s   SHORT = %d plies\n" % (
        which, "eval_v2.cpp draw_class -- the NARROWED set" if which == "v2"
        else "v1 is_practically_drawn -- all ten cases", EDGE, SHORT))

    # category is from the SIDE TO MOVE's view. A false positive = decisive for EITHER side.
    # cursed-win / blessed-loss are 50-move-rule draws: theoretically won, practically drawn -- counted apart.
    DECISIVE = {"win", "loss"}
    CURSED   = {"cursed-win", "blessed-loss"}

    results = {}
    print("DRAW CLASSIFIER ORACLE -- vs Lichess 7-piece tablebase")
    print("gate: ZERO LONG false positives (a flagged-drawn position won beyond %d plies)\n" % SHORT)
    print("  %-18s %7s  %6s %6s %6s" % ("case", "flagged", "FP", "short", "LONG"))

    for signame, (wp, bp) in SIGS.items():
        if wanted and signame not in wanted:
            continue
        fp_short, fp_long, cursed = [], [], 0
        tested = tries = 0
        while tested < N and tries < N * 80:
            tries += 1
            bd = random_position(wp, bp, rng)
            if bd is None:
                continue
            case = classifier(bd)
            if case is None:
                continue
            tested += 1
            cat, dtm = tb_lookup(bd.fen())
            if cat in DECISIVE:
                if dtm is not None and dtm <= SHORT:
                    fp_short.append((bd.fen(), cat, case, dtm))
                else:
                    fp_long.append((bd.fen(), cat, case, dtm))
            elif cat in CURSED:
                cursed += 1
        results[signame] = (tested, fp_short, fp_long, cursed)
        mark = "✅" if not fp_long else "☠️"
        print("  %-18s %7d  %6d %6d %6d  %s%s" % (
            signame, tested, len(fp_short) + len(fp_long), len(fp_short), len(fp_long), mark,
            "   (cursed %d)" % cursed if cursed else ""))

    save_cache()
    print("\n(%d new tablebase queries; cache: %s)" % (_net[0], CACHE))

    any_short = any(r[1] for r in results.values())
    if any_short:
        print("\nSHORT false positives -- won, but within the horizon, so search finds the mate itself:")
        for signame, (tested, fs, fl, cu) in results.items():
            for fen, cat, case, dtm in fs[:4]:
                print("      %-18s %-5s dtm=%-3s %s" % (signame, cat, dtm, fen))

    total_long = sum(len(r[2]) for r in results.values())
    if total_long:
        print("\n☠️ %d LONG FALSE POSITIVES -- the rule discards a win search cannot see:" % total_long)
        for signame, (tested, fs, fl, cu) in results.items():
            if not fl:
                continue
            print("\n  %s -- %d/%d = %.1f%% of flagged positions" % (
                signame, len(fl), tested, 100.0 * len(fl) / max(1, tested)))
            for fen, cat, case, dtm in fl[:5]:
                print("      %-5s dtm=%-4s case=%-16s %s" % (cat, dtm, case, fen))
        print("\n★ Under the owner's asymmetric rule these cases CANNOT stay in a binary detector.")
        return 1

    print("\n✅ No LONG false positives in any case tested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
