# -*- coding: utf-8 -*-
"""Build the COLOUR-MIRRORED twin of an STS/EPD suite.

Why this exists: sts300.epd is 177 white-to-move against 123 black-to-move. That skew is harmless for
most work, but it makes STS the WRONG instrument for judging a colour-symmetry fix -- the bench asks
White's questions more often than Black's, so an eval that is differentially generous to one colour is
scored on a tilted board. Using it to arbitrate a symmetry repair measures partly the repair and partly
the tilt, with no way to separate them.

The mirrored suite is the same 300 questions asked of the other colour. Two readings follow:

  * `orig + mirror` is a colour-BALANCED positional score -- the honest arbiter for a symmetry change.
  * `orig - mirror` is a direct measurement of the eval's colour bias, and should be ~0 for an eval
    that respects antisymmetry. It is a bench-side check on the same defect `_eval_symmetry.py` finds
    statically, but expressed in the units we actually care about (move choice).

board.mirror() flips ranks, swaps colours, swaps side-to-move AND swaps castling rights, so the
position transform is exact. The c9 moves are UCI coordinates, so each one maps through square_mirror
on both endpoints; promotion pieces are unaffected (a promotion stays a promotion under a rank flip).

  pyrun diagnostics/make_mirror_suite.py [IN=suites/sts300.epd] [OUT=suites/sts300_mirror.epd]
"""
import os, re, sys

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess

IN = os.environ.get("IN", "suites/sts300.epd")
OUT = os.environ.get("OUT", "suites/sts300_mirror.epd")
if not os.path.isabs(IN):
    IN = os.path.join(THIS, IN)
if not os.path.isabs(OUT):
    OUT = os.path.join(THIS, OUT)

C8 = re.compile(r'\bc8\s+"([^"]*)"')
C9 = re.compile(r'\bc9\s+"([^"]*)"')
ID = re.compile(r'\bid\s+"([^"]*)"')
BM = re.compile(r'\bbm\s+([^;]+);')      # WAC-style: best move(s) in SAN, not UCI


def mirror_uci(u):
    """e2e4 -> e7e5 style: flip both endpoints vertically, keep any promotion suffix."""
    mv = chess.Move.from_uci(u)
    return chess.Move(chess.square_mirror(mv.from_square),
                      chess.square_mirror(mv.to_square),
                      promotion=mv.promotion).uci()


def mirror_bm_line(line):
    """WAC/tactical form: `<epd> bm <SAN...>; id "...";` -- SAN needs the board to parse and re-emit.

    wac.epd is 190 white-to-move vs 110 black-to-move (63/37), a WORSE skew than sts300's 59/41, so the
    tactical suite needs the same colour-balancing treatment as the positional one before it can judge
    a colour fix.
    """
    bm = BM.search(line)
    if not bm:
        return None
    b = chess.Board()
    try:
        b.set_epd(line)
    except Exception:
        return None
    m = b.mirror()
    moves = []
    for san in bm.group(1).split():
        try:
            mv = b.parse_san(san)
        except Exception:
            return None
        mm = chess.Move(chess.square_mirror(mv.from_square),
                        chess.square_mirror(mv.to_square), promotion=mv.promotion)
        if mm not in m.legal_moves:
            return None
        moves.append(m.san(mm))
    ident = ID.search(line)
    ops = "bm %s;" % " ".join(moves)
    if ident:
        ops += ' id "%s(mirror)";' % ident.group(1)
    return "%s %s" % (m.epd(), ops)


def main():
    out_lines, n, skipped = [], 0, 0
    for line in open(IN):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        c8, c9 = C8.search(line), C9.search(line)
        if not (c8 and c9):
            # Fall back to the bm/SAN form before giving up, so this one tool covers both suites.
            alt = mirror_bm_line(line)
            if alt:
                out_lines.append(alt)
                n += 1
            else:
                skipped += 1
            continue
        b = chess.Board()
        try:
            b.set_epd(line)
        except Exception:
            skipped += 1
            continue
        try:
            moves = [mirror_uci(u) for u in c9.group(1).split()]
        except Exception:
            skipped += 1
            continue

        m = b.mirror()
        # Only keep questions that are still well posed after the transform: every scored move must be
        # legal in the mirrored position. If the transform were correct this is automatic -- so a
        # failure here is a bug in THIS script, and it is loud rather than silently dropped.
        legal = {mv.uci() for mv in m.legal_moves}
        bad = [u for u in moves if u not in legal]
        if bad:
            print("  !! mirrored move not legal (%s) -- skipping %s" % (",".join(bad), line[:40]))
            skipped += 1
            continue

        ident = ID.search(line)
        ops = 'c9 "%s"; c8 "%s";' % (" ".join(moves), c8.group(1))
        if ident:
            ops += ' id "%s(mirror)";' % ident.group(1)
        out_lines.append("%s %s" % (m.epd(), ops))
        n += 1

    with open(OUT, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print("wrote %d mirrored positions -> %s" % (n, OUT))
    if skipped:
        print("skipped %d" % skipped)

    # Side-to-move census of both suites: the mirror must invert it exactly, which is the cheapest
    # possible proof that the transform did what it claims.
    def census(p):
        w = bl = 0
        for ln in open(p):
            parts = ln.split()
            if len(parts) > 1:
                if parts[1] == 'w':
                    w += 1
                elif parts[1] == 'b':
                    bl += 1
        return w, bl

    print("side-to-move   original %s   mirrored %s" % (census(IN), census(OUT)))


if __name__ == "__main__":
    main()
