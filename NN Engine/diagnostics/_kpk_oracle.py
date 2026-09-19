# -*- coding: utf-8 -*-
"""Authoritative rook-pawn KPvK win/draw oracle + drawing-rule validator.

Purpose: the chesscom-2200 conversion loss was a drawn lone rook-pawn KPvK scored ~+4.9 by our eval
(no draw detection). Before extending is_practically_drawn we must guarantee the new rule NEVER flags a
WON rook-pawn KPvK as drawn. This builds the exact game-theoretic oracle (retrograde fixpoint over KPvK
with a KQ-vs-K promotion shortcut) for pawns on the a/h file, then tests a candidate chebyshev-opposition
rule against it (reporting any false-draw = a won position the rule would wrongly zero).

Run (Windows): python diagnostics/_kpk_oracle.py            (oracle + rule check, no SF)
               python diagnostics/_kpk_oracle.py --sf        (also SF cp on the chesscom FENs)
"""
import sys, chess

WIN, DRAW = 1, 0  # value from White's perspective (Black has only a king -> cannot win)


def promo_value(board):
    """Value of a White pawn that has just promoted to a queen, Black to move (KQ vs K)."""
    if board.is_stalemate():
        return DRAW
    if board.is_insufficient_material():
        return DRAW
    # Black to move: if it can capture the (necessarily lone) queen, KvK -> draw.
    for mv in board.legal_moves:
        if board.is_capture(mv):
            return DRAW
    return WIN  # KQ vs K with a safe queen is a theoretical win


def key(board):
    """State key: placement + side to move ONLY.
    ☠️ Fixed 2026-09-14. This file used to key states by board.fen(), which INCLUDES the halfmove and fullmove
    counters. Every generated state is '... 0 1', but a king move's child is '... 1 1' and Black's reply bumps
    the fullmove number, so child lookups MISSED, val.get() returned None, and WIN never propagated except through
    pawn pushes and terminals -- the oracle labelled most won KPvK positions DRAW. A win-starved oracle cannot see
    a FALSE DRAW (the rule's flagged 'draw' is just agreed with), so every 'no false-draws' result this tool
    produced before the fix (incl. the June rook-pawn rule, '83,238 states, 0 false-draws') is UNVERIFIED."""
    return "%s %s - -" % (board.board_fen(), "w" if board.turn == chess.WHITE else "b")


def gen_states(files):
    """All legal KPvK positions with a single WHITE pawn on one of `files`, ranks 2-7, both stm."""
    states = []
    for pf in files:
        for pr in range(1, 7):              # ranks 2..7 (0-indexed 1..6)
            psq = chess.square(pf, pr)
            for wk in range(64):
                if wk == psq:
                    continue
                for bk in range(64):
                    if bk == psq or bk == wk:
                        continue
                    if chess.square_distance(wk, bk) <= 1:   # kings can't be adjacent
                        continue
                    for stm in (chess.WHITE, chess.BLACK):
                        b = chess.Board(None)
                        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                        b.set_piece_at(psq, chess.Piece(chess.PAWN, chess.WHITE))
                        b.turn = stm
                        if b.is_valid():
                            states.append(key(b))
    return states


def promo_best(board_before, mv):
    """--exact: White's best promotion on this move, QUEEN or ROOK. ★ Under-promotion to a rook wins some
    positions where queening stalemates; without it the oracle labels those DRAW and would hide a false draw."""
    best = DRAW
    for piece in (chess.QUEEN, chess.ROOK):
        nb = board_before.copy(stack=False)
        nb.push(chess.Move(mv.from_square, mv.to_square, promotion=piece))
        if promo_value(nb) == WIN:
            best = WIN
    return best


def solve(states, exact=False):
    """Retrograde fixpoint. Returns {fen: WIN|DRAW}. exact=True also considers rook under-promotion."""
    val = {}
    succ = {}
    # Seed terminals; record successors for the rest.
    for fen in states:
        b = chess.Board(fen)
        if b.is_checkmate():          # side to move is mated; only White can mate -> Black mated
            val[fen] = WIN
            continue
        if b.is_stalemate() or b.is_insufficient_material():
            val[fen] = DRAW
            continue
        children = []
        for mv in b.legal_moves:
            if b.piece_at(mv.from_square).piece_type == chess.PAWN and chess.square_rank(mv.to_square) == 7:
                # promotion: evaluate the KQ-vs-K (and, with exact, KR-vs-K) shortcut directly
                if mv.promotion not in (None, chess.QUEEN):
                    continue                               # one child per promotion square, not four
                if exact:
                    children.append(("T", promo_best(b, mv)))
                else:
                    nb = b.copy(stack=False)
                    nb.push(chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN))
                    children.append(("T", promo_value(nb)))
            else:
                nb = b.copy(stack=False)
                nb.push(mv)
                children.append(("S", key(nb)))
        succ[fen] = children

    changed = True
    while changed:
        changed = False
        for fen, children in succ.items():
            if fen in val:
                continue
            b_white = chess.Board(fen).turn == chess.WHITE
            cvals = []
            for kind, ref in children:
                cvals.append(ref if kind == "T" else val.get(ref))
            if b_white:
                if any(cv == WIN for cv in cvals):
                    val[fen] = WIN; changed = True
            else:
                if cvals and all(cv == WIN for cv in cvals):  # all known and winning
                    val[fen] = WIN; changed = True
    # Unresolved = Black holds = draw
    for fen in succ:
        val.setdefault(fen, DRAW)
    return val


def cheb(a, b):
    return chess.square_distance(a, b)


def candidate_rule(fen):
    """Mirror of the C++ rule we intend to add: lone rook-pawn KPvK is drawn if the defending king
    reaches the promotion corner no later than the pawn/attacker (chebyshev opposition)."""
    b = chess.Board(fen)
    psq = next(iter(b.pieces(chess.PAWN, chess.WHITE)))
    pfile = chess.square_file(psq)
    if pfile not in (0, 7):
        return False
    promo = chess.square(pfile, 7)
    dk = next(iter(b.pieces(chess.KING, chess.BLACK)))
    ak = next(iter(b.pieces(chess.KING, chess.WHITE)))
    dd, ad, pd = cheb(dk, promo), cheb(ak, promo), cheb(psq, promo)
    mode = candidate_rule.mode
    if mode == "strict":            # matches existing KBP/KN rook-pawn rules (no side-to-move)
        return dd <= min(pd, ad)
    # tempo-aware: defender one tempo behind when it is the attacker's move
    return dd <= min(pd, ad) + (0 if b.turn == chess.WHITE else 1)


candidate_rule.mode = "strict"


def engine_check():
    """--all-files --engine: the gate for eval v2's EXACT KPK bitbase (DRAW_V2_KPK_EXACT).

    Solves EVERY legal KPvK state with the pawn on files a-d (the bitbase's normalised domain; e-h is a file
    mirror) including rook under-promotion, then compares the engine's kpk_probe on each one. Both directions are
    reported, but a FALSE DRAW (oracle WIN, bitbase draw) is the one that must be zero: it would zero a won game.
    Run in WSL after a build:  pyrun diagnostics/_kpk_oracle.py --all-files --engine
    """
    import os, time
    engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, engine_dir)
    import ChessAI
    t0 = time.time()
    states = gen_states([0, 1, 2, 3])
    print(f"KPvK legal states, files a-d: {len(states)}  ({time.time()-t0:.0f}s)")
    val = solve(states, exact=True)
    wins = sum(1 for v in val.values() if v == WIN)
    print(f"oracle (exact, with rook under-promotion): WIN={wins}  DRAW={len(val)-wins}  ({time.time()-t0:.0f}s)")
    false_draw, false_win, bad_domain = [], [], 0
    for fen, v in val.items():
        b = chess.Board(fen)
        psq = next(iter(b.pieces(chess.PAWN, chess.WHITE)))
        wk = b.king(chess.WHITE)
        bk = b.king(chess.BLACK)
        got = ChessAI.kpk_win(wk, psq, bk, b.turn == chess.WHITE)
        if got < 0:
            bad_domain += 1
        elif v == WIN and got == 0:
            false_draw.append(fen)
        elif v == DRAW and got == 1:
            false_win.append(fen)
    print(f"engine bitbase vs oracle: FALSE-DRAWS={len(false_draw)} (must be 0)  false-wins={len(false_win)}  "
          f"out-of-domain={bad_domain}")
    for f in false_draw[:10]:
        print("    FALSE-DRAW:", f)
    for f in false_win[:10]:
        print("    false-win: ", f)
    ok = not false_draw and not false_win and bad_domain == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    if "--all-files" in sys.argv and "--engine" in sys.argv:
        engine_check()
    if "--sf" in sys.argv:
        import chess.engine
        SF = r"C:\Users\Kumodth\OneDrive\Desktop\Programming\Chess Engine\stockfish\stockfish-windows-x86-64-avx2.exe"
        eng = chess.engine.SimpleEngine.popen_uci(SF)
        fens = ["8/7k/8/8/6KP/8/8/8 w - - 4 59",
                "5R2/8/8/5r2/4k2P/8/6K1/8 w - - 6 56",
                "8/1R6/1P6/7p/2k1p2P/4P3/1r6/5K2 w - - 1 47"]
        for f in fens:
            info = eng.analyse(chess.Board(f), chess.engine.Limit(depth=30))
            print(f"SF {str(info['score'].white()):>10}  best={info['pv'][0] if info.get('pv') else '-'}  {f}")
        eng.quit()
        print()

    states = gen_states([0, 7])
    print(f"rook-pawn KPvK legal states: {len(states)}")
    val = solve(states)
    wins = sum(1 for v in val.values() if v == WIN)
    print(f"oracle: WIN={wins}  DRAW={len(val)-wins}")

    # Validate candidate rule: count false-draws (rule says draw but oracle says WIN) -> MUST be 0.
    for mode in ("strict", "tempo"):
        candidate_rule.mode = mode
        false_draw = []
        caught_draws = total_draws = 0
        for fen, v in val.items():
            r = candidate_rule(fen)
            if v == DRAW:
                total_draws += 1
                caught_draws += 1 if r else 0
            if r and v == WIN:
                false_draw.append(fen)
        print(f"rule[{mode:6}]: catches {caught_draws}/{total_draws} draws ; FALSE-DRAWS (won flagged drawn) = {len(false_draw)}")
        for f in false_draw[:8]:
            print("    FALSE-DRAW:", f)
    candidate_rule.mode = "strict"

    # Confirm the chesscom KPvK positions are oracle-draws and caught by the rule.
    print("\nchesscom KPvK check:")
    for f in ["8/7k/8/8/6KP/8/8/8 w - - 4 59", "7k/8/8/6KP/8/8/8/8 w - - 7 63"]:
        # normalise to white-pawn oracle key (these already have a white pawn)
        k = key(chess.Board(f))
        print(f"  oracle={'WIN' if val.get(k)==WIN else 'DRAW' if k in val else '?'}  rule={candidate_rule(f)}  {f}")


if __name__ == "__main__":
    main()
