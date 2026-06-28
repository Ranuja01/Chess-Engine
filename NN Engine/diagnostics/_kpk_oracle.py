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
                            states.append(b.fen())
    return states


def solve(states):
    """Retrograde fixpoint. Returns {fen: WIN|DRAW}."""
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
                # promotion: evaluate the KQ-vs-K shortcut directly
                nb = b.copy(stack=False)
                nb.push(chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN))
                children.append(("T", promo_value(nb)))
            else:
                nb = b.copy(stack=False)
                nb.push(mv)
                children.append(("S", nb.fen()))
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


def main():
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
        b = chess.Board(f)
        key = b.fen()
        print(f"  oracle={'WIN' if val.get(key)==WIN else 'DRAW' if key in val else '?'}  rule={candidate_rule(f)}  {f}")


if __name__ == "__main__":
    main()
