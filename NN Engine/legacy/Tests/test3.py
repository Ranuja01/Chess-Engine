import time
import chess
#import chess_eval
# Create a new board with a different FEN position


def find_legal_attackers(board, target_square):
    """
    Find legal attackers on the target square, differentiated by color.

    Args:
        board (chess.Board): The chess board state.
        target_square (int): The target square index.

    Returns:
        (list, list): Legal attackers for White and Black.
    """

    # Declare lists to store legal attackers for white and black
    white_attackers = []
    black_attackers = []
    
    # Get legal moves to the target square
    if (board.is_attacked_by(chess.BLACK, target_square)):
        for move in board.legal_moves:
            if move.to_square == target_square:
                if board.color_at(move.from_square):
                    white_attackers.append(chess.square_name(move.from_square))
                else:
                    black_attackers.append(chess.square_name(move.from_square))
        print(black_attackers)

board = chess.Board()
board.push(chess.Move.from_uci("d2d4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("a2a3"))
# Define a target square (e.g., d5)\
    
t0 = time.time()
target_square = chess.D4
find_legal_attackers(board, target_square)
t1 = time.time()

print(f"Time: {t1 - t0:.15f} seconds")



t0 = time.time()

total = 0
for move in board.legal_moves:
        if move.to_square == target_square:
            if (board.turn):
                total -= 1234
            else:
                print(board.piece_at(target_square).piece_type)
                total += 1234
            break
t1 = time.time()

print(f"Time: {t1 - t0:.15f} seconds")   