import chess
cimport cython

@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing
def find_legal_attackers(object board, int target_square):
    """
    Find legal attackers on the target square, differentiated by color.

    Args:
        board (chess.Board): The chess board state.
        target_square (int): The target square index.

    Returns:
        (list, list): Legal attackers for White and Black.
    """

    # Declare lists to store legal attackers for white and black
    cdef list white_attackers = []
    cdef list black_attackers = []
    
    # Get legal moves to the target square
    if (board.is_attacked_by(chess.BLACK, target_square)):
        for move in board.legal_moves:
            if move.to_square == target_square:
                if board.color_at(move.from_square):
                    white_attackers.append(chess.square_name(move.from_square))
                else:
                    black_attackers.append(chess.square_name(move.from_square))

    #return white_attackers, legal_black_attacking_pieces


