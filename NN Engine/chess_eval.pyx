# chess_eval.pyx

import chess  # Use regular import for Python libraries
cimport cython  # Import Cython-specific utilities


# Function to count the material on the board
# Using static typing for internal computations
def evaluate_board(board: object) -> int:
    """
    Evaluate the board and return a material score.
    
    :param board: A python-chess Board object
    :return: Material balance score
    """
    cdef int total = 0
    
    
    if (board.turn):
        if (board.is_checkmate()):
            total = 10000000
    else:
        if (board.is_checkmate()):
            total = 10000000  
    
    # Iterate over all pieces on the board
    for piece in board.piece_map().values():
        if piece.color == chess.WHITE:
            total -= get_piece_value(piece.piece_type)
        else:
            total += get_piece_value(piece.piece_type)

    return total


# Static typing for the piece value lookup
cdef int[7] piece_values = [0, 1000, 2700, 3000, 5000, 9000, 0]  # Piece values

# Function to return the piece value
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_piece_value(int piece_type):
    return piece_values[piece_type]
#python setup.py build_ext --inplace