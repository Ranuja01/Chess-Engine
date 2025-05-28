import chess  # Use regular import for Python libraries
cimport cython  # Import Cython-specific utilities
import Cython_Chess
from cython cimport boundscheck, wraparound

cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t

# Import functions from c++ file
cdef extern from "cpp_bitboard.h":
    void initialize_attack_tables()
    int placement_and_piece_eval(int moveNum, bint turn, uint8_t lastMovedToSquare, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t prevKings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    bint is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bint is_en_passant)
    
initialize_attack_tables()

cdef int values[7]
values[0] = 0      # No piece
values[1] = 1000   # Pawn
values[2] = 3150   # Knight
values[3] = 3250   # Bishop
values[4] = 5000   # Rook
values[5] = 9000   # Queen
values[6] = 0      # King

cpdef evaluate_board (object board):
    return wrapper (board)

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef int wrapper (object board):
    
    """
    Evaluation function

    Parameters:
    - board: The current board state
  
    Returns:
    - The evaluation for the given position
    """

    # Initialize bitmasks for the current board state
    cdef uint64_t pawns = board.pawns
    cdef uint64_t knights = board.knights
    cdef uint64_t bishops = board.bishops
    cdef uint64_t rooks = board.rooks
    cdef uint64_t queens = board.queens
    cdef uint64_t kings = board.kings
    
    cdef uint64_t occupied_white = board.occupied_co[True]
    cdef uint64_t occupied_black = board.occupied_co[False]
    cdef uint64_t occupied = board.occupied
    
    global values
    
    # Define variable to hold total evaluation and move number
    cdef int total = 0    
    cdef int moveNum = board.ply()
    
    # Define variables for the target move and square for the previously made move
    cdef object target_square
    cdef object target_move
    # Check if the board state is checkmate
    if board.is_checkmate():
        if board.turn:
            total = 9999999 - moveNum      
        else:
            total = -9999999 + moveNum
    else:
        # Call the c++ function   
        # print(board)
        # print()
        
        total += placement_and_piece_eval(moveNum, board.turn, 64, pawns, knights, bishops, rooks, queens, kings, 0, occupied_white, occupied_black, occupied)   
        # ** Code segment to see if a bad capture was made ** 
        
        # # Get the previous move made
        # if (board.move_stack):
        #     target_move = board.peek()
        #     print ("Cython-EEEE")
        #     # Check if the move was a capture
        #     if (is_capture(target_move.from_square, target_move.to_square, board.occupied_co[not board.turn], board.is_en_passant(target_move))):
                
        #         # Acquire the square that the move was made to
        #         target_square = target_move.to_square
                    
        #         # Check if there is a legal capture to the same square
        #         for move in Cython_Chess.generate_legal_captures(board,chess.BB_ALL,chess.BB_SQUARES[target_square]):
                    
        #             # If such a capture exists, assume that the last capture was a bad one and assume you will lose that piece
                    
        #             if (board.turn):
        #                 total -= values[board.piece_type_at(target_square)]
                        
        #             else:                            
        #                 total += values[board.piece_type_at(target_square)]
                    
        #             break    
    return total