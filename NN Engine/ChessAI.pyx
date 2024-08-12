# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:32:26 2024

@author: Kumodth
"""
import chess  # Use regular import for Python libraries
cimport cython  # Import Cython-specific utilities
from cython cimport boundscheck, wraparound
# Import Cython, numpy, chess, and tensorflow for defining the functions
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.array cimport array
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from cython.parallel import prange
from chess import Board
from chess import Move
import chess.polyglot

cimport chess as c_chess

import tensorflow as tf
#cimport tensorflow as ctf
from tensorflow.keras.models import Model

# Define data types for numpy arrays
ctypedef cnp.float32_t DTYPE_FLOAT
ctypedef cnp.int8_t DTYPE_INT

# Declare global models
cdef object blackModel
cdef object whiteModel

cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t

cdef struct MoveData:
    int a
    int b
    int c
    int d
    int score
    
cdef struct PredictionInfo:
    int x
    int y
    int w
    int z

# Define and initialize global arrays
cdef int layer[2][8][8]
cdef int placementLayer[2][8][8]
#cdef int placementLayer[2][64]
cdef int layer2[2][8][8]

# Initialize the arrays globally
cdef void initialize_layers():
    cdef int i, j, k

    # Initialize layer
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if i == 0:
                    layer[i][j][k] = [
                        [0,0,0,0,0,0,0,0], [0,0,5,5,5,5,3,0], [0,0,3,15,25,5,0,0], [0,0,3,25,30,15,0,0],
                        [0,0,3,25,30,15,0,0], [0,0,3,15,25,5,0,0], [0,0,5,5,5,5,3,0], [0,0,0,0,0,0,0,0]
                    ][j][k]
                else:
                    layer[i][j][k] = [
                        [0,0,0,0,0,0,0,0], [0,0,3,5,5,5,0,0], [0,0,5,25,15,3,0,0], [0,0,15,30,25,3,0,0],
                        [0,0,15,30,25,3,0,0], [0,0,5,25,15,3,0,0], [0,0,3,5,5,5,0,0], [0,0,0,0,0,0,0,0]
                    ][j][k]

    # Initialize placementLayer
    
    
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if i == 0:
                    placementLayer[i][j][k] = [
                        [0,0,0,0,0,0,0,0],
                        [-1000,0,2,5,5,2,0,0],
                        [-1000,0,3,15,15,5,0,0],
                        [0,-200,3,25,70,5,0,0],
                        [0,-200,3,25,70,5,0,0],
                        [-1000,0,3,15,15,5,0,0],
                        [-1000,0,2,5,5,2,0,0],
                        [0,0,0,0,0,0,0,0]
                    ][j][k]
                else:
                    placementLayer[i][j][k] = [
                        [0,0,0,0,0,0,0,0],
                        [0,0,2,5,5,2,0,-1000],
                        [0,0,5,15,15,3,0,-1000],
                        [0,0,5,70,25,3,-200,0],
                        [0,0,5,70,25,3,-200,0],
                        [0,0,5,15,15,3,0,-1000],
                        [0,0,2,5,5,2,0,-1000],
                        [0,0,0,0,0,0,0,0]
                    ][j][k]
    '''
    for i in range(2):
        for j in range(64):
                if i == 0:
                    placementLayer[i][j] = [0, -150, -200, 0, 0, -200, -150, 0, 
                                            0, 0, 0, -200, -200, 0, 0, 0, 
                                            0, 2, 3, 3, 3, 3, 2, 0,
                                            0, 5, 15, 25, 25, 15, 5, 0,
                                            0, 5, 15, 70, 70, 15, 5, 0, 
                                            0, 2, 5, 5, 5, 5, 2, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0][j]
                else:
                    placementLayer[i][j] = [0, 0, 0, 0, 0, 0, 0, 0, 
                                            0, 0, 0, 0, 0, 0, 0, 0, 
                                            0, 2, 5, 5, 5, 2, 5, 2, 
                                            0, 5, 15, 70, 70, 15, 5, 15, 
                                            0, 5, 15, 25, 25, 15, 5, 25, 
                                            0, 2, 3, 3, 3, 3, 2, 3, 
                                            0, 0, 0, -200, -200, 0, 0, 0, 
                                            0, -150, -200, 0, 0, -200, -150, 0][j]
    '''
    # Initialize layer2
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if i == 0:
                    layer2[i][j][k] = [
                        [0,0,1,2,3,15,40,0], [0,0,2,5,15,25,40,0], [0,0,3,5,25,35,40,0], [0,0,3,5,25,35,40,0],
                        [0,0,3,5,25,35,40,0], [0,0,3,5,25,35,40,0], [0,0,2,5,15,25,40,0], [0,0,1,2,3,15,40,0]
                    ][j][k]
                else:
                    layer2[i][j][k] = [
                        [0,40,15,3,2,1,0,0], [0,40,25,15,5,2,0,0], [0,40,35,25,5,3,0,0], [0,40,35,25,5,3,0,0],
                        [0,40,35,25,5,3,0,0], [0,40,35,25,5,3,0,0], [0,40,25,15,5,2,0,0], [0,40,15,3,2,1,0,0]
                    ][j][k]



# Declare class ChessAI
@cython.cclass
cdef class ChessAI:
    
    cdef object blackModel
    cdef object whiteModel
    cdef object pgnBoard
    cdef list boardPieces
    cdef int numMove
    cdef int numIterations
    #cdef bool isComputerMove
    #cdef bool computerThinking

    def __cinit__(self, object black_model, object white_model, object board):
        self.blackModel = black_model
        self.whiteModel = white_model
        self.pgnBoard = board
        self.numMove = 0
        self.numIterations = 0
        #self.isComputerMove = False
        #self.computerThinking = False
        
        # Call the initialization function once at module load
        initialize_layers()

    cpdef alphaBetaWrapper(self, int curDepth, int depthLimit):
        self.numIterations = 0            
        if (len(self.pgnBoard.move_stack) < 21):
            return self.opening_book(curDepth, depthLimit)
        
        
        return self.alphaBeta(curDepth, depthLimit)

    @cython.ccall
    @cython.exceptval(check=False)
    cdef MoveData opening_book(self, curDepth, depthLimit):
        cdef MoveData best_move
        cdef str cur
        
        best_move.a = -1
        best_move.b = -1
        best_move.c = -1
        best_move.d = -1
        best_move.score = -99999999
        
        with chess.polyglot.open_reader("M11.2.bin") as reader:
            # Find all entries for the current board position
            entries = list(reader.find_all(self.pgnBoard))

            
            # Sort entries by weight to find the best move
            if entries:
                best_entry = max(entries, key=lambda e: e.weight)
                print(f"Best Move: {best_entry.move}, Weight: {best_entry.weight}, Learn: {best_entry.learn}")
            
                cur = best_entry.move.uci()
                
                best_move.score = 0
                best_move.a = ord(cur[0]) - 96
                best_move.b = int(cur[1])
                best_move.c = ord(cur[2]) - 96
                best_move.d = int(cur[3])
                
                return best_move
            else:
                return self.alphaBeta(curDepth, depthLimit)
                print("No moves found in the book for this position.")


    # Define the alphaBeta function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef MoveData alphaBeta(self, int curDepth, int depthLimit):
        cdef int alpha = -9999998
        cdef int beta = 9999998

        #cdef int highestScore = -99999999
        #cdef int a, b, c, d
        
        cdef MoveData bestMove
        bestMove.a = -1
        bestMove.b = -1
        bestMove.c = -1
        bestMove.d = -1
        bestMove.score = -99999999
                          
        #cdef int a, b, c, d
        cdef str cur
        #cdef int index
        
        # cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] filteredPrediction = np.zeros(4096, dtype=np.float32)
        # cdef cnp.ndarray[DTYPE_INT, ndim=4] inputBoard = np.array([encode_board(self.pgnBoard)], dtype=np.int8)
        # cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prediction = self.blackModel.predict(inputBoard, verbose=0)
        cdef list moves_list
        moves_list = reorder_legal_moves(self.pgnBoard)
        for move in moves_list:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0

            
            self.pgnBoard.push(move)

            self.numMove += 1
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta)
            self.numMove -= 1
            self.pgnBoard.pop()
            '''
            if (len(self.pgnBoard.move_stack) == 51):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("1ST MOVE: {}, {}\n".format(score, move.uci()))
            '''
            if score > bestMove.score:
                cur = move.uci()
                
                bestMove.score = score
                bestMove.a = ord(cur[0]) - 96
                bestMove.b = int(cur[1])
                bestMove.c = ord(cur[2]) - 96
                bestMove.d = int(cur[3])

            alpha = max(alpha, bestMove.score)

            if beta <= alpha:
                self.numMove += 1
                print(self.numIterations)
                return bestMove

        if curDepth == 0:
            self.numMove += 1
            print(self.numIterations)
            return bestMove
        

    # Define the maximizer function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall    
    @cython.inline
    cdef int maximizer(self, int curDepth, int depthLimit, int alpha, int beta):
        cdef int highestScore = -9999999
                                 
        cdef int a, b, c, d
        cdef str cur
        #cdef int index
        cdef int target_square
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] filteredPrediction = np.zeros(4096, dtype=np.float32)

        if curDepth >= depthLimit:
            # target_square = self.pgnBoard.peek().to_square
            # if not (self.pgnBoard.is_attacked_by(chess.BLACK, target_square)) or curDepth >= 6:
            self.numIterations += 1
            return evaluate_board(self.pgnBoard)
            #depthLimit += 1
            #print("Maximizer extra 2: ")

        #cdef cnp.ndarray[DTYPE_INT, ndim=4] inputBoard = np.array([encode_board(self.pgnBoard)], dtype=np.int8)
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prediction = self.blackModel.predict(inputBoard, verbose=0)

        cdef list moves_list
        moves_list = reorder_legal_moves(self.pgnBoard)
        for move in moves_list:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0

            
            self.pgnBoard.push(move)

            self.numMove += 1
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta)
            self.numMove -= 1
            self.pgnBoard.pop()
            '''
            if (len(self.pgnBoard.move_stack) == 53):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("3RD MOVE: {}, {}\n".format(score, move.uci()))

            if (len(self.pgnBoard.move_stack) == 55):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("5TH MOVE: {}, {}\n".format(score, move.uci()))
            '''
            if score > highestScore:
                highestScore = score

            alpha = max(alpha, highestScore)

            if beta <= alpha:
                return highestScore
        
        if (len(moves_list) == 0):
            if self.pgnBoard.is_checkmate():                
                return -100000000
             
        
        return highestScore

    # Define the minimizer function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int minimizer(self, int curDepth, int depthLimit, int alpha, int beta):
        cdef int lowestScore = 9999999 - len(self.pgnBoard.move_stack)
        
        #cdef int a, b, c, d
        cdef str cur
        #cdef int index
        cdef int target_square
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=1] filteredPrediction = np.zeros(4096, dtype=np.float32)

        if curDepth >= depthLimit:
            # target_square = self.pgnBoard.peek().to_square
            # if not (self.pgnBoard.is_attacked_by(chess.WHITE, target_square)) or curDepth >= 6:
            self.numIterations += 1
            return evaluate_board(self.pgnBoard)
            # depthLimit += 1
            #print("Minimizer extra 1: ")

        #cdef cnp.ndarray[DTYPE_INT, ndim=4] inputBoard = np.array([encode_board(self.pgnBoard)], dtype=np.int8)
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prediction = self.whiteModel.predict(inputBoard, verbose=0)

        cdef list moves_list
        moves_list = reorder_legal_moves(self.pgnBoard)
        for move in moves_list:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0

            
            self.pgnBoard.push(move)
            
            self.numMove += 1
            score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
            self.numMove -= 1
            self.pgnBoard.pop()
            '''
            if (len(self.pgnBoard.move_stack) == 52):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("2ND MOVE: {}, {}\n".format(score, move.uci()))
                    
            if (len(self.pgnBoard.move_stack) == 54):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("4TH MOVE: {}, {}\n".format(score, move.uci()))

            if (len(self.pgnBoard.move_stack) == 56):
                with open('Unfiltered_Full.txt', 'a') as file:
                    file.write("6TH MOVE: {}, {}\n".format(score, move.uci()))
            '''
            if score < lowestScore:
                lowestScore = score

            beta = min(beta, lowestScore)

            if beta <= alpha:
                '''
                if lowestScore == 9999999:
                    print("AAA: ", alpha, curDepth)
                    print("BBB: ", beta)
                    print("CCC: ", move)                    
                ''' 
                return lowestScore

        if (lowestScore == 9999999):
            #print("AAAA")
            if self.pgnBoard.is_checkmate():                   
                #print("BBBB")
                return 100000000
            elif self.pgnBoard.is_stalemate():
                #print("STALEMATE")
                return -100000000
            elif self.pgnBoard.can_claim_draw():
                print("OTHER")
                return -100000000
        return lowestScore

# Define the Cython function
cdef cnp.ndarray[DTYPE_FLOAT, ndim=3] encode_board(object board):

    # Define piece mappings
    cdef dict piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Initialize a 12-channel tensor
    cdef cnp.ndarray[DTYPE_FLOAT, ndim=3] encoded_board = np.zeros((8, 8, 12), dtype=np.float32)


    # Populate the tensor
    cdef int i, j
    for i in range(8):
        for j in range(8):
            # chess.square expects (file, rank) with 0-indexed file
            piece = board.piece_at(chess.square(j, 7-i))
            if piece:
                channel = piece_to_channel[piece.symbol()]
                encoded_board[i, j, channel] = 1.0
    
    return encoded_board

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
@cython.inline
def reorder_legal_moves(object board):
    """
    Reorder legal moves for the given board state,
    prioritizing capture moves using python-chess.
    """
    cdef list legal_moves = []
    cdef list capture_moves = []
    cdef list non_capture_moves = []
    cdef object move

    # Iterate through all legal moves
    for move in board.legal_moves:
        if board.is_capture(move):
            capture_moves.append(move)
        else:
            non_capture_moves.append(move)

    # Append capture moves first followed by non-capture moves
    legal_moves.extend(capture_moves)
    legal_moves.extend(non_capture_moves)

    return legal_moves

# Function to evaluate the board
cdef int evaluate_board2(object board):
    cdef int total = 0
    cdef int square
    cdef object piece
    cdef int values[7]

    # Initialize the array in C-style
    values[0] = 0      # No piece
    values[1] = 1000   # Pawn
    values[2] = 2700   # Knight
    values[3] = 3000   # Bishop
    values[4] = 5000   # Rook
    values[5] = 9000   # Queen
    values[6] = 0      # King
    
    
    # Check for checkmate
    if board.is_checkmate():
        if board.turn:
            total = 100000000
            
        else:
            total = -100000000
    else:
        # Iterate through all squares on the board and evaluate piece values
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Evaluate based on piece color
                if piece.color == board.turn:
                    total -= values[piece.piece_type]
                else:
                    total += values[piece.piece_type]
                     
    return total

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef evaluate_board(object board):
    cdef int total = 0
    cdef object piece
    cdef uint8_t  square
    cdef uint8_t  x, y
    #cdef int activeLayer[2][8][8]
    #cdef int activePlacementLayer[2][8][8]

    cdef int moveNum = len(board.move_stack)
    
    # Define the layers as 2D arrays
    #cdef int layer[2][8][8]
    #cdef int placementLayer[2][8][8]
    #cdef int layer2[2][8][8]
    
    # Determine active layers based on move count
    #cdef int[:,:,:] activeLayer = layer2 if moveNum >= 18 else layer
    cdef int[:,:,:] activePlacementLayer = layer2 if moveNum >= 40 else placementLayer

    cdef int values[7]

    # Initialize the array in C-style
    values[0] = 0      # No piece
    values[1] = 1000   # Pawn
    values[2] = 3150   # Knight
    values[3] = 3250   # Bishop
    values[4] = 5000   # Rook
    values[5] = 9000   # Queen
    values[6] = 0      # King

    
    # Iterate through all squares on the board and evaluate piece values
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            y = square >> 3
            x = square - (y << 3)
            
            # Evaluate based on piece color
            if piece.color:
                total -= values[piece.piece_type]
                
                if not (piece.piece_type == chess.ROOK) or moveNum >= 40:
                    total -= activePlacementLayer[0][x][y]
                    #print(activePlacementLayer[0][x][y])
                    
                    if (piece.piece_type == chess.PAWN):
                        total -= (y + 1) * 25
                
                if (x == 3 and y == 3 or
                    x == 3 and y == 4 or
                    x == 4 and y == 3 or
                    x == 4 and y == 4):
                    total -= 150
            else:
                total += values[piece.piece_type]
                
                if not (piece.piece_type == chess.ROOK) or moveNum >= 40:
                    total += activePlacementLayer[1][x][y]
                    
                    if (piece.piece_type == chess.PAWN):
                        total += (8 - y) * 25
    
                if (x == 3 and y == 3 or
                    x == 3 and y == 4 or
                    x == 4 and y == 3 or
                    x == 4 and y == 4):
                    total += 150
   
    # current_castling_rights = board.castling_rights

    # move2 = board.pop()
    # move1 = board.pop()
    
    # initial_castling_rights = board.castling_rights
    
    # board.push(move1)
    # board.push(move2)
    
    # # Compare the initial castling rights with the current ones
    # # If there is a difference, castling has occurred
    # if current_castling_rights != initial_castling_rights:
    #     return True
    # return False
    

    return total

cdef bint is_promotion_move_enhanced(object move, object board, int y):
    
    # Check if the move is a promotion
    if move.promotion is not None:
        return True
    # Check if the move is to the promotion rank
    if (board.turn and y - 1 == 7) or \
       (not(board.turn) and y - 1 == 0):
        # Check if a pawn is moving to the last rank
        from_square = move.from_square
        piece = board.piece_at(from_square)

        if piece and piece.piece_type == 1:
           
            return True
    
    return False

# Define the Cython function
cdef PredictionInfo predictionInfo(int prediction):
    
    cdef PredictionInfo coords
    # Get the starting square by integer dividing by 64
    pieceToBeMoved = prediction // 64
    
    # Get location square via the remainder
    squareToBeMovedTo = prediction % 64

    # Coordinates of the piece to be moved
    coords.x = pieceToBeMoved // 8 + 1
    coords.y = pieceToBeMoved % 8 + 1
    
    # Coordinates of the square to be moved to
    coords.w = squareToBeMovedTo // 8 + 1
    coords.z = squareToBeMovedTo % 8 + 1

    return coords

# Define the Cython function
cdef int reversePrediction(int x, int y, int i, int j):
    
    return (((x - 1) * 8 + y) - 1) * 64 + ((i - 1) * 8 + j)
