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
from cython cimport nogil

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

cdef whiteCastledIndex = -1
cdef blackCastledIndex = -1

# Initialize the arrays globally
cdef void initialize_layers(board):
    cdef int i, j, k
    cdef int increment = 5
    # Initialize layer
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if i == 0:
                    layer[i][j][k] = [
                        [0,0,0,0,0,0,0,0],
                        [0,0,3,3,4,5,5,0],
                        [0,0,3,4,5,6,0,0],
                        [0,0,3,5,7,8,0,0],
                        [0,0,3,5,7,8,0,0],
                        [0,0,3,4,7,6,0,0],
                        [0,0,3,3,4,5,5,0],
                        [0,0,0,0,0,0,0,0]
                    ][j][k]
                else:
                    layer[i][j][k] = [
                        [0,0,0,0,0,0,0,0],
                        [0,5,5,4,3,3,0,0],
                        [0,0,6,7,6,3,0,0],
                        [0,0,8,8,7,3,0,0],
                        [0,0,8,8,7,3,0,0],
                        [0,0,6,7,6,3,0,0],
                        [0,5,5,4,3,3,0,0],
                        [0,0,0,0,0,0,0,0]
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
                        [0,-200,3,175,220,5,0,0],
                        [0,-200,3,175,220,5,0,0],
                        [-1000,0,3,15,15,5,0,0],
                        [-1000,0,2,5,5,2,0,0],
                        [0,0,0,0,0,0,0,0]
                    ][j][k]
                else:
                    placementLayer[i][j][k] = [
                        [0,0,0,0,0,0,0,0],
                        [0,0,2,5,5,2,0,-1000],
                        [0,0,5,15,15,3,0,-1000],
                        [0,0,5,220,175,3,-200,0],
                        [0,0,5,220,175,3,-200,0],
                        [0,0,5,15,15,3,0,-1000],
                        [0,0,2,5,5,2,0,-1000],
                        [0,0,0,0,0,0,0,0]
                    ][j][k]
   
    # Initialize layer2
    for i in range(2):
        for j in range(8):
            for k in range(8):
                if i == 0:
                    layer2[i][j][k] = [
                        [0,0,1,2,3,15,40,0],
                        [0,0,2,5,15,25,40,0],
                        [0,0,3,5,25,35,40,0],
                        [0,0,3,5,25,35,40,0],
                        [0,0,3,5,25,35,40,0],
                        [0,0,3,5,25,35,40,0],
                        [0,0,2,5,15,25,40,0],
                        [0,0,1,2,3,15,40,0]
                    ][j][k]
                else:
                    layer2[i][j][k] = [
                        [0,40,15,3,2,1,0,0],
                        [0,40,25,15,5,2,0,0],
                        [0,40,35,25,5,3,0,0],
                        [0,40,35,25,5,3,0,0],
                        [0,40,35,25,5,3,0,0],
                        [0,40,35,25,5,3,0,0],
                        [0,40,25,15,5,2,0,0],
                        [0,40,15,3,2,1,0,0]
                    ][j][k]
    temp = chess.Board(None)  # Create an empty board with no pieces
    king_piece = chess.Piece(chess.KING, chess.WHITE)  # Create a king piece of the given color
    temp.set_piece_at(board.king(chess.WHITE), king_piece)

    for square in temp.attacks(temp.king(chess.WHITE)):
        y = square >> 3
        x = square - (y << 3)
        layer[1][x][y] += increment
        
        temp2 = chess.Board(None)  # Create an empty board with no pieces
        king_piece = chess.Piece(chess.KING, chess.WHITE)  # Create a king piece of the given color
        temp2.set_piece_at(square, king_piece)
        
        for square in temp2.attacks(temp2.king(chess.WHITE)):
            
            y = square >> 3
            x = square - (y << 3)
            layer[1][x][y] += increment
    
    temp.turn = False
    temp = chess.Board(None)  # Create an empty board with no pieces
    king_piece = chess.Piece(chess.KING, chess.BLACK)  # Create a king piece of the given color
    temp.set_piece_at(board.king(chess.BLACK), king_piece)

    for square in temp.attacks(temp.king(chess.BLACK)):
        y = square >> 3
        x = square - (y << 3)
        layer[0][x][y] += increment
            
        temp2 = chess.Board(None)  # Create an empty board with no pieces
        temp2.turn = False
        king_piece = chess.Piece(chess.KING, chess.BLACK)  # Create a king piece of the given color
        temp2.set_piece_at(square, king_piece)
        
        for square in temp2.attacks(temp2.king(chess.BLACK)):
            y = square >> 3
            x = square - (y << 3)
            
            layer[0][x][y] += increment

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
        initialize_layers(self.pgnBoard)

    cpdef alphaBetaWrapper(self, int curDepth, int depthLimit):
        initialize_layers(self.pgnBoard)          
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
        moves_list = self.reorder_legal_moves()
        cdef int num_legal_moves = len(moves_list)
        cdef int best_move_index = -1
        cdef int count = 1
        cdef int depthUsage = 0
        print("Num Moves: ", num_legal_moves)
        #moves_list = reorder_capture_moves(self.pgnBoard)
        
        self.numIterations = 0
        
        self.pgnBoard.push(moves_list[0])
        score = self.minimizer(curDepth + 1, depthLimit + 1, alpha, beta)
        self.pgnBoard.pop()
        
        if score > bestMove.score:
            cur = moves_list[0].uci()
            
            bestMove.score = score
            bestMove.a = ord(cur[0]) - 96
            bestMove.b = int(cur[1])
            bestMove.c = ord(cur[2]) - 96
            bestMove.d = int(cur[3])
            
            best_move_index = 0
            
        alpha = max(alpha, bestMove.score)
        
        for move in moves_list[1:]:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0
            
            if (count <= 4):
                depthUsage = depthLimit + 1
            elif (count >= 30):
                depthUsage = depthLimit - 1
            else:
                depthUsage = depthLimit
            self.pgnBoard.push(move)
            score = self.minimizer(curDepth + 1, depthUsage, alpha, alpha + 1)
                        
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                score = self.minimizer(curDepth + 1, depthLimit, alpha, beta)
            
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
                
                best_move_index = count
                
            alpha = max(alpha, bestMove.score)
            count += 1
            if beta <= alpha:
                self.numMove += 1
                print(self.numIterations)
                print("Best: ", best_move_index)
                return bestMove

        if curDepth == 0:
            self.numMove += 1
            print(self.numIterations)
            print("Best: ", best_move_index)
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
            self.numIterations += 1
            return evaluate_board(self.pgnBoard)

        #cdef cnp.ndarray[DTYPE_INT, ndim=4] inputBoard = np.array([encode_board(self.pgnBoard)], dtype=np.int8)
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prediction = self.blackModel.predict(inputBoard, verbose=0)

        cdef list moves_list
        moves_list = reorder_capture_moves(self.pgnBoard)
        #moves_list = self.reorder_legal_moves()
        
        for move in moves_list:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0
            
            self.pgnBoard.push(move)
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta)
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
                # if (self.pgnBoard.can_claim_threefold_repetition()):
                #     score = -100000000
                # else:
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
            self.numIterations += 1
            return evaluate_board(self.pgnBoard)

        #cdef cnp.ndarray[DTYPE_INT, ndim=4] inputBoard = np.array([encode_board(self.pgnBoard)], dtype=np.int8)
        #cdef cnp.ndarray[DTYPE_FLOAT, ndim=2] prediction = self.whiteModel.predict(inputBoard, verbose=0)

        cdef list moves_list
        moves_list = reorder_capture_moves(self.pgnBoard)
        
        for move in moves_list:
            
            #index = reversePrediction(a, b, c, d) - 1
            #filteredPrediction[index] = prediction[0, index]

            #for i in range(15):
            # index = np.argmax(filteredPrediction)
            # result = predictionInfo(index)
            # a, b, c, d = result.x, result.y, result.w, result.z
            
            # filteredPrediction[index] = 0
            
            self.pgnBoard.push(move)
            score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
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

        if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):
            if self.pgnBoard.is_checkmate():
                return 100000000
            elif self.pgnBoard.is_stalemate():
                return -100000000
            elif self.pgnBoard.can_claim_draw():
                return -100000000
        return lowestScore

    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef reorder_legal_moves(self):
        
        cdef int alpha = -9999998
        cdef int beta = 9999998
        
        cdef int score = -99999999
        cdef int highestScore = -99999999
        cdef str cur
        cdef list moves_list
        cdef int count = 1
        cdef int depth = 3
          
        moves_list = reorder_capture_moves(self.pgnBoard)
        self.pgnBoard.push(moves_list[0])
        highestScore = self.minimizer(1, depth, alpha, beta)
        self.pgnBoard.pop()

        alpha = max(alpha, highestScore)
        
        for move in moves_list[1:]:
            
            self.pgnBoard.push(move)
            score = self.minimizer(1, depth, alpha, alpha + 1)
            
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                score = self.minimizer(1, depth, alpha, beta)
            
            self.pgnBoard.pop()
            
            if score > highestScore:
                highestScore = score
                
                # Shift the other elements down
                for j in range(count, 0, -1):                
                    moves_list[j] = moves_list[j-1]

                # Place the stored element at the front
                moves_list[0] = move
            count += 1
            alpha = max(alpha, highestScore)

            if beta <= alpha:
                break
        
        return moves_list

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
def reorder_capture_moves(object board):
    
    cdef list legal_moves = []
    cdef list capture_moves = []
    #cdef list check_moves = []
    cdef list non_capture_moves = []
    cdef object move

    # Iterate through all legal moves
    for move in board.legal_moves:
        if board.is_capture(move):
            capture_moves.append(move)
        # elif board.gives_check(move):
        #     check_moves.append(move)
        else:
            non_capture_moves.append(move)

    # Append capture moves first followed by non-capture moves
    legal_moves.extend(capture_moves)
    # legal_moves.extend(check_moves)
    legal_moves.extend(non_capture_moves)

    return legal_moves

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
@cython.inline
cdef int placement_and_piece_eval(object board, uint8_t square, bint colour, uint8_t piece_type, int moveNum, int values [7], int[:,:,:] activePlacementLayer, int[:,:,:] activeAttackingLayer):
    
    cdef int total = 0
    cdef uint8_t  x, y

    y = square >> 3
    x = square - (y << 3)

    # Evaluate based on piece color
    if colour:
        total -= values[piece_type]
        
        if not (piece_type == 4) or moveNum >= 40:
            total -= activePlacementLayer[0][x][y]
            
            if (piece_type == 1):
                total -= (y + 1) * 25
        for attack in board.attacks(square): 
            y = attack >> 3
            x = attack - (y << 3)
            if (piece_type == 1 or piece_type == 5):
                total -= activeAttackingLayer[0][x][y] >> 2
            else:    
                total -= activeAttackingLayer[0][x][y]  
                
    else:
        total += values[piece_type]
        if not (piece_type == 4) or moveNum >= 40:
            total += activePlacementLayer[1][x][y]
            
            if (piece_type == 1):
                total += (8 - y) * 25
        for attack in board.attacks(square): 
            y = attack >> 3
            x = attack - (y << 3)
            if (piece_type == 1 or piece_type == 5):
                total -= activeAttackingLayer[1][x][y] >> 2
            else:    
                total -= activeAttackingLayer[1][x][y] 
                
    return total

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef int evaluate_board(object board):
    
    global whiteCastledIndex
    global blackCastledIndex
    
    cdef int total = 0
    cdef int subTotal = 0
    cdef object piece
    cdef uint8_t  square
    cdef uint8_t  x, y
    cdef uint8_t i
    cdef int moveNum = len(board.move_stack)
    cdef object target_square
    cdef object target_move
    
    cdef white_ksc = chess.Move.from_uci('e1g1')
    cdef white_qsc = chess.Move.from_uci('e1c1')
    cdef black_ksc = chess.Move.from_uci('e8g8')
    cdef black_qsc = chess.Move.from_uci('e8c8')
        
    # Determine active layers based on move count
    cdef int[:,:,:] activeAttackingLayer = layer
    cdef int[:,:,:] activePlacementLayer = layer2 if moveNum >= 40 else placementLayer

    cdef int values[7]
    cdef int castle_index = -1

    # Initialize the array in C-style
    values[0] = 0      # No piece
    values[1] = 1000   # Pawn
    values[2] = 3150   # Knight
    values[3] = 3250   # Bishop
    values[4] = 5000   # Rook
    values[5] = 9000   # Queen
    values[6] = 0      # King
    
    # Iterate through all squares on the board and evaluate piece values
    if board.is_checkmate():
        if board.turn:
            total = -100000000            
        else:
            total = 100000000
    else:
        for square in chess.SquareSet(board.occupied):
            piece = board.piece_at(square)
            total += placement_and_piece_eval(board, square, piece.color, piece.piece_type, moveNum, values, activePlacementLayer, activeAttackingLayer)
            
            
        if (len(array('i', board.pieces(chess.BISHOP, chess.WHITE))) == 2):
            total -= 315
        if (len(array('i', board.pieces(chess.KNIGHT, chess.WHITE))) == 2):
            total -= 300
        if (len(array('i', board.pieces(chess.BISHOP, chess.BLACK))) == 2):
            total += 315
        if (len(array('i', board.pieces(chess.KNIGHT, chess.BLACK))) == 2):
            total += 300
        
        castle_index = move_index (board, white_ksc, white_qsc)
        if (castle_index != -1):
            total -= max(3000 - ((castle_index-1) >> 1) * 100 - moveNum * 50, 0)
        
        castle_index = move_index (board, black_ksc, black_qsc)
        if (castle_index != -1):            
            total += max(3000 - ((castle_index-1) >> 1) * 100 - moveNum * 50, 0)
            #print("AAAA", castle_index, moveNum, 3000 - ((castle_index-1) >> 1) * 100 - moveNum * 50)
        
        target_move = board.peek()
        if (board.is_capture(target_move)):
            target_square = target_move.to_square
            for move in board.legal_moves:
                    if move.to_square == target_square:
                        if (board.turn):
                            total -= values[board.piece_at(target_square).piece_type]
                        else:                        
                            total += values[board.piece_at(target_square).piece_type]
                        break      
    return total

import chess

cdef int move_index(object board, object move1, object move2):
    for index, move in enumerate(board.move_stack):
        if move == move1 or move == move2:
            return index
    return -1

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
