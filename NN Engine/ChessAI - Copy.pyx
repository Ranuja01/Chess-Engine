# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:32:26 2024

@author: Kumodth
"""
import chess  # Use regular import for Python libraries
cimport cython  # Import Cython-specific utilities
from cython cimport boundscheck, wraparound
import json
import marshal
import moveSchema_pb2  # Import the generated code
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
from joblib import Parallel, delayed
from numba import njit
from timeit import default_timer as timer
from functools import lru_cache
import Cython_Chess
import multiprocessing
import time
import itertools
from typing import Iterator
import tensorflow as tf
from tensorflow.keras.models import Model

# Import data structures from the c++ standard library
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t

# Import functions from c++ file
cdef extern from "cpp_bitboard.h":
    uint8_t scan_reversed_size(uint64_t bb)
    void scan_reversed(uint64_t bb, vector[uint8_t] &result)
    void scan_forward(uint64_t bb, vector[uint8_t] &result)
    int getPPIncrement(int square, bint colour, uint64_t opposingPawnMask, int ppIncrement, int x)
    uint64_t attacks_mask(bint colour, uint64_t occupied, uint8_t square, uint8_t pieceType)
    uint64_t attackersMask(bint colour, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co)
    uint64_t slider_blockers(uint8_t king, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t occupied_co_opp, uint64_t occupied_co, uint64_t occupied)
    uint64_t betweenPieces(uint8_t a, uint8_t b)
    uint64_t ray(uint8_t a, uint8_t b)
    bint is_capture(uint8_t from_square, uint8_t to_square, uint64_t occupied_co, bint is_en_passant)
    void initialize_attack_tables()
    void setAttackingLayer(uint64_t occupied_white, uint64_t occupied_black, uint64_t kings, int increment);
    int placement_and_piece_midgame(uint8_t square, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    int placement_and_piece_endgame(uint8_t square, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    int placement_and_piece_eval(int moveNum, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t prevKings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    void initializeZobrist()
    uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask);
    void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bint isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion)
    int accessCache(uint64_t key)
    void addToCache(uint64_t key,int value)
    string accessOpponentMoveGenCache(uint64_t key);
    void addToOpponentMoveGenCache(uint64_t key,char* data, int length);
    string accessCurPlayerMoveGenCache(uint64_t key);
    void addToCurPlayerMoveGenCache(uint64_t key,char* data, int length);
    int printCacheStats()
    int printOpponentMoveGenCacheStats();
    int printCurPlayerMoveGenCacheStats();
    void evictOldEntries(int numToEvict)
    void evictOpponentMoveGenEntries(int numToEvict)
    void evictCurPlayerMoveGenEntries(int numToEvict)
    
# Create struct to hold information regarding the chosen move by the engine
cdef struct MoveData:
    int a
    int b
    int c
    int d
    int promotion
    int score

cdef uint64_t prevKings
cdef int blackCastledIndex = -1
cdef int whiteCastledIndex = -1

cdef object white_ksc = chess.Move.from_uci('e1g1')
cdef object white_qsc = chess.Move.from_uci('e1c1')
cdef object black_ksc = chess.Move.from_uci('e8g8')
cdef object black_qsc = chess.Move.from_uci('e8c8')

cdef int values[7]
values[0] = 0      # No piece
values[1] = 1000   # Pawn
values[2] = 3150   # Knight
values[3] = 3250   # Bishop
values[4] = 5000   # Rook
values[5] = 9000   # Queen
values[6] = 0      # King

# Define class for the chess engine
@cython.cclass
cdef class ChessAI:
    
    # Member variables for neural networks
    cdef object blackModel
    cdef object whiteModel
    
    # Member variable to hold the chess board
    cdef object pgnBoard
    
    # Member variables to hold the move number, the number of static evaluations
    cdef int numMove
    cdef int numIterations
    
    # Dictionary to hold time thresholds for using higher depth
    cdef dict move_times
    
    # Member variable to hold zobrist hash for the given position
    cdef uint64_t zobrist
    
    # Lists to hold moves, alpha and beta values for previous move searches for iterative deepening
    cdef list moves_list
    cdef list alpha_list
    cdef list beta_list
    cdef list beta_move_list
    
    # Absolute time limit for move making
    cdef int time_limit
    
    # Max depth for quiescence search
    cdef int quiescenceDepth
    
    # Constructor for chess engine
    def __cinit__(self, object black_model, object white_model, object board):
        
        # Initialize member variables
        self.blackModel = black_model
        self.whiteModel = white_model
        self.pgnBoard = board
        self.numMove = 0
        self.numIterations = 0
        self.move_times = {}
        self.moves_list = []
        self.alpha_list = []
        self.beta_list = []
        self.beta_move_list = []
        self.move_times[4] = 5.0
        self.move_times[5] = 5.5
        self.time_limit = 60
        self.quiescenceDepth = 6
        
        for i in range(6,26):
            self.move_times[i] = 2.5
        
        # Initialize attack tables for move generation
        initialize_attack_tables()
        Cython_Chess.inititalize()
        
        # Initialize zobrist tables for hashing
        initializeZobrist()
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False])    
    
    # Function to set global variable for white castling index
    def setWhiteCastledIndex(self, index):
        global whiteCastledIndex
        whiteCastledIndex = index
    
    # Function to set global variable for black castling index
    def setBlackCastledIndex(self, index):
        global blackCastledIndex
        blackCastledIndex = index            

    # Function to wrap the 
    def alphaBetaWrapper(self, int curDepth, int depthLimit):
        
        # Start timer
        t0= timer()
        
        # Call global variables into the context of this function
        global prevKings
        global whiteCastledIndex
        global blackCastledIndex
        
        # Initialize the lists required for iterative deepening
        self.moves_list = []
        self.alpha_list = []
        self.beta_list = []
        self.beta_move_list = []
        self.numIterations = 0
        
        print(whiteCastledIndex,blackCastledIndex)
        cdef int cacheSize = printCacheStats()
        # print()
        # cdef int OpponentMoveGenCacheSize = printOpponentMoveGenCacheStats()
        # print()
        # cdef int CurPlayerMoveGenCacheSize = printCurPlayerMoveGenCacheStats()
        
        # if (OpponentMoveGenCacheSize > 450000):
        #     evictOpponentMoveGenEntries(OpponentMoveGenCacheSize - 450000)
        
        # if (CurPlayerMoveGenCacheSize > 450000):
        #     evictCurPlayerMoveGenEntries(CurPlayerMoveGenCacheSize - 450000)
        
        # Code segment to control cache size
        if (self.pgnBoard.ply() < 30):
            if (cacheSize > 8000000):
                evictOldEntries(cacheSize - 8000000)                
        elif(self.pgnBoard.ply() < 50):
            if (cacheSize > 16000000):
                evictOldEntries(cacheSize - 16000000)
        elif(self.pgnBoard.ply() < 75):
            if (cacheSize > 32000000):
                evictOldEntries(cacheSize - 32000000)
        else:
            if (cacheSize > 64000000):
                evictOldEntries(cacheSize - 64000000)
        
        # Set the variable for where the king was located before move selection is started
        prevKings = self.pgnBoard.kings
        
        # Define variables to hold generated move data
        cdef MoveData result
        cdef int a, b, c, d,promo,val
        cdef object move
        
        self.zobrist = generateZobristHash(self.pgnBoard.pawns,self.pgnBoard.knights,self.pgnBoard.bishops,self.pgnBoard.rooks,self.pgnBoard.queens,self.pgnBoard.kings,self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False])
        
        # Code segment to check if the opponent has castled and set the castled index
        if (len(self.pgnBoard.move_stack) > 0):
            move = self.pgnBoard.pop()
            
            if (self.pgnBoard.turn):
                if (whiteCastledIndex == -1):
                    if (self.pgnBoard.is_castling(move)):
                        print ("WHITE CASTLED")
                        whiteCastledIndex = self.pgnBoard.ply()
            else:
                if (blackCastledIndex == -1):
                    if (self.pgnBoard.is_castling(move)):
                        print ("BLACK CASTLED")
                        blackCastledIndex = self.pgnBoard.ply()
            
            self.pgnBoard.push(move)
        
        # Code segment to check if either side has lost castling rights
        if (self.pgnBoard.turn):
            if (whiteCastledIndex == -1):
                if not(self.pgnBoard.has_castling_rights(True)):
                    print("WHITE CASTLING LOST")
                    whiteCastledIndex = 121
            if (blackCastledIndex == -1):
                if not(self.pgnBoard.has_castling_rights(False)):
                    print("BLACK CASTLING LOST")
                    blackCastledIndex = 121
        else:
            if (whiteCastledIndex == -1):
                if not(self.pgnBoard.has_castling_rights(True)):
                    print("WHITE CASTLING LOST")
                    whiteCastledIndex = 121
            if (blackCastledIndex == -1):
                if not(self.pgnBoard.has_castling_rights(False)):
                    print("BLACK CASTLING LOST")
                    blackCastledIndex = 121    
            
        print(whiteCastledIndex,blackCastledIndex)
        
        # If less than 30 plies have been played, check the opening book
        if (len(self.pgnBoard.move_stack) < 30):
            result = self.opening_book(curDepth, depthLimit)
                                      
            a = result.a
            b = result.b
            c = result.c
            d = result.d
            promo = result.promotion
            val = result.score
                        
            t1 = timer()
            # Check if an entry exists in the opening book
            if not((a,b,c,d) == (-1,-1,-1,-1)):  
                print(a,b,c,d)
                print()
                print("Evaluation: Book Move")
                print ("Time Taken: ", t1 - t0)
                print("Move: ", self.pgnBoard.ply())
                print()
                
                # Convert the coordinates to alphanumeric representation
                x = chr(a + 96)
                y = str(b)
                i = chr(c + 96)
                j = str(d)
                if (promo == -1):
                    move = chess.Move.from_uci(x+y+i+j)
                    
                    # Check if the engine has castled and set the castling index
                    if (self.pgnBoard.turn):
                        if (whiteCastledIndex == -1):
                            if (self.pgnBoard.is_castling(move)):
                                print ("WHITE CASTLED")
                                whiteCastledIndex = self.pgnBoard.ply()
                    else:
                        if (blackCastledIndex == -1):
                            if (self.pgnBoard.is_castling(move)):
                                print ("BLACK CASTLED")
                                blackCastledIndex = self.pgnBoard.ply()
                                
                    return move
                else:
                    return chess.Move.from_uci(x+y+i+j+chr(promo + 96))
        
        # Call the alpha beta algorithm to make a move decision
        result = self.alphaBeta(curDepth=0, depthLimit=4, t0 = timer())
        val = result.score
        t1 = timer()
        dif = t1 - t0
        new_depth = 5
        
        # Check if the move generation time and value is low enough to warrant a deeper search
        while(dif <= self.move_times[new_depth-1] and new_depth <= 25 and val < 9000000):
            
            a = result.a
            b = result.b
            c = result.c
            d = result.d
            promo = result.promotion
            
            if (val <= -15000):
                return None
            print(a,b,c,d)
            print()
            print("TRYING DEPTH: ", new_depth)
            t0_new = timer()
            result = self.alphaBeta(curDepth=0, depthLimit=new_depth,  t0 = timer())
            new_depth += 1
            val = result.score
            t1 = timer()
            dif = t1 - t0_new
        
        a = result.a
        b = result.b
        c = result.c
        d = result.d
        promo = result.promotion
        val = result.score
        print(a,b,c,d)        
        
        if not((a,b,c,d) == (-1,-1,-1,-1)):  
            
            print()
            print("Evaluation: ", val)
            print("Positions Analyzed: ",self.numIterations)
            print("Average Static Analysis Speed: ",self.numIterations/ (t1 - t0))
            print ("Time Taken: ", t1 - t0)
            print("Move: ", self.pgnBoard.ply())
            print()
            
            # Convert the coordinates to alphanumeric representation
            x = chr(a + 96)
            y = str(b)
            i = chr(c + 96)
            j = str(d)
            
            # Check if the move is a promoting move
            if (promo == -1):
                move = chess.Move.from_uci(x+y+i+j)
                
                # Check if the engine has castled and set the castling index
                if (self.pgnBoard.turn):
                    if (whiteCastledIndex == -1):
                        if (self.pgnBoard.is_castling(move)):
                            print ("WHITE CASTLED")
                            whiteCastledIndex = self.pgnBoard.ply()
                else:
                    if (blackCastledIndex == -1):
                        if (self.pgnBoard.is_castling(move)):
                            print ("BLACK CASTLED")
                            blackCastledIndex = self.pgnBoard.ply()
                            
                return move
            else:
                return chess.Move.from_uci(x+y+i+j+chr(promo + 96))
        else:
            return None
    
    # Function for opening book moves
    @cython.ccall
    @cython.exceptval(check=False)
    cdef MoveData opening_book(self, curDepth, depthLimit):
        cdef MoveData best_move
        cdef str cur
        
        best_move.a = -1
        best_move.b = -1
        best_move.c = -1
        best_move.d = -1
        best_move.promotion = -1
        best_move.score = -99999999
        
        # Open the polyglot book
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
                best_move.b = ord(cur[1]) - ord('0')
                best_move.c = ord(cur[2]) - 96
                best_move.d = ord(cur[3]) - ord('0')
                
                
            return best_move
            
                
    # Define the alphaBeta function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef MoveData alphaBeta(self, int curDepth, int depthLimit, double t0):
        
        # Initialize alpha and beta values
        cdef int alpha = -9999998
        cdef int beta = 9999998
        
        # Initialize variables to hold the current move, its string representation and score
        cdef int score
        cdef object move
        cdef str cur
        
        # Define and initialize the struct to return the best move
        cdef MoveData bestMove
        bestMove.a = -1
        bestMove.b = -1
        bestMove.c = -1
        bestMove.d = -1
        bestMove.promotion = -1
        bestMove.score = -99999999
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied    
    
        # Define lists to hold move lists and respective scores acquired in previous search
        cdef list moves_list
        cdef list alpha_list
        cdef list beta_list
        cdef list beta_move_list
        
        # Define variable to hold the zobrist hash for the current board state
        cdef uint64_t curHash = self.zobrist
        
        # Call function to sort moves based on a previous search
        moves_list, alpha_list,beta_list, beta_move_list = self.reorder_legal_moves(alpha,beta, depthLimit)
        
        # Define the razor threshold, if not the first move search iteration, razor more aggressively
        cdef int razorThreshold
        if (self.alpha_list == []):
            razorThreshold = max (int(750 * .75** (depthLimit - 5)), 200)
        else:
            razorThreshold = max (int(300 * .75** (depthLimit - 5)), 50)
        
        # After the in scope lists have been initialized, the global ones can be reset
        self.alpha_list = []    
        self.beta_list = []
        self.beta_move_list = []
        
        # Define the number of moves, the best move index and the current index
        cdef int num_legal_moves = len(moves_list)
        cdef int best_move_index = -1
        cdef int count = 1
        
        # Define the depth that should be used
        cdef int depthUsage = 0                
        
        # Define variables to hold information for zobrist hashing
        cdef bint isCapture
        cdef int promotion = 0
        
        # Define variables to hold information on repeating moves
        cdef bint repetitionFlag = False
        cdef object repetitionMove = None
        cdef int repetitionScore = 0
        
        if (depthLimit >= 5):
            print("Num Moves: ", num_legal_moves)        
        
        # Check if the move is a promoting move
        if (moves_list[0].promotion):
            promotion = moves_list[0].promotion
        else:
            promotion = 0
        
        # Acquire the zobrist hash for the new position if the given move was made
        isCapture = is_capture(moves_list[0].from_square, moves_list[0].to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(moves_list[0]))
        updateZobristHashForMove(self.zobrist, moves_list[0].from_square, moves_list[0].to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
        
        # Make the move and call the minimizer
        self.pgnBoard.push(moves_list[0])
        score = self.minimizer(curDepth + 1, depthLimit, alpha, beta, beta_list[0], beta_move_list[0])
        
        # Check if the position is repeating after the move
        if (self.pgnBoard.is_repetition(2)):
            repetitionFlag = True
            repetitionMove = moves_list[0]
            repetitionScore = score
            score = -100000000
        
        # Check if the move causes a stalemate
        if (self.pgnBoard.is_stalemate()):            
            score = -100000000
        
        # Undo the move and restore the zobrist hash
        self.pgnBoard.pop()
        self.zobrist = curHash
        
        if (depthLimit >= 5):
            print(0,score,alpha_list[0],moves_list[0])
        
        # Assign the best move struct and alpha
        if score > bestMove.score:
            cur = moves_list[0].uci()
            
            bestMove.score = score
            bestMove.a = ord(cur[0]) - 96
            bestMove.b = ord(cur[1]) - ord('0')
            bestMove.c = ord(cur[2]) - 96
            bestMove.d = ord(cur[3]) - ord('0')
            if (moves_list[0].promotion):
                bestMove.promotion = ord(cur[4]) - 96
            else:
                bestMove.promotion = -1
            best_move_index = 0
            
        alpha = max(alpha, bestMove.score)
        
        # Append the global moves list and alpha list to store the current score
        self.moves_list = moves_list
        self.alpha_list.append(score)
        
        # Check if the search time has exceeded
        if (timer() - t0 >= self.time_limit):
            return bestMove
        
        for move in moves_list[1:]:
            
            # Razoring
            if (not(alpha_list[count] == None)):
                if (alpha - alpha_list[count] > razorThreshold) and alpha < 9000000:                    
                    break
            
            # Late move reduction
            if (count >= 35):
                depthUsage = depthLimit - 1
            else:
                depthUsage = depthLimit
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Make the move and call the minimizer
            self.pgnBoard.push(move)            
            score = self.minimizer(curDepth + 1, depthUsage, alpha, alpha+1, beta_list[count], beta_move_list[count])           
                        
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                
                # Ensure there is a score for the given index
                if (not(alpha_list[count] == None)):
                    
                    # Pop the lists for the re-search
                    self.beta_list.pop()
                    self.beta_move_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, beta, beta_list[count],beta_move_list[count])
                else:
                    
                    # Pop the lists for the re-search
                    self.beta_list.pop()
                    self.beta_move_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, beta, beta_list[count],beta_move_list[count])                
            
            # Check if the position is repeating after the move
            if (self.pgnBoard.is_repetition(2)):
                repetitionFlag = True
                repetitionMove = move
                repetitionScore = score
                score = -100000000
            
            # Check if the move causes a stalemate
            if (self.pgnBoard.is_stalemate()):            
                score = -100000000
            
            # Undo the move, restore the zobrist hash and append the score list for the current move
            self.pgnBoard.pop()
            self.zobrist = curHash
            self.alpha_list.append(score)
            
            if (depthLimit >= 5):
                print(count,score,alpha_list[count], move)
            
            # Check if the current move's score is better than the existing best move
            if score > bestMove.score:
                cur = move.uci()
                
                bestMove.score = score
                bestMove.a = ord(cur[0]) - 96
                bestMove.b = ord(cur[1]) - ord('0')
                bestMove.c = ord(cur[2]) - 96
                bestMove.d = ord(cur[3]) - ord('0')
                
                if (move.promotion):
                    bestMove.promotion = ord(cur[4]) - 96
                else:
                    bestMove.promotion = -1
                best_move_index = count
                
            alpha = max(alpha, bestMove.score)
            count += 1
            
            # Check for a beta cutoff 
            if beta <= alpha:
                self.numMove += 1
                if (depthLimit >= 5):
                    print()                
                    print("Best: ", best_move_index)
                
                for i in range(num_legal_moves - count):
                    self.alpha_list.append(None)
                # print(self.alpha_list)
                return bestMove
            
            # Check if the time limit is exceeded
            if (timer() - t0 >= self.time_limit):
                
                # Check if a repeating move is detected
                if (repetitionFlag):
                    # Check if the best move excluding the repeating move is not good enough to be played
                    if (alpha < repetitionScore):
                        if (alpha <= -500):
                            cur = repetitionMove.uci()
                            
                            bestMove.score = 0
                            bestMove.a = ord(cur[0]) - 96
                            bestMove.b = ord(cur[1]) - ord('0')
                            bestMove.c = ord(cur[2]) - 96
                            bestMove.d = ord(cur[3]) - ord('0')
                            
                            bestMove.promotion = -1
                            
                            return bestMove
                
                if (repetitionFlag):
                    if (alpha < repetitionScore):
                        if (alpha <= -500):
                            cur = repetitionMove.uci()
                            
                            bestMove.score = 0
                            bestMove.a = ord(cur[0]) - 96
                            bestMove.b = ord(cur[1]) - ord('0')
                            bestMove.c = ord(cur[2]) - 96
                            bestMove.d = ord(cur[3]) - ord('0')
                            
                            bestMove.promotion = -1
                            
                            return bestMove                    
                return bestMove
        
        # Fill the non utilized alpha list to full capacity
        for i in range(num_legal_moves - count):
            self.alpha_list.append(None)
        
        if curDepth == 0:
            self.numMove += 1
            print(repetitionFlag, repetitionMove, repetitionScore, alpha)
            # Check if a repeating move is detected
            if (repetitionFlag):
                
                # Check if the best move excluding the repeating move is not good enough to be played
                if (alpha < repetitionScore):                    
                    if (alpha <= -500):
                    
                        cur = repetitionMove.uci()
                        
                        bestMove.score = 0
                        bestMove.a = ord(cur[0]) - 96
                        bestMove.b = ord(cur[1]) - ord('0')
                        bestMove.c = ord(cur[2]) - 96
                        bestMove.d = ord(cur[3]) - ord('0')
                        
                        bestMove.promotion = -1
                        
            if (depthLimit >= 5):
                print()            
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
        
        # If the depth limit is reached, evaluate the current position
        if curDepth >= depthLimit:
            self.numIterations += 1
            return evaluate_board(self.pgnBoard,self.zobrist)            
            # return self.quiescenceMax(alpha, beta, 0)
        
        # Initialize variables to hold the highest score, the current score and current move
        cdef int highestScore = -9999999
        cdef int score
        cdef object move                        
        
        # Define variable to hold the zobrist hash for the current board state
        cdef uint64_t curHash = self.zobrist        
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied
        
        # Define variables to hold information for zobrist hashing
        cdef int promotion = 0     
        cdef bint isCapture

        # Acquire a moves list ordered such that captures are first
        # cdef list moves_list = list(self.reorder_capture_moves(chess.BB_ALL, self.pgnBoard))
        
        for move in self.reorder_capture_moves(chess.BB_ALL, self.pgnBoard):
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the given move and call the minimizer
            self.pgnBoard.push(move)
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta, [],[])
            
            # Undo the move and restore the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            # ** Code for testing purposes **
            
            # if (self.pgnBoard == chess.Board("1r3rk1/p1p2p1p/2p5/3p2p1/8/1P5P/P1PBBK2/RN6 b - - 0 18")):                
            #     print ("MAX: ",score, move)          
                
            # if (self.pgnBoard == chess.Board("1r3rk1/p1p4p/2p5/3p1pB1/8/1P5P/P1P1BK2/RN6 b - - 0 19")):            
            #     print ("MAX2: ",score, move)
            
            # if (self.pgnBoard == chess.Board("5rk1/6pp/Q1pqpp2/3p3P/1P1P4/4P3/5P1P/5RK1 b - - 0 23")):
            #     print ("MAX3: ",score, move)
            
            # Acquire the highest score and alpha
            if score > highestScore:
                highestScore = score

            alpha = max(alpha, highestScore)
            
            # Beta cutoff
            if beta <= alpha:
                return highestScore
        
        # Check if the last move resulted in a checkmate for the opposing player
        if (highestScore == -9999999):
            self.numIterations += 1
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
    cdef int minimizer(self, int curDepth, int depthLimit, int alpha, int beta, list beta_list_og, list beta_moves_list):
        
        # If the depth limit is reached, evaluate the current position
        if curDepth >= depthLimit:            
            self.numIterations += 1            
            return evaluate_board(self.pgnBoard,self.zobrist)
            # return self.quiescenceMin(alpha, beta, 0)
        
        # Define the lowest score with respect to the number of moves played
        cdef int lowestScore = 9999999 - len(self.pgnBoard.move_stack)
        
        # Define variables to hold the current move, score and index
        cdef int score
        cdef object move                
        cdef int count = 0
        
        # Define and initialize the razoring threshold
        cdef int razorThreshold
        if (depthLimit == 4):
            razorThreshold = max (int(1000 * .75** (depthLimit - 5)), 200) 
        else:
            razorThreshold = max (int(750 * .75** (depthLimit - 5)), 50)
            
        # Define variable to hold the zobrist hash for the current board state
        cdef uint64_t curHash = self.zobrist
        
        # Define lists to hold the current scores and copy of the expected scores
        cdef list cur_beta_list = []
        cdef list beta_list = beta_list_og.copy()
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        # Define variables to hold information for zobrist hashing
        cdef int promotion = 0
        cdef bint isCapture
        
        # Variable to hold full length of the list
        cdef int length = 0
        
        # Check for the second recursive depth to use the stored moves list instead of generating it
        cdef list moves_list
        if (curDepth == 1):
            moves_list = beta_moves_list.copy()
            quicksort_ascending_wrapper(beta_list, moves_list)
            self.beta_move_list.append(moves_list)
            length = len(moves_list)
           
        # Check for the second recursive depth
        if curDepth == 1:
            for move in moves_list:
                
                # Razoring
                if (not(beta_list[count] == None)):
                    if (beta_list[count] - beta > razorThreshold):
                        count+=1
                        cur_beta_list.append(None)
                        continue
                
                # Check if the move is a promoting move
                if (move.promotion):
                    promotion = move.promotion
                else:
                    promotion = 0
                
                # Acquire the zobrist hash for the new position if the given move was made
                isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
                updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
                
                # Push the given move and call the maximizer
                self.pgnBoard.push(move)
                score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
                
                # Undo the move and reset the zobrist hash
                self.pgnBoard.pop()
                self.zobrist = curHash
                
                # ** Code for testing purposes **
                
                if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1bpn2/2p5/2PPPP2/2NB4/P1Q2P1P/R1B2RK1 w - - 0 14")):         
                    # print(moves_list)
                    print ("MIN: ",score, move, alpha, beta, lowestScore)
                
                # if (self.pgnBoard == chess.Board("1r3rk1/p1p4p/2p5/3p1pp1/8/1P5P/P1PBBK2/RN6 w - - 0 19")):
                #     print ("MIN2: ",score, move, alpha, beta)
                
                # if (self.pgnBoard == chess.Board("5rk1/p1p4p/2p5/3p1pB1/8/1r5P/P1P1BK2/RN6 w - - 0 20")):
                #     print ("MIN3: ",score, move, alpha, beta)    
                
                # Store the move scores           
                cur_beta_list.append(score)
                    
                # Find the lowest score and beta
                if score < lowestScore:
                    lowestScore = score

                beta = min(beta, lowestScore)
                count+=1
                
                if beta <= alpha:
                      
                    # Fill up the remaining list to capacity
                    for i in range(length - count):
                        cur_beta_list.append(None)
                    self.beta_list.append(cur_beta_list)
                    return score
            
            # Check if no moves are available, inidicating a game ending move was made previously
            if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):            
                self.numIterations += 1

                # Fill up the remaining list to capacity
                for i in range(length - count):
                    cur_beta_list.append(None)
                self.beta_list.append(cur_beta_list)
                
                if self.pgnBoard.is_checkmate():
                    return 100000000
                else:
                    return min(beta,lowestScore)
           
            # Fill up the remaining list to capacity
            for i in range(length - count):
                cur_beta_list.append(None)
            self.beta_list.append(cur_beta_list)    
        else: # If not the second recursive depth, take advantage of the yielding feature to increase speed
            for move in self.reorder_capture_moves(chess.BB_ALL, self.pgnBoard):
                
                # Check if the move is a promoting move
                if (move.promotion):
                    promotion = move.promotion
                else:
                    promotion = 0
                
                # Acquire the zobrist hash for the new position if the given move was made
                isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
                updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
                
                # Push the given move and call the maximizer
                self.pgnBoard.push(move)
                score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
                
                # Undo the move and reset the zobrist hash
                self.pgnBoard.pop()
                self.zobrist = curHash
                
                # ** Code for testing purposes **
                
                # if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1bpn2/2p5/2PPPP2/2NB4/P1Q2P1P/R1B2RK1 w - - 0 14")):         
                #     # print(moves_list)
                #     print ("MIN: ",score, move, alpha, beta, lowestScore)
                
                # if (self.pgnBoard == chess.Board("1r3rk1/p1p4p/2p5/3p1pp1/8/1P5P/P1PBBK2/RN6 w - - 0 19")):
                #     print ("MIN2: ",score, move, alpha, beta)
                
                # if (self.pgnBoard == chess.Board("5rk1/p1p4p/2p5/3p1pB1/8/1r5P/P1P1BK2/RN6 w - - 0 20")):
                #     print ("MIN3: ",score, move, alpha, beta)    
                
                    
                # Find the lowest score and beta
                if score < lowestScore:
                    lowestScore = score

                beta = min(beta, lowestScore)
                count+=1
                
                if beta <= alpha:                    
                    return score
            
            # Check if no moves are available, inidicating a game ending move was made previously
            if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):            
                self.numIterations += 1
                
                if self.pgnBoard.is_checkmate():
                    return 100000000
                else:
                    return min(beta,lowestScore)
               
        return lowestScore
        
    # Define the quiescence maximizer function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int quiescenceMax(self, int alpha, int beta, int quiescenceDepth):
        self.numIterations += 1
        # Get an evaluation and see if the evaluation is close to the alpha and beta or exceeds the quiescence depth
        cdef int evaluation = evaluate_board(self.pgnBoard, self.zobrist)
        if (quiescenceDepth >= self.quiescenceDepth) or evaluation - 1500 >= beta or evaluation + 1500 <= alpha or evaluation >= 9000000:
            return evaluation
        
        # Define variable to hold the zobrist hash for the current board state 
        cdef uint64_t curHash = self.zobrist   
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied
        
        # Define variables to hold information for zobrist hashing
        cdef bint isCapture
        cdef int promotion = 0
        
        # Make a copy of alpha before it may change
        cdef int alphaCopy = alpha
        
        # Update alpha
        alpha = max(alpha, evaluation)
    
        # Search through all capture moves (and other tactical moves if applicable)
        for move in Cython_Chess.generate_legal_captures(self.pgnBoard,chess.BB_ALL,chess.BB_ALL):
        
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the given move and call the quiescence minimizer using aspiration windows
            self.pgnBoard.push(move)
            # score = self.quiescenceMin(alpha, beta ,quiescenceDepth+1)
            score = self.quiescenceMin(alpha - 750, beta + 750 ,quiescenceDepth+1)

            if alpha < score and score < beta:
                score = self.quiescenceMin(alpha, beta ,quiescenceDepth+1)
            
            # Undo the move and reset the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            # Update alpha
            alpha = max(alpha, score)
            
            # Beta cutoff
            if alpha >= beta:
                return score  
        
        # In the case where the alpha value changes, without a beta cutoff having occurred, this move is likely to be a good one
        if (alpha != alphaCopy):
            return alpha
        # If the alpha value has not changed, the return should be the evaluation acquired in this recursive iteration
        return evaluation          
    
    # Define the quiescence minimizer function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int quiescenceMin(self, int alpha, int beta, int quiescenceDepth):
        
        self.numIterations += 1
        # Get an evaluation and see if the evaluation is close to the alpha and beta or exceeds the quiescence depth
        cdef int evaluation = evaluate_board(self.pgnBoard, self.zobrist)        
        if (quiescenceDepth >= self.quiescenceDepth) or evaluation + 1500 <= alpha or evaluation - 1500 >= beta or evaluation <= -9000000:
            return evaluation
        
        # Define variable to hold the zobrist hash for the current board state
        cdef uint64_t curHash = self.zobrist
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        # Define variables to hold information for zobrist hashing
        cdef int promotion = 0
        cdef bint isCapture
        
        # Make a copy of alpha before it may change
        cdef int betaCopy = beta
        
        # Update beta
        beta = min(beta, evaluation)
    
        # Search through all capture moves (and other tactical moves if applicable)
        for move in Cython_Chess.generate_legal_captures(self.pgnBoard,chess.BB_ALL,chess.BB_ALL):
        # for move in self.non_quiescence_moves(self.pgnBoard):            
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the given move and call the quiescence maximizer using aspiration windows
            self.pgnBoard.push(move)
            # score = self.quiescenceMax(alpha, beta ,quiescenceDepth+1)
            score = self.quiescenceMax(alpha - 750, beta + 750 ,quiescenceDepth+1)

            if alpha < score and score < beta:
                score = self.quiescenceMax(alpha, beta ,quiescenceDepth+1) 
            
            # Undo the move and reset the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash

            # Update beta                                      
            beta = min(beta, score)
            
            # Beta cutoff
            if beta <= alpha:
                return score  # Alpha cutoff
        
        # In the case where the beta value changes, without a beta cutoff having occurred, this move is likely to be a good one
        if (beta != betaCopy):
            return beta
        # If the alpha value has not changed, the return should be the evaluation acquired in this recursive iteration
        return evaluation 
    
    # Define the pre-minimizer function to be used when trtying to reorder moves for full search
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple preMinimizer(self, int curDepth, int depthLimit, int alpha, int beta):
        
        # Define the lowest score with respect to the number of moves played
        cdef int lowestScore = 9999999 - len(self.pgnBoard.move_stack)
        
        # Define variables to hold the current move, score and index
        cdef int score
        cdef object move     
        cdef int count = 0
        
        # Define lists to hold the current scores
        cdef list beta_list = []
        
        # Define variable to hold the zobrist hash for the current board state
        cdef uint64_t curHash = self.zobrist
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        # Define variables to hold information for zobrist hashing
        cdef int promotion = 0
        cdef bint isCapture
        
        # If the depth limit is reached, evaluate the current position
        if curDepth >= depthLimit:            
            self.numIterations += 1            
            return evaluate_board(self.pgnBoard,self.zobrist)
        
        # Acquire list of moves where captures appear first
        cdef list moves_list = list(self.reorder_capture_moves(chess.BB_ALL, self.pgnBoard))
        
        # Variable to hold full length of the list
        cdef int length = len(moves_list)
        
        for move in moves_list:
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the given move and call the maximizer
            self.pgnBoard.push(move)
            score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
            
            # Undo the move and reset the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            # ** Code segment for testing
            
            # if (self.pgnBoard == chess.Board("r1b2r1k/ppq4p/1b1p1p2/n2Bp2p/7N/2P4P/P2Q1PP1/RN3RK1 w - - 2 19")):         
            #     # print(moves_list)
            #     print ("MIN: ",score, move, alpha, beta, lowestScore)
            
            # Append list to keep score of the move
            beta_list.append(score)
            count += 1
            
            # Update lowest score
            if score < lowestScore:
                lowestScore = score

            # Update beta
            beta = min(beta, lowestScore)

            # Beta cutoff
            if beta <= alpha:
                
                # Fill up the scores list to capacity
                for i in range(length - count):
                    beta_list.append(None)
                
                return beta, beta_list, moves_list
        
        # If no moves were available, check if a the opponent is in checkmate
        if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):
            self.numIterations += 1            
            if self.pgnBoard.is_checkmate():
                return 100000000, beta_list, moves_list
            
        return lowestScore, beta_list, moves_list
    
    # Standing position evaluation function
    def ev(self, object board):
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False])
        return evaluate_board(board,self.zobrist)
    
    # Function to order pre order moves for full search
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple reorder_legal_moves(self,int alpha,int beta, depthLimit):
        
        # Define variables to hold current scores, moves. index and highest score
        cdef int score = -99999999
        cdef int highestScore = -99999999
        cdef object move
        cdef int count = 1   
        
        # Define lists to hold scores and moves for reordering purposes
        cdef list moves_list
        cdef list alpha_list = []
        cdef list beta_list = []
        cdef list cur_beta_list = []
        cdef list beta_move_list = []
        cdef list cur_beta_move_list = []
        
        # Define depth for preliminary search             
        cdef int depth = depthLimit - 2
        
        # Define variable to hold zobrist hash of current position
        cdef uint64_t curHash = self.zobrist
        
        # Define variables to hold information for zobrist hashing
        cdef bint isCapture
        cdef int promotion = 0
        
        # Initialize bitmasks for the current board state
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        # Check if this is the first iteration of iterative deepening and if a moves list has already been define
        if (self.alpha_list == []):
            moves_list = list(Cython_Chess.generate_legal_moves(self.pgnBoard,chess.BB_ALL,chess.BB_ALL))
        else:
            moves_list = self.moves_list        
        
        # Check if the move is a promoting move
        if (moves_list[0].promotion):
            promotion = moves_list[0].promotion
        else:
            promotion = 0
        
        # Acquire the zobrist hash for the new position if the given move was made
        isCapture = is_capture(moves_list[0].from_square, moves_list[0].to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(moves_list[0]))
        updateZobristHashForMove(self.zobrist, moves_list[0].from_square, moves_list[0].to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black,promotion)
        
        self.pgnBoard.push(moves_list[0])        
        highestScore, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, beta)
        self.pgnBoard.pop()
        
        self.zobrist = curHash
        
        alpha = max(alpha, highestScore)
        alpha_list.append(highestScore)
        beta_list.append(cur_beta_list)
        beta_move_list.append(cur_beta_move_list)
        # print(0,highestScore,alpha, moves_list[0])  
        for move in moves_list[1:]:
            
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            self.pgnBoard.push(move)
            
            score, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, alpha + 1)
            
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                score, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, beta)
                        
            self.pgnBoard.pop()
            self.zobrist = curHash
            alpha_list.append(score)
            beta_list.append(cur_beta_list)
            beta_move_list.append(cur_beta_move_list)
            # print(count,score,alpha, move)  
            if score > highestScore:
                highestScore = score
            count += 1
            alpha = max(alpha, highestScore)
              
        if (self.alpha_list == []):
            quicksort(alpha_list, moves_list, beta_list, beta_move_list, 0, len(alpha_list) - 1)
            return moves_list,alpha_list,beta_list,beta_move_list
        else:
            # print(self.beta_list, len(self.beta_list))
            quicksort_wrapper(self.alpha_list, moves_list, self.beta_list, self.beta_move_list, alpha_list,beta_list, beta_move_list)    
            # print()
            # print(self.beta_list, len(self.beta_list), len(moves_list), len(self.alpha_list), len(beta_list), len(alpha_list))
            return moves_list,self.alpha_list,self.beta_list,self.beta_move_list
            
        # quicksort(alpha_list, moves_list, beta_list, 0, len(alpha_list) - 1)
        # return moves_list,alpha_list,beta_list    
    def reorder_capture_moves(self, uint64_t mask, object board) -> Iterator[chess.Move]:
        
        cdef object move
                
        for move in Cython_Chess.generate_legal_captures(board,mask,chess.BB_ALL):
            yield move
        for move in Cython_Chess.generate_legal_moves(board,mask,chess.BB_ALL):
            if not is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move)):
                yield move
                
    def non_quiescence_moves(self, object board) -> Iterator[chess.Move]:
        
        cdef object move
                
        for move in Cython_Chess.generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
            if (is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move)) or board.gives_check(move)) or move.promotion:
                yield move

            
cdef void quicksort(list values, list objects, list betas, list betaMoves, int left, int right):
    if left >= right:
        return

    pivot = values[left + (right - left) // 2]
    cdef int i = left
    cdef int j = right
    cdef int temp_value
    cdef object temp_object
    cdef list temp_list
    cdef list temp_moves_list

    while i <= j:
        while values[i] > pivot:
            i += 1
        while values[j] < pivot:
            j -= 1

        if i <= j:
            # Swap values
            temp_value = values[i]
            values[i] = values[j]
            values[j] = temp_value

            # Swap objects
            temp_object = objects[i]
            objects[i] = objects[j]
            objects[j] = temp_object
            
            # Swap lists
            temp_list = betas[i]
            betas[i] = betas[j]
            betas[j] = temp_list
            
            temp_moves_list = betaMoves[i]
            betaMoves[i] = betaMoves[j]
            betaMoves[j] = temp_moves_list 

            i += 1
            j -= 1

    # Recursively sort the partitions
    quicksort(values, objects, betas, betaMoves, left, j)
    quicksort(values, objects, betas, betaMoves, i, right)
    
cdef void quicksort_ascending_wrapper(list values, list objects):
    cdef int count = 0
    for i in values:
        if (i == None):
            break
        count += 1
    cdef list values_sub_list = values[:count]
    cdef list objects_sub_list = objects[:count]
    quicksort_ascending(values_sub_list, objects_sub_list, 0, len(values_sub_list) - 1)

    # Update the original lists
    # values[:count] = values_sub_list
    # objects[:count] = objects_sub_list
    values[:] = values_sub_list + values[count:]
    objects[:] = objects_sub_list + objects[count:]

cdef void quicksort_wrapper(list alphas, list objects, list betas, list betaMoves, list preAlphas, list preBetas, list preBetaMoves):
    cdef int count = 0
    cdef int index = 0
    cdef int maxVal = alphas[0]
    for i in range (len(alphas)):
        
        if (alphas[i] == None):
            break
        if (alphas[i] > maxVal):
            maxVal = alphas[i]
            index = i
        count += 1
    
    cdef int tempAlpha = alphas.pop(index)
    cdef list tempBeta = betas.pop(index)
    cdef object tempObject = objects.pop(index)    
    cdef list tempBetaMoves = betaMoves.pop(index)
    
    alphas.insert(0, tempAlpha)
    betas.insert(0, tempBeta)
    objects.insert(0, tempObject)
    betaMoves.insert(0,tempBetaMoves)
    # print(betas)
    cdef list alphas_sub_list = alphas[1:count] + preAlphas[count:]
    cdef list objects_sub_list = objects[1:]
    cdef list betas_sub_list = betas[1:count] + preBetas[count:]
    cdef list beta_moves_sub_list = betaMoves[1:count] + preBetaMoves[count:]
    
    quicksort(alphas_sub_list, objects_sub_list, betas_sub_list,beta_moves_sub_list, 0, len(alphas_sub_list) - 1)
    
    alphas[1:] = alphas_sub_list    
    betas[1:] = betas_sub_list
    objects[1:] = objects_sub_list
    betaMoves[1:] = beta_moves_sub_list
    # print(betas[0])
        
cdef void quicksort_ascending(list values, list objects, int left, int right):
    if left >= right:
        return

    pivot = values[left + (right - left) // 2]
    cdef int i = left
    cdef int j = right
    cdef int temp_value
    cdef object temp_object
    cdef list temp_list

    while i <= j:
        while values[i] < pivot:
            i += 1
        while values[j] > pivot:
            j -= 1

        if i <= j:
            # Swap values
            temp_value = values[i]
            values[i] = values[j]
            values[j] = temp_value

            # Swap objects
            temp_object = objects[i]
            objects[i] = objects[j]
            objects[j] = temp_object
            
            i += 1
            j -= 1

    # Recursively sort the partitions
    quicksort_ascending(values, objects, left, j)
    quicksort_ascending(values, objects, i, right)    

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef int evaluate_board(object board,uint64_t zobrist):
    
    cdef uint64_t pawns = board.pawns
    cdef uint64_t knights = board.knights
    cdef uint64_t bishops = board.bishops
    cdef uint64_t rooks = board.rooks
    cdef uint64_t queens = board.queens
    cdef uint64_t kings = board.kings
    
    cdef uint64_t occupied_white = board.occupied_co[True]
    cdef uint64_t occupied_black = board.occupied_co[False]
    cdef uint64_t occupied = board.occupied
    # cdef uint64_t curZobrist = generateZobristHash(pawns,knights,bishops,rooks,queens,kings,occupied_white,occupied_black)
    #zobrist = curZobrist
    # if (curZobrist != zobrist):
    #     print(curZobrist, zobrist, board.fen(), board.move_stack)
    cdef int cache_result = accessCache(zobrist)
    
    if (cache_result != 0):
        # print(cache_result,board.fen())
        return cache_result
        
    global prevKings
    global whiteCastledIndex
    global blackCastledIndex
    
    global white_ksc
    global white_qsc
    global black_ksc
    global black_qsc
    
    global values
    
    cdef int total = 0    
    cdef int moveNum = board.ply()
    cdef object target_square
    cdef object target_move
    
    
    cdef int castle_index = -1
    cdef bint horizonMitigation = False

    # Initialize the array in C-style
    
    # if (board.occupied == 10746666234248479586):
    #     print(board)
    # Iterate through all squares on the board and evaluate piece values
    if board.is_checkmate():
        if board.turn:
            total = 9999999 - moveNum      
        else:
            total = -9999999 + moveNum
    # elif board.is_stalemate():
    #     total = -100000000
    else:
        total += placement_and_piece_eval(moveNum, pawns, knights, bishops, rooks, queens, kings, prevKings, occupied_white, occupied_black, occupied)
        

        # if (whiteCastledIndex == -1):        
        #     castle_index = move_index (board, white_ksc, white_qsc)
        #     if (castle_index != -1):
        #         total -= max(1500 - ((castle_index-1) >> 1) * 25 - moveNum * 25, 500)
        # else:
        #     total -= max(1500 - ((whiteCastledIndex-1) >> 1) * 25 - moveNum * 25, 500)
        
        # if (blackCastledIndex == -1):       
        #     castle_index = move_index (board, black_ksc, black_qsc)
        #     if (castle_index != -1):            
        #         total += max(1500 - ((castle_index-1) >> 1) * 25 - moveNum * 25, 500)
        # else:
        #     total += max(1500 - ((blackCastledIndex-1) >> 1) * 25 - moveNum * 25, 500)
        
        
        target_move = board.peek()
        
        if (is_capture(target_move.from_square, target_move.to_square, board.occupied_co[not board.turn], board.is_en_passant(target_move))):
            #if (board.is_capture(target_move)):    
            target_square = target_move.to_square
            #for move in board.generate_legal_captures():
            for move in Cython_Chess.generate_legal_captures(board,chess.BB_ALL,chess.BB_ALL):
                if move.to_square == target_square:
                    if (board.turn):
                        total -= values[board.piece_type_at(target_square)]
                        
                    else:                            
                        total += values[board.piece_type_at(target_square)]
                    horizonMitigation = True
                    break      
    
    #print(board.fen)
    # print(board.move_stack)
    # print(total)
    # print()
    
    
    # if (board == chess.Board("r1b2q1r/pp2kp2/2p1pNpp/8/2BQ1P2/bR2P3/P5PP/5RK1 b - - 7 18")):
    #     print(total)
    #     print(board.move_stack)
    # if (total == 2780):
    #     print(board.fen)
    #     print(total)
    #     print(board.move_stack)
    
    # if (total == 3825):
    #     print(board.fen(),total)
    if not(horizonMitigation):
        addToCache(zobrist, total)
    # if (board == chess.Board("5rk1/p1p4p/2p5/3p1pB1/8/1r5P/P1P1BK2/RN6 w - - 0 20")):                
    #     print (total, board.move_stack)          
    return total

cdef int move_index(object board, object move1, object move2):
    
    cdef int index
    cdef object move
    
    for index, move in enumerate(board.move_stack):
        if move == move1 or move == move2:
            return index
    return -1


