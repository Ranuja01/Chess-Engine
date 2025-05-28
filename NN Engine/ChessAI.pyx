# -*- coding: utf-8 -*-
"""
@author: Ranuja Pinnaduwage

This file holds the main chess engine and its components

"""
import chess  # Use regular import for Python libraries
cimport cython  # Import Cython-specific utilities
from cython cimport boundscheck, wraparound
import json
import marshal
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
from timeit import default_timer as timer
from functools import lru_cache
import Cython_Chess
import multiprocessing
import time
import itertools
from typing import Iterator
import tensorflow as tf
from tensorflow.keras.models import Model
from operator import itemgetter

# Import data structures from the c++ standard library
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t
    
# cdef extern from "nnue.h":
#     int evaluate_position(const uint64_t* bitboards)
#     void init_session(const char* model_path)
#     void load_all_weights()
#     int run_inference(const uint64_t* bitboards)
#     void load_model()
#     int run_inference_quantized(const uint64_t* bitboards)

# Import functions from c++ file
cdef extern from "cpp_bitboard.h":
    bool get_horizon_mitigation_flag()
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
    int placement_and_piece_eval(int moveNum, bint turn, uint8_t lastMovedToSquare, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t prevKings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    void initializeZobrist()
    uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, bint whiteToMove);
    void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bint isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion)
    int accessCache(uint64_t key)
    void addToCache(uint64_t key,int value)
    string accessOpponentMoveGenCache(uint64_t key);
    void addToOpponentMoveGenCache(uint64_t key,char* data, int length);
    string accessCurPlayerMoveGenCache(uint64_t key);
    void addToCurPlayerMoveGenCache(uint64_t key,char* data, int length);
    int printCacheStats()
    int getCacheStats()
    int printOpponentMoveGenCacheStats();
    int printCurPlayerMoveGenCacheStats();
    void evictOldEntries(int numToEvict)
    void evictOpponentMoveGenEntries(int numToEvict)
    void evictCurPlayerMoveGenEntries(int numToEvict)
    void printLayers()
    
    bint is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bint turn)
    
    bint is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bint turn)
    
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

cdef const char* nnue_model_path = b"/mnt/c/Users/Kumodth/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/NNUE_treesearch_21_to_41.onnx"

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
    
    cdef bint is_training
    
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
        self.move_times[3] = 5.0
        self.move_times[4] = 5.0
        self.move_times[5] = 5.5
        self.time_limit = 60
        self.quiescenceDepth = 6
        
        self.is_training = False
        
        for i in range(6,26):
            self.move_times[i] = 2.5
        
        # Initialize attack tables for move generation
        initialize_attack_tables()
        Cython_Chess.inititalize()
        # init_session(nnue_model_path)
        # load_all_weights()
        # load_model()
        
        # Initialize zobrist tables for hashing
        initializeZobrist()
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False], board.turn)    
    
    # Function to set global variable for white castling index
    def setWhiteCastledIndex(self, index):
        global whiteCastledIndex
        whiteCastledIndex = index
    
    # Function to set global variable for black castling index
    def setBlackCastledIndex(self, index):
        global blackCastledIndex
        blackCastledIndex = index            

    
    def create_test_data(self, object board):
        self.is_training = True
        self.pgnBoard = board
        
        # Initialize the lists required for iterative deepening
        self.moves_list = []
        self.alpha_list = []
        self.beta_list = []
        self.beta_move_list = []
        self.numIterations = 0
        
        cdef int cacheSize = getCacheStats()
        
        if (self.pgnBoard.ply() < 30):
            if (cacheSize > 16000000):
                evictOldEntries(cacheSize - 16000000)                
        elif(self.pgnBoard.ply() < 50):
            if (cacheSize > 32000000):
                evictOldEntries(cacheSize - 32000000)
        elif(self.pgnBoard.ply() < 75):
            if (cacheSize > 64000000):
                evictOldEntries(cacheSize - 64000000)
        else:
            if (cacheSize > 128000000):
                evictOldEntries(cacheSize - 128000000)
        
        cdef int a, b, c, d,promo,val
        cdef object move
        
        self.zobrist = generateZobristHash(self.pgnBoard.pawns,self.pgnBoard.knights,self.pgnBoard.bishops,self.pgnBoard.rooks,self.pgnBoard.queens,self.pgnBoard.kings,self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False], self.pgnBoard.turn)
        
        result = self.alphaBeta(curDepth=0, depthLimit=3, t0 = timer())
        moves_list ,_,_,_= self.reorder_legal_moves(-9999998,9999999, 3)
        length = len(self.alpha_list)
        
        
        self.is_training = False
        return (moves_list [:length], self.alpha_list)
        
    # Function to wrap the 
    def alphaBetaWrapper(self):
        
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
        
        self.zobrist = generateZobristHash(self.pgnBoard.pawns,self.pgnBoard.knights,self.pgnBoard.bishops,self.pgnBoard.rooks,self.pgnBoard.queens,self.pgnBoard.kings,self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False], self.pgnBoard.turn)
        
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
            result = self.opening_book()
                                      
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
        result = self.alphaBeta(curDepth=0, depthLimit=3, t0 = timer())
        val = result.score
        t1 = timer()
        dif = t1 - t0
        new_depth = 4
        print(dif)
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
    cdef MoveData opening_book(self):
        
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
                            
    
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef MoveData alphaBeta(self, int curDepth, int depthLimit, double t0):
        
        """
        Define the alphaBeta function

        Parameters:
        - curDepth: The starting depth
        - depthLimit: The maximum depth limit
        - t0: The starting time to keep track of the cutoff time

        Returns:
        - A MoveData struct instance with the bestmove's coordinates, scores and potential promotions
        """
        
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
            razorThreshold = max (int(750 * .75** (depthLimit - 4)), 200)
        else:
            razorThreshold = max (int(300 * .75** (depthLimit - 4)), 50)
        
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
        score = self.minimizer(curDepth + 1, depthLimit, alpha, beta, beta_list[0], beta_move_list[0], isCapture)
        
        # Check if the position is repeating after the move
        if (self.pgnBoard.is_repetition(2)):
            repetitionFlag = True
            repetitionMove = moves_list[0]
            repetitionScore = score
            score = -100000000
        
        # Check if the move causes a stalemate
        if (is_stalemate(self.pgnBoard.clean_castling_rights(), occupied, occupied_white, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.occupied_co[self.pgnBoard.turn],
                              pawns, knights, bishops, rooks, queens, kings, (-1 if self.pgnBoard.ep_square is None else self.pgnBoard.ep_square), self.pgnBoard.turn)):            
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
        
        if (alpha - alpha_list[0] > razorThreshold):
            razorThreshold += alpha - alpha_list[0]
        
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
                depthUsage = depthLimit
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
            score = self.minimizer(curDepth + 1, depthUsage, alpha, alpha+1, beta_list[count], beta_move_list[count], isCapture)           
                        
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                
                # Ensure there is a score for the given index
                if (not(alpha_list[count] == None)):
                    
                    # Pop the lists for the re-search
                    self.beta_list.pop()
                    self.beta_move_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, beta, beta_list[count],beta_move_list[count], isCapture)
                else:
                    
                    # Pop the lists for the re-search
                    self.beta_list.pop()
                    self.beta_move_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, beta, beta_list[count],beta_move_list[count], isCapture)                
            
            # Check if the position is repeating after the move
            if (self.pgnBoard.is_repetition(2)):
                repetitionFlag = True
                repetitionMove = move
                repetitionScore = score
                score = -100000000
            
            # Check if the move causes a stalemate
            if (is_stalemate(self.pgnBoard.clean_castling_rights(), occupied, occupied_white, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.occupied_co[self.pgnBoard.turn],
                                  pawns, knights, bishops, rooks, queens, kings, (-1 if self.pgnBoard.ep_square is None else self.pgnBoard.ep_square), self.pgnBoard.turn)):            
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
            # print(repetitionFlag, repetitionMove, repetitionScore, alpha)
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
        
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall    
    @cython.inline
    cdef int maximizer(self, int curDepth, int depthLimit, int alpha, int beta):
        
        """
        Define the maximizer function

        Parameters:
        - curDepth: The current depth
        - depthLimit: The maximum depth limit
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move

        Returns:
        - The best score found by the maximizer
        """
        
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
        
        for move in Cython_Chess.generate_ordered_moves(self.pgnBoard, chess.BB_ALL, chess.BB_ALL):
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the given move and call the minimizer
            
            try:
                # code that might raise an exception
                self.pgnBoard.push(move)
            except Exception as e:
                print(f"Something went wrong: {e}")
                print(self.pgnBoard)
                print(self.pgnBoard.move_stack)
                # self.pgnBoard.pop()
                for i in Cython_Chess.generate_legal_moves(self.pgnBoard, chess.BB_ALL, chess.BB_ALL):
                    print(i)
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta, [],[], False)
            
            # Undo the move and restore the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            # ** Code for testing purposes **
            
            # if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1bpn2/2p1P3/2PP1P2/2NB4/P1Q2P1P/R1B2RK1 b - - 0 14")):                
            #     print ("MAX: ",score, move)          
                
            # if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1Ppn2/8/2Pp1P2/2NB4/P1Q2P1P/R1B2RK1 b - - 0 15")):            
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

    
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int minimizer(self, int curDepth, int depthLimit, int alpha, int beta, list beta_list_og, list beta_moves_list, bint prevCapture):
        
        """
        Define the minimizer function

        Parameters:
        - curDepth: The current depth
        - depthLimit: The maximum depth limit
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move
        - beta_list_og: A list of preliminary scores
        - beta_moves_list: A list of pre ordered moves corresponding to the above list
        - prevCapture: A boolean storing whether the last move was a capture

        Returns:
        - The best score found by the minimizer
        """
        
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
        # cdef int razorThreshold
        # if (depthLimit == 3):
        #     razorThreshold = max (int(2000 * .75** (depthLimit - 4)), 200) 
        # else:
        #     razorThreshold = max (int(1500 * .75** (depthLimit - 4)), 50)
            
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
                # if (not(beta_list[count] == None)):
                #     if (beta_list[count] - beta > razorThreshold):
                #         count+=1
                #         cur_beta_list.append(None)
                #         # if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1bpn2/2p5/2PPPP2/2NB4/P1Q2P1P/R1B2RK1 w - - 0 14")):         
                #         #     # print(moves_list)
                #         #     print("REMOVED: ", move, beta_list[count], alpha, beta, count)
                #         # if (prevCapture):
                #         #     continue
                #         # else:
                #         #     break
                #         continue
                
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
                
                # Store the move scores           
                cur_beta_list.append(score)
                    
                # Find the lowest score and beta
                if score < lowestScore:
                    lowestScore = score

                beta = min(beta, lowestScore)
                
                # if (not(beta_list[count] == None)):
                #     if (beta_list[count] - beta > razorThreshold):
                #         razorThreshold += beta_list[count] - beta
                
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
                    return lowestScore
                else:
                    return min(beta,lowestScore)
           
            # Fill up the remaining list to capacity
            for i in range(length - count):
                cur_beta_list.append(None)
            self.beta_list.append(cur_beta_list)    
        else: # If not the second recursive depth, take advantage of the yielding feature to increase speed
            for move in Cython_Chess.generate_ordered_moves(self.pgnBoard, chess.BB_ALL, chess.BB_ALL):
                
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
                
                # if (self.pgnBoard == chess.Board("r4rk1/p2nqppp/1p1bpn2/4P3/2Pp1P2/2NB4/P1Q2P1P/R1B2RK1 w - - 0 15")):
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
                    return lowestScore
                else:
                    return min(beta,lowestScore)
               
        return lowestScore
        
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int quiescenceMax(self, int alpha, int beta, int quiescenceDepth):
        
        """
        Define the quiescence maximizer function

        Parameters:        
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move
        - quiescenceDepth: The max quiescence depth

        Returns:
        - The best score found by the quiescence maximizer
        """
        
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

    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef int quiescenceMin(self, int alpha, int beta, int quiescenceDepth):
        
        """
        Define the quiescence minimizer function

        Parameters:        
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move
        - quiescenceDepth: The max quiescence depth

        Returns:
        - The best score found by the quiescence minimizer
        """
        
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
    
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple preMinimizer(self, int curDepth, int depthLimit, int alpha, int beta):
        
        """
        Define the pre-minimizer function to be used when reordering moves for full search

        Parameters:
        - curDepth: The current depth
        - depthLimit: The maximum depth limit
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move

        Returns:
        - A tuple which contains the best score, the list of scores for each move as well as the order of the moves
        """
        
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
        cdef list moves_list = list(Cython_Chess.generate_ordered_moves(self.pgnBoard, chess.BB_ALL, chess.BB_ALL))
        
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
                return lowestScore, beta_list, moves_list
            
        return lowestScore, beta_list, moves_list
    
    # Standing position evaluation function
    def ev(self, object board):
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False], self.pgnBoard.turn)
        return evaluate_board(board,self.zobrist)

    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple reorder_legal_moves(self,int alpha,int beta, depthLimit):
        
        """
        Function to order pre order moves for full search

        Parameters:
        - alpha: The alpha value representing the score of the best maximizer move
        - beta: The beta value representing the score of the best minimizer move
        - depthLimit: The maximum depth limit

        Returns:
        - A tuple which contains the list of scores for each move, as well as the moves list for both the first and second recursive iteration
        """
        
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
        cdef int depth = depthLimit - 1
        
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
        
        # Check if this is the first iteration of iterative deepening and if a moves list has already been defined
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
        
        # Push the move and call the preliminary minimizer, then undo the move and reset the zobrist hash
        self.pgnBoard.push(moves_list[0])        
        highestScore, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, beta)
        self.pgnBoard.pop()        
        self.zobrist = curHash
        
        # Update alpha
        alpha = max(alpha, highestScore)
        
        # Increment the appropriate moves and scores list
        alpha_list.append(highestScore)
        beta_list.append(cur_beta_list)
        beta_move_list.append(cur_beta_move_list)
        
        # print(0,highestScore,alpha, moves_list[0])  
        for move in moves_list[1:]:
            
            # Check if the move is a promoting move
            if (move.promotion):
                promotion = move.promotion
            else:
                promotion = 0
            
            # Acquire the zobrist hash for the new position if the given move was made
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black, promotion)
            
            # Push the move and call the preliminary minimizer using pvs
            self.pgnBoard.push(move)            
            score, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, alpha + 1)
            
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                score, cur_beta_list, cur_beta_move_list = self.preMinimizer(1, depth, alpha, beta)
            
            # Undo the move and reset the zobrist hash
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            # Increment the appropriate moves and scores list
            alpha_list.append(score)
            beta_list.append(cur_beta_list)
            beta_move_list.append(cur_beta_move_list)
            
            # print(count,score,alpha, move)
            
            # Update the highest score
            if score > highestScore:
                highestScore = score
            count += 1
            
            # Update alpha
            alpha = max(alpha, highestScore)
        
        # Check if this is the first iteration of iterative deepening
        if (self.alpha_list == []):
            # Call quicksort to sort the moves and their associative scores
            quicksort(alpha_list, moves_list, beta_list, beta_move_list, 0, len(alpha_list) - 1)
            return moves_list,alpha_list,beta_list,beta_move_list
        else:
            # Call the quicksort wrapper to first reorder previous iteration moves before the preliminary search moves
            quicksort_wrapper(self.alpha_list, moves_list, self.beta_list, self.beta_move_list, alpha_list,beta_list, beta_move_list)    
            # print()
            # print(self.beta_list, len(self.beta_list), len(moves_list), len(self.alpha_list), len(beta_list), len(alpha_list))
            return moves_list,self.alpha_list,self.beta_list,self.beta_move_list
     
    def reorder_capture_moves(self, uint64_t mask, object board) -> Iterator[chess.Move]:
        
        """
        Function to order capture moves before other moves

        Parameters:
        - mask: The starting location
        - board: The current board state

        Yields:
        - Legal chess moves
        """
        
        cdef object move
        
        # Generate and yield capture moves first 
        for move in Cython_Chess.generate_legal_captures(board,mask,chess.BB_ALL):
            yield move
            
        # Generate all legal moves and yield moves that aren't captures
        for move in Cython_Chess.generate_legal_moves(board,mask,chess.BB_ALL):
            if not is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move)):
                yield move
   
    def reorder_capture_moves2(self, uint64_t mask, object board) -> Iterator[chess.Move]:
        """
        Function to order capture moves before other moves
    
        Parameters:
        - mask: The starting location
        - board: The current board state
    
        Yields:
        - Legal chess moves
        """
        cdef object move
        cdef set capture_moves = set(Cython_Chess.generate_legal_captures(board, mask, chess.BB_ALL))
    
        # Yield all captures first
        yield from capture_moves
    
        # Cache occupied squares for efficiency
        cdef uint64_t enemy_pieces = board.occupied_co[not board.turn]
    
        # Yield non-capture moves
        for move in Cython_Chess.generate_legal_moves(board, mask, chess.BB_ALL):
            if move not in capture_moves:
                yield move
 
   
    # Function to return moves that are either captures, checks or promotions
    def non_quiescence_moves(self, object board) -> Iterator[chess.Move]:
        
        cdef object move
                
        for move in Cython_Chess.generate_legal_moves(board,chess.BB_ALL,chess.BB_ALL):
            if (is_capture(move.from_square, move.to_square, board.occupied_co[not board.turn], board.is_en_passant(move)) or board.gives_check(move)) or move.promotion:
                yield move
        
cdef void quicksort(list values, list objects, list betas, list betaMoves, int left, int right):
    
    """
    Function to sort moves lists and scores list by the alpha list 

    Parameters:
    - values: The list of alpha values
    - objects: The list of moves
    - betas: The list of lists containing scores for the second recursive iteration
    - betaMoves: The list of lists containing moves for the second recursive iteration
    - left: The left index
    - right: The right index

    """
    
    # Check if the list bounds have overlaped
    if left >= right:
        return

    # Define the pivot as the middle value
    pivot = values[left + (right - left) // 2]
    
    # Define starting point 
    cdef int i = left
    cdef int j = right
    
    # Define variables to temporarily hold swap indices
    cdef int temp_value
    cdef object temp_object
    cdef list temp_list
    cdef list temp_moves_list

    # Loop while the sublist bounds don't cross each other
    while i <= j:
        
        # Keep shortening the list until two values are found above and below the pivot
        # This is checking for when something left of the pivot is less than the pivot and vice versa
        while values[i] > pivot:
            i += 1
        while values[j] < pivot:
            j -= 1

        if i <= j:
            # Swap alpha values
            temp_value = values[i]
            values[i] = values[j]
            values[j] = temp_value

            # Swap first iteration moves
            temp_object = objects[i]
            objects[i] = objects[j]
            objects[j] = temp_object
            
            # Swap second iteration scores lists
            temp_list = betas[i]
            betas[i] = betas[j]
            betas[j] = temp_list
            
            # Swap second iteration moves lists
            temp_moves_list = betaMoves[i]
            betaMoves[i] = betaMoves[j]
            betaMoves[j] = temp_moves_list 

            i += 1
            j -= 1

    # Recursively sort the partitions
    quicksort(values, objects, betas, betaMoves, left, j)
    quicksort(values, objects, betas, betaMoves, i, right)

cdef void quicksort_ascending_wrapper(list values, list objects):
    
    """
    Function to wrap the ascending quicksort function

    Parameters:
    - values: The list of alpha values
    - objects: The list of moves

    """
    
    # Section of code to find the index where the given lists start becoming NULL
    cdef int count = 0
    for i in values:
        if (i == None):
            break
        count += 1
    
    # Sort the non NULL items
    cdef list values_sub_list = values[:count]
    cdef list objects_sub_list = objects[:count]
    quicksort_ascending(values_sub_list, objects_sub_list, 0, len(values_sub_list) - 1)
    # sort_by_alpha(values_sub_list, objects_sub_list)

    # Update the original lists
    values[:] = values_sub_list + values[count:]
    objects[:] = objects_sub_list + objects[count:]

cdef void quicksort_wrapper(list alphas, list objects, list betas, list betaMoves, list preAlphas, list preBetas, list preBetaMoves):
    
    """
    Function to wrap the descending quicksort function

    Parameters:
    - alphas: The list of alpha values from the previous full search
    - objects: The list of moves from the previous full search
    - betas: The list of lists containing scores for the second recursive iteration from the previous full search
    - betaMoves: The list of lists containing moves for the second recursive iteration from the previous full search
    - preAlphas: The list of alpha values from the preliminary search
    - preBetas: The list of lists containing scores for the second recursive iteration from the preliminary search
    - preBetas: The list of lists containing moves for the second recursive iteration from the preliminary search

    """
    
    # Define variables to hold index when the lists become null, the max index, and the max value
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
    
    # Move the max entry to the front without pop/insert overhead
    alphas[:] = [alphas[index]] + alphas[:index] + alphas[index+1:]
    objects[:] = [objects[index]] + objects[:index] + objects[index+1:]
    betas[:] = [betas[index]] + betas[:index] + betas[index+1:]
    betaMoves[:] = [betaMoves[index]] + betaMoves[:index] + betaMoves[index+1:]

    
    # Create a list that consists of the non null values excluding the first one
    # Complete the lists with the preliminary search values
    cdef list alphas_sub_list = alphas[1:count] + preAlphas[count:]
    cdef list objects_sub_list = objects[1:]
    cdef list betas_sub_list = betas[1:count] + preBetas[count:]
    cdef list beta_moves_sub_list = betaMoves[1:count] + preBetaMoves[count:]
    
    # Call quicksort to sort the sub lists
    quicksort(alphas_sub_list, objects_sub_list, betas_sub_list,beta_moves_sub_list, 0, len(alphas_sub_list) - 1)
    
    # Add the sublists to the first indices
    alphas[1:] = alphas_sub_list    
    betas[1:] = betas_sub_list
    objects[1:] = objects_sub_list
    betaMoves[1:] = beta_moves_sub_list
    
cdef void quicksort_ascending(list values, list objects, int left, int right):
    
    """
    Function to sort moves lists and scores list by the alpha list in ascending order    

    Parameters:
    - values: The list of alpha values
    - objects: The list of moves
    - left: The left index
    - right: The right index

    """
    
    # Check if the list bounds have overlaped
    if left >= right:
        return

    # Define the pivot as the middle value
    pivot = values[left + (right - left) // 2]
    
    # Define variables to temporarily hold swap indices
    cdef int i = left
    cdef int j = right
    cdef int temp_value
    cdef object temp_object
    cdef list temp_list

    # Loop while the sublist bounds don't cross each other
    while i <= j:
        
        # Keep shortening the list until two values are found below and above the pivot
        # This is checking for when something left of the pivot is greater than the pivot and vice versa
        while values[i] < pivot:
            i += 1
        while values[j] > pivot:
            j -= 1

        if i <= j:
            # Swap alpha values
            temp_value = values[i]
            values[i] = values[j]
            values[j] = temp_value

            # Swap moves
            temp_object = objects[i]
            objects[i] = objects[j]
            objects[j] = temp_object
            
            i += 1
            j -= 1

    # Recursively sort the partitions
    quicksort_ascending(values, objects, left, j)
    quicksort_ascending(values, objects, i, right)    



def sort_by_alpha(list values, list objects):
    """
    Sorts the two lists 'values' and 'objects' in ascending order based on 'values'.

    Parameters:
      values  - List of numeric alpha values.
      objects - List of corresponding objects/moves.
    """
    # Combine the two lists into a list of tuples (value, object)
    combined = zip(values, objects)
    
    # Use sorted with itemgetter(0) to sort the tuples by the first element (the alpha value)
    sorted_combined = sorted(combined, key=itemgetter(0))
    
    # Unzip the sorted tuples back into separate lists
    # Note: zip(*...) returns tuples, so we wrap them with list() to ensure we have lists
    sorted_values, sorted_objects = list(zip(*sorted_combined))
    
    # Update the original lists in-place
    values[:] = sorted_values
    objects[:] = sorted_objects

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef evaluate_board1(object board,uint64_t zobrist):
        
    cdef uint64_t pawns = board.pawns
    cdef uint64_t knights = board.knights
    cdef uint64_t bishops = board.bishops
    cdef uint64_t rooks = board.rooks
    cdef uint64_t queens = board.queens
    cdef uint64_t kings = board.kings
    
    cdef uint64_t occupied_white = board.occupied_co[True]
    cdef uint64_t occupied_black = board.occupied_co[False]
    
    cdef uint64_t[12] bitboards
    cdef int total = 0    
    cdef int moveNum = board.ply()

    # Access the cache
    cdef int cache_result = accessCache(zobrist)
    
    # Check if the cache has the given position
    if (cache_result != 0):        
        return cache_result

    if board.is_checkmate():
        if board.turn:
            total = 9999999 - moveNum      
        else:
            total = -9999999 + moveNum
    
    else:
         bitboards = [
            pawns & occupied_white,    # White pawns
            knights & occupied_white,  # White knights
            bishops & occupied_white,  # White bishops
            rooks & occupied_white,    # White rooks
            queens & occupied_white,   # White queens
            kings & occupied_white,    # White kings
            pawns & occupied_black,    # Black pawns
            knights & occupied_black,  # Black knights
            bishops & occupied_black,  # Black bishops
            rooks & occupied_black,    # Black rooks
            queens & occupied_black,   # Black queens
            kings & occupied_black     # Black kings
        ]
          
    
    # total = evaluate_position(bitboards)
    # total = run_inference(bitboards)
    # total = run_inference_quantized(bitboards)
    
    addToCache(zobrist, total)
           
    return total

@boundscheck(False)
@wraparound(False)
@cython.exceptval(check=False)
@cython.nonecheck(False)
@cython.ccall
cdef int evaluate_board(object board,uint64_t zobrist):
    
    """
    Evaluation function

    Parameters:
    - board: The current board state
    - zobrist: The zobrist hash for the current position
  
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
    
    # Access the cache
    cdef int cache_result = accessCache(zobrist)
    
    # Check if the cache has the given position
    if (cache_result != 0):        
        return cache_result
    
    # Acquire global variables
    global prevKings
    # global whiteCastledIndex
    # global blackCastledIndex
    
    # global white_ksc
    # global white_qsc
    # global black_ksc
    # global black_qsc
    
    global values
    
    # Define variable to hold total evaluation and move number
    cdef int total = 0    
    cdef int moveNum = board.ply()
    
    # Define variables for the target move and square for the previously made move
    cdef object target_square
    cdef object target_move
        
    # cdef int castle_index = -1
    cdef bint horizonMitigation = False

    # Check if the board state is checkmate
    if is_checkmate(board.clean_castling_rights(), occupied, occupied_white, board.occupied_co[not board.turn], board.occupied_co[board.turn],
                          pawns, knights, bishops, rooks, queens, kings, (-1 if board.ep_square is None else board.ep_square), board.turn):
        if board.turn:
            total = 9999999 - moveNum      
        else:
            total = -9999999 + moveNum
    
    else:
        # Call the c++ function 
        total += placement_and_piece_eval(moveNum, board.turn, board.peek().to_square, pawns, knights, bishops, rooks, queens, kings, prevKings, occupied_white, occupied_black, occupied)
        horizonMitigation = get_horizon_mitigation_flag()        
        # if (total == -1396):
        #     print(board)
        #     print(total)
        #     print("Horizon Mitigation: ", horizonMitigation)
        #     print (board.move_stack[-8:])
        # print (board)
        # print()
        # printLayers()
        # print()
        
        # Additional castling logic        

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
        
        
        # ** Code segment to see if a bad capture was made ** 
        
        # # Get the previous move made
        # target_move = board.peek()
        
        # # Check if the move was a capture
        # if (is_capture(target_move.from_square, target_move.to_square, board.occupied_co[not board.turn], board.is_en_passant(target_move))):
            
        #     # Acquire the square that the move was made to
        #     target_square = target_move.to_square
            
        #     # # Check if there is a legal capture to the same square
        #     # for move in Cython_Chess.generate_legal_captures(board,chess.BB_ALL,chess.BB_ALL):
                
        #     #     # If such a capture exists, assume that the last capture was a bad one and assume you will lose that piece
        #     #     if move.to_square == target_square:
        #     #         if (board.turn):
        #     #             total -= values[board.piece_type_at(target_square)]
                        
        #     #         else:                            
        #     #             total += values[board.piece_type_at(target_square)]
        #     #         horizonMitigation = True
        #     #         break
                
        #     # Check if there is a legal capture to the same square
        #     for move in Cython_Chess.generate_legal_captures(board,chess.BB_ALL,chess.BB_SQUARES[target_square]):
                
        #         # If such a capture exists, assume that the last capture was a bad one and assume you will lose that piece
                
        #         if (board.turn):
        #             total -= values[board.piece_type_at(target_square)]
                    
        #         else:                            
        #             total += values[board.piece_type_at(target_square)]
                    
        #         # Set the flag for a position where move order matters  
        #         horizonMitigation = True
        #         break
            
    # Avoid the adding this evaluation to the cache if the move order matters
    if not(horizonMitigation):
        addToCache(zobrist, total)
           
    return total

cdef int move_index(object board, object move1, object move2):
    
    cdef int index
    cdef object move
    
    for index, move in enumerate(board.move_stack):
        if move == move1 or move == move2:
            return index
    return -1
