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
from joblib import Parallel, delayed
from numba import njit
from timeit import default_timer as timer
from functools import lru_cache
import Cython_Chess

cimport chess as c_chess

import tensorflow as tf
#cimport tensorflow as ctf
from tensorflow.keras.models import Model

# Define data types for numpy arrays
ctypedef cnp.float32_t DTYPE_FLOAT
ctypedef cnp.int8_t DTYPE_INT

from libcpp.vector cimport vector
cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t
    

cdef extern from "cpp_bitboard.h":
    uint8_t scan_reversed_size(uint64_t bb)
    void scan_reversed(uint64_t bb, vector[uint8_t] &result)
    void scan_forward(uint64_t bb, vector[uint8_t] &result)
    int getPPIncrement(int square, bint colour, uint64_t opposingPawnMask, int ppIncrement, int x)
    uint64_t attacks_mask(bint colour, uint64_t occupied, uint8_t square, uint8_t pieceType)
    uint64_t attackersMask(bint color, uint8_t square, uint64_t occupied, uint64_t queens_and_rooks, uint64_t queens_and_bishops, uint64_t kings, uint64_t knights, uint64_t pawns, uint64_t occupied_co)
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
    void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bint isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask)
    int accessCache(uint64_t key)
    void addToCache(uint64_t key,int value)
    int printCacheStats()
    void evictOldEntries(int numToEvict)
    
    
cdef struct MoveData:
    int a
    int b
    int c
    int d
    int promotion
    int score
    
cdef struct PredictionInfo:
    int x
    int y
    int w
    int z

cdef uint64_t prevKings
cdef int blackCastledIndex = -1
cdef int whiteCastledIndex = -1

@cython.cclass
cdef class ChessAI:
    
    cdef object blackModel
    cdef object whiteModel
    cdef object pgnBoard
    cdef list boardPieces
    cdef int numMove
    cdef int numIterations
    cdef dict move_times
    cdef uint64_t zobrist
    cdef list moves_list
    cdef list alpha_list
    cdef list beta_list

    def __cinit__(self, object black_model, object white_model, object board):
        self.blackModel = black_model
        self.whiteModel = white_model
        self.pgnBoard = board
        self.numMove = 0
        self.numIterations = 0
        self.move_times = {}
        self.moves_list = []
        self.alpha_list = []
        self.beta_list = []
        self.move_times[4] = 5.0
        self.move_times[5] = 5.5
        
        for i in range(6,26):
            self.move_times[i] = 2.0
        
        # Call the initialization function once at module load
        #initialize_layers(self.pgnBoard)
        initialize_attack_tables()
        Cython_Chess.inititalize()
        initializeZobrist()
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False])
        #setAttackingLayer(self.pgnBoard.occupied_co[True], self.pgnBoard.occupied_co[False], self.pgnBoard.kings,5)
    
    def setWhiteCastledIndex(self, index):
        global whiteCastledIndex
        whiteCastledIndex = index
        
    def setBlackCastledIndex(self, index):
        global blackCastledIndex
        blackCastledIndex = index
            
    def get_move_cache(self):
        return self.move_cache

    def alphaBetaWrapper(self, int curDepth, int depthLimit):
        #initialize_layers(self.pgnBoard)
        t0= timer()
        global prevKings
        global whiteCastledIndex
        global blackCastledIndex
        
        self.moves_list = []
        self.alpha_list = []
        self.beta_list = []
        self.numIterations = 0
        
        print(whiteCastledIndex,blackCastledIndex)
        cdef int cacheSize = printCacheStats()
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
        printCacheStats()
        
        prevKings = self.pgnBoard.kings
        setAttackingLayer(self.pgnBoard.occupied_co[True], self.pgnBoard.occupied_co[False], self.pgnBoard.kings,5)  
        #Cython_Chess.test4 (self.pgnBoard,5)
        
        cdef MoveData result
        cdef int a, b, c, d,promo,val
        cdef object move
        self.zobrist = generateZobristHash(self.pgnBoard.pawns,self.pgnBoard.knights,self.pgnBoard.bishops,self.pgnBoard.rooks,self.pgnBoard.queens,self.pgnBoard.kings,self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False])
        
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
        if (len(self.pgnBoard.move_stack) < 30):
            result = self.opening_book(curDepth, depthLimit)
                                      
            a = result.a
            b = result.b
            c = result.c
            d = result.d
            promo = result.promotion
            val = result.score
                        
            t1 = timer()
            if not((a,b,c,d) == (-1,-1,-1,-1)):  
                print(a,b,c,d)
                print()
                print("Evaluation: Book Move")
                print ("Time Taken: ", t1 - t0)
                print("Move: ", self.pgnBoard.ply())
                print()
                
                x = chr(a + 96)
                y = str(b)
                i = chr(c + 96)
                j = str(d)
                if (promo == -1):
                    move = chess.Move.from_uci(x+y+i+j)
                    
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
        result = self.alphaBeta(curDepth=0, depthLimit=4)
        val = result.score
        t1 = timer()
        dif = t1 - t0
        new_depth = 5
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
            result = self.alphaBeta(curDepth=0, depthLimit=new_depth)
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
            
            x = chr(a + 96)
            y = str(b)
            i = chr(c + 96)
            j = str(d)
            if (promo == -1):
                move = chess.Move.from_uci(x+y+i+j)
                
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
        #with chess.polyglot.open_reader("../M11.2.bin") as reader:
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
        cdef int score
        cdef object move
        #cdef int highestScore = -99999999
        #cdef int a, b, c, d
        
        cdef MoveData bestMove
        bestMove.a = -1
        bestMove.b = -1
        bestMove.c = -1
        bestMove.d = -1
        bestMove.promotion = -1
        bestMove.score = -99999999
    
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied    
    
        cdef str cur
        cdef list moves_list
        cdef list alpha_list
        cdef list beta_list
        cdef uint64_t curHash = self.zobrist
        moves_list, alpha_list, beta_list = self.reorder_legal_moves(alpha,beta, depthLimit)
        
        cdef int razorThreshold
        if (self.alpha_list == []):
            razorThreshold = max (int(750 * .75** (depthLimit - 6)), 200)
        else:
            razorThreshold = max (int(300 * .75** (depthLimit - 6)), 50)
        # razorThreshold = max (int(750 * .75** (depthLimit - 6)), 200)
        self.alpha_list = []    
        self.beta_list = []
        
        cdef int num_legal_moves = len(moves_list)
        cdef int best_move_index = -1
        cdef int count = 1
        cdef int depthUsage = 0                
        cdef bint isCapture
        
        if (depthLimit >= 5):
            print("Num Moves: ", num_legal_moves)        
                
        isCapture = is_capture(moves_list[0].from_square, moves_list[0].to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(moves_list[0]))
        updateZobristHashForMove(self.zobrist, moves_list[0].from_square, moves_list[0].to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
        
        self.pgnBoard.push(moves_list[0])
        score = self.minimizer(curDepth + 1, depthLimit, alpha, alpha_list[0], beta_list[0])
        # if alpha < score and score < beta:
        #     score = self.minimizer(curDepth + 1, depthLimit, alpha, beta)
        if (self.pgnBoard.is_repetition(2) or self.pgnBoard.is_stalemate()):
            print("adasdkjgasd")
            score = -100000000
            
        self.pgnBoard.pop()
        self.zobrist = curHash
        
        if (depthLimit >= 5):
            print(0,score,alpha_list[0],moves_list[0])
        if score > bestMove.score:
            cur = moves_list[0].uci()
            
            bestMove.score = score
            bestMove.a = ord(cur[0]) - 96
            bestMove.b = int(cur[1])
            bestMove.c = ord(cur[2]) - 96
            bestMove.d = int(cur[3])
            if (moves_list[0].promotion):
                bestMove.promotion = ord(cur[4]) - 96
            else:
                bestMove.promotion = -1
            best_move_index = 0
            
        alpha = max(alpha, bestMove.score)
        
        self.moves_list = moves_list
        self.alpha_list.append(score)
        
        for move in moves_list[1:]:
            
            # Razoring
            if (not(alpha_list[count] == None)):
                if (alpha - alpha_list[count] > razorThreshold):                    
                    break
            
            # Late move reduction
            if (count >= 35):
                depthUsage = depthLimit - 1
            else:
                depthUsage = depthLimit
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
            
            self.pgnBoard.push(move)
            #print(beta_list[count])
            score = self.minimizer(curDepth + 1, depthUsage, alpha, alpha+1, beta_list[count])           
                        
            #If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                if (not(alpha_list[count] == None)):
                    self.beta_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, alpha_list[count], beta_list[count])
                else:
                    self.beta_list.pop()
                    score = self.minimizer(curDepth + 1, depthUsage, alpha, beta, beta_list[count])
                # if alpha < score and score < beta:
                #     score = self.minimizer(curDepth + 1, depthUsage, alpha, beta)            
            
            if (self.pgnBoard.is_repetition(2) or self.pgnBoard.is_stalemate()):
                print("adasdkjgasd")
                score = -100000000
            # if (self.pgnBoard == chess.Board("r1bqnrk1/ppp1bppp/2p5/8/B2P1B2/2P5/PP3PPP/RN1QR1K1 w - - 3 12")):
            #     print(move, len(self.beta_list[count]), count)
            self.pgnBoard.pop()
            self.zobrist = curHash
            self.alpha_list.append(score)
            
            if (depthLimit >= 5):
                print(count,score,alpha_list[count], move)
            
            if score > bestMove.score:
                cur = move.uci()
                
                bestMove.score = score
                bestMove.a = ord(cur[0]) - 96
                bestMove.b = int(cur[1])
                bestMove.c = ord(cur[2]) - 96
                bestMove.d = int(cur[3])
                if (move.promotion):
                    bestMove.promotion = ord(cur[4]) - 96
                else:
                    bestMove.promotion = -1
                best_move_index = count
                
            alpha = max(alpha, bestMove.score)
            count += 1
            if beta <= alpha:
                self.numMove += 1
                if (depthLimit >= 5):
                    print()                
                    print("Best: ", best_move_index)
                
                for i in range(num_legal_moves - count):
                    self.alpha_list.append(None)
                # print(self.alpha_list)
                return bestMove

        for i in range(num_legal_moves - count):
            self.alpha_list.append(None)
        
        if curDepth == 0:
            self.numMove += 1
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
        cdef int highestScore = -9999999
        cdef int score
        cdef object move                        
        #cdef int index
        cdef int target_square
        cdef uint64_t curHash = self.zobrist
        cdef bint isCapture
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        if curDepth >= depthLimit:
            self.numIterations += 1
            return evaluate_board(self.pgnBoard,self.zobrist)

        #cdef list moves_list = self.reorder_capture_moves()
        #moves_list = self.get_legal_moves()
        # moves_list = list(self.reorder_capture_moves(curDepth))
        # #moves_list = self.reorder_legal_moves()
        # if (curDepth == 2):
        #     print(moves_list)
        for move in self.reorder_capture_moves():
            # if (curDepth == 2):
            #     print(self.pgnBoard.fen(), move)
            #     print(self.pgnBoard.move_stack)
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
            
            self.pgnBoard.push(move)
            score = self.minimizer(curDepth + 1, depthLimit, alpha, beta, [])
            self.pgnBoard.pop()
            self.zobrist = curHash
            # if (self.pgnBoard == chess.Board("rq2r1k1/2pn1p1p/1p1bpnp1/p3P3/2PP1P2/2NB4/P1Q2P1P/R1B3RK b - - 0 16")):
            #     #print("My Moves:", list(self.reorder_capture_moves()))
            #     # print()
            #     # print("Th moves: ", list (self.pgnBoard.legal_moves))
            #     # print()
            #     # if score == 14960:
            #     print ("MAX: ",score, move)
            #     # print()
            #     # print(list(self.reorder_capture_moves()))
            #     # print()
            # if (self.pgnBoard == chess.Board("r3r1k1/1qpn1p1p/1p1bpnp1/p3P3/2PP1P2/2NB4/P1Q2PRP/R1B4K b - - 2 17")):
            #     # print("My Moves:", self.reorder_capture_moves())
            #     # print()
            #     # print("Th moves: ", list (self.pgnBoard.legal_moves))
            #     # print()
            #     print ("MAX2: ",score, move)
            if score > highestScore:
                highestScore = score

            alpha = max(alpha, highestScore)

            if beta <= alpha:
                return highestScore
        
        if (highestScore == -9999999):
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
    cdef int minimizer(self, int curDepth, int depthLimit, int alpha, int beta, list beta_list):
        cdef int lowestScore = 9999999 - len(self.pgnBoard.move_stack)
        cdef int score
        cdef object move
        #cdef int a, b, c, d
        #cdef str cur
        #cdef int index
        cdef int target_square
        cdef int preBeta = beta
        cdef int count = 0
        cdef int razorThreshold
        if (depthLimit == 4):
            razorThreshold = max (int(1250 * .75** (depthLimit - 6)), 200) 
        else:
            razorThreshold = max (int(750 * .75** (depthLimit - 6)), 50)
        cdef uint64_t curHash = self.zobrist
        cdef bint isCapture
        cdef list cur_beta_list = []
        
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        if curDepth >= depthLimit:            
            self.numIterations += 1
            #print("AAA", self.numIterations)
            return evaluate_board(self.pgnBoard,self.zobrist)
        
        
        cdef list moves_list = list(self.reorder_capture_moves())
        cdef int length = len(moves_list)
        # if curDepth == 1:
        #     moves_list = list(self.pgnBoard.generate_legal_moves())
        #     quicksort_ascending_wrapper(beta_list, moves_list)
        # else:
        #     moves_list = list(self.reorder_capture_moves())
        
        # if (self.pgnBoard == chess.Board("rnbqkb1r/ppp3pp/5p2/3pp3/3PnB2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 0 6")):
        #     print("My Moves:", self.reorder_capture_moves())
        #     print()
        #     print("Th moves: ", list (self.pgnBoard.legal_moves))
        # if curDepth == 1:
        #     print(beta_list)
        for move in moves_list:
            
            if curDepth == 1:
                # print("AAAA", move,length , self.pgnBoard.fen())
                if (not(beta_list[count] == None)):
                    if (beta_list[count] - beta > razorThreshold):
                        # if self.pgnBoard == chess.Board("2b2b1r/3q2p1/4p1kp/2PpQ3/1P1P1Pn1/4P3/7P/4KB1R w K - 0 24"):
                        #     print("Removed:  ", beta_list[count], beta, move)
                        count+=1
                        cur_beta_list.append(None)
                        continue
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
            
            self.pgnBoard.push(move)
            score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
            self.pgnBoard.pop()
            self.zobrist = curHash
            # if (self.pgnBoard == chess.Board("rq2r1k1/2pn1p1p/1p1bpnp1/p7/2PPPP2/2NB4/P1Q2P1P/R1B3RK w - - 1 16")):
            #     # print("My Moves:", self.reorder_capture_moves())
            #     # print()
            #     # print("Th moves: ", list (self.pgnBoard.legal_moves))
            #     # print()
            #     print ("MIN: ",score, move, alpha, beta, lowestScore)
            # if (self.pgnBoard == chess.Board("r3r1k1/1qpn1p1p/1p1bpnp1/p3P3/2PP1P2/2NB4/P1Q2P1P/R1B3RK w - - 1 17")):
            #     # print("My Moves:", self.reorder_capture_moves())
            #     # print()
            #     # print("Th moves: ", list (self.pgnBoard.legal_moves))
            #     # print()
            #     print ("MIN2: ",score, move, alpha, beta)
            if curDepth == 1:
                # print(count)
                cur_beta_list.append(score)
            if score < lowestScore:
                lowestScore = score

            beta = min(beta, lowestScore)
            count+=1
            if beta <= alpha:
                if curDepth == 1:
                    # print(count)
                    for i in range(length - count):
                        cur_beta_list.append(None)
                    self.beta_list.append(cur_beta_list)
                return beta
                # return lowestScore
        if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):     
            if curDepth == 1:
                for i in range(length - count):
                    cur_beta_list.append(None)
                self.beta_list.append(cur_beta_list)
            if self.pgnBoard.is_checkmate():
                return 100000000
            else:
                return beta
        
        if curDepth == 1:
            for i in range(length - count):
                cur_beta_list.append(None)
            self.beta_list.append(cur_beta_list)    
        
        return lowestScore
    
    # Define the minimizer function
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple preMinimizer(self, int curDepth, int depthLimit, int alpha, int beta):
        cdef int lowestScore = 9999999 - len(self.pgnBoard.move_stack)
        cdef int score
        cdef object move
        cdef int target_square
        cdef uint64_t curHash = self.zobrist
        cdef list beta_list = []
        cdef bint isCapture
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        if curDepth >= depthLimit:            
            self.numIterations += 1            
            return evaluate_board(self.pgnBoard,self.zobrist)

        cdef list moves_list = list(self.reorder_capture_moves())
        cdef int length = len(moves_list)
        cdef int count = 0
        for move in moves_list:
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
            
            self.pgnBoard.push(move)
            score = self.maximizer(curDepth + 1, depthLimit, alpha, beta)
            self.pgnBoard.pop()
            self.zobrist = curHash
            
            beta_list.append(score)
            count += 1
            if score < lowestScore:
                lowestScore = score

            beta = min(beta, lowestScore)

            if beta <= alpha:
                
                for i in range(length - count):
                    beta_list.append(None)
                
                return beta, beta_list
                
        if (lowestScore == 9999999 - len(self.pgnBoard.move_stack)):            
            if self.pgnBoard.is_checkmate():
                return 100000000, beta_list
            
            
        return lowestScore, beta_list
    
    
    def ev(self, object board):
        #initialize_layers(board)
        setAttackingLayer(board.occupied_co[True], board.occupied_co[False], board.kings,5)
        self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False])
        return evaluate_board(board,self.zobrist)
    
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef tuple reorder_legal_moves(self,int alpha,int beta, depthLimit):
        
        cdef int score = -99999999
        cdef int highestScore = -99999999
        cdef str cur
        cdef list moves_list
        cdef list alpha_list = []
        cdef list beta_list = []
        cdef list curList = []
        cdef int count = 1
        cdef int depth = depthLimit - 2
        cdef uint64_t curHash = self.zobrist
        cdef bint isCapture
        cdef uint64_t pawns = self.pgnBoard.pawns
        cdef uint64_t knights = self.pgnBoard.knights
        cdef uint64_t bishops = self.pgnBoard.bishops
        cdef uint64_t rooks = self.pgnBoard.rooks
        cdef uint64_t queens = self.pgnBoard.queens
        cdef uint64_t kings = self.pgnBoard.kings
        
        cdef uint64_t occupied_white = self.pgnBoard.occupied_co[True]
        cdef uint64_t occupied_black = self.pgnBoard.occupied_co[False]
        cdef uint64_t occupied = self.pgnBoard.occupied   
        
        if (self.alpha_list == []):
            moves_list = list(Cython_Chess.generate_legal_moves(self.pgnBoard,chess.BB_ALL,chess.BB_ALL))
        else:
            moves_list = self.moves_list
        # moves_list = list(Cython_Chess.generate_legal_moves(self.pgnBoard,chess.BB_ALL,chess.BB_ALL))
        
        isCapture = is_capture(moves_list[0].from_square, moves_list[0].to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(moves_list[0]))
        updateZobristHashForMove(self.zobrist, moves_list[0].from_square, moves_list[0].to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
        self.pgnBoard.push(moves_list[0])
        
        highestScore, curList = self.preMinimizer(1, depth, alpha, beta)
        self.pgnBoard.pop()
        
        self.zobrist = curHash
        
        alpha = max(alpha, highestScore)
        alpha_list.append(highestScore)
        beta_list.append(curList)
        for move in moves_list[1:]:
            
            isCapture = is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move))
            updateZobristHashForMove(self.zobrist, move.from_square, move.to_square, isCapture, pawns, knights, bishops, rooks, queens, kings, occupied_white, occupied_black)
            
            self.pgnBoard.push(move)
            
            score, curList = self.preMinimizer(1, depth, alpha, alpha + 1)
            
            # If the score is within the window, re-search with full window
            if alpha < score and score < beta:
                score, curList = self.preMinimizer(1, depth, alpha, beta)
                        
            self.pgnBoard.pop()
            self.zobrist = curHash
            alpha_list.append(score)
            beta_list.append(curList)
            
            if score > highestScore:
                highestScore = score
            count += 1
            alpha = max(alpha, highestScore)
                
        if (self.alpha_list == []):
            quicksort(alpha_list, moves_list, beta_list, 0, len(alpha_list) - 1)
            return moves_list,alpha_list,beta_list
        else:
            # print(self.beta_list, len(self.beta_list))
            quicksort_wrapper(self.alpha_list, moves_list, self.beta_list,alpha_list,beta_list)    
            # print()
            # print(self.beta_list, len(self.beta_list), len(moves_list), len(self.alpha_list), len(beta_list), len(alpha_list))
            return moves_list,self.alpha_list,self.beta_list
            
        # quicksort(alpha_list, moves_list, beta_list, 0, len(alpha_list) - 1)
        # return moves_list,alpha_list,beta_list    
    def reorder_capture_moves(self) -> Iterator[chess.Move]:
        
        cdef object move
                
        for move in Cython_Chess.generate_legal_captures(self.pgnBoard,chess.BB_ALL,chess.BB_ALL):
            yield move
        for move in Cython_Chess.generate_legal_moves(self.pgnBoard,chess.BB_ALL,chess.BB_ALL):
            if not is_capture(move.from_square, move.to_square, self.pgnBoard.occupied_co[not self.pgnBoard.turn], self.pgnBoard.is_en_passant(move)):
                yield move
    
    @boundscheck(False)
    @wraparound(False)
    @cython.exceptval(check=False)
    @cython.nonecheck(False)
    @cython.ccall
    @cython.inline
    cdef list get_legal_moves(self):
        # Create a Cython integer for the occupied mask
        cdef unsigned long long occupied_mask = self.pgnBoard.occupied
        
        # Check if the moves for this board state are already cached
        if occupied_mask in self.move_cache:
            return self.move_cache[occupied_mask]
        
        # Generate legal moves
        legal_moves = self.reorder_capture_moves()
        
        # Store the generated moves in the hash map
        self.move_cache[occupied_mask] = legal_moves
        
        return legal_moves


cdef void quicksort(list values, list objects, list betas, int left, int right):
    if left >= right:
        return

    pivot = values[left + (right - left) // 2]
    cdef int i = left
    cdef int j = right
    cdef int temp_value
    cdef object temp_object
    cdef list temp_list

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

            i += 1
            j -= 1

    # Recursively sort the partitions
    quicksort(values, objects, betas, left, j)
    quicksort(values, objects, betas, i, right)
    
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

cdef void quicksort_wrapper(list alphas, list objects, list betas, list preAlphas, list preBetas):
    cdef int count = 0
    for i in range (len(alphas)):
        if (alphas[i] == None):
            break
        count += 1
        
    
    cdef list alphas_sub_list = alphas[:count]
    cdef list objects_sub_list = objects[:count]
    cdef list betas_sub_list = betas[:count]
    # print(betas)
    cdef list remaining_alphas = []
    cdef list remaining_objects = []
    cdef list remaining_betas = []

    quicksort(alphas_sub_list, objects_sub_list, betas_sub_list, 0, len(alphas_sub_list) - 1)
    
    if (count < len(alphas)):
        remaining_alphas = preAlphas[count:]
        remaining_objects = objects[count:]
        remaining_betas = preBetas[count:]
        
        quicksort(remaining_alphas, remaining_objects, remaining_betas, 0, len(remaining_alphas) - 1)
        
    alphas[:] = alphas_sub_list + remaining_alphas
    objects[:] = objects_sub_list + remaining_objects
    betas[:] = betas_sub_list + remaining_betas
        
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
    #cdef uint64_t curZobrist = generateZobristHash(pawns,knights,bishops,rooks,queens,kings,occupied_white,occupied_black)
    #zobrist = curZobrist
    # if (curZobrist != zobrist):
    #     print(curZobrist, zobrist, board.fen())
    cdef int cache_result = accessCache(zobrist)
    
    if (cache_result != 0):
        # print(cache_result,board.fen())
        return cache_result
        
    global prevKings
    global whiteCastledIndex
    global blackCastledIndex
    
    cdef int total = 0
    cdef int subTotal = 0
    cdef object piece
    cdef uint8_t  square
    cdef uint8_t  x, y
    cdef uint8_t i
    cdef int moveNum = board.ply()
    cdef object target_square
    cdef object target_move
    cdef uint8_t  kingSeparation
    cdef uint8_t size
    
    cdef white_ksc = chess.Move.from_uci('e1g1')
    cdef white_qsc = chess.Move.from_uci('e1c1')
    cdef black_ksc = chess.Move.from_uci('e8g8')
    cdef black_qsc = chess.Move.from_uci('e8c8')
    
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
    
    
    # if (board.occupied == 10746666234248479586):
    #     print(board)
    # Iterate through all squares on the board and evaluate piece values
    if board.is_checkmate():
        if board.turn:
            total = 100000000            
        else:
            total = -100000000
    else:
        total += placement_and_piece_eval(moveNum, pawns, knights, bishops, rooks, queens, kings, prevKings, occupied_white, occupied_black, occupied)
        

        if (whiteCastledIndex == -1):        
            castle_index = move_index (board, white_ksc, white_qsc)
            if (castle_index != -1):
                total -= max(3000 - ((castle_index-1) >> 1) * 50 - moveNum * 25, 0)
        else:
            total -= max(3000 - ((whiteCastledIndex-1) >> 1) * 50 - moveNum * 25, 0)
        
        if (blackCastledIndex == -1):       
            castle_index = move_index (board, black_ksc, black_qsc)
            if (castle_index != -1):            
                total += max(3000 - ((castle_index-1) >> 1) * 50 - moveNum * 25, 0)
        else:
            total += max(3000 - ((blackCastledIndex-1) >> 1) * 50 - moveNum * 25, 0)
        
        
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
    addToCache(zobrist, total)
    
    return total

cdef int move_index(object board, object move1, object move2):
    
    cdef int index
    cdef object move
    
    for index, move in enumerate(board.move_stack):
        if move == move1 or move == move2:
            return index
    return -1


