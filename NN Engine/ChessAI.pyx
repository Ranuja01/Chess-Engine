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
from libcpp.unordered_map cimport unordered_map

cdef extern from "stdint.h":
    ctypedef signed char int8_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned long long uint64_t
    
cdef extern from "search_engine.h":
    
    cdef struct BoardState:
        uint64_t pawns
        uint64_t knights
        uint64_t bishops
        uint64_t rooks
        uint64_t queens
        uint64_t kings

        uint64_t occupied_white
        uint64_t occupied_black
        uint64_t occupied

        uint64_t promoted

        bint turn
        uint64_t castling_rights

        int ep_square
        int halfmove_clock
        int fullmove_number
        
    cdef struct MoveData:
        int a
        int b
        int c
        int d
        int promotion
        int score
        int num_iterations
       
    cdef struct Move:
        uint8_t from_square
        uint8_t to_square
        uint8_t promotion
        
    void initialize_engine(vector[BoardState]& state_history, unordered_map[uint64_t, int]& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bint turn, bint side_to_play)
    void set_current_state(vector[BoardState]& state_history, unordered_map[uint64_t, int]& position_count, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied, uint64_t occupied_white, uint64_t occupied_black, uint64_t promoted, uint64_t castling_rights, int ep_square, int halfmove_clock, int fullmove_number, bint turn)
    MoveData get_engine_move(vector[BoardState]& state_history, unordered_map[uint64_t, int]& position_count)

# cdef extern from "nnue.h":
#     int evaluate_position(const uint64_t* bitboards)
#     void init_session(const char* model_path)
#     void load_all_weights()
#     int run_inference(const uint64_t* bitboards)
#     void load_model()
#     int run_inference_quantized(const uint64_t* bitboards)


cdef extern from "cache_management.h":
    void initializeZobrist()
    uint64_t generateZobristHash(uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, bint whiteToMove);
    void updateZobristHashForMove(uint64_t& hash, uint8_t fromSquare, uint8_t toSquare, bint isCapture, uint64_t pawnsMask, uint64_t knightsMask, uint64_t bishopsMask, uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask, uint64_t occupied_whiteMask, uint64_t occupied_blackMask, int promotion)
    int accessCache(uint64_t key)
    void addToCache(uint64_t key, int max_size, int value)   
    int printCacheStats()
        

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
    int placement_and_piece_eval(int moveNum, bint turn, uint64_t pawns, uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens, uint64_t kings, uint64_t occupied_white, uint64_t occupied_black, uint64_t occupied)
    
    void printLayers()
    
    bint is_checkmate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bint turn)
    
    bint is_stalemate(uint64_t preliminary_castling_mask, uint64_t occupiedMask, uint64_t occupiedWhite, uint64_t opposingPieces, uint64_t ourPieces, uint64_t pawnsMask, uint64_t knightsMask,
	  			  uint64_t bishopsMask,	uint64_t rooksMask, uint64_t queensMask, uint64_t kingsMask,  int ep_square, bint turn)
    
# Create struct to hold information regarding the chosen move by the engine
cdef struct MoveData_Cython:
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

    cdef bint side_to_play
    
    
    cdef vector[BoardState] state_history
    cdef unordered_map[uint64_t, int] position_count
    
    # Constructor for chess engine
    def __cinit__(self, object black_model, object white_model, object board, bint side_to_play):
        
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
        self.move_times[6] = 5.5
        self.time_limit = 60
        self.quiescenceDepth = 6
        
        self.is_training = False
        
        self.side_to_play = side_to_play
        
        for i in range(7,26):
            self.move_times[i] = 2.5
        
        # Initialize attack tables for move generation
        initialize_attack_tables()
        Cython_Chess.inititalize()
        # init_session(nnue_model_path)
        # load_all_weights()
        # load_model()
        
        # Initialize zobrist tables for hashing
        initializeZobrist()
        initialize_engine(self.state_history, self.position_count, board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied, board.occupied_co[True],board.occupied_co[False],
                          board.promoted, board.castling_rights, (-1 if board.ep_square is None else board.ep_square), board.halfmove_clock, board.fullmove_number, board.turn, self.side_to_play)
        
        # self.zobrist = generateZobristHash(board.pawns,board.knights,board.bishops,board.rooks,board.queens,board.kings,board.occupied_co[True],board.occupied_co[False], board.turn)    
    
    # Function to set global variable for white castling index
    def setWhiteCastledIndex(self, index):
        global whiteCastledIndex
        whiteCastledIndex = index
    
    # Function to set global variable for black castling index
    def setBlackCastledIndex(self, index):
        global blackCastledIndex
        blackCastledIndex = index            

    
    # def create_test_data(self, object board):
    #     self.is_training = True
    #     self.pgnBoard = board
        
    #     # Initialize the lists required for iterative deepening
    #     self.moves_list = []
    #     self.alpha_list = []
    #     self.beta_list = []
    #     self.beta_move_list = []
    #     self.numIterations = 0
        
    #     cdef int cacheSize = getCacheStats()
        
    #     if (self.pgnBoard.ply() < 30):
    #         if (cacheSize > 16000000):
    #             evictOldEntries(cacheSize - 16000000)                
    #     elif(self.pgnBoard.ply() < 50):
    #         if (cacheSize > 32000000):
    #             evictOldEntries(cacheSize - 32000000)
    #     elif(self.pgnBoard.ply() < 75):
    #         if (cacheSize > 64000000):
    #             evictOldEntries(cacheSize - 64000000)
    #     else:
    #         if (cacheSize > 128000000):
    #             evictOldEntries(cacheSize - 128000000)
        
    #     cdef int a, b, c, d,promo,val
    #     cdef object move
        
    #     self.zobrist = generateZobristHash(self.pgnBoard.pawns,self.pgnBoard.knights,self.pgnBoard.bishops,self.pgnBoard.rooks,self.pgnBoard.queens,self.pgnBoard.kings,self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False], self.pgnBoard.turn)
        
    #     result = self.alphaBeta(curDepth=0, depthLimit=3, t0 = timer())
    #     moves_list ,_,_,_= self.reorder_legal_moves(-9999998,9999999, 3)
    #     length = len(self.alpha_list)
        
        
    #     self.is_training = False
    #     return (moves_list [:length], self.alpha_list)
    
    # Function to wrap the 
    def alphaBetaWrapper(self):

        cdef MoveData_Cython result
        cdef int a, b, c, d,promo,val
        # cdef int x,y,i,j
        cdef object move
        # If less than 30 plies have been played, check the opening book
        if (len(self.pgnBoard.move_stack) < 30):
            t0 = timer()
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
                    return move
                else:
                    return chess.Move.from_uci(x+y+i+j+chr(promo + 96))

        cdef MoveData move_data        
        if (len(self.pgnBoard.move_stack) > 0):
            print(self.pgnBoard.ep_square)            
            set_current_state(self.state_history, self.position_count, self.pgnBoard.pawns, self.pgnBoard.knights, self.pgnBoard.bishops, self.pgnBoard.rooks, self.pgnBoard.queens,
                              self.pgnBoard.kings, self.pgnBoard.occupied, self.pgnBoard.occupied_co[True],self.pgnBoard.occupied_co[False], self.pgnBoard.promoted, 
                              self.pgnBoard.castling_rights, (-1 if self.pgnBoard.ep_square is None else self.pgnBoard.ep_square), self.pgnBoard.halfmove_clock,
                              self.pgnBoard.fullmove_number, self.pgnBoard.turn)
        t0= timer() 
        move_data = get_engine_move(self.state_history, self.position_count)
        t1 = timer()        
        
        if not((move_data.a,move_data.b,move_data.c,move_data.d) == (-1,-1,-1,-1)):  
            
            print()
            print("Evaluation: ", move_data.score)
            print("Positions Analyzed: ",move_data.num_iterations)
            print("Average Static Analysis Speed: ", move_data.num_iterations / (t1 - t0))
            print ("Time Taken: ", t1 - t0)
            print("Move: ", self.pgnBoard.ply())
            print()
            
            
            # Convert the coordinates to alphanumeric representation
            x = chr(int(move_data.a) + 96)
            y = str(move_data.b)
            i = chr(int(move_data.c) + 96)
            j = str(move_data.d)
            
            # Check if the move is a promoting move
            if (move_data.promotion == 1):
                move = chess.Move.from_uci(x+y+i+j)                            
                return move
            else:
                if move_data.promotion == 2:
                    promotion_char = 'n'  # knight
                elif move_data.promotion == 3:
                    promotion_char = 'b'  # bishop
                elif move_data.promotion == 4:
                    promotion_char = 'r'  # rook
                elif move_data.promotion == 5:
                    promotion_char = 'q'  # queen
                
                return chess.Move.from_uci(x+y+i+j+promotion_char)
        else:
            return None
        
    
    
    # Function for opening book moves
    @cython.ccall
    @cython.exceptval(check=False)
    cdef MoveData_Cython opening_book(self):
        
        cdef MoveData_Cython best_move
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
                            
    
